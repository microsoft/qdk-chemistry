"""Observable dynamic mode decomposition implementation.

This module implements the observable dynamic mode decomposition (ODMD)
algorithm, which measures the overlap between the initial state and a series of state evolved
after different time steps. These overlap values compose two Hankel matrices X and X', connected
by DMD system matrix A. The estimated ground state energy are then obtained by diagonalizing A.

Reference:
    Shen, Y., Camps, D., Szasz, A., Darbha, S., Klymko, K., Williams, D.B., Tubman, N.M. and
    Van Beeumen, R., 2025. Estimating eigenenergies from quantum dynamics: A unified
    noise-resilient measurement-driven approach. Quantum, 9, p.1836.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import numpy as np

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.base import ControlledEvolutionCircuitMapper
from qdk_chemistry.data import (
    Circuit,
    ControlledTimeEvolutionUnitary,
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
)
from qdk_chemistry.utils import Logger

from .base import PhaseEstimation, PhaseEstimationSettings
from .measure_observable import measure_observable

__all__: list[str] = ["DynamicModeDecomposition", "DynamicModeDecompositionSettings"]


class DynamicModeDecompositionSettings(PhaseEstimationSettings):
    """Settings for the Dynamic Mode Decomposition algorithm."""

    def __init__(self):
        """Initialize the settings for ODMD.

        Args:
            shots_per_observable: The number of shots to execute per measuring a bit in the iterative phase estimation.

        """
        super().__init__()
        self._set_default("hankel_rows", "int", -1, "the row number of Hankel matrix X")
        self._set_default("initial_hankel_columns", "int", -1, "the initial column number of Hankel matrix X")
        self._set_default("evolution_time", "float", 0.0, "Time dt in the evolution unitary U = exp(-i H dt)")
        self._set_default("eigen_converge_tol", "float", 1e-5, "Convergence tolerance for Hamiltonian eigenvalue")
        self._set_default("shots_per_observable", "int", 40, "the number of shots to test one observable")
        self._set_default("max_hankel_columns", "int", 200, "Column number limit of Hankel matrix X")


class DynamicModeDecomposition(PhaseEstimation):
    """ODMD algorithm implementation."""

    def __init__(
        self,
        hankel_rows: int,
        initial_hankel_columns: int,
        time_step: float,
        eigen_converge_tol: float = 1e-3,
        shots_per_observable: int = 40,
        max_hankel_columns: int = 200,
    ):
        """Initialize DynamicModeDecomposition with the given settings.

        Args:
            hankel_rows: The row count of the Hankel matrix X. Must be a positive integer.
            initial_hankel_columns: The initial column count of the Hankel matrix X. Must be greater than 1.
            time_step: Time parameter dt used in the time-evolution unitary U = exp(-i H dt)``.
            Must be a non-negative number.
            eigen_converge_tol: Convergence tolerance for Hamiltonian eigenvalue.
            Must be a positive float.
            shots_per_observable: The number of shots to execute per observable measurement. Defaults to 40.
            Must be a positive integer.
            max_hankel_columns: The maximum column count limit of Hankel matrix X. Defaults to 200.
            Must be greater than or equal to initial_hankel_columns.

        Raises:
            ValueError: If hankel_rows or initial_hankel_columns are not positive integers,
            if initial_hankel_columns exceeds max_hankel_columns, if time_step is negative,
            if eigen_converge_tol is not positive, or if shots_per_observable is not positive.

        """
        Logger.trace_entering()
        super().__init__(evolution_time=time_step)
        self._settings = DynamicModeDecompositionSettings()
        self._settings.set("hankel_rows", hankel_rows)
        self._settings.set("initial_hankel_columns", initial_hankel_columns)
        self._settings.set("evolution_time", time_step)
        self._settings.set("eigen_converge_tol", eigen_converge_tol)
        self._settings.set("shots_per_observable", shots_per_observable)
        self._settings.set("max_hankel_columns", max_hankel_columns)
        self._iteration_circuits: list[Circuit] | None = None
        self._last_converged: bool | None = None
        self._phase_fraction_history: list[float] = []

        # check validity of inputs
        if hankel_rows < 1:
            raise ValueError("hankel_rows must be a positive integer.")
        if initial_hankel_columns < 2:
            raise ValueError("initial_hankel_columns must be larger than 1.")
        if initial_hankel_columns > max_hankel_columns:
            raise ValueError("initial_hankel_columns must be no more than max_hankel_columns.")
        if time_step <= 0.0:
            raise ValueError("time_step must be a positive float.")
        if eigen_converge_tol <= 0.0:
            raise ValueError("eigen_converge_tol must be a positive float.")
        if shots_per_observable < 1:  # currently we use classical Hadamard test for measuring observable
            raise ValueError("shots_per_observable must be a positive integer.")

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: ControlledEvolutionCircuitMapper,
        circuit_executor: CircuitExecutor,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run the ODMD with the given state preparation circuit and qubit Hamiltonian.

        Args:
            state_preparation: The state preparation circuit.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            evolution_builder: The time evolution builder to use.
            circuit_mapper: The controlled evolution circuit mapper to use.
            circuit_executor: The executor to run quantum circuits.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            QpeResult: The result of the phase estimation.

        """
        hankel_rows = self._settings.get("hankel_rows")
        initial_hankel_columns = self._settings.get("initial_hankel_columns")
        max_hankel_columns = self._settings.get("max_hankel_columns")
        initial_number_measurement = hankel_rows + initial_hankel_columns
        observable_array = np.zeros(hankel_rows + max_hankel_columns)

        # Generate controlled U = exp(-iH dt) by evolution_builder, it will be connected by circuit_mapper
        ctrl_time_evol_unitary = self.create_ctrl_evol_unitary(qubit_hamiltonian, evolution_builder, [0])
        shots = self._settings.get("shots_per_observable")

        # Calculate initial hankel_rows + initial_hankel_columns - 1 observables
        for i in range(initial_number_measurement - 1):
            observable_power = i + 1
            ctrl_evol_circuit = self._create_ctrl_time_evol_circuit(
                controlled_evolution=ctrl_time_evol_unitary,
                power=observable_power,
                circuit_mapper=circuit_mapper,
            )
            Logger.info(f"Measuring observable with {observable_power} power of controlled unitary.")
            observable_array[i] = measure_observable(
                state_preparation=state_preparation,
                num_system_qubits=qubit_hamiltonian.num_qubits,
                ctrl_time_evol_unitary_circuit=ctrl_evol_circuit,
                circuit_executor=circuit_executor,
                shots=shots,
                noise=noise,
            )

        phase_fraction = 0.0
        observe_index = initial_number_measurement - 2
        rank_k = initial_hankel_columns - 1

        evolution_time = self._settings.get("evolution_time")
        eigen_converge_tol = self._settings.get("eigen_converge_tol")
        convergence_tolerance = eigen_converge_tol * evolution_time / (2.0 * np.pi)
        converged = False
        stop_reason = "max_hankel_columns_reached"
        odmd_iterations = 0
        self._phase_fraction_history = []

        # Compose Hankel matrices
        hankel_x = np.zeros((hankel_rows, rank_k))
        hankel_x_prime = np.zeros((hankel_rows, rank_k))

        for v in range(rank_k):
            hankel_x[:, v] = observable_array[v : v + hankel_rows]
        hankel_x_prime[:, 0 : rank_k - 1] = hankel_x[:, 1:rank_k]
        hankel_x_prime[:, rank_k - 1] = observable_array[rank_k : rank_k + hankel_rows]

        while True:
            if rank_k == max_hankel_columns:
                Logger.warn(f"reached the limit of rank_k {max_hankel_columns}! ODMD iteration will stop here!")
                break

            # Quantum part: obtain the observables (overlaps) by Hadamard tests
            rank_k += 1
            observe_index += 1
            observable_power = observe_index + 1
            ctrl_evol_circuit = self._create_ctrl_time_evol_circuit(
                controlled_evolution=ctrl_time_evol_unitary,
                power=observable_power,
                circuit_mapper=circuit_mapper,
            )
            Logger.info(f"Measuring observable with {observable_power} power of controlled unitary.")
            observable_array[observe_index] = measure_observable(
                state_preparation=state_preparation,
                num_system_qubits=qubit_hamiltonian.num_qubits,
                ctrl_time_evol_unitary_circuit=ctrl_evol_circuit,
                circuit_executor=circuit_executor,
                shots=shots,
                noise=noise,
            )

            # Classical part: update Hankel matrices, pseudo-inverse to get DMD matrix A
            hankel_x = np.column_stack((hankel_x, hankel_x_prime[:, rank_k - 2]))
            hankel_x_prime = np.column_stack((hankel_x_prime, observable_array[rank_k : rank_k + hankel_rows]))

            dmd_a = hankel_x_prime @ np.linalg.pinv(hankel_x)

            # Classical part: extract ground state energy from A by diagonalization
            eigenvalues = np.linalg.eigvals(dmd_a)  # eigensolver for general matrix

            log_eigs = np.log(eigenvalues)

            Logger.info(f"log_eigs: {log_eigs}")

            # Pick the mode with smallest |real| part.
            min_real_idx = np.argmin(np.abs(log_eigs.real))
            phase = -log_eigs[min_real_idx].imag
            phase_fraction = phase / (2.0 * np.pi)
            odmd_iterations += 1
            converged = self._record_phase_fraction_and_check_convergence(
                phase_fraction=phase_fraction,
                convergence_tolerance=convergence_tolerance,
            )

            Logger.info(f"iteration {rank_k - initial_hankel_columns + 1}, phase_fraction {phase_fraction}")
            if converged:
                stop_reason = "converged"
                break

        self._last_converged = converged

        result_metadata = {
            "converged": converged,
            "stop_reason": stop_reason,
            "iterations": odmd_iterations,
            "final_hankel_columns": rank_k,
            "max_hankel_columns": max_hankel_columns,
            "eigen_converge_tol": eigen_converge_tol,
            "phase_fraction_convergence_tolerance": convergence_tolerance,
            "phase_fraction_history": list(self._phase_fraction_history),
        }

        # Create and return the result
        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=phase_fraction,
            evolution_time=self.settings().get("evolution_time"),
            metadata=result_metadata,
        )

    def _record_phase_fraction_and_check_convergence(
        self,
        phase_fraction: float,
        convergence_tolerance: float,
    ) -> bool:
        """Record phase history and evaluate convergence for the current ODMD run.

        Args:
            phase_fraction: Newly estimated phase fraction for this iteration.
            convergence_tolerance: Convergence threshold on absolute phase difference.

        Returns:
            bool: ``True`` when converged under the current criterion, else ``False``.

        """
        current_phase_fraction = float(phase_fraction)
        self._phase_fraction_history.append(current_phase_fraction)

        if len(self._phase_fraction_history) < 2:
            return False

        previous_phase_fraction = self._phase_fraction_history[-2]
        return abs(abs(current_phase_fraction) - abs(previous_phase_fraction)) <= convergence_tolerance

    def is_converged(self) -> bool | None:
        """Return convergence state from the most recent ODMD run.

        Returns:
            bool | None: ``True`` if the latest run converged, ``False`` if it stopped
            before convergence, or ``None`` if no run has been executed yet.

        """
        return self._last_converged

    def create_ctrl_evol_unitary(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        evolution_builder: TimeEvolutionBuilder,
        control_indices: list[int] | None = None,
    ) -> ControlledTimeEvolutionUnitary:
        """Create the controlled one time step evolution unitary operator U = exp(-iH dt).

        Args:
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            evolution_builder: The time evolution builder to use.
            control_indices: The list of control qubit indices

        """
        evolution_time = self._settings.get("evolution_time")

        time_evol_unitary = self._create_time_evolution(
            qubit_hamiltonian=qubit_hamiltonian,
            time=evolution_time,
            evolution_builder=evolution_builder,
        )

        return ControlledTimeEvolutionUnitary(
            time_evolution_unitary=time_evol_unitary,
            control_indices=control_indices,
        )

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "ODMD"
