"""Observable dynamic mode decomposition implementation.

This module implements the observable dynamic mode decomposition (ODMD)
algorithm, which measures overlaps between an initial state and states evolved
for different time steps. These overlap values compose two Hankel matrices
``X`` and ``X'``, connected by the DMD system matrix ``A``. The estimated
ground-state energy is then obtained from diagonalizing ``A``.

In the current implementation, overlap measurements are evaluated with a
Hadamard test in the ``X`` basis, i.e., using ``Re(<psi|U|psi>)`` samples.

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
from qdk_chemistry.algorithms.hadamard_test_generator.base import HadamardTestGenerator
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

__all__: list[str] = ["DynamicModeDecomposition", "DynamicModeDecompositionSettings"]


def measure_observable(
    state_preparation: Circuit,
    num_system_qubits: int,
    ctrl_time_evol_unitary_circuit: Circuit,
    hadamard_test_generator: HadamardTestGenerator,
    circuit_executor: CircuitExecutor,
    shots: int,
    noise: QuantumErrorProfile | None = None,
) -> float:
    """Measure ``Re(<psi|U|psi>)`` using a backend-supported Hadamard test.

    Args:
        state_preparation: Circuit that prepares the trial state ``|psi>``.
        num_system_qubits: Number of system qubits acted on by ``state_preparation`` and ``U``.
        ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing ``U``.
        hadamard_test_generator: Hadamard-test circuit generator used to build the
            measurement circuit in the ``"X"`` basis.
        circuit_executor: Backend executor used to run the generated measurement circuit.
        shots: Number of shots used for the expectation estimate.
        noise: Optional noise profile for noisy simulation.

    Returns:
        Real-valued overlap estimate computed from ancilla measurement counts.

    """
    circuit = hadamard_test_generator.run(
        state_preparation, num_system_qubits, ctrl_time_evol_unitary_circuit, test_basis="X"
    )

    executor_data = circuit_executor.run(circuit, shots=shots, noise=noise)
    bitstring_result = executor_data.bitstring_counts
    Logger.info(
        "Measured real observable from Hadamard test, "
        f"{bitstring_result.get('0', 0)} zeros, {bitstring_result.get('1', 0)} ones"
    )
    observable_value = (bitstring_result.get("0", 0) - bitstring_result.get("1", 0)) / shots
    Logger.info(f"Measured observable value {observable_value}")
    return observable_value


class DynamicModeDecompositionSettings(PhaseEstimationSettings):
    """Settings for the Dynamic Mode Decomposition algorithm."""

    def __init__(self):
        """Initialize the settings for ODMD."""
        super().__init__()
        self._set_default("hankel_rows", "int", -1, "Row number of Hankel matrix X")
        self._set_default("hankel_columns", "int", -1, "Column number of Hankel matrices X and X'")
        self._set_default("evolution_time", "float", 0.0, "Time dt in the evolution unitary U = exp(-i H dt)")
        self._set_default(
            "shots_per_observable", "int", 100, "Number of shots to use when measuring each observable in ODMD"
        )


class DynamicModeDecomposition(PhaseEstimation):
    """ODMD algorithm implementation."""

    def __init__(
        self,
        hankel_rows: int,
        hankel_columns: int,
        evolution_time: float,
        shots_per_observable: int = 100,
    ):
        """Initialize DynamicModeDecomposition with the given settings.

        Args:
            hankel_rows: The row count of the Hankel matrix X. Must be a positive integer.
            hankel_columns: The column count of both Hankel matrices X and X'. Must be a positive integer.
            evolution_time: Time parameter dt in time-evolution unitary U = exp(-i H dt)``. It must be positive.
            shots_per_observable: The number of shots to use when measuring each observable in ODMD.

        """
        Logger.trace_entering()
        super().__init__(evolution_time=evolution_time)
        self._settings = DynamicModeDecompositionSettings()
        self._settings.set("hankel_rows", hankel_rows)
        self._settings.set("hankel_columns", hankel_columns)
        self._settings.set("evolution_time", evolution_time)
        self._settings.set("shots_per_observable", shots_per_observable)
        self._iteration_circuits: list[Circuit] | None = None

        # check validity of inputs
        if hankel_rows < 1:
            raise ValueError("hankel_rows must be a positive integer.")
        if hankel_columns < 1:
            raise ValueError("hankel_columns must be a positive integer.")
        if evolution_time <= 0.0:
            raise ValueError("evolution_time must be a positive float.")
        if shots_per_observable < 1:  # currently we use classical Hadamard test for measuring observable
            raise ValueError("shots_per_observable must be a positive integer.")

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: ControlledEvolutionCircuitMapper,
        hadamard_test_generator: HadamardTestGenerator,
        circuit_executor: CircuitExecutor,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run the ODMD with the given state preparation circuit and qubit Hamiltonian.

        Args:
            state_preparation: Circuit that prepares the trial state ``|psi>``.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            evolution_builder: The time evolution builder to use.
            circuit_mapper: The controlled evolution circuit mapper to use.
            hadamard_test_generator: Hadamard-test circuit generator used for overlap
                estimation; ODMD uses its ``"X"``-basis measurement path.
            circuit_executor: Backend executor used to run the generated measurement circuit.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            QpeResult: The result of the phase estimation.

        """
        hankel_rows = self._settings.get("hankel_rows")
        hankel_columns = self._settings.get("hankel_columns")
        number_of_measurements = hankel_rows + hankel_columns
        observable_array = np.zeros(number_of_measurements)

        # Generate controlled U = exp(-iH dt) by evolution_builder, it will be connected by circuit_mapper
        ctrl_time_evol_unitary = self.create_ctrl_evol_unitary(qubit_hamiltonian, evolution_builder, [0])
        shots = self._settings.get("shots_per_observable")

        # Calculate observables needed for fixed-size Hankel matrices X and X'.
        for i in range(number_of_measurements):
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
                hadamard_test_generator=hadamard_test_generator,
                circuit_executor=circuit_executor,
                shots=shots,
                noise=noise,
            )

        hankel_x = np.zeros((hankel_rows, hankel_columns))
        hankel_x_prime = np.zeros((hankel_rows, hankel_columns))
        for v in range(hankel_columns):
            hankel_x[:, v] = observable_array[v : v + hankel_rows]
            hankel_x_prime[:, v] = observable_array[v + 1 : v + 1 + hankel_rows]

        dmd_a = hankel_x_prime @ np.linalg.pinv(hankel_x)
        eigenvalues = np.linalg.eigvals(dmd_a)
        log_eigs = np.log(eigenvalues)

        Logger.info(f"log_eigs: {log_eigs}")

        min_real_idx = int(np.argmin(np.abs(log_eigs.real)))
        phase = -log_eigs[min_real_idx].imag
        phase_fraction = phase / (2.0 * np.pi)

        result_metadata = {
            "hankel_rows": hankel_rows,
            "hankel_columns": hankel_columns,
            "shots_per_observable": shots,
            "selected_mode_index": min_real_idx,
        }

        # Create and return the result
        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=phase_fraction,
            evolution_time=self.settings().get("evolution_time"),
            metadata=result_metadata,
        )

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
