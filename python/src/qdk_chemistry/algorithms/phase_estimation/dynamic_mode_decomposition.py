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
import copy

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3

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

__all__: list[str] = ["DynamicModeDecomposition", "DynamicModeDecompositionSettings"]


class DynamicModeDecompositionSettings(PhaseEstimationSettings):
    """Settings for the Dynamic Mode Decomposition algorithm."""

    def __init__(self):
        """Initialize the settings for ODMD.

        Args:
            shots_per_observable: The number of shots to execute per measuring a bit in the iterative phase estimation.

        """
        super().__init__()
        self._set_default("vector_dim", "int", -1, "the row number of Hankel matrix X")
        self._set_default("initial_rank_k", "int", -1, "the initial column number of Hankel matrix X")
        self._set_default("evolution_time", "float", 0.0, "Time dt in the evolution unitary U = exp(-i H dt)")
        self._set_default("shots_per_observable", "int", 40, "the number of shots to test one observable")
        self._set_default("max_rank_k", "int", 200, "Column number limit of Hankel matrix X")
        self._set_default("measure_imag", "bool", False, "flag for testing imaginary part of observables")


class DynamicModeDecomposition(PhaseEstimation):
    """ODMD algorithm implementation."""

    def __init__(
        self,
        vector_dim: int,
        initial_rank_k: int,
        evolution_time: float,
        shots_per_observable: int = 40,
        max_rank_k: int = 200,
        measure_imag: bool = False
    ):
        """Initialize DynamicModeDecomposition with the given settings.

        Args:
            vector_dim: The row number of the Hankel matrix X. Must be a positive integer.
            initial_rank_k: The initial column number of the Hankel matrix X. Must be a positive integer.
            evolution_time: Time parameter dt used in the time-evolution unitary U = exp(-i H dt)``.
            Must be a non-negative number.
            shots_per_observable: The number of shots to execute per observable measurement. Defaults to 40.
            Must be a positive integer.
            max_rank_k: The maximum column number limit of Hankel matrix X. Defaults to 200.
            Must be greater than or equal to initial_rank_k.
            measure_imag: The flag for testing imaginary part of observables. Default is False.

        Raises:
            ValueError: If vector_dim or initial_rank_k are not positive integers, if initial_rank_k exceeds
            max_rank_k, if evolution_time is negative, or if shots_per_observable is not positive.

        """
        Logger.trace_entering()
        super().__init__(evolution_time=evolution_time)
        self._settings = DynamicModeDecompositionSettings()
        self._settings.set("vector_dim", vector_dim)
        self._settings.set("initial_rank_k", initial_rank_k)
        self._settings.set("evolution_time", evolution_time)
        self._settings.set("shots_per_observable", shots_per_observable)
        self._settings.set("max_rank_k", max_rank_k)
        self._settings.set("measure_imag", measure_imag)
        self._iteration_circuits: list[Circuit] | None = None

        # check validity of inputs
        if vector_dim < 1:
            raise ValueError("vector_dim must be a positive integer.")
        if initial_rank_k < 2:
            raise ValueError("initial_rank_k must be larger than 1.")
        if initial_rank_k > max_rank_k:
            raise ValueError("initial_rank_k must be no more than max_rank_k.")
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
        vector_dim = self._settings.get("vector_dim")
        initial_rank_k = self._settings.get("initial_rank_k")
        max_rank_k = self._settings.get("max_rank_k")
        measure_imag = self._settings.get("measure_imag")
        initial_number_measurement = vector_dim + initial_rank_k
        observable_array = np.zeros(vector_dim + max_rank_k, dtype=complex)

        # Generate controlled U = exp(-iH dt) by evolution_builder, it will be connected by circuit_mapper
        ctrl_time_evol_unitary = self.create_ctrl_one_time_evol(qubit_hamiltonian, evolution_builder, [0])

        # Calculate initial vector_dim + initial_rank_k - 1 observables
        for i in range(initial_number_measurement - 1):
            observable_array[i] = self._measure_observables(
                state_preparation=state_preparation,
                num_system_qubits=qubit_hamiltonian.num_qubits,
                ctrl_time_evol_unitary=ctrl_time_evol_unitary,
                circuit_mapper=circuit_mapper,
                observable_power=i + 1,
                circuit_executor=circuit_executor,
                noise=noise,
            )

        last_phase_fraction = 1.0
        phase_fraction = 0.0
        observe_index = initial_number_measurement - 2
        rank_k = initial_rank_k - 1

        # Compose Hankel matrices
        if measure_imag:
            hankel_x = np.zeros((vector_dim, rank_k), dtype=complex)
            hankel_x_prime = np.zeros((vector_dim, rank_k), dtype=complex)
        else:
            hankel_x = np.zeros((vector_dim, rank_k))
            hankel_x_prime = np.zeros((vector_dim, rank_k))

        for v in range(rank_k):
            hankel_x[:, v] = observable_array[v : v + vector_dim]
        hankel_x_prime[:, 0 : rank_k - 1] = hankel_x[:, 1:rank_k]
        hankel_x_prime[:, rank_k - 1] = observable_array[rank_k : rank_k + vector_dim]

        while abs(phase_fraction - last_phase_fraction) > 1e-6:
            if rank_k == max_rank_k:
                Logger.warn(f"reached the limit of rank_k {max_rank_k}! ODMD iteration will stop here!")
                break

            last_phase_fraction = phase_fraction

            # Quantum part: obtain the observables (overlaps) by Hadamard tests
            rank_k += 1
            observe_index += 1
            observable_array[observe_index] = self._measure_observables(
                state_preparation=state_preparation,
                num_system_qubits=qubit_hamiltonian.num_qubits,
                ctrl_time_evol_unitary=ctrl_time_evol_unitary,
                circuit_mapper=circuit_mapper,
                observable_power=observe_index + 1,
                circuit_executor=circuit_executor,
                noise=noise,
            )

            # Classical part: update Hankel matrices, pseudo-inverse to get DMD matrix A
            hankel_x = np.column_stack((hankel_x, hankel_x_prime[:, rank_k - 2]))
            hankel_x_prime = np.column_stack((hankel_x_prime, observable_array[rank_k : rank_k + vector_dim]))

            dmd_a = hankel_x_prime @ np.linalg.pinv(hankel_x)

            # Classical part: extract ground state energy from A by diagonalization
            eigenvalues = np.linalg.eigvals(dmd_a)  # eigensolver for general matrix

            log_eigs = np.log(eigenvalues)

            Logger.info(f"log_eigs: {log_eigs}")

            # Pick the mode with smallest |real| and negative imaginary part.
            negative_imag_mask = log_eigs.imag < 0.0
            if np.any(negative_imag_mask):
                candidate_log_eigs = log_eigs[negative_imag_mask]
            else:
                candidate_log_eigs = log_eigs

            min_real_idx = np.argmin(np.abs(candidate_log_eigs.real))
            phase = candidate_log_eigs[min_real_idx].imag
            phase_fraction = phase / (2.0 * np.pi)

            Logger.info(f"iteration {rank_k - initial_rank_k + 1}, phase_fraction {phase_fraction}")

        # Create and return the result
        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=phase_fraction,
            evolution_time=self.settings().get("evolution_time"),
        )

    def create_ctrl_one_time_evol(
        self, qubit_hamiltonian: QubitHamiltonian, evolution_builder: TimeEvolutionBuilder, control_indices: list[int] | None = None
    ) -> ControlledTimeEvolutionUnitary:
        """Create the controlled one time step evolution unitary operator U = exp(-iH dt).

        Args:
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            evolution_builder: The time evolution builder to use.
            control_indices:

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

    def create_iteration_circuit(
        self,
        state_preparation: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary: ControlledTimeEvolutionUnitary,
        circuit_mapper: ControlledEvolutionCircuitMapper,
        observable_power: int,
    ) -> list[Circuit]:
        r"""Construct n-th single ODMD iteration circuit for gauging Re and Im of (<\psi|exp(-iH n dt)|\psi>).

        Args:
            state_preparation: Trial-state preparation circuit that prepares the initial state on the system qubits.
            num_system_qubits: number of qubits representing the quantum state
            ctrl_time_evol_unitary: The controlled one time step evolution unitary operator U = exp(-iH dt).
            circuit_mapper: The controlled evolution circuit mapper to use.
            observable_power: Current observable power.

        Returns:
            A quantum circuit gauging <\psi|exp(-iH n dt)|\psi>.

        """
        # Build the base circuit with registers
        ancilla = QuantumRegister(1, "ancilla")
        system_target = QuantumRegister(num_system_qubits, "system")
        classical = ClassicalRegister(1, f"c{observable_power}")
        circuit = QuantumCircuit(ancilla, system_target, classical)

        # Apply state preparation
        self._append_state_preparation(circuit, state_preparation, system_target)

        # Prepare the ancilla qubit
        control = ancilla[0]
        target_qubits = list(system_target)
        circuit.h(control)

        # Apply control U^n = exp(-iH n dt)
        ctrl_time_evol_circuit = self._create_ctrl_time_evol_circuit(
            controlled_evolution=ctrl_time_evol_unitary, power=observable_power, circuit_mapper=circuit_mapper
        )
        cu_circuit = qasm3.loads(ctrl_time_evol_circuit.qasm)
        mapping = [control, *target_qubits]
        circuit.compose(cu_circuit, qubits=mapping, inplace=True)

        circuit_imag = copy.deepcopy(circuit)

        # Final Hadamard and measurement for real part and imag part
        circuit.h(control)
        circuit.measure(control, classical[0])

        circuit_imag.sdg(control)
        circuit_imag.h(control)
        circuit_imag.measure(control, classical[0])

        Logger.info(
            f"Completed ODMD observable {observable_power}, producing circuit with {circuit.num_qubits} "
            f"qubits and {circuit.num_clbits} classical bits."
        )
        return [Circuit(qasm=qasm3.dumps(circuit)), Circuit(qasm=qasm3.dumps(circuit_imag))]

    def _append_state_preparation(
        self, circuit: QuantumCircuit, state_preparation: Circuit, system_target: QuantumRegister
    ) -> None:
        """Apply the state preparation circuit to the system qubits.

        Args:
            circuit: The quantum circuit to modify.
            state_preparation: The state preparation circuit.
            system_target: The system qubit register.

        """
        state_prep_circuit = qasm3.loads(state_preparation.qasm)
        if state_prep_circuit.num_qubits != len(system_target):
            raise ValueError(
                "state_preparation must prepare the same number of system qubits as the target register "
                f"(expected {len(system_target)}, received {state_prep_circuit.num_qubits}).",
            )
        state_prep_circuit.name = "state_preparation"
        try:
            circuit.compose(state_prep_circuit.to_gate(), qubits=system_target, inplace=True)
        except AssertionError:
            circuit.compose(state_prep_circuit, qubits=system_target, inplace=True)

    def _measure_observables(
        self,
        state_preparation: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary: ControlledTimeEvolutionUnitary,
        circuit_mapper: ControlledEvolutionCircuitMapper,
        observable_power: int,
        circuit_executor: CircuitExecutor,
        noise: QuantumErrorProfile | None = None,
    ) -> complex:
        """Measure a single ODMD observable using Hadamard tests.

        Args:
            state_preparation: Trial-state preparation circuit.
            num_system_qubits: Number of system qubits for the circuit.
            ctrl_time_evol_unitary: Controlled time-evolution unitary.
            circuit_mapper: Circuit mapper for controlled evolution.
            observable_power: Observable time-step power.
            circuit_executor: Executor to run the circuits.
            noise: Optional noise model.

        Returns:
            The complex-valued observable.

        """
        circuit_real, circuit_imag = self.create_iteration_circuit(
            state_preparation=state_preparation,
            num_system_qubits=num_system_qubits,
            ctrl_time_evol_unitary=ctrl_time_evol_unitary,
            circuit_mapper=circuit_mapper,
            observable_power=observable_power,
        )

        shots = self._settings.get("shots_per_observable")

        executor_data = circuit_executor.run(circuit_real, shots=shots, noise=noise)
        bitstring_result = executor_data.bitstring_counts
        Logger.info(
            f"measured real part of observable power {observable_power}, "
            f"{bitstring_result.get('0', 0)} zeros, {bitstring_result.get('1', 0)} ones"
        )
        observable_real = (bitstring_result.get("0", 0) - bitstring_result.get("1", 0)) / shots

        if self._settings.get("measure_imag"):
            executor_data = circuit_executor.run(circuit_imag, shots=shots, noise=noise)
            bitstring_result = executor_data.bitstring_counts
            Logger.info(
                f"measured imag part of observable power {observable_power}, "
                f"{bitstring_result.get('0', 0)} zeros, {bitstring_result.get('1', 0)} ones"
            )
            observable_imag = (bitstring_result.get("0", 0) - bitstring_result.get("1", 0)) / shots
            observable_value = observable_real + 1j * observable_imag
        else:
            observable_value = observable_real
        
        Logger.info(f"measured observable power {observable_power}, value {observable_value}")
        return observable_value

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "ODMD"
