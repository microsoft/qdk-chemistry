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
        """Initialize the settings for ODMD

        Args:
            shots_per_observable: The number of shots to execute per measuring a bit in the iterative phase estimation.

        """
        super().__init__()
        self._set_default(
            "shots_per_observable",
            "int",
            3,
            "The number of shots to execute per observable (overlap \bra{\psi}\exp{-i\hat{H} n dt}\ket{\psi})).",
        )
        self._set_default("vector_dim", "int", -1, "the row number of Hankel matrix X")
        self._set_default("rank_k", "int", -1, "the column number of Hankel matrix X")
        self._set_default("shots_per_observable", "int", 20, "the number of shots to test one observable")

class DynamicModeDecomposition(PhaseEstimation):
    """ODMD algorithm implementation."""

    def __init__(
        self,
        vector_dim: int,
        rank_k: int,
        evolution_time: float,
        shots_per_observable: int = 20
    ):
        Logger.trace_entering()
        super().__init__(evolution_time=evolution_time)
        self._settings = DynamicModeDecompositionSettings()
        self._settings.set("vector_dim", vector_dim)
        self._settings.set("rank_k", rank_k)
        self._settings.set("evolution_time", evolution_time)
        self._settings.set("shots_per_observable", shots_per_observable)
        self._iteration_circuits: list[Circuit] | None = None

        # check validity of inputs
        if (vector_dim < 1 or rank_k < 1):
            raise ValueError("vector_dim and rank_k must be a positive integer.")
        if (evolution_time < 0.0):
            raise ValueError("evolution_time must be a positive double number.")
        if (shots_per_observable < 1): # currently we use classical Hadamard test for measuring observable
            raise ValueError("shots_per_observable must be a positive integer.")

    def _run_impl(self,
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
        # quantum part: obtain the observables (overlaps) by Hadamard tests
        vector_dim = self._settings.get("vector_dim")
        rank_k = self._settings.get("rank_k")
        total_number_measurement = vector_dim + rank_k
        observable_array = np.zeros(total_number_measurement)
        shots = self._settings.get("shots_per_observable")

        ctrl_time_evol_unitary = self.create_ctrl_one_time_evol(
            qubit_hamiltonian, 
            evolution_builder
        )

        for i in range (0, total_number_measurement):
            circuit_real = self.create_iteration_circuit(
                state_preparation,
                qubit_hamiltonian.num_qubits,
                ctrl_time_evol_unitary,
                circuit_mapper,
                i + 1
            )

            executor_data = circuit_executor.run(circuit_real, shots=shots, noise=noise)
            bitstring_result = executor_data.bitstring_counts # a dictionary
            Logger.info(f"measured real part of {i + 1} observable, {bitstring_result.get("0", 0)} zeros, {bitstring_result.get("1", 0)} ones")
            observable_real = (bitstring_result.get("0", 0) - bitstring_result.get("1", 0)) / shots
            Logger.info(f"measured real part of {i + 1} observable, value {observable_real}")
            observable_array[i] = observable_real

        # classical part: generate Hankel matrices, pseudo-inverse to get DMD matrix A
        X = np.zeros((vector_dim, rank_k))
        for v in range(0, rank_k):
            X[:, v] = observable_array[v : v + vector_dim]
        X_prime = np.zeros((vector_dim, rank_k))
        X_prime[:, 0:rank_k - 1] = X[:, 1:rank_k]
        X_prime[:, rank_k - 1] = observable_array[rank_k : rank_k + vector_dim]
        A = X_prime @ np.linalg.pinv(X)

        # classical part: extract ground state energy from A by diagonalization
        eigenvalues = np.linalg.eigvals(A) # eigensolver for general matrix

        log_Eigs = np.log(eigenvalues)

        print(log_Eigs)

        # Pick the mode with smallest |real| and negative imaginary part.
        negative_imag_mask = log_Eigs.imag < 0.0
        if np.any(negative_imag_mask):
            candidate_log_Eigs = log_Eigs[negative_imag_mask]
        else:
            candidate_log_Eigs = log_Eigs

        min_real_idx = np.argmin(np.abs(candidate_log_Eigs.real))
        phase = candidate_log_Eigs[min_real_idx].imag
        phase_fraction = phase / (2.0 * np.pi)

        print(phase_fraction)
        # Create and return the result
        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=phase_fraction,
            evolution_time=self.settings().get("evolution_time"),
        )

    def create_ctrl_one_time_evol(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        evolution_builder: TimeEvolutionBuilder,
        control_indices: list = [0]
    ) -> ControlledTimeEvolutionUnitary:
        """Create the controlled one time step evolution unitary operator U = exp(-iH dt)

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

        ctrl_one_time_step_evol_unitary = ControlledTimeEvolutionUnitary(
            time_evolution_unitary=time_evol_unitary,
            control_indices=control_indices,
        )

        return ctrl_one_time_step_evol_unitary

    def create_iteration_circuit(
        self,
        state_preparation: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary: ControlledTimeEvolutionUnitary,
        circuit_mapper: ControlledEvolutionCircuitMapper,
        observable_index: int,
    ) -> Circuit:
        """Construct n-th single ODMD iteration circuit for gauging Re(<\psi|exp(-iH n dt)|\psi>). Since we only approximate ground state energy and don't find the state, we don't measure the imaginary part for now.

        Args:
            state_preparation: Trial-state preparation circuit that prepares the initial state on the system qubits.
            ctrl_time_evol_unitary: The controlled one time step evolution unitary operator U = exp(-iH dt).
            circuit_mapper: The controlled evolution circuit mapper to use.
            observable_index: Current observable index (0-based).

        Returns:
            A quantum circuit gauging <\psi|exp(-iH n dt)|\psi>.

        """
        # Build the base circuit with registers
        ancilla = QuantumRegister(1, "ancilla")
        system_target = QuantumRegister(num_system_qubits, "system")
        classical = ClassicalRegister(1, f"c{observable_index}") # ??
        circuit = QuantumCircuit(ancilla, system_target, classical)

        # Apply state preparation
        self._append_state_preparation(circuit, state_preparation, system_target)

        # Prepare the ancilla qubit
        control = ancilla[0]
        target_qubits = list(system_target)
        circuit.h(control)

        # Apply control U^n = exp(-iH n dt)
        ctrl_time_evol_circuit = self._create_ctrl_time_evol_circuit(
            controlled_evolution=ctrl_time_evol_unitary,
            power=observable_index,
            circuit_mapper=circuit_mapper
        )
        cu_circuit = qasm3.loads(ctrl_time_evol_circuit.qasm)
        mapping = [control, *target_qubits] # ??
        circuit.compose(cu_circuit, qubits=mapping, inplace=True)

        # Final Hadamard and measurement for real part
        circuit.h(control)
        circuit.measure(control, classical[0])

        Logger.info(
            f"Completed ODMD observable {observable_index + 1}, producing circuit with {circuit.num_qubits} "
            f"qubits and {circuit.num_clbits} classical bits."
        )
        return Circuit(qasm=qasm3.dumps(circuit))

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

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "ODMD"