"""Iterative phase estimation circuit builder.

This module implements the circuit-building component of the Kitaev-style iterative
quantum phase estimation (IQPE) algorithm. It constructs the iteration circuits
without executing them, enabling standalone resource estimation and circuit preview.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import (
    Circuit,
    QubitHamiltonian,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import QpeCircuitBuilder, QpeCircuitBuilderSettings

__all__: list[str] = ["IterativeQpeCircuitBuilder", "IterativeQpeCircuitBuilderSettings"]


class IterativeQpeCircuitBuilderSettings(QpeCircuitBuilderSettings):
    """Settings for the Iterative Phase Estimation Builder."""

    def __init__(self):
        """Initialize the settings for the Iterative Phase Estimation Builder."""
        super().__init__()
        self._set_default("phase_correction", "double", 0.0, "The accumulated phase feedback from prior iterations.")
        self._set_default(
            "num_iteration", "int", -1, "The specific iteration to build. Default to -1 to build all iterations."
        )


class IterativeQpeCircuitBuilder(QpeCircuitBuilder):
    """Iterative Phase Estimation circuit builder.

    Constructs the quantum circuits for each IQPE iteration without executing them.
    Can be used standalone for resource estimation or composed inside IterativePhaseEstimation.

    """

    def __init__(self, num_bits: int = -1, phase_correction: float = 0.0, num_iteration: int = -1):
        """Initialize the IterativeQpeCircuitBuilder.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            phase_correction: The accumulated phase feedback from prior iterations. Default to 0.0.
            num_iteration: The specific iteration to build. Default to -1 (build all iterations).

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = IterativeQpeCircuitBuilderSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("phase_correction", phase_correction)
        self._settings.set("num_iteration", num_iteration)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[Circuit]:
        """Build IQPE iteration circuits.

        Uses settings ``phase_correction`` (default 0.0) and ``num_iteration``
        (default -1). When ``num_iteration`` is negative, all iteration circuits
        are returned. When positive, only the circuit for that single iteration
        (0-based) is returned.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build circuits.

        Returns:
            A list of quantum circuits, one per phase bit iteration (or a single-element
            list when ``num_iteration`` is set to a specific iteration index).

        Raises:
            ValueError: If ``num_iteration`` >= ``num_bits``.

        """
        num_bits = self.settings().get("num_bits")
        phase_correction = self.settings().get("phase_correction")
        num_iteration = self.settings().get("num_iteration")

        if num_iteration >= num_bits:
            raise ValueError(f"num_iteration ({num_iteration}) must be less than num_bits ({num_bits}).")

        iterations = [num_iteration] if num_iteration >= 0 else range(num_bits)
        circuits: list[Circuit] = []
        for iteration in iterations:
            circuit = self._create_iteration_circuit(
                state_preparation=state_preparation,
                qubit_hamiltonian=qubit_hamiltonian,
                iteration=iteration,
                total_iterations=num_bits,
                phase_correction=phase_correction,
            )
            circuits.append(circuit)

        Logger.warn(
            f"Builder iteration circuit with dummy phase_correction={phase_correction}. "
            f"This is non-adaptive and intended for resource estimation only."
        )
        return circuits

    def _create_iteration_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        iteration: int,
        total_iterations: int,
        phase_correction: float = 0.0,
    ) -> Circuit:
        """Construct a single IQPE iteration circuit.

        Args:
            state_preparation: Trial-state preparation circuit that prepares the initial state on the system qubits.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            iteration: Current iteration index (0-based), where 0 corresponds to the most-significant bit.
            total_iterations: Total number of phase bits to measure across all iterations.
            phase_correction: Feedback phase angle to apply before controlled unitary, defaults to 0.0.

        Returns:
            A quantum circuit implementing one IQPE iteration.

        """
        _validate_iteration_inputs(iteration, total_iterations)
        num_system_qubits = qubit_hamiltonian.num_qubits
        power = 2 ** (total_iterations - iteration - 1)
        ctrl_unitary_circuit = self._create_controlled_circuit(qubit_hamiltonian, power)

        if state_preparation._qsharp_op and ctrl_unitary_circuit._qsharp_op:  # noqa: SLF001
            return self._create_circuit_from_qsharp_op(
                state_preparation, ctrl_unitary_circuit, phase_correction, num_system_qubits
            )

        if state_preparation.get_qiskit_circuit() and ctrl_unitary_circuit.get_qiskit_circuit():
            return self._create_circuit_from_qiskit(state_preparation, ctrl_unitary_circuit, phase_correction)

        raise RuntimeError(
            "Failed to create iteration circuit: Q# operations or Qiskit dependencies are not available."
        )

    def _create_circuit_from_qsharp_op(
        self,
        state_preparation: Circuit,
        controlled_unitary_circuit: Circuit,
        phase_correction: float,
        num_system_qubits: int,
    ) -> Circuit:
        """Create a Circuit object from a Q# operation.

        Args:
            state_preparation: Circuit object containing a Q# operation for state preparation.
            controlled_unitary_circuit: Circuit object containing a Q# operation for the controlled unitary.
            phase_correction: Feedback phase angle to apply before controlled unitary.
            num_system_qubits: Number of system qubits.

        Returns:
            A Circuit object representing the IQPE iteration.

        """
        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        ctrl_unitary_op = controlled_unitary_circuit._qsharp_op  # noqa: SLF001
        iterative_parameters = {
            "statePrep": state_prep_op,
            "repControlledEvolution": ctrl_unitary_op,
            "accumulatePhase": phase_correction,
            "control": 0,
            "systems": [i + 1 for i in range(num_system_qubits)],
        }
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.IterativePhaseEstimation.MakeIQPECircuit,
                parameter=iterative_parameters,
            )
        )

    def _create_circuit_from_qiskit(
        self, state_preparation: Circuit, controlled_unitary_circuit: Circuit, phase_correction: float
    ) -> Circuit:
        """Create a Circuit object from Qiskit QuantumCircuit objects.

        Args:
            state_preparation: Circuit object containing a Qiskit QuantumCircuit for state preparation.
            controlled_unitary_circuit: Circuit object containing a Qiskit QuantumCircuit for the controlled unitary.
            phase_correction: Feedback phase angle to apply before controlled unitary.

        Returns:
            A Circuit object representing the IQPE iteration.

        """
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3  # noqa: PLC0415

        state_prep_qc = state_preparation.get_qiskit_circuit()
        ctrl_unitary_qc = controlled_unitary_circuit.get_qiskit_circuit()
        ancilla = QuantumRegister(1, "ancilla")
        system_target = QuantumRegister(state_prep_qc.num_qubits, "system")
        classical = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(ancilla, system_target, classical)
        circuit.append(state_prep_qc.to_gate(), system_target)
        control = ancilla[0]
        target_qubits = list(system_target)
        circuit.h(control)

        # Apply phase correction if provided
        if phase_correction:
            circuit.rz(phase_correction, control)

        # Append the controlled unitary circuit
        circuit.append(ctrl_unitary_qc.to_gate(), [control, *target_qubits])
        circuit.h(control)
        circuit.measure(control, classical[0])

        return Circuit(qasm=qasm3.dumps(circuit))

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "iterative"


def _validate_iteration_inputs(iteration: int, total_iterations: int) -> None:
    """Validate iteration parameters for IQPE circuit construction.

    Args:
        iteration: The current iteration index (0-based).
        total_iterations: The total number of iterations.

    """
    if total_iterations <= 0:
        raise ValueError("total_iterations must be a positive integer.")
    if iteration < 0 or iteration >= total_iterations:
        raise ValueError(
            f"iteration index {iteration} is outside the valid range [0, {total_iterations - 1}].",
        )
