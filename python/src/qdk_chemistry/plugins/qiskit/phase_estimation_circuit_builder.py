"""Qiskit-based phase estimation circuit builder.

This module provides the Qiskit-specific implementation of the standard QPE circuit builder,
extending the base StandardQpeCircuitBuilder with Qiskit QuantumCircuit support.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3
from qiskit.synthesis.qft.qft_decompose_full import synth_qft_full

from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import (
    IterativeQpeCircuitBuilder,
    QpeCircuitBuilderSettings,
    StandardQpeCircuitBuilder,
)
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.iterative_builder import (
    _validate_iteration_inputs,
)
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitHamiltonian
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitIterativeQpeCircuitBuilder", "QiskitStandardQpeCircuitBuilder"]


class QiskitStandardQpeCircuitBuilderSettings(QpeCircuitBuilderSettings):
    """Settings for the Qiskit Standard Phase Estimation Builder."""

    def __init__(self):
        """Initialize the settings for the Qiskit Standard Phase Estimation Builder."""
        super().__init__()
        self._set_default("qft_do_swaps", "bool", True, "Whether to apply swap gates in the QFT.")


class QiskitStandardQpeCircuitBuilder(StandardQpeCircuitBuilder):
    """Qiskit-based standard (QFT-based) phase estimation circuit builder.

    Extends StandardQpeCircuitBuilder to add support for Qiskit QuantumCircuit objects.
    Constructs the full QPE circuit (state prep, controlled unitaries, inverse QFT,
    measurements) without executing it.

    """

    def __init__(
        self,
        num_bits: int = -1,
        qft_do_swaps: bool = True,
        circuit_mapper: AlgorithmRef | None = None,
        unitary_builder: AlgorithmRef | None = None,
    ):
        """Initialize QiskitStandardQpeCircuitBuilder with the given settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            qft_do_swaps: Whether to apply swap gates in the QFT. Defaults to True.
            circuit_mapper: Optional algorithm reference for the circuit mapper.
            unitary_builder: Optional algorithm reference for the unitary builder.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits, circuit_mapper=circuit_mapper, unitary_builder=unitary_builder)
        self._settings = QiskitStandardQpeCircuitBuilderSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("qft_do_swaps", qft_do_swaps)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[Circuit]:
        """Build the standard QPE circuit.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.

        Returns:
            A list containing a single standard QPE circuit.

        """
        Logger.trace_entering()
        circuit = self.build_circuit(state_preparation, qubit_hamiltonian)
        Logger.info("Built standard QPE circuit using Qiskit.")
        return [circuit]

    def build_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> Circuit:
        """Build the standard QPE circuit using Qiskit.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.

        Returns:
            The constructed QPE quantum circuit.

        """
        Logger.trace_entering()
        num_bits = self._settings.get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")
        ancilla = QuantumRegister(num_bits, "ancilla")
        system = QuantumRegister(qubit_hamiltonian.num_qubits, "system")
        classical = ClassicalRegister(num_bits, "c")
        qc = QuantumCircuit(ancilla, system, classical)

        Logger.debug(f"Creating traditional QPE circuit with {num_bits} ancilla qubits and measurements.")
        state_prep = state_preparation.get_qiskit_circuit()
        if state_prep.num_qubits != qubit_hamiltonian.num_qubits:
            raise ValueError(
                "state_preparation must prepare the same number of system qubits as the Hamiltonian "
                f"(expected {qubit_hamiltonian.num_qubits}, received {state_prep.num_qubits}).",
            )

        qc.compose(state_prep, qubits=system, inplace=True)

        for idx in range(num_bits):
            qc.h(ancilla[idx])

        for ancilla_idx in range(num_bits):
            power = 2**ancilla_idx
            self._append_controlled_unitary(
                circuit=qc,
                qubit_hamiltonian=qubit_hamiltonian,
                control_qubit=ancilla[ancilla_idx],
                target_qubits=system,
                power=power,
            )

        inverse_qft = synth_qft_full(
            num_bits, do_swaps=self._settings.get("qft_do_swaps"), inverse=True, name="Inverse QFT"
        )
        qc.compose(inverse_qft.to_gate(), qubits=ancilla, inplace=True)
        qc.measure(ancilla, classical)
        Logger.debug(f"Completed standard QPE circuit with {qc.num_qubits} qubits.")

        return Circuit(qasm3.dumps(qc))

    def _append_controlled_unitary(
        self,
        circuit: QuantumCircuit,
        qubit_hamiltonian: QubitHamiltonian,
        control_qubit: int,
        target_qubits: list,
        *,
        power: int,
    ) -> None:
        """Apply the controlled unitary to the circuit.

        Args:
            circuit: The quantum circuit to modify.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            control_qubit: The control qubit.
            target_qubits: List of target qubits.
            power: The power to which the controlled unitary is raised.

        """
        ctrl_unitary_circuit = self._create_controlled_circuit(qubit_hamiltonian=qubit_hamiltonian, power=power)
        cu_circuit = ctrl_unitary_circuit.get_qiskit_circuit()

        mapping = [control_qubit, *target_qubits]
        circuit.compose(cu_circuit, qubits=mapping, inplace=True)

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "qiskit_standard"


class QiskitIterativeQpeCircuitBuilderSettings(QpeCircuitBuilderSettings):
    """Settings for the Qiskit Iterative Phase Estimation Builder."""

    def __init__(self):
        """Initialize the settings for the Qiskit Iterative Phase Estimation Builder."""
        super().__init__()
        self._set_default("phase_correction", "double", 0.0, "The accumulated phase feedback from prior iterations.")
        self._set_default(
            "num_iteration", "int", -1, "The specific iteration to build. Default to -1 to build all iterations."
        )


class QiskitIterativeQpeCircuitBuilder(IterativeQpeCircuitBuilder):
    """Qiskit-based iterative phase estimation circuit builder.

    Extends IterativeQpeCircuitBuilder to add support for Qiskit QuantumCircuit objects.

    """

    def __init__(
        self,
        num_bits: int = -1,
        phase_correction: float = 0.0,
        num_iteration: int = -1,
    ):
        """Initialize QiskitIterativeQpeCircuitBuilder with the given settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            phase_correction: The accumulated phase feedback from prior iterations. Default to 0.0.
            num_iteration: The specific iteration to build. Default to -1 (build all iterations).

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = QiskitIterativeQpeCircuitBuilderSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("phase_correction", phase_correction)
        self._settings.set("num_iteration", num_iteration)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[Circuit]:
        """Build IQPE iteration circuits using Qiskit.

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
        Logger.trace_entering()
        num_bits = self._settings.get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")
        phase_correction = self._settings.get("phase_correction")
        num_iteration = self._settings.get("num_iteration")

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

        Logger.info(f"Built {len(circuits)} iteration circuit(s) with phase_correction={phase_correction}.")
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
        """Construct a single IQPE iteration circuit using Q# or Qiskit.

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
        return "qiskit_iterative"
