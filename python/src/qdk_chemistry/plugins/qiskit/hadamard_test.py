"""Qiskit Hadamard test circuit builder implementation."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.hadamard_test.base import HadamardTestBasis
from qdk_chemistry.algorithms.hadamard_test.circuit_builder.base import HadamardTestCircuitBuilder
from qdk_chemistry.data import Circuit, UnitaryRepresentation
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitHadamardTestCircuitBuilder"]


class QiskitHadamardTestCircuitBuilder(HadamardTestCircuitBuilder):
    """Hadamard test circuit builder based on the Qiskit framework."""

    def __init__(
        self,
    ):
        """Initialize QiskitHadamardTestCircuitBuilder."""
        Logger.trace_entering()
        super().__init__()

    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        unitary: UnitaryRepresentation,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
        num_ancilla_qubits: int = 0,
    ) -> Circuit:
        r"""Build a Hadamard test circuit using the Qiskit backend.

        The target unitary is mapped into a controlled evolution circuit internally; the
        resulting controlled unitary circuit must place its ancilla qubit at index 0.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            unitary: Unitary representation :math:`U` (e.g. a time-evolution unitary built with the desired power).
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X``, ``HadamardTestBasis.Y``, or
              ``HadamardTestBasis.Z``).
            num_ancilla_qubits: Number of ancilla qubits needed by the controlled evolution (0 if none).

        Returns:
            Circuit containing the OpenQASM3 representation of the Qiskit Hadamard test circuit.

        Raises:
            ModuleNotFoundError: If Qiskit is not installed.

        """
        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3  # noqa: PLC0415
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Qiskit is required to use QiskitHadamardTestCircuitBuilder. "
                "Install qiskit or use QdkHadamardTestCircuitBuilder."
            ) from err

        num_system_qubits = unitary.get_num_qubits()
        ctrl_time_evol_unitary_circuit = self._create_controlled_circuit(unitary)

        # Build the base circuit with registers.
        ancilla = QuantumRegister(1, "ancilla")
        system_target = QuantumRegister(num_system_qubits, "system")
        registers = [ancilla, system_target]
        if num_ancilla_qubits > 0:
            extra_ancillas = QuantumRegister(num_ancilla_qubits, "ancilla_extra")
            registers.append(extra_ancillas)
        classical = ClassicalRegister(1, "c")
        registers.append(classical)
        circuit = QuantumCircuit(*registers)

        # Apply state preparation.
        try:
            state_prep_qc = state_preparation_circuit.get_qiskit_circuit()
        except (AttributeError, RuntimeError) as err:
            raise ValueError(
                "Input state_preparation_circuit cannot be used for QiskitHadamardTestCircuitBuilder."
            ) from err
        circuit.append(state_prep_qc.to_gate(), system_target)

        # Prepare ancilla and apply controlled time evolution.
        control = ancilla[0]
        target_qubits = list(system_target)
        if num_ancilla_qubits > 0:
            target_qubits += list(extra_ancillas)

        circuit.h(control)

        try:
            ctrl_evol_qc = ctrl_time_evol_unitary_circuit.get_qiskit_circuit()
        except (AttributeError, RuntimeError) as err:
            raise ValueError(
                "Input ctrl_time_evol_unitary_circuit cannot be used for QiskitHadamardTestCircuitBuilder."
            ) from err
        circuit.append(ctrl_evol_qc.to_gate(), [control, *target_qubits])

        # Final basis rotation and measurement on the control qubit.
        if test_basis is HadamardTestBasis.X:
            circuit.h(control)
        elif test_basis is HadamardTestBasis.Y:
            circuit.sdg(control)
            circuit.h(control)
        elif test_basis is HadamardTestBasis.Z:
            pass
        else:
            raise ValueError(f"Unsupported test basis: {test_basis}.")
        circuit.measure(control, classical[0])

        Logger.debug(f"Completed qiskit circuit for measurement on {test_basis.value} basis.")
        return Circuit(qasm=qasm3.dumps(circuit))

    def name(self) -> str:
        """Return the name of the QiskitHadamardTestCircuitBuilder algorithm."""
        return "qiskit"
