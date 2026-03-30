"""Qiskit Hadamard test generator implementation."""
# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.hadamard_test.base import HadamardTest, HadamardTestBasis
from qdk_chemistry.data import Circuit
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitHadamardTest"]


class QiskitHadamardTest(HadamardTest):
    """Hadamard test circuit generator based on Qiskit framework."""

    def __init__(
        self,
    ):
        """Initialize QiskitHadamardTest."""
        Logger.trace_entering()
        super().__init__()

    def _build_hadamard_test_circuit(
        self,
        state_preparation_circuit: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary_circuit: Circuit,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
    ) -> Circuit:
        r"""Build a Hadamard test circuit using the Qiskit backend.

        Currently, the function only accepts the controlled unitary circuit whose index of ancilla qubit is 0.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            num_system_qubits: Number of qubits in the system register.
            ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X`` or ``HadamardTestBasis.Y``).

        Returns:
            Circuit containing the OpenQASM3 representation of the Qiskit Hadamard test circuit.

        Raises:
            ModuleNotFoundError: If Qiskit is not installed.

        """
        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3  # noqa: PLC0415
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Qiskit is required to use QiskitHadamardTest. Install qiskit or use QdkHadamardTest."
            ) from err

        # Build the base circuit with registers.
        ancilla = QuantumRegister(1, "ancilla")
        system_target = QuantumRegister(num_system_qubits, "system")
        classical = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(ancilla, system_target, classical)

        # Apply state preparation.
        try:
            state_prep_qc = state_preparation_circuit.get_qiskit_circuit()
        except (AttributeError, RuntimeError) as err:
            raise ValueError("Input state_preparation_circuit cannot be used for QiskitHadamardTest.") from err
        circuit.append(state_prep_qc.to_gate(), system_target)

        # Prepare ancilla and apply controlled time evolution.
        control = ancilla[0]
        target_qubits = list(system_target)

        circuit.h(control)

        try:
            ctrl_evol_qc = ctrl_time_evol_unitary_circuit.get_qiskit_circuit()
        except (AttributeError, RuntimeError) as err:
            raise ValueError("Input ctrl_time_evol_unitary_circuit cannot be used for QiskitHadamardTest.") from err
        circuit.append(ctrl_evol_qc.to_gate(), [control, *target_qubits])

        # Final basis rotation and measurement on the control qubit.
        if test_basis is HadamardTestBasis.X:
            circuit.h(control)
        elif test_basis is HadamardTestBasis.Y:
            circuit.sdg(control)
            circuit.h(control)
        circuit.measure(control, classical[0])

        Logger.debug(f"Completed qiskit circuit for measurement on {test_basis.value} basis.")
        return Circuit(qasm=qasm3.dumps(circuit))

    def name(self) -> str:
        """Return the name of the QiskitHadamardTest algorithm."""
        return "qiskit_hadamard_test"
