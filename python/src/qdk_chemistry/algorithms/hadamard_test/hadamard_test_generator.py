"""Hadamard test circuit generator implementations for Q# and Qiskit backends."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.algorithms.hadamard_test.base import HadamardTestGenerator
from qdk_chemistry.data import Circuit
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["QiskitHadamardGenerator", "QsharpHadamardGenerator"]


class QsharpHadamardGenerator(HadamardTestGenerator):
    """Hadamard test circuit generator based on Q# framework."""

    def __init__(
        self,
    ):
        """Initialize QsharpHadamardGenerator."""
        Logger.trace_entering()
        super().__init__()

    def _run_impl(
        self,
        state_preparation: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary_circuit: Circuit,
        test_basis: str = "X",
    ) -> Circuit:
        r"""Build a Hadamard test circuit using the Q# backend.

        Args:
            state_preparation: Circuit that prepares the trial state on system qubits.
            num_system_qubits: Number of qubits in the system register.
            ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.
            test_basis: Measurement basis for the control qubit. Supported values are ``"X"``, ``"Y"``, and ``"Z"``.

        Returns:
            Circuit containing compiled and rendered Q# Hadamard test artifacts.

        """
        if test_basis not in {"X", "Y", "Z"}:
            raise ValueError("currently test_basis can only be X, Y or Z.")

        try:
            state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        except AttributeError as err:
            raise ValueError("Input state_preparation cannot be used for QsharpHadamardGenerator.") from err

        try:
            ctrl_evol_op = ctrl_time_evol_unitary_circuit._qsharp_op  # noqa: SLF001
        except AttributeError as err:
            raise ValueError(
                "Input ctrl_time_evol_unitary_circuit cannot be used for QsharpHadamardGenerator."
            ) from err

        systems = list(i for i in range(1, num_system_qubits + 1))
        hadamard_test_qsc = qsharp.circuit(
            QSHARP_UTILS.HadamardTest.MakeHadamardCircuit,
            state_prep_op,
            ctrl_evol_op,
            test_basis,
            0,
            systems,
        )
        hadamard_test_qir = qsharp.compile(
            QSHARP_UTILS.HadamardTest.MakeHadamardCircuit,
            state_prep_op,
            ctrl_evol_op,
            test_basis,
            0,
            systems,
        )

        Logger.info("Completed qsharp circuit for real observable measurement.")
        return Circuit(qsharp=hadamard_test_qsc, qir=hadamard_test_qir)

    def name(self) -> str:
        """Return the name of the QsharpHadamardGenerator algorithm."""
        return "qsharp_hadamard_generator"


class QiskitHadamardGenerator(HadamardTestGenerator):
    """Hadamard test circuit generator based on Qiskit framework."""

    def __init__(
        self,
    ):
        """Initialize QiskitHadamardGenerator.

        Raises:
            ModuleNotFoundError: If Qiskit is not installed.

        """
        Logger.trace_entering()
        super().__init__()

    def _run_impl(
        self,
        state_preparation: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary_circuit: Circuit,
        test_basis: str = "X",
    ) -> Circuit:
        r"""Build a Hadamard test circuit using the Qiskit backend.

        Args:
            state_preparation: Circuit that prepares the trial state on system qubits.
            num_system_qubits: Number of qubits in the system register.
            ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.
            test_basis: Measurement basis for the control qubit. Supported values are ``"X"``, ``"Y"``, and ``"Z"``.

        Returns:
            Circuit containing the OpenQASM3 representation of the Qiskit Hadamard test circuit.

        """
        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3
        except ModuleNotFoundError as err:
            raise ModuleNotFoundError(
                "Qiskit is required to use QiskitHadamardGenerator. Install qiskit or use QsharpHadamardGenerator."
            ) from err

        # Build the base circuit with registers.
        ancilla = QuantumRegister(1, "ancilla")
        system_target = QuantumRegister(num_system_qubits, "system")
        classical = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(ancilla, system_target, classical)

        # Apply state preparation.
        try:
            state_prep_qc = state_preparation.get_qiskit_circuit()
        except (AttributeError, RuntimeError) as err:
            raise ValueError("Input state_preparation cannot be used for QiskitHadamardGenerator.") from err
        circuit.append(state_prep_qc.to_gate(), system_target)

        # Prepare ancilla and apply controlled time evolution.
        control = ancilla[0]
        target_qubits = list(system_target)

        circuit.h(control)

        try:
            ctrl_evol_qc = ctrl_time_evol_unitary_circuit.get_qiskit_circuit()
        except (AttributeError, RuntimeError) as err:
            raise ValueError(
                "Input ctrl_time_evol_unitary_circuit cannot be used for QiskitHadamardGenerator."
            ) from err
        circuit.append(ctrl_evol_qc.to_gate(), [control, *target_qubits])

        # Final basis rotation and measurement on the control qubit.
        if test_basis == "X":
            circuit.h(control)
        elif test_basis == "Y":
            circuit.sdg(control)
            circuit.h(control)
        elif test_basis != "Z":  # if it is "Z", then do nothing; otherwise, raise error
            raise ValueError("currently test_basis can only be X, Y or Z.")
        circuit.measure(control, classical[0])

        Logger.info("Completed qiskit circuit for real observable measurement.")
        return Circuit(qasm=qasm3.dumps(circuit))

    def name(self) -> str:
        """Return the name of the QiskitHadamardGenerator algorithm."""
        return "qiskit_hadamard_generator"
