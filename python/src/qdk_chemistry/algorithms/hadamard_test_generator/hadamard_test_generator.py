"""Hadamard test circuit generator implementations for Q# and Qiskit backends."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.hadamard_test_generator.base import (
    HadamardTest,
    HadamardTestBasis,
    basis_to_qsharp_pauli,
)
from qdk_chemistry.data import Circuit
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["QdkHadamardTest"]


class QdkHadamardTest(HadamardTest):
    """Hadamard test circuit generator based on Q# framework."""

    def __init__(
        self,
    ):
        """Initialize QdkHadamardTest."""
        Logger.trace_entering()
        super().__init__()

    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        num_system_qubits: int,
        ctrl_time_evol_unitary_circuit: Circuit,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
    ) -> Circuit:
        r"""Build a Hadamard test circuit using the Q# backend.

        Currently, the function only accepts the controlled unitary circuit whose index of ancilla qubit is 0.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            num_system_qubits: Number of qubits in the system register.
            ctrl_time_evol_unitary_circuit: Controlled evolution circuit implementing the target unitary.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X`` or ``HadamardTestBasis.Y``).

        Returns:
            Circuit containing compiled and rendered Q# Hadamard test artifacts.

        """
        if not isinstance(test_basis, HadamardTestBasis):
            raise TypeError("test_basis must be an instance of HadamardTestBasis.")

        Logger.debug(f"Building qsharp circuit for measurement on {test_basis.value} basis.")

        qsharp_basis = basis_to_qsharp_pauli(test_basis)

        state_prep_op = state_preparation_circuit._qsharp_op  # noqa: SLF001
        if state_prep_op is None:
            raise ValueError("Input state_preparation_circuit cannot be used for QdkHadamardTest.")

        ctrl_evol_op = ctrl_time_evol_unitary_circuit._qsharp_op  # noqa: SLF001
        if ctrl_evol_op is None:
            raise ValueError("Input ctrl_time_evol_unitary_circuit cannot be used for QdkHadamardTest.")

        hadamard_parameters = {
            "statePrep": state_prep_op,
            "repControlledEvolution": ctrl_evol_op,
            "testBasis": qsharp_basis,
            "control": 0,
            "systems": [i + 1 for i in range(num_system_qubits)],
        }

        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.HadamardTest.HadamardTest,
                parameter=hadamard_parameters,
            )
        )

    def name(self) -> str:
        """Return the name of the QdkHadamardTest algorithm."""
        return "qdk_hadamard_test"
