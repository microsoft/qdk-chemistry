"""Q# Hadamard test circuit builder implementation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.hadamard_test.circuit_builder.base import HadamardTestCircuitBuilder
from qdk_chemistry.algorithms.hadamard_test.hadamard_test import HadamardTestBasis, basis_to_qsharp_pauli
from qdk_chemistry.data import AlgorithmRef, Circuit, UnitaryRepresentation
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["QdkHadamardTestCircuitBuilder"]


class QdkHadamardTestCircuitBuilder(HadamardTestCircuitBuilder):
    """Hadamard test circuit builder based on the Q# framework."""

    def __init__(
        self,
        controlled_circuit_mapper: AlgorithmRef | None = None,
        test_basis: HadamardTestBasis = HadamardTestBasis.X,
    ):
        """Initialize QdkHadamardTestCircuitBuilder.

        Args:
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.
            test_basis: Measurement basis for the control qubit (``HadamardTestBasis.X`` or ``HadamardTestBasis.Y``).

        """
        Logger.trace_entering()
        super().__init__(
            controlled_circuit_mapper=controlled_circuit_mapper,
            test_basis=test_basis,
        )

    def _run_impl(
        self,
        state_preparation_circuit: Circuit,
        unitary: UnitaryRepresentation,
    ) -> Circuit:
        r"""Build a Hadamard test circuit using the Q# backend.

        The target unitary is mapped into a controlled evolution circuit internally; the
        resulting controlled unitary circuit must place its control qubit at index 0.

        Args:
            state_preparation_circuit: Circuit that prepares the trial state on system qubits.
            unitary: Unitary representation :math:`U` (e.g. a time-evolution unitary built with the desired power).

        Returns:
            Circuit containing compiled and rendered Q# Hadamard test artifacts.

        """
        test_basis = HadamardTestBasis(self._settings.get("test_basis"))
        Logger.debug(f"Building qsharp circuit for measurement on {test_basis.value} basis.")

        qsharp_basis = basis_to_qsharp_pauli(test_basis)

        num_system_qubits = unitary.get_num_qubits()
        ctrl_time_evol_unitary_circuit = self._create_controlled_circuit(unitary)

        state_prep_op = state_preparation_circuit._qsharp_op  # noqa: SLF001
        if state_prep_op is None:
            raise ValueError("Input state_preparation_circuit is not a Q# callable circuit.")

        ctrl_evol_op = ctrl_time_evol_unitary_circuit._qsharp_op  # noqa: SLF001
        if ctrl_evol_op is None:
            raise ValueError("Input ctrl_time_evol_unitary_circuit is not a Q# callable circuit.")

        hadamard_parameters = {
            "statePrep": state_prep_op,
            "repControlledEvolution": ctrl_evol_op,
            "testBasis": qsharp_basis,
            "numSystemQubits": num_system_qubits,
        }

        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.HadamardTest.HadamardTest,
                parameter=hadamard_parameters,
            )
        )

    def name(self) -> str:
        """Return the name of the QdkHadamardTestCircuitBuilder algorithm."""
        return "qdk"
