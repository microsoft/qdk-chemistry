"""QDK/Chemistry SELECT oracle mappers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod

from qdk import qsharp

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.block_encoding import Select
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = ["MultiControlledSelectMapper", "SelectMapper", "SelectMapperFactory"]


class SelectMapper(Algorithm):
    r"""Abstract base class for SELECT oracle mappers.

    Subclasses implement a specific strategy for constructing the SELECT circuit
    from a :class:`~qdk_chemistry.data.unitary_representation.containers.block_encoding.Select` data object.
    """

    def type_name(self) -> str:
        """Return the algorithm type name.

        Returns:
            str: The type name ``"select_mapper"``.

        """
        return "select_mapper"

    @abstractmethod
    def _run_impl(self, select: Select) -> Circuit:
        """Create a SELECT circuit from a Select data object."""


class SelectMapperFactory(AlgorithmFactory):
    """Factory class for creating SELECT oracle mapper instances."""

    def algorithm_type_name(self) -> str:
        """Return select_mapper as the algorithm type name."""
        return "select_mapper"

    def default_algorithm_name(self) -> str:
        """Return multi_controlled_select as the default algorithm name."""
        return "multi_controlled_select"


class MultiControlledSelectMapper(SelectMapper):
    r"""Multi-controlled SELECT oracle mapper for LCU block encodings.

    Converts a :class:`~qdk_chemistry.data.unitary_representation.containers.block_encoding.Select` data object
    into a Q# callable that implements the SELECT oracle:

    .. math::

        \mathrm{SELECT} = \sum_j |j\rangle\langle j| \otimes \mathrm{sign}(\alpha_j) \cdot P_j

    Each Pauli string :math:`P_j` is applied controlled on the ancilla state
    :math:`|j\rangle`, with a phase correction for negative coefficients.

    """

    def __init__(self):
        """Initialize the MultiControlledSelectMapper."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"multi_controlled_select"``.

        """
        return "multi_controlled_select"

    def _run_impl(self, select: Select) -> Circuit:
        """Create a SELECT circuit from a Select data object.

        Converts each controlled operation's Pauli string into Q# ``Pauli`` enums
        and packages them with sign phases into a ``DefaultSelectParams`` struct,
        then wraps it as a Circuit containing the Q# callable and factory.

        Args:
            select: The SELECT oracle data object containing controlled operations,
                phases, and qubit layout.

        Returns:
            Circuit: A Circuit wrapping the Q# SELECT callable and factory.

        """
        pauli_terms: list[list[qsharp.Pauli]] = []
        for op in select.controlled_operations:
            base_paulis = [qsharp.Pauli.I] * select.num_target_qubits
            for i, pauli_char in enumerate(op.operation):
                if pauli_char != "I":
                    base_paulis[i] = getattr(qsharp.Pauli, pauli_char)
            pauli_terms.append(base_paulis)
        phases = [int(s) for s in select.phases]
        select_params = QSHARP_UTILS.Select.PauliSelectParams(pauliTerms=pauli_terms, signs=phases)
        qsharp_op = QSHARP_UTILS.Select.MakeSelectOp(select_params)
        factory_params = {
            "params": select_params,
            "numSelectQubits": select.num_prepare_qubits,
            "numTargetQubits": select.num_target_qubits,
        }
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.Select.MakeSelectCircuit,
            parameter=factory_params,
        )
        return Circuit(qsharp_op=qsharp_op, qsharp_factory=qsharp_factory)
