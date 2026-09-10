"""QDK/Chemistry sequence structure controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any, cast

from qdk import qsharp

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    BatchedExponentiatedPauliTerm,
    ConjugatedExponentiatedPauliTerm,
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import ControlledCircuitMapper

__all__: list[str] = ["ControlledPauliSequenceMapper"]


class ControlledPauliSequenceMapper(ControlledCircuitMapper):
    r"""Controlled evolution circuit mapper using Pauli product formula term sequences.

    Given a time-evolution operator expressed as a Pauli product formula
    :math:`U(t) \approx \left[ U_{\mathrm{step}}(t / r) \right]^{r}`, this mapper constructs
    a controlled version of :math:`U(t)` using the following pattern:

    1. Each Pauli operator :math:`P_j` is basis-rotated into the :math:`Z` basis.
    2. Qubits involved in :math:`P_j` are entangled into a sequence using CNOT gates.
    3. A controlled :math:`R_z` rotation implements
        :math:`e^{-i\,\theta_j\,P_j} \;\rightarrow\; \text{CRZ}(2 \theta_j)`.
    4. The basis rotations and entangling operations are uncomputed.

    Terms are handed to Q# in a sparse encoding: each term contributes only the qubit
    indices it acts on and their Pauli axes, rather than one Pauli per system qubit.

    Notes:
        * Currently supports only single-control-qubit scenarios.
        * Requires a ``PauliProductFormulaContainer`` for the time evolution unitary.

    """

    def __init__(self):
        """Initialize the PauliSequenceMapper."""
        super().__init__()

    def name(self) -> str:
        """Return the algorithm name."""
        return "pauli_sequence"

    def type_name(self) -> str:
        """Return controlled_circuit_mapper as the algorithm type name."""
        return "controlled_circuit_mapper"

    @staticmethod
    def _encode_pauli_terms(
        pauli_terms: list[dict[int, str]],
    ) -> tuple[list[list[int]], list[list[qsharp.Pauli]]]:
        """Encode sparse Pauli maps as parallel Q# index and axis arrays."""
        pauli_indices: list[list[int]] = []
        pauli_ops: list[list[qsharp.Pauli]] = []
        for pauli_term in pauli_terms:
            indices: list[int] = []
            ops: list[qsharp.Pauli] = []
            for index, pauli in pauli_term.items():
                indices.append(index)
                ops.append(getattr(qsharp.Pauli, pauli))
            pauli_indices.append(indices)
            pauli_ops.append(ops)
        return pauli_indices, pauli_ops

    @classmethod
    def _encode_flat_terms(cls, terms: list[ExponentiatedPauliTerm], repetitions: int) -> Any:
        """Encode an unstructured repeated Pauli sequence."""
        pauli_indices, pauli_ops = cls._encode_pauli_terms([term.pauli_term for term in terms])
        return QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(
            pauliIndices=pauli_indices,
            pauliOps=pauli_ops,
            pauliCoefficients=[term.angle for term in terms],
            repetitions=repetitions,
        )

    @classmethod
    def _encode_group(cls, term: ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm) -> Any:
        """Encode one plain exponential or equal-angle batch."""
        pauli_terms = [term.pauli_term] if isinstance(term, ExponentiatedPauliTerm) else list(term.pauli_terms)
        pauli_indices, pauli_ops = cls._encode_pauli_terms(pauli_terms)
        return QSHARP_UTILS.PauliExp.SparsePauliExpGroupParams(
            pauliIndices=pauli_indices,
            pauliOps=pauli_ops,
            angle=term.angle,
        )

    @classmethod
    def _encode_block(
        cls,
        term: ExponentiatedPauliTerm | BatchedExponentiatedPauliTerm | ConjugatedExponentiatedPauliTerm,
    ) -> Any:
        """Encode a direct group or a structured Q# ``within``/``apply`` block."""
        if isinstance(term, ConjugatedExponentiatedPauliTerm):
            within_groups = [cls._encode_group(group) for group in term.within_terms]
            apply_groups = [cls._encode_group(group) for group in term.apply_terms]
        else:
            within_groups = []
            apply_groups = [cls._encode_group(term)]
        return QSHARP_UTILS.PauliExp.ConjugatedSparsePauliExpParams(
            withinGroups=within_groups,
            applyGroups=apply_groups,
        )

    def _run_impl(self, unitary: UnitaryRepresentation) -> Circuit:
        r"""Construct a quantum circuit implementing the controlled unitary.

        Args:
            unitary: The unitary representation containing the Hamiltonian
                and evolution parameters. Control and target indices are
                read from settings.

        Returns:
            Circuit: A quantum circuit implementing the controlled unitary :math:`U`
            where :math:`U` is the time evolution operator :math:`\exp(-i H t)`.

        Raises:
            ValueError: If the unitary container type is not supported.
            ValueError: If multiple control qubits are provided.

        """
        unitary_container = unitary.get_container()
        if not isinstance(unitary_container, PauliProductFormulaContainer):
            raise ValueError(
                f"The {unitary.get_container_type()} container type is not supported. "
                "PauliSequenceMapper only supports PauliProductFormula container for the unitary."
            )

        control_indices = self._get_control_indices()
        if len(control_indices) != 1:
            raise ValueError("PauliSequenceMapper currently only supports a single control qubit.")

        target_indices = self._get_target_indices(unitary)

        structured = bool(unitary_container.conjugating_terms) or any(
            not isinstance(term, ExponentiatedPauliTerm) for term in unitary_container.step_terms
        )
        if structured:
            mapped_params = QSHARP_UTILS.PauliExp.StructuredSparseRepPauliExpParams(
                conjugatingGroups=[self._encode_group(term) for term in unitary_container.conjugating_terms],
                stepBlocks=[self._encode_block(term) for term in unitary_container.step_terms],
                repetitions=unitary_container.step_reps,
            )
            program = QSHARP_UTILS.ControlledPauliExp.MakeStructuredRepControlledPauliExpCircuit
            controlled_unitary_op = QSHARP_UTILS.ControlledPauliExp.MakeStructuredRepControlledPauliExpOp(mapped_params)
        else:
            terms = cast("list[ExponentiatedPauliTerm]", unitary_container.step_terms)
            mapped_params = self._encode_flat_terms(terms, unitary_container.step_reps)
            program = QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpCircuit
            controlled_unitary_op = QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpOp(mapped_params)

        qsharp_factory = QsharpFactoryData(
            program=program,
            parameter={"params": mapped_params, "control": control_indices[0], "systems": target_indices},
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=controlled_unitary_op)
