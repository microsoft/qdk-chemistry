"""QDK/Chemistry sequence structure controlled circuit mapper."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.circuit_mapper.pauli_sequence_mapper import PauliSequenceMapper
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, _pauli_evolution_parameters

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
    When the formula declares disjoint layers, their controlled rotations share two
    rotation rounds per layer. The declared boundaries are used without regrouping;
    formulas without layer metadata retain term-by-term controlled evolution.

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
            evo_params = QSHARP_UTILS.PauliExp.StructuredSparseRepPauliExpParams(
                conjugatingGroups=[
                    PauliSequenceMapper._encode_group(term)  # noqa: SLF001 - shared structured lowering
                    for term in unitary_container.conjugating_terms
                ],
                stepBlocks=[
                    PauliSequenceMapper._encode_block(term)  # noqa: SLF001 - shared structured lowering
                    for term in unitary_container.step_terms
                ],
                repetitions=unitary_container.step_reps,
            )
            program = QSHARP_UTILS.ControlledPauliExp.MakeStructuredRepControlledPauliExpCircuit
            controlled_unitary_op = QSHARP_UTILS.ControlledPauliExp.MakeStructuredRepControlledPauliExpOp(evo_params)
            parameters = {"params": evo_params}
        else:
            evo_params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(**_pauli_evolution_parameters(unitary_container))
            layer_offsets = list(unitary_container.layer_offsets or ())
            program = QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpCircuit
            controlled_unitary_op = QSHARP_UTILS.ControlledPauliExp.MakeRepControlledPauliExpOp(
                evo_params, layer_offsets
            )
            parameters = {"params": evo_params, "layerOffsets": layer_offsets}

        parameters.update(control=control_indices[0], systems=target_indices)
        qsharp_factory = QsharpFactoryData(
            program=program,
            parameter=parameters,
        )

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=controlled_unitary_op)
