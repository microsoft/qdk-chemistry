"""Batched controlled Pauli-product-formula circuit mapping."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, _pauli_evolution_parameters

from .controlled_pauli_sequence_mapper import ControlledPauliSequenceMapper

__all__ = ["ControlledBatchedPauliSequenceMapper"]


class ControlledBatchedPauliSequenceMapper(ControlledPauliSequenceMapper):
    """Control a Pauli product formula by batching consecutive disjoint supports.

    Uses the same input and single-control settings as
    :class:`~qdk_chemistry.algorithms.controlled_circuit_mapper.ControlledPauliSequenceMapper`.
    Each batch shares two rounds of rotations, surrounded by local basis and
    parity computations. Identity terms retain their phase on the control.

    Term order, signed angles, and symbolic repetitions are unchanged. No
    partition metadata is required in the product-formula container: disjoint
    supports are detected from its ordered sparse terms. No extra ancillas
    are allocated. Shared-control CNOTs still have serial two-qubit depth.
    """

    def name(self) -> str:
        """Return the opt-in batched mapper's registry name."""
        return "batched_pauli_sequence"

    def _map_sequence(self, container: PauliProductFormulaContainer, control: int, systems: list[int]) -> Circuit:
        """Interleave controlled-rotation gadgets only within disjoint data supports."""
        parameters = _pauli_evolution_parameters(container)
        batch_offsets = [0]
        occupied: set[int] = set()
        for term_index, support in enumerate(parameters["pauliIndices"]):
            if not support:
                # Identity phases act on the control, not on a parity target.
                if batch_offsets[-1] != term_index:
                    batch_offsets.append(term_index)
                batch_offsets.append(term_index + 1)
                occupied.clear()
            else:
                if not occupied.isdisjoint(support):
                    batch_offsets.append(term_index)
                    occupied.clear()
                occupied.update(support)
        if batch_offsets[-1] != len(container.step_terms):
            batch_offsets.append(len(container.step_terms))

        params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(**parameters)
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.BatchedControlledPauliExp.MakeRepControlledPauliExpCircuit,
                parameter={"params": params, "batchOffsets": batch_offsets, "control": control, "systems": systems},
            ),
            qsharp_op=QSHARP_UTILS.BatchedControlledPauliExp.MakeRepControlledPauliExpOp(params, batch_offsets),
        )
