r"""Vacuum-annihilating term grouper: group Pauli terms whose amplitudes cancel on the vacuum.

The **flipped-qubit set** of a Pauli string is the set of positions carrying :math:`X` or
:math:`Y`, i.e. the qubits whose :math:`|0\rangle` and :math:`|1\rangle` get exchanged.  On the
all-zero state,

.. math::

    c\,P\,|0\ldots0\rangle = c\,i^{n_Y}\,|b_F\rangle,

with :math:`F` the flipped-qubit set, :math:`n_Y` the number of :math:`Y` factors, and
:math:`b_F` the bit string with exactly the qubits in :math:`F` set.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper, TermGrouperSettings
from qdk_chemistry.data import FlatPartition, LayeredPartition
from qdk_chemistry.data.qubit_operator.containers.sparse_pauli_decomposition import _sparse_terms

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qdk_chemistry.data import QubitOperator

__all__ = ["VacuumAnnihilatingTermGrouper", "VacuumAnnihilatingTermGrouperSettings"]

_COMMUTATION_CHUNK_SIZE = 256


def _index_word(word: dict[int, str], index: int, by_qubit: dict[int, dict[str, dict[int, int]]]) -> None:
    """Add a local term ordinal to bounded masks without allocating by physical qubit or total term count."""
    chunk, offset = divmod(index, _COMMUTATION_CHUNK_SIZE)
    bit = 1 << offset
    for qubit, axis in word.items():
        chunks = by_qubit.setdefault(qubit, {}).setdefault(axis, {})
        chunks[chunk] = chunks.get(chunk, 0) | bit


def _anticommuting_masks(word: dict[int, str], by_qubit: dict[int, dict[str, dict[int, int]]]) -> dict[int, int]:
    """Resolve odd-overlap parity per indexed term, before projecting conflicts to owning groups."""
    odd: dict[int, int] = {}
    for qubit, axis in word.items():
        for other_axis, chunks in by_qubit.get(qubit, {}).items():
            if axis != other_axis:
                for chunk, mask in chunks.items():
                    odd[chunk] = odd.get(chunk, 0) ^ mask
    return {chunk: mask for chunk, mask in odd.items() if mask}


def _commuting_group(
    indices: Iterable[int], words: list[dict[int, str]], keys: list[tuple[frozenset[int], int]]
) -> bool:
    """Certify full commutation; same flipped-set/Y-parity words commute, so only cross-bucket pairs are checked."""
    buckets: dict[tuple[frozenset[int], int], list[int]] = {}
    for index in indices:
        buckets.setdefault(keys[index], []).append(index)
    by_qubit: dict[int, dict[str, dict[int, int]]] = {}
    position = 0
    for members in buckets.values():
        if any(_anticommuting_masks(words[index], by_qubit) for index in members):
            return False
        for index in members:
            _index_word(words[index], position, by_qubit)
            position += 1
    return True


def _vacuum_residuals(
    indices: Iterable[int], keys: list[tuple[frozenset[int], int]], amplitudes: list[float]
) -> dict[tuple[frozenset[int], int], float]:
    """Sum each real or imaginary vacuum component, excluding the diagonal eigenphase."""
    values: dict[tuple[frozenset[int], int], list[float]] = {}
    for index in indices:
        if keys[index][0]:
            values.setdefault(keys[index], []).append(amplitudes[index])
    return {key: math.fsum(pending) for key, pending in values.items()}


class VacuumAnnihilatingTermGrouperSettings(TermGrouperSettings):
    """Settings for the :class:`VacuumAnnihilatingTermGrouper`.

    Attributes:
        tolerance: Absolute coefficient residual per flipped-set/Y-parity component, shared across output groups.

    """

    def __init__(self):
        """Initialise the settings for VacuumAnnihilatingTermGrouper."""
        super().__init__()
        self._set_default(
            "tolerance",
            "double",
            1e-9,
            "Absolute tolerance on the residual vacuum amplitude of each "
            "flipped-set/Y-parity component, shared across output groups.",
        )


class VacuumAnnihilatingTermGrouper(TermGrouper):
    r"""Construct commuting, vacuum-preserving groups with disjoint Pauli layers.

    An existing compatible :class:`~qdk_chemistry.data.LayeredPartition` is retained exactly.
    Otherwise compatible outer groups are preserved where the residual budget permits, and
    cancellation partners split across incompatible groups are reunited. Existing disjoint
    layers are retained whole or as ordered fragments; flat or unpartitioned inputs acquire
    layers from their full Pauli supports. Repartitioning can change the Trotter approximation,
    but never the Hamiltonian's terms, coefficients, or stored order.

    Only words with the same flipped-qubit set can cancel on the vacuum. Within one Y-count
    parity these words commute, and their signed real coefficients must sum to zero within
    ``tolerance``. Exact zero-sum cuts form indivisible cancellation blocks, which are packed
    together only when all cross-block Pauli pairs commute. The blocks stay within one outer
    group but may span several layers: XX and YY on one bond must occupy different layers.
    Diagonal I/Z words may leave a vacuum eigenphase, which the consumer corrects.

    Tolerance is a coefficient-level budget for each flipped-set/Y-parity component, shared
    across the output groups. It is not a bound on a repeated circuit's leakage; the final
    emitted product formula must still be validated after filtering and angle scaling.

    Regroup operators carrying flat partitions before adding them to layered output; operator addition
    only merges partitions of the same kind.

    """

    def __init__(self):
        """Initialise the VacuumAnnihilatingTermGrouper."""
        super().__init__()
        self._settings = VacuumAnnihilatingTermGrouperSettings()

    def name(self) -> str:
        """Return ``vacuum_annihilating`` as the algorithm name."""
        return "vacuum_annihilating"

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Return a copy of ``qubit_hamiltonian`` partitioned into vacuum-annihilating groups.

        Args:
            qubit_hamiltonian: Hamiltonian to partition.

        Returns:
            QubitOperator: A new operator carrying a :class:`~qdk_chemistry.data.LayeredPartition`.

        Raises:
            ValueError: If ``tolerance`` is negative or not finite.
            ValueError: If a coefficient is not a finite real number.
            ValueError: If terms flipping the same qubits do not cancel on the vacuum.
            ValueError: If a supplied layer has overlapping Pauli supports.
            TypeError: If the incoming partition is neither flat nor layered.

        """
        tolerance = self._settings.get("tolerance")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"tolerance must be finite and non-negative, got {tolerance}.")

        coefficients = np.asarray(qubit_hamiltonian.coefficients)
        if np.any(np.iscomplex(coefficients)) or not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "VacuumAnnihilatingTermGrouper requires finite, real coefficients, since vacuum "
                "amplitudes are compared against a real cancellation tolerance."
            )

        words = [dict(word) for word, _ in _sparse_terms(qubit_hamiltonian.get_container())]
        keys: list[tuple[frozenset[int], int]] = []
        amplitudes: list[float] = []
        for index, word in enumerate(words):
            n_y = sum(axis == "Y" for axis in word.values())
            keys.append((frozenset(q for q, axis in word.items() if axis in ("X", "Y")), n_y % 2))
            amplitudes.append(float(coefficients[index].real) * (-1) ** (n_y // 2))

        incoming = qubit_hamiltonian.term_partition
        old_groups: list[list[int]] = []
        if isinstance(incoming, LayeredPartition):
            for group_index, layers in enumerate(incoming.groups):
                old_groups.append([i for layer in layers for i in layer])
                for layer_index, layer in enumerate(layers):
                    occupied: set[int] = set()
                    for i in layer:
                        if not occupied.isdisjoint(words[i]):
                            raise ValueError(
                                f"Group {group_index}, layer {layer_index} must have disjoint Pauli supports."
                            )
                        occupied.update(words[i])
        elif isinstance(incoming, FlatPartition):
            old_groups = [list(group) for group in incoming.groups]
        elif incoming is not None:
            raise TypeError(f"Expected FlatPartition or LayeredPartition, got {type(incoming).__name__}.")
        traversal = incoming.all_indices() if incoming is not None else list(range(len(words)))
        rank = {index: position for position, index in enumerate(traversal)}
        old_owner = {i: group_index for group_index, group in enumerate(old_groups) for i in group}
        buckets: dict[tuple[frozenset[int], int], list[int]] = {}
        for i in traversal:
            buckets.setdefault(keys[i], []).append(i)
        for (flipped, _), residual in _vacuum_residuals(traversal, keys, amplitudes).items():
            if abs(residual) > tolerance:
                raise ValueError(
                    "VacuumAnnihilatingTermGrouper cannot certify the Hamiltonian: terms "
                    f"flip qubits {sorted(flipped)} and leave an uncancelled vacuum amplitude of {residual:.3g}. "
                    "The computational vacuum is not an eigenstate within tolerance in this encoding."
                )

        residuals = [_vacuum_residuals(group, keys, amplitudes) for group in old_groups]
        preserved = {
            g
            for g, group in enumerate(old_groups)
            if all(abs(value) <= tolerance for value in residuals[g].values()) and _commuting_group(group, words, keys)
        }
        # Approximate preserved groups and the repair pool share one budget per component.
        # Recompute from original coefficients, rather than subtracting rounded group sums.
        repair = [i for i in traversal if old_owner.get(i) not in preserved]
        repair_residuals = _vacuum_residuals(repair, keys, amplitudes)
        spent: dict[tuple[frozenset[int], int], list[float]] = {}
        for g in preserved:
            for key, value in residuals[g].items():
                spent.setdefault(key, []).append(abs(value))
        for key, value in repair_residuals.items():
            spent.setdefault(key, []).append(abs(value))
        if any(math.fsum(values) > tolerance for values in spent.values()):
            preserved = {g for g in preserved if all(value == 0.0 for value in residuals[g].values())}
            repair = [i for i in traversal if old_owner.get(i) not in preserved]
        if isinstance(incoming, LayeredPartition) and len(preserved) == len(old_groups):
            return self._with_partition(qubit_hamiltonian, incoming)

        kept = sorted(preserved)
        groups = [old_groups[g] for g in kept]
        owners = {i: g for g, group in enumerate(groups) for i in group}
        repair_set = set(repair)
        blocks: list[list[int]] = []
        for (flipped, _), members in buckets.items():
            indices = [i for i in members if i in repair_set]
            if not indices:
                continue
            if not flipped:
                blocks.append(indices)
                continue
            # Only exact cuts are safe: an approximately-zero prefix may strand its remainder.
            current: list[int] = []
            pending: list[float] = []
            for index in indices:
                current.append(index)
                pending.append(amplitudes[index])
                if math.fsum(pending) == 0.0:
                    blocks.append(current)
                    current, pending = [], []
            if current:
                blocks.append(current)

        placed: dict[int, dict[str, dict[int, int]]] = {}
        placed_indices: list[int] = []
        owner_masks: dict[int, dict[int, int]] = {}
        affinities: dict[int, set[int]] = {}
        for block in sorted(blocks, key=lambda members: min(rank[i] for i in members)):
            forbidden: set[int] = set()
            preferred: set[int] = set()
            for i in block:
                for chunk, mask in _anticommuting_masks(words[i], placed).items():
                    remaining = mask
                    while remaining:
                        bit = remaining & -remaining
                        position = chunk * _COMMUTATION_CHUNK_SIZE + bit.bit_length() - 1
                        owner = owners[placed_indices[position]]
                        forbidden.add(owner)
                        remaining &= ~owner_masks[owner][chunk]
                preferred.update(affinities.get(old_owner.get(i, -1), ()))
            available = preferred - forbidden
            group_index = (
                min(available)
                if available
                else next((g for g in range(len(kept), len(groups)) if g not in forbidden), len(groups))
            )
            if group_index == len(groups):
                groups.append([])
            groups[group_index].extend(block)
            for i in block:
                owners[i] = group_index
                if i in old_owner:
                    affinities.setdefault(old_owner[i], set()).add(group_index)
                chunk, offset = divmod(len(placed_indices), _COMMUTATION_CHUNK_SIZE)
                masks = owner_masks.setdefault(group_index, {})
                masks[chunk] = masks.get(chunk, 0) | 1 << offset
                _index_word(words[i], len(placed_indices), placed)
                placed_indices.append(i)

        group_layers: list[list[tuple[int, ...]]] = [[] for _ in groups]
        if isinstance(incoming, LayeredPartition):
            for g, original in enumerate(kept):
                group_layers[g] = list(incoming.groups[original])
            # Intersections retain the producer's layers without an O(groups * terms) rescan.
            for original, layers in enumerate(incoming.groups):
                if original in preserved:
                    continue
                for layer in layers:
                    fragments: dict[int, list[int]] = {}
                    for i in layer:
                        fragments.setdefault(owners[i], []).append(i)
                    for g, fragment in fragments.items():
                        group_layers[g].append(tuple(fragment))
        else:
            for g, group in enumerate(groups):
                disjoint_layers: list[list[int]] = []
                occupied_layers: dict[int, set[int]] = {}
                first_free: dict[int, int] = {}
                for i in sorted(group, key=rank.__getitem__):
                    # Every smaller layer is occupied on a qubit attaining this lower bound.
                    layer_index = max((first_free.get(qubit, 0) for qubit in words[i]), default=0)
                    while any(layer_index in occupied_layers.get(qubit, ()) for qubit in words[i]):
                        layer_index += 1
                    if layer_index == len(disjoint_layers):
                        disjoint_layers.append([])
                    disjoint_layers[layer_index].append(i)
                    for qubit in words[i]:
                        occupied = occupied_layers.setdefault(qubit, set())
                        occupied.add(layer_index)
                        cursor = first_free.get(qubit, 0)
                        while cursor in occupied:
                            cursor += 1
                        first_free[qubit] = cursor
                group_layers[g] = [tuple(layer) for layer in disjoint_layers]

        spent = {}
        for group in groups:
            for key, value in _vacuum_residuals(group, keys, amplitudes).items():
                spent.setdefault(key, []).append(abs(value))
        if any(math.fsum(values) > tolerance for values in spent.values()):
            raise ValueError("Vacuum residuals after grouping exceed the shared tolerance.")
        order = sorted(range(len(groups)), key=lambda g: min((rank[i] for i in groups[g]), default=len(words)))
        partition = LayeredPartition(
            strategy="vacuum_annihilating", groups=tuple(tuple(group_layers[g]) for g in order)
        )
        return self._with_partition(qubit_hamiltonian, partition)
