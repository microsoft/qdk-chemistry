"""NetworkX-backed term groupers using DSATUR graph coloring."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import networkx as nx

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper
from qdk_chemistry.data import FlatPartition
from qdk_chemistry.data.qubit_operator.containers.sparse_pauli_decomposition import SparsePauliDecompositionContainer
from qdk_chemistry.utils.pauli_commutation import (
    do_pauli_labels_commute,
    do_pauli_labels_qw_commute,
    do_pauli_maps_commute,
    do_pauli_maps_qw_commute,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from qdk_chemistry.data import QubitOperator

__all__ = ["NxFullCommutingTermGrouper", "NxQubitWiseCommutingTermGrouper"]

_Term = TypeVar("_Term")


def _dsatur_commutation_grouping(
    pauli_strings: Sequence[_Term],
    commutes: Callable[[_Term, _Term], bool],
) -> tuple[tuple[int, ...], ...]:
    """Partition Pauli terms using networkx DSATUR graph coloring.

    Builds the non-commutation graph (vertices = Pauli terms, edges between
    non-commuting pairs) and colors it using the saturation-largest-first
    (DSATUR) strategy, which greedily colors the vertex with the highest
    saturation degree (most distinct colors among neighbours).

    Args:
        pauli_strings: Pauli labels or sparse Pauli maps to partition.
        commutes: Predicate returning ``True`` when two terms commute.

    Returns:
        Tuple of groups; each group is a tuple of indices into ``pauli_strings``.

    """
    n = len(pauli_strings)
    if n == 0:
        return ()

    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(1, n):
        for j in range(i):
            if not commutes(pauli_strings[i], pauli_strings[j]):
                g.add_edge(i, j)

    coloring = nx.coloring.greedy_color(g, strategy="saturation_largest_first")

    color_to_indices: dict[int, list[int]] = {}
    for node, color in sorted(coloring.items()):
        color_to_indices.setdefault(color, []).append(node)

    return tuple(tuple(indices) for indices in color_to_indices.values())


def _dsatur_partition(
    qubit_hamiltonian: QubitOperator,
    label_commutes: Callable[[str, str], bool],
    map_commutes: Callable[[dict[int, str], dict[int, str]], bool],
) -> tuple[tuple[int, ...], ...]:
    """Color the non-commutation graph, comparing sparse words without building labels."""
    container = qubit_hamiltonian.get_container()
    if isinstance(container, SparsePauliDecompositionContainer):
        maps = [dict(word) for word, _ in container.iter_sparse_terms()]
        return _dsatur_commutation_grouping(maps, map_commutes)
    return _dsatur_commutation_grouping(qubit_hamiltonian.pauli_strings, label_commutes)


class NxFullCommutingTermGrouper(TermGrouper):
    """Group terms by full Pauli commutation using networkx DSATUR.

    Uses the saturation-largest-first (DSATUR) graph coloring strategy
    from networkx, which typically produces fewer groups than the built-in
    greedy first-fit algorithm.

    """

    def name(self) -> str:
        """Return ``nx_commuting`` as the algorithm name."""
        return "nx_commuting"

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Return a copy of ``qubit_hamiltonian`` with a full-commutation DSATUR partition.

        Args:
            qubit_hamiltonian: Hamiltonian to partition.

        Returns:
            QubitOperator: New instance with a ``FlatPartition`` (strategy ``"nx_commuting"``).

        """
        groups = _dsatur_partition(qubit_hamiltonian, do_pauli_labels_commute, do_pauli_maps_commute)
        partition = FlatPartition(strategy="nx_commuting", groups=groups)
        return self._with_partition(qubit_hamiltonian, partition)


class NxQubitWiseCommutingTermGrouper(TermGrouper):
    """Group terms by qubit-wise commutation using networkx DSATUR.

    Uses the saturation-largest-first (DSATUR) graph coloring strategy
    from networkx, which typically produces fewer groups than the built-in
    greedy first-fit algorithm.

    """

    def name(self) -> str:
        """Return ``nx_qubit_wise_commuting`` as the algorithm name."""
        return "nx_qubit_wise_commuting"

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Return a copy with a qubit-wise commutation DSATUR partition.

        Args:
            qubit_hamiltonian: Hamiltonian to partition.

        Returns:
            QubitOperator: New instance with a ``FlatPartition`` (strategy ``"nx_qubit_wise_commuting"``).

        """
        groups = _dsatur_partition(qubit_hamiltonian, do_pauli_labels_qw_commute, do_pauli_maps_qw_commute)
        partition = FlatPartition(strategy="nx_qubit_wise_commuting", groups=groups)
        return self._with_partition(qubit_hamiltonian, partition)
