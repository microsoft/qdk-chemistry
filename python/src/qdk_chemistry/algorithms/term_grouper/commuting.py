"""Commutation-based term groupers (full and qubit-wise)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Callable

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper
from qdk_chemistry.data import FlatPartition, QubitHamiltonian
from qdk_chemistry.utils.pauli_commutation import do_pauli_labels_commute, do_pauli_labels_qw_commute

__all__ = ["FullCommutingTermGrouper", "QubitWiseCommutingTermGrouper"]


def _greedy_color_partition(
    pauli_strings: list[str],
    commutes: Callable[[str, str], bool],
) -> tuple[tuple[int, ...], ...]:
    """Partition Pauli labels into commuting groups via greedy graph coloring.

    Conceptually builds the non-commutation graph (an edge between every pair
    of labels that do *not* commute) and colors it greedily, assigning each
    label the lowest color not used by any non-commuting neighbour.  Labels
    sharing a color form a commuting group.

    Args:
        pauli_strings: Pauli labels to partition.
        commutes: Predicate returning ``True`` when two labels commute.

    Returns:
        Tuple of groups; each group is a tuple of indices into ``pauli_strings``.

    """
    groups: list[list[int]] = []
    group_labels: list[list[str]] = []

    for i, pauli_str in enumerate(pauli_strings):
        placed = False
        for group, labels in zip(groups, group_labels, strict=True):
            if all(commutes(pauli_str, existing) for existing in labels):
                group.append(i)
                labels.append(pauli_str)
                placed = True
                break
        if not placed:
            groups.append([i])
            group_labels.append([pauli_str])

    return tuple(tuple(g) for g in groups)


class FullCommutingTermGrouper(TermGrouper):
    """Group terms by full Pauli commutation (``[P_i, P_j] = 0``).

    The resulting :class:`~qdk_chemistry.data.FlatPartition` stores a
    :class:`~qdk_chemistry.data.QubitHamiltonian` partition where every pair of
    terms in the same group commutes globally.  Useful for Trotter-style
    decompositions, which can exponentiate a commuting block as a single
    ordered product without splitting error.

    """

    def name(self) -> str:
        """Return ``commuting`` as the algorithm name."""
        return "commuting"

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> QubitHamiltonian:
        """Return a copy of ``qubit_hamiltonian`` with a full-commutation partition.

        Args:
            qubit_hamiltonian: Hamiltonian to partition.

        Returns:
            QubitHamiltonian: New instance carrying a :class:`~qdk_chemistry.data.FlatPartition` with strategy ``"commuting"``.

        """
        groups = _greedy_color_partition(qubit_hamiltonian.pauli_strings, do_pauli_labels_commute)
        partition = FlatPartition(strategy="commuting", groups=groups)
        return QubitHamiltonian(
            pauli_strings=list(qubit_hamiltonian.pauli_strings),
            coefficients=qubit_hamiltonian.coefficients.copy(),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            term_partition=partition,
        )


class QubitWiseCommutingTermGrouper(TermGrouper):
    """Group terms by qubit-wise commutation.

    Two labels qubit-wise commute when, on every qubit position, the two
    single-qubit Paulis individually commute (i.e. one is identity or both are
    equal).  All members of a group can be measured in a single basis, which
    is the property exploited by :class:`~qdk_chemistry.algorithms.QdkEnergyEstimator`
    for measurement-cost reduction.

    """

    def name(self) -> str:
        """Return ``qubit_wise_commuting`` as the algorithm name."""
        return "qubit_wise_commuting"

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> QubitHamiltonian:
        """Return a copy of ``qubit_hamiltonian`` with a qubit-wise commutation partition.

        Args:
            qubit_hamiltonian: Hamiltonian to partition.

        Returns:
            QubitHamiltonian: New instance carrying a :class:`~qdk_chemistry.data.FlatPartition` with strategy ``"qubit_wise_commuting"``.

        """
        groups = _greedy_color_partition(qubit_hamiltonian.pauli_strings, do_pauli_labels_qw_commute)
        partition = FlatPartition(strategy="qubit_wise_commuting", groups=groups)
        return QubitHamiltonian(
            pauli_strings=list(qubit_hamiltonian.pauli_strings),
            coefficients=qubit_hamiltonian.coefficients.copy(),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            term_partition=partition,
        )
