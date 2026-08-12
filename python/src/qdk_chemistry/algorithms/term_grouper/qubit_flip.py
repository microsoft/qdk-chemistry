r"""Qubit-flip term grouper: group Pauli terms that flip the same qubits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper
from qdk_chemistry.data import FlatPartition, QubitOperator
from qdk_chemistry.utils.pauli_qubit_flip import pauli_label_flipped_qubits

__all__ = ["QubitFlipTermGrouper"]


class QubitFlipTermGrouper(TermGrouper):
    r"""Group Pauli terms that flip the same set of qubits.

    Terms land in the same group when they carry :math:`X`/:math:`Y` on the same
    qubits and differ only by :math:`Z`/:math:`I` factors; terms that flip nothing
    (diagonal :math:`I`/:math:`Z` strings) form a single group.

    Such terms connect the same pairs of basis states, so they are the only ones
    whose amplitudes can cancel.  Keeping them contiguous lets a Trotterised
    evolution reproduce a cancellation that the full operator has but no single
    Pauli string can, since a unitary :math:`e^{-i\theta P}` never annihilates a
    state while a sum of terms may:

    .. math::

        e^{-it\sum_i P_i}|\psi\rangle
        \approx \prod_i e^{-it P_i}|\psi\rangle .

    This is the coarsest partition with that property, so groups stay as large as
    possible while preserving every cancellation.

    The motivating case is fermionic chemistry: each excitation
    :math:`a_p^\dagger a_q` (or :math:`a_p^\dagger a_r^\dagger a_s a_q`) annihilates
    the all-zero reference, yet only the *weighted sum* of its Pauli strings cancels.
    Those strings share a flipped-qubit set, so this grouper reassembles them without
    needing the fermionic provenance.  Their :math:`Z` parts then differ by even-size
    subsets of the shared flip set, hence group members also pairwise commute and can
    be exponentiated term by term.

    """

    def name(self) -> str:
        """Return ``qubit_flip`` as the algorithm name."""
        return "qubit_flip"

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Return a copy of ``qubit_hamiltonian`` partitioned by flipped-qubit set.

        Args:
            qubit_hamiltonian: Hamiltonian to partition.

        Returns:
            QubitOperator: New instance with a ``FlatPartition`` (strategy ``"qubit_flip"``).

        """
        buckets: dict[frozenset[int], list[int]] = {}
        for index, label in enumerate(qubit_hamiltonian.pauli_strings):
            buckets.setdefault(pauli_label_flipped_qubits(label), []).append(index)

        # Order groups deterministically: the diagonal group first, then by first member.
        ordered = sorted(buckets.items(), key=lambda item: (len(item[0]) > 0, item[1][0]))
        partition = FlatPartition(
            strategy="qubit_flip",
            groups=tuple(tuple(indices) for _, indices in ordered),
        )
        return QubitOperator(
            pauli_strings=list(qubit_hamiltonian.pauli_strings),
            coefficients=qubit_hamiltonian.coefficients.copy(),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            term_partition=partition,
        )
