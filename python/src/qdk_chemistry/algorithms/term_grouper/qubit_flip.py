r"""Qubit-flip term grouper: group Pauli terms that flip the same qubits.

The **flipped-qubit set** of a Pauli string is the set of positions carrying
:math:`X` or :math:`Y` (its :math:`XY`-support), i.e. the qubits whose
:math:`|0\rangle` and :math:`|1\rangle` get exchanged; :math:`I` and :math:`Z`
leave the bit value alone.  On the all-zero state,

.. math::

    P\,|0\ldots0\rangle = i^{n_Y}\,|b_F\rangle,

with :math:`F` the flipped-qubit set, :math:`n_Y` the number of :math:`Y`
factors, and :math:`b_F` the bit string with exactly the qubits in :math:`F` set.
Groups are refined by the parity of :math:`n_Y`; see :class:`QubitFlipTermGrouper`.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper
from qdk_chemistry.data import FlatPartition, QubitOperator

__all__ = ["QubitFlipTermGrouper"]


class QubitFlipTermGrouper(TermGrouper):
    r"""Group Pauli terms that flip the same set of qubits with the same :math:`Y` parity.

    Terms land in the same group when they carry :math:`X`/:math:`Y` on the same qubits and agree
    on the parity of their :math:`Y` count.  Terms that flip nothing (diagonal :math:`I`/:math:`Z`
    strings) form a single group.

    Terms sharing a flipped-qubit set connect the same pairs of basis states, so they are the only
    ones whose amplitudes can cancel.  Keeping them contiguous lets a Trotterised evolution
    reproduce a cancellation that the full operator has but no single Pauli string can, since a
    unitary :math:`e^{-i\theta P}` never annihilates a state while a sum of terms may:

    .. math::

        e^{-it\sum_i P_i}|\psi\rangle
        \approx \prod_i e^{-it P_i}|\psi\rangle .

    The parity refinement makes a group internally commuting.  Two strings sharing a flipped-qubit
    set disagree only inside it, on positions where one carries :math:`X` and the other :math:`Y`,
    and the count of those has parity :math:`n_Y^{(a)} + n_Y^{(b)} \bmod 2`.  Equal parity means an
    even number of anticommuting positions.  Without it :math:`X` and :math:`Y` would share a group
    despite anticommuting, and a symmetric Trotter formula could no longer treat the group as a
    single exponential.

    The split loses no cancellation: with real (Hermitian) coefficients the even-parity terms
    contribute :math:`\pm 1` and the odd-parity ones :math:`\pm i` to :math:`P|0\ldots0\rangle`, so
    the two sub-sums are the real and imaginary parts of the total and vanish separately.  This is
    the coarsest partition that preserves every cancellation and keeps groups internally commuting.

    The motivating case is fermionic chemistry: each excitation :math:`a_p^\dagger a_q` (or
    :math:`a_p^\dagger a_r^\dagger a_s a_q`) annihilates the all-zero reference, yet only the
    *weighted sum* of its Pauli strings cancels.  Those strings share a flipped-qubit set and an
    even :math:`Y` count -- ``XX`` and ``YY``, for instance -- so this grouper reassembles them
    without needing the fermionic provenance.

    """

    def name(self) -> str:
        """Return ``qubit_flip`` as the algorithm name."""
        return "qubit_flip"

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Return a copy of ``qubit_hamiltonian`` partitioned by flipped-qubit set and Y parity.

        Args:
            qubit_hamiltonian: Hamiltonian to partition.

        Returns:
            QubitOperator: New instance with a ``FlatPartition`` (strategy ``"qubit_flip"``).

        """
        buckets: dict[tuple[frozenset[int], int], list[int]] = {}
        for index, label in enumerate(qubit_hamiltonian.pauli_strings):
            # Labels follow the Qiskit convention: the rightmost character is qubit 0.
            flipped = frozenset(len(label) - position - 1 for position, axis in enumerate(label) if axis in "XY")
            # Same-support strings anticommute exactly when their Y counts differ in parity.
            buckets.setdefault((flipped, label.count("Y") % 2), []).append(index)

        # Order groups deterministically: the diagonal group first, then by first member.
        ordered = sorted(buckets.items(), key=lambda item: (len(item[0][0]) > 0, item[1][0]))
        partition = FlatPartition(
            strategy="qubit_flip",
            groups=tuple(tuple(indices) for _, indices in ordered),
        )
        return QubitOperator(
            pauli_strings=list(qubit_hamiltonian.pauli_strings),
            coefficients=qubit_hamiltonian.coefficients.copy(),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            tapering=qubit_hamiltonian.tapering,
            term_partition=partition,
        )
