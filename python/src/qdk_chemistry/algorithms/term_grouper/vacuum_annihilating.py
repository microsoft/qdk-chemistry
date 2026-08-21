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

import numpy as np

from qdk_chemistry.algorithms.term_grouper.base import TermGrouper, TermGrouperSettings
from qdk_chemistry.data import FlatPartition, QubitOperator

__all__ = ["VacuumAnnihilatingTermGrouper", "VacuumAnnihilatingTermGrouperSettings"]


class VacuumAnnihilatingTermGrouperSettings(TermGrouperSettings):
    """Settings for the :class:`VacuumAnnihilatingTermGrouper`.

    Attributes:
        tolerance: Absolute tolerance on the vacuum amplitude of a flipped-qubit set.

    """

    def __init__(self):
        """Initialise the settings for VacuumAnnihilatingTermGrouper."""
        super().__init__()
        self._set_default(
            "tolerance",
            "double",
            1e-9,
            "Absolute tolerance below which a group's accumulated vacuum amplitude counts as zero.",
        )


class VacuumAnnihilatingTermGrouper(TermGrouper):
    r"""Group Pauli *terms* whose weighted amplitudes cancel on :math:`|0\ldots0\rangle`.

    Only terms sharing a flipped-qubit set reach the same basis state :math:`|b_F\rangle`, so
    they are the only ones whose amplitudes can cancel.  Within such a set the grouper
    accumulates :math:`c_j i^{n_Y^{(j)}}` in index order and closes a group as soon as the
    running sum vanishes, giving the finest partition into groups that each satisfy

    .. math::

        \sum_{j \in g} c_j\, P_j\, |0\ldots0\rangle = 0 .

    Terms left over at the end of a flipped-qubit set cannot cancel, so the operator does not
    annihilate the vacuum and grouping fails with a :class:`ValueError`.  Diagonal
    (:math:`I`/:math:`Z`) strings never move the vacuum and form a single group that only phases
    it, which a consumer can correct for, so their sum is left unconstrained.

    Groups are additionally restricted to a single :math:`Y`-count parity, which makes their
    members commute: two strings sharing a flipped-qubit set disagree only inside it, on
    positions where one carries :math:`X` and the other :math:`Y`, and the number of those has
    parity :math:`n_Y^{(a)} + n_Y^{(b)} \bmod 2`.  No cancellation is lost, since with real
    coefficients the even-parity terms contribute :math:`\pm 1` and the odd-parity ones
    :math:`\pm i`, so the two sub-sums are the real and imaginary parts of the total.

    The motivating case is fermionic chemistry, here each excitation annihilates the all-zero
    reference through the *weighted sum* of its Pauli strings,
    and which is what the ``ControlledSwapPauliSequenceMapper`` requires.

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
            QubitOperator: New instance with a ``FlatPartition`` (strategy ``"vacuum_annihilating"``).

        Raises:
            ValueError: If ``tolerance`` is negative or not finite.
            ValueError: If a coefficient is not a finite real number.
            ValueError: If terms flipping the same qubits do not cancel on the vacuum.

        """
        tolerance = self._settings.get("tolerance")
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"tolerance must be finite and non-negative, got {tolerance}.")

        flipped_qubits = str.maketrans("IXYZ", "0110")

        coefficients = np.asarray(qubit_hamiltonian.coefficients)
        if np.any(np.iscomplex(coefficients)) or not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "VacuumAnnihilatingTermGrouper requires finite, real coefficients, since vacuum "
                "amplitudes are compared against a real cancellation tolerance."
            )

        buckets: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for index, label in enumerate(qubit_hamiltonian.pauli_strings):
            # Labels follow the Qiskit convention: the rightmost character is qubit 0.
            flipped = int(label.translate(flipped_qubits), 2)
            n_y = label.count("Y")
            # Same-support strings anticommute exactly when their Y counts differ in parity.
            # Within one parity i^{n_Y} is a common factor of 1 or i times this sign.
            amplitude = float(coefficients[index].real) * (-1) ** (n_y // 2)
            buckets.setdefault((flipped, n_y % 2), []).append((index, amplitude))

        diagonal: tuple[int, ...] = ()
        groups: list[tuple[int, ...]] = []
        for (flipped, _), entries in buckets.items():
            if not flipped:
                diagonal = tuple(index for index, _ in entries)
                continue

            # Certify the whole flipped-qubit set first: splitting on an approximately-zero prefix
            # would otherwise strand a remainder that the set as a whole cancels.
            residual = math.fsum(amplitude for _, amplitude in entries)
            if abs(residual) > tolerance:
                qubits = [qubit for qubit in range(flipped.bit_length()) if flipped >> qubit & 1]
                raise ValueError(
                    f"VacuumAnnihilatingTermGrouper cannot group terms {[index for index, _ in entries]}: "
                    f"they flip qubits {qubits} and leave an uncancelled vacuum amplitude of {residual:.3g}. "
                    "The Hamiltonian does not annihilate |0...0> in this encoding, so no ordering of "
                    "its Pauli strings preserves the vacuum under Trotterisation."
                )

            current: list[int] = []
            pending: list[float] = []
            for index, amplitude in entries:
                current.append(index)
                pending.append(amplitude)
                if math.fsum(pending) == 0.0:
                    groups.append(tuple(current))
                    current, pending = [], []
            if current:
                groups.append(tuple(current))

        ordered = ([diagonal] if diagonal else []) + sorted(groups, key=lambda group: group[0])
        partition = FlatPartition(strategy="vacuum_annihilating", groups=tuple(ordered))
        return QubitOperator(
            pauli_strings=list(qubit_hamiltonian.pauli_strings),
            coefficients=qubit_hamiltonian.coefficients.copy(),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            tapering=qubit_hamiltonian.tapering,
            term_partition=partition,
        )
