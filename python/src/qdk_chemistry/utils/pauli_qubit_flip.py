r"""Utilities describing which qubits a Pauli term flips.

The **flipped-qubit set** of a Pauli string is the set of positions carrying
:math:`X` or :math:`Y` (its :math:`XY`-support), i.e. the qubits whose
:math:`|0\rangle` and :math:`|1\rangle` get exchanged; :math:`I` and :math:`Z`
leave the bit value alone.  Strings sharing a flipped-qubit set differ only by
:math:`Z`/:math:`I` factors, so they connect the same pairs of basis states and
are the only ones whose amplitudes can cancel.  On the all-zero state,

.. math::

    P\,|0\ldots0\rangle = i^{n_Y}\,|b_F\rangle,

with :math:`F` the flipped-qubit set, :math:`n_Y` the number of :math:`Y`
factors, and :math:`b_F` the bit string with exactly the qubits in :math:`F` set.

Used by the ``qubit_flip`` term grouper
(:class:`~qdk_chemistry.algorithms.term_grouper.QubitFlipTermGrouper`) and by the
CSWAP-sandwich mapper, which checks that an evolution fixes its vacuum register.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

__all__: list[str] = [
    "pauli_label_flipped_qubits",
    "pauli_label_zero_state_action",
    "pauli_map_flipped_qubits",
    "pauli_map_zero_state_action",
]

_VALID_AXES = frozenset("IXYZ")
_FLIPPING_AXES = frozenset("XY")


def pauli_label_flipped_qubits(label: str) -> frozenset[int]:
    r"""Return the qubits a Pauli label flips.

    The label uses the Qiskit / ``SparsePauliOp`` convention: the rightmost
    character corresponds to qubit 0.

    Args:
        label: Pauli string label (e.g. ``"XIZI"``).

    Returns:
        The set of qubit indices carrying :math:`X` or :math:`Y`.

    Raises:
        ValueError: If the label contains a character other than ``I``, ``X``, ``Y`` or ``Z``.

    Examples:
        >>> sorted(pauli_label_flipped_qubits("XIZY"))
        [0, 3]
        >>> pauli_label_flipped_qubits("ZZ")
        frozenset()

    """
    n = len(label)
    flipped: set[int] = set()
    for position, axis in enumerate(label):
        if axis not in _VALID_AXES:
            raise ValueError(f"Invalid character {axis!r} in Pauli label; expected 'I', 'X', 'Y', or 'Z'.")
        if axis in _FLIPPING_AXES:
            flipped.add(n - position - 1)
    return frozenset(flipped)


def pauli_map_flipped_qubits(pauli_map: dict[int, str]) -> frozenset[int]:
    r"""Return the qubits a sparse Pauli term flips.

    Args:
        pauli_map: Sparse Pauli term mapping qubit index to Pauli axis.

    Returns:
        The set of qubit indices carrying :math:`X` or :math:`Y`.

    Raises:
        ValueError: If the mapping contains an axis other than ``I``, ``X``, ``Y`` or ``Z``.

    """
    flipped: set[int] = set()
    for qubit, axis in pauli_map.items():
        if axis not in _VALID_AXES:
            raise ValueError(f"Invalid Pauli axis {axis!r} on qubit {qubit}; expected 'I', 'X', 'Y', or 'Z'.")
        if axis in _FLIPPING_AXES:
            flipped.add(qubit)
    return frozenset(flipped)


def pauli_label_zero_state_action(label: str) -> tuple[frozenset[int], complex]:
    r"""Return the basis state and amplitude produced by a Pauli label acting on :math:`|0\ldots0\rangle`.

    Args:
        label: Pauli string label (e.g. ``"XIZI"``), the rightmost character being qubit 0.

    Returns:
        A ``(flipped_qubits, amplitude)`` pair where ``flipped_qubits`` is the set of
        qubits raised to :math:`|1\rangle` and ``amplitude`` is the prefactor :math:`i^{n_Y}`.

    Raises:
        ValueError: If the label contains a character other than ``I``, ``X``, ``Y`` or ``Z``.

    Examples:
        >>> pauli_label_zero_state_action("ZI")
        (frozenset(), (1+0j))

    """
    return pauli_label_flipped_qubits(label), 1j ** label.count("Y")


def pauli_map_zero_state_action(pauli_map: dict[int, str]) -> tuple[frozenset[int], complex]:
    r"""Return the basis state and amplitude produced by a sparse Pauli term acting on :math:`|0\ldots0\rangle`.

    Args:
        pauli_map: Sparse Pauli term mapping qubit index to Pauli axis.

    Returns:
        A ``(flipped_qubits, amplitude)`` pair, as for :func:`pauli_label_zero_state_action`.

    Raises:
        ValueError: If the mapping contains an axis other than ``I``, ``X``, ``Y`` or ``Z``.

    """
    flipped = pauli_map_flipped_qubits(pauli_map)
    num_y = sum(1 for axis in pauli_map.values() if axis == "Y")
    return flipped, 1j**num_y
