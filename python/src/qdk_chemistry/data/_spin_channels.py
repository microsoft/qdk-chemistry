# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
"""Internal per-spin-channel helpers over symmetry-blocked data.

These helpers are **not** part of the public API. User code should read
symmetry-blocked tensors and index sets through their primitive accessors
(``SymmetryBlockedTensor.block([...])`` and
``SymmetryBlockedIndexSet.indices(SymmetryLabel([...]))``) with explicit
:class:`~qdk_chemistry.data.symmetry.SymmetryLabel` values.

They exist only for internal library and test use, where they additionally
encapsulate the trivial (spin-free) vs spin-resolved label selection so the
same call works for restricted, unrestricted, and model (symmetry-free)
orbitals.
"""

from qdk_chemistry._core.data.symmetry import (
    AxisName,
    SpinValue,
    SymmetryBlockedTensorRank1,
    SymmetryBlockedTensorRank2,
    SymmetryLabel,
    spin_channel_indices,
)

__all__ = [
    "spin_channel_indices",
    "spin_channel_matrix",
    "spin_channel_vector",
]


def spin_channel_matrix(coefficients: SymmetryBlockedTensorRank2, channel: SpinValue):
    """Per-spin dense block of a rank-2 (coefficient-like) SymmetryBlockedTensor.

    Returns the block for ``channel``; for a restricted (spin-equivalent) tensor the
    alpha and beta blocks share storage, so either channel selection succeeds.

    Args:
        coefficients: A rank-2 SymmetryBlockedTensor (e.g. orbital coefficients).
        channel: The spin channel to select, e.g. ``axes.alpha()`` or ``axes.beta()``.

    Returns:
        The requested spin channel's dense block as a NumPy array.

    """
    slots = coefficients.symmetries()

    def _slot_label(index: int) -> SymmetryLabel:
        spin_axis = index < len(slots) and slots[index].has_axis(AxisName.Spin)
        return SymmetryLabel([channel]) if spin_axis else SymmetryLabel([])

    return coefficients.block([_slot_label(0), _slot_label(1)])


def spin_channel_vector(energies: SymmetryBlockedTensorRank1, channel: SpinValue):
    """Per-spin dense block of a rank-1 (energy-like) SymmetryBlockedTensor.

    Returns the block for ``channel``; for a restricted (spin-equivalent) tensor the
    alpha and beta blocks share storage, so either channel selection succeeds.

    Args:
        energies: A rank-1 SymmetryBlockedTensor (e.g. orbital energies).
        channel: The spin channel to select, e.g. ``axes.alpha()`` or ``axes.beta()``.

    Returns:
        The requested spin channel's dense block as a NumPy array.

    """
    label = SymmetryLabel([channel])
    slots = energies.symmetries()
    if not (slots and slots[0].has_axis(AxisName.Spin)):
        label = SymmetryLabel([])
    return energies.block([label])
