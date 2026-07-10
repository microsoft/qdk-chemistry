"""Single-particle symmetry types and symmetry-blocked storage primitives.

This module exposes the single-particle symmetry types used to block
quantum-chemistry tensors (orbital coefficients, energies, integrals, reduced
density matrices) by conserved single-particle quantum numbers, together with
the symmetry-blocked tensor storage primitives (the rank/scalar variants listed
below) and their index-set companion :class:`SymmetryBlockedIndexSet`.

In this release only the spin axis (:math:`S_z`) is populated, supporting
restricted (RHF/ROHF) and unrestricted (UHF) references.

Exposed symmetry types are:

- :class:`AxisName`: Enumeration of single-particle symmetry axes.
- :func:`axis_name_to_string`: Human-readable name for an :class:`AxisName`.
- :class:`SymmetryAxisValue`: Abstract value carried by a single symmetry axis.
- :class:`SpinValue`: Concrete spin-1/2 axis value (stored as :math:`2 M_s`).
- :class:`SymmetryAxis`: One named symmetry partition with its admissible labels.
- :class:`SymmetryProduct`: An ordered set of axes a basis is blocked under.
- :class:`SymmetryLabel`: A composite addressing key, one value per axis.

Exposed storage types are:

- :class:`SymmetryBlockedScalarCount`: One scalar count per symmetry sector.
- :class:`SymmetryBlockedTensorRank1` / :class:`SymmetryBlockedTensorRank1Complex`
- :class:`SymmetryBlockedTensorRank2` / :class:`SymmetryBlockedTensorRank2Complex`
- :class:`SymmetryBlockedTensorRank3` / :class:`SymmetryBlockedTensorRank3Complex`
- :class:`SymmetryBlockedTensorRank4` / :class:`SymmetryBlockedTensorRank4Complex`
- :class:`SymmetryBlockedSparseMapRank4`
- :class:`SymmetryBlockedIndexSet`

Errors are surfaced as standard Python exceptions mapped from the underlying
C++ standard-library exceptions.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core.data.symmetry import (
    AxisName,
    SpinValue,
    SymmetryAxis,
    SymmetryAxisValue,
    SymmetryBlockedIndexSet,
    SymmetryBlockedScalarCount,
    SymmetryBlockedSparseMapRank4,
    SymmetryBlockedTensorRank1,
    SymmetryBlockedTensorRank1Complex,
    SymmetryBlockedTensorRank2,
    SymmetryBlockedTensorRank2Complex,
    SymmetryBlockedTensorRank3,
    SymmetryBlockedTensorRank3Complex,
    SymmetryBlockedTensorRank4,
    SymmetryBlockedTensorRank4Complex,
    SymmetryLabel,
    SymmetryProduct,
    axes,
    axis_name_to_string,
    spin_channel_indices,
)

__all__ = [
    "AxisName",
    "SpinValue",
    "SymmetryAxis",
    "SymmetryAxisValue",
    "SymmetryBlockedIndexSet",
    "SymmetryBlockedScalarCount",
    "SymmetryBlockedSparseMapRank4",
    "SymmetryBlockedTensorRank1",
    "SymmetryBlockedTensorRank1Complex",
    "SymmetryBlockedTensorRank2",
    "SymmetryBlockedTensorRank2Complex",
    "SymmetryBlockedTensorRank3",
    "SymmetryBlockedTensorRank3Complex",
    "SymmetryBlockedTensorRank4",
    "SymmetryBlockedTensorRank4Complex",
    "SymmetryLabel",
    "SymmetryProduct",
    "axes",
    "axis_name_to_string",
    "spin_channel_indices",
    "spin_channel_matrix",
    "spin_channel_vector",
    "spin_index_set",
]


from collections.abc import Sequence


def spin_index_set(
    num_modes: int,
    alpha: Sequence[int],
    beta: Sequence[int],
    *,
    equivalent: bool = True,
) -> SymmetryBlockedIndexSet:
    """Build a spin-resolved SymmetryBlockedIndexSet from alpha/beta index lists.

    Args:
        num_modes: Universe size (total number of orbitals per spin channel).
        alpha: Sorted active/inactive indices for the alpha channel.
        beta: Sorted active/inactive indices for the beta channel.
        equivalent: Whether the spin axis labels share storage (restricted).

    Returns:
        A SymmetryBlockedIndexSet carrying the spin symmetry and per-channel indices.

    """
    sym = SymmetryProduct([axes.spin(1, equivalent)])
    alpha_label = SymmetryLabel([axes.alpha()])
    beta_label = SymmetryLabel([axes.beta()])
    extents = {alpha_label: num_modes, beta_label: num_modes}
    indices = {
        alpha_label: list(alpha),
        beta_label: list(beta),
    }
    return SymmetryBlockedIndexSet(sym, extents, indices)


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
