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
)

__all__ = [
    "AxisName",
    "SpinValue",
    "SymmetryAxis",
    "SymmetryAxisValue",
    "SymmetryBlockedIndexSet",
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
]
