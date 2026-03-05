"""Fermion mode ordering convention for fermion-to-qubit mappings.

This module defines the :class:`FermionModeOrder` enum, which tracks how
fermionic modes (e.g. spin-orbitals) are ordered before applying a qubit
encoding:

- **blocked**: all alpha modes first, then all beta
  ``[α₀, α₁, …, αₙ₋₁, β₀, β₁, …, βₙ₋₁]``
- **interleaved**: alternating alpha/beta
  ``[α₀, β₀, α₁, β₁, …, αₙ₋₁, βₙ₋₁]``
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from enum import StrEnum

__all__ = ["FermionModeOrder"]


class FermionModeOrder(StrEnum):
    """Fermion mode ordering convention used when mapping fermions to qubits.

    Attributes:
        BLOCKED: All alpha modes first, then all beta: ``[α₀, α₁, …, β₀, β₁, …]``.
        INTERLEAVED: Alternating alpha and beta: ``[α₀, β₀, α₁, β₁, …]``.

    """

    BLOCKED = "blocked"
    INTERLEAVED = "interleaved"
