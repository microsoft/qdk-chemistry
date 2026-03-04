"""Symmetries data class for encoding quantum number constraints.

Encapsulates the physical symmetries of an electronic state (particle number, spin)
for use by algorithms that exploit conserved quantum numbers, such as the
symmetry-conserving Bravyi-Kitaev qubit mapping.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdk_chemistry.data import Ansatz, Wavefunction

__all__ = ["Symmetries"]


class Symmetries:
    """Immutable container for the conserved quantum numbers of an electronic state.

    Stores the number of alpha and beta electrons in the active space. Derived
    quantities such as total particle number, :math:`S_z` projection, and spin
    multiplicity are available as properties.

    Args:
        n_alpha: Number of alpha (spin-up) electrons in the active space.
        n_beta: Number of beta (spin-down) electrons in the active space.

    Raises:
        ValueError: If ``n_alpha`` or ``n_beta`` is negative.

    Examples:
        >>> sym = Symmetries(n_alpha=2, n_beta=2)
        >>> sym.n_particles
        4

        >>> sym = Symmetries.from_wavefunction(wfn)

    """

    __slots__ = ("_n_alpha", "_n_beta")

    def __init__(self, n_alpha: int, n_beta: int) -> None:
        """Initialize Symmetries with active-space electron counts."""
        if n_alpha < 0:
            raise ValueError(f"n_alpha must be non-negative, got {n_alpha}")
        if n_beta < 0:
            raise ValueError(f"n_beta must be non-negative, got {n_beta}")
        self._n_alpha = int(n_alpha)
        self._n_beta = int(n_beta)

    # -- Factory methods -------------------------------------------------------

    @classmethod
    def from_wavefunction(cls, wavefunction: Wavefunction) -> Symmetries:
        """Construct ``Symmetries`` from a :class:`~qdk_chemistry.data.Wavefunction`.

        Reads the active-space electron counts via ``get_active_num_electrons()``.

        Args:
            wavefunction: A wavefunction carrying active-space electron information.

        Returns:
            A new ``Symmetries`` instance.

        """
        n_alpha, n_beta = wavefunction.get_active_num_electrons()
        return cls(n_alpha=int(n_alpha), n_beta=int(n_beta))

    @classmethod
    def from_ansatz(cls, ansatz: Ansatz) -> Symmetries:
        """Construct ``Symmetries`` from an :class:`~qdk_chemistry.data.Ansatz`.

        Delegates to :meth:`from_wavefunction` using the ansatz's wavefunction.

        Args:
            ansatz: An ansatz bundling a Hamiltonian and wavefunction.

        Returns:
            A new ``Symmetries`` instance.

        """
        return cls.from_wavefunction(ansatz.get_wavefunction())

    # -- Read-only properties --------------------------------------------------

    @property
    def n_alpha(self) -> int:
        """Number of alpha (spin-up) electrons in the active space."""
        return self._n_alpha

    @property
    def n_beta(self) -> int:
        """Number of beta (spin-down) electrons in the active space."""
        return self._n_beta

    @property
    def n_particles(self) -> int:
        """Total number of active electrons (``n_alpha + n_beta``)."""
        return self._n_alpha + self._n_beta

    @property
    def sz(self) -> float:
        r"""Spin projection quantum number :math:`S_z = (n_\alpha - n_\beta) / 2`."""
        return (self._n_alpha - self._n_beta) / 2

    @property
    def spin_multiplicity(self) -> int:
        r"""Spin multiplicity :math:`2S + 1 = n_\alpha - n_\beta + 1`."""
        return self._n_alpha - self._n_beta + 1

    # -- Dunder methods --------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"Symmetries(n_alpha={self._n_alpha}, n_beta={self._n_beta})"

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Symmetries):
            return NotImplemented
        return self._n_alpha == other._n_alpha and self._n_beta == other._n_beta

    def __hash__(self) -> int:
        """Return a hash."""
        return hash((self._n_alpha, self._n_beta))
