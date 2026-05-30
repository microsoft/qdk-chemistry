"""Symmetries data class for encoding physical symmetries of an electronic state.

The :class:`SymmetriesV1` class is a general container for symmetry information
that quantum algorithms can exploit to reduce circuit depth or qubit count.

.. note::
    This class was previously named ``Symmetries``. The name ``Symmetries`` is
    now used for the single-particle symmetry vocabulary in
    :mod:`qdk_chemistry.data.symmetry`. The legacy name remains available as a
    deprecated alias for :class:`SymmetriesV1`.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qdk_chemistry.data.base import DataClass

if TYPE_CHECKING:
    import h5py

    from qdk_chemistry.data import Ansatz, Wavefunction

__all__ = ["SymmetriesV1"]


class SymmetriesV1(DataClass):
    r"""Immutable container for the physical symmetries of an electronic state.

    ``SymmetriesV1`` serves as the central object for communicating symmetry
    information to algorithms that can exploit it.

    Args:
        n_alpha: Number of alpha (spin-up) electrons in the active space.
        n_beta: Number of beta (spin-down) electrons in the active space.

    Raises:
        ValueError: If ``n_alpha`` or ``n_beta`` is negative.

    Examples:
        >>> sym = SymmetriesV1(n_alpha=2, n_beta=2)
        >>> sym.n_particles
        4

        >>> sym = SymmetriesV1.from_wavefunction(wfn)

    """

    # Class attribute for filename validation
    _data_type_name = "symmetries"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(self, n_alpha: int, n_beta: int) -> None:
        """Initialize SymmetriesV1 with active-space electron counts."""
        if n_alpha < 0:
            raise ValueError(f"n_alpha must be non-negative, got {n_alpha}")
        if n_beta < 0:
            raise ValueError(f"n_beta must be non-negative, got {n_beta}")
        self._n_alpha = int(n_alpha)
        self._n_beta = int(n_beta)
        # Make instance immutable after construction (handled by base class)
        super().__init__()

    # -- Factory methods -------------------------------------------------------

    @classmethod
    def from_wavefunction(cls, wavefunction: Wavefunction) -> SymmetriesV1:
        """Construct ``SymmetriesV1`` from a :class:`~qdk_chemistry.data.Wavefunction`.

        Reads the active-space electron counts via ``get_active_num_electrons()``.

        Args:
            wavefunction: A wavefunction carrying active-space electron information.

        Returns:
            A new ``SymmetriesV1`` instance.

        """
        n_alpha, n_beta = wavefunction.get_active_num_electrons()
        return cls(n_alpha=int(n_alpha), n_beta=int(n_beta))

    @classmethod
    def from_ansatz(cls, ansatz: Ansatz) -> SymmetriesV1:
        """Construct ``SymmetriesV1`` from an :class:`~qdk_chemistry.data.Ansatz`.

        Delegates to :meth:`from_wavefunction` using the ansatz's wavefunction.

        Args:
            ansatz: An ansatz bundling a Hamiltonian and wavefunction.

        Returns:
            A new ``SymmetriesV1`` instance.

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
        r"""Spin multiplicity :math:`2S + 1 = |n_\alpha - n_\beta| + 1`."""
        return abs(self._n_alpha - self._n_beta) + 1

    # -- Dunder methods --------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"SymmetriesV1(n_alpha={self._n_alpha}, n_beta={self._n_beta})"

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, SymmetriesV1):
            return NotImplemented
        return self._n_alpha == other._n_alpha and self._n_beta == other._n_beta

    def __hash__(self) -> int:
        """Return a hash."""
        return hash((self._n_alpha, self._n_beta))

    # -- DataClass interface ---------------------------------------------------

    def get_summary(self) -> str:
        """Get a human-readable summary of the symmetries.

        Returns:
            str: Summary string describing the symmetries.

        """
        return (
            f"SymmetriesV1\n"
            f"  Alpha electrons: {self._n_alpha}\n"
            f"  Beta electrons: {self._n_beta}\n"
            f"  Total particles: {self.n_particles}\n"
            f"  Sz: {self.sz}\n"
            f"  Spin multiplicity: {self.spin_multiplicity}"
        )

    def to_json(self) -> dict[str, Any]:
        """Convert the symmetries to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the symmetries.

        """
        data = {
            "n_alpha": self._n_alpha,
            "n_beta": self._n_beta,
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the symmetries to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write the symmetries to.

        """
        self._add_hdf5_version(group)
        group.attrs["n_alpha"] = self._n_alpha
        group.attrs["n_beta"] = self._n_beta

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> SymmetriesV1:
        """Create a SymmetriesV1 from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized data.

        Returns:
            SymmetriesV1: New instance reconstructed from JSON data.

        Raises:
            RuntimeError: If version field is missing or incompatible.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls(
            n_alpha=json_data["n_alpha"],
            n_beta=json_data["n_beta"],
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> SymmetriesV1:
        """Load a SymmetriesV1 from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file containing the data.

        Returns:
            SymmetriesV1: New instance reconstructed from HDF5 data.

        Raises:
            RuntimeError: If version attribute is missing or incompatible.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls(
            n_alpha=int(group.attrs["n_alpha"]),
            n_beta=int(group.attrs["n_beta"]),
        )
