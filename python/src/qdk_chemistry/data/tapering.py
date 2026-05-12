"""Tapering specification for symmetry-conserving encodings.

A :class:`TaperingSpecification` describes which qubits to taper and the eigenvalue
to assign to each, enabling post-mapping qubit reduction for encodings like
symmetry-conserving Bravyi-Kitaev (SCBK).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdk_chemistry.data import Symmetries

__all__: list[str] = []


class TaperingSpecification:
    """Immutable specification for post-mapping qubit tapering.

    Describes which qubits are constrained by symmetry and the Z eigenvalue
    (+1 or -1) to substitute for each.  When attached to a
    :class:`~qdk_chemistry.data.MajoranaMapping`, the mapper applies this
    tapering automatically after the Majorana-to-Pauli mapping.

    Attributes:
        qubit_indices (tuple[int, ...]): Qubit indices to taper (0-indexed, pre-taper numbering).
        eigenvalues (tuple[int, ...]): Corresponding Z eigenvalues (+1 or -1).
        source_num_qubits (int): Number of qubits in the pre-taper register.
        source_encoding (str): Encoding of the pre-taper mapping (e.g. ``"bravyi-kitaev"``).

    """

    __slots__ = ("_eigenvalues", "_qubit_indices", "_source_encoding", "_source_num_qubits")

    _qubit_indices: tuple[int, ...]
    _eigenvalues: tuple[int, ...]
    _source_num_qubits: int
    _source_encoding: str

    def __init__(
        self,
        qubit_indices: tuple[int, ...] | list[int],
        eigenvalues: tuple[int, ...] | list[int],
        source_num_qubits: int,
        source_encoding: str = "",
    ) -> None:
        """Initialize a TaperingSpecification.

        Args:
            qubit_indices (tuple[int, ...] | list[int]): Qubit indices to taper (0-indexed).
            eigenvalues (tuple[int, ...] | list[int]): Corresponding Z eigenvalues (+1 or -1).
            source_num_qubits (int): Number of qubits in the pre-taper register.
            source_encoding (str): Encoding name of the pre-taper mapping.

        Raises:
            ValueError: If lengths don't match, indices out of range, duplicates, or invalid eigenvalues.

        """
        qi = tuple(qubit_indices)
        ev = tuple(eigenvalues)
        if len(qi) != len(ev):
            raise ValueError(f"qubit_indices length ({len(qi)}) must match eigenvalues length ({len(ev)})")
        if len(set(qi)) != len(qi):
            raise ValueError("qubit_indices must not contain duplicates")
        for q in qi:
            if q < 0 or q >= source_num_qubits:
                raise ValueError(f"Qubit index {q} out of range [0, {source_num_qubits})")
        for v in ev:
            if v not in (1, -1):
                raise ValueError(f"Eigenvalue must be +1 or -1, got {v}")

        object.__setattr__(self, "_qubit_indices", qi)
        object.__setattr__(self, "_eigenvalues", ev)
        object.__setattr__(self, "_source_num_qubits", source_num_qubits)
        object.__setattr__(self, "_source_encoding", source_encoding)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("TaperingSpecification is immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("TaperingSpecification is immutable")

    @property
    def qubit_indices(self) -> tuple[int, ...]:
        """Qubit indices to taper (0-indexed, pre-taper numbering)."""
        return self._qubit_indices

    @property
    def eigenvalues(self) -> tuple[int, ...]:
        """Z eigenvalues (+1 or -1) for each tapered qubit."""
        return self._eigenvalues

    @property
    def source_num_qubits(self) -> int:
        """Number of qubits in the pre-taper register."""
        return self._source_num_qubits

    @property
    def source_encoding(self) -> str:
        """Encoding name of the pre-taper mapping."""
        return self._source_encoding

    @property
    def num_tapered(self) -> int:
        """Number of qubits removed by this tapering."""
        return len(self._qubit_indices)

    @classmethod
    def symmetry_conserving_bravyi_kitaev(cls, num_modes: int, symmetries: Symmetries) -> TaperingSpecification:
        """Create a tapering specification for symmetry-conserving Bravyi-Kitaev.

        Tapers the two Z₂ symmetry qubits of the Bravyi-Kitaev encoding:
        total electron-number parity (qubit n-1) and alpha-spin parity
        (qubit n/2-1), reducing the qubit count by 2.

        Args:
            num_modes (int): Number of spin-orbitals (= number of Bravyi-Kitaev qubits).
            symmetries (Symmetries): Electron counts for the target symmetry sector.

        Returns:
            TaperingSpecification: Tapering specification for symmetry-conserving Bravyi-Kitaev.

        Raises:
            ValueError: If num_modes < 4 or odd, or electron counts exceed available orbitals.

        """
        n = num_modes
        if n < 4 or n % 2 != 0:
            raise ValueError(f"Symmetry-conserving Bravyi-Kitaev requires an even num_modes >= 4, got {n}")
        if symmetries.n_alpha < 0 or symmetries.n_beta < 0:
            raise ValueError("n_alpha and n_beta must be non-negative")
        if symmetries.n_alpha > n // 2:
            raise ValueError(f"n_alpha ({symmetries.n_alpha}) exceeds spatial orbitals ({n // 2})")
        if symmetries.n_beta > n // 2:
            raise ValueError(f"n_beta ({symmetries.n_beta}) exceeds spatial orbitals ({n // 2})")

        ev_total = 1 if (symmetries.n_alpha + symmetries.n_beta) % 2 == 0 else -1
        ev_alpha = 1 if symmetries.n_alpha % 2 == 0 else -1

        q_alpha = n // 2 - 1
        q_total = n - 1

        return cls(
            qubit_indices=(q_alpha, q_total),
            eigenvalues=(ev_alpha, ev_total),
            source_num_qubits=n,
            source_encoding="bravyi-kitaev",
        )

    @classmethod
    def parity_two_qubit_reduction(cls, num_modes: int, symmetries: Symmetries) -> TaperingSpecification:
        """Create a tapering specification for parity encoding two-qubit reduction.

        Tapers the same two Z₂ symmetry qubits as the symmetry-conserving
        Bravyi-Kitaev encoding: total electron-number parity (qubit n-1) and
        alpha-spin parity (qubit n/2-1).

        Args:
            num_modes (int): Number of spin-orbitals (= number of parity qubits).
            symmetries (Symmetries): Electron counts for the target symmetry sector.

        Returns:
            TaperingSpecification: Tapering specification for parity two-qubit reduction.

        Raises:
            ValueError: If num_modes < 4 or odd, or electron counts exceed available orbitals.

        """
        n = num_modes
        if n < 4 or n % 2 != 0:
            raise ValueError(f"Parity two-qubit reduction requires an even num_modes >= 4, got {n}")
        if symmetries.n_alpha < 0 or symmetries.n_beta < 0:
            raise ValueError("n_alpha and n_beta must be non-negative")
        if symmetries.n_alpha > n // 2:
            raise ValueError(f"n_alpha ({symmetries.n_alpha}) exceeds spatial orbitals ({n // 2})")
        if symmetries.n_beta > n // 2:
            raise ValueError(f"n_beta ({symmetries.n_beta}) exceeds spatial orbitals ({n // 2})")

        ev_total = 1 if (symmetries.n_alpha + symmetries.n_beta) % 2 == 0 else -1
        ev_alpha = 1 if symmetries.n_alpha % 2 == 0 else -1

        q_alpha = n // 2 - 1
        q_total = n - 1

        return cls(
            qubit_indices=(q_alpha, q_total),
            eigenvalues=(ev_alpha, ev_total),
            source_num_qubits=n,
            source_encoding="parity",
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            dict[str, Any]: Dictionary representation.

        """
        return {
            "qubit_indices": list(self._qubit_indices),
            "eigenvalues": list(self._eigenvalues),
            "source_num_qubits": self._source_num_qubits,
            "source_encoding": self._source_encoding,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TaperingSpecification:
        """Deserialize from a JSON dictionary.

        Args:
            data (dict[str, Any]): Dictionary produced by :meth:`to_json`.

        Returns:
            TaperingSpecification: The deserialized tapering specification.

        """
        return cls(
            qubit_indices=tuple(data["qubit_indices"]),
            eigenvalues=tuple(data["eigenvalues"]),
            source_num_qubits=data["source_num_qubits"],
            source_encoding=data.get("source_encoding", ""),
        )

    def __repr__(self) -> str:
        """Return a repr string."""
        return (
            f"TaperingSpecification(qubit_indices={self._qubit_indices}, "
            f"eigenvalues={self._eigenvalues}, "
            f"source_num_qubits={self._source_num_qubits})"
        )

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, TaperingSpecification):
            return NotImplemented
        return (
            self._qubit_indices == other._qubit_indices
            and self._eigenvalues == other._eigenvalues
            and self._source_num_qubits == other._source_num_qubits
            and self._source_encoding == other._source_encoding
        )

    def __hash__(self) -> int:
        """Return hash."""
        return hash((self._qubit_indices, self._eigenvalues, self._source_num_qubits, self._source_encoding))
