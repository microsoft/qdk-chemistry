"""Majorana-to-Pauli mapping data class for fermion-to-qubit encodings.

This module provides the :class:`MajoranaMapping` data class, which stores
the mapping of 2N Majorana operators to Pauli strings for a given f2q encoding.
Standard encodings (Jordan-Wigner, Bravyi-Kitaev, parity) are available via
class-method factories; custom encodings can be constructed from a Pauli table.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry._core.data import MajoranaMapping as _CoreMajoranaMapping
from qdk_chemistry.data.base import DataClass

if TYPE_CHECKING:
    import h5py

__all__: list[str] = []


class MajoranaMapping(DataClass):
    """Immutable data class mapping 2N Majorana operators to Pauli strings.

    A ``MajoranaMapping`` stores a table of 2N entries, one per Majorana operator
    gamma_k (k = 0, ..., 2N-1), where N is the number of fermionic modes (spin-orbitals).
    Each entry is a single Pauli string representing φ(gamma_k) under the chosen
    fermion-to-qubit encoding.

    The mapping is validated at construction time by checking the Clifford algebra
    anticommutation relations: {gamma_i, gamma_j} = 2δ_{ij} · I. This guarantees that the
    mapping defines a valid encoding.

    Pauli strings use the same **little-endian** convention as
    :class:`~qdk_chemistry.data.QubitHamiltonian`: qubit 0 is the rightmost character.

    Attributes:
        table (tuple[str, ...]): Tuple of 2N dense Pauli strings in little-endian format.
        num_modes (int): Number of fermionic modes (spin-orbitals), N.
        num_qubits (int): Number of qubits required by this encoding.
        name (str): Human-readable name of the encoding (may be empty for custom mappings).

    Examples:
        Built-in encodings:

        >>> jw = MajoranaMapping.jordan_wigner(num_modes=4)
        >>> jw.num_modes
        4
        >>> jw.num_qubits
        4

        Custom encoding from Pauli string labels:

        >>> custom = MajoranaMapping(table=["IX", "IY", "XZ", "YZ"], name="my-jw")
        >>> custom.table
        ('IX', 'IY', 'XZ', 'YZ')

    """

    _data_type_name = "majorana_mapping"
    _serialization_version = "0.1.0"

    def __init__(
        self,
        table: list[str] | tuple[str, ...],
        name: str = "",
        *,
        _core: _CoreMajoranaMapping | None = None,
    ) -> None:
        """Initialize a MajoranaMapping from a list of dense Pauli-string labels.

        Args:
            table (list[str] | tuple[str, ...]): 2N Pauli strings in little-endian format (qubit 0 = rightmost char).
            name (str): Optional human-readable label for the encoding. Default ``""``.

        Raises:
            ValueError: If the table is invalid (wrong size, bad characters, or Clifford algebra violation).

        """
        if _core is not None:
            # Fast path: accept a pre-validated C++ core object directly
            # (used by factory classmethods to skip re-parsing and re-validation)
            self._core = _core
        else:
            # Build C++ core object (validates Clifford algebra)
            self._core = _CoreMajoranaMapping(list(table), name)

        # Cache immutable properties from the core
        self._table = self._core.table
        self._name = self._core.name
        self._num_modes = self._core.num_modes
        self._num_qubits = self._core.num_qubits

        # Mark immutable
        super().__init__()

    @property
    def table(self) -> tuple[str, ...]:
        """Tuple of 2N dense Pauli strings in little-endian format (qubit 0 = rightmost char)."""
        return self._table

    @property
    def num_modes(self) -> int:
        """Number of fermionic modes (spin-orbitals)."""
        return self._num_modes

    @property
    def num_qubits(self) -> int:
        """Number of qubits required by this encoding."""
        return self._num_qubits

    @property
    def name(self) -> str:
        """Human-readable name of the encoding (may be empty for custom mappings)."""
        return self._name

    @property
    def core(self) -> _CoreMajoranaMapping:
        """Access the underlying C++ MajoranaMapping object."""
        return self._core

    @classmethod
    def jordan_wigner(cls, num_modes: int) -> MajoranaMapping:
        """Construct a Jordan-Wigner encoding.

        Args:
            num_modes (int): Number of fermionic modes (spin-orbitals).

        Returns:
            MajoranaMapping: Mapping with name ``"jordan-wigner"``.

        """
        core = _CoreMajoranaMapping.jordan_wigner(num_modes)
        return cls(table=[], _core=core)

    @classmethod
    def bravyi_kitaev(cls, num_modes: int) -> MajoranaMapping:
        """Construct a Bravyi-Kitaev encoding.

        Args:
            num_modes (int): Number of fermionic modes (spin-orbitals).

        Returns:
            MajoranaMapping: Mapping with name ``"bravyi-kitaev"``.

        """
        core = _CoreMajoranaMapping.bravyi_kitaev(num_modes)
        return cls(table=[], _core=core)

    @classmethod
    def parity(cls, num_modes: int) -> MajoranaMapping:
        """Construct a parity encoding.

        Args:
            num_modes (int): Number of fermionic modes (spin-orbitals).

        Returns:
            MajoranaMapping: Mapping with name ``"parity"``.

        """
        core = _CoreMajoranaMapping.parity(num_modes)
        return cls(table=[], _core=core)

    @classmethod
    def from_mode_pairs(
        cls,
        pairs: list[tuple[str, str]],
        name: str = "",
    ) -> MajoranaMapping:
        """Construct from (gamma_even, gamma_odd) mode pairs.

        Args:
            pairs (list[tuple[str, str]]): List of (gamma_{2k}, gamma_{2k+1}) Pauli string pairs, one per mode.
            name (str): Optional human-readable label. Default ``""``.

        Returns:
            MajoranaMapping: The constructed mapping.

        Raises:
            ValueError: If pairs are invalid or Clifford algebra validation fails.

        """
        table: list[str] = []
        for even, odd in pairs:
            table.append(even)
            table.append(odd)
        return cls(table=table, name=name)

    @classmethod
    def _from_core(cls, core: _CoreMajoranaMapping) -> MajoranaMapping:
        """Construct from an already-validated C++ core object."""
        return cls(table=[], _core=core)

    def get_summary(self) -> str:
        """Get a human-readable summary of the mapping.

        Returns:
            str: Summary string.

        """
        lines = []
        label = f"MajoranaMapping '{self._name}'" if self._name else "MajoranaMapping (unnamed)"
        lines.append(label)
        lines.append(f"  Modes: {self._num_modes}, Qubits: {self._num_qubits}")
        for k, pauli_str in enumerate(self._table):
            lines.append(f"  gamma_{k} → {pauli_str}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            dict[str, Any]: Dictionary representation.

        """
        data: dict[str, Any] = {
            "table": list(self._table),
            "name": self._name,
        }
        return self._add_json_version(data)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> MajoranaMapping:
        """Deserialize from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary produced by :meth:`to_json`.

        Returns:
            MajoranaMapping: The deserialized mapping.

        """
        cls._validate_json_version("0.1.0", json_data)
        return cls(
            table=json_data["table"],
            name=json_data.get("name", ""),
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group to write to.

        """
        self._add_hdf5_version(group)
        group.attrs["name"] = self._name
        group.attrs["num_modes"] = self._num_modes
        # Store table as array of strings

        group.create_dataset("table", data=np.array(list(self._table), dtype="S"))

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> MajoranaMapping:
        """Load from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group to read from.

        Returns:
            MajoranaMapping: The deserialized mapping.

        """
        cls._validate_hdf5_version("0.1.0", group)
        table = [s.decode("utf-8") if isinstance(s, bytes) else s for s in group["table"][()]]
        name = group.attrs.get("name", "")
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return cls(table=table, name=name)

    def __repr__(self) -> str:
        """Return a repr string."""
        label = f"'{self._name}', " if self._name else ""
        return f"MajoranaMapping({label}num_modes={self._num_modes}, num_qubits={self._num_qubits})"
