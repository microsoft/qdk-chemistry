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
from qdk_chemistry.data.tapering import TaperingSpecification

if TYPE_CHECKING:
    import h5py

    from qdk_chemistry.data import Symmetries

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
        phases: list[int] | tuple[int, ...] | None = None,
        tapering: TaperingSpecification | None = None,
        *,
        _core: _CoreMajoranaMapping | None = None,
    ) -> None:
        """Initialize a MajoranaMapping from a list of dense Pauli-string labels.

        Args:
            table (list[str] | tuple[str, ...]): 2N Pauli strings in little-endian format (qubit 0 = rightmost char).
            name (str): Optional human-readable label for the encoding. Default ``""``.
            phases (list[int] | None): Optional 2N sign factors (+1 or -1) per Majorana operator. Default all +1.
            tapering (TaperingSpecification | None): Optional post-mapping tapering specification.

        Raises:
            ValueError: If the table is invalid (wrong size, bad characters, or Clifford algebra violation).

        """
        if _core is not None:
            self._core = _core
        else:
            self._core = _CoreMajoranaMapping(list(table), name, list(phases) if phases else [])

        # Cache immutable properties from the core
        self._table = self._core.table
        # Allow Python-level name override (e.g. SCBK wraps a BK core)
        self._name = name if name else self._core.name
        self._num_modes = self._core.num_modes
        self._num_qubits = self._core.num_qubits
        self._phases = self._core.phases
        self._tapering = tapering

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
        """Effective number of qubits after any tapering.

        For untapered encodings this equals the number of qubits in the Pauli
        table.  For tapering-based encodings (e.g. symmetry-conserving
        Bravyi-Kitaev, parity with two-qubit reduction) this reflects the
        reduced qubit count that downstream consumers will see.
        """
        if self._tapering is not None:
            return self._num_qubits - self._tapering.num_tapered
        return self._num_qubits

    @property
    def name(self) -> str:
        """Human-readable name of the encoding (may be empty for custom mappings)."""
        return self._name

    @property
    def phases(self) -> tuple[int, ...]:
        """Tuple of per-entry sign factors (+1 or -1). All +1 for standard encodings."""
        return self._phases

    @property
    def tapering(self) -> TaperingSpecification | None:
        """Post-mapping tapering specification, or None for untapered encodings."""
        return self._tapering

    @property
    def base_encoding(self) -> str:
        """The base encoding name used for the Majorana-to-Pauli table.

        For standard encodings this equals :attr:`name`. For tapering-based
        encodings like symmetry-conserving Bravyi-Kitaev, this returns the
        underlying encoding (e.g. ``"bravyi-kitaev"``) while :attr:`name`
        returns the final encoding label
        (e.g. ``"symmetry-conserving-bravyi-kitaev"``).
        """
        return self._core.name

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
    def parity(
        cls,
        num_modes: int,
        symmetries: Symmetries | None = None,
    ) -> MajoranaMapping:
        """Construct a parity encoding, optionally with two-qubit reduction.

        When ``symmetries`` is provided, the mapping includes a
        :class:`~qdk_chemistry.data.TaperingSpecification` that tapers the two
        Z₂ symmetry qubits (total electron-number parity and alpha-spin
        parity), reducing the qubit count by 2.  This is the same two-qubit
        reduction used by Qiskit Nature's ``ParityMapper(num_particles=...)``.

        Args:
            num_modes (int): Number of fermionic modes (spin-orbitals).
            symmetries (Symmetries | None): If provided, enables two-qubit reduction for the target symmetry sector.

        Returns:
            MajoranaMapping: Mapping with name ``"parity"`` (untapered) or ``"parity-2q-reduced"`` (tapered).

        """
        core = _CoreMajoranaMapping.parity(num_modes)
        if symmetries is not None:
            tapering = TaperingSpecification.parity_two_qubit_reduction(num_modes, symmetries)
            return cls(table=[], name="parity-2q-reduced", tapering=tapering, _core=core)
        return cls(table=[], _core=core)

    @classmethod
    def symmetry_conserving_bravyi_kitaev(
        cls,
        num_modes: int,
        symmetries: Symmetries,
    ) -> MajoranaMapping:
        """Construct a symmetry-conserving Bravyi-Kitaev (SCBK) encoding.

        Combines the standard Bravyi-Kitaev mapping with a
        :class:`~qdk_chemistry.data.TaperingSpecification` that removes the two Z₂
        symmetry qubits (total electron-number parity and alpha-spin parity),
        reducing the qubit count by 2.

        When passed to :meth:`~qdk_chemistry.algorithms.QubitMapper.run`, the
        mapper applies the BK mapping first, then tapers the symmetry qubits
        automatically.

        Args:
            num_modes (int): Number of fermionic modes (spin-orbitals). Must be even and >= 4.
            symmetries (Symmetries): Electron counts for the target symmetry sector.

        Returns:
            MajoranaMapping: BK mapping with SCBK tapering, name ``"symmetry-conserving-bravyi-kitaev"``.

        Raises:
            ValueError: If num_modes < 4 or odd, or electron counts are invalid.

        Examples:
            >>> from qdk_chemistry.data import MajoranaMapping, Symmetries
            >>> mapping = MajoranaMapping.symmetry_conserving_bravyi_kitaev(8, Symmetries(2, 2))
            >>> mapping.name
            'symmetry-conserving-bravyi-kitaev'
            >>> mapping.base_encoding
            'bravyi-kitaev'
            >>> mapping.tapering.num_tapered
            2

        """
        tapering = TaperingSpecification.symmetry_conserving_bravyi_kitaev(num_modes, symmetries)
        core = _CoreMajoranaMapping.bravyi_kitaev(num_modes)
        return cls(
            table=[],
            name="symmetry-conserving-bravyi-kitaev",
            tapering=tapering,
            _core=core,
        )

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
        # Only include phases if any are non-default (-1)
        if any(p != 1 for p in self._phases):
            data["phases"] = list(self._phases)
        if self._tapering is not None:
            data["tapering"] = self._tapering.to_json()
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
        tapering_data = json_data.get("tapering")
        tapering = TaperingSpecification.from_json(tapering_data) if tapering_data else None
        return cls(
            table=json_data["table"],
            name=json_data.get("name", ""),
            phases=json_data.get("phases"),
            tapering=tapering,
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
