"""QDK/Chemistry Quantum Phase Estimation Results module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from collections.abc import Iterable, Sequence
from typing import Any

import h5py
import numpy as np

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.phase import energy_alias_candidates, energy_from_phase, resolve_energy_aliases

__all__: list[str] = []


class QpeResult(DataClass):
    """Structured output for quantum phase estimation workflows."""

    # Class attribute for filename validation
    _data_type_name = "qpe_result"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        method: str,
        evolution_time: float,
        phase_fraction: float,
        phase_angle: float,
        canonical_phase_fraction: float,
        canonical_phase_angle: float,
        raw_energy: float,
        branching: tuple[float, ...],
        resolved_energy: float | None = None,
        bits_msb_first: tuple[int, ...] | None = None,
        bitstring_msb_first: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialize a QPE result.

        Args:
            method: Identifier for the algorithm or workflow that produced the result.
            evolution_time: Evolution time ``t`` used in ``U = exp(-i H t)``.
            phase_fraction:  Raw measured phase fraction in ``[0, 1)``.
            phase_angle: Raw measured phase angle in radians.
            canonical_phase_fraction:  Alias-resolved phase fraction consistent with the selected energy branch.
            canonical_phase_angle: Alias-resolved phase angle in radians.
            raw_energy: Energy computed directly from ``phase_fraction``.
            branching: Sorted tuple of all alias energy candidates considered.
            resolved_energy: Alias energy selected with the optional reference value, if available.
            bits_msb_first: Tuple of measured bits ordered from MSB to LSB, when provided.
            bitstring_msb_first: Measured bitstring representation, when provided.
            metadata: Optional metadata dictionary.

        """
        Logger.trace_entering()
        self.method = method
        self.evolution_time = evolution_time
        self.phase_fraction = phase_fraction
        self.phase_angle = phase_angle
        self.canonical_phase_fraction = canonical_phase_fraction
        self.canonical_phase_angle = canonical_phase_angle
        self.raw_energy = raw_energy
        self.branching = branching
        self.resolved_energy = resolved_energy
        self.bits_msb_first = bits_msb_first
        self.bitstring_msb_first = bitstring_msb_first
        self.metadata = metadata
        # Make instance immutable after construction (handled by base class)
        super().__init__()

    @classmethod
    def from_phase_fraction(
        cls,
        *,
        method: str,
        phase_fraction: float,
        evolution_time: float,
        branch_shifts: Iterable[int] = range(-2, 3),
        bits_msb_first: Sequence[int] | None = None,
        bitstring_msb_first: str | None = None,
        reference_energy: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> "QpeResult":
        """Construct a :class:`QpeResult` from a measured phase fraction.

        Args:
            method: Phase estimation algorithm or workflow label.
            phase_fraction: Measured phase fraction in ``[0, 1)``.
            evolution_time: Evolution time ``t`` used in ``U = exp(-i H t)``.
            branch_shifts: Integer multiples of ``2π / t`` examined when forming alias candidates.
            bits_msb_first: Optional measured bits ordered from MSB to LSB.
            bitstring_msb_first: Optional string representation of the measured bits.
            reference_energy: Optional target value used to select the canonical alias branch.
            metadata: Optional dictionary copied into the result for caller-defined context.

        Returns:
            QpeResult: Populated :class:`QpeResult` instance reflecting the supplied data.

        """
        Logger.trace_entering()
        method_label = str(method.value) if hasattr(method, "value") else str(method)

        normalized_phase = float(phase_fraction % 1.0)
        phase_angle = float(normalized_phase * (2 * np.pi))
        raw_energy = energy_from_phase(normalized_phase, evolution_time=evolution_time)

        branching = tuple(
            energy_alias_candidates(
                raw_energy,
                evolution_time=evolution_time,
                shift_range=branch_shifts,
            )
        )

        resolved = None
        if reference_energy is not None:
            resolved = resolve_energy_aliases(
                raw_energy,
                evolution_time=evolution_time,
                reference_energy=reference_energy,
                shift_range=branch_shifts,
            )

        canonical_phase_fraction = normalized_phase
        canonical_phase_angle = phase_angle
        if resolved is not None:
            resolved_angle = float(resolved * evolution_time)
            canonical_phase_angle = float((resolved_angle + 2 * np.pi) % (2 * np.pi))
            canonical_phase_fraction = float(canonical_phase_angle / (2 * np.pi))

        normalized_bits: tuple[int, ...] | None = None
        bitstring = bitstring_msb_first
        if bits_msb_first is not None:
            normalized_bits = tuple(int(bit) for bit in bits_msb_first)
            if bitstring is None:
                bitstring = "".join(str(bit) for bit in normalized_bits)

        metadata_copy = dict(metadata) if metadata is not None else None

        return cls(
            method=method_label,
            evolution_time=float(evolution_time),
            phase_fraction=normalized_phase,
            phase_angle=phase_angle,
            canonical_phase_fraction=canonical_phase_fraction,
            canonical_phase_angle=canonical_phase_angle,
            raw_energy=raw_energy,
            branching=branching,
            resolved_energy=resolved,
            bits_msb_first=normalized_bits,
            bitstring_msb_first=bitstring,
            metadata=metadata_copy,
        )

    # DataClass interface implementation
    def get_summary(self) -> str:
        """Get a human-readable summary of the QPE result.

        Returns:
            str: Summary string describing the QPE result.

        """
        lines = [
            f"QPE Result ({self.method})",
            f"  Evolution time: {self.evolution_time}",
            f"  Phase fraction: {self.phase_fraction:.6f}",
            f"  Raw energy: {self.raw_energy:.6f}",
        ]
        if self.resolved_energy is not None:
            lines.append(f"  Resolved energy: {self.resolved_energy:.6f}")
        if self.bitstring_msb_first is not None:
            lines.append(f"  Bitstring: {self.bitstring_msb_first}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Convert the QPE result to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the QPE result.

        """
        data = {
            "method": self.method,
            "evolution_time": self.evolution_time,
            "phase_fraction": self.phase_fraction,
            "phase_angle": self.phase_angle,
            "canonical_phase_fraction": self.canonical_phase_fraction,
            "canonical_phase_angle": self.canonical_phase_angle,
            "raw_energy": self.raw_energy,
            "branching": list(self.branching),
        }

        if self.resolved_energy is not None:
            data["resolved_energy"] = self.resolved_energy
        if self.bits_msb_first is not None:
            data["bits_msb_first"] = list(self.bits_msb_first)
        if self.bitstring_msb_first is not None:
            data["bitstring_msb_first"] = self.bitstring_msb_first
        if self.metadata is not None:
            data["metadata"] = self.metadata

        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the QPE result to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write the QPE result to.

        """
        self._add_hdf5_version(group)
        group.attrs["method"] = self.method
        group.attrs["evolution_time"] = self.evolution_time
        group.attrs["phase_fraction"] = self.phase_fraction
        group.attrs["phase_angle"] = self.phase_angle
        group.attrs["canonical_phase_fraction"] = self.canonical_phase_fraction
        group.attrs["canonical_phase_angle"] = self.canonical_phase_angle
        group.attrs["raw_energy"] = self.raw_energy

        group.create_dataset("branching", data=np.array(self.branching))

        if self.resolved_energy is not None:
            group.attrs["resolved_energy"] = self.resolved_energy
        if self.bits_msb_first is not None:
            group.create_dataset("bits_msb_first", data=np.array(self.bits_msb_first))
        if self.bitstring_msb_first is not None:
            group.attrs["bitstring_msb_first"] = self.bitstring_msb_first
        if self.metadata is not None:
            # Store metadata as JSON string since HDF5 doesn't handle nested dicts well
            group.attrs["metadata"] = json.dumps(self.metadata)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "QpeResult":
        """Create a QPE result from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized QPE result data.

        Returns:
            QpeResult: New instance reconstructed from the JSON data.

        Raises:
            RuntimeError: If version field is missing or incompatible.

        """
        cls._validate_json_version(cls._serialization_version, json_data)

        return cls(
            method=json_data["method"],
            evolution_time=json_data["evolution_time"],
            phase_fraction=json_data["phase_fraction"],
            phase_angle=json_data["phase_angle"],
            canonical_phase_fraction=json_data["canonical_phase_fraction"],
            canonical_phase_angle=json_data["canonical_phase_angle"],
            raw_energy=json_data["raw_energy"],
            branching=tuple(json_data["branching"]),
            resolved_energy=json_data.get("resolved_energy"),
            bits_msb_first=tuple(json_data["bits_msb_first"]) if "bits_msb_first" in json_data else None,
            bitstring_msb_first=json_data.get("bitstring_msb_first"),
            metadata=json_data.get("metadata"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "QpeResult":
        """Load a QPE result from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file containing the QPE result data.

        Returns:
            QpeResult: New instance reconstructed from the HDF5 data.

        Raises:
            RuntimeError: If version attribute is missing or incompatible.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)

        branching = tuple(group["branching"][:])

        bits_msb_first = None
        if "bits_msb_first" in group:
            bits_msb_first = tuple(group["bits_msb_first"][:])

        metadata = None
        if "metadata" in group.attrs:
            metadata = json.loads(group.attrs["metadata"])

        return cls(
            method=group.attrs["method"],
            evolution_time=group.attrs["evolution_time"],
            phase_fraction=group.attrs["phase_fraction"],
            phase_angle=group.attrs["phase_angle"],
            canonical_phase_fraction=group.attrs["canonical_phase_fraction"],
            canonical_phase_angle=group.attrs["canonical_phase_angle"],
            raw_energy=group.attrs["raw_energy"],
            branching=branching,
            resolved_energy=group.attrs.get("resolved_energy"),
            bits_msb_first=bits_msb_first,
            bitstring_msb_first=group.attrs.get("bitstring_msb_first"),
            metadata=metadata,
        )
