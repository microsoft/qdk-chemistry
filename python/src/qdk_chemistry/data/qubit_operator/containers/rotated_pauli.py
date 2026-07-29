"""Rotated-Pauli qubit operator container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer

if TYPE_CHECKING:
    from collections.abc import Sequence

    import h5py

__all__ = ["RotatedPauliContainer"]


class RotatedPauliContainer(QubitOperatorContainer):
    """Container for a linear combination of rotated Pauli terms.

    Each term ``i`` is the Pauli string ``pauli_strings[i]`` scaled by
    ``coefficients[i]`` and dressed by the orbital-rotation basis vector
    ``rotations[i]`` (``None`` for un-rotated identity terms).
    """

    def __init__(
        self,
        pauli_strings: Sequence[str],
        coefficients: Sequence[complex] | np.ndarray,
        rotations: Sequence[np.ndarray | None],
        num_qubits: int,
        encoding: str | None,
        fermion_mode_order: str | None,
    ) -> None:
        """Initialize a rotated-Pauli container."""
        self.pauli_strings = tuple(pauli_strings)
        self.coefficients = np.asarray(coefficients, dtype=complex)
        self.rotations = tuple(
            None if rotation is None else np.asarray(rotation, dtype=float) for rotation in rotations
        )
        self._num_qubits = num_qubits
        if not len(self.pauli_strings) == len(self.coefficients) == len(self.rotations):
            raise ValueError("pauli_strings, coefficients, and rotations must have equal length")
        super().__init__(encoding, fermion_mode_order)

    @property
    def type(self) -> str:
        """Return the container type."""
        return "rotated_pauli"

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self._num_qubits

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_arg(h, self.to_json())

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary."""
        return self._add_json_version(
            {
                "container_type": self.type,
                "num_qubits": self.num_qubits,
                "encoding": self.encoding,
                "fermion_mode_order": str(self.fermion_mode_order) if self.fermion_mode_order is not None else None,
                "pauli_strings": list(self.pauli_strings),
                "coefficients": {
                    "real": self.coefficients.real.tolist(),
                    "imag": self.coefficients.imag.tolist(),
                },
                "rotations": [None if rotation is None else rotation.tolist() for rotation in self.rotations],
            }
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the container to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["payload"] = json.dumps(self.to_json())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> RotatedPauliContainer:
        """Create a rotated-Pauli container from JSON."""
        cls._validate_json_version(cls._serialization_version, json_data)
        coefficients = np.array(json_data["coefficients"]["real"], dtype=float) + 1j * np.array(
            json_data["coefficients"]["imag"], dtype=float
        )
        rotations = [
            None if rotation is None else np.asarray(rotation, dtype=float) for rotation in json_data["rotations"]
        ]
        return cls(
            json_data["pauli_strings"],
            coefficients,
            rotations,
            json_data["num_qubits"],
            json_data.get("encoding"),
            json_data.get("fermion_mode_order"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RotatedPauliContainer:
        """Create a rotated-Pauli container from HDF5."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls.from_json(json.loads(group.attrs["payload"]))

    def get_summary(self) -> str:
        """Return a summary of the rotated-Pauli container."""
        return (
            f"Rotated Pauli Operator\n  Number of qubits: {self.num_qubits}\n"
            f"  Number of terms: {len(self.pauli_strings)}\n"
        )
