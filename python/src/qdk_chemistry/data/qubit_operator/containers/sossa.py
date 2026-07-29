"""Sum-of-squares (SOSSA) qubit operator container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer
from qdk_chemistry.data.qubit_operator.containers.rotated_pauli import RotatedPauliContainer

if TYPE_CHECKING:
    import h5py
    from numpy.typing import ArrayLike

__all__ = ["SOSSAContainer"]


class SOSSAContainer(QubitOperatorContainer):
    """Container for a sum-of-squares qubit operator.

    The SOSSA generators are split into three rotated-Pauli combinations:
    :attr:`d1` (particle one-body), :attr:`q1` (hole one-body), and :attr:`sf`
    (spin-free two-body). Each is a :class:`RotatedPauliContainer` whose
    ``rotations`` hold the Givens rotation angles derived from the orbital
    rotation ``U`` (one angle vector per orbital for ``d1``/``q1``, one per
    rank/basis for ``sf``). The ``inner_coefficients`` give the inner-PREPARE
    conditional distribution and ``energy_shift`` the constant offset. The
    circuit builder derives the outer LCU coefficients and the block-encoding
    normalization from these.
    """

    def __init__(
        self,
        num_spatial_orbitals: int,
        energy_shift: float,
        num_ranks: int,
        num_bases: int,
        num_copies: int,
        d1: RotatedPauliContainer,
        q1: RotatedPauliContainer,
        sf: RotatedPauliContainer,
        inner_coefficients: ArrayLike,
        encoding: str | None,
        fermion_mode_order: str | None,
    ) -> None:
        """Initialize a sum-of-squares container."""
        self.num_spatial_orbitals = num_spatial_orbitals
        self.energy_shift = float(energy_shift)
        self.num_ranks = num_ranks
        self.num_bases = num_bases
        self.num_copies = num_copies
        self.d1 = d1
        self.q1 = q1
        self.sf = sf
        self.inner_coefficients = inner_coefficients
        if num_spatial_orbitals <= 0 or not isfinite(self.energy_shift):
            raise ValueError("invalid sum-of-squares container")
        if not all(isinstance(part, RotatedPauliContainer) for part in (self.d1, self.q1, self.sf)):
            raise TypeError("d1, q1, and sf must be RotatedPauliContainer instances")
        super().__init__(encoding, fermion_mode_order)

    @property
    def type(self) -> str:
        """Return the container type."""
        return "sossa"

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits (two spin-orbitals per spatial orbital)."""
        return 2 * self.num_spatial_orbitals

    @property
    def num_positive_one_body_terms(self) -> int:
        """Return the number of D1 (particle) one-body terms."""
        return len(self.d1.rotations)

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_arg(h, self.to_json())

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary."""
        return self._add_json_version(
            {
                "container_type": self.type,
                "num_spatial_orbitals": self.num_spatial_orbitals,
                "energy_shift": self.energy_shift,
                "num_ranks": self.num_ranks,
                "num_bases": self.num_bases,
                "num_copies": self.num_copies,
                "d1": self.d1.to_json(),
                "q1": self.q1.to_json(),
                "sf": self.sf.to_json(),
                "inner_coefficients": np.asarray(self.inner_coefficients).tolist(),
                "encoding": self.encoding,
                "fermion_mode_order": str(self.fermion_mode_order) if self.fermion_mode_order is not None else None,
            }
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the container to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["payload"] = json.dumps(self.to_json())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> SOSSAContainer:
        """Create a sum-of-squares container from JSON."""
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls(
            json_data["num_spatial_orbitals"],
            json_data["energy_shift"],
            json_data["num_ranks"],
            json_data["num_bases"],
            json_data["num_copies"],
            RotatedPauliContainer.from_json(json_data["d1"]),
            RotatedPauliContainer.from_json(json_data["q1"]),
            RotatedPauliContainer.from_json(json_data["sf"]),
            np.asarray(json_data["inner_coefficients"], dtype=float),
            json_data.get("encoding"),
            json_data.get("fermion_mode_order"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> SOSSAContainer:
        """Create a sum-of-squares container from HDF5."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls.from_json(json.loads(group.attrs["payload"]))

    def get_summary(self) -> str:
        """Return a summary of the sum-of-squares container."""
        return (
            f"SOS Qubit Operator\n  Number of qubits: {self.num_qubits}\n"
            f"  Number of spatial orbitals: {self.num_spatial_orbitals}\n"
            f"  D1/Q1/SF terms: {len(self.d1.rotations)}/{len(self.q1.rotations)}/{len(self.sf.rotations)}\n"
        )
