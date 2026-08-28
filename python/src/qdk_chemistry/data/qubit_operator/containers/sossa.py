"""Sum-of-squares (SOSSA) qubit operator container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer

if TYPE_CHECKING:
    import h5py

__all__ = ["FactorizedHamiltonianMetadata", "RotatedPaulis", "SOSContainer"]


def _complex_block_to_json(coeffs: np.ndarray) -> dict[str, Any]:
    """Serialize a complex coefficient array as split real/imaginary lists."""
    arr = np.asarray(coeffs, dtype=complex)
    return {"real": arr.real.tolist(), "imag": arr.imag.tolist()}


def _complex_block_from_json(data: dict[str, Any]) -> np.ndarray:
    """Rebuild a complex coefficient array from split real/imaginary lists."""
    return np.asarray(data["real"], dtype=float) + 1j * np.asarray(data["imag"], dtype=float)


@dataclass(frozen=True, eq=False)
class RotatedPaulis:
    """A block of rotated-Pauli generators: Givens ``angles`` [M, N-1], LCU ``coeffs`` [M, T], and ``paulis``."""

    angles: np.ndarray
    coeffs: np.ndarray
    paulis: tuple[str, ...]

    def __post_init__(self) -> None:
        """Coerce inputs to arrays and a tuple."""
        object.__setattr__(self, "angles", np.asarray(self.angles, dtype=float))
        object.__setattr__(self, "coeffs", np.asarray(self.coeffs, dtype=complex))
        object.__setattr__(self, "paulis", tuple(self.paulis))


@dataclass(frozen=True)
class FactorizedHamiltonianMetadata:
    r"""Dimensions and scalar constants of a factorized Hamiltonian's sum-of-squares form.

    Attributes:
        num_spatial_orbitals: Number of spatial orbitals :math:`N`.
        num_ranks: Number of DFTHC ranks :math:`R`.
        num_bases: Number of bases :math:`B` (``B + 1`` inner entries including the identity).
        num_copies: Number of copies :math:`C`.
        num_positive_one_body_terms: Number of D1 (particle) one-body generators.
        energy_shift: Constant offset :math:`E_{\text{SOS}} + E_{\text{nuc}}`.
        normalization: Block-encoding normalization :math:`\Lambda`, or ``None`` before it is built.

    """

    num_spatial_orbitals: int
    num_ranks: int
    num_bases: int
    num_copies: int
    num_positive_one_body_terms: int
    energy_shift: float
    normalization: float | None = None

    def __post_init__(self) -> None:
        """Coerce the shift and reject dimensions no register width can be built from."""
        object.__setattr__(self, "energy_shift", float(self.energy_shift))
        if self.num_spatial_orbitals <= 0 or not isfinite(self.energy_shift):
            raise ValueError("invalid sum-of-squares metadata")

    def to_json(self) -> dict[str, Any]:
        """Convert the metadata to a JSON dictionary."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FactorizedHamiltonianMetadata:
        """Create metadata from a JSON dictionary."""
        return cls(**data)


class SOSContainer(QubitOperatorContainer):
    """Container for a sum-of-squares qubit operator.

    The one-body and two-body generators are each a
    :class:`~qdk_chemistry.data.qubit_operator.containers.sossa.RotatedPaulis` block
    (``angles``, ``coeffs``, ``paulis``). ``one_body`` uses ``(X +/- iY) / 2``
    with the first ``metadata.num_positive_one_body_terms`` rows the D1 (particle)
    generators and the rest Q1 (hole); ``two_body`` uses ``Z`` with one
    ``[R * C, B + 1]`` coefficient row per ``(rank, copy)`` (columns ``0..B-1``
    rotated-``Z``, column ``B`` identity). Dimensions and scalar constants live in
    :class:`FactorizedHamiltonianMetadata`; the builder derives the inner-PREPARE distribution, outer
    coefficients, and normalization from these blocks.
    """

    _data_type_name = "sossa_container"
    _serialization_version = "0.2.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for SOSSA containers."""
        return "sossa_container"

    def __init__(
        self,
        one_body: RotatedPaulis,
        two_body: RotatedPaulis,
        encoding: str | None,
        fermion_mode_order: str | None,
        metadata: FactorizedHamiltonianMetadata,
    ) -> None:
        """Initialize a sum-of-squares container."""
        self.one_body = one_body
        self.two_body = two_body
        self.metadata = metadata
        if len(self.one_body.angles) != len(self.one_body.coeffs):
            raise ValueError("one-body angles and coefficients must have matching generator counts")
        if not 0 <= metadata.num_positive_one_body_terms <= len(self.one_body.angles):
            raise ValueError("num_positive_one_body_terms must be between 0 and the one-body generator count")
        expected_two_body = (metadata.num_ranks * metadata.num_copies, metadata.num_bases + 1)
        if self.two_body.coeffs.size and self.two_body.coeffs.shape != expected_two_body:
            raise ValueError("two_body_coeffs must have shape [num_ranks * num_copies, num_bases + 1]")
        super().__init__(encoding, fermion_mode_order)

    @property
    def type(self) -> str:
        """Return the container type."""
        return "sossa"

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits (two spin-orbitals per spatial orbital)."""
        return 2 * self.metadata.num_spatial_orbitals

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_arg(h, self.to_json())

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary."""
        return self._add_json_version(
            {
                "container_type": self.type,
                "metadata": self.metadata.to_json(),
                "one_body_angles": self.one_body.angles.tolist(),
                "one_body_coeffs": _complex_block_to_json(self.one_body.coeffs),
                "one_body_paulis": list(self.one_body.paulis),
                "two_body_angles": self.two_body.angles.tolist(),
                "two_body_coeffs": _complex_block_to_json(self.two_body.coeffs),
                "two_body_paulis": list(self.two_body.paulis),
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
    def from_json(cls, json_data: dict[str, Any]) -> SOSContainer:
        """Create a sum-of-squares container from JSON."""
        cls._validate_json_version(cls._serialization_version, json_data)
        one_body = RotatedPaulis(
            np.asarray(json_data["one_body_angles"], dtype=float),
            _complex_block_from_json(json_data["one_body_coeffs"]),
            tuple(json_data.get("one_body_paulis", ("X", "Y"))),
        )
        two_body = RotatedPaulis(
            np.asarray(json_data["two_body_angles"], dtype=float),
            _complex_block_from_json(json_data["two_body_coeffs"]),
            tuple(json_data.get("two_body_paulis", ("Z",))),
        )
        return cls(
            one_body,
            two_body,
            json_data.get("encoding"),
            json_data.get("fermion_mode_order"),
            FactorizedHamiltonianMetadata.from_json(json_data["metadata"]),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> SOSContainer:
        """Create a sum-of-squares container from HDF5."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls.from_json(json.loads(group.attrs["payload"]))

    def get_summary(self) -> str:
        """Return a summary of the sum-of-squares container."""
        num_d1 = self.metadata.num_positive_one_body_terms
        num_q1 = len(self.one_body.angles) - num_d1
        num_sf = len(self.two_body.angles)
        return (
            f"SOS Qubit Operator\n  Number of qubits: {self.num_qubits}\n"
            f"  D1/Q1/SF generators: {num_d1}/{num_q1}/{num_sf}\n"
        )
