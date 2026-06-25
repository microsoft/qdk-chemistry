r"""Sum of Squares Spectral Amplification (SOSSA) block encoding container.

References:
    Low, G. H. et al. "Fast quantum simulation of electronic structure by spectrum amplification."
    :cite:`Low2025`.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from math import ceil, log2
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from .block_encoding import BlockEncodingContainer, _wavefunction_from_hdf5, _wavefunction_to_hdf5

if TYPE_CHECKING:
    from qdk_chemistry.data import Wavefunction

__all__ = ["SOSSAContainer", "SOSSAInnerPrepare", "SOSSASelect"]


@dataclass(frozen=True)
class SOSSAInnerPrepare:
    r"""Inner (conditional) PREPARE oracle for the SOSSA block encoding.

    Prepares a superposition over bases :math:`b \in [0, B]` conditioned on :math:`x_o`.
    Uses coherent alias sampling over the 2D distribution :math:`[X_o][B+1]`.

    """

    conditional_coefficients: np.ndarray
    r"""2D amplitude array, shape :math:`[X_o, B+1]`. Row :math:`x_o` gives the inner distribution."""

    num_inner_qubits: int
    r"""Number of qubits in the :math:`b` register: :math:`\lceil\log_2(B+1)\rceil`."""

    num_bases: int
    """Number of bases B (B+1 entries including identity term)."""

    free_rider_data: np.ndarray | None = None
    r"""Optional 2D boolean array, shape :math:`[X_o, n_{\text{fr}}]`.

    Classical bits loaded into the free-rider register by the 2D QROM
    alongside alias sampling data. Each row gives the free-rider bits
    for the corresponding :math:`x_o` condition value.
    """

    def to_json(self) -> dict[str, Any]:
        """Save to a JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "conditional_coefficients": self.conditional_coefficients.tolist(),
            "num_inner_qubits": self.num_inner_qubits,
            "num_bases": self.num_bases,
        }
        if self.free_rider_data is not None:
            data["free_rider_data"] = self.free_rider_data.tolist()
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SOSSAInnerPrepare":
        """Load from a JSON dictionary."""
        fr_data = np.array(data["free_rider_data"], dtype=bool) if "free_rider_data" in data else None
        return cls(
            conditional_coefficients=np.array(data["conditional_coefficients"], dtype=float),
            num_inner_qubits=data["num_inner_qubits"],
            num_bases=data["num_bases"],
            free_rider_data=fr_data,
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save to HDF5."""
        group.create_dataset("conditional_coefficients", data=self.conditional_coefficients)
        group.attrs["num_inner_qubits"] = self.num_inner_qubits
        group.attrs["num_bases"] = self.num_bases
        if self.free_rider_data is not None:
            group.create_dataset("free_rider_data", data=self.free_rider_data)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SOSSAInnerPrepare":
        """Load from HDF5."""
        free_rider = np.array(group["free_rider_data"]) if "free_rider_data" in group else None
        return cls(
            conditional_coefficients=np.array(group["conditional_coefficients"]),
            num_inner_qubits=int(group.attrs["num_inner_qubits"]),
            num_bases=int(group.attrs["num_bases"]),
            free_rider_data=free_rider,
        )


@dataclass(frozen=True)
class SOSSASelect:
    r"""SELECT oracle for the SOSSA block encoding.

    The SELECT oracle applies Givens rotations + SpinSwap + Majorana operators controlled on
    the :math:`(x_o, b)` state. Rotation angles define the orbital basis transformation.

    """

    one_body_rotation_angles: np.ndarray
    r"""Givens rotation angles for D1/Q1, shape :math:`[N, N-1]`."""

    two_body_rotation_angles: np.ndarray
    r"""Givens rotation angles for SF generators, shape :math:`[R \cdot (B+1), N-1]`."""

    num_orbitals: int
    """Number of spatial orbitals N (system register size = 2N spin-orbitals)."""

    num_ranks: int
    """Number of DFTHC ranks R."""

    num_copies: int
    """Number of copies C."""

    num_bases: int
    """Number of bases B."""

    num_positive_one_body_terms: int
    """Number of D1 entries (indices [0, num_d1) in x_o)."""

    def to_json(self) -> dict[str, Any]:
        """Save to a JSON-serializable dictionary."""
        return {
            "one_body_rotation_angles": self.one_body_rotation_angles.tolist(),
            "two_body_rotation_angles": self.two_body_rotation_angles.tolist(),
            "num_orbitals": self.num_orbitals,
            "num_ranks": self.num_ranks,
            "num_copies": self.num_copies,
            "num_bases": self.num_bases,
            "num_positive_one_body_terms": self.num_positive_one_body_terms,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SOSSASelect":
        """Load from a JSON dictionary."""
        return cls(
            one_body_rotation_angles=np.array(data["one_body_rotation_angles"], dtype=float),
            two_body_rotation_angles=np.array(data["two_body_rotation_angles"], dtype=float),
            num_orbitals=data["num_orbitals"],
            num_ranks=data["num_ranks"],
            num_copies=data["num_copies"],
            num_bases=data["num_bases"],
            num_positive_one_body_terms=data["num_positive_one_body_terms"],
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save to HDF5."""
        group.create_dataset("one_body_rotation_angles", data=self.one_body_rotation_angles)
        group.create_dataset("two_body_rotation_angles", data=self.two_body_rotation_angles)
        group.attrs["num_orbitals"] = self.num_orbitals
        group.attrs["num_ranks"] = self.num_ranks
        group.attrs["num_copies"] = self.num_copies
        group.attrs["num_bases"] = self.num_bases
        group.attrs["num_positive_one_body_terms"] = self.num_positive_one_body_terms

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SOSSASelect":
        """Load from HDF5."""
        return cls(
            one_body_rotation_angles=np.array(group["one_body_rotation_angles"]),
            two_body_rotation_angles=np.array(group["two_body_rotation_angles"]),
            num_orbitals=int(group.attrs["num_orbitals"]),
            num_ranks=int(group.attrs["num_ranks"]),
            num_copies=int(group.attrs["num_copies"]),
            num_bases=int(group.attrs["num_bases"]),
            num_positive_one_body_terms=int(group.attrs["num_positive_one_body_terms"]),
        )


class SOSSAContainer(BlockEncodingContainer):
    r"""Container for the Sum of Squares Spectral Amplification (SOSSA) block encoding.

    The walk operator is (:cite:`Low2025`, Eq. 77):

    .. math::

        W = \mathrm{Ref}_{a,B} \cdot U^\dagger \cdot \mathrm{Ref}_B \cdot U

    where :math:`U = \text{OuterPREP} \cdot \text{within\{InnerPREP\} apply\{SELECT\}}`.

    """

    _data_type_name = "sossa_container"
    _serialization_version = "0.1.0"

    def __init__(
        self,
        outer_prepare: "Wavefunction",
        inner_prepare: SOSSAInnerPrepare,
        select: SOSSASelect,
        normalization: float,
        power: int = 1,
        energy_shift: float = 0.0,
    ) -> None:
        r"""Initialize a SOSSAContainer.

        Args:
            outer_prepare: The outer PREPARE Wavefunction.
            inner_prepare: The inner (conditional) PREPARE oracle data.
            select: The SELECT oracle data (Givens rotations + Spin swap + Majorana).
            normalization: The block encoding normalization :math:`\Lambda`.
            power: Number of times to apply the walk operator.
            energy_shift: Energy shift :math:`E_{\text{SOS}} + E_{\text{nuc}}`
                to add when recovering total energy from the measured phase.

        """
        self._power = power
        self.outer_prepare = outer_prepare
        self.inner_prepare = inner_prepare
        self.select = select
        self.normalization = normalization
        self.energy_shift = energy_shift

        super().__init__()

    @property
    def power(self) -> int:
        """Number of times to apply the walk operator."""
        return self._power

    @property
    def num_qubits(self) -> int:
        """Total number of qubits to be allocated in QPE or other callers.

        System register: 2N spin-orbitals.
        Ancilla: x_o register + inner register (b + free-rider) + 2 spin qubits.
        This doesn't equal the total qubits of SOSSA since the SOSSA circuit allocate
        and free ancillary qubits internally.
        """
        num_system = 2 * self.select.num_orbitals
        # Outer register: ceil(log2(x_o_dim))
        x_o_dim = self.select.num_orbitals + self.select.num_ranks * self.select.num_copies
        num_outer = ceil(log2(x_o_dim)) if x_o_dim > 1 else 1
        # Inner register: b bits + free-rider bits
        num_ranks = self.select.num_ranks
        rank_bits = ceil(log2(num_ranks)) if num_ranks > 1 else 0
        num_free_rider_bits = 2 + rank_bits  # isSF + dvsq + rank
        num_inner = self.inner_prepare.num_inner_qubits + num_free_rider_bits
        # Spin register: 2 (spinDQ, spinSF)
        num_ancilla = num_outer + num_inner + 2
        return num_system + num_ancilla

    @property
    def type(self) -> str:
        """Get the type of the unitary container."""
        return "sossa"

    def to_json(self) -> dict[str, Any]:
        """Save the SOSSAContainer to a JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "container_type": self.type,
            "power": self.power,
            "normalization": self.normalization,
            "energy_shift": self.energy_shift,
            "outer_prepare": self.outer_prepare.to_json(),
            "inner_prepare": self.inner_prepare.to_json(),
            "select": self.select.to_json(),
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the SOSSAContainer to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["power"] = self.power
        group.attrs["normalization"] = self.normalization
        group.attrs["energy_shift"] = self.energy_shift
        _wavefunction_to_hdf5(self.outer_prepare, group.create_group("outer_prepare"))
        self.inner_prepare.to_hdf5(group.create_group("inner_prepare"))
        self.select.to_hdf5(group.create_group("select"))

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "SOSSAContainer":
        """Create a SOSSAContainer from a JSON dictionary."""
        cls._validate_json_version(cls._serialization_version, json_data)

        from qdk_chemistry.data import Wavefunction  # noqa: PLC0415

        outer_prepare = Wavefunction.from_json(json_data["outer_prepare"])
        inner_prepare = SOSSAInnerPrepare.from_json(json_data["inner_prepare"])
        select = SOSSASelect.from_json(json_data["select"])

        return cls(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            normalization=json_data["normalization"],
            power=json_data.get("power", 1),
            energy_shift=json_data.get("energy_shift", 0.0),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SOSSAContainer":
        """Load a SOSSAContainer from an HDF5 group."""
        outer_prepare = _wavefunction_from_hdf5(group["outer_prepare"])
        inner_prepare = SOSSAInnerPrepare.from_hdf5(group["inner_prepare"])
        select = SOSSASelect.from_hdf5(group["select"])
        return cls(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            normalization=float(group.attrs["normalization"]),
            power=int(group.attrs["power"]),
            energy_shift=float(group.attrs.get("energy_shift", 0.0)),
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the SOSSA container."""
        n = self.select.num_orbitals
        r = self.select.num_ranks
        b = self.select.num_bases
        c = self.select.num_copies
        return (
            f"SOSSA Container (DFTHC block encoding):\n"
            f"  Power: {self.power}\n"
            f"  Orbitals N={n}, Ranks R={r}, Bases B={b}, Copies C={c}\n"
            f"  Xo = N + R*C = {n + r * c}\n"
            f"  Normalization Lambda = {self.normalization:.6f}\n"
            f"  Outer PREPARE: {self.outer_prepare.get_orbitals().num_modes()} qubits\n"
            f"  Inner PREPARE: {self.inner_prepare.num_inner_qubits} qubits, {b + 1} basis entries\n"
            f"  System: {2 * n} spin-orbitals\n"
        )
