r"""QDK/Chemistry SOSSA (Sum of Squares with Ancilla) block encoding container.

References:
    Low, G. H. et al. "Quantum simulation of chemistry with sublinear scaling
    in basis size." arXiv:2502.15882v1 (2025).

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from .block_encoding import BlockEncodingContainer, wavefunction_from_hdf5, wavefunction_to_hdf5

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

    The SELECT oracle applies Givens rotations + Majorana operators controlled on
    the :math:`(x_o, b)` state. Rotation angles define the orbital basis transformation.

    """

    rotation_angles: np.ndarray
    r"""Givens rotation angles for D1/Q1, shape :math:`[N, N-1]`."""

    sf_rotation_angles: np.ndarray
    r"""Givens rotation angles for SF generators, shape :math:`[R \cdot (B+1), N-1]`."""

    num_orbitals: int
    """Number of spatial orbitals N (system register size = 2N spin-orbitals)."""

    num_ranks: int
    """Number of DFTHC ranks R."""

    num_copies: int
    """Number of copies C."""

    num_bases: int
    """Number of bases B."""

    num_d1: int
    """Number of D1 entries (indices [0, num_d1) in x_o)."""

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save to HDF5."""
        group.create_dataset("rotation_angles", data=self.rotation_angles)
        group.create_dataset("sf_rotation_angles", data=self.sf_rotation_angles)
        group.attrs["num_orbitals"] = self.num_orbitals
        group.attrs["num_ranks"] = self.num_ranks
        group.attrs["num_copies"] = self.num_copies
        group.attrs["num_bases"] = self.num_bases
        group.attrs["num_d1"] = self.num_d1

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SOSSASelect":
        """Load from HDF5."""
        return cls(
            rotation_angles=np.array(group["rotation_angles"]),
            sf_rotation_angles=np.array(group["sf_rotation_angles"]),
            num_orbitals=int(group.attrs["num_orbitals"]),
            num_ranks=int(group.attrs["num_ranks"]),
            num_copies=int(group.attrs["num_copies"]),
            num_bases=int(group.attrs["num_bases"]),
            num_d1=int(group.attrs["num_d1"]),
        )


class SOSSAContainer(BlockEncodingContainer):
    r"""Container for the SOSSA (Sum of Squares with Ancilla) block encoding.

    Stores the two-level PREPARE and rotation-based SELECT for the DFTHC walk operator.

    The walk operator is (arXiv:2502.15882, Eq. 77):

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
        quantum_walk: bool = True,
    ) -> None:
        r"""Initialize a SOSSAContainer.

        Args:
            outer_prepare: The outer PREPARE oracle data.
            inner_prepare: The inner (conditional) PREPARE oracle data.
            select: The SELECT oracle data (Givens rotations + Majorana).
            normalization: The block encoding normalization :math:`\Lambda`.
            power: Number of times to apply the walk operator.
            quantum_walk: Whether to use quantum walk (always True for SOSSA/QPE).

        """
        self._power = power
        self._quantum_walk = quantum_walk
        self.outer_prepare = outer_prepare
        self.inner_prepare = inner_prepare
        self.select = select
        self.normalization = normalization

        super().__init__()

    @property
    def power(self) -> int:
        """Number of times to apply the walk operator."""
        return self._power

    @property
    def quantum_walk(self) -> bool:
        """Whether to wrap with a quantum walk operator."""
        return self._quantum_walk

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (system + ancilla registers).

        System register: 2N spin-orbitals.
        Ancilla: x_o + b + spin + alias-compare + flag qubits.

        """
        num_system = 2 * self.select.num_orbitals
        num_ancilla = self.outer_prepare.get_orbitals().num_modes() + self.inner_prepare.num_inner_qubits
        # Additional ancilla: 2 spin qubits + flag + keep register (approximate)
        num_ancilla += 3
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
            "quantum_walk": self.quantum_walk,
            "normalization": self.normalization,
            "outer_prepare": self.outer_prepare.to_json(),
            "inner_prepare": {
                "conditional_coefficients": self.inner_prepare.conditional_coefficients.tolist(),
                "num_inner_qubits": self.inner_prepare.num_inner_qubits,
                "num_bases": self.inner_prepare.num_bases,
                **(
                    {"free_rider_data": self.inner_prepare.free_rider_data.tolist()}
                    if self.inner_prepare.free_rider_data is not None
                    else {}
                ),
            },
            "select": {
                "rotation_angles": self.select.rotation_angles.tolist(),
                "sf_rotation_angles": self.select.sf_rotation_angles.tolist(),
                "num_orbitals": self.select.num_orbitals,
                "num_ranks": self.select.num_ranks,
                "num_copies": self.select.num_copies,
                "num_bases": self.select.num_bases,
                "num_d1": self.select.num_d1,
            },
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the SOSSAContainer to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["power"] = self.power
        group.attrs["quantum_walk"] = self.quantum_walk
        group.attrs["normalization"] = self.normalization
        wavefunction_to_hdf5(self.outer_prepare, group.create_group("outer_prepare"))
        self.inner_prepare.to_hdf5(group.create_group("inner_prepare"))
        self.select.to_hdf5(group.create_group("select"))

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "SOSSAContainer":
        """Create a SOSSAContainer from a JSON dictionary."""
        cls._validate_json_version(cls._serialization_version, json_data)

        from qdk_chemistry.data import Wavefunction  # noqa: PLC0415

        outer_prepare = Wavefunction.from_json(json_data["outer_prepare"])

        ip = json_data["inner_prepare"]
        fr_data = np.array(ip["free_rider_data"], dtype=bool) if "free_rider_data" in ip else None
        inner_prepare = SOSSAInnerPrepare(
            conditional_coefficients=np.array(ip["conditional_coefficients"], dtype=float),
            num_inner_qubits=ip["num_inner_qubits"],
            num_bases=ip["num_bases"],
            free_rider_data=fr_data,
        )

        sel = json_data["select"]
        select = SOSSASelect(
            rotation_angles=np.array(sel["rotation_angles"], dtype=float),
            sf_rotation_angles=np.array(sel["sf_rotation_angles"], dtype=float),
            num_orbitals=sel["num_orbitals"],
            num_ranks=sel["num_ranks"],
            num_copies=sel["num_copies"],
            num_bases=sel["num_bases"],
            num_d1=sel["num_d1"],
        )

        return cls(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            normalization=json_data["normalization"],
            power=json_data.get("power", 1),
            quantum_walk=json_data.get("quantum_walk", True),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SOSSAContainer":
        """Load a SOSSAContainer from an HDF5 group."""
        outer_prepare = wavefunction_from_hdf5(group["outer_prepare"])
        inner_prepare = SOSSAInnerPrepare.from_hdf5(group["inner_prepare"])
        select = SOSSASelect.from_hdf5(group["select"])
        return cls(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            normalization=float(group.attrs["normalization"]),
            power=int(group.attrs["power"]),
            quantum_walk=bool(group.attrs.get("quantum_walk", True)),
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
            f"  Quantum Walk: {'Yes' if self.quantum_walk else 'No'}"
        )
