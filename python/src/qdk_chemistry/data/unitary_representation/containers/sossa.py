r"""Sum of Squares Spectral Amplification (SOSSA) block encoding container.

References:
    Low, G. H. et al. "Fast quantum simulation of electronic structure by spectrum amplification."
    :cite:`Low2025`.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from dataclasses import dataclass
from math import ceil, log2
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.qubit_operator.containers.sossa import FactorizedHamiltonianMetadata

from .block_encoding import _wavefunction_from_hdf5, _wavefunction_to_hdf5
from .quantum_walk import QuantumWalkContainer

if TYPE_CHECKING:
    from qdk_chemistry.data import Wavefunction

__all__ = ["SOSSAInnerPrepare", "SOSSASelect", "SOSSAWalkContainer", "sossa_register_bits"]


def sossa_register_bits(num_orbitals: int, num_ranks: int, num_bases: int, num_copies: int) -> dict[str, int]:
    r"""Return the structural ancilla register widths shared across the SOSSA workflow.

    These depend only on the block-encoding dimensions ``(N, R, B, C)`` and are the single
    source of truth for the builder, the walk container, and the circuit mapper.

    Args:
        num_orbitals: Number of spatial orbitals ``N``.
        num_ranks: Number of DFTHC ranks ``R``.
        num_bases: Number of bases ``B`` (``B + 1`` inner entries including the identity term).
        num_copies: Number of copies ``C``.

    Returns:
        dict: ``xo_bits`` (outer index), ``b_bits`` (inner index), ``rank_bits``, and
        ``num_free_rider_bits`` (``2 + rank_bits`` for ``isSF``, ``dvsq``, and the rank).

    """
    x_o_dim = num_orbitals + num_ranks * num_copies
    rank_bits = ceil(log2(num_ranks)) if num_ranks > 1 else 0
    return {
        "xo_bits": ceil(log2(x_o_dim)) if x_o_dim > 1 else 1,
        "b_bits": ceil(log2(num_bases + 1)) if num_bases + 1 > 1 else 1,
        "rank_bits": rank_bits,
        "num_free_rider_bits": 2 + rank_bits,
    }


@dataclass(frozen=True)
class SOSSAInnerPrepare:
    r"""Inner (conditional) PREPARE oracle for the SOSSA block encoding.

    Prepares a superposition over bases :math:`b \in [0, B]` conditioned on :math:`x_o`.
    Uses coherent alias sampling over the 2D distribution :math:`[X_o][B+1]`.

    """

    conditional_coefficients: np.ndarray
    r"""2D amplitude array, shape :math:`[X_o, B+1]`. Row :math:`x_o` gives the inner distribution."""

    free_rider_data: np.ndarray | None = None
    r"""Optional 2D boolean array, shape :math:`[X_o, n_{\text{fr}}]`."""

    def to_json(self) -> dict[str, Any]:
        """Save to a JSON-serializable dictionary."""
        data: dict[str, Any] = {"conditional_coefficients": self.conditional_coefficients.tolist()}
        if self.free_rider_data is not None:
            data["free_rider_data"] = self.free_rider_data.tolist()
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SOSSAInnerPrepare":
        """Load from a JSON dictionary."""
        fr_data = np.array(data["free_rider_data"], dtype=bool) if "free_rider_data" in data else None
        return cls(
            conditional_coefficients=np.array(data["conditional_coefficients"], dtype=float),
            free_rider_data=fr_data,
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save to HDF5."""
        group.create_dataset("conditional_coefficients", data=self.conditional_coefficients)
        if self.free_rider_data is not None:
            group.create_dataset("free_rider_data", data=self.free_rider_data)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SOSSAInnerPrepare":
        """Load from HDF5."""
        free_rider = np.array(group["free_rider_data"]) if "free_rider_data" in group else None
        return cls(
            conditional_coefficients=np.array(group["conditional_coefficients"]),
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

    def to_json(self) -> dict[str, Any]:
        """Save to a JSON-serializable dictionary."""
        return {
            "one_body_rotation_angles": self.one_body_rotation_angles.tolist(),
            "two_body_rotation_angles": self.two_body_rotation_angles.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "SOSSASelect":
        """Load from a JSON dictionary."""
        return cls(
            one_body_rotation_angles=np.array(data["one_body_rotation_angles"], dtype=float),
            two_body_rotation_angles=np.array(data["two_body_rotation_angles"], dtype=float),
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save to HDF5."""
        group.create_dataset("one_body_rotation_angles", data=self.one_body_rotation_angles)
        group.create_dataset("two_body_rotation_angles", data=self.two_body_rotation_angles)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SOSSASelect":
        """Load from HDF5."""
        return cls(
            one_body_rotation_angles=np.array(group["one_body_rotation_angles"]),
            two_body_rotation_angles=np.array(group["two_body_rotation_angles"]),
        )


class SOSSAWalkContainer(QuantumWalkContainer):
    r"""Container for the Sum of Squares Spectral Amplification (SOSSA) block encoding.

    The walk operator is defined inline above Eq. (9) of :cite:`Low2025` and derived in
    its Appendix A 2, with spectrum :math:`e^{\pm i \arccos(E_k/\Lambda - 1)}`:

    .. math::

        W = \mathrm{Ref}_{a,B} \cdot U^\dagger \cdot \mathrm{Ref}_B \cdot U

    where :math:`U = \text{OuterPREP} \cdot \text{within\{InnerPREP\} apply\{SELECT\}}`.

    """

    _serialization_version = "0.1.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for SOSSA-walk containers.

        Distinct from the ``"sossa_container"`` claimed by
        :class:`qdk_chemistry.data.qubit_operator.containers.sossa.SOSContainer`,
        which holds the operator this block encoding is built from.

        Returns:
            ``"sossa_walk_container"``.

        """
        return "sossa_walk_container"

    def __init__(
        self,
        outer_prepare: "Wavefunction",
        inner_prepare: SOSSAInnerPrepare,
        select: SOSSASelect,
        metadata: FactorizedHamiltonianMetadata,
        power: int = 1,
    ) -> None:
        r"""Initialize a SOSSAWalkContainer.

        Args:
            outer_prepare: The outer PREPARE Wavefunction, whose amplitudes are the
                normalized generator one-norms :math:`c_{x_o}`.
            inner_prepare: The inner (conditional) PREPARE oracle data.
            select: The SELECT oracle data (Givens rotations + Spin swap + Majorana).
            metadata: Dimensions and scalar constants carried from the SOS qubit operator.
            power: Number of times to apply the walk operator.

        Raises:
            ValueError: If ``metadata.normalization`` is unset.

        """
        if metadata.normalization is None:
            raise ValueError(
                "metadata.normalization is unset; the block encoding stage must supply it, because "
                "outer_prepare stores only the normalized amplitudes and cannot recover the scale."
            )

        self._power = power
        self.outer_prepare = outer_prepare
        self.inner_prepare = inner_prepare
        self.select = select
        self.metadata = metadata

        super().__init__()

    @property
    def power(self) -> int:
        """Number of times to apply the walk operator."""
        return self._power

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the carried metadata.

        Args:
            name (str): Attribute name

        Returns:
            Any: The corresponding metadata attribute

        Raises:
            AttributeError: If neither this container nor its metadata provides the attribute

        """
        # Underscored names must not forward: DataClass.__setattr__ probes `_initialized`
        # during construction, before `metadata` has been assigned.
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        metadata = self.__dict__.get("metadata")
        if metadata is not None and hasattr(metadata, name):
            return getattr(metadata, name)
        return super().__getattr__(name)

    @property
    def num_qubits(self) -> int:
        """Total number of qubits to be allocated in QPE or other callers.

        System register: 2N spin-orbitals.
        Ancilla: x_o register + inner register (b + free-rider) + 2 spin qubits.
        This doesn't equal the total qubits of SOSSA since the SOSSA circuit allocate
        and free ancillary qubits internally.
        """
        meta = self.metadata
        num_system = 2 * meta.num_spatial_orbitals
        reg_bits = sossa_register_bits(meta.num_spatial_orbitals, meta.num_ranks, meta.num_bases, meta.num_copies)
        num_outer = reg_bits["xo_bits"]
        # Inner register: logical b bits + free-rider bits; spin register: 2 (spinDQ, spinSF).
        num_inner = reg_bits["b_bits"] + reg_bits["num_free_rider_bits"]
        num_ancilla = num_outer + num_inner + 2
        return num_system + num_ancilla

    @property
    def type(self) -> str:
        """Get the type of the unitary container."""
        return "sossa_walk"

    def to_json(self) -> dict[str, Any]:
        """Save the SOSSAWalkContainer to a JSON-serializable dictionary."""
        data: dict[str, Any] = {
            "container_type": self.type,
            "power": self.power,
            "metadata": self.metadata.to_json(),
            "outer_prepare": self.outer_prepare.to_json(),
            "inner_prepare": self.inner_prepare.to_json(),
            "select": self.select.to_json(),
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the SOSSAWalkContainer to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["power"] = self.power
        group.attrs["metadata"] = json.dumps(self.metadata.to_json())
        _wavefunction_to_hdf5(self.outer_prepare, group.create_group("outer_prepare"))
        self.inner_prepare.to_hdf5(group.create_group("inner_prepare"))
        self.select.to_hdf5(group.create_group("select"))

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "SOSSAWalkContainer":
        """Create a SOSSAWalkContainer from a JSON dictionary."""
        cls._validate_json_version(cls._serialization_version, json_data)

        from qdk_chemistry.data import Wavefunction  # noqa: PLC0415

        outer_prepare = Wavefunction.from_json(json_data["outer_prepare"])
        inner_prepare = SOSSAInnerPrepare.from_json(json_data["inner_prepare"])
        select = SOSSASelect.from_json(json_data["select"])

        return cls(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            metadata=FactorizedHamiltonianMetadata.from_json(json_data["metadata"]),
            power=json_data.get("power", 1),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SOSSAWalkContainer":
        """Load a SOSSAWalkContainer from an HDF5 group."""
        outer_prepare = _wavefunction_from_hdf5(group["outer_prepare"])
        inner_prepare = SOSSAInnerPrepare.from_hdf5(group["inner_prepare"])
        select = SOSSASelect.from_hdf5(group["select"])
        return cls(
            outer_prepare=outer_prepare,
            inner_prepare=inner_prepare,
            select=select,
            metadata=FactorizedHamiltonianMetadata.from_json(json.loads(group.attrs["metadata"])),
            power=int(group.attrs["power"]),
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the SOSSA container."""
        n = self.metadata.num_spatial_orbitals
        r = self.metadata.num_ranks
        b = self.metadata.num_bases
        c = self.metadata.num_copies
        b_bits = sossa_register_bits(n, r, b, c)["b_bits"]
        return (
            f"SOSSA Container (DFTHC block encoding):\n"
            f"  Power: {self.power}\n"
            f"  Orbitals N={n}, Ranks R={r}, Bases B={b}, Copies C={c}\n"
            f"  Xo = N + R*C = {n + r * c}\n"
            f"  Normalization Lambda = {self.normalization:.6f}\n"
            f"  Outer PREPARE: {self.outer_prepare.get_orbitals().num_modes()} qubits\n"
            f"  Inner PREPARE: {b_bits} qubits, {b + 1} basis entries\n"
            f"  System: {2 * n} spin-orbitals\n"
        )

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_arg(h, self.to_json())

    def eigenvalue_from_phase(self, phase_fraction: float) -> float:
        r"""Recover a Hamiltonian eigenvalue from the SOSSA walk operator phase.

        For the SOSSA walk operator, QPE measures :math:`\varphi` such that:

        .. math::

            E = \Lambda (1 + \cos 2\pi\varphi) + E_{\text{SOS}}

        Args:
            phase_fraction: Measured phase fraction :math:`\varphi \in [0, 1)`.

        Returns:
            float: The corresponding Hamiltonian eigenvalue.

        """
        phi = phase_fraction % 1.0
        # Evaluated as the algebraically equal 2cos^2(pi*phi). The literal
        # 1 + cos(2*pi*phi) cancels catastrophically near phi = 1/2, which is exactly
        # where the amplified ground state sits: both terms approach 1 and -1, so the
        # difference loses every significant digit. The squared half-angle form keeps
        # full relative precision there because cos(pi*phi) is computed small and accurate.
        return float(2.0 * self.normalization * np.cos(np.pi * phi) ** 2 + self.energy_shift)

    def combine(self, other: "SOSSAWalkContainer") -> "SOSSAWalkContainer":  # type: ignore[override]
        """Not supported for SOSSA containers.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("SOSSAWalkContainer does not support combining containers.")
