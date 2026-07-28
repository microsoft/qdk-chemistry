"""QDK/Chemistry FOQCS-LCU block encoding container module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any

import h5py

from qdk_chemistry.data._hashing import _hash_float, _hash_int, _hash_str

from .block_encoding import BlockEncodingContainer

__all__ = ["FoqcsContainer", "FoqcsFamily"]


@dataclass(frozen=True)
class FoqcsFamily:
    """A homogeneous family of Pauli terms in a FOQCS-LCU block encoding.

    A family groups translationally-equivalent terms that share a single Pauli
    pattern (a field direction for 1-body terms or a homogeneous coupling for
    2-body terms).  All terms in a family carry the same Hamiltonian coefficient
    and are prepared by a single balanced Dicke state.

    Attributes:
        paulis: The Pauli pattern, e.g. ``("X",)`` for a field or ``("Z", "Z")`` for a coupling.
        offset: Nearest-neighbour separation ``k`` for 2-body families (``0`` for 1-body).
        abs_coeff: The normalized sub-PREP amplitude magnitude for this family.
        phase: The sub-PREP phase correction (radians) for this family.

    """

    paulis: tuple[str, ...]
    offset: int
    abs_coeff: float
    phase: float


class FoqcsContainer(BlockEncodingContainer):
    r"""Container for a FOQCS-LCU block encoding of a spin-model Hamiltonian.

    FOQCS-LCU (Fast One-Qubit Control Select – Linear Combination of Unitaries)
    block-encodes a translationally-structured spin Hamiltonian by grouping its
    terms into homogeneous :class:`FoqcsFamily` blocks.  Each family is loaded
    with a balanced Dicke state and selected transversally, giving the block
    encoding

    .. math::

        B[H] = \mathrm{PREP}(c^*)^\dagger \cdot \mathrm{SELECT} \cdot \mathrm{PREP}(c)

    which encodes :math:`H / \lambda` in the
    :math:`\langle 0|_\mathrm{anc} B |0\rangle_\mathrm{anc}` subspace.  The
    ancilla register is laid out as ``[subPrepReg | xReg | zReg]`` with
    ``subPrepReg`` of length ``num_families`` and ``xReg`` / ``zReg`` each of
    length ``num_sites``.

    References:
        F. Della Chiara, M. Nibbi, Y. Shen, D. Camps, R. Van Beeumen,
        "Efficient LCU block encodings through Dicke states preparation",
        2025, arXiv:2507.20887.

    """

    # Class attribute for filename validation
    _data_type_name = "foqcs_container"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self,
        num_sites: int,
        families: list[FoqcsFamily],
        lambda_: float,
        power: int = 1,
    ) -> None:
        r"""Initialize a FoqcsContainer.

        Args:
            num_sites: Number of spin sites (system qubits) ``L``.
            families: The homogeneous Pauli-term families defining the block encoding.
            lambda_: The 1-norm :math:`\lambda` used for eigenvalue-phase conversion.
            power: Number of times to apply the block encoding (for :math:`B[H]^\mathrm{power}`).

        """
        self._num_sites = num_sites
        self._families = list(families)
        self._lambda = lambda_
        self._power = power
        super().__init__()

    @property
    def power(self) -> int:
        """Number of times to apply the block encoding.

        Returns:
            int: The power value.

        """
        return self._power

    @property
    def num_sites(self) -> int:
        """Number of spin sites (system qubits).

        Returns:
            int: The site count ``L``.

        """
        return self._num_sites

    @property
    def families(self) -> list[FoqcsFamily]:
        """The homogeneous Pauli-term families.

        Returns:
            list[FoqcsFamily]: The families defining the block encoding.

        """
        return self._families

    @property
    def num_families(self) -> int:
        """Number of Pauli-term families (sub-PREP qubits).

        Returns:
            int: The family count.

        """
        return len(self._families)

    @property
    def lambda_(self) -> float:
        r"""The 1-norm :math:`\lambda` of the block-encoded Hamiltonian.

        Returns:
            float: The normalization factor.

        """
        return self._lambda

    @property
    def num_target_qubits(self) -> int:
        """Number of target (system) qubits.

        Returns:
            int: Equal to ``num_sites``.

        """
        return self._num_sites

    @property
    def num_prepare_ancillas(self) -> int:
        """Number of ancilla qubits in the PREPARE register.

        The layout is ``[subPrepReg | xReg | zReg]``: one qubit per family plus
        two site-length registers.

        Returns:
            int: ``num_families + 2 * num_sites``.

        """
        return self.num_families + 2 * self._num_sites

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (system + ancilla).

        Returns:
            int: The combined qubit count.

        """
        return self.num_target_qubits + self.num_prepare_ancillas

    @property
    def type(self) -> str:
        """Get the type of the unitary container.

        Returns:
            str: The type string ``"foqcs"``.

        """
        return "foqcs"

    def to_json(self) -> dict[str, Any]:
        """Save the FoqcsContainer to a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Dictionary representation including container type, power,
                lattice size, normalization, and per-family data.

        """
        data: dict[str, Any] = {
            "container_type": self.type,
            "power": self.power,
            "num_sites": self._num_sites,
            "lambda": self._lambda,
            "families": [
                {
                    "paulis": list(f.paulis),
                    "offset": f.offset,
                    "abs_coeff": f.abs_coeff,
                    "phase": f.phase,
                }
                for f in self._families
            ],
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the FoqcsContainer to an HDF5 group.

        Args:
            group: HDF5 group to write container data to.

        """
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["power"] = self.power
        group.attrs["num_sites"] = self._num_sites
        group.attrs["lambda"] = self._lambda
        families_group = group.create_group("families")
        for i, f in enumerate(self._families):
            fam_group = families_group.create_group(f"family_{i}")
            fam_group.attrs["paulis"] = list(f.paulis)
            fam_group.attrs["offset"] = f.offset
            fam_group.attrs["abs_coeff"] = f.abs_coeff
            fam_group.attrs["phase"] = f.phase

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "FoqcsContainer":
        """Create a FoqcsContainer from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized FOQCS data.

        Returns:
            FoqcsContainer: The deserialized instance.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        families = [
            FoqcsFamily(
                paulis=tuple(f["paulis"]),
                offset=int(f["offset"]),
                abs_coeff=float(f["abs_coeff"]),
                phase=float(f["phase"]),
            )
            for f in json_data["families"]
        ]
        return cls(
            num_sites=int(json_data["num_sites"]),
            families=families,
            lambda_=float(json_data["lambda"]),
            power=int(json_data.get("power", 1)),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "FoqcsContainer":
        """Load a FoqcsContainer from an HDF5 group.

        Args:
            group: HDF5 group to read container data from.

        Returns:
            FoqcsContainer: The deserialized instance.

        """
        families_group = group["families"]
        families = []
        for key in sorted(families_group.keys(), key=lambda k: int(k.split("_")[1])):
            fam_group = families_group[key]
            families.append(
                FoqcsFamily(
                    paulis=tuple(str(p) for p in fam_group.attrs["paulis"]),
                    offset=int(fam_group.attrs["offset"]),
                    abs_coeff=float(fam_group.attrs["abs_coeff"]),
                    phase=float(fam_group.attrs["phase"]),
                )
            )
        return cls(
            num_sites=int(group.attrs["num_sites"]),
            families=families,
            lambda_=float(group.attrs["lambda"]),
            power=int(group.attrs["power"]),
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the FOQCS container.

        Returns:
            str: Multi-line summary describing the block encoding.

        """
        family_lines = "\n".join(
            f"    {''.join(f.paulis)} (offset {f.offset}): |c|={f.abs_coeff:.4f}, phase={f.phase:.4f}"
            for f in self._families
        )
        return (
            f"FOQCS Container:\n"
            f"  Power: {self.power}\n"
            f"  Sites: {self._num_sites}, ancilla: {self.num_prepare_ancillas} qubits\n"
            f"  Lambda: {self._lambda:.6f}\n"
            f"  Families ({self.num_families}):\n{family_lines}"
        )

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "foqcs_container")
        _hash_int(h, self._power)
        _hash_int(h, self._num_sites)
        _hash_float(h, self._lambda)
        for f in self._families:
            _hash_str(h, "".join(f.paulis))
            _hash_int(h, f.offset)
            _hash_float(h, f.abs_coeff)
            _hash_float(h, f.phase)

    def eigenvalue_from_phase(self, phase_fraction: float) -> float:
        """Not applicable for a raw block encoding.

        A plain FOQCS block encoding does not define an eigenvalue-phase
        relationship on its own.  Wrap it in an
        :class:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.LCUWalkContainer`
        for QPE with qubitization.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError(
            "FoqcsContainer does not define an eigenvalue-phase relationship. "
            "Wrap it in an LCUWalkContainer to use QPE."
        )

    def combine(self, other: "BlockEncodingContainer") -> "BlockEncodingContainer":
        """Combining FOQCS block encodings is not supported.

        Args:
            other: The container to append after this one.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("FoqcsContainer does not support combination.")
