"""Rotated-Pauli and sum-of-squares qubit operator containers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.qubit_operator_containers.base import QubitOperatorContainer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import h5py

    from qdk_chemistry.data.qubit_operator import QubitOperator

__all__ = [
    "FermionParity",
    "RotatedMode",
    "RotatedPauliContainer",
    "RotatedPauliTerm",
    "SOSContainer",
    "SOSGenerator",
    "SOSGeneratorKind",
    "SpinPolicy",
]


class SOSGeneratorKind(Enum):
    """Kind of fermionic sum-of-squares generator."""

    D1 = "d1"
    Q1 = "q1"
    SF = "sf"


class FermionParity(Enum):
    """Fermion parity of a sum-of-squares generator."""

    Even = "even"
    Odd = "odd"


class SpinPolicy(Enum):
    """Spin metadata carried by a sum-of-squares generator."""

    Specific = "specific"
    Summed = "summed"


@dataclass(frozen=True, eq=False)
class RotatedMode:
    """A normalized spatial orbital and its spin channel."""

    basis_vector: np.ndarray
    spin: int

    def __post_init__(self) -> None:
        """Validate and freeze the rotated mode."""
        vector = np.array(self.basis_vector, dtype=float, copy=True)
        if self.spin not in (0, 1):
            raise ValueError("rotated mode spin must be 0 or 1")
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("rotated mode basis_vector must be finite")
        if abs(float(np.linalg.norm(vector)) - 1.0) > 1e-10:
            raise ValueError("rotated mode basis_vector must be normalized")
        vector.flags.writeable = False
        object.__setattr__(self, "basis_vector", vector)


@dataclass(frozen=True, eq=False)
class RotatedPauliTerm:
    """A single term of a rotated-Pauli operator."""

    coefficient: complex
    center: Mapping[int, int]
    mode: RotatedMode | None = None

    def __post_init__(self) -> None:
        """Validate the term."""
        coefficient = complex(self.coefficient)
        center = dict(self.center)
        if not isfinite(coefficient.real) or not isfinite(coefficient.imag):
            raise ValueError("rotated-Pauli coefficient must be finite")
        if (len(center) == 0) == (self.mode is not None):
            raise ValueError("non-identity Pauli centers require a rotated mode")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "center", center)


class RotatedPauliContainer(QubitOperatorContainer):
    """Container for a linear combination of rotated Pauli terms."""

    def __init__(
        self,
        terms: Sequence[RotatedPauliTerm],
        num_qubits: int,
        encoding: str | None,
        fermion_mode_order: str | None,
    ) -> None:
        """Initialize a rotated-Pauli container."""
        self.terms = tuple(terms)
        self._num_qubits = num_qubits
        if not self.terms or num_qubits <= 0:
            raise ValueError("a rotated-Pauli container requires terms and positive num_qubits")
        super().__init__(encoding, fermion_mode_order)

    @property
    def type(self) -> str:
        """Return the container type."""
        return "rotated_pauli"

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self._num_qubits

    @property
    def coefficient_one_norm(self) -> float:
        """Return the coefficient one-norm."""
        return float(sum(abs(term.coefficient) for term in self.terms))

    @property
    def lcu_normalization(self) -> float:
        """Return the LCU normalization."""
        return self.coefficient_one_norm

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.type)
        _hash_arg(h, self.to_json())

    def to_json(self) -> dict[str, Any]:
        """Convert the container to a JSON dictionary."""
        terms = []
        for term in self.terms:
            mode = None
            if term.mode is not None:
                mode = {"basis_vector": term.mode.basis_vector.tolist(), "spin": term.mode.spin}
            terms.append(
                {
                    "coefficient": {"real": term.coefficient.real, "imag": term.coefficient.imag},
                    "center": {str(key): value for key, value in term.center.items()},
                    "mode": mode,
                }
            )
        return self._add_json_version(
            {
                "container_type": self.type,
                "num_qubits": self.num_qubits,
                "encoding": self.encoding,
                "fermion_mode_order": str(self.fermion_mode_order) if self.fermion_mode_order is not None else None,
                "terms": terms,
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
        terms = []
        for item in json_data["terms"]:
            mode_data = item["mode"]
            mode = RotatedMode(np.asarray(mode_data["basis_vector"]), mode_data["spin"]) if mode_data else None
            coefficient = complex(item["coefficient"]["real"], item["coefficient"]["imag"])
            center = {int(key): value for key, value in item["center"].items()}
            terms.append(RotatedPauliTerm(coefficient, center, mode))
        return cls(terms, json_data["num_qubits"], json_data.get("encoding"), json_data.get("fermion_mode_order"))

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RotatedPauliContainer:
        """Create a rotated-Pauli container from HDF5."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls.from_json(json.loads(group.attrs["payload"]))

    def get_summary(self) -> str:
        """Return a summary of the rotated-Pauli container."""
        return f"Rotated Pauli Operator\n  Number of qubits: {self.num_qubits}\n  Number of terms: {len(self.terms)}\n"


@dataclass(frozen=True, eq=False)
class SOSGenerator:
    """One generator in a sum-of-squares representation."""

    kind: SOSGeneratorKind
    parity: FermionParity
    spin_policy: SpinPolicy
    operator: QubitOperator
    spin: int | None = None
    source_index: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate generator metadata and nested operator type."""
        from qdk_chemistry.data.qubit_operator import QubitOperator  # noqa: PLC0415

        if not isinstance(self.operator, QubitOperator):
            raise TypeError("SOSGenerator requires a QubitOperator")
        object.__setattr__(self, "source_index", tuple(self.source_index))
        is_sf = self.kind == SOSGeneratorKind.SF
        if is_sf != (self.parity == FermionParity.Even) or is_sf != (self.spin_policy == SpinPolicy.Summed):
            raise ValueError("generator kind, parity, and spin policy disagree")
        if (self.spin_policy == SpinPolicy.Specific) != (self.spin is not None):
            raise ValueError("invalid generator spin metadata")
        if self.spin is not None and self.spin not in (0, 1):
            raise ValueError("invalid generator spin metadata")

    @property
    def lcu_normalization(self) -> float:
        """Return the nested operator LCU normalization."""
        container = self.operator.get_container()
        if not isinstance(container, RotatedPauliContainer):
            from qdk_chemistry.data.qubit_operator import PauliLCUContainer  # noqa: PLC0415

            if isinstance(container, PauliLCUContainer):
                return container.schatten_norm
            raise TypeError("SOS generators require rotated-Pauli or Pauli-LCU containers")
        return container.lcu_normalization


class SOSContainer(QubitOperatorContainer):
    """Container for a sum-of-squares qubit operator."""

    def __init__(
        self,
        num_spatial_orbitals: int,
        num_qubits: int,
        energy_shift: float,
        generators: Sequence[SOSGenerator],
        encoding: str | None,
        fermion_mode_order: str | None,
    ) -> None:
        """Initialize a sum-of-squares container."""
        self.num_spatial_orbitals = num_spatial_orbitals
        self._num_qubits = num_qubits
        self.energy_shift = float(energy_shift)
        self.generators = tuple(generators)
        if num_spatial_orbitals <= 0 or num_qubits <= 0 or not self.generators or not isfinite(self.energy_shift):
            raise ValueError("invalid sum-of-squares container")
        if any(generator.operator.num_qubits != num_qubits for generator in self.generators):
            raise ValueError("generator dimensions do not match the SOS container")
        super().__init__(encoding, fermion_mode_order)

    @property
    def type(self) -> str:
        """Return the container type."""
        return "sos"

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits."""
        return self._num_qubits

    @property
    def square_root_normalization(self) -> float:
        """Return the square-root normalization."""
        return float(sqrt(sum(generator.lcu_normalization**2 for generator in self.generators)))

    @property
    def normalization(self) -> float:
        """Return the SOS block-encoding normalization."""
        return 0.5 * self.square_root_normalization**2

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
                "num_qubits": self.num_qubits,
                "energy_shift": self.energy_shift,
                "encoding": self.encoding,
                "fermion_mode_order": str(self.fermion_mode_order) if self.fermion_mode_order is not None else None,
                "generators": [
                    {
                        "kind": generator.kind.value,
                        "parity": generator.parity.value,
                        "spin_policy": generator.spin_policy.value,
                        "operator": generator.operator.to_json(),
                        "spin": generator.spin,
                        "source_index": list(generator.source_index),
                    }
                    for generator in self.generators
                ],
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
        from qdk_chemistry.data.qubit_operator import QubitOperator  # noqa: PLC0415

        cls._validate_json_version(cls._serialization_version, json_data)
        generators = tuple(
            SOSGenerator(
                SOSGeneratorKind(item["kind"]),
                FermionParity(item["parity"]),
                SpinPolicy(item["spin_policy"]),
                QubitOperator.from_json(item["operator"]),
                item.get("spin"),
                tuple(item.get("source_index", ())),
            )
            for item in json_data["generators"]
        )
        return cls(
            json_data["num_spatial_orbitals"],
            json_data["num_qubits"],
            json_data["energy_shift"],
            generators,
            json_data.get("encoding"),
            json_data.get("fermion_mode_order"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> SOSContainer:
        """Create a sum-of-squares container from HDF5."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls.from_json(json.loads(group.attrs["payload"]))

    def get_summary(self) -> str:
        """Return a summary of the sum-of-squares container."""
        return (
            f"SOS Qubit Operator\n  Number of qubits: {self.num_qubits}\n"
            f"  Number of generators: {len(self.generators)}\n"
        )
