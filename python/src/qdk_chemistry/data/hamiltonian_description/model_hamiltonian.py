"""QDK/Chemistry model Hamiltonian description base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypeAlias

import h5py
import numpy as np

from qdk_chemistry._core.data import Hamiltonian, LatticeGeometry
from qdk_chemistry.data._hashing import _hash_array, _hash_float, _hash_int, _hash_str, _hash_uint
from qdk_chemistry.data.hamiltonian_description.base import HamiltonianDescription

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_operator import QubitOperator

__all__: list[str] = ["ModelHamiltonianDescription"]

ModelParameter: TypeAlias = float | np.ndarray | Mapping[int, float | np.ndarray]
"""A model parameter: a scalar, a per-site or per-edge array, or a value for each neighbor shell."""


class ModelHamiltonianDescription(HamiltonianDescription):
    """Abstract class for a model Hamiltonian defined by a lattice and named model parameters.

    A parameter is a float, a per-site or per-edge array, or a mapping from neighbor shell to either.
    A subclass implements :attr:`kind` and :meth:`materialize`, takes the lattice followed by each
    model parameter as a keyword argument, and is added to :meth:`from_json`.
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for model Hamiltonian descriptions.

        Returns:
            ``"model_hamiltonian_description"``.

        """
        return "model_hamiltonian_description"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(self, lattice: LatticeGeometry, parameters: Mapping[str, ModelParameter]) -> None:
        """Initialize a model Hamiltonian description.

        Args:
            lattice: The lattice the model is defined on.
            parameters: The model parameters by name.

        """
        self.lattice = lattice
        self.parameters: Mapping[str, ModelParameter] = MappingProxyType(
            {str(name): _freeze_parameter(value) for name, value in parameters.items()}
        )
        super().__init__()

    @property
    def kind(self) -> str:
        """Return the model this description holds, as recorded in its serialized form."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement kind")

    def materialize(self) -> "Hamiltonian | QubitOperator":
        """Build the model Hamiltonian.

        Returns:
            Hamiltonian | QubitOperator: The fermionic Hamiltonian or the qubit operator of the model.

        Raises:
            NotImplementedError: If the subclass does not implement it.

        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement materialize()")

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.kind)
        _hash_str(h, self.lattice.content_hash())
        _hash_uint(h, len(self.parameters))
        for name in sorted(self.parameters):
            _hash_str(h, name)
            _hash_parameter(h, self.parameters[name])

    def get_summary(self) -> str:
        """Get summary of the model Hamiltonian description.

        Returns:
            str: Summary string describing the model, the lattice size and the model parameters.

        """
        parameters = (
            ", ".join(f"{name}={_describe_parameter(value)}" for name, value in self.parameters.items()) or "none"
        )
        return f"{type(self).__name__}\n  Lattice sites: {self.lattice.num_sites}\n  Parameters: {parameters}"

    def to_json(self) -> dict[str, Any]:
        """Convert the model Hamiltonian description to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the model Hamiltonian description.

        """
        data = {
            "kind": self.kind,
            "lattice": json.loads(self.lattice.to_json()),
            "parameters": {name: _encode_parameter(value) for name, value in self.parameters.items()},
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the model Hamiltonian description to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to.

        """
        self._add_hdf5_version(group)
        group.attrs["model_hamiltonian_description"] = json.dumps(self.to_json())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "ModelHamiltonianDescription":
        """Create the model Hamiltonian description of the recorded kind from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data.

        Returns:
            ModelHamiltonianDescription

        Raises:
            ValueError: If ``json_data["kind"]`` is not a known model.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        kind = json_data.get("kind")
        lattice = LatticeGeometry.from_json(json.dumps(json_data["lattice"]))
        parameters = {name: _decode_parameter(value) for name, value in json_data["parameters"].items()}
        if kind == "fermi_hubbard":
            from qdk_chemistry.data.hamiltonian_description.fermi_hubbard import (  # noqa: PLC0415
                FermiHubbardModelHamiltonianDescription,
            )

            return FermiHubbardModelHamiltonianDescription(lattice, **parameters)
        raise ValueError(f"Unknown ModelHamiltonianDescription kind: {kind!r}. Expected 'fermi_hubbard'.")

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ModelHamiltonianDescription":
        """Load a model Hamiltonian description from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from.

        Returns:
            ModelHamiltonianDescription

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        raw = group.attrs["model_hamiltonian_description"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls.from_json(json.loads(raw))


def _freeze_value(value: float | np.ndarray) -> float | np.ndarray:
    """Return a scalar as a float, or a read-only float copy of an array."""
    array = np.array(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    array.setflags(write=False)
    return array


def _freeze_parameter(value: ModelParameter) -> ModelParameter:
    """Return an immutable copy of a parameter, with any shell mapping keyed and sorted by integer shell."""
    if isinstance(value, Mapping):
        shells = {int(shell): _freeze_value(shell_value) for shell, shell_value in value.items()}
        return MappingProxyType(dict(sorted(shells.items())))
    return _freeze_value(value)


def _hash_parameter(h, value: ModelParameter) -> None:
    """Feed a parameter into the hasher, tagged with its kind."""
    if isinstance(value, Mapping):
        _hash_str(h, "shells")
        _hash_uint(h, len(value))
        for shell, shell_value in value.items():
            _hash_int(h, shell)
            _hash_parameter(h, shell_value)
    elif isinstance(value, np.ndarray):
        _hash_str(h, "array")
        _hash_array(h, value)
    else:
        _hash_str(h, "float")
        _hash_float(h, value)


def _describe_parameter(value: ModelParameter) -> str:
    """Describe a parameter by its value, its array shape or its shells."""
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{shell}: {_describe_parameter(v)}" for shell, v in value.items()) + "}"
    if isinstance(value, np.ndarray):
        return f"array{value.shape}"
    return f"{value:g}"


def _encode_parameter(value: ModelParameter) -> Any:
    """Encode a parameter as a JSON number, nested list, or object keyed by shell."""
    if isinstance(value, Mapping):
        return {str(shell): _encode_parameter(v) for shell, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _decode_parameter(value: Any) -> ModelParameter:
    """Decode a parameter written by :func:`_encode_parameter`."""
    if isinstance(value, dict):
        return {int(shell): np.asarray(v, dtype=float) if isinstance(v, list) else v for shell, v in value.items()}
    if isinstance(value, list):
        return np.asarray(value, dtype=float)
    return value
