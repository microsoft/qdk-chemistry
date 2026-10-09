"""QDK/Chemistry model Hamiltonian description base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import h5py

from qdk_chemistry._core.data import Hamiltonian, LatticeGeometry
from qdk_chemistry.data._hashing import _hash_float, _hash_str, _hash_uint
from qdk_chemistry.data.hamiltonian_description.base import HamiltonianDescription

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_operator import QubitOperator

__all__: list[str] = ["ModelHamiltonianDescription"]


class ModelHamiltonianDescription(HamiltonianDescription):
    """Abstract class for a model Hamiltonian defined by a lattice and named model parameters.

    A subclass implements :meth:`materialize` and :meth:`data_type_name`, and takes the lattice
    followed by each model parameter as a keyword argument.
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

    def __init__(self, lattice: LatticeGeometry, parameters: Mapping[str, float]) -> None:
        """Initialize a model Hamiltonian description.

        Args:
            lattice: The lattice the model is defined on.
            parameters: The model parameters by name.

        """
        self.lattice = lattice
        self.parameters: Mapping[str, float] = MappingProxyType(
            {str(name): float(value) for name, value in parameters.items()}
        )
        super().__init__()

    @abstractmethod
    def materialize(self) -> "Hamiltonian | QubitOperator":
        """Build the model Hamiltonian.

        Returns:
            Hamiltonian | QubitOperator: The fermionic Hamiltonian or the qubit operator of the model.

        """

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, self.get_data_type_name())
        _hash_str(h, self.lattice.content_hash())
        _hash_uint(h, len(self.parameters))
        for name in sorted(self.parameters):
            _hash_str(h, name)
            _hash_float(h, self.parameters[name])

    def get_summary(self) -> str:
        """Get summary of the model Hamiltonian description.

        Returns:
            str: Summary string describing the model, the lattice size and the model parameters.

        """
        parameters = ", ".join(f"{name}={value:g}" for name, value in self.parameters.items()) or "none"
        return f"{type(self).__name__}\n  Lattice sites: {self.lattice.num_sites}\n  Parameters: {parameters}"

    def to_json(self) -> dict[str, Any]:
        """Convert the model Hamiltonian description to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the model Hamiltonian description.

        """
        data = {"lattice": json.loads(self.lattice.to_json()), "parameters": dict(self.parameters)}
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the model Hamiltonian description to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to.

        """
        self._add_hdf5_version(group)
        group.attrs["lattice"] = self.lattice.to_json()
        parameters = group.create_group("parameters")
        for name, value in self.parameters.items():
            parameters.attrs[name] = value

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "ModelHamiltonianDescription":
        """Create a model Hamiltonian description from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data.

        Returns:
            ModelHamiltonianDescription

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        parameters: dict[str, Any] = json_data["parameters"]
        return cls(LatticeGeometry.from_json(json.dumps(json_data["lattice"])), **parameters)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ModelHamiltonianDescription":
        """Load a model Hamiltonian description from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from.

        Returns:
            ModelHamiltonianDescription

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        lattice_json = group.attrs["lattice"]
        if isinstance(lattice_json, bytes):
            lattice_json = lattice_json.decode("utf-8")
        parameters: dict[str, Any] = {name: float(value) for name, value in group["parameters"].attrs.items()}
        return cls(LatticeGeometry.from_json(lattice_json), **parameters)
