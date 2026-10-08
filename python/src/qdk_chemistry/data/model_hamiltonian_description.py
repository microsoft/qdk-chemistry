"""QDK/Chemistry model Hamiltonian description."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from collections.abc import Mapping
from typing import Any

import h5py

from qdk_chemistry._core.data import LatticeGeometry
from qdk_chemistry.data._hashing import _hash_float, _hash_str, _hash_uint
from qdk_chemistry.data.base import DataClass

__all__: list[str] = ["ModelHamiltonianDescription"]


class ModelHamiltonianDescription(DataClass):
    """A lattice model Hamiltonian described by its lattice and named model parameters.

    A unitary builder that evolves the model directly, rather than its qubit operator, reads the
    lattice and the parameters it needs, for example ``t`` and ``u`` for the Fermi-Hubbard model.
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for a model Hamiltonian description.

        Returns:
            ``"model_hamiltonian_description"``.

        """
        return "model_hamiltonian_description"

    _serialization_version = "0.1.0"

    def __init__(self, lattice: LatticeGeometry, parameters: Mapping[str, float]) -> None:
        """Initialize a model Hamiltonian description.

        Args:
            lattice: The lattice the model is defined on.
            parameters: The model parameters by name, for example ``{"t": 1.0, "u": 4.0}``.

        Raises:
            TypeError: If ``lattice`` is not a :class:`~qdk_chemistry.data.LatticeGeometry`.

        """
        if not isinstance(lattice, LatticeGeometry):
            raise TypeError(f"lattice must be a LatticeGeometry, got {type(lattice).__name__}.")
        self.lattice = lattice
        self.parameters = {str(name): float(value) for name, value in parameters.items()}
        super().__init__()

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "model_hamiltonian_description")
        _hash_str(h, self.lattice.content_hash())
        _hash_uint(h, len(self.parameters))
        for name in sorted(self.parameters):
            _hash_str(h, name)
            _hash_float(h, self.parameters[name])

    def get_summary(self) -> str:
        """Get a human-readable summary of the model Hamiltonian description.

        Returns:
            str: Summary of the lattice size and the model parameters.

        """
        parameters = ", ".join(f"{name}={value:g}" for name, value in self.parameters.items()) or "none"
        return f"Model Hamiltonian Description\n  Lattice sites: {self.lattice.num_sites}\n  Parameters: {parameters}"

    def to_json(self) -> dict[str, Any]:
        """Convert the model Hamiltonian description to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: The lattice and the model parameters.

        """
        data = {"lattice": json.loads(self.lattice.to_json()), "parameters": dict(self.parameters)}
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the model Hamiltonian description to an HDF5 group.

        Args:
            group: HDF5 group or file to write the description to.

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
            json_data: Dictionary containing the serialized description.

        Returns:
            ModelHamiltonianDescription: New instance reconstructed from JSON data.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        lattice = LatticeGeometry.from_json(json.dumps(json_data["lattice"]))
        return cls(lattice=lattice, parameters=json_data["parameters"])

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ModelHamiltonianDescription":
        """Load a model Hamiltonian description from an HDF5 group.

        Args:
            group: HDF5 group or file containing the description.

        Returns:
            ModelHamiltonianDescription: New instance reconstructed from HDF5 data.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        lattice_json = group.attrs["lattice"]
        if isinstance(lattice_json, bytes):
            lattice_json = lattice_json.decode("utf-8")
        parameters = {name: float(value) for name, value in group["parameters"].attrs.items()}
        return cls(lattice=LatticeGeometry.from_json(lattice_json), parameters=parameters)
