"""QDK/Chemistry model Hamiltonian description."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import h5py

from qdk_chemistry._core.data import Hamiltonian, LatticeGeometry, LatticeGraph
from qdk_chemistry.data._hashing import _hash_float, _hash_str, _hash_uint
from qdk_chemistry.data.base import DataClass

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_operator import QubitOperator

__all__: list[str] = ["ModelHamiltonianDescription"]


class ModelHamiltonianDescription(DataClass):
    """A lattice model Hamiltonian described by its model name, lattice and named model parameters.

    A unitary builder that evolves the model directly, rather than its qubit operator, reads the
    lattice and the parameters it needs. :meth:`materialize` builds the Hamiltonian itself.
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for a model Hamiltonian description.

        Returns:
            ``"model_hamiltonian_description"``.

        """
        return "model_hamiltonian_description"

    _serialization_version = "0.1.0"

    def __init__(self, model: str, lattice: LatticeGeometry, parameters: Mapping[str, float]) -> None:
        """Initialize a model Hamiltonian description.

        Args:
            model: The model name, for example ``"hubbard"``.
            lattice: The lattice the model is defined on.
            parameters: The model parameters by name, for example ``{"epsilon": 0.0, "t": 1.0, "U": 4.0}``.

        Raises:
            TypeError: If ``lattice`` is not a :class:`~qdk_chemistry.data.LatticeGeometry`.

        """
        if not isinstance(lattice, LatticeGeometry):
            raise TypeError(f"lattice must be a LatticeGeometry, got {type(lattice).__name__}.")
        self.model = str(model)
        self.lattice = lattice
        self.parameters = {str(name): float(value) for name, value in parameters.items()}
        super().__init__()

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "model_hamiltonian_description")
        _hash_str(h, self.model)
        _hash_str(h, self.lattice.content_hash())
        _hash_uint(h, len(self.parameters))
        for name in sorted(self.parameters):
            _hash_str(h, name)
            _hash_float(h, self.parameters[name])

    def get_summary(self) -> str:
        """Get a human-readable summary of the model Hamiltonian description.

        Returns:
            str: Summary of the model, the lattice size and the model parameters.

        """
        parameters = ", ".join(f"{name}={value:g}" for name, value in self.parameters.items()) or "none"
        return (
            f"Model Hamiltonian Description\n  Model: {self.model}\n"
            f"  Lattice sites: {self.lattice.num_sites}\n  Parameters: {parameters}"
        )

    def materialize(self) -> "Hamiltonian | QubitOperator":
        """Build the model Hamiltonian on the nearest-neighbor bonds of the lattice.

        Calls the ``create_<model>_hamiltonian`` function of :mod:`qdk_chemistry.utils.model_hamiltonians`
        with the parameters as its keyword arguments, for example ``epsilon``, ``t`` and ``U`` for ``"hubbard"``.

        Returns:
            Hamiltonian | QubitOperator: A :class:`~qdk_chemistry.data.Hamiltonian` for the fermionic models
            ``"huckel"``, ``"hubbard"`` and ``"ppp"``, or a :class:`~qdk_chemistry.data.QubitOperator` for the
            spin models ``"heisenberg"`` and ``"ising"``.

        Raises:
            ValueError: If no create function exists for the model.

        """
        from qdk_chemistry.utils.model_hamiltonians import (  # noqa: PLC0415
            create_heisenberg_hamiltonian,
            create_hubbard_hamiltonian,
            create_huckel_hamiltonian,
            create_ising_hamiltonian,
            create_ppp_hamiltonian,
        )

        factories: dict[str, Callable[..., Hamiltonian | QubitOperator]] = {
            "huckel": create_huckel_hamiltonian,
            "hubbard": create_hubbard_hamiltonian,
            "ppp": create_ppp_hamiltonian,
            "heisenberg": create_heisenberg_hamiltonian,
            "ising": create_ising_hamiltonian,
        }
        factory = factories.get(self.model.lower())
        if factory is None:
            raise ValueError(f"No Hamiltonian factory for model {self.model!r}; available: {', '.join(factories)}.")
        return factory(LatticeGraph.from_geometry(self.lattice), **self.parameters)

    def to_json(self) -> dict[str, Any]:
        """Convert the model Hamiltonian description to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: The model name, the lattice and the model parameters.

        """
        data = {
            "model": self.model,
            "lattice": json.loads(self.lattice.to_json()),
            "parameters": dict(self.parameters),
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the model Hamiltonian description to an HDF5 group.

        Args:
            group: HDF5 group or file to write the description to.

        """
        self._add_hdf5_version(group)
        group.attrs["model"] = self.model
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
        return cls(model=json_data["model"], lattice=lattice, parameters=json_data["parameters"])

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ModelHamiltonianDescription":
        """Load a model Hamiltonian description from an HDF5 group.

        Args:
            group: HDF5 group or file containing the description.

        Returns:
            ModelHamiltonianDescription: New instance reconstructed from HDF5 data.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        model, lattice_json = (
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in (group.attrs["model"], group.attrs["lattice"])
        )
        parameters = {name: float(value) for name, value in group["parameters"].attrs.items()}
        return cls(model=model, lattice=LatticeGeometry.from_json(lattice_json), parameters=parameters)
