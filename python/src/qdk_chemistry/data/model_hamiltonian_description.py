"""QDK/Chemistry model Hamiltonian descriptions."""

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

from qdk_chemistry._core.data import Hamiltonian, LatticeGeometry, LatticeGraph
from qdk_chemistry.data._hashing import _hash_float, _hash_str, _hash_uint
from qdk_chemistry.data.base import DataClass

if TYPE_CHECKING:
    from qdk_chemistry.data.qubit_operator import QubitOperator

__all__: list[str] = ["FermiHubbardModelHamiltonianDescription", "ModelHamiltonianDescription"]


class ModelHamiltonianDescription(DataClass):
    """Abstract base of the lattice model Hamiltonians described by a lattice and named model parameters.

    A unitary builder that evolves the model directly, rather than its qubit operator, reads the
    lattice and the parameters it needs. :meth:`materialize` builds the Hamiltonian itself.

    A subclass implements :meth:`materialize`, declares its own :meth:`data_type_name`, and takes
    the lattice followed by each model parameter as a keyword argument, which :meth:`from_json`
    and :meth:`from_hdf5` rely on.
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for model Hamiltonian descriptions.

        Returns:
            ``"model_hamiltonian_description"``.

        """
        return "model_hamiltonian_description"

    _serialization_version = "0.1.0"

    def __init__(self, lattice: LatticeGeometry, parameters: Mapping[str, float]) -> None:
        """Initialize a model Hamiltonian description.

        Args:
            lattice: The lattice the model is defined on.
            parameters: The model parameters by name.

        Raises:
            TypeError: If the class does not implement :meth:`materialize`, or ``lattice`` is not a
                :class:`~qdk_chemistry.data.LatticeGeometry`.

        """
        if getattr(type(self).materialize, "__isabstractmethod__", False):
            raise TypeError(f"Can't instantiate abstract class {type(self).__name__} without materialize().")
        if not isinstance(lattice, LatticeGeometry):
            raise TypeError(f"lattice must be a LatticeGeometry, got {type(lattice).__name__}.")
        self.lattice = lattice
        self.parameters: Mapping[str, float] = MappingProxyType(
            {str(name): float(value) for name, value in parameters.items()}
        )
        super().__init__()

    @abstractmethod
    def materialize(self) -> "Hamiltonian | QubitOperator":
        """Build the model Hamiltonian.

        Returns:
            Hamiltonian | QubitOperator: A :class:`~qdk_chemistry.data.Hamiltonian` for a fermionic model,
            or a :class:`~qdk_chemistry.data.QubitOperator` for a spin model.

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
        """Get a human-readable summary of the model Hamiltonian description.

        Returns:
            str: Summary of the model, the lattice size and the model parameters.

        """
        parameters = ", ".join(f"{name}={value:g}" for name, value in self.parameters.items()) or "none"
        return f"{type(self).__name__}\n  Lattice sites: {self.lattice.num_sites}\n  Parameters: {parameters}"

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
        parameters: dict[str, Any] = json_data["parameters"]
        return cls(LatticeGeometry.from_json(json.dumps(json_data["lattice"])), **parameters)

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
        parameters: dict[str, Any] = {name: float(value) for name, value in group["parameters"].attrs.items()}
        return cls(LatticeGeometry.from_json(lattice_json), **parameters)


class FermiHubbardModelHamiltonianDescription(ModelHamiltonianDescription):
    r"""The Fermi-Hubbard model on the nearest-neighbor bonds of a lattice.

    .. math::

        H = \sum_{i,\sigma} \epsilon\, n_{i\sigma}
          - t \sum_{\langle i,j \rangle, \sigma} (a^\dagger_{i\sigma} a_{j\sigma} + \text{h.c.})
          + U \sum_i n_{i\uparrow} n_{i\downarrow}
    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for a Fermi-Hubbard model description.

        Returns:
            ``"fermi_hubbard_model_hamiltonian_description"``.

        """
        return "fermi_hubbard_model_hamiltonian_description"

    def __init__(self, lattice: LatticeGeometry, t: float, u: float, epsilon: float = 0.0) -> None:
        """Initialize a Fermi-Hubbard model description.

        Args:
            lattice: The lattice the model is defined on.
            t: The nearest-neighbor hopping integral.
            u: The on-site Coulomb repulsion.
            epsilon: The on-site orbital energy.

        """
        super().__init__(lattice, {"t": t, "u": u, "epsilon": epsilon})

    def materialize(self) -> Hamiltonian:
        """Build the Fermi-Hubbard Hamiltonian with ``create_hubbard_hamiltonian``.

        Returns:
            Hamiltonian: The Fermi-Hubbard Hamiltonian on the nearest-neighbor bonds of the lattice.

        """
        from qdk_chemistry.utils.model_hamiltonians import create_hubbard_hamiltonian  # noqa: PLC0415

        return create_hubbard_hamiltonian(
            LatticeGraph.from_geometry(self.lattice),
            epsilon=self.parameters["epsilon"],
            t=self.parameters["t"],
            U=self.parameters["u"],
        )
