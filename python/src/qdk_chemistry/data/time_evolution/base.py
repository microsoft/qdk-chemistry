"""QDK/Chemistry time evolution base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py

from qdk_chemistry.data.base import DataClass

from .containers.base import TimeEvolutionUnitaryContainer
from .containers.pauli_product_formula import PauliProductFormulaContainer

__all__: list[str] = []


class TimeEvolutionUnitary(DataClass):
    """Data class for a time evolution unitary.

    Attributes:
        container (TimeEvolutionUnitaryContainer): The container for representing the time evolution unitary.

    """

    # Class attribute for filename validation
    _data_type_name = "time_evolution_unitary"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(self, container: TimeEvolutionUnitaryContainer) -> None:
        """Initialize a TimeEvolutionUnitary."""
        self._container = container
        super().__init__()

    def get_container_type(self) -> str:
        """Get the type of the time evolution unitary container.

        Returns:
            The type of the time evolution unitary.

        """
        return self._container.type

    def get_container(self) -> TimeEvolutionUnitaryContainer:
        """Get the time evolution unitary container.

        Returns:
            The time evolution unitary container.

        """
        return self._container

    def get_num_qubits(self) -> int:
        """Get the number of qubits the time evolution unitary acts on.

        Returns:
            The number of qubits.

        """
        return self._container.num_qubits

    def to_json(self) -> dict[str, Any]:
        """Convert the TimeEvolutionUnitary to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the TimeEvolutionUnitary

        """
        return self._container.to_json()

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the TimeEvolutionUnitary to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to

        """
        self._container.to_hdf5(group)

    def get_summary(self) -> str:
        """Get summary of time evolution unitary.

        Returns:
            str: Summary string describing the TimeEvolutionUnitary's contents and properties

        """
        return self._container.get_summary()

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "TimeEvolutionUnitary":
        """Create TimeEvolutionUnitary from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            TimeEvolutionUnitary

        """
        if "container_type" not in json_data:
            raise ValueError("JSON data must contain 'container_type' field.")
        container_type = json_data["container_type"]

        if container_type == "pauli_product_formula":
            container = PauliProductFormulaContainer.from_json(json_data)
        else:
            raise ValueError(f"Unsupported container type: {container_type}")

        return cls(container=container)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "TimeEvolutionUnitary":
        """Load an instance from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from

        Returns:
            TimeEvolutionUnitary

        """
        container_type = group.attrs.get("container_type")
        if container_type == "pauli_product_formula":
            container = PauliProductFormulaContainer.from_hdf5(group)
        else:
            raise ValueError(f"Unsupported container type: {container_type}")
        return cls(container=container)
