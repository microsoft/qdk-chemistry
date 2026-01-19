"""QDK/Chemistry time evolution container base module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import abstractmethod
from typing import Any

import h5py

from qdk_chemistry.data.base import DataClass

__all__: list[str] = ["TimeEvolutionUnitaryContainer"]


class TimeEvolutionUnitaryContainer(DataClass):
    """Abstract class for a time evolution unitary container."""

    # Class attribute for filename validation
    _data_type_name = "time_evolution_unitary"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    @property
    @abstractmethod
    def type(self) -> str:
        """Get the type of the time evolution unitary container.

        Returns:
            The type of the time evolution unitary container.

        """

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Get the number of qubits the time evolution unitary acts on.

        Returns:
            The number of qubits.

        """

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Convert the TimeEvolutionUnitaryContainer to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the TimeEvolutionUnitaryContainer

        """

    @abstractmethod
    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the TimeEvolutionUnitaryContainer to an HDF5 group.

        Args:
            group: HDF5 group or file to write data to

        """

    @classmethod
    @abstractmethod
    def from_json(cls, json_data: dict[str, Any]) -> "TimeEvolutionUnitaryContainer":
        """Create TimeEvolutionUnitaryContainer from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            TimeEvolutionUnitaryContainer

        """

    @classmethod
    @abstractmethod
    def from_hdf5(cls, group: h5py.Group) -> "TimeEvolutionUnitaryContainer":
        """Load an instance from an HDF5 group.

        Args:
            group: HDF5 group or file to read data from

        Returns:
            TimeEvolutionUnitaryContainer

        """

    @abstractmethod
    def get_summary(self) -> str:
        """Get summary of time evolution unitary.

        Returns:
            str: Summary string describing the TimeEvolutionUnitaryContainer's contents and properties

        """
