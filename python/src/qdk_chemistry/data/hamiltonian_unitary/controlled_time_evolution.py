"""QDK/Chemistry controlled time evolution module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py

from qdk_chemistry.data.base import DataClass

from .base import TimeEvolutionUnitary

__all__: list[str] = ["ControlledTimeEvolutionUnitary"]


class ControlledTimeEvolutionUnitary(DataClass):
    """Data class for a controlled time evolution unitary."""

    # Class attribute for filename validation
    _data_type_name = "controlled_time_evolution_unitary"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(self, time_evolution_unitary: TimeEvolutionUnitary, control_indices: list[int]):
        """Initialize a ControlledTimeEvolutionUnitary.

        Args:
            time_evolution_unitary: The time evolution unitary to be controlled.
            control_indices: The control qubit indices.

        """
        self.time_evolution_unitary = time_evolution_unitary
        self.control_indices = control_indices
        super().__init__()

    def get_unitary_container_type(self) -> str:
        """Get the type of the time evolution unitary container.

        Returns:
            The type of the time evolution unitary container.

        """
        return self.time_evolution_unitary.get_container_type()

    def get_num_total_qubits(self) -> int:
        """Get the total number of qubits including control qubits.

        Returns:
            The total number of qubits (time evolution qubits + control qubits).

        """
        return self.time_evolution_unitary.get_num_qubits() + len(self.control_indices)

    def to_json(self) -> dict[str, Any]:
        """Convert the ControlledTimeEvolutionUnitary to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the ControlledTimeEvolutionUnitary

        """
        data: dict[str, Any] = {}
        data["time_evolution_unitary"] = self.time_evolution_unitary.to_json()
        data["control_indices"] = self.control_indices
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the ControlledTimeEvolutionUnitary to an HDF5 group.

        Args:
            group: HDF5 group or file to write the controlled time evolution unitary to

        """
        self._add_hdf5_version(group)

        # Write simple attributes
        group.attrs["control_indices"] = self.control_indices

        # Create subgroup for the nested object
        teu_group = group.create_group("time_evolution_unitary")
        self.time_evolution_unitary.to_hdf5(teu_group)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "ControlledTimeEvolutionUnitary":
        """Create ControlledTimeEvolutionUnitary from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            ControlledTimeEvolutionUnitary

        """
        time_evolution_unitary = TimeEvolutionUnitary.from_json(json_data["time_evolution_unitary"])
        control_indices = json_data["control_indices"]
        return cls(
            time_evolution_unitary=time_evolution_unitary,
            control_indices=control_indices,
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ControlledTimeEvolutionUnitary":
        """Load a ControlledTimeEvolutionUnitary from an HDF5 group.

        Args:
            group: The HDF5 group containing the serialized object.

        Returns:
            ControlledTimeEvolutionUnitary

        """
        # Verify version
        cls._validate_hdf5_version(cls._serialization_version, group)

        # Load simple attributes
        control_indices = list(group.attrs["control_indices"])

        # Load nested TimeEvolutionUnitary
        teu_group = group["time_evolution_unitary"]
        time_evolution_unitary = TimeEvolutionUnitary.from_hdf5(teu_group)

        return cls(
            time_evolution_unitary=time_evolution_unitary,
            control_indices=control_indices,
        )

    def get_summary(self) -> str:
        """Get summary of controlled time evolution unitary.

        Returns:
            str: Summary string describing the ControlledTimeEvolutionUnitary's contents and properties

        """
        line = "Controlled Time Evolution Unitary:\n"
        line += f"  Control Indices: {self.control_indices}\n"
        line += "  Time Evolution Unitary Summary:\n"
        teu_summary = self.time_evolution_unitary.get_summary()
        teu_summary_indented = "\n".join("    " + summary_line for summary_line in teu_summary.splitlines())
        line += teu_summary_indented
        return line
