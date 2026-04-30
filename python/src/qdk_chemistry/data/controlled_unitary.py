"""QDK/Chemistry controlled unitary module."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py

from qdk_chemistry.data.base import DataClass

from .unitary_representation.base import UnitaryRepresentation

__all__: list[str] = ["ControlledUnitary"]


class ControlledUnitary(DataClass):
    """Data class for a controlled unitary."""

    # Class attribute for filename validation
    _data_type_name = "controlled_unitary"

    # Serialization version for this class
    _serialization_version = "0.1.0"

    def __init__(
        self, unitary: UnitaryRepresentation, control_indices: list[int], target_indices: list[int] | None = None
    ):
        """Initialize a ControlledUnitary.

        Args:
            unitary: The unitary to be controlled.
            control_indices: The control qubit indices.
            target_indices: The target qubit indices. If None, defaults to all qubit indices
                in range(total_qubits) that are not in control_indices.

        """
        self.unitary = unitary
        self.control_indices = control_indices
        if target_indices is not None:
            target_indices_set = set(target_indices)
            control_indices_set = set(control_indices)
            if target_indices_set & control_indices_set:
                raise ValueError("target_indices and control_indices must not overlap.")
        self._target_indices = target_indices
        super().__init__()

    @property
    def target_indices(self) -> list[int]:
        """Get the target qubit indices.

        Returns:
            The target qubit indices. If not explicitly set, returns all qubit indices
            excluding the control indices.

        """
        if self._target_indices is not None:
            return self._target_indices
        total_qubits = self.get_num_total_qubits()
        return [i for i in range(total_qubits) if i not in self.control_indices]

    def get_unitary_container_type(self) -> str:
        """Get the type of the unitary container.

        Returns:
            The type of the unitary container.

        """
        return self.unitary.get_container_type()

    def get_num_total_qubits(self) -> int:
        """Get the total number of qubits including control qubits.

        Returns:
            The total number of qubits (unitary qubits + control qubits).

        """
        return self.unitary.get_num_qubits() + len(self.control_indices)

    def to_json(self) -> dict[str, Any]:
        """Convert the ControlledUnitary to a dictionary for JSON serialization.

        Returns:
            dict: Dictionary representation of the ControlledUnitary

        """
        data: dict[str, Any] = {}
        data["unitary"] = self.unitary.to_json()
        data["control_indices"] = self.control_indices
        data["target_indices"] = self.target_indices
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the ControlledUnitary to an HDF5 group.

        Args:
            group: HDF5 group or file to write the controlled unitary to

        """
        self._add_hdf5_version(group)

        # Write simple attributes
        group.attrs["control_indices"] = self.control_indices
        group.attrs["target_indices"] = self.target_indices

        # Create subgroup for the nested object
        unitary_group = group.create_group("unitary")
        self.unitary.to_hdf5(unitary_group)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "ControlledUnitary":
        """Create ControlledUnitary from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            ControlledUnitary

        """
        unitary = UnitaryRepresentation.from_json(json_data["unitary"])
        control_indices = json_data["control_indices"]
        target_indices = json_data.get("target_indices")
        return cls(
            unitary=unitary,
            control_indices=control_indices,
            target_indices=target_indices,
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "ControlledUnitary":
        """Load a ControlledUnitary from an HDF5 group.

        Args:
            group: The HDF5 group containing the serialized object.

        Returns:
            ControlledUnitary

        """
        # Verify version
        cls._validate_hdf5_version(cls._serialization_version, group)

        # Load simple attributes
        control_indices = list(group.attrs["control_indices"])
        target_indices = list(group.attrs["target_indices"]) if "target_indices" in group.attrs else None

        # Load nested UnitaryRepresentation
        unitary_group = group["unitary"]
        unitary = UnitaryRepresentation.from_hdf5(unitary_group)

        return cls(
            unitary=unitary,
            control_indices=control_indices,
            target_indices=target_indices,
        )

    def get_summary(self) -> str:
        """Get summary of controlled unitary.

        Returns:
            str: Summary string describing the ControlledUnitary's contents and properties

        """
        line = "Controlled Unitary:\n"
        line += f"  Control Indices: {self.control_indices}\n"
        line += "  Unitary Summary:\n"
        unitary_summary = self.unitary.get_summary()
        unitary_summary_indented = "\n".join("    " + summary_line for summary_line in unitary_summary.splitlines())
        line += unitary_summary_indented
        return line
