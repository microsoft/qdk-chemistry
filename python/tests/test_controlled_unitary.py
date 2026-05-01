"""Test for ControlledUnitary in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py
import pytest

from qdk_chemistry.data import ControlledUnitary, UnitaryContainer, UnitaryRepresentation


class MockUnitaryContainer(UnitaryContainer):
    """Mock implementation of UnitaryContainer for testing purposes."""

    _data_type_name = "mock_unitary_container"
    _serialization_version = "0.1.0"

    def __init__(self, num_qubits: int):
        """Initialize the mock container."""
        self._num_qubits = num_qubits
        super().__init__()

    @property
    def type(self) -> str:
        """Get the type of the unitary container."""
        return "mock"

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits the unitary acts on."""
        return self._num_qubits

    def to_json(self) -> dict[str, Any]:
        """Convert the UnitaryRepresentation to a dictionary for JSON serialization."""
        data = {
            "container_type": self.type,
            "num_qubits": self._num_qubits,
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the UnitaryRepresentation to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["num_qubits"] = self._num_qubits

    def get_summary(self) -> str:
        """Get a summary string for the mock container."""
        return f"Mock Unitary with {self._num_qubits} qubits"


def create_mock_unitary(num_qubits: int) -> UnitaryRepresentation:
    """Create a mock UnitaryRepresentation for testing."""
    container = MockUnitaryContainer(num_qubits=num_qubits)
    return UnitaryRepresentation(container=container)


class TestControlledUnitary:
    """Tests for ControlledUnitary."""

    def test_basic_properties(self):
        """Test basic properties of ControlledUnitary."""
        unitary_rep = create_mock_unitary(num_qubits=4)
        cteu = ControlledUnitary(unitary_rep, control_indices=[1])

        assert cteu.control_indices == [1]
        assert cteu.get_num_total_qubits() == 5
        assert cteu.get_unitary_container_type() == "mock"

    def test_to_json_serialization(self):
        """Test JSON serialization."""
        unitary_rep = create_mock_unitary(num_qubits=6)
        cteu = ControlledUnitary(unitary_rep, control_indices=[2])

        json_data = cteu.to_json()
        assert "unitary" in json_data
        assert json_data["control_indices"] == [2]

    def test_to_hdf5_roundtrip(self, tmp_path):
        """Test HDF5 serialization and deserialization."""
        unitary_rep = create_mock_unitary(num_qubits=8)
        cteu = ControlledUnitary(unitary_rep, control_indices=[0])

        file_path = tmp_path / "cte_unitary.h5"

        with h5py.File(file_path, "w") as f:
            grp = f.create_group("cte")
            cteu.to_hdf5(grp)
        assert file_path.exists()
        assert file_path.stat().st_size > 0
        with h5py.File(file_path, "r") as f:
            grp = f["cte"]
            control_indices = grp.attrs["control_indices"]
            assert list(control_indices) == [0]

            unitary_group = grp["unitary"]
            num_qubits = unitary_group.attrs["num_qubits"]
            assert num_qubits == 8

    def test_summary_format(self):
        """Test the summary format of ControlledUnitary."""
        unitary_rep = create_mock_unitary(num_qubits=3)
        cteu = ControlledUnitary(unitary_rep, control_indices=[5])

        summary = cteu.get_summary()
        assert "Controlled Unitary" in summary
        assert "Control Indices: [5]" in summary
        assert "Mock Unitary" in summary

    def test_rejects_duplicate_control_indices(self):
        """Test that duplicate control indices raise ValueError."""
        unitary_rep = create_mock_unitary(num_qubits=4)
        with pytest.raises(ValueError, match="control_indices must not contain duplicates"):
            ControlledUnitary(unitary_rep, control_indices=[0, 0])

    def test_rejects_duplicate_target_indices(self):
        """Test that duplicate target indices raise ValueError."""
        unitary_rep = create_mock_unitary(num_qubits=3)
        with pytest.raises(ValueError, match="target_indices must not contain duplicates"):
            ControlledUnitary(unitary_rep, control_indices=[0], target_indices=[1, 1, 2])

    def test_rejects_target_indices_length_mismatch(self):
        """Test that target_indices length must match unitary qubit count."""
        unitary_rep = create_mock_unitary(num_qubits=4)
        with pytest.raises(ValueError, match="target_indices length"):
            ControlledUnitary(unitary_rep, control_indices=[0], target_indices=[1, 2])

    def test_rejects_overlapping_control_and_target(self):
        """Test that overlapping control and target indices raise ValueError."""
        unitary_rep = create_mock_unitary(num_qubits=3)
        with pytest.raises(ValueError, match="must not overlap"):
            ControlledUnitary(unitary_rep, control_indices=[0], target_indices=[0, 1, 2])

    def test_default_target_indices_skips_controls(self):
        """Test that default target_indices assigns first available indices, skipping controls."""
        unitary_rep = create_mock_unitary(num_qubits=3)

        # Control at index 0 → targets should be [1, 2, 3]
        cu = ControlledUnitary(unitary_rep, control_indices=[0])
        assert cu.target_indices == [1, 2, 3]

        # Control at index 2 → targets should be [0, 1, 3]
        cu = ControlledUnitary(unitary_rep, control_indices=[2])
        assert cu.target_indices == [0, 1, 3]

        # Controls at indices 0 and 2 → targets should be [1, 3, 4]
        cu = ControlledUnitary(unitary_rep, control_indices=[0, 2])
        assert cu.target_indices == [1, 3, 4]

        # Controls at indices 4 → default targets should be [0, 1, 2]
        cu = ControlledUnitary(unitary_rep, control_indices=[4])
        assert cu.target_indices == [0, 1, 2]
