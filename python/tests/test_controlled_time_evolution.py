"""Test for ControlledTimeEvolutionUnitary in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py

from qdk_chemistry.data import ControlledTimeEvolutionUnitary, TimeEvolutionUnitary, TimeEvolutionUnitaryContainer


class MockTimeEvolutionUnitaryContainer(TimeEvolutionUnitaryContainer):
    """Mock implementation of TimeEvolutionUnitaryContainer for testing purposes."""

    _data_type_name = "mock_time_evolution_unitary_container"
    _serialization_version = "0.1.0"

    def __init__(self, num_qubits: int):
        """Initialize the mock container."""
        self._num_qubits = num_qubits
        super().__init__()

    @property
    def type(self) -> str:
        """Get the type of the time evolution unitary container."""
        return "mock"

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits the time evolution unitary acts on."""
        return self._num_qubits

    def to_json(self) -> dict[str, Any]:
        """Convert the TimeEvolutionUnitary to a dictionary for JSON serialization."""
        data = {
            "container_type": self.type,
            "num_qubits": self._num_qubits,
        }
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the TimeEvolutionUnitary to an HDF5 group."""
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["num_qubits"] = self._num_qubits

    def get_summary(self):
        """Get a summary string for the mock container."""
        return f"Mock Time Evolution Unitary with {self._num_qubits} qubits"


def create_mock_time_evolution_unitary(num_qubits: int) -> TimeEvolutionUnitary:
    """Create a mock TimeEvolutionUnitary for testing."""
    container = MockTimeEvolutionUnitaryContainer(num_qubits=num_qubits)
    return TimeEvolutionUnitary(container=container)


class TestControlledTimeEvolutionUnitary:
    """Tests for ControlledTimeEvolutionUnitary."""

    def test_basic_properties(self):
        """Test basic properties of ControlledTimeEvolutionUnitary."""
        teu = create_mock_time_evolution_unitary(num_qubits=4)
        cteu = ControlledTimeEvolutionUnitary(teu, control_index=1)

        assert cteu.control_index == 1
        assert cteu.get_num_system_qubits() == 4
        assert cteu.get_unitary_container_type() == "mock"

    def test_to_json_serialization(self):
        """Test JSON serialization."""
        teu = create_mock_time_evolution_unitary(num_qubits=6)
        cteu = ControlledTimeEvolutionUnitary(teu, control_index=2)

        json_data = cteu.to_json()
        assert "time_evolution_unitary" in json_data
        assert json_data["control_index"] == 2

    def test_to_hdf5_roundtrip(self, tmp_path):
        """Test HDF5 serialization and deserialization."""
        teu = create_mock_time_evolution_unitary(num_qubits=8)
        cteu = ControlledTimeEvolutionUnitary(teu, control_index=0)

        file_path = tmp_path / "cte_unitary.h5"

        with h5py.File(file_path, "w") as f:
            grp = f.create_group("cte")
            cteu.to_hdf5(grp)
        assert file_path.exists()
        assert file_path.stat().st_size > 0
        with h5py.File(file_path, "r") as f:
            grp = f["cte"]
            control_index = grp.attrs["control_index"]
            assert control_index == 0

            teu_group = grp["time_evolution_unitary"]
            num_qubits = teu_group.attrs["num_qubits"]
            assert num_qubits == 8

    def test_summary_format(self):
        """Test the summary format of ControlledTimeEvolutionUnitary."""
        teu = create_mock_time_evolution_unitary(num_qubits=3)
        cteu = ControlledTimeEvolutionUnitary(teu, control_index=5)

        summary = cteu.get_summary()
        assert "Controlled Time Evolution Unitary" in summary
        assert "Control Index: 5" in summary
        assert "Mock Time Evolution Unitary" in summary
