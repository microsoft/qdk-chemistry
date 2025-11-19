"""Tests for MeasurementData data class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import tempfile
from pathlib import Path

import numpy as np

from qdk_chemistry.data.estimator_data import MeasurementData
from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian


def test_measurement_data_creation():
    """Test basic MeasurementData creation."""
    ham = QubitHamiltonian(["ZI", "XI"], np.array([1.0, 0.5]))

    data = MeasurementData(
        hamiltonians=[ham],
        bitstring_counts=[{"0": 50, "1": 50}],
        shots_list=[100],
    )

    assert len(data.hamiltonians) == 1
    assert len(data.bitstring_counts) == 1
    assert data.shots_list == [100]


def test_measurement_data_default_values():
    """Test MeasurementData with default values."""
    ham = QubitHamiltonian(["Z"], np.array([1.0]))

    data = MeasurementData(hamiltonians=[ham])

    assert len(data.hamiltonians) == 1
    assert data.bitstring_counts == []
    assert data.shots_list == []


def test_measurement_data_json_serialization():
    """Test MeasurementData JSON serialization."""
    ham1 = QubitHamiltonian(["Z"], np.array([1.0]))
    ham2 = QubitHamiltonian(["X"], np.array([0.5]))

    data = MeasurementData(
        hamiltonians=[ham1, ham2],
        bitstring_counts=[{"0": 60, "1": 40}, {"0": 55, "1": 45}],
        shots_list=[100, 100],
    )

    # Test to_json returns dict
    json_dict = data.to_json()
    assert isinstance(json_dict, dict)
    assert "0" in json_dict
    assert "1" in json_dict

    # Verify Hamiltonian data
    assert "hamiltonian" in json_dict["0"]
    assert "paulis" in json_dict["0"]["hamiltonian"]
    assert "coefficients" in json_dict["0"]["hamiltonian"]
    assert "bitstring" in json_dict["0"]
    assert "shots" in json_dict["0"]

    # Verify values
    assert json_dict["0"]["shots"] == 100
    assert json_dict["0"]["bitstring"] == {"0": 60, "1": 40}


def test_measurement_data_json_file_io():
    """Test MeasurementData JSON file I/O."""
    ham = QubitHamiltonian(["ZZ", "IZ"], np.array([1.0, -0.5]))

    data = MeasurementData(
        hamiltonians=[ham],
        bitstring_counts=[{"00": 25, "01": 25, "10": 25, "11": 25}],
        shots_list=[100],
    )

    with tempfile.NamedTemporaryFile(suffix=".measurement_data.json", delete=False) as tmp:
        filename = tmp.name

    try:
        # Save to file
        data.to_json_file(filename)
        assert Path(filename).exists()

        # Verify file contents
        with open(filename) as f:
            loaded = json.load(f)

        assert "0" in loaded
        assert loaded["0"]["shots"] == 100
        assert "00" in loaded["0"]["bitstring"]
    finally:
        Path(filename).unlink()


def test_measurement_data_hdf5_file_io():
    """Test MeasurementData HDF5 file I/O."""
    ham1 = QubitHamiltonian(["Z"], np.array([1.0]))
    ham2 = QubitHamiltonian(["XI", "YI"], np.array([0.5, -0.3]))

    data = MeasurementData(
        hamiltonians=[ham1, ham2],
        bitstring_counts=[{"0": 70, "1": 30}, {"0": 55, "1": 45}],
        shots_list=[100, 100],
    )

    with tempfile.NamedTemporaryFile(suffix=".measurement_data.h5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Save to file
        data.to_hdf5_file(filename)
        assert Path(filename).exists()

        # Load from file
        loaded_data = MeasurementData.from_hdf5_file(filename)

        # Verify data
        assert len(loaded_data.hamiltonians) == 2
        assert len(loaded_data.bitstring_counts) == 2
        assert loaded_data.shots_list == [100, 100]

        # Verify Hamiltonian data
        np.testing.assert_array_equal(
            loaded_data.hamiltonians[0].coefficients,
            data.hamiltonians[0].coefficients,
        )
    finally:
        Path(filename).unlink()


def test_measurement_data_summary():
    """Test MeasurementData summary string."""
    ham = QubitHamiltonian(["Z"], np.array([1.0]))

    data = MeasurementData(
        hamiltonians=[ham, ham, ham],
        bitstring_counts=[{"0": 50, "1": 50}] * 3,
        shots_list=[100, 200, 150],
    )

    summary = data.get_summary()
    assert isinstance(summary, str)
    assert "3" in summary  # Number of Hamiltonians
    assert "450" in summary  # Total shots
    assert "Measurement" in summary


def test_measurement_data_empty_measurements():
    """Test MeasurementData with no measurements collected."""
    ham = QubitHamiltonian(["Z"], np.array([1.0]))

    data = MeasurementData(hamiltonians=[ham])

    # Should handle empty lists gracefully
    json_dict = data.to_json()
    assert "0" in json_dict
    assert json_dict["0"]["bitstring"] is None
    assert json_dict["0"]["shots"] == 0


def test_measurement_data_immutability():
    """Test that MeasurementData is immutable after construction."""
    ham = QubitHamiltonian(["Z"], np.array([1.0]))
    measurement = MeasurementData([ham], [{"0": 50, "1": 50}], [100])

    try:
        measurement.shots_list = [200]
        raise AssertionError("Should not be able to modify MeasurementData after construction")
    except AttributeError:
        pass  # Expected - DataClass should be immutable
