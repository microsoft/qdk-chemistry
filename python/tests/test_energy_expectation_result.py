"""Tests for EnergyExpectationResult data class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import tempfile
from pathlib import Path

import numpy as np

from qdk_chemistry.data.estimator_data import EnergyExpectationResult


def test_energy_expectation_result_creation():
    """Test basic EnergyExpectationResult creation."""
    result = EnergyExpectationResult(
        energy_expectation_value=-1.5,
        energy_variance=0.01,
        expvals_each_term=[np.array([1.0, -0.5])],
        variances_each_term=[np.array([0.001, 0.002])],
    )

    assert result.energy_expectation_value == -1.5
    assert result.energy_variance == 0.01
    assert len(result.expvals_each_term) == 1
    assert len(result.variances_each_term) == 1


def test_energy_expectation_result_json_serialization():
    """Test EnergyExpectationResult JSON serialization."""
    result = EnergyExpectationResult(
        energy_expectation_value=-2.0,
        energy_variance=0.05,
        expvals_each_term=[np.array([1.0, -1.0]), np.array([0.5])],
        variances_each_term=[np.array([0.01, 0.02]), np.array([0.001])],
    )

    # Test to_json returns dict
    json_dict = result.to_json()
    assert isinstance(json_dict, dict)
    assert json_dict["energy_expectation_value"] == -2.0
    assert json_dict["energy_variance"] == 0.05
    assert len(json_dict["expvals_each_term"]) == 2
    assert len(json_dict["variances_each_term"]) == 2

    # Verify arrays are converted to lists
    assert isinstance(json_dict["expvals_each_term"][0], list)
    assert json_dict["expvals_each_term"][0] == [1.0, -1.0]


def test_energy_expectation_result_json_file_io():
    """Test EnergyExpectationResult JSON file I/O."""
    result = EnergyExpectationResult(
        energy_expectation_value=-1.0,
        energy_variance=0.001,
        expvals_each_term=[np.array([0.5, 0.5])],
        variances_each_term=[np.array([0.0001, 0.0002])],
    )

    with tempfile.NamedTemporaryFile(suffix=".energy_expectation_result.json", delete=False) as tmp:
        filename = tmp.name

    try:
        # Save to file
        result.to_json_file(filename)
        assert Path(filename).exists()

        # Verify file contents
        with open(filename) as f:
            data = json.load(f)

        assert data["energy_expectation_value"] == -1.0
        assert data["energy_variance"] == 0.001
        assert len(data["expvals_each_term"]) == 1
    finally:
        Path(filename).unlink()


def test_energy_expectation_result_hdf5_file_io():
    """Test EnergyExpectationResult HDF5 file I/O."""
    result = EnergyExpectationResult(
        energy_expectation_value=-3.5,
        energy_variance=0.1,
        expvals_each_term=[np.array([1.0, 2.0, 3.0])],
        variances_each_term=[np.array([0.01, 0.02, 0.03])],
    )

    with tempfile.NamedTemporaryFile(suffix=".energy_expectation_result.h5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Save to file
        result.to_hdf5_file(filename)
        assert Path(filename).exists()

        # Load from file
        loaded_result = EnergyExpectationResult.from_hdf5_file(filename)

        # Verify data
        assert loaded_result.energy_expectation_value == result.energy_expectation_value
        assert loaded_result.energy_variance == result.energy_variance
        assert len(loaded_result.expvals_each_term) == len(result.expvals_each_term)
        np.testing.assert_array_equal(loaded_result.expvals_each_term[0], result.expvals_each_term[0])
    finally:
        Path(filename).unlink()


def test_energy_expectation_result_summary():
    """Test EnergyExpectationResult summary string."""
    result = EnergyExpectationResult(
        energy_expectation_value=-1.5,
        energy_variance=0.04,
        expvals_each_term=[np.array([1.0])],
        variances_each_term=[np.array([0.01])],
    )

    summary = result.get_summary()
    assert isinstance(summary, str)
    assert "Energy" in summary
    assert "-1.5" in summary
    assert "Variance" in summary


def test_energy_expectation_result_immutability():
    """Test that EnergyExpectationResult is immutable after construction."""
    energy = EnergyExpectationResult(-1.0, 0.01, [np.array([1.0])], [np.array([0.01])])

    try:
        energy.energy_expectation_value = -2.0
        raise AssertionError("Should not be able to modify EnergyExpectationResult after construction")
    except AttributeError:
        pass  # Expected - DataClass should be immutable
