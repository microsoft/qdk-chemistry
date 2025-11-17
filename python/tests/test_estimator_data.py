"""Test estimator data class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import tempfile

import numpy as np

from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.data.estimator_data import EnergyExpectationResult, MeasurementData

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_measurement_data_serialization():
    """Test serialization and deserialization of MeasurementData."""
    # Create sample MeasurementData
    measurement_data = MeasurementData(
        bitstring_counts=[{"00": 5000, "11": 5000}, {"1": 6000, "0": 4000}, None],
        hamiltonians=[
            QubitHamiltonian(["ZZ"], np.array([1.0])),
            QubitHamiltonian(["IX"], np.array([1.0])),
            QubitHamiltonian(["IY"], np.array([1.0])),
        ],
        shots_list=[10000, 10000, 0],
    )

    # Serialize to dictionary
    measurement_data_dict = measurement_data.to_dict()
    assert len(measurement_data_dict) == 3
    assert measurement_data_dict["0"]["hamiltonian"]["paulis"] == ["ZZ"]
    assert measurement_data_dict["0"]["shots"] == 10000
    assert measurement_data_dict["1"]["hamiltonian"]["paulis"] == ["IX"]
    assert measurement_data_dict["1"]["shots"] == 10000
    assert measurement_data_dict["2"]["hamiltonian"]["paulis"] == ["IY"]
    assert measurement_data_dict["2"]["shots"] == 0
    assert measurement_data_dict["0"]["bitstring"] == {"00": 5000, "11": 5000}
    assert measurement_data_dict["1"]["bitstring"] == {"1": 6000, "0": 4000}
    assert measurement_data_dict["2"]["bitstring"] is None

    # Save to json file
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".measurement_data.json", delete=False) as tmpfile:
        measurement_data.to_json_file(tmpfile.name)
        tmpfile_path = tmpfile.name

    # Load from json file and verify contents
    with open(tmpfile_path) as f:
        data = json.load(f)
    assert data == measurement_data_dict


def test_energy_expectation_result_structure():
    """Test EnergyExpectationResult TypedDict structure."""
    sample_result: EnergyExpectationResult = {
        "energy_expectation_value": -1.234,
        "energy_variance": 0.056,
        "expvals_each_term": [np.array([0.5, -0.5]), np.array([1.0])],
        "variances_each_term": [np.array([0.1, 0.1]), np.array([0.05])],
    }

    assert np.isclose(
        sample_result["energy_expectation_value"],
        -1.234,
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )
    assert np.isclose(
        sample_result["energy_variance"],
        0.056,
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )
    assert np.allclose(
        sample_result["expvals_each_term"][0],
        np.array([0.5, -0.5]),
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )
    assert np.allclose(
        sample_result["variances_each_term"][1],
        np.array([0.05]),
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )
