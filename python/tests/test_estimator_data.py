"""Test estimator data class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import tempfile
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from qdk_chemistry.data import QubitOperator
from qdk_chemistry.data.estimator_data import EnergyExpectationResult, MeasurementData
from qdk_chemistry.data.term_partition import LayeredPartition

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_measurement_data_serialization():
    """Test serialization and deserialization of MeasurementData."""
    # Create sample MeasurementData
    measurement_data = MeasurementData(
        bitstring_counts=[{"00": 5000, "11": 5000}, {"1": 6000, "0": 4000}, None],
        hamiltonians=[
            QubitOperator(["ZZ"], np.array([1.0])),
            QubitOperator(["IX"], np.array([1.0])),
            QubitOperator(["IY"], np.array([1.0])),
        ],
        shots_list=[10000, 10000, 0],
    )

    # Serialize to dictionary
    measurement_data_dict = measurement_data.to_dict()
    assert len(measurement_data_dict) == 4  # 3 measurements + version field
    assert "version" in measurement_data_dict
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
    with open(tmpfile_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data == measurement_data_dict


@pytest.mark.parametrize("format_name", ["json", "hdf5"])
def test_measurement_data_mixed_sparse_roundtrip(format_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Roundtrip dense and packed operators with partition metadata without expanding labels."""
    dense = QubitOperator(["ZX"], np.array([0.5]))
    packed = QubitOperator.from_sparse_terms(
        100_000,
        [{0: "X", 99_999: "Y"}, {}],
        np.array([1.25 + 0.25j, -0.5]),
        term_partition=LayeredPartition(strategy="test", groups=(((0,),), ((1,),))),
    )
    original = MeasurementData([dense, packed], [{"00": 3}, None], [3, 0])
    monkeypatch.setattr(type(packed.pauli_strings), "__getitem__", Mock(side_effect=AssertionError("Dense labels")))
    original_hash = original.content_hash(0)

    if format_name == "json":
        payload = json.loads(json.dumps(original.to_json()))
        assert payload["version"] == payload["1"]["hamiltonian"]["version"] == "0.2.0"
        restored = MeasurementData.from_json(payload)
    else:
        with h5py.File("mixed.measurement_data.h5", "w", driver="core", backing_store=False) as group:
            original.to_hdf5(group)
            assert group.attrs["version"] == group["hamiltonian_1"].attrs["version"] == "0.2.0"
            restored = MeasurementData.from_hdf5(group)

    assert restored.bitstring_counts == original.bitstring_counts
    assert restored.shots_list == original.shots_list
    for actual, expected in zip(restored.hamiltonians, original.hamiltonians, strict=True):
        assert actual.has_sparse_terms == expected.has_sparse_terms
        assert actual.num_qubits == expected.num_qubits
        np.testing.assert_array_equal(actual.coefficients, expected.coefficients)
        assert list(actual.iter_sparse_terms()) == list(expected.iter_sparse_terms())
        assert actual.term_partition == expected.term_partition
    assert restored.content_hash(0) == original_hash


@pytest.mark.parametrize("nested", [False, True])
def test_measurement_data_sparse_version_validation(nested: bool) -> None:
    """Reject an old schema version at either packed serialization boundary."""
    original = MeasurementData([QubitOperator.from_sparse_terms(1, [{0: "Z"}], np.array([1.0]))])
    payload = original.to_json()
    versioned_data = payload["0"]["hamiltonian"] if nested else payload
    versioned_data["version"] = "0.1.0"
    with pytest.raises(RuntimeError):
        MeasurementData.from_json(payload)


def test_energy_expectation_result_structure() -> None:
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
