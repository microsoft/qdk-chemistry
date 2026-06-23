"""Tests for QpeResult data class."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from qdk_chemistry.data import QpeResult
from qdk_chemistry.data.unitary_representation.containers.block_encoding import LCUContainer
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import QuantumWalkContainer
from tests.reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_qpe_result_creation():
    """Test basic QpeResult creation."""
    result = QpeResult.from_phase_fraction(
        method="IQPE",
        phase_fraction=0.25,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=1.0),
        bits_msb_first=[0, 1],
    )

    assert result.method == "IQPE"
    assert result.phase_fraction == 0.25
    assert result.bits_msb_first == (0, 1)
    assert result.bitstring_msb_first == "01"


def test_qpe_result_json_serialization():
    """Test QPE result JSON serialization round-trip."""
    result = QpeResult.from_phase_fraction(
        method="IQPE",
        phase_fraction=0.125,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=2.0),
        bits_msb_first=[0, 0, 1],
    )

    # Test to_json returns dict
    json_dict = result.to_json()
    assert isinstance(json_dict, dict)
    assert json_dict["method"] == "IQPE"
    assert json_dict["phase_fraction"] == 0.125
    assert "raw_energy" in json_dict
    assert "branching" in json_dict


def test_qpe_result_json_file_io():
    """Test QPE result JSON file I/O."""
    result = QpeResult.from_phase_fraction(
        method="QPE",
        phase_fraction=0.5,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=1.0),
    )

    with tempfile.NamedTemporaryFile(suffix=".qpe_result.json", delete=False) as tmp:
        filename = tmp.name

    try:
        # Save to file
        result.to_json_file(filename)
        assert Path(filename).exists()

        # Load from file
        loaded_result = QpeResult.from_json_file(filename)

        # Verify data
        assert loaded_result.method == result.method
        assert loaded_result.phase_fraction == result.phase_fraction
        assert loaded_result.raw_energy == result.raw_energy
    finally:
        Path(filename).unlink()


def test_qpe_result_hdf5_file_io():
    """Test QPE result HDF5 file I/O."""
    result = QpeResult.from_phase_fraction(
        method="IQPE",
        phase_fraction=0.375,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=3.0),
        metadata={"test": "data"},
    )

    with tempfile.NamedTemporaryFile(suffix=".qpe_result.hdf5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Save to file
        result.to_hdf5_file(filename)
        assert Path(filename).exists()

        # Load from file
        loaded_result = QpeResult.from_hdf5_file(filename)

        # Verify data
        assert loaded_result.method == result.method
        assert loaded_result.phase_fraction == result.phase_fraction
        assert loaded_result.resolved_energy == result.resolved_energy
    finally:
        Path(filename).unlink()


def test_qpe_result_summary():
    """Test QPE result summary string."""
    result = QpeResult.from_phase_fraction(
        method="IQPE",
        phase_fraction=0.25,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=1.0),
    )

    summary = result.get_summary()
    assert isinstance(summary, str)
    assert "IQPE" in summary
    assert "Phase fraction" in summary
    assert "Raw energy" in summary


def test_qpe_result_immutability():
    """Test that QpeResult is immutable after construction."""
    qpe = QpeResult.from_phase_fraction(
        method="IQPE",
        phase_fraction=0.25,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=1.0),
    )

    try:
        qpe.method = "different"
        raise AssertionError("Should not be able to modify QpeResult after construction")
    except AttributeError:
        pass  # Expected - DataClass should be immutable


def test_qpe_result_from_json():
    """Test QPE result creation from JSON dictionary."""
    json_data = {
        "version": QpeResult._serialization_version,
        "method": "IQPE",
        "phase_fraction": 0.25,
        "phase_angle": 1.5707963267948966,
        "canonical_phase_fraction": 0.25,
        "canonical_phase_angle": 1.5707963267948966,
        "raw_energy": -1.0,
        "branching": [-1.0, -0.5, 0.0],
        "resolved_energy": -1.0,
        "bits_msb_first": [0, 1],
        "bitstring_msb_first": "01",
        "metadata": {"test": "value"},
    }

    result = QpeResult.from_json(json_data)

    assert result.method == "IQPE"
    assert result.phase_fraction == 0.25
    assert result.raw_energy == -1.0
    assert result.branching == (-1.0, -0.5, 0.0)
    assert result.resolved_energy == -1.0
    assert result.bits_msb_first == (0, 1)
    assert result.bitstring_msb_first == "01"
    assert result.metadata == {"test": "value"}


def test_qpe_result_from_json_minimal():
    """Test QPE result creation from minimal JSON dictionary."""
    json_data = {
        "version": QpeResult._serialization_version,
        "method": "QPE",
        "phase_fraction": 0.5,
        "phase_angle": 3.141592653589793,
        "canonical_phase_fraction": 0.5,
        "canonical_phase_angle": 3.141592653589793,
        "raw_energy": -2.0,
        "branching": [-2.0],
    }

    result = QpeResult.from_json(json_data)

    assert result.method == "QPE"
    assert result.phase_fraction == 0.5
    assert result.resolved_energy is None
    assert result.bits_msb_first is None
    assert result.bitstring_msb_first is None
    assert result.metadata is None


def test_qpe_result_json_roundtrip():
    """Test QPE result JSON serialization/deserialization roundtrip."""
    original = QpeResult.from_phase_fraction(
        method="IQPE",
        phase_fraction=0.375,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=2.5),
        bits_msb_first=[1, 1, 0],
        metadata={"source": "test", "iteration": 5},
    )

    # Serialize to JSON
    json_data = original.to_json()

    # Deserialize from JSON
    restored = QpeResult.from_json(json_data)

    # Verify all fields match
    assert restored.method == original.method
    assert restored.phase_fraction == original.phase_fraction
    assert restored.phase_angle == original.phase_angle
    assert restored.canonical_phase_fraction == original.canonical_phase_fraction
    assert restored.canonical_phase_angle == original.canonical_phase_angle
    assert restored.raw_energy == original.raw_energy
    assert restored.branching == original.branching
    assert restored.resolved_energy == original.resolved_energy
    assert restored.bits_msb_first == original.bits_msb_first
    assert restored.bitstring_msb_first == original.bitstring_msb_first
    assert restored.metadata == original.metadata


def test_qpe_result_hdf5_roundtrip():
    """Test QPE result HDF5 serialization/deserialization roundtrip."""
    original = QpeResult.from_phase_fraction(
        method="QPE",
        phase_fraction=0.125,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=1.0),
        bits_msb_first=[0, 0, 1],
        metadata={"tag": "experiment_1"},
    )

    with tempfile.NamedTemporaryFile(suffix=".qpe_result.hdf5", delete=False) as tmp:
        filename = tmp.name

    try:
        # Save to HDF5
        with h5py.File(filename, "w") as f:
            original.to_hdf5(f)

        # Load from HDF5
        with h5py.File(filename, "r") as f:
            restored = QpeResult.from_hdf5(f)

        # Verify all fields match
        assert restored.method == original.method
        assert restored.phase_fraction == original.phase_fraction
        assert restored.phase_angle == original.phase_angle
        assert restored.canonical_phase_fraction == original.canonical_phase_fraction
        assert restored.canonical_phase_angle == original.canonical_phase_angle
        assert restored.raw_energy == original.raw_energy
        assert restored.branching == original.branching
        assert restored.resolved_energy == original.resolved_energy
        assert restored.bits_msb_first == original.bits_msb_first
        assert restored.bitstring_msb_first == original.bitstring_msb_first
        assert restored.metadata == original.metadata
    finally:
        Path(filename).unlink()


def test_qpe_result_from_file_json():
    """Test QPE result loading from file with explicit JSON format."""
    result = QpeResult.from_phase_fraction(
        method="IQPE",
        phase_fraction=0.25,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=1.0),
    )

    with tempfile.NamedTemporaryFile(suffix=".qpe_result.json", delete=False) as tmp:
        filename = tmp.name

    try:
        result.to_file(filename, "json")
        loaded = QpeResult.from_file(filename, "json")

        assert loaded.method == result.method
        assert loaded.phase_fraction == result.phase_fraction
    finally:
        Path(filename).unlink()


def test_qpe_result_from_file_hdf5():
    """Test QPE result loading from file with explicit HDF5 format."""
    result = QpeResult.from_phase_fraction(
        method="QPE",
        phase_fraction=0.5,
        eigenvalue_from_phase=lambda phi: PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=2.0),
    )

    with tempfile.NamedTemporaryFile(suffix=".qpe_result.hdf5", delete=False) as tmp:
        filename = tmp.name

    try:
        result.to_file(filename, "hdf5")
        loaded = QpeResult.from_file(filename, "hdf5")

        assert loaded.method == result.method
        assert loaded.phase_fraction == result.phase_fraction
    finally:
        Path(filename).unlink()


def test_qpe_result_from_phase_fraction_qubitization():
    """Test QpeResult creation from qubitization phase measurement via from_phase_fraction."""
    lambda_val = 5.0
    phase_fraction = 0.25
    expected_energy = 0.0  # λ·cos(π/2)

    result = QpeResult.from_phase_fraction(
        method="qubitization_qpe",
        phase_fraction=phase_fraction,
        eigenvalue_from_phase=lambda phi: QuantumWalkContainer.eigenvalue_from_phase(phi, scale=lambda_val),
        bits_msb_first=[0, 1, 0, 0],
    )

    assert result.method == "qubitization_qpe"
    assert result.phase_fraction == phase_fraction
    assert np.isclose(
        result.raw_energy,
        expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )
    assert result.branching == (result.raw_energy,)
    assert result.resolved_energy is None
    assert result.bits_msb_first == (0, 1, 0, 0)
    assert result.bitstring_msb_first == "0100"
    assert result.metadata is None


@pytest.mark.parametrize(
    ("phi", "scale", "expected"),
    [
        (0.0, 2.0, 0.0),
        (0.1, 2.0, 0.3141592653589793),
        (0.75, 1.0, -1.5707963267948966),
    ],
    ids=["zero_phase", "positive_energy", "negative_wrap"],
)
def test_ppf_eigenvalue_from_phase(phi, scale, expected):
    """Time evolution eigenvalue_from_phase maps phase fraction to correct energy."""
    assert np.isclose(
        PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=scale),
        expected,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )


def test_ppf_eigenvalue_from_phase_roundtrip():
    """Verify roundtrip: E → phase → E for time evolution (principal branch)."""
    t = 2.0
    energy = 0.5  # |E*t| = 1.0 < π, stays in principal branch
    # Convention: U = e^{iHt}, so eigenvalue = e^{iEt}, phase_fraction = Et/(2π)
    phi = (energy * t / (2 * np.pi)) % 1.0
    assert np.isclose(
        PauliProductFormulaContainer.eigenvalue_from_phase(phi, scale=t),
        energy,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )


@pytest.mark.parametrize(
    ("phi", "scale", "expected"),
    [
        (0.0, 5.0, 5.0),
        (0.25, 3.0, 0.0),
        (0.5, 4.0, -4.0),
    ],
    ids=["zero_phase", "quarter_phase", "half_phase"],
)
def test_qw_eigenvalue_from_phase(phi, scale, expected):
    """Quantum walk eigenvalue_from_phase maps phase fraction to correct energy."""
    assert np.isclose(
        QuantumWalkContainer.eigenvalue_from_phase(phi, scale=scale),
        expected,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )


def test_qw_eigenvalue_from_phase_roundtrip():
    """Verify roundtrip: E → phase → E for quantum walk."""
    lam = 6.0
    energy = -2.5
    phi = np.arccos(energy / lam) / (2 * np.pi)
    assert np.isclose(
        QuantumWalkContainer.eigenvalue_from_phase(phi, scale=lam),
        energy,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )


def test_lcu_eigenvalue_from_phase_raises():
    """LCUContainer.eigenvalue_from_phase raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        LCUContainer.eigenvalue_from_phase(0.25, scale=1.0)
