"""Test for noise model functionalities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import pytest

from qdk_chemistry.noise_models import (
    QuantumErrorProfile,
    SupportedErrorTypes,
    SupportedGate,
)


def test_profile_dumping(simple_error_profile):
    """Test dumping quantum error profile to YAML."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        simple_error_profile.to_yaml(tmp_file.name)


def test_yaml_save_and_load_equivalence(simple_error_profile):
    """Test that a saved error profile gives the same values on loading."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        simple_error_profile.to_yaml(tmp_file.name)
        loaded_profile = QuantumErrorProfile.from_yaml(tmp_file.name)
        assert simple_error_profile == loaded_profile


def test_basis_gates(simple_error_profile):
    """Check for valid basis gates in all available quantum error profiles."""
    basis_gates = simple_error_profile.basis_gates
    assert isinstance(basis_gates, list)
    assert len(basis_gates) > 0
    for gate in basis_gates:
        assert isinstance(gate, str)
        assert len(gate) > 0
    assert "measure" not in basis_gates
    assert "h" in basis_gates
    assert "cx" in basis_gates


def test_quantum_error_profile_unsupported_qubit_number():
    """Test QuantumErrorProfile with unsupported number of qubits."""
    with pytest.raises(ValueError, match="Unsupported number of qubits"):
        QuantumErrorProfile(
            name="test",
            description="test profile",
            errors={
                SupportedGate.H: {
                    "type": SupportedErrorTypes.DEPOLARIZING_ERROR,
                    "rate": 0.01,
                    "num_qubits": 3,  # Unsupported - should be 1 or 2
                }
            },
        )


def test_quantum_error_profile_from_yaml_file_not_found():
    """Test loading from non-existent YAML file."""
    with pytest.raises(FileNotFoundError, match=r"File .* not found"):
        QuantumErrorProfile.from_yaml("non_existent_file.yaml")


def test_quantum_error_profile_from_yaml_empty_file():
    """Test loading from empty YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")  # Empty file
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match=r"YAML file .* is empty or invalid"):
            QuantumErrorProfile.from_yaml(temp_path)
    finally:
        Path(temp_path).unlink()


def test_quantum_error_profile_from_yaml_invalid_keys():
    """Test loading YAML with invalid keys."""
    yaml_content = """
name: test_profile
description: test description
invalid_key: invalid_value
errors:
  h:
    type: depolarizing_error
    rate: 0.01
    num_qubits: 1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid keys in YAML file"):
            QuantumErrorProfile.from_yaml(temp_path)
    finally:
        Path(temp_path).unlink()


def test_supported_gate_from_string_invalid():
    """Test SupportedGate.from_string and direct string conversion with invalid gate string."""
    with pytest.raises(ValueError, match="Unknown gate type"):
        SupportedGate.from_string("invalid_gate")
    with pytest.raises(ValueError, match="is not a valid SupportedGate"):
        SupportedGate("invalid_gate")


def test_supported_gate_from_string_valid():
    """Test SupportedGate.from_string and direct string conversion with valid gate strings."""
    assert SupportedGate.from_string("h") == SupportedGate.H
    assert SupportedGate.from_string("CX") == SupportedGate.CX  # Test case insensitivity
    assert SupportedGate("h") == SupportedGate.H
    assert SupportedGate("cY") == SupportedGate.CY


def test_supported_gate_to_string():
    """Test conversion of SupportedGate enum to string."""
    assert str(SupportedGate.H) == "h"
    assert str(SupportedGate.CX) == "cx"
    assert str(SupportedGate.CY) == "cy"


def test_quantum_error_profile_from_yaml_minimal():
    """Test loading minimal YAML file with defaults."""
    yaml_content = """
errors:
  h:
    type: depolarizing_error
    rate: 0.01
    num_qubits: 1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        profile = QuantumErrorProfile.from_yaml(temp_path)
        assert profile.name == "default"  # Default name
        assert profile.description == "No description provided"  # Default description
        assert len(profile.errors) == 1
    finally:
        Path(temp_path).unlink()


def test_to_dict_conversion():
    """Test to_dict method converts enums to strings properly."""
    profile = QuantumErrorProfile(
        name="test",
        description="test profile",
        errors={
            SupportedGate.H: {
                "type": SupportedErrorTypes.DEPOLARIZING_ERROR,
                "rate": 0.01,
                "num_qubits": 1,
            }
        },
    )

    result_dict = profile.to_dict()

    # Check that enum keys and values are converted to strings
    assert "h" in result_dict["errors"]
    assert result_dict["errors"]["h"]["type"] == "depolarizing_error"
    assert result_dict["errors"]["h"]["rate"] == 0.01
    assert result_dict["errors"]["h"]["num_qubits"] == 1
