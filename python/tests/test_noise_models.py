"""Test for noise model functionalities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from qdk.simulation import NoiseConfig

from qdk_chemistry.data.noise_models import (
    GateErrorDef,
    QuantumErrorProfile,
    SupportedErrorTypes,
    SupportedGate,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_profile_dumping(simple_error_profile):
    """Test dumping quantum error profile to YAML."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        simple_error_profile.to_yaml_file(tmp_file.name)


def test_yaml_save_and_load_equivalence(simple_error_profile):
    """Test that a saved error profile gives the same values on loading."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        simple_error_profile.to_yaml_file(tmp_file.name)
        loaded_profile = QuantumErrorProfile.from_yaml_file(tmp_file.name)
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
        QuantumErrorProfile.from_yaml_file("non_existent_file.yaml")


def test_quantum_error_profile_from_yaml_file_empty():
    """Test loading from empty YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")  # Empty file
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match=r"YAML file .* is empty or invalid"):
            QuantumErrorProfile.from_yaml_file(temp_path)
    finally:
        Path(temp_path).unlink()


def test_quantum_error_profile_from_yaml_file_invalid_keys():
    """Test loading YAML with invalid keys."""
    yaml_content = f"""
version: {QuantumErrorProfile._serialization_version}
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
            QuantumErrorProfile.from_yaml_file(temp_path)
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


def test_quantum_error_profile_from_minimal_yaml_file():
    """Test loading minimal YAML file with defaults."""
    yaml_content = f"""
version: {QuantumErrorProfile._serialization_version}
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
        profile = QuantumErrorProfile.from_yaml_file(temp_path)
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


def test_quantum_error_profile_initialization_defaults():
    """Test QuantumErrorProfile initialization with default values."""
    profile = QuantumErrorProfile()
    assert profile.name == "default"
    assert profile.description == "No description provided"
    assert len(profile.errors) == 0
    assert len(profile.one_qubit_gates) == 0
    assert len(profile.two_qubit_gates) == 0


def test_quantum_error_profile_initialization_with_enum_keys():
    """Test QuantumErrorProfile initialization using SupportedGate enum keys."""
    profile = QuantumErrorProfile(
        name="test_enum",
        description="test with enums",
        errors={
            SupportedGate.H: {
                "type": SupportedErrorTypes.DEPOLARIZING_ERROR,
                "rate": 0.01,
                "num_qubits": 1,
            },
            SupportedGate.CX: {
                "type": SupportedErrorTypes.DEPOLARIZING_ERROR,
                "rate": 0.02,
                "num_qubits": 2,
            },
        },
    )
    assert len(profile.errors) == 2
    assert SupportedGate.H in profile.errors
    assert SupportedGate.CX in profile.errors
    assert "h" in profile.one_qubit_gates
    assert "cx" in profile.two_qubit_gates


def test_quantum_error_profile_initialization_with_string_keys():
    """Test QuantumErrorProfile initialization using string keys."""
    profile = QuantumErrorProfile(
        name="test_strings",
        description="test with strings",
        errors={
            "h": {
                "type": "depolarizing_error",
                "rate": 0.01,
                "num_qubits": 1,
            },
            "cx": {
                "type": "depolarizing_error",
                "rate": 0.02,
                "num_qubits": 2,
            },
        },
    )
    assert len(profile.errors) == 2
    assert SupportedGate.H in profile.errors
    assert SupportedGate.CX in profile.errors


def test_quantum_error_profile_initialization_with_gate_error_def():
    """Test QuantumErrorProfile initialization using GateErrorDef objects."""
    profile = QuantumErrorProfile(
        name="test_gate_error_def",
        description="test with GateErrorDef",
        errors={
            SupportedGate.H: GateErrorDef(
                type=SupportedErrorTypes.DEPOLARIZING_ERROR,
                rate=0.01,
                num_qubits=1,
            )
        },
    )
    assert len(profile.errors) == 1
    assert profile.errors[SupportedGate.H]["type"] == SupportedErrorTypes.DEPOLARIZING_ERROR
    assert profile.errors[SupportedGate.H]["rate"] == 0.01
    assert profile.errors[SupportedGate.H]["num_qubits"] == 1


def test_quantum_error_profile_one_and_two_qubit_gate_classification():
    """Test that gates are correctly classified as 1-qubit or 2-qubit."""
    profile = QuantumErrorProfile(
        name="test_classification",
        description="test gate classification",
        errors={
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "x": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "y": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
            "cz": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
        },
    )
    assert sorted(profile.one_qubit_gates) == ["h", "x", "y"]
    assert sorted(profile.two_qubit_gates) == ["cx", "cz"]


def test_quantum_error_profile_basis_gates_exclusion():
    """Test that specific gates are excluded from basis_gates."""
    profile = QuantumErrorProfile(
        name="test_exclusion",
        description="test basis gates exclusion",
        errors={
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "reset": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "barrier": {"type": "depolarizing_error", "rate": 0.0, "num_qubits": 1},
            "measure": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
        },
    )
    basis_gates = profile.basis_gates
    assert "h" in basis_gates
    assert "cx" in basis_gates
    assert "reset" not in basis_gates
    assert "barrier" not in basis_gates
    assert "measure" not in basis_gates


def test_quantum_error_profile_equality():
    """Test equality comparison between QuantumErrorProfile instances."""
    profile1 = QuantumErrorProfile(
        name="test1",
        description="description1",
        errors={"h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )
    profile2 = QuantumErrorProfile(
        name="test1",
        description="description1",
        errors={"h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )
    profile3 = QuantumErrorProfile(
        name="test2",
        description="description1",
        errors={"h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )

    assert profile1 == profile2
    assert profile1 != profile3
    assert profile1 != "not a profile"


def test_quantum_error_profile_hash():
    """Test that QuantumErrorProfile instances are hashable."""
    profile1 = QuantumErrorProfile(
        name="test",
        description="test",
        errors={"h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )
    profile2 = QuantumErrorProfile(
        name="test",
        description="test",
        errors={"h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )

    # Should be able to use in sets and as dict keys
    profile_set = {profile1, profile2}
    assert len(profile_set) == 1  # Same profiles should hash to same value

    profile_dict = {profile1: "value1"}
    assert profile_dict[profile2] == "value1"


def test_quantum_error_profile_immutability():
    """Test that QuantumErrorProfile instances are immutable after construction."""
    profile = QuantumErrorProfile(
        name="test",
        description="test",
        errors={"h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )

    with pytest.raises(AttributeError, match="Cannot modify immutable"):
        profile.name = "new_name"

    with pytest.raises(AttributeError, match="Cannot modify immutable"):
        profile.description = "new_description"

    with pytest.raises(AttributeError, match="Cannot modify immutable"):
        profile.errors = {}


def test_quantum_error_profile_get_summary():
    """Test get_summary method returns proper string representation."""
    profile = QuantumErrorProfile(
        name="test_summary",
        description="test summary method",
        errors={
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
        },
    )

    summary = profile.get_summary()
    assert isinstance(summary, str)
    assert "Quantum Error Profile" in summary
    assert "name: test_summary" in summary
    assert "description: test summary method" in summary
    assert "gate: h" in summary
    assert "gate: cx" in summary
    assert "type: depolarizing_error" in summary
    assert "rate: 0.01" in summary
    assert "rate: 0.02" in summary


def test_quantum_error_profile_to_json():
    """Test to_json method serialization."""
    profile = QuantumErrorProfile(
        name="test_json",
        description="test json serialization",
        errors={
            SupportedGate.H: {
                "type": SupportedErrorTypes.DEPOLARIZING_ERROR,
                "rate": 0.01,
                "num_qubits": 1,
            }
        },
    )

    json_data = profile.to_json()

    assert isinstance(json_data, dict)
    assert "version" in json_data
    assert json_data["version"] == QuantumErrorProfile._serialization_version
    assert json_data["name"] == "test_json"
    assert json_data["description"] == "test json serialization"
    assert "h" in json_data["errors"]
    assert json_data["errors"]["h"]["type"] == "depolarizing_error"
    assert json_data["errors"]["h"]["rate"] == 0.01
    assert json_data["errors"]["h"]["num_qubits"] == 1


def test_quantum_error_profile_from_json():
    """Test from_json method deserialization."""
    json_data = {
        "version": QuantumErrorProfile._serialization_version,
        "name": "test_from_json",
        "description": "test json deserialization",
        "errors": {
            "h": {
                "type": "depolarizing_error",
                "rate": 0.015,
                "num_qubits": 1,
            }
        },
    }

    profile = QuantumErrorProfile.from_json(json_data)

    assert profile.name == "test_from_json"
    assert profile.description == "test json deserialization"
    assert len(profile.errors) == 1
    assert SupportedGate.H in profile.errors
    assert profile.errors[SupportedGate.H]["rate"] == 0.015


def test_quantum_error_profile_json_roundtrip():
    """Test that to_json and from_json are inverses."""
    original = QuantumErrorProfile(
        name="roundtrip_test",
        description="test json roundtrip",
        errors={
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
        },
    )

    json_data = original.to_json()
    restored = QuantumErrorProfile.from_json(json_data)

    assert original == restored


def test_quantum_error_profile_hdf5_save_and_load():
    """Test saving and loading QuantumErrorProfile to/from HDF5."""
    profile = QuantumErrorProfile(
        name="test_hdf5",
        description="test hdf5 serialization",
        errors={
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
        },
    )

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        # Save to HDF5
        with h5py.File(temp_path, "w") as f:
            group = f.create_group("error_profile")
            profile.to_hdf5(group)

        # Load from HDF5
        with h5py.File(temp_path, "r") as f:
            loaded_profile = QuantumErrorProfile.from_hdf5(f["error_profile"])

        assert loaded_profile.name == profile.name
        assert loaded_profile.description == profile.description
        assert len(loaded_profile.errors) == len(profile.errors)

        assert loaded_profile == profile
    finally:
        Path(temp_path).unlink()


def test_quantum_error_profile_case_insensitive_gate_names():
    """Test that gate names are case-insensitive."""
    profile1 = QuantumErrorProfile(
        name="test",
        errors={"h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )
    profile2 = QuantumErrorProfile(
        name="test",
        errors={"H": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )

    # Both should create the same gate key
    assert SupportedGate.H in profile1.errors
    assert SupportedGate.H in profile2.errors


def test_quantum_error_profile_case_insensitive_error_types():
    """Test that error types are case-insensitive."""
    profile1 = QuantumErrorProfile(
        name="test",
        errors={"h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1}},
    )
    profile2 = QuantumErrorProfile(
        name="test",
        errors={"h": {"type": "DEPOLARIZING_ERROR", "rate": 0.01, "num_qubits": 1}},
    )

    # Both should create the same error type
    assert profile1.errors[SupportedGate.H]["type"] == SupportedErrorTypes.DEPOLARIZING_ERROR
    assert profile2.errors[SupportedGate.H]["type"] == SupportedErrorTypes.DEPOLARIZING_ERROR


def test_supported_error_types_enum():
    """Test SupportedErrorTypes enum values."""
    assert SupportedErrorTypes.DEPOLARIZING_ERROR == "depolarizing_error"
    assert str(SupportedErrorTypes.DEPOLARIZING_ERROR) == "depolarizing_error"

    # Test case-insensitive creation
    assert SupportedErrorTypes("depolarizing_error") == SupportedErrorTypes.DEPOLARIZING_ERROR
    assert SupportedErrorTypes("DEPOLARIZING_ERROR") == SupportedErrorTypes.DEPOLARIZING_ERROR
    assert SupportedErrorTypes("Depolarizing_Error") == SupportedErrorTypes.DEPOLARIZING_ERROR


def test_gate_error_def_structure() -> None:
    """Test GateErrorDef TypedDict structure."""
    error_def: GateErrorDef = {
        "type": SupportedErrorTypes.DEPOLARIZING_ERROR,
        "rate": 0.01,
        "num_qubits": 1,
    }

    assert error_def["type"] == SupportedErrorTypes.DEPOLARIZING_ERROR
    assert error_def["rate"] == 0.01
    assert error_def["num_qubits"] == 1


def test_quantum_error_profile_yaml_with_all_supported_gates():
    """Test YAML serialization with multiple gate types."""
    profile = QuantumErrorProfile(
        name="comprehensive",
        description="test multiple gates",
        errors={
            "h": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "x": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "y": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "z": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "cx": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
            "cz": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 2},
            "swap": {"type": "depolarizing_error", "rate": 0.03, "num_qubits": 2},
        },
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_path = f.name

    try:
        profile.to_yaml_file(temp_path)
        loaded = QuantumErrorProfile.from_yaml_file(temp_path)

        assert loaded == profile
        assert len(loaded.errors) == 7
        assert len(loaded.one_qubit_gates) == 4
        assert len(loaded.two_qubit_gates) == 3
    finally:
        Path(temp_path).unlink()


def test_noise_model_to_qdk_conversion(simple_error_profile):
    """Test conversion of QuantumErrorProfile to QDK-compatible noise configuration."""
    qdk_noise_config = simple_error_profile.to_qdk_noise_config()
    cx_err = simple_error_profile.errors["cx"]["rate"]
    h_err = simple_error_profile.errors["h"]["rate"]
    assert isinstance(qdk_noise_config, NoiseConfig)
    for component in ["x", "y", "z"]:
        assert np.isclose(
            getattr(qdk_noise_config.h, component),
            h_err / 3,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        ), f"h.{component} mismatch"

    two_qubit_components = ["ix", "iy", "iz", "xi", "xx", "xy", "xz", "yi", "yx", "yy", "yz", "zi", "zx", "zy", "zz"]
    for component in two_qubit_components:
        assert np.isclose(
            getattr(qdk_noise_config.cx, component),
            cx_err / 15,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        ), f"cx.{component} mismatch"


def test_noise_model_to_qdk_conversion_correct_gate_name():
    """Test that the conversion to QDK noise config uses correct gate names."""
    test_profile = QuantumErrorProfile(
        name="test_gate_names",
        description="test correct gate names in QDK conversion",
        errors={
            "sdg": {"type": "depolarizing_error", "rate": 0.01, "num_qubits": 1},
            "tdg": {"type": "depolarizing_error", "rate": 0.02, "num_qubits": 1},
            "sxdg": {"type": "depolarizing_error", "rate": 0.03, "num_qubits": 1},
            "measure": {"type": "depolarizing_error", "rate": 0.04, "num_qubits": 1},
        },
    )
    gate_name_mapping = {
        "sdg": "s_adj",
        "tdg": "t_adj",
        "sxdg": "sx_adj",
        "measure": "mresetz",
    }
    qdk_noise_config = test_profile.to_qdk_noise_config()
    for gate_name in ["sdg", "tdg", "sxdg", "measure"]:
        for component in ["x", "y", "z"]:
            assert np.isclose(
                getattr(getattr(qdk_noise_config, gate_name_mapping[gate_name]), component),
                test_profile.errors[gate_name]["rate"] / 3,
                atol=float_comparison_absolute_tolerance,
                rtol=float_comparison_relative_tolerance,
            ), f"{gate_name}.{component} mismatch"


def test_to_qdk_noise_config_warns_on_unsupported_gate():
    """Test that unsupported gates log a warning and are skipped."""
    # Create a profile with a gate that QDK doesn't support
    profile = QuantumErrorProfile(
        name="test",
        description="test profile",
        errors={
            SupportedGate.CRZ: GateErrorDef(
                type=SupportedErrorTypes.DEPOLARIZING_ERROR,
                rate=0.01,
                num_qubits=2,
            ),
        },
    )

    with patch("qdk_chemistry.data.noise_models.Logger") as mock_logger:
        qdk_noise_config = profile.to_qdk_noise_config()

        assert isinstance(qdk_noise_config, NoiseConfig)
        mock_logger.warn.assert_called_once()
        warning_message = mock_logger.warn.call_args[0][0]
        assert "crz" in warning_message.lower()
        assert "not supported in QDK" in warning_message
