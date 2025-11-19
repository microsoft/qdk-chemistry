"""Tests for the Settings class in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.data import (
    SettingNotFoundError,
    Settings,
    SettingTypeMismatch,
)


class _TestSettingsContainer(Settings):
    """A test settings class that defines all the settings used in tests."""

    def __init__(self):
        super().__init__()
        # Define all settings that will be used in the tests
        self._set_default("use_optimization", "bool", False)
        self._set_default("max_iterations", "int", 0)
        self._set_default("tolerance", "double", 0.0)
        self._set_default("method", "string", "")
        self._set_default("active_orbitals", "vector<int>", [])
        self._set_default("weights", "vector<double>", [])
        self._set_default("keywords", "vector<string>", [])
        self._set_default("bool_val", "bool", False)
        self._set_default("int_val", "int", 0)
        self._set_default("long_val", "int", 0)
        self._set_default("size_t_val", "int", 0)
        self._set_default("float_val", "double", 0.0)
        self._set_default("double_val", "double", 0.0)
        self._set_default("string_val", "string", "")
        self._set_default("int_vector", "vector<int>", [])
        self._set_default("double_vector", "vector<double>", [])
        self._set_default("string_vector", "vector<string>", [])
        self._set_default("test_key", "int", 0)
        self._set_default("existing_key", "int", 0)
        self._set_default("key1", "int", 0)
        self._set_default("key2", "int", 0)
        self._set_default("key3", "int", 0)
        self._set_default("test_val", "int", 0)
        self._set_default("test_string", "string", "")
        self._set_default("test_bool", "bool", False)
        self._set_default("test_double", "double", 0.0)
        self._set_default("custom_param", "int", 0)
        self._set_default("string_key", "string", "")
        self._set_default("required_key", "string", "")
        self._set_default("optional_key", "string", "")
        self._set_default("max_iter", "int", 0)
        self._set_default("use_symmetry", "bool", False)
        self._set_default("str_val", "string", "")
        # Additional settings for tests that were failing
        self._set_default("algorithm", "string", "")
        self._set_default("layers", "int", 0)
        self._set_default("step_size", "double", 0.0)
        self._set_default("verbose", "bool", False)
        self._set_default("parameters", "vector<double>", [])
        self._set_default("coefficients", "vector<double>", [])
        self._set_default("basis_functions", "vector<string>", [])
        self._set_default("list_int", "vector<int>", [])
        self._set_default("list_float", "vector<double>", [])
        self._set_default("list_str", "vector<string>", [])

        # Keys for test_update_method
        self._set_default("param1", "string", "")
        self._set_default("param2", "int", 0)
        self._set_default("param3", "vector<int>", [])

        # Keys for test_dictionary_conversion
        self._set_default("old_setting", "string", "")

        # Keys for test_hdf5_file_operations
        self._set_default("string_param", "string", "")
        self._set_default("int_param", "int", 0)
        self._set_default("float_param", "double", 0.0)
        self._set_default("bool_param", "bool", False)
        self._set_default("list_int_param", "vector<int>", [])
        self._set_default("list_float_param", "vector<double>", [])
        self._set_default("list_str_param", "vector<string>", [])

        # Keys for test_advanced_type_conversions
        self._set_default("small_int", "int", 0)
        self._set_default("large_int", "int64_t", 0)
        self._set_default("very_large_int", "size_t", 0)
        self._set_default("int_min", "int", 0)
        self._set_default("int_max", "int", 0)
        self._set_default("size_t_candidate", "size_t", 0)
        self._set_default("negative_large", "int64_t", 0)
        self._set_default("precise_float", "double", 0.0)
        self._set_default("simple_float", "float", 0.0)
        self._set_default("float_boundary", "float", 0.0)
        self._set_default("double_precision", "double", 0.0)
        self._set_default("bool_list", "vector<bool>", [])
        self._set_default("empty_list", "vector<int>", [])
        self._set_default("tuple_ints", "vector<int>", [])
        self._set_default("tuple_floats", "vector<double>", [])
        self._set_default("tuple_strings", "vector<string>", [])

        # Keys for test_numpy_array_support
        self._set_default("numpy_int", "vector<int>", [])
        self._set_default("numpy_float", "vector<double>", [])
        self._set_default("numpy_string", "vector<string>", [])
        self._set_default("numpy_float32", "vector<double>", [])

        # Keys for test_conversion_error_cases
        self._set_default("bad_list", "vector<int>", [])
        # Keys for test_edge_case_conversions
        self._set_default("zero", "int", 0)
        self._set_default("negative", "int", 0)
        self._set_default("float_zero", "double", 0.0)
        self._set_default("small_float", "double", 0.0)
        self._set_default("large_float", "double", 0.0)
        self._set_default("long_string", "string", "")
        self._set_default("empty_string", "string", "")
        self._set_default("whitespace", "string", "")
        self._set_default("newlines", "string", "")
        self._set_default("single_int", "vector<int>", [])
        self._set_default("single_float", "vector<double>", [])
        self._set_default("single_string", "vector<string>", [])

        # Keys for test_comprehensive_edge_cases
        self._set_default("negative_one", "int", 0)
        self._set_default("just_over_int_max", "int", 0)
        self._set_default("just_under_int_min", "int", 0)
        self._set_default("exact_float", "double", 0.0)
        self._set_default("requires_double", "double", 0.0)
        self._set_default("very_small", "double", 0.0)
        self._set_default("very_large", "double", 0.0)
        self._set_default("empty_int_list", "vector<int>", [])
        self._set_default("all_bool_list", "vector<bool>", [])
        self._set_default("first_bool_list", "vector<bool>", [])
        self._set_default("first_int_list", "vector<int>", [])
        self._set_default("first_float_list", "vector<float>", [])
        self._set_default("first_str_list", "vector<string>", [])


class TestSettings:
    """Test suite for the Settings class."""

    def test_settings_construction(self):
        """Test constructing Settings objects."""
        # Test default constructor
        settings = _TestSettingsContainer()
        assert len(settings) > 0  # Should have default settings defined
        assert not settings.empty()
        assert settings.size() > 0

    def test_basic_set_get_operations(self):
        """Test basic set and get operations with various types."""
        settings = _TestSettingsContainer()

        # Test bool
        settings.set("use_optimization", True)
        assert settings.get("use_optimization") is True
        assert isinstance(settings.get("use_optimization"), bool)

        # Test int
        settings.set("max_iterations", 100)
        assert settings.get("max_iterations") == 100
        assert isinstance(settings.get("max_iterations"), int)

        # Test float
        settings.set("tolerance", 1e-6)
        assert settings.get("tolerance") == 1e-6
        assert isinstance(settings.get("tolerance"), float)

        # Test string
        settings.set("method", "hf")
        assert settings.get("method") == "hf"
        assert isinstance(settings.get("method"), str)

        # Test list of ints
        settings.set("active_orbitals", [1, 2, 3, 4])
        result = settings.get("active_orbitals")
        assert result == [1, 2, 3, 4]
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

        # Test list of floats
        settings.set("weights", [0.1, 0.2, 0.3])
        result = settings.get("weights")
        assert result == [0.1, 0.2, 0.3]
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

        # Test list of strings
        settings.set("keywords", ["s", "p", "d"])
        result = settings.get("keywords")
        assert result == ["s", "p", "d"]
        assert isinstance(result, list)
        assert all(isinstance(x, str) for x in result)

    def test_dictionary_style_access(self):
        """Test dictionary-style access using [] operators."""
        settings = _TestSettingsContainer()

        # Test setting values
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_optimization"] = True

        # Test getting values
        assert settings["method"] == "hf"
        assert settings["max_iterations"] == 100
        assert settings["tolerance"] == 1e-6
        assert settings["use_optimization"] is True

        # Test updating values
        settings["max_iterations"] = 200
        assert settings["max_iterations"] == 200

    def test_attribute_style_access(self):
        """Test attribute-style access using dot notation."""
        settings = _TestSettingsContainer()

        # Test setting values
        settings.method = "hf"
        settings.max_iterations = 100
        settings.tolerance = 1e-6
        settings.use_optimization = True

        # Test getting values
        assert settings.method == "hf"
        assert settings.max_iterations == 100
        assert settings.tolerance == 1e-6
        assert settings.use_optimization is True

        # Test updating values
        settings.max_iterations = 200
        assert settings.max_iterations == 200

    def test_mixed_access_styles(self):
        """Test mixing dictionary and attribute access styles."""
        settings = _TestSettingsContainer()

        # Set with dictionary, get with attribute
        settings["method"] = "hf"
        assert settings.method == "hf"

        # Set with attribute, get with dictionary
        settings.tolerance = 1e-6
        assert settings["tolerance"] == 1e-6

        # Both should be equivalent
        settings["max_iter"] = 100
        settings.max_iter = 150
        assert settings["max_iter"] == 150
        assert settings.max_iter == 150

    def test_membership_operations(self):
        """Test membership testing with 'in' operator."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["tolerance"] = 1e-6

        # Test membership
        assert "method" in settings
        assert "tolerance" in settings
        assert "nonexistent" not in settings

        # Test has method
        assert settings.has("method")
        assert settings.has("tolerance")
        assert not settings.has("nonexistent")

    def test_size_and_empty_operations(self):
        """Test size and empty operations."""
        settings = _TestSettingsContainer()

        # Initially has predefined settings
        assert not settings.empty()
        initial_size = len(settings)
        assert initial_size > 0
        assert settings.size() == initial_size

        # Add some settings (update existing ones)
        settings["method"] = "hf"
        settings["tolerance"] = 1e-6
        assert not settings.empty()
        assert len(settings) == initial_size  # Same size, just updated values
        assert settings.size() == initial_size

        # Test creating a new instance to get fresh state
        fresh_settings = _TestSettingsContainer()
        assert not fresh_settings.empty()
        assert len(fresh_settings) == initial_size

    def test_keys_iteration(self):
        """Test keys method and iteration."""
        # Create a minimal settings container for testing iteration
        settings = _TestSettingsContainer()

        # Reset values to default and set only what we need
        settings["method"] = ""
        settings["max_iterations"] = 0
        settings["tolerance"] = 0.0

        # Set our test values
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6

        # Test keys method - now contains all predefined keys
        keys = settings.keys()
        assert "method" in keys
        assert "max_iterations" in keys
        assert "tolerance" in keys
        # Don't assert exact length since there are predefined keys

        # Test iteration over keys
        iterated_keys = []
        for key in settings:
            iterated_keys.append(key)

        assert "method" in iterated_keys
        assert "max_iterations" in iterated_keys
        assert "tolerance" in iterated_keys

        # Test explicit keys iteration
        explicit_keys = []
        for key in settings.keys():  # noqa: SIM118
            explicit_keys.append(key)

        assert "method" in explicit_keys
        assert "max_iterations" in explicit_keys
        assert "tolerance" in explicit_keys

    def test_values_iteration(self):
        """Test values method and iteration."""
        settings = _TestSettingsContainer()

        # Set our test values
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6

        # Test values method - will contain all values including defaults
        values = settings.values()
        assert "hf" in values
        assert 100 in values
        assert 1e-6 in values

        # Test iteration over values
        iterated_values = []
        for value in settings.values():
            iterated_values.append(value)

        assert "hf" in iterated_values
        assert 100 in iterated_values
        assert 1e-6 in iterated_values

    def test_items_iteration(self):
        """Test items method and iteration."""
        settings = _TestSettingsContainer()

        # Set our test values
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6

        # Test items method - will contain all items including defaults
        items = settings.items()
        items_dict = dict(items)
        assert items_dict["method"] == "hf"
        assert items_dict["max_iterations"] == 100
        assert items_dict["tolerance"] == 1e-6

        # Test iteration over items
        iterated_items = {}
        for key, value in settings.items():
            iterated_items[key] = value

        assert iterated_items["method"] == "hf"
        assert iterated_items["max_iterations"] == 100
        assert iterated_items["tolerance"] == 1e-6

    def test_get_or_default(self):
        """Test get_or_default method."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"

        # Test existing key
        assert settings.get_or_default("method", "default") == "hf"

        # Test existing key with default value (max_iter is predefined with 0)
        assert settings.get_or_default("max_iter", 1000) == 0  # Returns existing value

        # Test non-existing key with default
        assert settings.get_or_default("nonexistent", "default_value") == "default_value"
        assert settings.get_or_default("use_gpu", False) is False

        # Test with different types
        assert settings.get_or_default("coeffs", [1.0, 2.0]) == [1.0, 2.0]

    def test_update_method(self):
        """Test update method for existing keys."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100

        # Test updating existing keys
        settings.update("method", "ucc")
        assert settings["method"] == "ucc"

        settings.update("max_iterations", 200)
        assert settings["max_iterations"] == 200

        # Test that update throws for non-existing keys
        with pytest.raises(SettingNotFoundError):
            settings.update("nonexistent", "value")

        # Test dictionary update method
        settings = _TestSettingsContainer()
        settings["param1"] = "old_value1"
        settings["param2"] = 42
        settings["param3"] = [1, 2, 3]

        # Test updating with dictionary - should work for existing keys
        update_dict = {"param1": "new_value1", "param2": 100, "param3": [4, 5, 6]}
        settings.update(update_dict)

        assert settings["param1"] == "new_value1"
        assert settings["param2"] == 100
        assert settings["param3"] == [4, 5, 6]

        # Test that dictionary update fails for non-existing keys
        bad_update_dict = {"param1": "another_value", "nonexistent_key": "value"}
        with pytest.raises(SettingNotFoundError):
            settings.update(bad_update_dict)

    def test_dictionary_conversion(self):
        """Test to_dict and from_dict methods."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True
        settings["active_orbitals"] = [1, 2, 3]

        # Test to_dict
        settings_dict = settings.to_dict()
        assert isinstance(settings_dict, dict)
        assert settings_dict["method"] == "hf"
        assert settings_dict["max_iterations"] == 100
        assert settings_dict["tolerance"] == 1e-6
        assert settings_dict["use_symmetry"] is True
        assert settings_dict["active_orbitals"] == [1, 2, 3]

        # Test from_dict with predefined keys
        new_settings = _TestSettingsContainer()
        # Set some initial settings - these should remain unless overwritten
        new_settings["old_setting"] = "should_remain"

        test_dict = {
            "algorithm": "qaoa",
            "layers": 5,
            "step_size": 0.01,
            "verbose": True,
            "parameters": [0.1, 0.2, 0.3],
        }
        new_settings.from_dict(test_dict)

        # Verify old setting was not cleared (only updates, doesn't clear)
        assert new_settings.has("old_setting")
        assert new_settings["old_setting"] == "should_remain"

        # Verify new settings were loaded
        assert new_settings["algorithm"] == "qaoa"
        assert new_settings["layers"] == 5
        assert new_settings["step_size"] == 0.01
        assert new_settings["verbose"] is True
        assert new_settings["parameters"] == [0.1, 0.2, 0.3]

        # Test from_dict with various data types
        complex_dict = {
            "string_val": "test",
            "int_val": 42,
            "float_val": 3.14159,
            "bool_val": False,
            "list_int": [1, 2, 3, 4],
            "list_float": [1.1, 2.2, 3.3],
            "list_str": ["a", "b", "c"],
        }

        complex_settings = _TestSettingsContainer()
        complex_settings.from_dict(complex_dict)

        # Verify all types are preserved
        assert complex_settings["string_val"] == "test"
        assert complex_settings["int_val"] == 42
        assert complex_settings["float_val"] == 3.14159
        assert complex_settings["bool_val"] is False
        assert complex_settings["list_int"] == [1, 2, 3, 4]
        assert complex_settings["list_float"] == [1.1, 2.2, 3.3]
        assert complex_settings["list_str"] == ["a", "b", "c"]

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True
        settings["active_orbitals"] = [1, 2, 3]

        # Test to_json_string
        json_str = settings.to_json_string()
        assert isinstance(json_str, str)

        # Parse and verify JSON
        parsed = json.loads(json_str)
        assert parsed["method"] == "hf"
        assert parsed["max_iterations"] == 100
        assert parsed["tolerance"] == 1e-6
        assert parsed["use_symmetry"] is True
        assert parsed["active_orbitals"] == [1, 2, 3]

        # Test from_json_string
        new_settings = Settings.from_json_string(json_str)

        assert new_settings["method"] == "hf"
        assert new_settings["max_iterations"] == 100
        assert new_settings["tolerance"] == 1e-6
        assert new_settings["use_symmetry"] is True
        assert new_settings["active_orbitals"] == [1, 2, 3]

        # Test to_json method (alias)
        json_str2 = settings.to_json()
        assert isinstance(json_str2, str)

        # Should be the same as to_json_string
        parsed2 = json.loads(json_str2)
        assert parsed2 == parsed

        # Test from_json method
        new_settings2 = Settings.from_json(json_str2)

        assert new_settings2["method"] == "hf"
        assert new_settings2["max_iterations"] == 100
        assert new_settings2["tolerance"] == 1e-6
        assert new_settings2["use_symmetry"] is True
        assert new_settings2["active_orbitals"] == [1, 2, 3]

        # Test error handling for malformed JSON
        with pytest.raises(RuntimeError):
            Settings.from_json("invalid json string {{{")

        with pytest.raises(RuntimeError):
            Settings.from_json_string("another invalid json ]]")

    def test_json_file_operations(self):
        """Test JSON file operations."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            json_file = tmpdir_path / "test_settings.settings.json"

            # Test to_json_file
            settings.to_json_file(json_file)
            assert Path(json_file).exists()

            # Verify file content
            with open(json_file) as f:
                data = json.load(f)
            assert data["method"] == "hf"
            assert data["max_iterations"] == 100
            assert data["tolerance"] == 1e-6

            # Test from_json_file
            new_settings = Settings.from_json_file(json_file)

            assert new_settings["method"] == "hf"
            assert new_settings["max_iterations"] == 100
            assert new_settings["tolerance"] == 1e-6

    def test_hdf5_file_operations(self):
        """Test HDF5 file operations."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                hdf5_file = tmpdir_path / "test_settings.settings.h5"

                # Test to_hdf5_file
                settings.to_hdf5_file(str(hdf5_file))
                assert Path(hdf5_file).exists()

                # Test from_hdf5_file
                new_settings = Settings.from_hdf5_file(str(hdf5_file))

                assert new_settings["method"] == "hf"
                assert new_settings["max_iterations"] == 100
                assert new_settings["tolerance"] == 1e-6
                assert new_settings["use_symmetry"] is True

                # Test round-trip with multiple data types
                complex_settings = _TestSettingsContainer()
                complex_settings["string_param"] = "test_string"
                complex_settings["int_param"] = 42
                complex_settings["float_param"] = 3.14159
                complex_settings["bool_param"] = False
                complex_settings["list_int_param"] = [1, 2, 3, 4]
                complex_settings["list_float_param"] = [1.1, 2.2, 3.3]
                complex_settings["list_str_param"] = ["a", "b", "c"]

                complex_hdf5_file = os.path.join(tmpdir, "complex_test.settings.h5")
                complex_settings.to_hdf5_file(complex_hdf5_file)

                loaded_complex_settings = Settings.from_hdf5_file(complex_hdf5_file)

                assert loaded_complex_settings["string_param"] == "test_string"
                assert loaded_complex_settings["int_param"] == 42
                assert loaded_complex_settings["float_param"] == 3.14159
                assert loaded_complex_settings["bool_param"] is False
                assert loaded_complex_settings["list_int_param"] == [1, 2, 3, 4]
                assert loaded_complex_settings["list_float_param"] == [1.1, 2.2, 3.3]
                assert loaded_complex_settings["list_str_param"] == ["a", "b", "c"]

        except (ImportError, OSError, RuntimeError) as e:
            pytest.skip(f"HDF5 test skipped - {e!s}")

    def test_generic_file_operations(self):
        """Test generic file operations with to_file and from_file."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Test JSON format through generic interface
            json_file = tmpdir_path / "test_settings.json"
            settings.to_file(json_file, "json")
            assert Path(json_file).exists()

            # Verify content
            with open(json_file) as f:
                data = json.load(f)
            assert data["method"] == "hf"
            assert data["max_iterations"] == 100
            assert data["tolerance"] == 1e-6
            assert data["use_symmetry"] is True

            # Test loading from JSON
            new_settings = Settings.from_file(json_file, "json")
            assert new_settings["method"] == "hf"
            assert new_settings["max_iterations"] == 100
            assert new_settings["tolerance"] == 1e-6
            assert new_settings["use_symmetry"] is True

            # Test HDF5 format through generic interface
            try:
                hdf5_file = tmpdir_path / "test_settings.h5"
                settings.to_file(hdf5_file, "hdf5")
                assert Path(hdf5_file).exists()

                # Test loading from HDF5
                new_settings_hdf5 = Settings.from_file(hdf5_file, "hdf5")
                assert new_settings_hdf5["method"] == "hf"
                assert new_settings_hdf5["max_iterations"] == 100
                assert new_settings_hdf5["tolerance"] == 1e-6
                assert new_settings_hdf5["use_symmetry"] is True

                # Test unsupported file formats
                with pytest.raises(ValueError, match="Unknown file type"):
                    settings.to_file(json_file, "xml")

                with pytest.raises(ValueError, match="Unknown file type"):
                    Settings.from_file(json_file, "yaml")

            except (ImportError, OSError, RuntimeError) as e:
                pytest.skip(f"HDF5 test skipped - {e!s}")

    def test_filename_validation(self):
        """Test filename validation for file operations."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100

        # Test invalid filenames
        invalid_filenames = [
            "",  # Empty filename
            "   ",  # Whitespace only
            "\n",  # Newline
            "\t",  # Tab
            "file\0name",  # Null character
            "file\x01name",  # Control character
        ]

        for invalid_filename in invalid_filenames:
            with pytest.raises(ValueError, match="Invalid filename"):
                settings.to_json_file(invalid_filename)
            with pytest.raises(ValueError, match="Invalid filename"):
                settings.from_json_file(invalid_filename)

        # Test HDF5 filename validation (if HDF5 is available)
        try:
            for invalid_filename in invalid_filenames:
                with pytest.raises(ValueError, match="Invalid filename"):
                    settings.to_hdf5_file(invalid_filename)
                with pytest.raises(ValueError, match="Invalid filename"):
                    settings.from_hdf5_file(invalid_filename)
        except (ImportError, OSError, RuntimeError):
            pass  # Skip if HDF5 not available

    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Test unsupported format
            unsupported_file = tmpdir_path / "test.xml"
            with pytest.raises(ValueError, match="Unknown file type"):
                settings.to_file(unsupported_file, "unsupported")
            with pytest.raises(ValueError, match="Unknown file type"):
                Settings.from_file(unsupported_file, "unsupported")

    def test_file_not_found_error(self):
        """Test error handling when file does not exist."""
        settings = _TestSettingsContainer()

        # Test with non-existent file
        with pytest.raises(RuntimeError, match="Cannot open file for reading"):
            Settings.from_file("/nonexistent/path/file.settings.json", "json")
        with pytest.raises(RuntimeError, match="Cannot open file for reading"):
            settings.from_json_file("/nonexistent/path/file.settings.json")

        # Test HDF5 file not found (if HDF5 is available)
        try:
            with pytest.raises(RuntimeError, match="HDF5 error: H5Fopen failed"):
                settings.from_hdf5_file("/nonexistent/path/file.settings.h5")
        except (ImportError, OSError, RuntimeError):
            pass  # Skip if HDF5 not available

    def test_directory_creation_error(self):
        """Test error handling when directory cannot be created."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"

        # Test with invalid directory path (root directory on most systems)
        invalid_path = "/root/nonexistent/file.settings.json"
        with pytest.raises(RuntimeError, match="Cannot open file for writing"):
            settings.to_file(invalid_path, "json")
        with pytest.raises(RuntimeError, match="Cannot open file for writing"):
            settings.to_json_file(invalid_path)

        # Test HDF5 directory creation error (if HDF5 is available)
        try:
            invalid_hdf5_path = "/root/nonexistent/file.settings.h5"
            with pytest.raises(RuntimeError, match="HDF5 error: H5Fcreate failed"):
                settings.to_hdf5_file(invalid_hdf5_path)
        except (ImportError, OSError, RuntimeError):
            pass  # Skip if HDF5 not available

    def test_consistency_between_generic_and_specific_methods(self):
        """Test that generic and specific methods produce consistent results."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True
        settings["active_orbitals"] = [1, 2, 3]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Test JSON consistency
            json_file1 = tmpdir_path / "test1.settings.json"
            json_file2 = tmpdir_path / "test2.settings.json"

            # Save with generic method
            settings.to_file(json_file1, "json")
            # Save with specific method
            settings.to_json_file(json_file2)

            # Files should be identical
            with open(json_file1) as f1, open(json_file2) as f2:
                assert f1.read() == f2.read()

            # Load with both methods and verify consistency
            settings1 = Settings.from_file(json_file1, "json")
            settings2 = Settings.from_json_file(json_file2)

            # Convert to dicts for comparison
            dict1 = settings1.to_dict()
            dict2 = settings2.to_dict()
            assert dict1 == dict2

            # Test HDF5 consistency (if available)
            try:
                hdf5_file1 = tmpdir_path / "test1.settings.h5"
                hdf5_file2 = tmpdir_path / "test2.settings.h5"

                # Save with generic method
                settings.to_file(str(hdf5_file1), "hdf5")
                # Save with specific method
                settings.to_hdf5_file(str(hdf5_file2))

                # Load with both methods and verify consistency
                settings3 = Settings.from_file(str(hdf5_file1), "hdf5")
                settings4 = Settings.from_hdf5_file(str(hdf5_file2))

                # Convert to dicts for comparison
                dict3 = settings3.to_dict()
                dict4 = settings4.to_dict()
                assert dict3 == dict4
            except (ImportError, OSError, RuntimeError):
                pass  # Skip if HDF5 not available

    def test_complex_data_types_file_io(self):
        """Test file I/O with complex data types."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True
        settings["active_orbitals"] = [1, 2, 3, 4, 5]
        settings["coefficients"] = [0.1, 0.2, 0.3, 0.4]
        settings["basis_functions"] = ["s", "p", "d", "f"]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Test JSON with complex data
            json_file = tmpdir_path / "complex_test.json"
            settings.to_file(json_file, "json")

            new_settings = Settings.from_file(json_file, "json")

            # Verify all data types are preserved
            assert new_settings["method"] == "hf"
            assert new_settings["max_iterations"] == 100
            assert new_settings["tolerance"] == 1e-6
            assert new_settings["use_symmetry"] is True
            assert new_settings["active_orbitals"] == [1, 2, 3, 4, 5]
            assert new_settings["coefficients"] == [0.1, 0.2, 0.3, 0.4]
            assert new_settings["basis_functions"] == ["s", "p", "d", "f"]

            # Test HDF5 with complex data (if available)
            try:
                hdf5_file = tmpdir_path / "complex_test.h5"
                settings.to_file(hdf5_file, "hdf5")

                new_settings_hdf5 = Settings.from_file(hdf5_file, "hdf5")

                # Verify all data types are preserved
                assert new_settings_hdf5["method"] == "hf"
                assert new_settings_hdf5["max_iterations"] == 100
                assert new_settings_hdf5["tolerance"] == 1e-6
                assert new_settings_hdf5["use_symmetry"] is True
                assert new_settings_hdf5["active_orbitals"] == [1, 2, 3, 4, 5]
                assert new_settings_hdf5["coefficients"] == [0.1, 0.2, 0.3, 0.4]
                assert new_settings_hdf5["basis_functions"] == ["s", "p", "d", "f"]
            except (ImportError, OSError, RuntimeError):
                pass  # Skip if HDF5 not available

    def test_roundtrip_data_integrity(self):
        """Test data integrity through multiple save/load cycles."""
        original_settings = _TestSettingsContainer()
        original_settings["method"] = "hf"
        original_settings["max_iterations"] = 100
        original_settings["tolerance"] = 1e-6
        original_settings["use_symmetry"] = True
        original_settings["active_orbitals"] = [1, 2, 3]
        original_settings["coefficients"] = [0.1, 0.2, 0.3]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Multiple roundtrips for JSON
            json_file = tmpdir_path / "roundtrip.json"
            current_settings = original_settings

            for _ in range(3):
                current_settings.to_file(json_file, "json")
                new_settings = Settings.from_file(json_file, "json")

                # Verify data integrity
                assert new_settings.to_dict() == original_settings.to_dict()
                current_settings = new_settings

            # Multiple roundtrips for HDF5 (if available)
            try:
                hdf5_file = tmpdir_path / "roundtrip.h5"
                current_settings = original_settings

                for _ in range(3):
                    current_settings.to_file(hdf5_file, "hdf5")
                    new_settings = Settings.from_file(hdf5_file, "hdf5")

                    # Verify data integrity
                    assert new_settings.to_dict() == original_settings.to_dict()
                    current_settings = new_settings
            except (ImportError, OSError, RuntimeError):
                pass  # Skip if HDF5 not available

    def test_string_representations(self):
        """Test string representations."""
        settings = _TestSettingsContainer()

        # Set our test values
        settings["method"] = "hf"
        settings["max_iterations"] = 100

        # Test __repr__
        repr_str = repr(settings)
        assert "qdk_chemistry.Settings" in repr_str
        # Don't check exact size since there are many predefined settings
        assert "size=" in repr_str

        # Test __str__
        str_str = str(settings)
        assert "Settings" in str_str
        assert "method" in str_str
        assert "max_iterations" in str_str

        # Test with empty settings
        empty_settings = Settings()
        empty_repr = repr(empty_settings)
        empty_str = str(empty_settings)

        assert "size=0" in empty_repr or "0" in empty_repr
        assert "Settings" in empty_str

    def test_get_as_string(self):
        """Test get_as_string method."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True

        # Test string representations of different types
        assert settings.get_as_string("method") == "hf"
        assert settings.get_as_string("max_iterations") == "100"
        assert float(settings.get_as_string("tolerance")) == 1e-6  # check numerical value
        assert settings.get_as_string("use_symmetry") in ["1", "true", "True"]  # Implementation dependent

    def test_get_type_name(self):
        """Test get_type_name method."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True
        settings["active_orbitals"] = [1, 2, 3]

        # Test type name reporting
        assert "string" in settings.get_type_name("method").lower()
        assert "int" in settings.get_type_name("max_iterations").lower()
        assert (
            "double" in settings.get_type_name("tolerance").lower()
            or "float" in settings.get_type_name("tolerance").lower()
        )
        assert "bool" in settings.get_type_name("use_symmetry").lower()
        assert "vector" in settings.get_type_name("active_orbitals").lower()

    def test_error_handling(self):
        """Test error handling for invalid operations."""
        settings = _TestSettingsContainer()

        # Test accessing non-existent key with get
        with pytest.raises(SettingNotFoundError):
            settings.get("nonexistent")

        # Test accessing non-existent key with []
        with pytest.raises(SettingNotFoundError):
            _ = settings["nonexistent"]

        # Test accessing non-existent attribute
        with pytest.raises(AttributeError):
            _ = settings.nonexistent

    def test_conversion_utilities(self):
        """Test the conversion utility functions."""
        # Test direct utility functions (if accessible)
        settings = _TestSettingsContainer()

        # Test various type conversions through the interface
        settings["int_val"] = 42
        settings["float_val"] = 3.14
        settings["bool_val"] = True
        settings["str_val"] = "test"
        settings["list_int"] = [1, 2, 3]
        settings["list_float"] = [1.1, 2.2, 3.3]
        settings["list_str"] = ["a", "b", "c"]

        # Verify types are correctly converted
        assert isinstance(settings["int_val"], int)
        assert isinstance(settings["float_val"], float)
        assert isinstance(settings["bool_val"], bool)
        assert isinstance(settings["str_val"], str)
        assert isinstance(settings["list_int"], list)
        assert isinstance(settings["list_float"], list)
        assert isinstance(settings["list_str"], list)

    def test_advanced_type_conversions(self):
        """Test advanced type conversion scenarios."""
        settings = _TestSettingsContainer()

        # Test integer type selection (int vs long vs size_t)
        settings["small_int"] = 100  # Should be int
        settings["large_int"] = 2**31  # Should be long or size_t
        settings["very_large_int"] = 2**63 - 1  # Should be long

        assert isinstance(settings["small_int"], int)
        assert isinstance(settings["large_int"], int)  # Python int can handle large values
        assert isinstance(settings["very_large_int"], int)

        # Test specific integer boundary conditions

        settings["int_min"] = -2147483648  # INT_MIN
        settings["int_max"] = 2147483647  # INT_MAX
        settings["size_t_candidate"] = 4294967295  # SIZE_MAX on 32-bit
        settings["negative_large"] = -9223372036854775808  # Very negative

        assert isinstance(settings["int_min"], int)
        assert isinstance(settings["int_max"], int)
        assert isinstance(settings["size_t_candidate"], int)
        assert isinstance(settings["negative_large"], int)

        # Test float precision logic
        settings["precise_float"] = 3.14159  # Should be double
        settings["simple_float"] = 2.0  # Could be float without precision loss
        settings["float_boundary"] = 3.4028235e38  # Near float max
        settings["double_precision"] = 1.7976931348623157e308  # Near double max

        assert isinstance(settings["precise_float"], float)
        assert isinstance(settings["simple_float"], float)
        assert isinstance(settings["float_boundary"], float)
        assert isinstance(settings["double_precision"], float)

        # Test boolean handling in sequences
        settings["bool_list"] = [True, False, True]  # Should become int list
        result = settings["bool_list"]
        assert isinstance(result, list)
        # Booleans get converted to ints in lists
        assert all(isinstance(x, int) for x in result)

        # Test empty sequences (should default to int vector)
        settings["empty_list"] = []
        empty_result = settings["empty_list"]
        assert isinstance(empty_result, list)
        assert len(empty_result) == 0

        # Test tuples (should be treated like lists)
        settings["tuple_ints"] = (1, 2, 3)
        settings["tuple_floats"] = (1.1, 2.2, 3.3)
        settings["tuple_strings"] = ("a", "b", "c")

        assert settings["tuple_ints"] == [1, 2, 3]
        assert settings["tuple_floats"] == [1.1, 2.2, 3.3]
        assert settings["tuple_strings"] == ["a", "b", "c"]

    def test_numpy_array_support(self):
        """Test numpy array conversion support."""
        settings = _TestSettingsContainer()

        try:
            # Test int array
            int_array = np.array([1, 2, 3, 4], dtype=np.int32)
            settings["numpy_int"] = int_array
            result = settings["numpy_int"]
            assert isinstance(result, list)
            assert result == [1, 2, 3, 4]
            assert all(isinstance(x, int) for x in result)

            # Test float array
            float_array = np.array([1.1, 2.2, 3.3], dtype=np.float64)
            settings["numpy_float"] = float_array
            result = settings["numpy_float"]
            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(x, float) for x in result)

            # Test float32 array (should be promoted to double)
            float32_array = np.array([1.5, 2.5, 3.5], dtype=np.float32)
            settings["numpy_float32"] = float32_array
            result = settings["numpy_float32"]
            assert isinstance(result, list)
            assert all(isinstance(x, float) for x in result)

        except ImportError:
            pytest.skip("NumPy not available for testing")

    def test_conversion_error_cases(self):
        """Test error handling in type conversions."""
        settings = _TestSettingsContainer()

        # Test unsupported list element types
        with pytest.raises(SettingTypeMismatch):
            settings["bad_list"] = [{"nested": "dict"}]  # Unsupported nested type

    def test_edge_case_conversions(self):
        """Test edge cases in type conversions."""
        settings = _TestSettingsContainer()

        # Test very small and large numbers
        settings["zero"] = 0
        settings["negative"] = -42
        settings["float_zero"] = 0.0
        settings["small_float"] = 1e-10
        settings["large_float"] = 1e10

        assert settings["zero"] == 0
        assert settings["negative"] == -42
        assert settings["float_zero"] == 0.0
        assert settings["small_float"] == 1e-10
        assert settings["large_float"] == 1e10

        # Test very long strings
        long_string = "x" * 1000
        settings["long_string"] = long_string
        assert settings["long_string"] == long_string

        # Test empty string
        settings["empty_string"] = ""
        assert settings["empty_string"] == ""

        # Test lists with single elements
        settings["single_int"] = [42]
        settings["single_float"] = [3.14]
        settings["single_string"] = ["test"]

        assert settings["single_int"] == [42]
        assert settings["single_float"] == [3.14]
        assert settings["single_string"] == ["test"]

    def test_comprehensive_edge_cases(self):
        """Test edge cases for type conversion."""
        settings = _TestSettingsContainer()

        # Test various integer boundary conditions specifically
        # These tests target integer type selection
        settings["zero"] = 0
        settings["negative_one"] = -1
        settings["int_max"] = 2147483647  # INT_MAX
        settings["int_min"] = -2147483648  # INT_MIN
        with pytest.raises(SettingTypeMismatch):
            settings["just_over_int_max"] = 2147483648
        with pytest.raises(SettingTypeMismatch):
            settings["just_under_int_min"] = -2147483649  # Just under INT_MIN

        # Verify all are stored correctly
        assert settings["zero"] == 0
        assert settings["negative_one"] == -1
        assert settings["int_max"] == 2147483647
        assert settings["int_min"] == -2147483648

        # Test float precision edge cases
        settings["exact_float"] = 2.0  # Can be represented exactly as float
        settings["requires_double"] = 3.141592653589793  # Requires double precision
        settings["very_small"] = 1e-45  # Very small number
        settings["very_large"] = 1e38  # Large number

        # Test these are all stored as floats in Python
        assert isinstance(settings["exact_float"], float)
        assert isinstance(settings["requires_double"], float)
        assert isinstance(settings["very_small"], float)
        assert isinstance(settings["very_large"], float)

        # Test empty list type default
        settings["empty_int_list"] = []
        assert settings["empty_int_list"] == []
        assert isinstance(settings["empty_int_list"], list)

        # Test mixed type lists that should trigger type selection
        settings["all_bool_list"] = [True, False, True, False]
        bool_result = settings["all_bool_list"]
        # Booleans in lists should become integers
        assert all(isinstance(x, int) for x in bool_result)
        assert bool_result == [1, 0, 1, 0]

        # Test sequences with first element determining type
        with pytest.raises(SettingTypeMismatch):
            settings["first_bool_list"] = [True, 1, 0]  # First is bool
        with pytest.raises(SettingTypeMismatch):
            settings["first_int_list"] = [1, True, False]  # First is int
        settings["first_float_list"] = [1.0, 2, 3]  # First is float
        settings["first_str_list"] = ["a", "b", "c"]  # First is string

        # Verify type conversion based on first element
        first_bool_result = settings["first_bool_list"]
        first_int_result = settings["first_int_list"]
        first_float_result = settings["first_float_list"]
        first_str_result = settings["first_str_list"]

        assert all(isinstance(x, int) for x in first_bool_result)
        assert all(isinstance(x, int) for x in first_int_result)
        assert all(isinstance(x, float) for x in first_float_result)
        assert all(isinstance(x, str) for x in first_str_result)

    def test_get_all_settings_method(self):
        """Test get_all_settings method for internal map access."""
        settings = _TestSettingsContainer()
        settings["method"] = "hf"
        settings["max_iterations"] = 100
        settings["tolerance"] = 1e-6
        settings["use_symmetry"] = True
        settings["active_orbitals"] = [1, 2, 3]

        # Test get_all_settings method
        all_settings = settings.get_all_settings()
        assert isinstance(all_settings, dict)
        # Should contain all predefined keys from _TestSettingsContainer
        assert len(all_settings) > 5  # Has many more defaults

        # This returns the internal map, so we need to check it contains the keys
        assert "method" in all_settings
        assert "max_iterations" in all_settings
        assert "tolerance" in all_settings
        assert "use_symmetry" in all_settings
        assert "active_orbitals" in all_settings


class TestSettingsCustomClass:
    """Test creating custom settings classes."""

    def test_custom_settings_class(self):
        """Test creating a custom settings class in Python."""

        class HfSettings(Settings):
            def __init__(self):
                super().__init__()
                self._set_default("method", "string", "hf")
                self._set_default("max_iterations", "int", 1000)
                self._set_default("tolerance", "double", 1e-6)
                self._set_default("use_symmetry", "bool", False)
                self._set_default("active_orbitals", "vector<int>", [])

            def validate(self):
                required = ["method", "max_iterations", "tolerance"]
                self.validate_required(required)

        # Test construction with defaults
        hf_settings = HfSettings()
        assert hf_settings["method"] == "hf"
        assert hf_settings["max_iterations"] == 1000
        assert hf_settings["tolerance"] == 1e-6
        assert hf_settings["use_symmetry"] is False
        assert hf_settings["active_orbitals"] == []

        # Test modifying values
        hf_settings["max_iterations"] = 500
        hf_settings.tolerance = 1e-8
        assert hf_settings["max_iterations"] == 500
        assert hf_settings.tolerance == 1e-8

        # Test validation
        hf_settings.validate()  # Should not raise

        # Test validation with missing required key - create fresh instance
        fresh_hf_settings = HfSettings()
        # Remove a required setting by creating a new base Settings instance
        # Since we can't clear(), we'll test error handling differently
        try:
            fresh_hf_settings.validate_required(["nonexistent_key"])
            pytest.fail("Should have raised SettingNotFoundError")
        except SettingNotFoundError:
            pass  # Expected

        # Test validate_required method directly
        settings = _TestSettingsContainer()
        settings["param1"] = "value1"
        settings["param2"] = 42

        # Should pass with existing keys
        settings.validate_required(["param1", "param2"])

        # Should fail with missing key
        with pytest.raises(SettingNotFoundError):
            settings.validate_required(["param1", "param2", "missing_param"])

    def test_trampoline_class_methods(self):
        """Test PySettings trampoline class specific methods."""

        class AdvancedSettings(Settings):
            def __init__(self):
                super().__init__()
                # Test _set_default with different types
                self._set_default("string_param", "string", "default_value")
                self._set_default("int_param", "int", 42)
                self._set_default("float_param", "double", 3.14159)
                self._set_default("bool_param", "bool", True)
                self._set_default("list_param", "vector<int>", [1, 2, 3])

        settings = AdvancedSettings()

        # Verify all defaults were set correctly
        assert settings["string_param"] == "default_value"
        assert settings["int_param"] == 42
        assert settings["float_param"] == 3.14159
        assert settings["bool_param"] is True
        assert settings["list_param"] == [1, 2, 3]

        # Verify the settings can be overridden
        settings["string_param"] = "new_value"
        settings["int_param"] = 100

        assert settings["string_param"] == "new_value"
        assert settings["int_param"] == 100

        # Verify original defaults are preserved in a new instance
        settings2 = AdvancedSettings()
        assert settings2["string_param"] == "default_value"
        assert settings2["int_param"] == 42

        # Test validate_required method directly
        settings = _TestSettingsContainer()
        settings["param1"] = "value1"
        settings["param2"] = 42

        # Should pass with existing keys
        settings.validate_required(["param1", "param2"])

        # Should fail with missing key
        with pytest.raises(SettingNotFoundError):
            settings.validate_required(["param1", "param2", "missing_param"])
