"""Comprehensive tests for the ConfigurationSet class in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import pickle

import pytest

from qdk_chemistry.data import (
    Configuration,
    ConfigurationSet,
)

from .test_helpers import create_test_orbitals


class TestConfigurationSetBasics:
    """Test ConfigurationSet basic functionality."""

    @pytest.fixture
    def test_orbitals(self):
        """Create test orbitals with 4 spatial orbitals."""
        return create_test_orbitals(4)

    @pytest.fixture
    def test_configurations(self):
        """Create a list of test configurations."""
        return [
            Configuration("2200"),  # doubly occupied first two orbitals
            Configuration("2ud0"),  # mixed occupation
            Configuration("2du0"),  # mixed occupation (different)
            Configuration("udud"),  # all singly occupied
        ]

    @pytest.fixture
    def test_configuration_set(self, test_configurations, test_orbitals):
        """Create a test ConfigurationSet."""
        return ConfigurationSet(test_configurations, test_orbitals)

    def test_construction(self, test_configurations, test_orbitals):
        """Test ConfigurationSet construction."""
        config_set = ConfigurationSet(test_configurations, test_orbitals)
        assert config_set is not None
        assert len(config_set) == 4

    def test_construction_empty(self, test_orbitals):
        """Test ConfigurationSet construction with empty list."""
        config_set = ConfigurationSet([], test_orbitals)
        assert config_set is not None
        assert len(config_set) == 0
        assert config_set.empty()

    def test_construction_single(self, test_orbitals):
        """Test ConfigurationSet construction with single configuration."""
        configs = [Configuration("2200")]
        config_set = ConfigurationSet(configs, test_orbitals)
        assert len(config_set) == 1
        assert not config_set.empty()

    def test_len(self, test_configuration_set):
        """Test __len__ method."""
        assert len(test_configuration_set) == 4

    def test_empty(self, test_configuration_set, test_orbitals):
        """Test empty method."""
        assert not test_configuration_set.empty()
        empty_set = ConfigurationSet([], test_orbitals)
        assert empty_set.empty()

    def test_getitem(self, test_configuration_set, test_configurations):
        """Test __getitem__ access."""
        for i, expected_config in enumerate(test_configurations):
            config = test_configuration_set[i]
            assert config.to_string() == expected_config.to_string()

    def test_getitem_out_of_range(self, test_configuration_set):
        """Test __getitem__ raises IndexError for out of range."""
        with pytest.raises(IndexError):
            _ = test_configuration_set[100]

    def test_at(self, test_configuration_set, test_configurations):
        """Test at method with bounds checking."""
        config = test_configuration_set.at(0)
        assert config.to_string() == test_configurations[0].to_string()

    def test_iteration(self, test_configuration_set, test_configurations):
        """Test iteration over ConfigurationSet."""
        configs_from_iter = list(test_configuration_set)
        assert len(configs_from_iter) == len(test_configurations)
        for i, config in enumerate(configs_from_iter):
            assert config.to_string() == test_configurations[i].to_string()

    def test_get_configurations(self, test_configuration_set, test_configurations):
        """Test configurations property."""
        configs = test_configuration_set.configurations
        assert len(configs) == len(test_configurations)
        for i, config in enumerate(configs):
            assert config.to_string() == test_configurations[i].to_string()

    def test_get_orbitals(self, test_configuration_set, test_orbitals):
        """Test orbitals property."""
        orbitals = test_configuration_set.orbitals
        assert orbitals is not None
        assert orbitals.get_num_molecular_orbitals() == test_orbitals.get_num_molecular_orbitals()

    def test_get_summary(self, test_configuration_set):
        """Test summary property."""
        summary = test_configuration_set.summary
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_repr(self, test_configuration_set):
        """Test __repr__ method."""
        repr_str = repr(test_configuration_set)
        assert isinstance(repr_str, str)

    def test_str(self, test_configuration_set):
        """Test __str__ method."""
        str_repr = str(test_configuration_set)
        assert isinstance(str_repr, str)


class TestConfigurationSetEquality:
    """Test ConfigurationSet equality comparisons."""

    @pytest.fixture
    def test_orbitals(self):
        """Create test orbitals."""
        return create_test_orbitals(4)

    def test_equality_same(self, test_orbitals):
        """Test equality for identical ConfigurationSets."""
        configs = [Configuration("2200"), Configuration("2ud0")]
        set1 = ConfigurationSet(configs, test_orbitals)
        set2 = ConfigurationSet(configs, test_orbitals)
        # Note: equality checks if orbitals point to the same object
        # and if configurations are equal
        assert set1 == set2  # Sets with same configs and orbitals should be equal

    def test_inequality_different_configs(self, test_orbitals):
        """Test inequality for different configurations."""
        configs1 = [Configuration("2200")]
        configs2 = [Configuration("2ud0")]
        set1 = ConfigurationSet(configs1, test_orbitals)
        set2 = ConfigurationSet(configs2, test_orbitals)
        assert set1 != set2

    def test_inequality_different_size(self, test_orbitals):
        """Test inequality for different sizes."""
        configs1 = [Configuration("2200")]
        configs2 = [Configuration("2200"), Configuration("2ud0")]
        set1 = ConfigurationSet(configs1, test_orbitals)
        set2 = ConfigurationSet(configs2, test_orbitals)
        assert set1 != set2


class TestConfigurationSetSerialization:
    """Test ConfigurationSet serialization and deserialization."""

    @pytest.fixture
    def test_orbitals(self):
        """Create test orbitals."""
        return create_test_orbitals(4)

    @pytest.fixture
    def test_configuration_set(self, test_orbitals):
        """Create a test ConfigurationSet."""
        configs = [
            Configuration("2200"),
            Configuration("2ud0"),
            Configuration("udud"),
        ]
        return ConfigurationSet(configs, test_orbitals)

    def test_json_serialization(self, test_configuration_set):
        """Test JSON serialization round-trip."""
        # Serialize to JSON string
        json_str = test_configuration_set.to_json()
        assert isinstance(json_str, str)

        # Parse to verify valid JSON
        json_data = json.loads(json_str)
        assert isinstance(json_data, dict)

        # Deserialize
        reconstructed = ConfigurationSet.from_json(json_str)

        # Verify
        assert len(reconstructed) == len(test_configuration_set)
        for i in range(len(test_configuration_set)):
            assert reconstructed[i].to_string() == test_configuration_set[i].to_string()

    def test_json_file_serialization(self, test_configuration_set, tmp_path):
        """Test JSON file serialization round-trip."""
        filename = tmp_path / "test_config_set.json"

        # Save to file
        test_configuration_set.to_json_file(str(filename))
        assert filename.exists()

        # Load from file
        reconstructed = ConfigurationSet.from_json_file(str(filename))

        # Verify
        assert len(reconstructed) == len(test_configuration_set)
        for i in range(len(test_configuration_set)):
            assert reconstructed[i].to_string() == test_configuration_set[i].to_string()

    def test_hdf5_file_serialization(self, test_configuration_set, tmp_path):
        """Test HDF5 file serialization round-trip."""
        filename = tmp_path / "test_config_set.h5"

        # Save to file
        test_configuration_set.to_hdf5_file(str(filename))
        assert filename.exists()

        # Load from file
        reconstructed = ConfigurationSet.from_hdf5_file(str(filename))

        # Verify
        assert len(reconstructed) == len(test_configuration_set)
        for i in range(len(test_configuration_set)):
            assert reconstructed[i].to_string() == test_configuration_set[i].to_string()

    def test_generic_file_io_json(self, test_configuration_set, tmp_path):
        """Test generic file I/O with JSON format."""
        filename = tmp_path / "test_config_set_generic.json"

        # Save using generic method
        test_configuration_set.to_file(str(filename), "json")
        assert filename.exists()

        # Load using generic method
        reconstructed = ConfigurationSet.from_file(str(filename), "json")

        # Verify
        assert len(reconstructed) == len(test_configuration_set)

    def test_generic_file_io_hdf5(self, test_configuration_set, tmp_path):
        """Test generic file I/O with HDF5 format."""
        filename = tmp_path / "test_config_set_generic.h5"

        # Save using generic method
        test_configuration_set.to_file(str(filename), "hdf5")
        assert filename.exists()

        # Load using generic method
        reconstructed = ConfigurationSet.from_file(str(filename), "hdf5")

        # Verify
        assert len(reconstructed) == len(test_configuration_set)

    def test_pickle_serialization(self, test_configuration_set):
        """Test pickle serialization round-trip."""
        # Pickle
        pickled = pickle.dumps(test_configuration_set)
        assert isinstance(pickled, bytes)

        # Unpickle
        reconstructed = pickle.loads(pickled)

        # Verify
        assert len(reconstructed) == len(test_configuration_set)
        for i in range(len(test_configuration_set)):
            assert reconstructed[i].to_string() == test_configuration_set[i].to_string()


class TestConfigurationSetWithOrbitals:
    """Test ConfigurationSet interaction with Orbitals."""

    def test_orbitals_preserved_after_serialization(self, tmp_path):
        """Test that orbital information is preserved after serialization."""
        # Create orbitals with specific properties
        orbitals = create_test_orbitals(4)
        configs = [Configuration("2200"), Configuration("udud")]
        config_set = ConfigurationSet(configs, orbitals)

        # Serialize and deserialize
        filename = tmp_path / "test_orbitals_preserved.json"
        config_set.to_json_file(str(filename))
        reconstructed = ConfigurationSet.from_json_file(str(filename))

        # Verify orbitals are preserved
        orig_orbitals = config_set.orbitals
        recon_orbitals = reconstructed.orbitals

        assert recon_orbitals.get_num_molecular_orbitals() == orig_orbitals.get_num_molecular_orbitals()


class TestConfigurationSetEdgeCases:
    """Test ConfigurationSet edge cases and error handling."""

    def test_construction_with_none_orbitals_raises(self):
        """Test that construction with None orbitals raises an error."""
        configs = [Configuration("2200")]
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            ConfigurationSet(configs, None)

    def test_copy_constructor(self):
        """Test copy constructor."""
        orbitals = create_test_orbitals(4)
        configs = [Configuration("2200"), Configuration("2ud0")]
        original = ConfigurationSet(configs, orbitals)

        # Copy
        copied = ConfigurationSet(original)

        # Verify copy is equal to original
        assert len(copied) == len(original)
        for i in range(len(original)):
            assert copied[i].to_string() == original[i].to_string()
