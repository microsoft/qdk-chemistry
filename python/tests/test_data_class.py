"""Tests for the base class functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry._core.data import DataClass as DataClassCore
from qdk_chemistry.data import (
    Ansatz,
    BasisSet,
    CoupledClusterAmplitudes,
    DataClass,
    Hamiltonian,
    Orbitals,
    Settings,
    Structure,
    Wavefunction,
)


class TestDataClass:
    """Test cases for the DataClass base class interface."""

    def test_base_class_existence(self):
        """Test that DataClass class exists and can be imported."""
        assert hasattr(DataClass, "__name__")
        assert DataClass.__name__ == "DataClass"

    def test_structure_inherits_from_base(self):
        """Test that Structure inherits from DataClass."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        assert isinstance(s, DataClassCore)
        assert hasattr(s, "get_summary")
        assert hasattr(s, "to_json")
        assert hasattr(s, "to_json_file")
        assert hasattr(s, "to_hdf5_file")
        assert hasattr(s, "to_file")

    def test_settings_inherits_from_base(self):
        """Test that Settings inherits from DataClass."""

        class TestSettings(Settings):
            def __init__(self):
                super().__init__()
                self._set_default("test_param", "int", 42)

        settings = TestSettings()

        assert isinstance(settings, DataClassCore)
        assert hasattr(settings, "get_summary")
        assert hasattr(settings, "to_json")
        assert hasattr(settings, "to_json_file")
        assert hasattr(settings, "to_hdf5_file")
        assert hasattr(settings, "to_file")

    @pytest.mark.parametrize(
        "data_class",
        [Structure, Settings, BasisSet, Ansatz, Hamiltonian, Orbitals, Wavefunction, CoupledClusterAmplitudes],
    )
    def test_data_classes_have_base_interface(self, data_class):
        """Test that data classes have the required base class methods."""
        # Check that the class has the required methods as class attributes
        assert hasattr(data_class, "get_summary")
        # Note: to_json, to_file, etc. are inherited, so they should exist

    def test_base_class_methods_consistency(self):
        """Test that all data classes provide consistent base class interface."""
        # Create a simple structure to test the interface
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        # Test get_summary returns a string
        summary = s.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Test to_json returns a string
        json_str = s.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test file I/O methods exist and are callable
        assert callable(s.to_json_file)
        assert callable(s.to_hdf5_file)
        assert callable(s.to_file)

    def test_file_io_interface(self):
        """Test that the file I/O interface works consistently."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test JSON file I/O through base interface
            json_file = temp_path / "test_structure.structure.json"
            s.to_json_file(str(json_file))
            assert json_file.exists()

            # Test HDF5 file I/O through base interface
            h5_file = temp_path / "test_structure.structure.h5"
            s.to_hdf5_file(str(h5_file))
            assert h5_file.exists()

            # Test generic file I/O through base interface
            generic_json_file = temp_path / "test_generic.structure.json"
            s.to_file(str(generic_json_file), "json")
            assert generic_json_file.exists()

    def test_pathlib_support(self):
        """Test that base class file methods support pathlib.Path objects."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test that Path objects work with base class methods
            json_file = temp_path / "test_pathlib.structure.json"
            s.to_json_file(json_file)  # Should accept Path object
            assert json_file.exists()

            h5_file = temp_path / "test_pathlib.structure.h5"
            s.to_hdf5_file(h5_file)  # Should accept Path object
            assert h5_file.exists()

    def test_error_handling(self):
        """Test that base class methods handle errors appropriately."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        # Test invalid file type
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.xyz"

            with pytest.raises((ValueError, RuntimeError)):
                s.to_file(str(test_file), "invalid_format")

    def test_multiple_inheritance_classes(self):
        """Test classes that have multiple inheritance work correctly."""
        # Create a simple structure to verify it works with multiple inheritance
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        # Structure inherits from both DataClass and uses py::smart_holder
        # This tests that the multiple inheritance in the binding works
        assert isinstance(s, DataClassCore)

        # Test that we can still access Structure-specific methods
        assert hasattr(s, "get_num_atoms")
        assert s.get_num_atoms() == 1


class TestDataClassCompliance:
    """Test that all data classes properly implement the base interface."""

    def test_structure_compliance(self):
        """Test Structure class compliance with DataClass."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        # Test inheritance
        assert isinstance(s, DataClassCore)

        # Test required methods exist and work
        summary = s.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        json_str = s.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test file methods are callable
        assert callable(s.to_json_file)
        assert callable(s.to_hdf5_file)
        assert callable(s.to_file)

    def test_settings_compliance(self):
        """Test Settings class compliance with DataClass."""

        class TestSettings(Settings):
            def __init__(self):
                super().__init__()
                self._set_default("param", "int", 123)

        settings = TestSettings()

        # Test inheritance
        assert isinstance(settings, DataClassCore)

        # Test required methods exist and work
        summary = settings.get_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

        json_str = settings.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Test file methods are callable
        assert callable(settings.to_json_file)
        assert callable(settings.to_hdf5_file)
        assert callable(settings.to_file)

    def test_method_signatures_preserved(self):
        """Test that inheritance doesn't break existing method signatures."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        # Test that Structure-specific methods still work as expected
        assert s.get_num_atoms() == 2
        assert s.get_total_nuclear_charge() == 2

        # Test that base class methods work with expected signatures
        summary = s.get_summary()  # No arguments
        json_str = s.to_json()  # No arguments

        # Test that the methods return expected types
        assert isinstance(summary, str)
        assert isinstance(json_str, str)

    def test_binding_integrity(self):
        """Test that the pybind11 bindings work correctly with inheritance."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        # Test that the object can be used polymorphically
        def test_base_interface(obj: DataClass) -> str:
            return obj.get_summary()

        # This should work without issues if binding is correct
        summary = test_base_interface(s)
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Test that we can still access derived class methods
        assert s.get_num_atoms() == 2

    def test_method_resolution(self):
        """Test that method resolution works correctly with multiple inheritance."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        # Test that we can call methods that exist in both classes without ambiguity
        # get_summary exists in both DataClass (pure virtual) and Structure (implementation)
        summary1 = s.get_summary()
        summary2 = s.get_summary()

        assert summary1 == summary2
        assert isinstance(summary1, str)
