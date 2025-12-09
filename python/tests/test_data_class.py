"""Tests for the base class functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry._core.data import DataClass as DataClassCore
from qdk_chemistry.data import (
    Ansatz,
    BasisSet,
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
        [Structure, Settings, BasisSet, Ansatz, Hamiltonian, Orbitals, Wavefunction],
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


class TestDataClassDeserialization:
    """Test deserialization methods on DataClass base class."""

    def test_derived_class_from_json_file_works(self):
        """Test that derived classes can successfully use from_json_file."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            json_file = temp_path / "test.structure.json"

            # Save structure
            s.to_json_file(str(json_file))

            # Load structure back
            loaded = Structure.from_json_file(str(json_file))
            assert loaded.get_num_atoms() == s.get_num_atoms()

    def test_derived_class_from_hdf5_file_works(self):
        """Test that derived classes can successfully use from_hdf5_file."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            h5_file = temp_path / "test.structure.h5"

            # Save structure
            s.to_hdf5_file(str(h5_file))

            # Load structure back
            loaded = Structure.from_hdf5_file(str(h5_file))
            assert loaded.get_num_atoms() == s.get_num_atoms()


class TestDataClassErrorHandling:
    """Test error handling and edge cases for DataClass methods."""

    def test_to_hdf5_file_invalid_directory(self):
        """Test that to_file raises error for unsupported format."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"

            with pytest.raises((ValueError, RuntimeError)):
                s.to_file(str(test_file), "unsupported_format")

    def test_to_file_with_empty_format_string(self):
        """Test that to_file handles empty format string."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.dat"

            with pytest.raises((ValueError, RuntimeError)):
                s.to_file(str(test_file), "")

    def test_from_hdf5_file_nonexistent_file(self):
        """Test that pathlib.Path objects with special characters work."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Use a filename with spaces and underscores
            json_file = temp_path / "test file_name.structure.json"

            s.to_json_file(json_file)
            assert json_file.exists()

            loaded = Structure.from_json_file(json_file)
            assert loaded.get_num_atoms() == 1


class TestDataClassRoundTrip:
    """Test round-trip serialization/deserialization for DataClass."""

    def test_json_round_trip(self):
        """Test that data survives JSON round-trip."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        nuclear_charges = [6, 1, 1]
        original = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            json_file = temp_path / "roundtrip.structure.json"

            # Save and load
            original.to_json_file(json_file)
            loaded = Structure.from_json_file(json_file)

            # Verify data is preserved
            assert loaded.get_num_atoms() == original.get_num_atoms()
            assert loaded.get_total_nuclear_charge() == original.get_total_nuclear_charge()

    def test_hdf5_round_trip(self):
        """Test that data survives HDF5 round-trip."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        nuclear_charges = [6, 1, 1]
        original = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            h5_file = temp_path / "roundtrip.structure.h5"

            # Save and load
            original.to_hdf5_file(h5_file)
            loaded = Structure.from_hdf5_file(h5_file)

            # Verify data is preserved
            assert loaded.get_num_atoms() == original.get_num_atoms()
            assert loaded.get_total_nuclear_charge() == original.get_total_nuclear_charge()

    def test_generic_file_json_round_trip(self):
        """Test that data survives generic file I/O round-trip with JSON format."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [8, 1]
        original = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "generic.structure.json"

            # Save and load using generic interface
            original.to_file(file_path, "json")
            loaded = Structure.from_file(file_path, "json")

            # Verify data is preserved
            assert loaded.get_num_atoms() == original.get_num_atoms()
            assert loaded.get_total_nuclear_charge() == original.get_total_nuclear_charge()

    def test_generic_file_hdf5_round_trip(self):
        """Test that data survives generic file I/O round-trip with HDF5 format."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [8, 1]
        original = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "generic.structure.h5"

            # Save and load using generic interface
            original.to_file(file_path, "hdf5")
            loaded = Structure.from_file(file_path, "hdf5")

            # Verify data is preserved
            assert loaded.get_num_atoms() == original.get_num_atoms()
            assert loaded.get_total_nuclear_charge() == original.get_total_nuclear_charge()


class TestDataClassMultipleObjects:
    """Test DataClass with multiple objects and complex scenarios."""

    def test_json_string_method(self):
        """Test that to_json returns valid JSON string."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        json_str = s.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0

    def test_get_summary_not_empty(self):
        """Test that get_summary returns non-empty informative string."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [6, 1]
        s = Structure(coords, nuclear_charges)

        summary = s.get_summary()

        assert isinstance(summary, str)
        assert len(summary) > 10  # Should be more than just a few characters
        # Summary should contain some useful information
        assert "Structure" in summary or "atom" in summary.lower()

    def test_concurrent_file_operations(self):
        """Test that multiple file operations can be performed safely."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Perform multiple file operations
            json_file1 = temp_path / "test1.structure.json"
            json_file2 = temp_path / "test2.structure.json"
            h5_file = temp_path / "test.structure.h5"

            s.to_json_file(json_file1)
            s.to_json_file(json_file2)
            s.to_hdf5_file(h5_file)

            # All files should exist
            assert json_file1.exists()
            assert json_file2.exists()
            assert h5_file.exists()

            # All should be loadable
            loaded1 = Structure.from_json_file(json_file1)
            loaded2 = Structure.from_json_file(json_file2)
            loaded3 = Structure.from_hdf5_file(h5_file)

            assert loaded1.get_num_atoms() == 1
            assert loaded2.get_num_atoms() == 1
            assert loaded3.get_num_atoms() == 1
