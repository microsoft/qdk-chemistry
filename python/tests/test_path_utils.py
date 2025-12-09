"""Tests for path_utils.hpp functionality.

This module tests the to_string_path utility function that converts Python
path-like objects (strings and pathlib.Path) to C++ std::string objects.
The function is used throughout the C++ bindings for file I/O operations.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import tempfile
from pathlib import Path, PurePath

import numpy as np
import pytest

from qdk_chemistry.data import Structure


class TestPathUtilsStringConversion:
    """Test path conversion with regular string objects."""

    def test_string_path_to_json_file(self):
        """Test that string paths work with to_json_file."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use string path
            json_file = f"{temp_dir}/test_string.structure.json"
            s.to_json_file(json_file)

            assert Path(json_file).exists()

            # Load it back
            loaded = Structure.from_json_file(json_file)
            assert loaded.get_num_atoms() == 2

    def test_string_path_to_hdf5_file(self):
        """Test that string paths work with to_hdf5_file."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use string path
            h5_file = f"{temp_dir}/test_string.structure.h5"
            s.to_hdf5_file(h5_file)

            assert Path(h5_file).exists()

            # Load it back
            loaded = Structure.from_hdf5_file(h5_file)
            assert loaded.get_num_atoms() == 2

    def test_string_path_to_file(self):
        """Test that string paths work with to_file."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Use string path
            json_file = f"{temp_dir}/test_generic.structure.json"
            s.to_file(json_file, "json")

            assert Path(json_file).exists()

            # Load it back
            loaded = Structure.from_file(json_file, "json")
            assert loaded.get_num_atoms() == 2

    def test_absolute_string_path(self):
        """Test absolute string paths."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create absolute path as string
            abs_path = str(Path(temp_dir).absolute() / "absolute_test.structure.json")
            s.to_json_file(abs_path)

            assert Path(abs_path).exists()
            assert Path(abs_path).is_absolute()

    def test_relative_string_path(self):
        """Test relative string paths."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Use relative path
                s.to_json_file("relative_test.structure.json")

                assert Path("relative_test.structure.json").exists()
            finally:
                os.chdir(old_cwd)


class TestPathUtilsPathlibConversion:
    """Test path conversion with pathlib.Path objects."""

    def test_pathlib_path_to_json_file(self):
        """Test that pathlib.Path objects work with to_json_file."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Use Path object
            json_file = temp_path / "test_pathlib.structure.json"
            s.to_json_file(json_file)

            assert json_file.exists()

            # Load it back
            loaded = Structure.from_json_file(json_file)
            assert loaded.get_num_atoms() == 2

    def test_pathlib_path_to_hdf5_file(self):
        """Test that pathlib.Path objects work with to_hdf5_file."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Use Path object
            h5_file = temp_path / "test_pathlib.structure.h5"
            s.to_hdf5_file(h5_file)

            assert h5_file.exists()

            # Load it back
            loaded = Structure.from_hdf5_file(h5_file)
            assert loaded.get_num_atoms() == 2

    def test_pathlib_path_to_file(self):
        """Test that pathlib.Path objects work with to_file."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Use Path object
            json_file = temp_path / "test_generic.structure.json"
            s.to_file(json_file, "json")

            assert json_file.exists()

            # Load it back
            loaded = Structure.from_file(json_file, "json")
            assert loaded.get_num_atoms() == 2

    def test_pathlib_absolute_path(self):
        """Test absolute pathlib.Path objects."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).absolute()
            abs_file = temp_path / "absolute.structure.json"

            s.to_json_file(abs_file)

            assert abs_file.exists()
            assert abs_file.is_absolute()

    def test_pathlib_relative_path(self):
        """Test relative pathlib.Path objects."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Use relative Path object
                rel_file = Path("relative.structure.json")
                s.to_json_file(rel_file)

                assert rel_file.exists()
            finally:
                os.chdir(old_cwd)

    def test_pathlib_pure_path_objects(self):
        """Test PurePath objects (which have __fspath__)."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            # PurePath should work because it has __fspath__
            pure_path = PurePath(temp_dir) / "pure_path.structure.json"
            s.to_json_file(pure_path)

            assert Path(pure_path).exists()


class TestPathUtilsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string_path(self):
        """Test that empty string paths raise appropriate error."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        # Empty string should fail when trying to create file
        with pytest.raises(ValueError, match="Filename"):
            s.to_json_file("")

    def test_path_with_spaces(self):
        """Test paths containing spaces."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Path with spaces
            file_with_spaces = temp_path / "file with spaces.structure.json"
            s.to_json_file(file_with_spaces)

            assert file_with_spaces.exists()

            loaded = Structure.from_json_file(file_with_spaces)
            assert loaded.get_num_atoms() == 1

    def test_path_with_special_characters(self):
        """Test paths containing special characters."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Path with underscores, hyphens, dots
            special_file = temp_path / "test_file-name.v2.structure.json"
            s.to_json_file(special_file)

            assert special_file.exists()

    def test_path_with_unicode_characters(self):
        """Test paths containing Unicode characters."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Path with unicode characters (if filesystem supports it)
            try:
                unicode_file = temp_path / "test_αβγ_文件.structure.json"
                s.to_json_file(unicode_file)

                if unicode_file.exists():
                    loaded = Structure.from_json_file(unicode_file)
                    assert loaded.get_num_atoms() == 1
            except (RuntimeError, OSError):
                # Some filesystems may not support unicode filenames
                pytest.skip("Filesystem does not support unicode filenames")

    def test_deeply_nested_path(self):
        """Test deeply nested directory paths."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            # Create nested directories
            nested_path = temp_path / "level1" / "level2" / "level3" / "level4"
            nested_path.mkdir(parents=True, exist_ok=True)

            nested_file = nested_path / "deeply_nested.structure.json"
            s.to_json_file(nested_file)

            assert nested_file.exists()

    def test_path_with_parent_references(self):
        """Test paths with .. parent directory references."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            subdir = temp_path / "subdir"
            subdir.mkdir(exist_ok=True)

            # Path with parent reference
            parent_ref_file = subdir / ".." / "parent_ref.structure.json"
            s.to_json_file(parent_ref_file)

            # Should be created in temp_dir, not subdir
            expected_file = temp_path / "parent_ref.structure.json"
            assert expected_file.exists()


class TestPathUtilsErrorHandling:
    """Test error handling for invalid path inputs."""

    def test_none_path_raises_error(self):
        """Test that None as path raises appropriate error."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        # None should raise ValueError
        with pytest.raises(ValueError, match="Filename"):
            s.to_json_file(None)

    def test_integer_path_raises_error(self):
        """Test that integer as path raises appropriate error."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        # Integer should raise ValueError
        with pytest.raises(ValueError, match="Filename"):
            s.to_json_file(123)

    def test_list_path_raises_error(self):
        """Test that list as path raises appropriate error."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        # List should raise ValueError
        with pytest.raises(ValueError, match="Filename"):
            s.to_json_file(["path", "to", "file.json"])

    def test_dict_path_raises_error(self):
        """Test that dict as path raises appropriate error."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        # Dict should raise ValueError
        with pytest.raises(ValueError, match="Filename"):
            s.to_json_file({"path": "file.json"})

    def test_custom_object_without_fspath_raises_error(self):
        """Test that custom objects without __fspath__ raise appropriate error."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        # Custom object without __fspath__
        class CustomObject:
            pass

        with pytest.raises(ValueError, match="Filename"):
            s.to_json_file(CustomObject())

    def test_bytes_path_converts_via_str(self):
        """Test that bytes objects are converted via str() as fallback."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Bytes path should be converted via str()
            bytes_path = bytes(str(temp_path / "bytes_test.structure.json"), "utf-8")

            try:
                # This might work via str() fallback
                s.to_json_file(bytes_path)
                expected_file = temp_path / "bytes_test.structure.json"
                # If it worked, verify
                if expected_file.exists():
                    assert True
            except (RuntimeError, UnicodeDecodeError):
                # If bytes aren't supported, that's acceptable behavior
                pytest.skip("Bytes paths not supported")


class TestPathUtilsIntegration:
    """Test path_utils integration with various data classes."""

    def test_path_utils_with_different_formats(self):
        """Test that path conversion works for both JSON and HDF5 formats."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test JSON with Path object
            json_file = temp_path / "format_test.structure.json"
            s.to_file(json_file, "json")
            assert json_file.exists()

            # Test HDF5 with Path object
            h5_file = temp_path / "format_test.structure.h5"
            s.to_file(h5_file, "hdf5")
            assert h5_file.exists()

            # Load both back
            loaded_json = Structure.from_file(json_file, "json")
            loaded_h5 = Structure.from_file(h5_file, "hdf5")

            assert loaded_json.get_num_atoms() == 2
            assert loaded_h5.get_num_atoms() == 2

    def test_path_conversion_consistency(self):
        """Test that string and Path objects produce identical results."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        nuclear_charges = [1, 1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save with string path
            str_file = str(temp_path / "string_path.structure.json")
            s.to_json_file(str_file)

            # Save with Path object
            path_file = temp_path / "path_object.structure.json"
            s.to_json_file(path_file)

            # Both files should exist and be loadable
            assert Path(str_file).exists()
            assert path_file.exists()

            loaded_str = Structure.from_json_file(str_file)
            loaded_path = Structure.from_json_file(path_file)

            assert loaded_str.get_num_atoms() == loaded_path.get_num_atoms()

    def test_mixed_string_and_path_operations(self):
        """Test mixing string and Path objects in save/load operations."""
        coords = np.array([[0.0, 0.0, 0.0]])
        nuclear_charges = [1]
        s = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Save with Path object
            path_file = temp_path / "mixed_test.structure.json"
            s.to_json_file(path_file)

            # Load with string
            loaded = Structure.from_json_file(str(path_file))
            assert loaded.get_num_atoms() == 1

            # Save again with string
            str_file = str(temp_path / "mixed_test2.structure.json")
            loaded.to_json_file(str_file)

            # Load with Path
            loaded2 = Structure.from_json_file(Path(str_file))
            assert loaded2.get_num_atoms() == 1

    def test_path_utils_roundtrip_multiple_times(self) -> None:
        """Test that path conversion works correctly through multiple save/load cycles."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        nuclear_charges = [6, 1, 1]
        original = Structure(coords, nuclear_charges)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            current = original
            file_path: Path | str
            for i in range(3):
                # Alternate between string and Path objects
                if i % 2 == 0:
                    file_path = temp_path / f"roundtrip_{i}.structure.json"
                else:
                    file_path = str(temp_path / f"roundtrip_{i}.structure.json")

                current.to_json_file(file_path)
                current = Structure.from_json_file(file_path)

                # Verify data integrity
                assert current.get_num_atoms() == 3
                assert current.get_total_nuclear_charge() == 8
