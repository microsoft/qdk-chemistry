"""Tests for Structure class and related functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
    xyz_file_structure_tolerance,
)

try:
    from qdk_chemistry.data import Structure
except ImportError:
    pytest.skip("qdk_chemistry.data.Structure not available", allow_module_level=True)


class TestStructure:
    """Test cases for the Structure class."""

    def test_coordinates_nuclear_charges_constructor(self):
        """Test constructor with coordinates and nuclear charges."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1, 1]

        s = Structure(coords, nuclear_charges)
        assert not s.is_empty()
        assert s.get_num_atoms() == 2
        assert len(s) == 2
        assert s.get_total_nuclear_charge() == 2

        assert s.get_atom_nuclear_charge(0) == 1
        assert s.get_atom_nuclear_charge(1) == 1
        assert s.get_atom_symbol(0) == "H"
        assert s.get_atom_symbol(1) == "H"

        atom0_coords = s.get_atom_coordinates(0)
        assert np.allclose(
            atom0_coords,
            [0.0, 0.0, 0.0],
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_symbols_coordinates_constructor(self):
        """Test constructor with symbols and coordinates."""
        symbols = ["O", "H", "H"]
        coords = np.array(
            [[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000], [-0.757000, 0.586000, 0.000000]]
        )

        water = Structure(symbols, coords)
        assert water.get_num_atoms() == 3
        assert water.get_total_nuclear_charge() == 10
        assert water.get_atom_symbol(0) == "O"
        assert water.get_atom_symbol(1) == "H"
        assert water.get_atom_symbol(2) == "H"

    def test_dimension_validation(self):
        """Test that mismatched dimensions raise errors."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        nuclear_charges = [1]
        with pytest.raises(ValueError):  # noqa: PT011
            Structure(coords, nuclear_charges)

    def test_unknown_symbol(self):
        """Test that unknown symbols raise errors."""
        symbols = ["X"]
        coords = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError):  # noqa: PT011
            Structure(symbols, coords)

    def test_out_of_range_access(self):
        """Test that out-of-range access raises errors."""
        coords = np.array([[0.0, 0.0, 0.0]])
        symbols = ["H"]
        s = Structure(coords, symbols)

        with pytest.raises(IndexError):
            s.get_atom_coordinates(1)

        with pytest.raises(IndexError):
            s.get_atom_nuclear_charge(1)

        with pytest.raises(IndexError):
            s.get_atom_symbol(1)

    def test_xyz_serialization(self):
        """Test XYZ format serialization."""
        symbols = ["O", "H", "H"]
        coords = np.array(
            [[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000], [-0.757000, 0.586000, 0.000000]]
        )

        water = Structure(symbols, coords)

        xyz_string = water.to_xyz("Water molecule")
        assert "3" in xyz_string
        assert "Water molecule" in xyz_string
        assert "O" in xyz_string
        assert "H" in xyz_string

        water_copy = Structure.from_xyz(xyz_string)

        assert water_copy.get_num_atoms() == 3
        assert water_copy.get_atom_symbol(0) == "O"
        assert water_copy.get_atom_symbol(1) == "H"
        assert water_copy.get_atom_symbol(2) == "H"

        for i in range(3):
            original_coords = water.get_atom_coordinates(i)
            copy_coords = water_copy.get_atom_coordinates(i)
            assert np.allclose(
                original_coords,
                copy_coords,
                rtol=float_comparison_relative_tolerance,
                atol=xyz_file_structure_tolerance,
            )

    def test_xyz_file_io(self):
        """Test XYZ file I/O."""
        coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        symbols = ["H", "H"]
        s = Structure(coordinates=coordinates, symbols=symbols)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".structure.xyz", delete=False) as f:
            temp_file = f.name

        try:
            s.to_xyz_file(temp_file, "H2 molecule")
            assert Path(temp_file).exists()

            s_loaded = Structure.from_xyz_file(temp_file)

            assert s_loaded.get_num_atoms() == 2
            assert s_loaded.get_atom_symbol(0) == "H"
            assert s_loaded.get_atom_symbol(1) == "H"

            original_coords_0 = s.get_atom_coordinates(0)
            loaded_coords_0 = s_loaded.get_atom_coordinates(0)
            assert np.allclose(
                original_coords_0,
                loaded_coords_0,
                rtol=float_comparison_relative_tolerance,
                atol=xyz_file_structure_tolerance,
            )

            original_coords_1 = s.get_atom_coordinates(1)
            loaded_coords_1 = s_loaded.get_atom_coordinates(1)
            assert np.allclose(
                original_coords_1,
                loaded_coords_1,
                rtol=float_comparison_relative_tolerance,
                atol=xyz_file_structure_tolerance,
            )
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

    def test_json_serialization(self):
        """Test JSON serialization."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        symbols = ["C", "O"]
        s = Structure(symbols, coords)

        json_str = s.to_json()
        assert isinstance(json_str, str)

        j = json.loads(json_str)

        assert j["num_atoms"] == 2
        assert len(j["symbols"]) == 2
        assert j["symbols"][0] == "C"
        assert j["symbols"][1] == "O"
        assert len(j["nuclear_charges"]) == 2
        assert j["nuclear_charges"][0] == 6
        assert j["nuclear_charges"][1] == 8

        s_copy = Structure.from_json(json_str)

        assert s_copy.get_num_atoms() == 2
        assert s_copy.get_atom_symbol(0) == "C"
        assert s_copy.get_atom_symbol(1) == "O"

        original_coords_0 = s.get_atom_coordinates(0)
        copy_coords_0 = s_copy.get_atom_coordinates(0)
        assert np.allclose(
            original_coords_0,
            copy_coords_0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        original_coords_1 = s.get_atom_coordinates(1)
        copy_coords_1 = s_copy.get_atom_coordinates(1)
        assert np.allclose(
            original_coords_1,
            copy_coords_1,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_json_file_io(self):
        """Test JSON file I/O."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
        symbols = ["N", "N"]
        s = Structure(symbols, coords)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".structure.json", delete=False) as f:
            temp_file = f.name

        try:
            s.to_json_file(temp_file)
            assert Path(temp_file).exists()

            s_loaded = Structure.from_json_file(temp_file)

            assert s_loaded.get_num_atoms() == 2
            assert s_loaded.get_atom_symbol(0) == "N"
            assert s_loaded.get_atom_symbol(1) == "N"

            original_coords_0 = s.get_atom_coordinates(0)
            loaded_coords_0 = s_loaded.get_atom_coordinates(0)
            assert np.allclose(
                original_coords_0,
                loaded_coords_0,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

            original_coords_1 = s.get_atom_coordinates(1)
            loaded_coords_1 = s_loaded.get_atom_coordinates(1)
            assert np.allclose(
                original_coords_1,
                loaded_coords_1,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        finally:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

    def test_static_utility_functions(self):
        """Test static utility functions."""
        assert Structure.symbol_to_nuclear_charge("H") == 1
        assert Structure.symbol_to_nuclear_charge("C") == 6
        assert Structure.symbol_to_nuclear_charge("O") == 8
        with pytest.raises(ValueError):  # noqa: PT011
            Structure.symbol_to_nuclear_charge("Xx")

        assert Structure.nuclear_charge_to_symbol(1) == "H"
        assert Structure.nuclear_charge_to_symbol(6) == "C"
        assert Structure.nuclear_charge_to_symbol(8) == "O"

        with pytest.raises(ValueError):  # noqa: PT011
            Structure.nuclear_charge_to_symbol(200)

    def test_get_summary(self):
        """Test summary generation."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        symbols = ["C", "H", "H"]
        s = Structure(symbols, coords)

        summary = s.get_summary()
        assert "Number of atoms: 3" in summary
        assert "C1" in summary or "H2" in summary
        assert "H2" in summary

    def test_string_representations(self):
        """Test string representations (__str__ and __repr__)."""
        # Create minimal structure for testing
        s_single = Structure(["H"], np.array([[0.0, 0.0, 0.0]]))
        assert "Structure Summary" in str(s_single) or "1 atoms" in str(s_single)
        assert "atoms: 1" in repr(s_single)

        s_nonempty = Structure(["H"], np.array([[0.0, 0.0, 0.0]]))
        assert "Structure Summary" in str(s_nonempty)
        assert "atoms: 1" in repr(s_nonempty)

    def test_indexing_and_iteration(self):
        """Test indexing and iteration support."""
        s = Structure(["C", "H"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

        atom0 = s[0]
        assert atom0["symbol"] == "C"
        assert atom0["nuclear_charge"] == 6
        assert np.allclose(
            atom0["coordinates"],
            [0.0, 0.0, 0.0],
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        with pytest.raises(IndexError):
            s[2]

        symbols_from_iteration = []
        try:
            for atom in s:
                symbols_from_iteration.append(atom["symbol"])
            assert symbols_from_iteration == ["C", "H"]

            atoms_list = list(s)
            assert len(atoms_list) == 2
            assert atoms_list[0]["symbol"] == "C"
            assert atoms_list[1]["symbol"] == "H"
        except TypeError:
            assert s.get_num_atoms() == 2
            assert s.get_atom_symbol(0) == "C"
            assert s.get_atom_symbol(1) == "H"

    def test_properties_and_accessors(self):
        """Test various properties and accessor methods."""
        s = Structure(
            ["O", "H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]),
        )

        coords_matrix = s.get_coordinates()
        assert coords_matrix.shape == (3, 3)

        charges = s.get_nuclear_charges()
        assert np.array_equal(charges, [8, 1, 1])

        symbols = s.get_atomic_symbols()
        assert symbols == ["O", "H", "H"]

    def test_coordinate_matrix_operations(self):
        """Test operations with coordinate matrices."""
        new_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        s = Structure(["H", "H"], new_coords)

        retrieved_coords = s.get_coordinates()
        assert np.allclose(
            new_coords,
            retrieved_coords,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        s_charges = Structure(new_coords, [1, 1])
        assert np.array_equal(s_charges.get_nuclear_charges(), [1, 1])

    def test_nuclear_repulsion_energy(self):
        """Test calculation of nuclear repulsion energy."""
        # Test with empty arrays if supported, otherwise skip empty test
        try:
            empty_structure = Structure([], np.empty((0, 3)))
            assert np.isclose(
                empty_structure.calculate_nuclear_repulsion_energy(),
                0.0,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        except ValueError:
            # Empty structures not supported
            pass

        single_atom = Structure(["H"], np.array([[0.0, 0.0, 0.0]]))
        assert np.isclose(
            single_atom.calculate_nuclear_repulsion_energy(),
            0.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        h2_molecule = Structure(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
        h2_expected = 1.0 / 0.74
        assert np.isclose(
            h2_molecule.calculate_nuclear_repulsion_energy(),
            h2_expected,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        symbols = ["O", "H", "H"]
        coords = np.array(
            [[0.000000, 0.000000, 0.000000], [0.757000, 0.586000, 0.000000], [-0.757000, 0.586000, 0.000000]]
        )

        water = Structure(symbols, coords)
        o_h1_dist = np.sqrt(0.757**2 + 0.586**2)
        o_h2_dist = np.sqrt(0.757**2 + 0.586**2)
        h1_h2_dist = np.sqrt(1.514**2)

        expected_water_repulsion = 8.0 * 1.0 / o_h1_dist + 8.0 * 1.0 / o_h2_dist + 1.0 * 1.0 / h1_h2_dist

        assert np.isclose(
            water.calculate_nuclear_repulsion_energy(),
            expected_water_repulsion,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        custom_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        # Use integer nuclear charges as required by new immutable design
        custom_molecule = Structure(custom_coords, [1, 2])  # Changed from [1.5, 2.5] to [1, 2]

        custom_expected = (1 * 2) / 1.0  # Updated calculation
        assert np.isclose(
            custom_molecule.calculate_nuclear_repulsion_energy(),
            custom_expected,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )


class TestStructureEdgeCases:
    """Test edge cases and error conditions for Structure class."""

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        invalid_json_str = json.dumps({"invalid": "data"})

        with pytest.raises(RuntimeError):
            Structure.from_json(invalid_json_str)

    def test_invalid_xyz_format(self):
        """Test handling of invalid XYZ format."""
        # Test static method failures - Structure default constructor doesn't exist
        with pytest.raises(RuntimeError):  # Should raise std::runtime_error
            Structure.from_xyz("invalid\ncomment\nH 0 0 0")

        # Missing comment
        with pytest.raises(RuntimeError):
            Structure.from_xyz("invalid\ncomment\nH 0 0 0")

        with pytest.raises(RuntimeError):
            Structure.from_xyz("1")

        with pytest.raises(RuntimeError):
            Structure.from_xyz("1\ncomment\nH 0 0")

    def test_file_io_errors(self):
        """Test file I/O error handling."""
        with pytest.raises(RuntimeError):
            Structure.from_xyz_file("non_existent.structure.xyz")

        with pytest.raises(RuntimeError):
            Structure.from_json_file("non_existent.structure.json")

    def test_filename_validation(self):
        """Test filename validation for structure files."""
        coords = np.array([[0.0, 0.0, 0.0]])
        s = Structure(["H"], coords)

        with tempfile.NamedTemporaryFile(suffix=".structure.json", delete=False) as f:
            valid_json_filename = f.name
        try:
            s.to_json_file(valid_json_filename)
            Structure.from_json_file(valid_json_filename)
        finally:
            Path(valid_json_filename).unlink()

        with tempfile.NamedTemporaryFile(suffix=".structure.xyz", delete=False) as f:
            valid_xyz_filename = f.name
        try:
            s.to_xyz_file(valid_xyz_filename, "Test molecule")
            Structure.from_xyz_file(valid_xyz_filename)
        finally:
            Path(valid_xyz_filename).unlink()
        # invalid extensions
        with pytest.raises(ValueError):  # noqa: PT011
            s.to_json_file("invalid.json")
        with pytest.raises(ValueError):  # noqa: PT011
            s.from_json_file("invalid.json")
        with pytest.raises(ValueError):  # noqa: PT011
            s.to_xyz_file("invalid.xyz")
        with pytest.raises(ValueError):  # noqa: PT011
            s.from_xyz_file("invalid.xyz")


class TestStructureFileIO:
    """Test file I/O operations for Structure class."""

    def test_to_file_from_file_json(self):
        """Test generic to_file and from_file methods with JSON format."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        symbols = ["H", "H"]
        s1 = Structure(coords, symbols)

        with tempfile.NamedTemporaryFile(suffix=".structure.json", delete=False) as f:
            filename = f.name

        try:
            s1.to_file(filename, "json")
            s2 = Structure.from_file(filename, "json")

            assert s2.get_num_atoms() == s1.get_num_atoms()
            assert s2.get_atom_symbol(0) == s1.get_atom_symbol(0)
            assert s2.get_atom_symbol(1) == s1.get_atom_symbol(1)

            assert np.allclose(
                s2.get_atom_coordinates(0),
                s1.get_atom_coordinates(0),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                s2.get_atom_coordinates(1),
                s1.get_atom_coordinates(1),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        finally:
            Path(filename).unlink()

    def test_to_file_from_file_xyz(self):
        """Test generic to_file and from_file methods with XYZ format."""
        coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        symbols = ["O", "H", "H"]
        s1 = Structure(coords, symbols)

        with tempfile.NamedTemporaryFile(suffix=".structure.xyz", delete=False) as f:
            filename = f.name

        try:
            s1.to_file(filename, "xyz")
            s2 = Structure.from_file(filename, "xyz")

            assert s2.get_num_atoms() == s1.get_num_atoms()
            assert s2.get_atom_symbol(0) == s1.get_atom_symbol(0)
            assert s2.get_atom_symbol(1) == s1.get_atom_symbol(1)
            assert s2.get_atom_symbol(2) == s1.get_atom_symbol(2)

            assert np.allclose(
                s2.get_atom_coordinates(0),
                s1.get_atom_coordinates(0),
                rtol=float_comparison_relative_tolerance,
                atol=xyz_file_structure_tolerance,
            )
            assert np.allclose(
                s2.get_atom_coordinates(1),
                s1.get_atom_coordinates(1),
                rtol=float_comparison_relative_tolerance,
                atol=xyz_file_structure_tolerance,
            )
            assert np.allclose(
                s2.get_atom_coordinates(2),
                s1.get_atom_coordinates(2),
                rtol=float_comparison_relative_tolerance,
                atol=xyz_file_structure_tolerance,
            )
        finally:
            Path(filename).unlink()

    def test_hdf5_file_io(self):
        """Test HDF5 file I/O round-trip."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
        symbols = ["H", "H"]
        custom_masses = np.array([1.001, 0.999])
        custom_charges = np.array([0.9, 1.1])
        s1 = Structure(coords, symbols, custom_masses, custom_charges)

        with tempfile.NamedTemporaryFile(suffix=".structure.h5", delete=False) as f:
            filename = f.name

        try:
            # Save to HDF5
            s1.to_hdf5_file(filename)
            assert Path(filename).exists()

            # Load from HDF5
            s2 = Structure.from_hdf5_file(filename)

            # Verify structure properties
            assert s2.get_num_atoms() == s1.get_num_atoms()
            assert s2.get_atom_symbol(0) == s1.get_atom_symbol(0)
            assert s2.get_atom_symbol(1) == s1.get_atom_symbol(1)

            # Verify coordinates
            assert np.allclose(
                s2.get_atom_coordinates(0),
                s1.get_atom_coordinates(0),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                s2.get_atom_coordinates(1),
                s1.get_atom_coordinates(1),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

            # Verify custom masses and charges are preserved
            assert np.allclose(
                s2.get_masses(),
                custom_masses,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                s2.get_nuclear_charges(),
                custom_charges,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        finally:
            if Path(filename).exists():
                Path(filename).unlink()

    def test_to_file_from_file_errors(self):
        """Test error handling for generic file methods."""
        coords = np.array([[0.0, 0.0, 0.0]])
        symbols = ["H"]
        s = Structure(coords, symbols)
        with pytest.raises(ValueError):  # noqa: PT011
            s.to_file("test.structure.xyz", "invalid_format")
        with pytest.raises(ValueError):  # noqa: PT011
            Structure.from_file("test.structure.xyz", "invalid_format")
        with pytest.raises(RuntimeError):
            Structure.from_file("non_existent.structure.json", "json")

    def test_filename_validation_consistency(self):
        """Test that filename validation is consistent across all methods."""
        coords = np.array([[0.0, 0.0, 0.0]])
        symbols = ["H"]
        s = Structure(coords, symbols)

        invalid_filenames = [
            "test.json",
            "test.xyz",
            "test.structure",
            "structure.json",
            "structure.xyz",
        ]

        for filename in invalid_filenames:
            with pytest.raises(ValueError):  # noqa: PT011
                s.to_json_file(filename)
            with pytest.raises(ValueError):  # noqa: PT011
                s.from_json_file(filename)

            with pytest.raises(ValueError):  # noqa: PT011
                s.to_xyz_file(filename)
            with pytest.raises(ValueError):  # noqa: PT011
                s.from_xyz_file(filename)


class TestStructurePicklingAndRepr:
    """Test pickling support and string representation for Structure class."""

    def test_pickle_support(self):
        """Test that Structure objects can be pickled and unpickled."""
        # Create a test structure
        coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        symbols = ["O", "H", "H"]
        original = Structure(coords, symbols)

        # Test pickling and unpickling
        pickled_data = pickle.dumps(original)
        assert isinstance(pickled_data, bytes)

        # Unpickle and verify
        unpickled = pickle.loads(pickled_data)

        # Verify structure is preserved
        assert unpickled.get_num_atoms() == original.get_num_atoms()
        assert unpickled.get_total_nuclear_charge() == original.get_total_nuclear_charge()
        assert unpickled.get_atomic_symbols() == original.get_atomic_symbols()

        # Check coordinates are preserved
        assert np.allclose(
            unpickled.get_coordinates(),
            original.get_coordinates(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check masses and nuclear charges are preserved
        assert np.allclose(
            unpickled.get_masses(),
            original.get_masses(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            unpickled.get_nuclear_charges(),
            original.get_nuclear_charges(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_pickle_with_custom_masses_and_charges(self):
        """Test pickling with custom masses and nuclear charges."""
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        symbols = ["H", "H"]
        custom_masses = np.array([1.5, 2.0])  # Custom masses
        custom_charges = np.array([1.2, 1.8])  # Custom charges

        original = Structure(coords, symbols, custom_masses, custom_charges)

        # Pickle and unpickle
        pickled_data = pickle.dumps(original)
        unpickled = pickle.loads(pickled_data)

        # Verify custom values are preserved
        assert np.allclose(
            unpickled.get_masses(),
            custom_masses,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            unpickled.get_nuclear_charges(),
            custom_charges,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_pickle_round_trip_multiple_times(self):
        """Test that multiple pickle/unpickle cycles preserve data integrity."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        symbols = ["C", "N", "O"]
        original = Structure(coords, symbols)

        current = original
        # Multiple round trips
        for _ in range(3):
            pickled = pickle.dumps(current)
            current = pickle.loads(pickled)

            # Verify data integrity after each round trip
            assert current.get_num_atoms() == original.get_num_atoms()
            assert np.allclose(
                current.get_coordinates(),
                original.get_coordinates(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert current.get_atomic_symbols() == original.get_atomic_symbols()

    def test_repr_with_summary(self):
        """Test that __repr__ uses the summary function."""
        # Test with a non-empty structure
        coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        symbols = ["O", "H", "H"]
        structure = Structure(coords, symbols)

        # Get both repr and summary
        repr_output = repr(structure)
        summary_output = structure.get_summary()

        # They should be the same (or at least repr should contain summary content)
        assert repr_output == summary_output

        # Check that typical summary content is present
        assert "Number of atoms: 3" in repr_output
        assert "O" in repr_output
        assert "H" in repr_output

    def test_repr_single_atom(self):
        """Test __repr__ with single atom structure."""
        coords = np.array([[0.0, 0.0, 0.0]])
        symbols = ["H"]
        structure = Structure(coords, symbols)

        repr_output = repr(structure)

        # Should contain structure information
        assert "Number of atoms: 1" in repr_output
        assert "H" in repr_output

    def test_str_vs_repr_consistency(self):
        """Test that __str__ and __repr__ are consistent."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        symbols = ["C", "H"]
        structure = Structure(coords, symbols)

        str_output = str(structure)
        repr_output = repr(structure)

        # Both should use the summary function for non-empty structures
        assert str_output == repr_output

        # Both should contain key information
        assert "Number of atoms: 2" in str_output
        assert "Number of atoms: 2" in repr_output

    def test_pickle_preserves_repr_functionality(self):
        """Test that pickled structures still have proper repr functionality."""
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        symbols = ["N", "N"]
        original = Structure(coords, symbols)

        # Pickle and unpickle
        pickled_data = pickle.dumps(original)
        unpickled = pickle.loads(pickled_data)

        # Both should have same repr output
        original_repr = repr(original)
        unpickled_repr = repr(unpickled)

        assert original_repr == unpickled_repr
        assert "Number of atoms: 2" in unpickled_repr
        assert "N" in unpickled_repr
