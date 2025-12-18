"""Tests for the BasisSet class in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import contextlib
import pickle
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

import qdk_chemistry.algorithms as alg
from qdk_chemistry.data import AOType, BasisSet, Element, OrbitalType, Shell, Structure

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_basis_set_construction():
    """Test constructing BasisSet objects."""
    # Create a valid shell first for testing
    shell = Shell(0, OrbitalType.S, [1.0], [1.0])

    # Test constructor with name and shells (minimal)
    basis = BasisSet("", [shell])
    assert basis.get_name() == ""
    assert basis.get_atomic_orbital_type() == AOType.Spherical
    assert basis.get_num_shells() == 1
    assert basis.get_num_atomic_orbitals() == 1

    # Test constructor with name and shells
    basis_named = BasisSet("6-31G", [shell])
    assert basis_named.get_name() == "6-31G"
    assert basis_named.get_atomic_orbital_type() == AOType.Spherical
    assert basis_named.get_num_shells() == 1

    # Test constructor with name, shells, and basis type
    basis_cartesian = BasisSet("6-31G", [shell], AOType.Cartesian)
    assert basis_cartesian.get_name() == "6-31G"
    assert basis_cartesian.get_atomic_orbital_type() == AOType.Cartesian

    # Test copy constructor
    basis_copy = BasisSet(basis_cartesian)
    assert basis_copy.get_name() == "6-31G"
    assert basis_copy.get_atomic_orbital_type() == AOType.Cartesian


def test_shell_management():
    """Test creating and retrieving shells."""
    # Create shells first, then basis set
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),
        Shell(0, OrbitalType.P, [0.5], [1.0]),
    ]
    basis = BasisSet("STO-3G", shells)

    assert basis.get_num_shells() == 2
    assert basis.get_num_atoms() == 1
    assert basis.get_num_atomic_orbitals() == 4  # 1 s + 3 p

    # Test get_shells method - flattened from per-atom storage
    all_shells = basis.get_shells()
    assert len(all_shells) == 2
    assert all_shells[0].orbital_type == OrbitalType.S
    assert all_shells[1].orbital_type == OrbitalType.P

    # Get shells for atom 0
    atom_shells = basis.get_shells_for_atom(0)
    assert len(atom_shells) == 2
    assert atom_shells[0].orbital_type == OrbitalType.S
    assert atom_shells[1].orbital_type == OrbitalType.P

    # Get specific shell
    shell_0 = basis.get_shell(0)
    assert shell_0.orbital_type == OrbitalType.S
    assert shell_0.atom_index == 0

    # Test adding shell with Shell object - create shells directly in the list
    # Note: add_shell method doesn't exist, so we test shell creation instead
    exponents = np.array([2.0])
    coefficients = np.array([1.0])
    shell_obj = Shell(1, OrbitalType.S, exponents, coefficients)

    # Create a new basis set with additional shell
    all_shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),
        Shell(0, OrbitalType.P, [0.5], [1.0]),
        shell_obj,
    ]
    basis_extended = BasisSet("STO-3G-extended", all_shells)

    assert basis_extended.get_num_shells() == 3
    assert basis_extended.get_num_atoms() == 2  # Now we have atoms 0 and 1


def test_atomic_orbital_type_management():
    """Test basis type management."""
    # Create a valid shell for testing
    shell = Shell(0, OrbitalType.S, [1.0], [1.0])

    # Test default is spherical
    basis_spherical = BasisSet("test", [shell])
    assert basis_spherical.get_atomic_orbital_type() == AOType.Spherical

    # Test cartesian basis type with d orbital shell
    shells = [Shell(0, OrbitalType.D, [1.0], [2.0])]
    basis_cartesian = BasisSet("test", shells, AOType.Cartesian)
    assert basis_cartesian.get_atomic_orbital_type() == AOType.Cartesian

    # For cartesian d orbitals: 6 functions
    assert basis_cartesian.get_num_atomic_orbitals() == 6

    # Create spherical basis set with same shell
    basis_sph_test = BasisSet("test", shells, AOType.Spherical)
    # For spherical d orbitals: 5 functions
    assert basis_sph_test.get_num_atomic_orbitals() == 5


def test_shell_with_raw_primitives():
    """Test creating shells with raw primitive data."""
    # Create primitive data as separate lists
    exponents = [3.42525091, 0.62391373, 0.16885540]
    coefficients = [0.15432897, 0.53532814, 0.44463454]

    # Create shell with multiple primitives
    shells = [Shell(0, OrbitalType.S, exponents, coefficients)]
    basis = BasisSet("cc-pVDZ", shells)

    # Also test with explicit numpy arrays to ensure Eigen::VectorXd overload is called
    exponents_np = np.array([2.1, 0.5, 0.1])
    coefficients_np = np.array([0.6, 0.3, 0.1])
    # Create a new shell instead of using add_shell (which doesn't exist)
    p_shell = Shell(0, OrbitalType.P, exponents_np, coefficients_np)

    # Create a new basis set with both shells
    all_shells = [Shell(0, OrbitalType.S, exponents, coefficients), p_shell]
    basis_extended = BasisSet("cc-pVDZ-extended", all_shells)

    # Test original basis properties
    shell = basis.get_shell(0)
    assert shell.get_num_primitives() == 3
    assert len(shell.exponents) == 3
    assert len(shell.coefficients) == 3
    assert shell.exponents[0] == 3.42525091
    assert shell.coefficients[0] == 0.15432897

    # Test extended basis has both shells
    assert basis_extended.get_num_shells() == 2
    assert basis_extended.get_num_atomic_orbitals() == 4  # 1 s + 3 p functions


def test_orbital_type_enum():
    """Test OrbitalType enum functionality."""
    # Test that enum values work
    assert OrbitalType.S != OrbitalType.P

    # Test utility functions
    assert BasisSet.orbital_type_to_string(OrbitalType.S) == "s"
    assert BasisSet.orbital_type_to_string(OrbitalType.P) == "p"
    assert BasisSet.orbital_type_to_string(OrbitalType.D) == "d"
    # Test higher angular momentum orbitals
    assert BasisSet.orbital_type_to_string(OrbitalType.F) == "f"
    assert BasisSet.orbital_type_to_string(OrbitalType.G) == "g"
    assert BasisSet.orbital_type_to_string(OrbitalType.H) == "h"
    assert BasisSet.orbital_type_to_string(OrbitalType.I) == "i"

    assert BasisSet.string_to_orbital_type("s") == OrbitalType.S
    assert BasisSet.string_to_orbital_type("p") == OrbitalType.P
    assert BasisSet.string_to_orbital_type("d") == OrbitalType.D
    # Test higher angular momentum string conversions
    assert BasisSet.string_to_orbital_type("f") == OrbitalType.F
    assert BasisSet.string_to_orbital_type("g") == OrbitalType.G
    assert BasisSet.string_to_orbital_type("h") == OrbitalType.H
    assert BasisSet.string_to_orbital_type("i") == OrbitalType.I

    # Test angular momentum
    assert BasisSet.get_angular_momentum(OrbitalType.S) == 0
    assert BasisSet.get_angular_momentum(OrbitalType.P) == 1
    assert BasisSet.get_angular_momentum(OrbitalType.D) == 2
    assert BasisSet.get_angular_momentum(OrbitalType.F) == 3
    assert BasisSet.get_angular_momentum(OrbitalType.G) == 4
    assert BasisSet.get_angular_momentum(OrbitalType.H) == 5
    assert BasisSet.get_angular_momentum(OrbitalType.I) == 6

    # Test orbital sizes for different basis types
    assert BasisSet.get_num_orbitals_for_l(0, AOType.Spherical) == 1
    assert BasisSet.get_num_orbitals_for_l(1, AOType.Spherical) == 3
    assert BasisSet.get_num_orbitals_for_l(2, AOType.Spherical) == 5

    assert BasisSet.get_num_orbitals_for_l(0, AOType.Cartesian) == 1
    assert BasisSet.get_num_orbitals_for_l(1, AOType.Cartesian) == 3
    assert BasisSet.get_num_orbitals_for_l(2, AOType.Cartesian) == 6

    # Test basis type string conversion
    assert BasisSet.atomic_orbital_type_to_string(AOType.Spherical) == "spherical"
    assert BasisSet.atomic_orbital_type_to_string(AOType.Cartesian) == "cartesian"

    assert BasisSet.string_to_atomic_orbital_type("spherical") == AOType.Spherical
    assert BasisSet.string_to_atomic_orbital_type("cartesian") == AOType.Cartesian


def test_atomic_orbital_queries():
    """Test atomic orbital indexing and queries."""
    # Create shells for different atoms
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),  # atom 0: 1 s orbital
        Shell(0, OrbitalType.P, [0.5], [1.0]),  # atom 0: 3 p orbitals
        Shell(1, OrbitalType.S, [1.0], [1.0]),  # atom 1: 1 s orbital
    ]
    basis = BasisSet("Test", shells)

    # Total: 5 atomic orbitals (1s + 3p + 1s)
    assert basis.get_num_atomic_orbitals() == 5

    # Test atom indexing
    assert basis.get_atom_index_for_atomic_orbital(0) == 0  # First s orbital on atom 0
    assert basis.get_atom_index_for_atomic_orbital(1) == 0  # First p orbital on atom 0
    assert basis.get_atom_index_for_atomic_orbital(4) == 1  # s orbital on atom 1

    # Test atomic orbital indices for atom
    atom0_indices = basis.get_atomic_orbital_indices_for_atom(0)
    assert atom0_indices == [0, 1, 2, 3]  # 1s + 3p

    atom1_indices = basis.get_atomic_orbital_indices_for_atom(1)
    assert atom1_indices == [4]  # 1s


def test_shell_queries():
    """Test shell indexing and queries."""
    # Create shells
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),  # shell 0
        Shell(0, OrbitalType.P, [0.5], [1.0]),  # shell 1
        Shell(1, OrbitalType.S, [1.0], [1.0]),  # shell 2
    ]
    basis = BasisSet("Test", shells)

    # Test shell indices for atom
    atom0_shells = basis.get_shell_indices_for_atom(0)
    assert atom0_shells == [0, 1]

    atom1_shells = basis.get_shell_indices_for_atom(1)
    assert atom1_shells == [2]

    # Test shell indices for orbital type
    s_shells = basis.get_shell_indices_for_orbital_type(OrbitalType.S)
    assert s_shells == [0, 2]

    p_shells = basis.get_shell_indices_for_orbital_type(OrbitalType.P)
    assert p_shells == [1]

    # Test get_num_atomic_orbitals_for_atom
    assert basis.get_num_atomic_orbitals_for_atom(0) == 4  # 1s + 3p
    assert basis.get_num_atomic_orbitals_for_atom(1) == 1  # 1s

    # Test get_num_atomic_orbitals_for_orbital_type
    assert basis.get_num_atomic_orbitals_for_orbital_type(OrbitalType.S) == 2  # 2 s orbitals
    assert basis.get_num_atomic_orbitals_for_orbital_type(OrbitalType.P) == 3  # 3 p orbitals


def test_atomic_orbital_info():
    """Test atomic orbital info functionality."""
    shells = [Shell(0, OrbitalType.P, [1.0], [1.0])]  # 3 p orbitals
    basis = BasisSet("Test", shells)

    # Test getting atomic orbital info (now returns tuple)
    shell_idx, m_l = basis.get_atomic_orbital_info(1)  # Second p orbital
    assert shell_idx == 0
    assert m_l == 0  # py orbital (m_l = 0)

    # Test index conversions
    shell_idx2, m_l2 = basis.basis_to_shell_index(1)
    assert shell_idx2 == 0
    assert m_l2 == 0

    # Test additional atomic orbital queries
    shell_idx3, m_l3 = basis.get_atomic_orbital_info(0)  # First p orbital
    assert shell_idx3 == 0
    assert m_l3 == -1  # px orbital (m_l = -1)

    shell_idx4, m_l4 = basis.get_atomic_orbital_info(2)  # Third p orbital
    assert shell_idx4 == 0
    assert m_l4 == 1  # pz orbital (m_l = 1)


def test_structure_integration():
    """Test integration with molecular structure."""
    # Create a structure (water molecule)
    coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    nuclear_charges = [8, 1, 1]
    structure = Structure(coords, nuclear_charges)

    # Create shells consistent with structure
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),  # O 1s
        Shell(1, OrbitalType.S, [1.0], [1.0]),  # H 1s
        Shell(2, OrbitalType.S, [1.0], [1.0]),  # H 1s
    ]
    basis = BasisSet("STO-3G", shells, structure)

    assert basis.has_structure()
    retrieved_structure = basis.get_structure()
    assert retrieved_structure.get_num_atoms() == 3


def test_summary():
    """Test summary generation."""
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),
        Shell(0, OrbitalType.P, [0.5], [1.0]),
    ]
    basis = BasisSet("6-31G", shells)

    summary = basis.get_summary()
    assert isinstance(summary, str)
    assert "6-31G" in summary
    assert "shells: 2" in summary
    assert "atomic orbitals: 4" in summary


def test_json_serialization():
    """Test JSON serialization and deserialization."""
    # Create a basis set with data
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),
        Shell(0, OrbitalType.P, [0.5], [1.0]),
    ]
    basis_out = BasisSet("STO-3G", shells)

    # Test direct JSON conversion
    json_data = basis_out.to_json()
    assert isinstance(json_data, str)
    assert "STO-3G" in json_data

    basis_in = BasisSet.from_json(json_data)

    # Verify data was transferred correctly
    assert basis_in.get_name() == "STO-3G"
    assert basis_in.get_num_shells() == 2
    assert basis_in.get_num_atomic_orbitals() == 4

    # Test file-based serialization
    with tempfile.NamedTemporaryFile(suffix=".basis_set.json", mode="w", delete=False) as tmp:
        filename = tmp.name

    try:
        basis_out.to_json_file(filename)

        basis_file = BasisSet.from_json_file(filename)

        # Verify file-based serialization
        assert basis_file.get_name() == "STO-3G"
        assert basis_file.get_num_shells() == 2
        assert basis_file.get_num_atomic_orbitals() == 4
    finally:
        Path(filename).unlink()


def test_hdf5_serialization():
    """Test HDF5 serialization and deserialization."""
    # Create a basis set with data
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),
        Shell(0, OrbitalType.P, [0.5], [1.0]),
    ]
    basis_out = BasisSet("cc-pVDZ", shells, AOType.Spherical)

    try:
        with tempfile.NamedTemporaryFile(suffix=".basis_set.h5", delete=False) as tmp:
            filename = tmp.name

        basis_out.to_hdf5_file(filename)

        basis_in = BasisSet.from_hdf5_file(filename)

        # Verify data transfer
        assert basis_in.get_name() == "cc-pVDZ"
        assert basis_in.get_num_shells() == 2
        assert basis_in.get_num_atomic_orbitals() == 4

    except RuntimeError as e:
        pytest.skip(f"HDF5 test skipped - {e!s}")
    finally:
        if "filename" in locals():
            with contextlib.suppress(FileNotFoundError):
                Path(filename).unlink()


def test_error_handling():
    """Test error handling for invalid operations."""
    # Create a minimal shell for testing empty functions
    shell = Shell(0, OrbitalType.S, [1.0], [1.0])
    basis = BasisSet("Test", [shell])

    # Test invalid indices on basis set - expect IndexError not RuntimeError
    with pytest.raises(IndexError):
        basis.get_atom_index_for_atomic_orbital(10)  # Out of range

    with pytest.raises(IndexError):  # Also expect IndexError here
        basis.get_shell(10)  # Out of range

    with pytest.raises(IndexError):  # Also expect IndexError here
        basis.get_atomic_orbital_info(10)  # Out of range

    # Test invalid string to orbital type conversion - expect ValueError
    with pytest.raises(ValueError):  # noqa: PT011
        BasisSet.string_to_orbital_type("invalid")

    # Test accessing structure when none is set
    with pytest.raises(RuntimeError):
        basis.get_structure()


def test_complete_basis_set_workflow():
    """Test a complete workflow with the BasisSet class."""
    # Create structure for H2
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    nuclear_charges = [1, 1]
    structure = Structure(coords, nuclear_charges)

    # Create minimal basis (1s on each hydrogen)
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),  # H1 1s
        Shell(1, OrbitalType.S, [1.0], [1.0]),  # H2 1s
    ]
    # Create a basis set for a hydrogen molecule with structure
    basis = BasisSet("STO-3G", shells, structure)

    # Verify properties
    assert basis.get_num_atoms() == 2
    assert basis.get_num_shells() == 2
    assert basis.get_num_atomic_orbitals() == 2

    # Test serialization round-trip
    with tempfile.NamedTemporaryFile(suffix=".basis_set.json", mode="w", delete=False) as tmp:
        json_filename = tmp.name

    try:
        # Save to JSON file
        basis.to_json_file(json_filename)

        # Load from JSON file
        basis2 = BasisSet.from_json_file(json_filename)

        # Check equality of key properties
        assert basis2.get_name() == basis.get_name()
        assert basis2.get_num_shells() == basis.get_num_shells()
        assert basis2.get_num_atomic_orbitals() == basis.get_num_atomic_orbitals()

    finally:
        Path(json_filename).unlink()


def test_shell_creation():
    """Test Shell objects with raw primitive data."""
    # Test Shell creation with required parameters
    shell = Shell(0, OrbitalType.S, [1.0], [0.5])
    assert shell.atom_index == 0
    assert shell.orbital_type == OrbitalType.S
    assert shell.get_num_primitives() == 1
    assert len(shell.exponents) == 1
    assert len(shell.coefficients) == 1
    assert shell.exponents[0] == 1.0
    assert shell.coefficients[0] == 0.5

    # Test creating additional shells with all primitives upfront
    shell2 = Shell(0, OrbitalType.S, [1.0, 2.0], [0.5, 0.3])
    assert shell2.get_num_primitives() == 2
    assert len(shell2.exponents) == 2
    assert len(shell2.coefficients) == 2
    assert shell2.get_num_atomic_orbitals() == 1  # s orbital (spherical)
    assert shell2.get_angular_momentum() == 0

    # Test p shell with different basis types
    p_shell = Shell(1, OrbitalType.P, [1.0], [1.0])
    assert p_shell.get_num_atomic_orbitals(AOType.Spherical) == 3  # px, py, pz
    assert p_shell.get_num_atomic_orbitals(AOType.Cartesian) == 3  # same for p
    assert p_shell.get_angular_momentum() == 1

    # Test d shell with different basis types
    d_shell = Shell(2, OrbitalType.D, [1.0], [1.0])
    assert d_shell.get_num_atomic_orbitals(AOType.Spherical) == 5  # spherical d
    assert d_shell.get_num_atomic_orbitals(AOType.Cartesian) == 6  # cartesian d

    # Test shell creation with raw primitive data
    exponents = [1.0, 2.0, 3.0]
    coefficients = [0.1, 0.2, 0.3]
    contracted_shell = Shell(0, OrbitalType.S, exponents, coefficients)
    assert contracted_shell.get_num_primitives() == 3
    assert np.array_equal(contracted_shell.exponents, exponents)
    assert np.array_equal(contracted_shell.coefficients, coefficients)

    # Test with explicit numpy arrays to ensure Eigen::VectorXd overload is called
    exponents_np = np.array([4.0, 5.0])
    coefficients_np = np.array([0.4, 0.6])
    contracted_shell_np = Shell(1, OrbitalType.P, exponents_np, coefficients_np)
    assert contracted_shell_np.get_num_primitives() == 2
    assert np.array_equal(contracted_shell_np.exponents, exponents_np)
    assert np.array_equal(contracted_shell_np.coefficients, coefficients_np)


def test_utility_functions():
    """Test static utility functions."""
    # Test get_num_orbitals_for_l with different basis types
    # Spherical (the default)
    assert BasisSet.get_num_orbitals_for_l(0) == 1  # s: 1 orbital
    assert BasisSet.get_num_orbitals_for_l(1) == 3  # p: 3 orbitals
    assert BasisSet.get_num_orbitals_for_l(2) == 5  # d: 5 orbitals
    assert BasisSet.get_num_orbitals_for_l(3) == 7  # f: 7 orbitals

    # Cartesian
    assert BasisSet.get_num_orbitals_for_l(0, AOType.Cartesian) == 1  # s: 1 orbital
    assert BasisSet.get_num_orbitals_for_l(1, AOType.Cartesian) == 3  # p: 3 orbitals
    assert BasisSet.get_num_orbitals_for_l(2, AOType.Cartesian) == 6  # d: 6 orbitals
    assert BasisSet.get_num_orbitals_for_l(3, AOType.Cartesian) == 10  # f: 10 orbitals


def test_basis_set_file_io_generic():
    """Test generic file I/O methods for BasisSet."""
    # Create shells first
    shells = []

    # Create s shell for atom 0
    s_shell = Shell(0, OrbitalType.S, [1.0], [1.0])
    shells.append(s_shell)

    # Create p shell for atom 0
    p_shell = Shell(0, OrbitalType.P, [0.5], [0.8])
    shells.append(p_shell)

    # Create s shell for atom 1
    s_shell2 = Shell(1, OrbitalType.S, [2.0], [0.9])
    shells.append(s_shell2)

    # Create the basis set with shells
    basis = BasisSet("STO-3G", shells)

    # Test JSON file I/O
    with tempfile.NamedTemporaryFile(suffix=".basis_set.json") as tmp_json:
        json_filename = tmp_json.name

        # Save using generic method
        basis.to_file(json_filename, "json")

        # Load using generic method
        basis2 = BasisSet.from_file(json_filename, "json")

        # Check equality
        assert basis2.get_name() == basis.get_name()
        assert basis2.get_num_shells() == basis.get_num_shells()
        assert basis2.get_num_atomic_orbitals() == basis.get_num_atomic_orbitals()
        assert basis2.get_num_atoms() == basis.get_num_atoms()

    # Test HDF5 file I/O
    with tempfile.NamedTemporaryFile(suffix=".basis_set.h5") as tmp_hdf5:
        hdf5_filename = tmp_hdf5.name

        # Save using generic method
        basis.to_file(hdf5_filename, "hdf5")

        # Load using generic method
        basis3 = BasisSet.from_file(hdf5_filename, "hdf5")

        # Check equality
        assert basis3.get_name() == basis.get_name()
        assert basis3.get_num_shells() == basis.get_num_shells()
        assert basis3.get_num_atomic_orbitals() == basis.get_num_atomic_orbitals()
        assert basis3.get_num_atoms() == basis.get_num_atoms()

    # Test unsupported file type
    with pytest.raises(RuntimeError, match="Unsupported file type"):
        basis.to_file("test.basis_set.xyz", "xyz")

    with pytest.raises(RuntimeError, match="Unsupported file type"):
        basis = BasisSet.from_file("test.basis_set.xyz", "xyz")


def test_basis_set_hdf5_specific():
    """Test specific HDF5 file I/O methods for BasisSet."""
    # Create shells with vector data
    shells = []

    exponents_s = np.array([3.42525091, 0.62391373, 0.16885540])
    coefficients_s = np.array([0.15432897, 0.53532814, 0.44463454])
    s_shell = Shell(0, OrbitalType.S, exponents_s, coefficients_s)
    shells.append(s_shell)

    exponents_p = np.array([2.94124940, 0.68348310, 0.22228990])
    coefficients_p = np.array([-0.09996723, 0.39951283, 0.70011547])
    p_shell = Shell(0, OrbitalType.P, exponents_p, coefficients_p)
    shells.append(p_shell)

    # Create a basis set with multiple shells
    basis = BasisSet("6-31G", shells)

    # Test new to_hdf5_file method
    with tempfile.NamedTemporaryFile(suffix=".basis_set.h5") as tmp_hdf5:
        hdf5_filename = tmp_hdf5.name

        # Save using new method
        basis.to_hdf5_file(hdf5_filename)

        # Load using new method
        basis2 = BasisSet.from_hdf5_file(hdf5_filename)

        # Check equality
        assert basis2.get_name() == basis.get_name()
        assert basis2.get_num_shells() == basis.get_num_shells()
        assert basis2.get_num_atomic_orbitals() == basis.get_num_atomic_orbitals()
        assert basis2.get_num_atoms() == basis.get_num_atoms()

        # Check shells are preserved
        shells_orig = basis.get_shells()
        shells_loaded = basis2.get_shells()

        assert len(shells_orig) == len(shells_loaded)
        for orig, loaded in zip(shells_orig, shells_loaded, strict=True):
            assert orig.atom_index == loaded.atom_index
            assert orig.orbital_type == loaded.orbital_type
            assert np.allclose(
                orig.exponents,
                loaded.exponents,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                orig.coefficients,
                loaded.coefficients,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )


def test_basis_set_json_specific():
    """Test specific JSON file I/O methods for BasisSet."""
    # Create shells with vector data
    shells = []

    exponents_s = np.array([11720.0, 1759.0, 400.8, 113.7, 37.03, 13.27])
    coefficients_s = np.array([0.000710, 0.005470, 0.027837, 0.104800, 0.283062, 0.448719])
    s_shell = Shell(0, OrbitalType.S, exponents_s, coefficients_s)
    shells.append(s_shell)

    exponents_p = np.array([17.70, 3.854, 1.046, 0.2753])
    coefficients_p = np.array([0.043018, 0.228913, 0.508728, 0.460531])
    p_shell = Shell(0, OrbitalType.P, exponents_p, coefficients_p)
    shells.append(p_shell)

    # Create a basis set with multiple shells
    basis = BasisSet("cc-pVDZ", shells, AOType.Cartesian)

    # Test updated JSON file I/O methods
    with tempfile.NamedTemporaryFile(suffix=".basis_set.json") as tmp_json:
        json_filename = tmp_json.name

        # Save using to_json_file method
        basis.to_json_file(json_filename)

        # Load using from_json_file method
        basis2 = BasisSet.from_json_file(json_filename)

        # Check equality
        assert basis2.get_name() == basis.get_name()
        assert basis2.get_atomic_orbital_type() == basis.get_atomic_orbital_type()
        assert basis2.get_num_shells() == basis.get_num_shells()
        assert basis2.get_num_atomic_orbitals() == basis.get_num_atomic_orbitals()
        assert basis2.get_num_atoms() == basis.get_num_atoms()

        # Check shells are preserved
        shells_orig = basis.get_shells()
        shells_loaded = basis2.get_shells()

        assert len(shells_orig) == len(shells_loaded)
        for orig, loaded in zip(shells_orig, shells_loaded, strict=True):
            assert orig.atom_index == loaded.atom_index
            assert orig.orbital_type == loaded.orbital_type
            assert np.allclose(
                orig.exponents,
                loaded.exponents,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                orig.coefficients,
                loaded.coefficients,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )


def test_basis_set_file_io_validation():
    """Test filename validation for BasisSet file I/O."""
    # Create a shell first
    s_shell = Shell(0, OrbitalType.S, [1.0], [1.0])

    # Create basis with shell
    basis = BasisSet("STO-3G", [s_shell])

    # Test filename validation for JSON files
    with pytest.raises(ValueError, match=re.escape("'.basis_set.' before the file extension")):
        basis.to_json_file("test.json")

    with pytest.raises(ValueError, match=re.escape("'.basis_set.' before the file extension")):
        basis = BasisSet.from_json_file("test.json")

    # Test filename validation for HDF5 files
    with pytest.raises(ValueError, match=re.escape("'.basis_set.' before the file extension")):
        basis.to_hdf5_file("test.h5")

    with pytest.raises(ValueError, match=re.escape("'.basis_set.' before the file extension")):
        basis = BasisSet.from_hdf5_file("test.h5")

    # Test non-existent file
    with pytest.raises(RuntimeError, match="Unable to open BasisSet JSON file"):
        basis = BasisSet.from_json_file("nonexistent.basis_set.json")

    with pytest.raises(RuntimeError):
        basis = BasisSet.from_hdf5_file("nonexistent.basis_set.h5")


def test_basis_set_file_io_round_trip():
    """Test round-trip file I/O preserves data integrity."""
    # Create shells for different atoms
    shells = []

    # Atom 0: H-like with s and p shells
    s_shell0 = Shell(0, OrbitalType.S, np.array([1.24, 0.32]), np.array([0.6, 0.4]))
    shells.append(s_shell0)
    p_shell0 = Shell(0, OrbitalType.P, np.array([0.8, 0.2]), np.array([0.7, 0.3]))
    shells.append(p_shell0)

    # Atom 1: C-like with s, p, and d shells
    s_shell1 = Shell(
        1,
        OrbitalType.S,
        np.array([5.033151, 1.254400, 0.331126]),
        np.array([0.156285, 0.607684, 0.391957]),
    )
    shells.append(s_shell1)
    p_shell1 = Shell(
        1,
        OrbitalType.P,
        np.array([2.941249, 0.683483, 0.222289]),
        np.array([0.156285, 0.607684, 0.391957]),
    )
    shells.append(p_shell1)
    d_shell1 = Shell(1, OrbitalType.D, np.array([0.8, 0.2]), np.array([0.5, 0.5]))
    shells.append(d_shell1)

    # Create a complex basis set with all shells
    basis = BasisSet("complex-basis", shells, AOType.Spherical)

    # Test JSON round-trip
    with tempfile.NamedTemporaryFile(suffix=".basis_set.json") as tmp_json:
        json_filename = tmp_json.name

        # Save and reload
        basis.to_json_file(json_filename)
        basis_json = BasisSet.from_json_file(json_filename)

        # Check all properties are preserved
        assert basis_json.get_name() == basis.get_name()
        assert basis_json.get_atomic_orbital_type() == basis.get_atomic_orbital_type()
        assert basis_json.get_num_shells() == basis.get_num_shells()
        assert basis_json.get_num_atomic_orbitals() == basis.get_num_atomic_orbitals()
        assert basis_json.get_num_atoms() == basis.get_num_atoms()

        # Check shells in detail
        shells_orig = basis.get_shells()
        shells_json = basis_json.get_shells()

        assert len(shells_orig) == len(shells_json)
        for orig, loaded in zip(shells_orig, shells_json, strict=True):
            assert orig.atom_index == loaded.atom_index
            assert orig.orbital_type == loaded.orbital_type
            assert np.allclose(
                orig.exponents,
                loaded.exponents,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                orig.coefficients,
                loaded.coefficients,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

    # Test HDF5 round-trip
    with tempfile.NamedTemporaryFile(suffix=".basis_set.h5") as tmp_hdf5:
        hdf5_filename = tmp_hdf5.name

        # Save and reload
        basis.to_hdf5_file(hdf5_filename)
        basis_hdf5 = BasisSet.from_hdf5_file(hdf5_filename)

        # Check all properties are preserved
        assert basis_hdf5.get_name() == basis.get_name()
        assert basis_hdf5.get_num_shells() == basis.get_num_shells()
        assert basis_hdf5.get_num_atomic_orbitals() == basis.get_num_atomic_orbitals()
        assert basis_hdf5.get_num_atoms() == basis.get_num_atoms()

        # Check shells in detail
        shells_hdf5 = basis_hdf5.get_shells()

        assert len(shells_orig) == len(shells_hdf5)
        for orig, loaded in zip(shells_orig, shells_hdf5, strict=True):
            assert orig.atom_index == loaded.atom_index
            assert orig.orbital_type == loaded.orbital_type
            assert np.allclose(
                orig.exponents,
                loaded.exponents,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                orig.coefficients,
                loaded.coefficients,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )


def test_basis_set_consistency_between_methods():
    """Test that generic and specific methods produce consistent results."""
    # Create shells
    shells = []

    s_shell = Shell(0, OrbitalType.S, np.array([2.0, 0.5]), np.array([0.8, 0.2]))
    shells.append(s_shell)
    p_shell = Shell(0, OrbitalType.P, np.array([1.5, 0.3]), np.array([0.7, 0.3]))
    shells.append(p_shell)

    # Create basis with shells
    basis = BasisSet("test-consistency", shells)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Test JSON consistency
        json_file1 = tmpdir_path / "test1.basis_set.json"
        json_file2 = tmpdir_path / "test2.basis_set.json"

        # Save with generic method
        basis.to_file(json_file1, "json")
        # Save with specific method
        basis.to_json_file(json_file2)

        # Files should be identical (or at least functionally equivalent)
        basis1 = BasisSet.from_file(json_file1, "json")
        basis2 = BasisSet.from_json_file(json_file2)

        # Check both loaded basis sets are equivalent
        assert basis1.get_name() == basis2.get_name()
        assert basis1.get_num_shells() == basis2.get_num_shells()
        assert basis1.get_num_atomic_orbitals() == basis2.get_num_atomic_orbitals()

        # Test HDF5 consistency
        hdf5_file1 = tmpdir_path / "test1.basis_set.h5"
        hdf5_file2 = tmpdir_path / "test2.basis_set.h5"

        # Save with generic method
        basis.to_file(hdf5_file1, "hdf5")
        # Save with specific method
        basis.to_hdf5_file(hdf5_file2)

        # Load with both methods and verify consistency
        basis3 = BasisSet.from_file(hdf5_file1, "hdf5")
        basis4 = BasisSet.from_hdf5_file(hdf5_file2)

        # Check both loaded basis sets are equivalent
        assert basis3.get_name() == basis4.get_name()
        assert basis3.get_num_shells() == basis4.get_num_shells()
        assert basis3.get_num_atomic_orbitals() == basis4.get_num_atomic_orbitals()


def test_basis_set_metadata_management():
    """Test BasisSet metadata and name management."""
    shells = []
    s_shell = Shell(0, OrbitalType.S, np.array([2.0, 0.5]), np.array([0.8, 0.2]))
    shells.append(s_shell)
    p_shell = Shell(0, OrbitalType.P, np.array([1.5, 0.3]), np.array([0.7, 0.3]))
    shells.append(p_shell)

    # Test constructor with minimal shell list (default constructor doesn't exist)
    basis = BasisSet("", shells)
    assert basis.get_name() == ""

    # Test by creating a copy and modifying name (set_name method may not exist)
    # Instead test constructor with name
    basis_named = BasisSet("Custom-Basis", shells)
    assert basis_named.get_name() == "Custom-Basis"

    # Test constructor with name and basis type
    basis_with_type = BasisSet("6-31G", shells, AOType.Cartesian)
    assert basis_with_type.get_name() == "6-31G"
    assert basis_with_type.get_atomic_orbital_type() == AOType.Cartesian

    # Test copy constructor
    basis_copy = BasisSet(basis_with_type)
    assert basis_copy.get_name() == "6-31G"
    assert basis_copy.get_atomic_orbital_type() == AOType.Cartesian

    # Test structure management - create a minimal structure first
    coords = np.array([[0.0, 0.0, 0.0]])
    symbols = ["H"]
    structure = Structure(coords, symbols)

    # Test basis set with structure
    basis_with_structure = BasisSet("H-basis", shells, structure)
    assert basis_with_structure.has_structure()
    retrieved_structure = basis_with_structure.get_structure()
    assert retrieved_structure.get_num_atoms() == 1


def test_keyword_arguments():
    """Test using keyword arguments to ensure py::arg calls are covered."""
    # Test Shell constructor with keyword args - all parameters are required
    exponents = np.array([1.0])
    coefficients = np.array([1.0])
    shell1 = Shell(atom_index=0, orbital_type=OrbitalType.S, exponents=exponents, coefficients=coefficients)
    assert shell1.atom_index == 0
    assert shell1.orbital_type == OrbitalType.S

    # Test Shell constructor with vector data using keyword args
    exponents_vec = np.array([1.0, 2.0])
    coefficients_vec = np.array([0.5, 0.5])
    shell2 = Shell(
        atom_index=1,
        orbital_type=OrbitalType.P,
        exponents=exponents_vec,
        coefficients=coefficients_vec,
    )
    assert shell2.atom_index == 1
    assert shell2.get_num_primitives() == 2

    # Test BasisSet constructor with keyword args (need shells parameter)
    shells = [shell1]
    basis1 = BasisSet(name="STO-3G", shells=shells)
    assert basis1.get_name() == "STO-3G"

    # Test BasisSet constructor with name and atomic_orbital_type using keyword args
    basis2 = BasisSet(name="6-31G", shells=shells, atomic_orbital_type=AOType.Cartesian)
    assert basis2.get_name() == "6-31G"
    assert basis2.get_atomic_orbital_type() == AOType.Cartesian

    # Note: set_atomic_orbital_type and add_shell methods don't exist on BasisSet
    # Test get_atomic_orbital_type instead
    assert basis1.get_atomic_orbital_type() == AOType.Spherical  # Default
    assert basis2.get_atomic_orbital_type() == AOType.Cartesian

    # Test shell count
    assert basis1.get_num_shells() == 1


def test_explicit_pyarg_coverage():
    """Explicit test to ensure all py::arg calls are executed."""
    # Ensure py::arg calls are covered by using keyword arguments

    # Shell(atom_index, orbital_type, exponents, coefficients) py::arg - all parameters required
    exponents = np.array([1.0])
    coefficients = np.array([1.0])
    shell_basic = Shell(atom_index=0, orbital_type=OrbitalType.S, exponents=exponents, coefficients=coefficients)

    # Shell(atom_index, orbital_type, exponents, coefficients) py::arg
    exponents_vec = np.array([1.0, 2.0])
    coefficients_vec = np.array([0.5, 0.5])
    shell_vector = Shell(
        atom_index=1,
        orbital_type=OrbitalType.P,
        exponents=exponents_vec,
        coefficients=coefficients_vec,
    )

    # BasisSet(name, shells) py::arg - shells parameter is required
    shells = [shell_basic]
    basis1 = BasisSet(name="Test-Basis", shells=shells)

    # BasisSet(name, shells, atomic_orbital_type) py::arg
    basis2 = BasisSet(name="Test-Basis-2", shells=shells, atomic_orbital_type=AOType.Cartesian)

    # Verify the objects were created correctly
    assert basis1.get_name() == "Test-Basis"
    assert basis2.get_name() == "Test-Basis-2"
    assert shell_basic.atom_index == 0
    assert shell_vector.atom_index == 1


def test_explicit_shell_constructor_coverage():
    """Explicitly test Shell constructor with Eigen::VectorXd."""
    # Ensure we're calling the 4-parameter constructor
    # Use numpy arrays to ensure Eigen::VectorXd conversion
    exponents = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    coefficients = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    shell = Shell(0, OrbitalType.S, exponents, coefficients)

    assert shell.atom_index == 0
    assert shell.orbital_type == OrbitalType.S
    assert shell.get_num_primitives() == 3

    # Test creating another shell (add_primitive method doesn't exist)
    p_exponents = np.array([4.0, 5.0], dtype=np.float64)
    p_coefficients = np.array([0.4, 0.5], dtype=np.float64)
    shell_p = Shell(1, OrbitalType.P, p_exponents, p_coefficients)

    assert shell_p.get_num_primitives() == 2
    assert shell_p.exponents[0] == 4.0
    assert shell_p.coefficients[0] == 0.4


def test_basis_set_pickling_and_repr():
    """Test pickling support and string representation for BasisSet."""
    # Create a test basis set
    shells = [
        Shell(0, OrbitalType.S, [3.425251, 0.623914, 0.168855], [0.154329, 0.535328, 0.444635]),
        Shell(0, OrbitalType.P, [1.158, 0.325], [0.155916, 0.607684]),
    ]
    original = BasisSet("STO-3G", shells)

    # Test pickling and unpickling
    pickled_data = pickle.dumps(original)
    assert isinstance(pickled_data, bytes)

    # Unpickle and verify
    unpickled = pickle.loads(pickled_data)

    # Verify basis set is preserved
    assert unpickled.get_name() == original.get_name()
    assert unpickled.get_num_shells() == original.get_num_shells()
    assert unpickled.get_num_atomic_orbitals() == original.get_num_atomic_orbitals()
    assert unpickled.get_atomic_orbital_type() == original.get_atomic_orbital_type()

    # Verify shells are preserved
    original_shells = original.get_shells()
    unpickled_shells = unpickled.get_shells()
    assert len(original_shells) == len(unpickled_shells)

    for orig_shell, unpick_shell in zip(original_shells, unpickled_shells, strict=True):
        assert orig_shell.atom_index == unpick_shell.atom_index
        assert orig_shell.orbital_type == unpick_shell.orbital_type
        assert orig_shell.get_num_primitives() == unpick_shell.get_num_primitives()
        assert np.allclose(
            orig_shell.exponents,
            unpick_shell.exponents,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_shell.coefficients,
            unpick_shell.coefficients,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    # Test multiple round trips
    current = original
    for _ in range(3):
        pickled = pickle.dumps(current)
        current = pickle.loads(pickled)

        # Verify data integrity after each round trip
        assert current.get_name() == original.get_name()
        assert current.get_num_shells() == original.get_num_shells()
        assert current.get_num_atomic_orbitals() == original.get_num_atomic_orbitals()

    # Test __repr__ uses summary function
    repr_output = repr(original)
    summary_output = original.get_summary()

    # They should be the same
    assert repr_output == summary_output

    # Check that typical summary content is present
    assert "Basis set" in repr_output or "STO-3G" in repr_output

    # Test __str__ vs __repr__ consistency
    str_output = str(original)
    assert str_output == repr_output

    # Test that pickled objects still have proper repr functionality
    pickled_data = pickle.dumps(original)
    unpickled = pickle.loads(pickled_data)

    # Both should have same repr output
    original_repr = repr(original)
    unpickled_repr = repr(unpickled)

    assert original_repr == unpickled_repr


def test_basis_set_ecp_functionality():
    """Test basis set ECP (Effective Core Potential) functionality."""
    # Create a structure with multiple atoms
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = ["Cu", "O", "H"]
    structure = Structure(elements, positions)

    # Create shells for each atom
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),
        Shell(1, OrbitalType.S, [1.0], [1.0]),
        Shell(2, OrbitalType.S, [1.0], [1.0]),
    ]

    # Create the basis set
    basis = BasisSet("test-basis", shells, structure)

    # Test default ECP state
    assert not basis.has_ecp_electrons()
    assert basis.get_ecp_name() == "none"
    assert basis.get_ecp_electrons() == [0, 0, 0]

    # Test creating ECP with constructor
    ecp_name = "cc-pVDZ-PP"
    ecp_electrons = [10, 2, 0]
    # Create ECP shells for Cu (atom 0) and O (atom 1)
    ecp_shells = [
        Shell(0, OrbitalType.S, [2.0, 1.5], [0.5, 0.5]),  # Cu ECP S-shell
        Shell(0, OrbitalType.P, [2.0, 1.5], [0.5, 0.5]),  # Cu ECP P-shell
        Shell(1, OrbitalType.S, [1.0], [1.0]),  # O ECP S-shell
    ]
    basis_with_ecp = BasisSet("test-basis", shells, ecp_name, ecp_shells, ecp_electrons, structure)

    # Test getting ECP
    assert basis_with_ecp.has_ecp_electrons()
    assert basis_with_ecp.get_ecp_name() == ecp_name
    assert list(basis_with_ecp.get_ecp_electrons()) == ecp_electrons

    # Test ECP validation (wrong vector size should raise a ValueError)
    with pytest.raises(ValueError, match=r"ECP electrons vector size must match number of atoms"):
        BasisSet("test-basis", shells, "test-basis", ecp_shells, [10], structure)  # Only 1 element, but we have 3 atoms

    # Test ECP with JSON serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = str(Path(tmpdir) / "test_ecp.basis_set.json")
        basis_with_ecp.to_json_file(json_file)

        # Load and verify
        loaded_basis = BasisSet.from_json_file(json_file)
        assert loaded_basis.has_ecp_electrons()
        assert loaded_basis.get_ecp_name() == ecp_name
        assert list(loaded_basis.get_ecp_electrons()) == ecp_electrons

    # Test ECP with HDF5 serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_file = str(Path(tmpdir) / "test_ecp.basis_set.h5")
        basis_with_ecp.to_hdf5_file(hdf5_file)

        # Load and verify
        loaded_basis = BasisSet.from_hdf5_file(hdf5_file)
        assert loaded_basis.has_ecp_electrons()
        assert loaded_basis.get_ecp_name() == ecp_name
        assert list(loaded_basis.get_ecp_electrons()) == ecp_electrons

    # Test ECP with copy constructor
    basis_copy = BasisSet(basis_with_ecp)
    assert basis_copy.has_ecp_electrons()
    assert basis_copy.get_ecp_name() == ecp_name
    assert list(basis_copy.get_ecp_electrons()) == ecp_electrons


def test_basis_set_ecp_shells():
    """Test basis set ECP shells with radial powers."""
    # Create a structure with an atom that uses ECP
    positions = np.array([[0.0, 0.0, 0.0]])
    elements = ["Ag"]
    structure = Structure(elements, positions)

    # Create regular shells
    shells = [Shell(0, OrbitalType.S, [1.0], [1.0])]

    # Create ECP shells with radial powers
    ecp_exponents = np.array([10.0, 5.0, 2.0])
    ecp_coefficients = np.array([50.0, 20.0, 10.0])
    ecp_rpowers = np.array([0, 1, 2], dtype=np.int32)

    ecp_shell_s = Shell(0, OrbitalType.S, ecp_exponents, ecp_coefficients, ecp_rpowers)

    ecp_shell_p = Shell(0, OrbitalType.P, [8.0], [30.0], [1])

    ecp_shells = [ecp_shell_s, ecp_shell_p]

    # Create basis set with ECP shells
    basis = BasisSet("test-basis", shells, ecp_shells, structure)

    # Test ECP shell queries
    assert basis.has_ecp_shells()
    assert basis.get_num_ecp_shells() == 2

    # Test retrieving all ECP shells
    all_ecp_shells = basis.get_ecp_shells()
    assert len(all_ecp_shells) == 2

    # Test retrieving ECP shells for specific atom
    ecp_shells_atom0 = basis.get_ecp_shells_for_atom(0)
    assert len(ecp_shells_atom0) == 2
    assert ecp_shells_atom0[0].orbital_type == OrbitalType.S
    assert ecp_shells_atom0[1].orbital_type == OrbitalType.P

    # Test ECP shell properties
    shell_s = basis.get_ecp_shell(0)
    assert shell_s.atom_index == 0
    assert shell_s.orbital_type == OrbitalType.S
    assert shell_s.has_radial_powers()
    assert len(shell_s.rpowers) == 3
    assert np.array_equal(shell_s.rpowers, [0, 1, 2])
    assert np.array_equal(shell_s.exponents, [10.0, 5.0, 2.0])
    assert np.array_equal(shell_s.coefficients, [50.0, 20.0, 10.0])

    # Test ECP shell without radial powers (regular shell)
    regular_shell = shells[0]
    assert not regular_shell.has_radial_powers()


def test_basis_set_ecp_shells_serialization():
    """Test ECP shells serialization and deserialization."""
    # Create structure
    positions = np.array([[0.0, 0.0, 0.0]])
    elements = ["Ag"]
    structure = Structure(elements, positions)

    # Create shells and ECP shells
    shells = [Shell(0, OrbitalType.S, [1.0], [1.0])]
    ecp_shells = [Shell(0, OrbitalType.S, [10.0, 5.0], [50.0, 20.0], [0, 2])]

    # Create basis set with ECP shells and ECP metadata
    basis = BasisSet("test-basis", shells, "test-ecp", ecp_shells, [28], structure)

    # Test JSON serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = str(Path(tmpdir) / "test_ecp_shells.basis_set.json")
        basis.to_json_file(json_file)

        # Load and verify
        loaded_basis = BasisSet.from_json_file(json_file)
        assert loaded_basis.has_ecp_shells()
        assert loaded_basis.get_num_ecp_shells() == 1

        loaded_shell = loaded_basis.get_ecp_shell(0)
        assert loaded_shell.has_radial_powers()
        assert np.array_equal(loaded_shell.rpowers, [0, 2])
        assert np.array_equal(loaded_shell.exponents, [10.0, 5.0])
        assert np.array_equal(loaded_shell.coefficients, [50.0, 20.0])

    # Test HDF5 serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_file = str(Path(tmpdir) / "test_ecp_shells.basis_set.h5")
        basis.to_hdf5_file(hdf5_file)

        # Load and verify
        loaded_basis = BasisSet.from_hdf5_file(hdf5_file)
        assert loaded_basis.has_ecp_shells()
        assert loaded_basis.get_num_ecp_shells() == 1

        loaded_shell = loaded_basis.get_ecp_shell(0)
        assert loaded_shell.has_radial_powers()
        assert np.array_equal(loaded_shell.rpowers, [0, 2])


def test_basis_set_ecp_shells_copy():
    """Test that ECP shells are properly copied."""
    # Create structure and shells
    positions = np.array([[0.0, 0.0, 0.0]])
    elements = ["Ag"]
    structure = Structure(elements, positions)

    shells = [Shell(0, OrbitalType.S, [1.0], [1.0])]
    ecp_shells = [Shell(0, OrbitalType.S, [10.0, 5.0], [50.0, 20.0], [0, 2])]

    basis = BasisSet("test-basis", shells, "test-ecp", ecp_shells, [28], structure)

    # Test copy constructor
    basis_copy = BasisSet(basis)
    assert basis_copy.has_ecp_shells()
    assert basis_copy.get_num_ecp_shells() == basis.get_num_ecp_shells()

    # Verify ECP shell data is copied
    orig_shell = basis.get_ecp_shell(0)
    copy_shell = basis_copy.get_ecp_shell(0)
    assert copy_shell.has_radial_powers() == orig_shell.has_radial_powers()
    assert np.array_equal(copy_shell.rpowers, orig_shell.rpowers)
    assert np.array_equal(copy_shell.exponents, orig_shell.exponents)
    assert np.array_equal(copy_shell.coefficients, orig_shell.coefficients)


def test_basis_set_ecp_shells_multi_atom():
    """Test ECP shells in multi-atom systems."""
    # Create structure with multiple atoms
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    elements = ["Ag", "Au", "H"]
    structure = Structure(elements, positions)

    # Create regular shells for all atoms
    shells = [
        Shell(0, OrbitalType.S, [1.0], [1.0]),
        Shell(1, OrbitalType.S, [1.0], [1.0]),
        Shell(2, OrbitalType.S, [1.0], [1.0]),
    ]

    # Create ECP shells for Ag (atom 0) and Au (atom 1), but not H (atom 2)
    ecp_shells = [
        Shell(0, OrbitalType.S, [10.0], [50.0], [0]),
        Shell(0, OrbitalType.P, [8.0], [30.0], [1]),
        Shell(1, OrbitalType.D, [12.0], [40.0], [2]),
    ]

    basis = BasisSet("test-basis", shells, ecp_shells, structure)

    # Test total ECP shells
    assert basis.get_num_ecp_shells() == 3

    # Test ECP shells per atom
    ecp_shells_ag = basis.get_ecp_shells_for_atom(0)
    assert len(ecp_shells_ag) == 2
    assert ecp_shells_ag[0].orbital_type == OrbitalType.S
    assert ecp_shells_ag[1].orbital_type == OrbitalType.P

    ecp_shells_au = basis.get_ecp_shells_for_atom(1)
    assert len(ecp_shells_au) == 1
    assert ecp_shells_au[0].orbital_type == OrbitalType.D

    ecp_shells_h = basis.get_ecp_shells_for_atom(2)
    assert len(ecp_shells_h) == 0


def test_basis_set_from_basis_name():
    """Test creating basis set using from_basis_name static method."""
    # Create water structure
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements = ["H", "O", "H"]
    structure = Structure(elements, positions)

    # Create basis set using from_basis_name
    basis_set = "sto-3g"
    basis = BasisSet.from_basis_name(basis_set, structure)

    # Verify basis set properties
    assert basis.get_name() == basis_set
    assert basis.get_num_shells() == 5

    # Run SCF calculation to verify the basis set works
    scf_solver = alg.create("scf_solver")
    energy, determinant = scf_solver.run(structure, 0, 1, basis)

    # Check number of orbitals
    num_orbitals = determinant.get_orbitals().get_num_molecular_orbitals()
    assert num_orbitals == 7


def test_basis_set_from_element_map():
    """Test creating basis set using from_element_map static method."""
    # Create water structure
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements = ["H", "O", "H"]
    structure = Structure(elements, positions)

    # Create element-to-basis map
    element_basis_map = {"H": "cc-pvdz", "O": "sto-3g"}

    # Create basis set using from_element_map
    basis = BasisSet.from_element_map(element_basis_map, structure)

    # Verify basis set properties
    assert basis.get_name() == "custom_basis_set"
    assert basis.get_num_shells() == 9

    # Run SCF calculation to verify the basis set works
    scf_solver = alg.create("scf_solver")
    energy, determinant = scf_solver.run(structure, 0, 1, basis)

    # Check number of orbitals
    num_orbitals = determinant.get_orbitals().get_num_molecular_orbitals()
    assert num_orbitals == 15


def test_basis_set_from_index_map():
    """Test creating basis set using from_index_map static method."""
    # Create water structure
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements = ["H", "O", "H"]
    structure = Structure(elements, positions)

    # Create index-to-basis map with mixed basis sets
    index_basis_map = {0: "cc-pvtz", 1: "sto-3g", 2: "def2-svp"}

    # Create basis set using from_index_map
    basis = BasisSet.from_index_map(index_basis_map, structure)

    # Verify basis set properties
    assert basis.get_name() == "custom_basis_set"
    assert basis.get_num_shells() == 12

    # Run SCF calculation to verify the basis set works
    scf_solver = alg.create("scf_solver")
    energy, determinant = scf_solver.run(structure, 0, 1, basis)

    # Check number of orbitals
    num_orbitals = determinant.get_orbitals().get_num_molecular_orbitals()
    assert num_orbitals == 24


def test_basis_set_static_constants():
    """Test that static constant variables are accessible."""
    # Test that the static constants exist and have expected values
    assert BasisSet.custom_name == "custom_basis_set"
    assert BasisSet.custom_ecp_name == "custom_ecp"
    assert BasisSet.default_ecp_name == "default_ecp"


def test_get_supported_basis_set_names():
    """Test get_supported_basis_set_names static method."""
    # Get list of supported basis sets
    supported = BasisSet.get_supported_basis_set_names()

    # Verify it returns a list
    assert isinstance(supported, list)

    # Verify it's not empty
    assert len(supported) > 0

    # Verify some common basis sets are in the list
    assert "sto-3g" in supported
    assert "cc-pvdz" in supported
    assert "6-31g" in supported


def test_get_supported_elements_for_basis_set():
    """Test get_supported_elements_for_basis_set static method."""
    elements = BasisSet.get_supported_elements_for_basis_set("sto-3g")

    # Verify it returns a list
    assert isinstance(elements, list)

    # Verify it's not empty
    assert len(elements) > 0

    # Verify common elements are supported in STO-3G
    assert Element.H in elements
    assert Element.C in elements
    assert Element.O in elements
    assert Element.N in elements
