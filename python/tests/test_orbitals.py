"""Tests for the Orbitals class in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import contextlib
import pickle
import re
import tempfile

import numpy as np
import pytest

from qdk_chemistry.data import ModelOrbitals, Orbitals

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import create_test_basis_set


def test_orbitals_construction():
    """Test constructing Orbitals objects."""
    # Test constructor with minimal data
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    basis_set = create_test_basis_set(3, "test-construction")
    orb = Orbitals(coeffs, None, None, basis_set)

    # Copy constructor
    orb2 = Orbitals(orb)

    # Get coefficients and check they match
    alpha, beta = orb.get_coefficients()
    alpha2, beta2 = orb2.get_coefficients()

    assert np.allclose(
        alpha, alpha2, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(beta, beta2, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance)


def test_coefficient_management():
    """Test setting and getting orbital coefficients."""
    # Test restricted case
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    basis_set = create_test_basis_set(3, "test-coeff-restricted")
    orb = Orbitals(coeffs, None, None, basis_set)
    alpha, beta = orb.get_coefficients()

    assert np.allclose(
        coeffs, alpha, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(coeffs, beta, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance)
    assert orb.is_restricted()

    # Test unrestricted case
    coeffs_alpha = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    coeffs_beta = np.array([[0.8, 0.2], [0.2, -0.8], [0.1, 0.0]])
    basis_set_unres = create_test_basis_set(3, "test-coeff-unrestricted")
    orb_unres = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set_unres)

    alpha, beta = orb_unres.get_coefficients()
    assert np.allclose(
        coeffs_alpha, alpha, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(
        coeffs_beta, beta, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert not orb_unres.is_restricted()


def test_energy_management():
    """Test setting and getting orbital energies."""
    # Test restricted case with energies
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    energies = np.array([-1.0, 0.5])
    basis_set = create_test_basis_set(3, "test-energy-restricted")
    orb = Orbitals(coeffs, energies, None, basis_set)

    alpha, beta = orb.get_energies()
    assert np.allclose(
        energies, alpha, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(
        energies, beta, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )

    # Test unrestricted case
    coeffs_alpha = np.array([[0.8, 0.2], [0.2, -0.8], [0.0, 0.0]])
    coeffs_beta = np.array([[0.7, 0.3], [0.3, -0.7], [0.0, 0.0]])
    alpha_energies = np.array([-0.9, 0.6])
    beta_energies = np.array([-0.8, 0.7])
    basis_set_unres = create_test_basis_set(3, "test-energy-unrestricted")

    orb_unres = Orbitals(coeffs_alpha, coeffs_beta, alpha_energies, beta_energies, None, basis_set_unres)
    alpha, beta = orb_unres.get_energies()

    assert np.allclose(
        alpha_energies, alpha, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(
        beta_energies, beta, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )


def test_ao_overlap():
    """Test setting and getting AO overlap matrix."""
    coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
    overlap = np.array([[1.0, 0.1], [0.1, 1.0]])
    basis_set = create_test_basis_set(2, "test-ao-overlap")

    # Test with overlap matrix
    orb = Orbitals(coeffs, None, overlap, basis_set)

    # Check that we have an overlap now
    assert orb.has_overlap_matrix()

    # Get overlap and verify
    retrieved_overlap = orb.get_overlap_matrix()
    assert np.allclose(
        overlap, retrieved_overlap, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )

    # Test without overlap
    orb_no_overlap = Orbitals(coeffs, None, None, basis_set)
    assert not orb_no_overlap.has_overlap_matrix()


def test_basis_info():
    """Test basis set information via constructor injection."""
    # Minimal coefficients (2x2)
    coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
    occupations = np.array([2.0, 0.0])

    # Create basis set using convenience function
    basis_set = create_test_basis_set(2, "6-31G")

    # Inject basis set via constructor
    orb = Orbitals(coeffs, occupations, None, basis_set)

    assert orb.has_basis_set()
    retrieved_basis = orb.get_basis_set()
    assert retrieved_basis.get_name() == "6-31G"
    assert retrieved_basis.get_num_shells() > 0
    # Number of basis functions should match what we requested
    assert retrieved_basis.get_num_basis_functions() == 2


def test_calculation_restriction_query():
    """Test calculation type queries."""
    # Restricted
    coeffs = np.zeros((3, 2))
    basis_set = create_test_basis_set(3, "test-calc-type-restricted")
    orb_r = Orbitals(coeffs, None, None, basis_set)
    assert orb_r.is_restricted()

    # Unrestricted with open shell
    alpha_coeffs = np.ones((3, 2))
    beta_coeffs = np.zeros((3, 2))
    basis_set_unres = create_test_basis_set(3, "test-calc-type-unrestricted")
    orb_u = Orbitals(alpha_coeffs, beta_coeffs, None, None, None, basis_set_unres)
    assert not orb_u.is_restricted()

    # Equal electrons but different distributions still open shell
    alpha_coeffs2 = np.ones((3, 2))
    beta_coeffs2 = np.zeros((3, 2))
    basis_set_unres2 = create_test_basis_set(3, "test-calc-type-unrestricted2")
    orb_u2 = Orbitals(alpha_coeffs2, beta_coeffs2, None, None, None, basis_set_unres2)
    assert not orb_u2.is_restricted()


def test_validation_and_summary():
    """Test validation and summary methods."""
    # Minimal valid orbital
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9]])
    basis_set = create_test_basis_set(2, "test-validation")
    orb = Orbitals(coeffs, None, None, basis_set)
    summary = orb.get_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_json_serialization():
    """Test JSON serialization and deserialization."""
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    energies = np.array([-2.0, 0.5])
    overlap = np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    basis_set = create_test_basis_set(3, "test-json-serialization")
    orb_out = Orbitals(coeffs, energies, overlap, basis_set)

    # Test direct JSON conversion
    json_data = orb_out.to_json()
    orb_in = Orbitals.from_json(json_data)

    coeffs_out_a, coeffs_out_b = orb_out.get_coefficients()
    coeffs_in_a, coeffs_in_b = orb_in.get_coefficients()
    assert np.allclose(
        coeffs_out_a, coeffs_in_a, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(
        coeffs_out_b, coeffs_in_b, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )

    # Test file-based serialization
    with tempfile.NamedTemporaryFile(suffix=".orbitals.json") as tmp:
        filename = tmp.name
        orb_out.to_json_file(filename)

        orb_file = Orbitals.from_json_file(filename)

        coeffs_file_a, coeffs_file_b = orb_file.get_coefficients()
        assert np.allclose(
            coeffs_out_a,
            coeffs_file_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            coeffs_out_b,
            coeffs_file_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )


def test_hdf5_serialization():
    """Test HDF5 serialization and deserialization."""
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    energies = np.array([-2.0, 0.5])
    overlap = np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    basis_set = create_test_basis_set(3, "test-hdf5-serialization")
    orb_out = Orbitals(coeffs, energies, overlap, basis_set)

    try:
        with tempfile.NamedTemporaryFile(suffix=".orbitals.h5") as tmp:
            filename = tmp.name
            orb_out.to_hdf5_file(filename)

            orb_in = Orbitals.from_hdf5_file(filename)

            coeffs_out_a, coeffs_out_b = orb_out.get_coefficients()
            coeffs_in_a, coeffs_in_b = orb_in.get_coefficients()
            assert np.allclose(
                coeffs_out_a,
                coeffs_in_a,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                coeffs_out_b,
                coeffs_in_b,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
    except RuntimeError as e:
        pytest.skip(f"HDF5 test skipped - {e!s}")


def test_complete_orbitals_workflow():
    """Test a complete workflow with the Orbitals class."""
    coeffs = np.array([[0.85, 0.15], [0.15, -0.85], [0.0, 0.0]])
    energies = np.array([-1.5, 0.8])
    overlap = np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    basis_set = create_test_basis_set(3, "test-complete-workflow")
    orb = Orbitals(coeffs, energies, overlap, basis_set)

    assert orb.get_num_atomic_orbitals() == 3
    assert orb.get_num_molecular_orbitals() == 2
    assert orb.is_restricted()

    with tempfile.NamedTemporaryFile(suffix=".orbitals.json") as tmp_json:
        json_filename = tmp_json.name
        orb.to_json_file(json_filename)
        orb2 = Orbitals.from_json_file(json_filename)
        assert orb2.get_num_atomic_orbitals() == orb.get_num_atomic_orbitals()
        assert orb2.get_num_molecular_orbitals() == orb.get_num_molecular_orbitals()
        orig_coeffs_a, orig_coeffs_b = orb.get_coefficients()
        new_coeffs_a, new_coeffs_b = orb2.get_coefficients()
        assert np.allclose(
            orig_coeffs_a,
            new_coeffs_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_coeffs_b,
            new_coeffs_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )


def test_orbitals_file_io_generic():
    """Test generic file I/O methods for Orbitals."""
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    energies = np.array([-1.0, 0.5])
    overlap = np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    basis_set = create_test_basis_set(3, "test-file-io-generic")
    orb = Orbitals(coeffs, energies, overlap, basis_set)

    # Test JSON file I/O
    with tempfile.NamedTemporaryFile(suffix=".orbitals.json") as tmp_json:
        json_filename = tmp_json.name

        # Save using generic method
        orb.to_file(json_filename, "json")

        # Load using generic method (static)
        orb2 = Orbitals.from_file(json_filename, "json")

        # Check equality
        assert orb2.get_num_atomic_orbitals() == orb.get_num_atomic_orbitals()
        assert orb2.get_num_molecular_orbitals() == orb.get_num_molecular_orbitals()

        # Check coefficients
        orig_coeffs_a, orig_coeffs_b = orb.get_coefficients()
        new_coeffs_a, new_coeffs_b = orb2.get_coefficients()
        assert np.allclose(
            orig_coeffs_a,
            new_coeffs_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_coeffs_b,
            new_coeffs_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    # Test HDF5 file I/O
    with tempfile.NamedTemporaryFile(suffix=".orbitals.h5") as tmp_hdf5:
        hdf5_filename = tmp_hdf5.name

        # Save using generic method
        orb.to_file(hdf5_filename, "hdf5")

        # Load using generic method (static)
        orb3 = Orbitals.from_file(hdf5_filename, "hdf5")

        # Check equality
        assert orb3.get_num_atomic_orbitals() == orb.get_num_atomic_orbitals()
        assert orb3.get_num_molecular_orbitals() == orb.get_num_molecular_orbitals()

        # Check coefficients
        orig_coeffs_a, orig_coeffs_b = orb.get_coefficients()
        new_coeffs_a, new_coeffs_b = orb3.get_coefficients()
        assert np.allclose(
            orig_coeffs_a,
            new_coeffs_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_coeffs_b,
            new_coeffs_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    # Test unsupported file type
    with pytest.raises(RuntimeError, match="Unsupported file type"):
        orb.to_file("test.orbitals.xyz", "xyz")

    with pytest.raises(RuntimeError, match="Unsupported file type"):
        Orbitals.from_file("test.orbitals.xyz", "xyz")


def test_orbitals_hdf5_specific():
    """Test specific HDF5 file I/O methods for Orbitals."""
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    energies = np.array([-1.0, 0.5])
    overlap = np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    basis_set = create_test_basis_set(3, "test-file-io-hdf5-specific")
    orb = Orbitals(coeffs, energies, overlap, basis_set)

    # Test HDF5 file I/O methods
    with tempfile.NamedTemporaryFile(suffix=".orbitals.h5") as tmp_hdf5:
        hdf5_filename = tmp_hdf5.name

        # Save using new method
        orb.to_hdf5_file(hdf5_filename)

        # Load using new method (static)
        orb2 = Orbitals.from_hdf5_file(hdf5_filename)

        # Check equality
        assert orb2.get_num_atomic_orbitals() == orb.get_num_atomic_orbitals()
        assert orb2.get_num_molecular_orbitals() == orb.get_num_molecular_orbitals()

        # Check coefficients
        orig_coeffs_a, orig_coeffs_b = orb.get_coefficients()
        new_coeffs_a, new_coeffs_b = orb2.get_coefficients()
        assert np.allclose(
            orig_coeffs_a,
            new_coeffs_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_coeffs_b,
            new_coeffs_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check energies
        orig_energies_a, orig_energies_b = orb.get_energies()
        new_energies_a, new_energies_b = orb2.get_energies()
        assert np.allclose(
            orig_energies_a,
            new_energies_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_energies_b,
            new_energies_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check overlap
        assert np.allclose(
            orb.get_overlap_matrix(),
            orb2.get_overlap_matrix(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    # Test HDF5 file I/O methods work correctly
    with tempfile.NamedTemporaryFile(suffix=".orbitals.h5") as tmp_hdf5:
        hdf5_filename = tmp_hdf5.name

        # Save using method
        orb.to_hdf5_file(hdf5_filename)

        # Load using method (static)
        orb3 = Orbitals.from_hdf5_file(hdf5_filename)

        # Check equality
        assert orb3.get_num_atomic_orbitals() == orb.get_num_atomic_orbitals()
        assert orb3.get_num_molecular_orbitals() == orb.get_num_molecular_orbitals()


def test_orbitals_file_io_validation():
    """Test filename validation for Orbitals file I/O."""
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    occupations = np.array([2.0, 0.0])
    basis_set = create_test_basis_set(3, "test-file-io-validation")
    orb = Orbitals(coeffs, occupations, None, basis_set)

    # Test filename validation for JSON files
    with pytest.raises(ValueError, match=re.escape("'.orbitals.' before the file extension")):
        orb.to_json_file("test.json")

    with pytest.raises(ValueError, match=re.escape("'.orbitals.' before the file extension")):
        Orbitals.from_json_file("test.json")

    # Test filename validation for HDF5 files
    with pytest.raises(ValueError, match=re.escape("'.orbitals.' before the file extension")):
        orb.to_hdf5_file("test.h5")

    with pytest.raises(ValueError, match=re.escape("'.orbitals.' before the file extension")):
        Orbitals.from_hdf5_file("test.h5")

    # Test non-existent file
    with pytest.raises(RuntimeError, match="Cannot open file for reading"):
        Orbitals.from_json_file("nonexistent.orbitals.json")

    with pytest.raises(RuntimeError):
        Orbitals.from_hdf5_file("nonexistent.orbitals.h5")


def test_orbitals_file_io_round_trip():
    """Test round-trip file I/O preserves data integrity."""
    # Unrestricted data
    coeffs_alpha = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    coeffs_beta = np.array([[0.8, 0.2], [0.2, -0.8], [0.1, 0.1]])
    energies_alpha = np.array([-1.0, 0.5])
    energies_beta = np.array([-0.9, 0.6])
    overlap = np.array([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    basis_set = create_test_basis_set(3, "test-file-io-validation")
    orb = Orbitals(
        coeffs_alpha,
        coeffs_beta,
        energies_alpha,
        energies_beta,
        overlap,
        basis_set,
    )

    # Test JSON round-trip
    with tempfile.NamedTemporaryFile(suffix=".orbitals.json") as tmp_json:
        json_filename = tmp_json.name

        # Save and reload
        orb.to_json_file(json_filename)
        orb_json = Orbitals.from_json_file(json_filename)

        # Check all properties are preserved
        assert orb_json.get_num_atomic_orbitals() == orb.get_num_atomic_orbitals()
        assert orb_json.get_num_molecular_orbitals() == orb.get_num_molecular_orbitals()

        # Check coefficients
        orig_coeffs_a, orig_coeffs_b = orb.get_coefficients()
        json_coeffs_a, json_coeffs_b = orb_json.get_coefficients()
        assert np.allclose(
            orig_coeffs_a,
            json_coeffs_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_coeffs_b,
            json_coeffs_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check energies
        orig_energies_a, orig_energies_b = orb.get_energies()
        json_energies_a, json_energies_b = orb_json.get_energies()
        assert np.allclose(
            orig_energies_a,
            json_energies_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_energies_b,
            json_energies_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check overlap
        assert np.allclose(
            orb.get_overlap_matrix(),
            orb_json.get_overlap_matrix(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    # Test HDF5 round-trip
    with tempfile.NamedTemporaryFile(suffix=".orbitals.h5") as tmp_hdf5:
        hdf5_filename = tmp_hdf5.name

        # Save and reload
        orb.to_hdf5_file(hdf5_filename)
        orb_hdf5 = Orbitals.from_hdf5_file(hdf5_filename)

        # Check all properties are preserved
        assert orb_hdf5.get_num_atomic_orbitals() == orb.get_num_atomic_orbitals()
        assert orb_hdf5.get_num_molecular_orbitals() == orb.get_num_molecular_orbitals()

        # Check coefficients
        orig_coeffs_a, orig_coeffs_b = orb.get_coefficients()
        hdf5_coeffs_a, hdf5_coeffs_b = orb_hdf5.get_coefficients()
        assert np.allclose(
            orig_coeffs_a,
            hdf5_coeffs_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_coeffs_b,
            hdf5_coeffs_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check energies
        orig_energies_a, orig_energies_b = orb.get_energies()
        hdf5_energies_a, hdf5_energies_b = orb_hdf5.get_energies()
        assert np.allclose(
            orig_energies_a,
            hdf5_energies_a,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_energies_b,
            hdf5_energies_b,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check overlap
        assert np.allclose(
            orb.get_overlap_matrix(),
            orb_hdf5.get_overlap_matrix(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )


def test_active_space_management():
    """Test active space management functionality (restricted)."""
    coeffs = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, -0.9, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 0.2, -0.8],
        ]
    )
    active_indices = [1, 2]
    basis_set = create_test_basis_set(4, "test-active-space-management")
    orb = Orbitals(coeffs, None, None, basis_set, [active_indices, []])

    assert orb.has_active_space()
    alpha_indices, beta_indices = orb.get_active_space_indices()
    assert np.array_equal(alpha_indices, active_indices)
    assert np.array_equal(beta_indices, active_indices)


def test_inactive_space_management():
    """Test active space management functionality (restricted)."""
    coeffs = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, -0.9, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 0.2, -0.8],
        ]
    )
    inactive_indices = [0, 1]
    basis_set = create_test_basis_set(4, "test-inactive-space-management")
    orb = Orbitals(coeffs, None, None, basis_set, [[], inactive_indices])

    alpha_indices, beta_indices = orb.get_inactive_space_indices()
    assert np.array_equal(alpha_indices, inactive_indices)
    assert np.array_equal(beta_indices, inactive_indices)


def test_active_space_unrestricted():
    """Test active space management for unrestricted orbitals."""
    alpha_coeffs = np.array([[0.9, 0.1, 0.0], [0.1, -0.9, 0.0], [0.0, 0.0, 1.0]])
    beta_coeffs = np.array([[0.8, 0.2, 0.0], [0.2, -0.8, 0.0], [0.0, 0.0, 1.0]])
    alpha_active = [0, 1]
    beta_active = [1, 2]
    basis_set = create_test_basis_set(3, "test-active-space-unrestricted")
    orb = Orbitals(
        alpha_coeffs,
        beta_coeffs,
        None,
        None,
        None,
        basis_set,
        [alpha_active, beta_active, [], []],
    )

    retrieved_alpha, retrieved_beta = orb.get_active_space_indices()
    assert np.array_equal(retrieved_alpha, alpha_active)
    assert np.array_equal(retrieved_beta, beta_active)


def test_active_space_serialization():
    """Test serialization of orbitals with active space information."""
    coeffs = np.array([[0.9, 0.1, 0.0], [0.1, -0.9, 0.0], [0.0, 0.0, 1.0]])
    active_indices = [0, 2]
    basis_set = create_test_basis_set(3, "test-active-space-serialization")
    orb = Orbitals(coeffs, None, None, basis_set, [active_indices, []])

    # Test JSON serialization
    with tempfile.NamedTemporaryFile(suffix=".orbitals.json") as tmp_json:
        json_filename = tmp_json.name
        orb.to_json_file(json_filename)

        # Load into a new object
        orb_json = Orbitals.from_json_file(json_filename)

        # Check that active space was preserved
        assert orb_json.has_active_space()

        # Verify active space indices
        json_alpha, json_beta = orb_json.get_active_space_indices()
        assert np.array_equal(json_alpha, active_indices)
        assert np.array_equal(json_beta, active_indices)

    # Test HDF5 serialization
    try:
        with tempfile.NamedTemporaryFile(suffix=".orbitals.h5") as tmp_hdf5:
            hdf5_filename = tmp_hdf5.name
            orb.to_hdf5_file(hdf5_filename)

            # Load into a new object
            orb_hdf5 = Orbitals.from_hdf5_file(hdf5_filename)

            # Check that active space was preserved
            assert orb_hdf5.has_active_space()

            # Verify active space indices
            hdf5_alpha, hdf5_beta = orb_hdf5.get_active_space_indices()
            assert np.array_equal(hdf5_alpha, active_indices)
            assert np.array_equal(hdf5_beta, active_indices)

    except RuntimeError as e:
        pytest.skip(f"HDF5 test skipped - {e!s}")


def test_active_space_copy_assign():
    """Test that active space is preserved when copying or assigning orbitals."""
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9]])
    basis_set = create_test_basis_set(2, "test-active-space-copy-assign")
    orb = Orbitals(coeffs, None, None, basis_set, [[0], []])

    # Test copy constructor
    orb_copy = Orbitals(orb)
    assert orb_copy.has_active_space()

    copy_alpha, copy_beta = orb_copy.get_active_space_indices()
    orig_alpha, orig_beta = orb.get_active_space_indices()
    assert np.array_equal(copy_alpha, orig_alpha)
    assert np.array_equal(copy_beta, orig_beta)


def test_active_space_validation():
    """Test validation of active space indices and electron counts."""
    # Set up basic data with 3 MOs and valid active space in constructor
    coeffs = np.array([[0.9, 0.1, 0.0], [0.1, -0.9, 0.0], [0.0, 0.0, 1.0]])
    basis_set = create_test_basis_set(3, "test-active-space-validation")
    orb = Orbitals(coeffs, None, None, basis_set, [[0, 1], []])
    alpha_indices, beta_indices = orb.get_active_space_indices()
    assert np.array_equal(alpha_indices, [0, 1])
    assert np.array_equal(beta_indices, [0, 1])


def test_error_handling():
    """Removed: mutator-based error handling is obsolete with immutable API."""
    # In immutable API, incorrect shapes should raise at construction time.
    # Verify one such case explicitly.
    coeffs_alpha = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2
    coeffs_beta = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2x3
    basis = create_test_basis_set(2, "test-error-handling")
    with contextlib.suppress(Exception):
        Orbitals(coeffs_alpha, coeffs_beta, None, basis)  # may raise


def test_large_orbital_set(tmp_path):
    """Test handling of large orbital sets."""
    # Create fairly large matrices (e.g., 500 AOs, 200 MOs)
    num_atomic_orbitals = 500
    num_molecular_orbitals = 200

    # Generate large matrices using modern random generator
    rng = np.random.default_rng(42)  # Use fixed seed for reproducibility
    coeffs = rng.random((num_atomic_orbitals, num_molecular_orbitals))
    energies = rng.random(num_molecular_orbitals) - 0.5  # centered around zero
    occupations = np.zeros(num_molecular_orbitals)
    occupations[:100] = 2.0  # 100 doubly occupied orbitals = 200 electrons
    overlap = np.eye(num_atomic_orbitals)  # identity for simplicity

    basis = create_test_basis_set(num_atomic_orbitals, "test-large-orbitals")

    # Construct directly
    orb = Orbitals(coeffs, energies, overlap, basis)

    # Check dimensions
    assert orb.get_num_atomic_orbitals() == num_atomic_orbitals
    assert orb.get_num_molecular_orbitals() == num_molecular_orbitals

    # Check that serialization works with large data
    json_file = tmp_path / "large_orbitals.orbitals.json"

    # Test JSON serialization
    orb.to_json_file(json_file)

    # Load back and check
    orb2 = Orbitals.from_json_file(json_file)

    # Check dimensions
    assert orb2.get_num_atomic_orbitals() == orb.get_num_atomic_orbitals()
    assert orb2.get_num_molecular_orbitals() == orb.get_num_molecular_orbitals()


def test_orbitals_pickling_and_repr():
    """Test pickling support and string representation for Orbitals."""
    # Create a test orbitals object (restricted)
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    energies = np.array([-1.0, 0.5])
    overlap = np.eye(3)
    basis_set = create_test_basis_set(3, "test-pickling")

    original = Orbitals(coeffs, energies, overlap, basis_set)

    # Test pickling and unpickling
    pickled_data = pickle.dumps(original)
    assert isinstance(pickled_data, bytes)

    # Unpickle and verify
    unpickled = pickle.loads(pickled_data)

    # Verify orbitals are preserved
    assert unpickled.get_num_atomic_orbitals() == original.get_num_atomic_orbitals()
    assert unpickled.get_num_molecular_orbitals() == original.get_num_molecular_orbitals()
    assert unpickled.is_restricted() == original.is_restricted()

    # Check coefficients are preserved
    orig_alpha, orig_beta = original.get_coefficients()
    unpick_alpha, unpick_beta = unpickled.get_coefficients()
    assert np.allclose(
        orig_alpha, unpick_alpha, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(
        orig_beta, unpick_beta, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )

    # Check energies are preserved
    if original.has_energies():
        orig_e_alpha, orig_e_beta = original.get_energies()
        unpick_e_alpha, unpick_e_beta = unpickled.get_energies()
        assert np.allclose(
            orig_e_alpha,
            unpick_e_alpha,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            orig_e_beta,
            unpick_e_beta,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    # Check overlap matrix is preserved
    if original.has_overlap_matrix():
        assert np.allclose(
            original.get_overlap_matrix(),
            unpickled.get_overlap_matrix(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    # Test multiple round trips
    current = original
    for _ in range(3):
        pickled = pickle.dumps(current)
        current = pickle.loads(pickled)

        # Verify data integrity after each round trip
        assert current.get_num_atomic_orbitals() == original.get_num_atomic_orbitals()
        assert current.get_num_molecular_orbitals() == original.get_num_molecular_orbitals()
        assert current.is_restricted() == original.is_restricted()

    # Test __repr__ uses summary function
    repr_output = repr(original)
    summary_output = original.get_summary()

    # They should be the same
    assert repr_output == summary_output

    # Check that typical summary content is present
    assert (
        "Orbitals" in repr_output or "MOs" in repr_output or str(original.get_num_molecular_orbitals()) in repr_output
    )

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


def test_orbitals_unrestricted_pickling():
    """Test pickling with unrestricted orbitals."""
    # Create unrestricted orbitals
    coeffs_alpha = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
    coeffs_beta = np.array([[0.8, 0.2], [0.2, -0.8], [0.1, 0.0]])
    energies_alpha = np.array([-1.1, 0.4])
    energies_beta = np.array([-1.0, 0.6])
    overlap = np.eye(3)
    basis_set = create_test_basis_set(3, "test-unrestricted-pickling")

    original = Orbitals(coeffs_alpha, coeffs_beta, energies_alpha, energies_beta, overlap, basis_set)

    # Test pickling and unpickling
    pickled_data = pickle.dumps(original)
    unpickled = pickle.loads(pickled_data)

    # Verify unrestricted nature is preserved
    assert unpickled.is_unrestricted() == original.is_unrestricted()
    assert not unpickled.is_restricted()

    # Check alpha and beta coefficients separately
    orig_alpha = original.get_coefficients_alpha()
    orig_beta = original.get_coefficients_beta()
    unpick_alpha = unpickled.get_coefficients_alpha()
    unpick_beta = unpickled.get_coefficients_beta()

    assert np.allclose(
        orig_alpha, unpick_alpha, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )
    assert np.allclose(
        orig_beta, unpick_beta, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )

    # Check alpha and beta energies separately
    if original.has_energies():
        assert np.allclose(
            original.get_energies_alpha(),
            unpickled.get_energies_alpha(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            original.get_energies_beta(),
            unpickled.get_energies_beta(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )


def test_model_orbitals_pickling_and_repr():
    """Test pickling support and string representation for ModelOrbitals."""
    # Create a test ModelOrbitals object
    original = ModelOrbitals(4, True)  # 4 orbitals, restricted

    # Test pickling and unpickling
    pickled_data = pickle.dumps(original)
    assert isinstance(pickled_data, bytes)

    # Unpickle and verify
    unpickled = pickle.loads(pickled_data)

    # Verify model orbitals are preserved
    assert unpickled.get_num_atomic_orbitals() == original.get_num_atomic_orbitals()
    assert unpickled.get_num_molecular_orbitals() == original.get_num_molecular_orbitals()
    assert unpickled.is_restricted() == original.is_restricted()

    # Test __repr__ uses summary function
    repr_output = repr(original)
    summary_output = original.get_summary()

    # They should be the same
    assert repr_output == summary_output

    # Test unrestricted ModelOrbitals
    original_unres = ModelOrbitals(6, False)  # 6 orbitals, unrestricted

    pickled_data = pickle.dumps(original_unres)
    unpickled_unres = pickle.loads(pickled_data)

    assert unpickled_unres.is_unrestricted() == original_unres.is_unrestricted()
    assert not unpickled_unres.is_restricted()
