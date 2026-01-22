"""Parameterized tests for Hamiltonian with different container types.

This module tests the Hamiltonian class with both CanonicalFourCenterHamiltonianContainer
and DensityFittedHamiltonianContainer using pytest parametrization.

Both containers implement the same interface and should behave identically
for all Hamiltonian operations.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import math
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    DensityFittedHamiltonianContainer,
    Hamiltonian,
    ModelOrbitals,
    Orbitals,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import create_test_basis_set, create_test_hamiltonian, create_test_orbitals

# Container types for parametrization
CONTAINER_TYPES = ["canonical_four_center", "density_fitted"]


# =============================================================================
# Shared module-level variables for tests
# =============================================================================

_one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
_coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
_orbitals = Orbitals(_coeffs, None, None, create_test_basis_set(2))
_rng = np.random.default_rng(42)
_two_body = _rng.random(2**4)
_three_center = np.array(
    [
        [1.0, 1.0, 1.0, 1.0],
        [0.6, 0.6, 0.6, 0.6],
        [0.8, 0.8, 0.8, 0.8],
    ]
)


# =============================================================================
# Helper functions for creating containers directly
# =============================================================================


def create_non_zero_hamiltonian(container_type):
    """Create a Hamiltonian with non-zero integrals for testing."""
    if container_type == "canonical_four_center":
        container = CanonicalFourCenterHamiltonianContainer(_one_body, _two_body, _orbitals, 1.5, np.array([]))
    elif container_type == "density_fitted":
        container = DensityFittedHamiltonianContainer(_one_body, _three_center, _orbitals, 1.5, np.array([]))
    else:
        raise ValueError(f"Unknown container_type: {container_type}")

    return Hamiltonian(container)


# =============================================================================
# Parameterized Tests - Run for Both Container Types
# =============================================================================


class TestHamiltonian:
    """Parameterized tests that run for both container types."""

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_default_constructor(self, container_type):
        """Test default Hamiltonian construction."""
        h = create_test_hamiltonian(2, container_type)
        assert isinstance(h, Hamiltonian)
        assert h.has_one_body_integrals()
        assert h.has_two_body_integrals()
        assert h.has_orbitals()

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_size_and_electron_counts(self, container_type):
        """Test orbital count retrieval."""
        h = create_test_hamiltonian(3, container_type)
        assert isinstance(h, Hamiltonian)
        assert h.get_orbitals().get_num_molecular_orbitals() == 3

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_full_constructor(self, container_type):
        """Test full constructor with custom values."""
        h = create_non_zero_hamiltonian(container_type)
        assert h.has_one_body_integrals()
        assert h.has_two_body_integrals()
        assert h.has_orbitals()
        assert h.get_orbitals().get_num_molecular_orbitals() == 2
        assert h.get_core_energy() == 1.5

        aa, bb = h.get_one_body_integrals()
        assert np.array_equal(aa, _one_body)
        assert np.array_equal(bb, _one_body)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_one_body_integrals(self, container_type):
        """Test one-body integrals retrieval."""
        h = create_non_zero_hamiltonian(container_type)

        assert h.has_one_body_integrals()
        assert np.array_equal(h.get_one_body_integrals()[0], _one_body)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_two_body_integrals(self, container_type):
        """Test two-body integrals retrieval."""
        h = create_test_hamiltonian(2, container_type)

        aaaa, aabb, bbbb = h.get_two_body_integrals()
        # For restricted case, all should be the same
        assert np.array_equal(aabb, aaaa)
        assert np.array_equal(bbbb, aaaa)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_two_body_element_access(self, container_type):
        """Test two-body element access."""
        h = create_test_hamiltonian(2, container_type)
        val = h.get_two_body_element(0, 1, 1, 0)
        assert isinstance(val, float)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_active_space_management(self, container_type):
        """Test core energy retrieval."""
        h = create_non_zero_hamiltonian(container_type)
        assert h.get_core_energy() == 1.5

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_json_serialization(self, container_type):
        """Test JSON serialization roundtrip."""
        h = create_non_zero_hamiltonian(container_type)

        data = json.loads(h.to_json())
        assert isinstance(data, dict)
        assert data["container"]["core_energy"] == 1.5
        assert data["container"]["has_one_body_integrals"] is True
        assert data["container"]["has_two_body_integrals"] is True
        assert data["container"]["has_orbitals"] is True

        h2 = Hamiltonian.from_json(json.dumps(data))
        assert h2.get_orbitals().get_num_molecular_orbitals() == 2
        assert h2.get_core_energy() == 1.5
        assert h2.has_one_body_integrals()
        assert h2.has_two_body_integrals()
        assert h2.has_orbitals()

        # Verify integrals match
        assert np.allclose(
            h.get_one_body_integrals()[0],
            h2.get_one_body_integrals()[0],
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            h.get_one_body_integrals()[1],
            h2.get_one_body_integrals()[1],
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Compare two-body integrals
        h_aaaa, h_aabb, h_bbbb = h.get_two_body_integrals()
        h2_aaaa, h2_aabb, h2_bbbb = h2.get_two_body_integrals()
        assert np.allclose(
            h_aaaa, h2_aaaa, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.allclose(
            h_aabb, h2_aabb, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.allclose(
            h_bbbb, h2_bbbb, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_json_file_io(self, container_type):
        """Test JSON file I/O."""
        h = create_non_zero_hamiltonian(container_type)

        with tempfile.NamedTemporaryFile(suffix=".hamiltonian.json", delete=False) as f:
            filename = f.name
        try:
            h.to_json_file(filename)
            assert Path(filename).exists()
            h2 = Hamiltonian.from_json_file(filename)
            assert h2.get_orbitals().get_num_molecular_orbitals() == 2
            assert h2.get_core_energy() == 1.5
            assert h2.has_one_body_integrals()
            assert h2.has_two_body_integrals()
            assert h2.has_orbitals()

            # Compare integrals
            assert np.allclose(
                h.get_one_body_integrals()[0],
                h2.get_one_body_integrals()[0],
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

            h_aaaa, h_aabb, h_bbbb = h.get_two_body_integrals()
            h2_aaaa, h2_aabb, h2_bbbb = h2.get_two_body_integrals()
            assert np.allclose(
                h_aaaa, h2_aaaa, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )
            assert np.allclose(
                h_aabb, h2_aabb, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )
            assert np.allclose(
                h_bbbb, h2_bbbb, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )
        finally:
            Path(filename).unlink(missing_ok=True)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_hdf5_file_io(self, container_type):
        """Test HDF5 file I/O."""
        h = create_non_zero_hamiltonian(container_type)

        with tempfile.NamedTemporaryFile(suffix=".hamiltonian.h5", delete=False) as f:
            filename = f.name
        try:
            h.to_hdf5_file(filename)
            assert Path(filename).exists()
            h2 = Hamiltonian.from_hdf5_file(filename)
            assert h2.get_orbitals().get_num_molecular_orbitals() == 2
            assert h2.get_core_energy() == 1.5
            assert h2.has_one_body_integrals()
            assert h2.has_two_body_integrals()
            assert h2.has_orbitals()

            # Compare integrals
            assert np.allclose(
                h.get_one_body_integrals()[0],
                h2.get_one_body_integrals()[0],
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

            h_aaaa, h_aabb, h_bbbb = h.get_two_body_integrals()
            h2_aaaa, h2_aabb, h2_bbbb = h2.get_two_body_integrals()
            assert np.allclose(
                h_aaaa, h2_aaaa, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )
            assert np.allclose(
                h_aabb, h2_aabb, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )
            assert np.allclose(
                h_bbbb, h2_bbbb, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )
        finally:
            Path(filename).unlink(missing_ok=True)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_generic_file_io(self, container_type):
        """Test generic file I/O with both JSON and HDF5 formats."""
        h = create_non_zero_hamiltonian(container_type)

        # Test JSON
        with tempfile.NamedTemporaryFile(suffix=".hamiltonian.json", delete=False) as f:
            json_filename = f.name
        try:
            h.to_file(json_filename, "json")
            assert Path(json_filename).exists()
            h2 = Hamiltonian.from_file(json_filename, "json")
            assert h2.get_orbitals().get_num_molecular_orbitals() == 2
            assert np.allclose(
                h.get_one_body_integrals()[0],
                h2.get_one_body_integrals()[0],
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                h.get_one_body_integrals()[1],
                h2.get_one_body_integrals()[1],
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        finally:
            Path(json_filename).unlink(missing_ok=True)

        # Test HDF5
        with tempfile.NamedTemporaryFile(suffix=".hamiltonian.h5", delete=False) as f:
            hdf5_filename = f.name
        try:
            h.to_file(hdf5_filename, "hdf5")
            assert Path(hdf5_filename).exists()
            h3 = Hamiltonian.from_file(hdf5_filename, "hdf5")
            assert h3.get_orbitals().get_num_molecular_orbitals() == 2
            assert np.allclose(
                h.get_one_body_integrals()[0],
                h3.get_one_body_integrals()[0],
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                h.get_one_body_integrals()[1],
                h3.get_one_body_integrals()[1],
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        finally:
            Path(hdf5_filename).unlink(missing_ok=True)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_minimal_hamiltonian_json_roundtrip(self, container_type):
        """Test JSON roundtrip for minimal 1-orbital Hamiltonian."""
        h = create_test_hamiltonian(1, container_type)
        data = json.loads(h.to_json())
        assert data["container"]["core_energy"] == 0.0
        assert data["container"]["has_one_body_integrals"] is True
        assert data["container"]["has_two_body_integrals"] is True
        assert data["container"]["has_orbitals"] is True
        h2 = Hamiltonian.from_json(json.dumps(data))
        assert h2.get_orbitals().get_num_molecular_orbitals() == 1
        assert h2.get_core_energy() == 0.0
        assert h2.has_one_body_integrals()
        assert h2.has_two_body_integrals()
        assert h2.has_orbitals()

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_static_methods_exist(self, container_type):
        """Test that static methods exist and work."""
        h = create_non_zero_hamiltonian(container_type)

        with tempfile.NamedTemporaryFile(suffix=".hamiltonian.h5", delete=False) as f:
            filename = f.name
        try:
            h.to_hdf5_file(filename)
            assert Hamiltonian.from_hdf5_file(filename) is not None
        finally:
            Path(filename).unlink(missing_ok=True)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_repr_method(self, container_type):
        """Test that __repr__ returns the summary."""
        h = create_test_hamiltonian(2, container_type)
        repr_str = repr(h)
        summary_str = h.get_summary()
        assert repr_str == summary_str
        assert "Hamiltonian" in repr_str

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_str_method(self, container_type):
        """Test that __str__ returns the summary."""
        h = create_test_hamiltonian(2, container_type)
        str_str = str(h)
        summary_str = h.get_summary()
        assert str_str == summary_str
        assert "Hamiltonian" in str_str

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_pickling_hamiltonian(self, container_type):
        """Test that Hamiltonian can be pickled and unpickled correctly."""
        h = create_test_hamiltonian(3, container_type)

        # Test pickling round-trip
        pickled_data = pickle.dumps(h)
        h_restored = pickle.loads(pickled_data)

        # Verify core properties
        assert h_restored.has_one_body_integrals() == h.has_one_body_integrals()
        assert h_restored.has_two_body_integrals() == h.has_two_body_integrals()
        assert h_restored.has_orbitals() == h.has_orbitals()
        assert h_restored.get_core_energy() == h.get_core_energy()

        # Verify integral data
        if h.has_one_body_integrals():
            assert np.array_equal(h_restored.get_one_body_integrals()[0], h.get_one_body_integrals()[0])
            assert np.array_equal(h_restored.get_one_body_integrals()[1], h.get_one_body_integrals()[1])

        if h.has_two_body_integrals():
            h_aaaa, h_aabb, h_bbbb = h.get_two_body_integrals()
            h_restored_aaaa, h_restored_aabb, h_restored_bbbb = h_restored.get_two_body_integrals()
            assert np.allclose(h_restored_aaaa, h_aaaa)
            assert np.allclose(h_restored_aabb, h_aabb)
            assert np.allclose(h_restored_bbbb, h_bbbb)

        # Verify orbital consistency
        if h.has_orbitals():
            orig_orbs = h.get_orbitals()
            restored_orbs = h_restored.get_orbitals()
            assert orig_orbs.get_num_molecular_orbitals() == restored_orbs.get_num_molecular_orbitals()
            assert np.array_equal(orig_orbs.get_coefficients(), restored_orbs.get_coefficients())


# =============================================================================
# File I/O Validation Tests (container-independent)
# =============================================================================


class TestFileIOValidation:
    """Tests for file I/O validation (container-independent)."""

    def test_file_io_validation(self):
        """Test file I/O validation errors."""
        h = create_test_hamiltonian(2)
        with pytest.raises(RuntimeError, match="Unsupported file type"):
            h.to_file("test.txt", "txt")
        with pytest.raises(RuntimeError, match="Unsupported file type"):
            Hamiltonian.from_file("test.txt", "txt")
        with pytest.raises(RuntimeError, match="Unable to open Hamiltonian JSON file"):
            Hamiltonian.from_json_file("nonexistent.hamiltonian.json")
        with pytest.raises(RuntimeError, match="Unable to open Hamiltonian HDF5 file"):
            Hamiltonian.from_hdf5_file("nonexistent.hamiltonian.h5")


class TestInvalidContainerType:
    """Tests for invalid container type handling."""

    def test_invalid_container_type_raises(self):
        """Test that invalid container type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown container_type"):
            create_test_hamiltonian(2, "invalid_type")


# =============================================================================
# Restricted/Unrestricted Tests
# =============================================================================


class TestRestrictedUnrestricted:
    """Tests for restricted and unrestricted Hamiltonian behavior."""

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_restricted_hamiltonian_construction(self, container_type):
        """Test restricted Hamiltonian construction and properties."""
        # Create restricted orbitals
        coeffs = np.eye(3)
        basis_set = create_test_basis_set(3, "test-restricted")
        orbitals = Orbitals(coeffs, None, None, basis_set)

        assert orbitals.is_restricted()
        assert not orbitals.is_unrestricted()

        # Create restricted Hamiltonian
        rng = np.random.default_rng(42)
        one_body = rng.random((3, 3))
        one_body = 0.5 * (one_body + one_body.T)  # Make symmetric
        inactive_fock = rng.random((3, 3))

        if container_type == "canonical_four_center":
            two_body = rng.random(3**4)
            container = CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.0, inactive_fock)
        else:
            three_center = rng.random((3, 9))
            container = DensityFittedHamiltonianContainer(one_body, three_center, orbitals, 1.0, inactive_fock)

        h = Hamiltonian(container)

        # Verify Hamiltonian properties
        assert h.is_restricted()
        assert not h.is_unrestricted()
        assert h.has_one_body_integrals()
        assert h.has_two_body_integrals()
        assert h.has_orbitals()
        assert h.get_core_energy() == 1.0

        # Verify integral access
        assert np.array_equal(h.get_one_body_integrals()[0], one_body)
        assert np.array_equal(h.get_one_body_integrals()[1], one_body)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_unrestricted_hamiltonian_construction(self, container_type):
        """Test unrestricted Hamiltonian construction and properties."""
        # Create unrestricted orbitals with different alpha/beta coefficients
        coeffs_alpha = np.eye(2)
        coeffs_beta = np.array([[0.8, 0.6], [0.6, -0.8]])
        basis_set = create_test_basis_set(2, "test-unrestricted")
        orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

        # Verify orbitals are unrestricted
        assert not orbitals.is_restricted()
        assert orbitals.is_unrestricted()

        # Create unrestricted Hamiltonian with different alpha/beta integrals
        rng = np.random.default_rng(123)
        one_body_alpha = np.array([[1.0, 0.2], [0.2, 1.5]])
        one_body_beta = np.array([[1.1, 0.3], [0.3, 1.6]])
        inactive_fock_alpha = np.array([[0.5, 0.1], [0.1, 0.7]])
        inactive_fock_beta = np.array([[0.6, 0.2], [0.2, 0.8]])

        if container_type == "canonical_four_center":
            two_body_aaaa = rng.random(2**4)
            two_body_aabb = rng.random(2**4)
            two_body_bbbb = rng.random(2**4)
            container = CanonicalFourCenterHamiltonianContainer(
                one_body_alpha,
                one_body_beta,
                two_body_aaaa,
                two_body_aabb,
                two_body_bbbb,
                orbitals,
                2.0,
                inactive_fock_alpha,
                inactive_fock_beta,
            )
        else:
            # For density-fitted, use three-center integrals scaled appropriately
            base_three_center = np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.6, 0.6, 0.6, 0.6],
                    [0.8, 0.8, 0.8, 0.8],
                ]
            )
            three_center_aaaa = math.sqrt(0.5) * base_three_center
            three_center_aabb = base_three_center
            three_center_bbbb = math.sqrt(1.5) * base_three_center
            container = DensityFittedHamiltonianContainer(
                one_body_alpha,
                one_body_beta,
                three_center_aaaa,
                three_center_aabb,
                three_center_bbbb,
                orbitals,
                2.0,
                inactive_fock_alpha,
                inactive_fock_beta,
            )

        h = Hamiltonian(container)

        # Verify Hamiltonian properties
        assert not h.is_restricted()
        assert h.is_unrestricted()
        assert h.has_one_body_integrals()
        assert h.has_two_body_integrals()
        assert h.has_orbitals()
        assert h.get_core_energy() == 2.0

        # Verify separate alpha/beta integral access
        assert np.array_equal(h.get_one_body_integrals()[0], one_body_alpha)
        assert np.array_equal(h.get_one_body_integrals()[1], one_body_beta)

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_unrestricted_vs_restricted_serialization(self, container_type):
        """Test that restricted/unrestricted nature is preserved in serialization."""
        basis_set = create_test_basis_set(2, "test-serialization")

        # Test restricted Hamiltonian
        coeffs = np.eye(2)
        orbitals_restricted = Orbitals(coeffs, None, None, basis_set)

        one_body = np.array([[1.0, 0.1], [0.1, 1.0]])

        if container_type == "canonical_four_center":
            two_body = np.ones(16) * 0.5
            container_restricted = CanonicalFourCenterHamiltonianContainer(
                one_body, two_body, orbitals_restricted, 1.0, np.eye(2)
            )
        else:
            three_center = np.array(
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.3, 0.3, 0.3, 0.3],
                    [0.4, 0.4, 0.4, 0.4],
                ]
            )
            container_restricted = DensityFittedHamiltonianContainer(
                one_body, three_center, orbitals_restricted, 1.0, np.eye(2)
            )

        h_restricted = Hamiltonian(container_restricted)

        # Test unrestricted Hamiltonian
        coeffs_alpha = np.eye(2)
        coeffs_beta = np.array([[0.9, 0.1], [0.1, 0.9]])
        orbitals_unrestricted = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

        one_body_alpha = np.array([[1.0, 0.1], [0.1, 1.0]])
        one_body_beta = np.array([[1.1, 0.2], [0.2, 1.1]])

        if container_type == "canonical_four_center":
            two_body_aaaa = np.ones(16) * 1.0
            two_body_aabb = np.ones(16) * 2.0
            two_body_bbbb = np.ones(16) * 3.0
            container_unrestricted = CanonicalFourCenterHamiltonianContainer(
                one_body_alpha,
                one_body_beta,
                two_body_aaaa,
                two_body_aabb,
                two_body_bbbb,
                orbitals_unrestricted,
                2.0,
                np.eye(2),
                np.eye(2),
            )
        else:
            base_three_center = np.array(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [0.6, 0.6, 0.6, 0.6],
                    [0.8, 0.8, 0.8, 0.8],
                ]
            )
            three_center_aaaa = math.sqrt(0.5) * base_three_center
            three_center_aabb = base_three_center
            three_center_bbbb = math.sqrt(1.5) * base_three_center
            container_unrestricted = DensityFittedHamiltonianContainer(
                one_body_alpha,
                one_body_beta,
                three_center_aaaa,
                three_center_aabb,
                three_center_bbbb,
                orbitals_unrestricted,
                2.0,
                np.eye(2),
                np.eye(2),
            )

        h_unrestricted = Hamiltonian(container_unrestricted)

        # Test JSON serialization preserves restricted/unrestricted nature
        h_restricted_json = Hamiltonian.from_json(h_restricted.to_json())
        assert h_restricted_json.is_restricted()
        assert not h_restricted_json.is_unrestricted()

        h_unrestricted_json = Hamiltonian.from_json(h_unrestricted.to_json())
        assert not h_unrestricted_json.is_restricted()
        assert h_unrestricted_json.is_unrestricted()

        # Verify integral values are preserved
        assert np.array_equal(h_restricted.get_one_body_integrals()[0], h_restricted_json.get_one_body_integrals()[0])
        assert np.array_equal(h_restricted.get_one_body_integrals()[1], h_restricted_json.get_one_body_integrals()[1])
        assert np.array_equal(
            h_unrestricted.get_one_body_integrals()[0], h_unrestricted_json.get_one_body_integrals()[0]
        )
        assert np.array_equal(
            h_unrestricted.get_one_body_integrals()[1], h_unrestricted_json.get_one_body_integrals()[1]
        )

    @pytest.mark.parametrize("container_type", CONTAINER_TYPES)
    def test_active_space_consistency(self, container_type):
        """Test that active space handling works correctly for both restricted and unrestricted."""
        # Test restricted case with active space
        model_orbitals_restricted = ModelOrbitals(4, True)
        assert model_orbitals_restricted.is_restricted()
        assert model_orbitals_restricted.has_active_space()

        # Create restricted Hamiltonian
        one_body = np.eye(4)

        if container_type == "canonical_four_center":
            two_body = np.zeros(4**4)
            container_restricted = CanonicalFourCenterHamiltonianContainer(
                one_body, two_body, model_orbitals_restricted, 0.0, np.eye(4)
            )
        else:
            three_center = np.zeros((4, 16))
            container_restricted = DensityFittedHamiltonianContainer(
                one_body, three_center, model_orbitals_restricted, 0.0, np.eye(4)
            )

        h_restricted = Hamiltonian(container_restricted)
        assert h_restricted.is_restricted()

        # Test unrestricted case with active space
        model_orbitals_unrestricted = ModelOrbitals(4, False)
        assert not model_orbitals_unrestricted.is_restricted()
        assert model_orbitals_unrestricted.is_unrestricted()
        assert model_orbitals_unrestricted.has_active_space()

        # Create unrestricted Hamiltonian
        one_body_alpha = np.eye(4)
        one_body_beta = np.eye(4) * 1.1

        if container_type == "canonical_four_center":
            two_body_aaaa = np.zeros(4**4)
            two_body_aabb = np.zeros(4**4)
            two_body_bbbb = np.zeros(4**4)
            container_unrestricted = CanonicalFourCenterHamiltonianContainer(
                one_body_alpha,
                one_body_beta,
                two_body_aaaa,
                two_body_aabb,
                two_body_bbbb,
                model_orbitals_unrestricted,
                0.0,
                np.eye(4),
                np.eye(4),
            )
        else:
            three_center_aaaa = np.zeros((4, 16))
            three_center_aabb = np.zeros((4, 16))
            three_center_bbbb = np.zeros((4, 16))
            container_unrestricted = DensityFittedHamiltonianContainer(
                one_body_alpha,
                one_body_beta,
                three_center_aaaa,
                three_center_aabb,
                three_center_bbbb,
                model_orbitals_unrestricted,
                0.0,
                np.eye(4),
                np.eye(4),
            )

        h_unrestricted = Hamiltonian(container_unrestricted)
        assert h_unrestricted.is_unrestricted()

        # Verify active space information is accessible
        alpha_indices, beta_indices = model_orbitals_restricted.get_active_space_indices()
        assert len(alpha_indices) == 4  # All orbitals active by default
        assert len(beta_indices) == 4
        assert alpha_indices == beta_indices

        alpha_indices_unres, beta_indices_unres = model_orbitals_unrestricted.get_active_space_indices()
        assert len(alpha_indices_unres) == 4
        assert len(beta_indices_unres) == 4


# =============================================================================
# Density-Fitted Specific Tests
# =============================================================================


class TestDensityFittedSpecific:
    """Tests specific to DensityFittedHamiltonianContainer."""

    def test_three_center_integrals_storage(self):
        """Test three-center integral storage and retrieval."""
        one_body = np.eye(2)
        # Three-center: [n_aux x n_geminals] = [3 x 4]
        three_center = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.6, 0.6, 0.6, 0.6],
                [0.8, 0.8, 0.8, 0.8],
            ]
        )
        orbitals = create_test_orbitals(2)

        container = DensityFittedHamiltonianContainer(one_body, three_center, orbitals, 1.5, np.array([]))

        # Verify three-center storage
        tc = container.get_three_center_integrals()
        assert np.allclose(tc, three_center)

    def test_two_body_from_three_center_contraction(self):
        """Test two-body integrals computed from three-center contraction.

        The three_center integrals are chosen so that contraction
        (ij|kl) = sum_P (ij|P)(P|kl) produces expected two_body values.
        """
        one_body = np.eye(2)
        # These values produce two_body = 2.0 when contracted:
        # 1.0^2 + 0.6^2 + 0.8^2 = 1.0 + 0.36 + 0.64 = 2.0
        three_center = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.6, 0.6, 0.6, 0.6],
                [0.8, 0.8, 0.8, 0.8],
            ]
        )
        orbitals = create_test_orbitals(2)

        container = DensityFittedHamiltonianContainer(one_body, three_center, orbitals, 1.5, np.array([]))
        h = Hamiltonian(container)

        aaaa, _, _ = h.get_two_body_integrals()
        expected_two_body = 2.0 * np.ones(16)
        assert np.allclose(
            aaaa,
            expected_two_body,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_unrestricted_three_center_integrals(self):
        """Test three-center integrals for unrestricted case."""
        coeffs_alpha = np.eye(2)
        coeffs_beta = np.array([[0.8, 0.6], [0.6, -0.8]])
        basis_set = create_test_basis_set(2, "test-unrestricted-df")
        orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

        one_body_alpha = np.eye(2)
        one_body_beta = np.eye(2) * 1.1

        base_three_center = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.6, 0.6, 0.6, 0.6],
                [0.8, 0.8, 0.8, 0.8],
            ]
        )
        three_center_aaaa = math.sqrt(0.5) * base_three_center
        three_center_aabb = base_three_center
        three_center_bbbb = math.sqrt(1.5) * base_three_center

        container = DensityFittedHamiltonianContainer(
            one_body_alpha,
            one_body_beta,
            three_center_aaaa,
            three_center_aabb,
            three_center_bbbb,
            orbitals,
            1.5,
            np.eye(2),
            np.eye(2),
        )

        # Verify three-center retrieval for unrestricted
        tc_aaaa, tc_aabb, tc_bbbb = container.get_three_center_integrals()
        assert np.allclose(tc_aaaa, three_center_aaaa)
        assert np.allclose(tc_aabb, three_center_aabb)
        assert np.allclose(tc_bbbb, three_center_bbbb)


# =============================================================================
# Container Equivalence Tests
# =============================================================================


class TestContainerEquivalence:
    """Tests that verify both container types produce equivalent results."""

    def test_both_containers_have_same_interface(self):
        """Test that both containers expose the same Hamiltonian interface."""
        h_canonical = create_test_hamiltonian(2, "canonical_four_center")
        h_df = create_test_hamiltonian(2, "density_fitted")

        # Both should have the same interface methods
        assert h_canonical.has_one_body_integrals() == h_df.has_one_body_integrals()
        assert h_canonical.has_two_body_integrals() == h_df.has_two_body_integrals()
        assert h_canonical.has_orbitals() == h_df.has_orbitals()
        assert h_canonical.is_restricted() == h_df.is_restricted()

    def test_two_body_equivalence_with_matching_integrals(self):
        """Test that both containers produce the same two-body integrals.

        When three_center integrals are chosen such that their contraction
        equals the canonical two_body integrals, both containers should
        produce identical results.
        """
        one_body = np.eye(2)
        two_body = 2.0 * np.ones(16)
        orbitals = create_test_orbitals(2)

        # Three-center chosen so contraction equals two_body
        three_center = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.6, 0.6, 0.6, 0.6],
                [0.8, 0.8, 0.8, 0.8],
            ]
        )

        canonical = CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.5, np.array([]))
        density_fitted = DensityFittedHamiltonianContainer(one_body, three_center, orbitals, 1.5, np.array([]))

        h_canonical = Hamiltonian(canonical)
        h_df = Hamiltonian(density_fitted)

        can_aaaa, can_aabb, can_bbbb = h_canonical.get_two_body_integrals()
        df_aaaa, df_aabb, df_bbbb = h_df.get_two_body_integrals()

        assert np.allclose(
            df_aaaa,
            can_aaaa,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            df_aabb,
            can_aabb,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            df_bbbb,
            can_bbbb,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_one_body_equivalence(self):
        """Test that both containers have the same one-body integrals."""
        h_canonical = create_test_hamiltonian(3, "canonical_four_center")
        h_df = create_test_hamiltonian(3, "density_fitted")

        can_alpha, can_beta = h_canonical.get_one_body_integrals()
        df_alpha, df_beta = h_df.get_one_body_integrals()

        assert np.allclose(df_alpha, can_alpha)
        assert np.allclose(df_beta, can_beta)

    def test_unrestricted_two_body_equivalence(self):
        """Test that both containers produce the same two-body integrals for unrestricted case."""
        coeffs_alpha = np.eye(2)
        coeffs_beta = np.array([[0.8, 0.6], [0.6, -0.8]])
        basis_set = create_test_basis_set(2, "test-equiv-unrestricted")
        orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

        one_body_alpha = np.eye(2)
        one_body_beta = np.eye(2) * 1.1

        # Canonical two-body values
        two_body_aaaa = np.ones(16) * 1.0
        two_body_aabb = np.ones(16) * 2.0
        two_body_bbbb = np.ones(16) * 3.0

        # Three-center chosen to produce matching two-body
        base_three_center = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.6, 0.6, 0.6, 0.6],
                [0.8, 0.8, 0.8, 0.8],
            ]
        )
        three_center_aaaa = math.sqrt(0.5) * base_three_center
        three_center_aabb = base_three_center
        three_center_bbbb = math.sqrt(1.5) * base_three_center

        canonical = CanonicalFourCenterHamiltonianContainer(
            one_body_alpha,
            one_body_beta,
            two_body_aaaa,
            two_body_aabb,
            two_body_bbbb,
            orbitals,
            1.5,
            np.eye(2),
            np.eye(2),
        )
        density_fitted = DensityFittedHamiltonianContainer(
            one_body_alpha,
            one_body_beta,
            three_center_aaaa,
            three_center_aabb,
            three_center_bbbb,
            orbitals,
            1.5,
            np.eye(2),
            np.eye(2),
        )

        h_canonical = Hamiltonian(canonical)
        h_df = Hamiltonian(density_fitted)

        can_aaaa, can_aabb, can_bbbb = h_canonical.get_two_body_integrals()
        df_aaaa, df_aabb, df_bbbb = h_df.get_two_body_integrals()

        assert np.allclose(
            df_aaaa,
            can_aaaa,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            df_aabb,
            can_aabb,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            df_bbbb,
            can_bbbb,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
