"""Tests for the Hamiltonian class aligned with the immutable API."""

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

from qdk_chemistry.data import CanonicalFourCenterHamiltonianContainer, Hamiltonian, ModelOrbitals, Orbitals

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import create_test_basis_set, create_test_hamiltonian, create_test_orbitals


class TestHamiltonian:
    def test_default_constructor(self):
        h = create_test_hamiltonian(2)
        assert isinstance(h, Hamiltonian)
        assert h.has_one_body_integrals()
        assert h.has_two_body_integrals()
        assert h.has_orbitals()

    def test_size_and_electron_counts(self):
        h = create_test_hamiltonian(3)
        assert isinstance(h, Hamiltonian)
        assert h.get_orbitals().get_num_molecular_orbitals() == 3

    def test_full_constructor(self):
        one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
        rng = np.random.default_rng(0)
        two_body = rng.random(2**4)
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        orbitals = Orbitals(coeffs, None, None, create_test_basis_set(2))

        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.5, np.array([])))
        assert h.has_one_body_integrals()
        assert h.has_two_body_integrals()
        assert h.has_orbitals()
        assert h.get_orbitals().get_num_molecular_orbitals() == 2
        assert h.get_core_energy() == 1.5

        aa, bb = h.get_one_body_integrals()
        assert np.array_equal(aa, one_body)
        assert np.array_equal(bb, one_body)

        aaaa, aabb, bbbb = h.get_two_body_integrals()
        assert np.array_equal(aaaa, two_body)
        assert np.array_equal(aabb, two_body)
        assert np.array_equal(bbbb, two_body)

    def test_one_body_integrals(self):
        one_body = np.array([[1.0, 0.2], [0.2, 1.5]])
        two_body = np.zeros(2**4)
        orbitals = create_test_orbitals(2)
        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.0, np.array([])))
        assert h.has_one_body_integrals()
        assert np.array_equal(h.get_one_body_integrals()[0], one_body)

    def test_two_body_integrals(self):
        one_body = np.eye(2)
        rng = np.random.default_rng(1)
        two_body = rng.random(2**4)
        orbitals = create_test_orbitals(2)
        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 0.0, np.array([])))
        assert h.has_two_body_integrals()
        # get_two_body_integrals returns (aaaa, aabb, bbbb) tuple
        aaaa, aabb, bbbb = h.get_two_body_integrals()
        # For restricted case, all should be the same and equal to two_body
        assert np.array_equal(aaaa, two_body)
        assert np.array_equal(aabb, two_body)
        assert np.array_equal(bbbb, two_body)

    def test_two_body_element_access(self):
        h = create_test_hamiltonian(2)
        val = h.get_two_body_element(0, 1, 1, 0)
        assert isinstance(val, float)

    def test_active_space_management(self):
        one_body = np.eye(3)
        two_body = np.zeros(3**4)
        orbitals = create_test_orbitals(3)
        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 2.5, np.array([])))
        assert h.get_core_energy() == 2.5

    def test_json_serialization(self):
        one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
        rng = np.random.default_rng(42)
        two_body = rng.random(2**4)
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        orbitals = Orbitals(coeffs, None, None, create_test_basis_set(2))
        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.5, np.array([])))

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
        # Compare each component of the two-body integrals tuple
        h_aaaa, h_aabb, h_bbbb = h.get_two_body_integrals()
        h2_aaaa, h2_aabb, h2_bbbb = h2.get_two_body_integrals()
        assert np.allclose(
            h_aaaa,
            h2_aaaa,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            h_aabb,
            h2_aabb,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            h_bbbb,
            h2_bbbb,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_json_file_io(self):
        one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
        rng = np.random.default_rng(42)
        two_body = rng.random(2**4)
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        orbitals = Orbitals(coeffs, None, None, create_test_basis_set(2))
        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.5, np.array([])))

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
            # Compare each component of the two-body integrals tuple
            h_aaaa, h_aabb, h_bbbb = h.get_two_body_integrals()
            h2_aaaa, h2_aabb, h2_bbbb = h2.get_two_body_integrals()
            assert np.allclose(
                h_aaaa,
                h2_aaaa,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                h_aabb,
                h2_aabb,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                h_bbbb,
                h2_bbbb,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        finally:
            Path(filename).unlink(missing_ok=True)

    def test_hdf5_file_io(self):
        one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
        rng = np.random.default_rng(42)
        two_body = rng.random(2**4)
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        orbitals = Orbitals(coeffs, None, None, create_test_basis_set(2))
        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.5, np.array([])))

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
            # Compare each component of the two-body integrals tuple
            h_aaaa, h_aabb, h_bbbb = h.get_two_body_integrals()
            h2_aaaa, h2_aabb, h2_bbbb = h2.get_two_body_integrals()
            assert np.allclose(
                h_aaaa,
                h2_aaaa,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                h_aabb,
                h2_aabb,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                h_bbbb,
                h2_bbbb,
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
        finally:
            Path(filename).unlink(missing_ok=True)

    def test_generic_file_io(self):
        one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
        rng = np.random.default_rng(42)
        two_body = rng.random(2**4)
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        orbitals = Orbitals(coeffs, None, None, create_test_basis_set(2))
        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.5, np.array([])))

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

    def test_file_io_validation(self):
        h = create_test_hamiltonian(2)
        with pytest.raises(RuntimeError, match="Unsupported file type"):
            h.to_file("test.txt", "txt")
        with pytest.raises(RuntimeError, match="Unsupported file type"):
            Hamiltonian.from_file("test.txt", "txt")
        with pytest.raises(RuntimeError, match="Unable to open Hamiltonian JSON file"):
            Hamiltonian.from_json_file("nonexistent.hamiltonian.json")
        with pytest.raises(RuntimeError, match="Unable to open Hamiltonian HDF5 file"):
            Hamiltonian.from_hdf5_file("nonexistent.hamiltonian.h5")

    def test_minimal_hamiltonian_json_roundtrip(self):
        h = create_test_hamiltonian(1)
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

    def test_static_methods_exist(self):
        one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
        rng = np.random.default_rng(42)
        two_body = rng.random(2**4)
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        orbitals = Orbitals(coeffs, None, None, create_test_basis_set(2))
        fock = np.array([])
        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.5, fock))

        with tempfile.NamedTemporaryFile(suffix=".hamiltonian.h5", delete=False) as f:
            filename = f.name
        try:
            h.to_hdf5_file(filename)
            assert Hamiltonian.from_hdf5_file(filename) is not None
        finally:
            Path(filename).unlink(missing_ok=True)

    def test_repr_method(self):
        """Test that __repr__ returns the summary."""
        h = create_test_hamiltonian(2)
        repr_str = repr(h)
        summary_str = h.get_summary()
        assert repr_str == summary_str
        assert "Hamiltonian" in repr_str

    def test_str_method(self):
        """Test that __str__ returns the summary."""
        h = create_test_hamiltonian(2)
        str_str = str(h)
        summary_str = h.get_summary()
        assert str_str == summary_str
        assert "Hamiltonian" in str_str

    def test_pickling_hamiltonian(self):
        """Test that Hamiltonian can be pickled and unpickled correctly."""
        h = create_test_hamiltonian(3)

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
            assert np.array_equal(h_restored_aaaa, h_aaaa)
            assert np.array_equal(h_restored_aabb, h_aabb)
            assert np.array_equal(h_restored_bbbb, h_bbbb)

        # Verify orbital consistency
        if h.has_orbitals():
            orig_orbs = h.get_orbitals()
            restored_orbs = h_restored.get_orbitals()
            assert orig_orbs.get_num_molecular_orbitals() == restored_orbs.get_num_molecular_orbitals()
            assert np.array_equal(orig_orbs.get_coefficients(), restored_orbs.get_coefficients())

    def test_restricted_hamiltonian_construction(self):
        """Test restricted Hamiltonian construction and properties."""
        # Create restricted orbitals (default behavior)
        coeffs = np.eye(3)
        basis_set = create_test_basis_set(3, "test-restricted")
        orbitals = Orbitals(coeffs, None, None, basis_set)

        assert orbitals.is_restricted()
        assert not orbitals.is_unrestricted()

        # Create restricted Hamiltonian
        rng = np.random.default_rng(42)
        one_body = rng.random((3, 3))
        one_body = 0.5 * (one_body + one_body.T)  # Make symmetric
        two_body = rng.random(3**4)
        inactive_fock = rng.random((3, 3))

        h = Hamiltonian(CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals, 1.0, inactive_fock))

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

        aaaa, aabb, bbbb = h.get_two_body_integrals()
        assert np.array_equal(aaaa, two_body)
        assert np.array_equal(aabb, two_body)
        assert np.array_equal(bbbb, two_body)

    def test_unrestricted_hamiltonian_construction(self):
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
        two_body_aaaa = rng.random(2**4)
        two_body_aabb = rng.random(2**4)
        two_body_bbbb = rng.random(2**4)
        inactive_fock_alpha = np.array([[0.5, 0.1], [0.1, 0.7]])
        inactive_fock_beta = np.array([[0.6, 0.2], [0.2, 0.8]])

        h = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
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
        )

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
        aaaa, aabb, bbbb = h.get_two_body_integrals()
        assert np.array_equal(aaaa, two_body_aaaa)
        assert np.array_equal(aabb, two_body_aabb)
        assert np.array_equal(bbbb, two_body_bbbb)

    def test_unrestricted_vs_restricted_serialization(self):
        """Test that restricted/unrestricted nature is preserved in serialization."""
        # Test restricted Hamiltonian
        coeffs = np.eye(2)
        basis_set = create_test_basis_set(2, "test-serialization-restricted")
        orbitals_restricted = Orbitals(coeffs, None, None, basis_set)

        one_body = np.array([[1.0, 0.1], [0.1, 1.0]])
        two_body = np.ones(16) * 0.5
        h_restricted = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(one_body, two_body, orbitals_restricted, 1.0, np.eye(2))
        )

        # Test unrestricted Hamiltonian
        coeffs_alpha = np.eye(2)
        coeffs_beta = np.array([[0.9, 0.1], [0.1, 0.9]])
        orbitals_unrestricted = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

        one_body_alpha = np.array([[1.0, 0.1], [0.1, 1.0]])
        one_body_beta = np.array([[1.1, 0.2], [0.2, 1.1]])
        two_body_aaaa = np.ones(16) * 1.0
        two_body_aabb = np.ones(16) * 2.0
        two_body_bbbb = np.ones(16) * 3.0
        h_unrestricted = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
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
        )

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

    def test_active_space_consistency(self):
        """Test that active space handling works correctly for both restricted and unrestricted."""
        # Test restricted case with active space
        model_orbitals_restricted = ModelOrbitals(4, True)
        assert model_orbitals_restricted.is_restricted()
        assert model_orbitals_restricted.has_active_space()

        # Create restricted Hamiltonian
        one_body = np.eye(4)
        two_body = np.zeros(4**4)
        h_restricted = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(one_body, two_body, model_orbitals_restricted, 0.0, np.eye(4))
        )
        assert h_restricted.is_restricted()

        # Test unrestricted case with active space
        model_orbitals_unrestricted = ModelOrbitals(4, False)
        assert not model_orbitals_unrestricted.is_restricted()
        assert model_orbitals_unrestricted.is_unrestricted()
        assert model_orbitals_unrestricted.has_active_space()

        # Create unrestricted Hamiltonian
        one_body_alpha = np.eye(4)
        one_body_beta = np.eye(4) * 1.1
        two_body_aaaa = np.zeros(4**4)
        two_body_aabb = np.zeros(4**4)
        two_body_bbbb = np.zeros(4**4)
        h_unrestricted = Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(
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
        )
        assert h_unrestricted.is_unrestricted()

        # Verify active space information is accessible
        alpha_indices, beta_indices = model_orbitals_restricted.get_active_space_indices()
        assert len(alpha_indices) == 4  # All orbitals active by default
        assert len(beta_indices) == 4
        assert alpha_indices == beta_indices

        alpha_indices_unres, beta_indices_unres = model_orbitals_unrestricted.get_active_space_indices()
        assert len(alpha_indices_unres) == 4  # All orbitals active by default
        assert len(beta_indices_unres) == 4


def test_hamiltonian_data_type_name():
    """Test that Hamiltonian has the correct _data_type_name class attribute."""
    assert hasattr(Hamiltonian, "_data_type_name")
    assert Hamiltonian._data_type_name == "hamiltonian"
