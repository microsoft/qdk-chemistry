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

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import create_test_basis_set, create_test_hamiltonian, create_test_orbitals

try:
    from qdk_chemistry.data import Hamiltonian, Orbitals

    HAMILTONIAN_AVAILABLE = True
except ImportError:
    HAMILTONIAN_AVAILABLE = False


@pytest.mark.skipif(not HAMILTONIAN_AVAILABLE, reason="Hamiltonian class not available")
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

        h = Hamiltonian(one_body, two_body, orbitals, 1.5, np.array([]))
        assert h.has_one_body_integrals()
        assert h.has_two_body_integrals()
        assert h.has_orbitals()
        assert h.get_orbitals().get_num_molecular_orbitals() == 2
        assert h.get_core_energy() == 1.5
        np.testing.assert_array_equal(h.get_one_body_integrals(), one_body)
        np.testing.assert_array_equal(h.get_two_body_integrals(), two_body)

    def test_one_body_integrals(self):
        one_body = np.array([[1.0, 0.2], [0.2, 1.5]])
        two_body = np.zeros(2**4)
        orbitals = create_test_orbitals(2)
        h = Hamiltonian(one_body, two_body, orbitals, 0.0, np.array([]))
        assert h.has_one_body_integrals()
        np.testing.assert_array_equal(h.get_one_body_integrals(), one_body)

    def test_two_body_integrals(self):
        one_body = np.eye(2)
        rng = np.random.default_rng(1)
        two_body = rng.random(2**4)
        orbitals = create_test_orbitals(2)
        h = Hamiltonian(one_body, two_body, orbitals, 0.0, np.array([]))
        assert h.has_two_body_integrals()
        np.testing.assert_array_equal(h.get_two_body_integrals(), two_body)

    def test_two_body_element_access(self):
        h = create_test_hamiltonian(2)
        val = h.get_two_body_element(0, 1, 1, 0)
        assert isinstance(val, float)

    def test_active_space_management(self):
        one_body = np.eye(3)
        two_body = np.zeros(3**4)
        orbitals = create_test_orbitals(3)
        h = Hamiltonian(one_body, two_body, orbitals, 2.5, np.array([]))
        assert h.get_core_energy() == 2.5

    def test_json_serialization(self):
        one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
        rng = np.random.default_rng(42)
        two_body = rng.random(2**4)
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        orbitals = Orbitals(coeffs, None, None, create_test_basis_set(2))
        h = Hamiltonian(one_body, two_body, orbitals, 1.5, np.array([]))

        data = json.loads(h.to_json())
        assert isinstance(data, dict)
        assert data["core_energy"] == 1.5
        assert data["has_one_body_integrals"] is True
        assert data["has_two_body_integrals"] is True
        assert data["has_orbitals"] is True

        h2 = Hamiltonian.from_json(json.dumps(data))
        assert h2.get_orbitals().get_num_molecular_orbitals() == 2
        assert h2.get_core_energy() == 1.5
        assert h2.has_one_body_integrals()
        assert h2.has_two_body_integrals()
        assert h2.has_orbitals()
        assert np.allclose(
            h.get_one_body_integrals(),
            h2.get_one_body_integrals(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            h.get_two_body_integrals(),
            h2.get_two_body_integrals(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_json_file_io(self):
        one_body = np.array([[1.0, 0.5], [0.5, 2.0]])
        rng = np.random.default_rng(42)
        two_body = rng.random(2**4)
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        orbitals = Orbitals(coeffs, None, None, create_test_basis_set(2))
        h = Hamiltonian(one_body, two_body, orbitals, 1.5, np.array([]))

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
                h.get_one_body_integrals(),
                h2.get_one_body_integrals(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                h.get_two_body_integrals(),
                h2.get_two_body_integrals(),
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
        h = Hamiltonian(one_body, two_body, orbitals, 1.5, np.array([]))

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
                h.get_one_body_integrals(),
                h2.get_one_body_integrals(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                h.get_two_body_integrals(),
                h2.get_two_body_integrals(),
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
        h = Hamiltonian(one_body, two_body, orbitals, 1.5, np.array([]))

        with tempfile.NamedTemporaryFile(suffix=".hamiltonian.json", delete=False) as f:
            json_filename = f.name
        try:
            h.to_file(json_filename, "json")
            assert Path(json_filename).exists()
            h2 = Hamiltonian.from_file(json_filename, "json")
            assert h2.get_orbitals().get_num_molecular_orbitals() == 2
            assert np.allclose(
                h.get_one_body_integrals(),
                h2.get_one_body_integrals(),
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
                h.get_one_body_integrals(),
                h3.get_one_body_integrals(),
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
        with pytest.raises(RuntimeError, match="Cannot open file"):
            Hamiltonian.from_json_file("nonexistent.hamiltonian.json")
        with pytest.raises(RuntimeError, match="HDF5 error: H5Fopen failed"):
            Hamiltonian.from_hdf5_file("nonexistent.hamiltonian.h5")

    def test_minimal_hamiltonian_json_roundtrip(self):
        h = create_test_hamiltonian(1)
        data = json.loads(h.to_json())
        assert data["core_energy"] == 0.0
        assert data["has_one_body_integrals"] is True
        assert data["has_two_body_integrals"] is True
        assert data["has_orbitals"] is True
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
        h = Hamiltonian(one_body, two_body, orbitals, 1.5, fock)

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
            np.testing.assert_array_equal(h_restored.get_one_body_integrals(), h.get_one_body_integrals())

        if h.has_two_body_integrals():
            np.testing.assert_array_equal(h_restored.get_two_body_integrals(), h.get_two_body_integrals())

        # Verify orbital consistency
        if h.has_orbitals():
            orig_orbs = h.get_orbitals()
            restored_orbs = h_restored.get_orbitals()
            assert orig_orbs.get_num_molecular_orbitals() == restored_orbs.get_num_molecular_orbitals()
            np.testing.assert_array_equal(orig_orbs.get_coefficients(), restored_orbs.get_coefficients())
