"""Tests for PySCF stability checker functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pickle
import tempfile

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.data import StabilityResult, Structure

try:
    import qdk_chemistry.plugins.pyscf  # noqa: F401
    from qdk_chemistry.constants import ANGSTROM_TO_BOHR

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")


class TestStabilityResultIO:
    """Test class for StabilityResult I/O functionality."""

    def create_test_stability_result(self):
        """Create a test StabilityResult with sample data."""
        # Create sample eigenvalues and eigenvectors
        internal_eigenvalues = np.array([0.5, 1.2, 2.1])
        external_eigenvalues = np.array([0.3, 0.8])

        internal_eigenvectors = np.array([[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])

        external_eigenvectors = np.array([[0.7, 0.714], [-0.714, 0.7]])

        return StabilityResult(
            True,  # internal_stable
            False,  # external_stable
            internal_eigenvalues,
            internal_eigenvectors,
            external_eigenvalues,
            external_eigenvectors,
        )

    def test_stability_result_json_serialization(self):
        """Test JSON serialization and deserialization."""
        result_out = self.create_test_stability_result()

        # Test direct JSON conversion
        json_data = result_out.to_json()
        result_in = StabilityResult.from_json(json_data)

        # Verify basic properties
        assert result_in.is_internal_stable() == result_out.is_internal_stable()
        assert result_in.is_external_stable() == result_out.is_external_stable()
        assert result_in.internal_size() == result_out.internal_size()
        assert result_in.external_size() == result_out.external_size()

        # Verify eigenvalues are preserved
        assert np.allclose(result_out.get_internal_eigenvalues(), result_in.get_internal_eigenvalues())
        assert np.allclose(result_out.get_external_eigenvalues(), result_in.get_external_eigenvalues())

        # Verify eigenvectors are preserved
        assert np.allclose(result_out.get_internal_eigenvectors(), result_in.get_internal_eigenvectors())
        assert np.allclose(result_out.get_external_eigenvectors(), result_in.get_external_eigenvectors())

        # Test file-based serialization
        with tempfile.NamedTemporaryFile(suffix=".stability_result.json") as tmp:
            filename = tmp.name
            result_out.to_json_file(filename)

            result_file = StabilityResult.from_json_file(filename)

            # Verify data is preserved
            assert result_file.internal_size() == result_out.internal_size()
            assert result_file.external_size() == result_out.external_size()
            assert np.allclose(result_out.get_internal_eigenvalues(), result_file.get_internal_eigenvalues())
            assert np.allclose(result_out.get_external_eigenvalues(), result_file.get_external_eigenvalues())

    def test_stability_result_hdf5_serialization(self):
        """Test HDF5 serialization and deserialization."""
        result_out = self.create_test_stability_result()

        try:
            with tempfile.NamedTemporaryFile(suffix=".stability_result.h5") as tmp:
                filename = tmp.name
                result_out.to_hdf5_file(filename)

                result_in = StabilityResult.from_hdf5_file(filename)

                # Verify data preservation
                assert result_in.is_internal_stable() == result_out.is_internal_stable()
                assert result_in.is_external_stable() == result_out.is_external_stable()
                assert np.allclose(result_out.get_internal_eigenvalues(), result_in.get_internal_eigenvalues())
                assert np.allclose(result_out.get_external_eigenvalues(), result_in.get_external_eigenvalues())
                assert np.allclose(result_out.get_internal_eigenvectors(), result_in.get_internal_eigenvectors())
                assert np.allclose(result_out.get_external_eigenvectors(), result_in.get_external_eigenvectors())
        except RuntimeError as e:
            pytest.skip(f"HDF5 test skipped - {e!s}")

    def test_stability_result_file_io_generic(self):
        """Test generic file I/O methods for StabilityResult."""
        result = self.create_test_stability_result()

        # Test JSON file I/O
        with tempfile.NamedTemporaryFile(suffix=".stability_result.json") as tmp_json:
            json_filename = tmp_json.name

            # Save using generic method
            result.to_file(json_filename, "json")

            # Load using generic method (static)
            result2 = StabilityResult.from_file(json_filename, "json")

            # Check equality
            assert result2.internal_size() == result.internal_size()
            assert result2.external_size() == result.external_size()

            # Check eigenvalues
            assert np.allclose(result.get_internal_eigenvalues(), result2.get_internal_eigenvalues())
            assert np.allclose(result.get_external_eigenvalues(), result2.get_external_eigenvalues())

        # Test HDF5 file I/O
        with tempfile.NamedTemporaryFile(suffix=".stability_result.h5") as tmp_hdf5:
            hdf5_filename = tmp_hdf5.name

            # Save using generic method
            result.to_file(hdf5_filename, "hdf5")

            # Load using generic method (static)
            result3 = StabilityResult.from_file(hdf5_filename, "hdf5")

            # Check equality
            assert result3.internal_size() == result.internal_size()
            assert result3.external_size() == result.external_size()

            # Check eigenvalues
            assert np.allclose(result.get_internal_eigenvalues(), result3.get_internal_eigenvalues())
            assert np.allclose(result.get_external_eigenvalues(), result3.get_external_eigenvalues())

        # Test unsupported file type
        with pytest.raises(ValueError, match="Unsupported file type"):
            result.to_file("test.stability_result.xyz", "xyz")

        with pytest.raises(ValueError, match="Unsupported file type"):
            StabilityResult.from_file("test.stability_result.xyz", "xyz")

    def test_stability_result_file_io_validation(self):
        """Test filename validation for StabilityResult file I/O."""
        result = self.create_test_stability_result()

        # Test filename validation for JSON files
        with pytest.raises(ValueError, match=r"'.stability_result.' before the file extension"):
            result.to_json_file("test.json")

        with pytest.raises(ValueError, match=r"'.stability_result.' before the file extension"):
            StabilityResult.from_json_file("test.json")

        # Test filename validation for HDF5 files
        with pytest.raises(ValueError, match=r"'.stability_result.' before the file extension"):
            result.to_hdf5_file("test.h5")

        with pytest.raises(ValueError, match=r"'.stability_result.' before the file extension"):
            StabilityResult.from_hdf5_file("test.h5")

        # Test non-existent file
        with pytest.raises(RuntimeError, match="Failed to open file for reading"):
            StabilityResult.from_json_file("nonexistent.stability_result.json")

        with pytest.raises(RuntimeError):
            StabilityResult.from_hdf5_file("nonexistent.stability_result.h5")

    def test_stability_result_empty_data_io(self):
        """Test I/O with empty StabilityResult (no eigenvalues/eigenvectors)."""
        empty_result = StabilityResult()
        assert empty_result.is_stable()  # Default should be stable
        assert empty_result.empty()  # Method call, not property

        # Test JSON I/O with empty data
        json_data = empty_result.to_json()
        from_json = StabilityResult.from_json(json_data)
        assert from_json.is_stable() == empty_result.is_stable()
        assert from_json.internal_size() == empty_result.internal_size()
        assert from_json.external_size() == empty_result.external_size()

        # Test file I/O with empty data
        with tempfile.NamedTemporaryFile(suffix=".stability_result.json") as tmp:
            filename = tmp.name
            empty_result.to_json_file(filename)
            empty_from_file = StabilityResult.from_json_file(filename)
            assert empty_from_file.is_stable()
            assert empty_from_file.empty()  # Method call, not property

    def test_stability_result_pickle_serialization_and_repr(self):
        """Test pickle serialization support and string representation for StabilityResult."""
        # Create a test stability result
        original = self.create_test_stability_result()

        # Test pickle serialization and deserialization
        serialized_data = pickle.dumps(original)
        assert isinstance(serialized_data, bytes)

        # Deserialize and verify
        deserialized = pickle.loads(serialized_data)

        # Verify stability result is preserved
        assert deserialized.internal_size() == original.internal_size()
        assert deserialized.external_size() == original.external_size()
        assert deserialized.is_internal_stable() == original.is_internal_stable()
        assert deserialized.is_external_stable() == original.is_external_stable()

        # Check eigenvalues are preserved
        np.testing.assert_array_almost_equal(
            original.get_internal_eigenvalues(), deserialized.get_internal_eigenvalues()
        )
        np.testing.assert_array_almost_equal(
            original.get_external_eigenvalues(), deserialized.get_external_eigenvalues()
        )

        # Check eigenvectors are preserved
        np.testing.assert_array_almost_equal(
            original.get_internal_eigenvectors(), deserialized.get_internal_eigenvectors()
        )
        np.testing.assert_array_almost_equal(
            original.get_external_eigenvectors(), deserialized.get_external_eigenvectors()
        )

        # Test multiple round trips
        current = original
        for _ in range(3):
            serialized = pickle.dumps(current)
            current = pickle.loads(serialized)

            # Verify data integrity after each round trip
            assert current.internal_size() == original.internal_size()
            assert current.external_size() == original.external_size()
            assert current.is_stable() == original.is_stable()

        # Test __repr__ uses summary function
        repr_output = repr(original)
        summary_output = original.get_summary()

        # They should be the same
        assert repr_output == summary_output

        # Check that typical summary content is present
        assert "StabilityResult" in repr_output or "stable" in repr_output.lower()

        # Test __str__ vs __repr__ consistency
        str_output = str(original)
        assert str_output == repr_output

        # Test that serialized objects still have proper repr functionality
        serialized_data = pickle.dumps(original)
        deserialized = pickle.loads(serialized_data)

        # Both should have same repr output
        original_repr = repr(original)
        deserialized_repr = repr(deserialized)

        assert original_repr == deserialized_repr

    def test_stability_result_methods(self):
        """Test access methods for StabilityResult (consistent with Orbitals - no properties)."""
        result = self.create_test_stability_result()

        # Test basic stability methods
        assert isinstance(result.is_stable(), bool)
        assert isinstance(result.is_internal_stable(), bool)
        assert isinstance(result.is_external_stable(), bool)

        # Test data validation methods
        assert isinstance(result.empty(), bool)
        assert isinstance(result.has_internal_result(), bool)
        assert isinstance(result.has_external_result(), bool)

        # Test eigenvalue getter methods
        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()
        internal_eigenvectors = result.get_internal_eigenvectors()
        external_eigenvectors = result.get_external_eigenvectors()

        # Verify types and consistency
        assert isinstance(internal_eigenvalues, np.ndarray)
        assert isinstance(external_eigenvalues, np.ndarray)
        assert isinstance(internal_eigenvectors, np.ndarray)
        assert isinstance(external_eigenvectors, np.ndarray)

        # Test size methods
        assert result.internal_size() == len(internal_eigenvalues)
        assert result.external_size() == len(external_eigenvalues)


def create_water_structure():
    """Create a water molecule structure.

    Crawford geometry - same as used in C++ tests.
    """
    symbols = ["O", "H", "H"]
    coords = np.array(
        [
            [0.000000000, -0.0757918436, 0.000000000000],
            [0.866811829, 0.6014357793, -0.000000000000],
            [-0.866811829, 0.6014357793, -0.000000000000],
        ]
    )
    coords *= ANGSTROM_TO_BOHR
    return Structure(symbols, coords)


def create_stretched_n2_structure(distance_angstrom=1.2):
    """Create a Stretched nitrogen molecule (N2) structure.

    Crawford geometry - same as used in C++ tests.
    """
    symbols = ["N", "N"]
    coords = np.array(
        [
            [0.000000000, 0.000000000000, 0.000000000000],
            [distance_angstrom, 0.000000000000, -0.000000000000],
        ]
    )
    coords *= ANGSTROM_TO_BOHR
    return Structure(symbols, coords)


def create_o2_structure():
    """Create an oxygen molecule (O2) structure."""
    symbols = ["O", "O"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.208]])
    coords *= ANGSTROM_TO_BOHR
    return Structure(symbols, coords)


def create_bn_plus_structure():
    """Create a boron nitride cation (BN+) structure."""
    symbols = ["B", "N"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.2765],
        ]
    )
    coords *= ANGSTROM_TO_BOHR
    return Structure(symbols, coords)


def create_c2_plus_structure():
    """Create a carbon dimer cation (C2+) structure."""
    symbols = ["C", "C"]
    coords = np.array(
        [
            [0.0, 0.0, 0.6240212600],
            [0.0, 0.0, -0.6240212600],
        ]
    )
    coords *= ANGSTROM_TO_BOHR
    return Structure(symbols, coords)


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestPyscfStabilityChecker:
    """Test class for PySCF stability checker functionality."""

    def test_pyscf_stability_checker_rhf_water_stable(self):
        """Test PySCF stability checker on stable RHF water molecule."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "def2-svp")
        _, wavefunction = scf_solver.run(water, 0, 1)

        # Perform stability analysis
        stability_checker = algorithms.create("stability_checker", "pyscf")
        is_stable, result = stability_checker.run(wavefunction)

        # Check results
        assert result is not None
        assert isinstance(result, StabilityResult)
        assert isinstance(is_stable, bool)
        assert result.is_stable() is True  # Water RHF should be stable

        # Check that we have both internal and external results for RHF
        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()

        assert len(internal_eigenvalues) > 0
        assert len(external_eigenvalues) > 0

        # All eigenvalues should be positive for a stable system
        assert np.all(internal_eigenvalues > -1e-4)
        assert np.all(external_eigenvalues > -1e-4)

        # Check individual stability components
        assert result.is_internal_stable() is True
        assert result.is_external_stable() is True

        # Check eigenvectors
        internal_eigenvectors = result.get_internal_eigenvectors()
        external_eigenvectors = result.get_external_eigenvectors()

        assert internal_eigenvectors.shape[1] == len(internal_eigenvalues)
        assert external_eigenvectors.shape[1] == len(external_eigenvalues)

        # Test smallest eigenvalue methods
        smallest_internal = result.get_smallest_internal_eigenvalue()
        smallest_external = result.get_smallest_external_eigenvalue()
        smallest_overall = result.get_smallest_eigenvalue()

        # Reference values (from initial run)
        ref_smallest_internal = 1.1915284105596065
        ref_smallest_external = 0.1798655099964312
        ref_smallest_overall = 0.1798655099964312

        # Check against reference values with dynamic tolerance
        alg_tol = stability_checker.settings().get("davidson_tolerance")
        eigenvalue_tolerance = 10 * alg_tol
        assert abs(smallest_internal - ref_smallest_internal) < eigenvalue_tolerance
        assert abs(smallest_external - ref_smallest_external) < eigenvalue_tolerance
        assert abs(smallest_overall - ref_smallest_overall) < eigenvalue_tolerance

        # Test internal and external analysis separately
        # Create new stability checker with modified settings
        stability_checker_internal = algorithms.create("stability_checker", "pyscf")
        settings_internal = stability_checker_internal.settings()

        # Test only internal analysis
        settings_internal.set("internal", True)
        settings_internal.set("external", False)
        is_stable_internal, result_internal = stability_checker_internal.run(wavefunction)
        assert result_internal.is_internal_stable() is True
        assert result_internal.internal_size() > 0
        assert result_internal.external_size() == 0
        assert result_internal.is_stable() is True  # Should be True if internal is True and external is empty
        assert is_stable_internal is True

        # Test only external analysis
        stability_checker_external = algorithms.create("stability_checker", "pyscf")
        settings_external = stability_checker_external.settings()
        settings_external.set("internal", False)
        settings_external.set("external", True)
        is_stable_external, result_external = stability_checker_external.run(wavefunction)
        assert result_external.is_external_stable() is True
        assert is_stable_external is True
        assert result_external.internal_size() == 0
        assert result_external.external_size() > 0
        assert result_external.is_stable() is True  # Should be True if external is True and internal is empty

    def test_pyscf_stability_checker_uhf_o2(self):
        """Test PySCF stability checker on UHF oxygen molecule."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "def2-svp")
        _, wavefunction = scf_solver.run(o2, 0, 3)

        # Perform stability analysis (only internal, since external not supported for UHF)
        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("external", False)  # Explicitly disable external for UHF
        _, result = stability_checker.run(wavefunction)

        # Check results
        assert result is not None
        assert isinstance(result, StabilityResult)

        # For UHF, we should have internal results but no external results
        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()

        assert len(internal_eigenvalues) > 0
        assert result.is_internal_stable() is True
        assert len(external_eigenvalues) == 0  # No external analysis for UHF

        # Reference value (from initial run)
        ref_smallest_internal = 0.04196462242858642
        # Check against reference value with dynamic tolerance
        smallest_internal = result.get_smallest_internal_eigenvalue()
        alg_tol = stability_checker.settings().get("davidson_tolerance")
        eigenvalue_tolerance = 10 * alg_tol
        assert abs(smallest_internal - ref_smallest_internal) < eigenvalue_tolerance

        # Check eigenvectors
        internal_eigenvectors = result.get_internal_eigenvectors()
        external_eigenvectors = result.get_external_eigenvectors()
        assert internal_eigenvectors.size > 0
        assert external_eigenvectors.size == 0

        # Test that external analysis raises an error for UHF
        stability_checker_external = algorithms.create("stability_checker", "pyscf")
        settings_external = stability_checker_external.settings()
        settings_external.set("external", True)  # Request external analysis (not supported for UHF)
        # This should raise a ValueError
        with pytest.raises(ValueError, match=r"External stability analysis.*is not supported for UHF"):
            stability_checker_external.run(wavefunction)

    def test_pyscf_stability_checker_rohf_o2(self):
        """Test PySCF stability checker on ROHF oxygen molecule."""
        o2 = create_o2_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "def2-svp")
        scf_solver.settings().set("force_restricted", True)
        _, wavefunction = scf_solver.run(o2, 0, 3)

        # Perform stability analysis (only internal, since external not supported for ROHF)
        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("external", False)  # Explicitly disable external for ROHF
        _, result = stability_checker.run(wavefunction)

        # Test that external analysis raises an error for ROHF
        stability_checker_external = algorithms.create("stability_checker", "pyscf")
        settings_external = stability_checker_external.settings()
        settings_external.set("external", True)  # Request external analysis (not supported for ROHF)
        # This should raise a ValueError
        with pytest.raises(ValueError, match=r"External stability analysis.*is not supported for ROHF"):
            stability_checker_external.run(wavefunction)

        # Check results
        assert result is not None
        assert isinstance(result, StabilityResult)

        # For ROHF, we should have internal results but no external results (external gives warning)
        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()

        assert len(internal_eigenvalues) > 0
        assert len(external_eigenvalues) == 0  # No external analysis for ROHF
        assert result.is_internal_stable() is False  # ROHF O2 should be internally unstable

        # Reference value (from initial run)
        ref_smallest_internal = -0.00965024239084978
        # Check against reference value with dynamic tolerance
        smallest_internal = result.get_smallest_internal_eigenvalue()
        alg_tol = stability_checker.settings().get("davidson_tolerance")
        eigenvalue_tolerance = 10 * alg_tol
        assert abs(smallest_internal - ref_smallest_internal) < eigenvalue_tolerance

        # Check eigenvectors
        internal_eigenvectors = result.get_internal_eigenvectors()
        external_eigenvectors = result.get_external_eigenvectors()
        assert internal_eigenvectors.size > 0
        assert external_eigenvectors.size == 0

    def test_pyscf_stability_checker_no_analysis_requested(self):
        """Test stability checker when no analysis is requested."""
        water = create_water_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "sto-3g")
        _, wavefunction = scf_solver.run(water, 0, 1)

        stability_checker = algorithms.create("stability_checker", "pyscf")
        settings = stability_checker.settings()
        settings.set("internal", False)
        settings.set("external", False)  # No analysis requested

        is_stable, result = stability_checker.run(wavefunction)

        # Should return a result with empty arrays but stable defaults
        assert result is not None
        assert result.is_internal_stable() is True  # Default value
        assert result.is_external_stable() is True  # Default value
        assert result.is_stable() is True  # Both defaults are True
        assert is_stable is True  # Should also be True when both defaults are True
        assert len(result.get_internal_eigenvalues()) == 0
        assert len(result.get_external_eigenvalues()) == 0
        assert result.get_internal_eigenvectors().size == 0
        assert result.get_external_eigenvectors().size == 0
        assert result.internal_size() == 0
        assert result.external_size() == 0

    def test_pyscf_stability_checker_n2_rhf_1_2_angstrom(self):
        """Test PySCF stability checker on N2 at 1.2 Å with RHF - no internal, one external instability."""
        n2 = create_stretched_n2_structure(distance_angstrom=1.2)
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "def2-svp")
        _, wavefunction = scf_solver.run(n2, 0, 1)

        # Perform stability analysis
        stability_checker = algorithms.create("stability_checker", "pyscf")
        is_stable, result = stability_checker.run(wavefunction)

        # Check results - should be internally stable but externally unstable
        assert result is not None
        assert isinstance(result, StabilityResult)
        assert is_stable is False  # Overall unstable due to external instability
        assert result.is_internal_stable() is True  # No internal instability
        assert result.is_external_stable() is False  # One external instability
        assert result.is_stable() is False  # Overall unstable due to external

        # Check that we have both internal and external results
        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()

        assert len(internal_eigenvalues) > 0
        assert len(external_eigenvalues) > 0

        # All internal eigenvalues should be positive (stable)
        stability_tol = stability_checker.settings().get("stability_tolerance")
        assert np.all(internal_eigenvalues > stability_tol)

        # Should have at least one negative external eigenvalue (unstable)
        assert np.any(external_eigenvalues < stability_tol)

        # Print eigenvalues for reference
        smallest_internal = result.get_smallest_internal_eigenvalue()
        smallest_external = result.get_smallest_external_eigenvalue()
        # Reference values (from initial run)
        ref_smallest_internal = 0.6683982584414123
        ref_smallest_external = -0.04997491473779583

        # Check against reference values with dynamic tolerance
        alg_tol = stability_checker.settings().get("davidson_tolerance")
        eigenvalue_tolerance = 10 * alg_tol
        assert abs(smallest_internal - ref_smallest_internal) < eigenvalue_tolerance
        assert abs(smallest_external - ref_smallest_external) < eigenvalue_tolerance

    def test_pyscf_stability_checker_n2_rhf_1_6_angstrom(self):
        """Test PySCF stability checker on N2 molecule at 1.6 Å with RHF - should have one internal instability."""
        n2 = create_stretched_n2_structure(distance_angstrom=1.6)
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "def2-svp")
        _, wavefunction = scf_solver.run(n2, 0, 1)

        # Perform stability analysis
        stability_checker = algorithms.create("stability_checker", "pyscf")
        is_stable, result = stability_checker.run(wavefunction)

        # Check results - should have internal instability
        assert result is not None
        assert isinstance(result, StabilityResult)
        assert is_stable is False  # Overall unstable due to internal instability
        assert result.is_internal_stable() is False  # One internal instability

        # Check that we have both internal and external results
        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()

        assert len(internal_eigenvalues) > 0
        assert len(external_eigenvalues) > 0

        # Should have at least one negative internal eigenvalue (unstable)
        stability_tol = stability_checker.settings().get("stability_tolerance")
        assert np.any(internal_eigenvalues < stability_tol)

        # Print eigenvalues for reference
        smallest_internal = result.get_smallest_internal_eigenvalue()
        smallest_external = result.get_smallest_external_eigenvalue()
        # Reference values (from initial run)
        ref_smallest_internal = -0.26163382707539584
        ref_smallest_external = -0.28245222121208535

        # Check against reference values with dynamic tolerance
        alg_tol = stability_checker.settings().get("davidson_tolerance")
        eigenvalue_tolerance = 10 * alg_tol
        assert abs(smallest_internal - ref_smallest_internal) < eigenvalue_tolerance
        assert abs(smallest_external - ref_smallest_external) < eigenvalue_tolerance

    def test_pyscf_stability_checker_bn_plus_uhf(self):
        """Test PySCF stability checker on BN+ cation with UHF."""
        bn_plus = create_bn_plus_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "def2-svp")
        _, wavefunction = scf_solver.run(bn_plus, 1, 2)  # charge=1, multiplicity=2

        # Perform stability analysis (only internal for UHF)
        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("external", False)  # Explicitly disable external for UHF
        _, result = stability_checker.run(wavefunction)

        # Check results
        assert result is not None
        assert isinstance(result, StabilityResult)

        # For UHF, we should have internal results but no external results
        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()

        assert len(internal_eigenvalues) > 0
        assert len(external_eigenvalues) == 0  # No external analysis for UHF

        # Check that internal analysis worked
        assert result.is_internal_stable() is not None
        assert result.is_external_stable() is True  # Should be True (default) since no external analysis

        # Print smallest eigenvalue for reference
        ref_smallest_internal = -0.15872092489909065

        # Check against reference value with dynamic tolerance
        smallest_internal = result.get_smallest_internal_eigenvalue()
        alg_tol = stability_checker.settings().get("davidson_tolerance")
        eigenvalue_tolerance = 10 * alg_tol
        assert abs(smallest_internal - ref_smallest_internal) < eigenvalue_tolerance

        # Check that this system has internal instability
        stability_tol = stability_checker.settings().get("stability_tolerance")
        assert smallest_internal < stability_tol  # Should be unstable
        assert result.is_internal_stable() is False  # Should be internally unstable

        # Check eigenvectors
        internal_eigenvectors = result.get_internal_eigenvectors()
        external_eigenvectors = result.get_external_eigenvectors()

        assert internal_eigenvectors.size > 0
        assert external_eigenvectors.size == 0

    def test_pyscf_stability_checker_c2_plus_rohf(self):
        """Test PySCF stability checker on C2+ cation with ROHF."""
        c2_plus = create_c2_plus_structure()
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("basis_set", "def2-svp")
        scf_solver.settings().set("force_restricted", True)  # Force ROHF
        _, wavefunction = scf_solver.run(c2_plus, 1, 2)  # charge=1, multiplicity=2

        # Perform stability analysis (only internal for ROHF)
        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("external", False)  # Explicitly disable external for ROHF
        _, result = stability_checker.run(wavefunction)

        # Check results
        assert result is not None
        assert isinstance(result, StabilityResult)

        # For ROHF, we should have internal results but no external results
        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()

        assert len(internal_eigenvalues) > 0
        assert len(external_eigenvalues) == 0  # No external analysis for ROHF

        # Check that internal analysis worked
        assert result.is_internal_stable() is not None
        assert result.is_external_stable() is True  # Should be True (default) since no external analysis

        # Reference value (from initial run)
        smallest_internal = result.get_smallest_internal_eigenvalue()
        ref_smallest_internal = -0.08256762551795531

        # Check against reference value with dynamic tolerance
        alg_tol = stability_checker.settings().get("davidson_tolerance")
        eigenvalue_tolerance = 10 * alg_tol
        assert abs(smallest_internal - ref_smallest_internal) < eigenvalue_tolerance

        # Check that this system has internal instability
        stability_tol = stability_checker.settings().get("stability_tolerance")
        assert smallest_internal < stability_tol  # Should be unstable
        assert result.is_internal_stable() is False  # Should be internally unstable

        # Check eigenvectors
        internal_eigenvectors = result.get_internal_eigenvectors()
        external_eigenvectors = result.get_external_eigenvectors()

        assert internal_eigenvectors.size > 0
        assert external_eigenvectors.size == 0
