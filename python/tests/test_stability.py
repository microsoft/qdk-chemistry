"""Tests for PySCF stability checker functionality."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pickle
import tempfile
import warnings

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.constants import ANGSTROM_TO_BOHR
from qdk_chemistry.data import StabilityResult, Structure
from qdk_chemistry.utils import rotate_orbitals

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance

try:
    import pyscf  # noqa: F401

    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False

if PYSCF_AVAILABLE:
    import qdk_chemistry.plugins.pyscf as pyscf_plugin

    pyscf_plugin.load()

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
        assert np.allclose(
            result_out.get_internal_eigenvalues(),
            result_in.get_internal_eigenvalues(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            result_out.get_external_eigenvalues(),
            result_in.get_external_eigenvalues(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Verify eigenvectors are preserved
        assert np.allclose(
            result_out.get_internal_eigenvectors(),
            result_in.get_internal_eigenvectors(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            result_out.get_external_eigenvectors(),
            result_in.get_external_eigenvectors(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Test file-based serialization
        with tempfile.NamedTemporaryFile(suffix=".stability_result.json") as tmp:
            filename = tmp.name
            result_out.to_json_file(filename)

            result_file = StabilityResult.from_json_file(filename)

            # Verify data is preserved
            assert result_file.internal_size() == result_out.internal_size()
            assert result_file.external_size() == result_out.external_size()
            assert np.allclose(
                result_out.get_internal_eigenvalues(),
                result_file.get_internal_eigenvalues(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                result_out.get_external_eigenvalues(),
                result_file.get_external_eigenvalues(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

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
                assert np.allclose(
                    result_out.get_internal_eigenvalues(),
                    result_in.get_internal_eigenvalues(),
                    rtol=float_comparison_relative_tolerance,
                    atol=float_comparison_absolute_tolerance,
                )
                assert np.allclose(
                    result_out.get_external_eigenvalues(),
                    result_in.get_external_eigenvalues(),
                    rtol=float_comparison_relative_tolerance,
                    atol=float_comparison_absolute_tolerance,
                )
                assert np.allclose(
                    result_out.get_internal_eigenvectors(),
                    result_in.get_internal_eigenvectors(),
                    rtol=float_comparison_relative_tolerance,
                    atol=float_comparison_absolute_tolerance,
                )
                assert np.allclose(
                    result_out.get_external_eigenvectors(),
                    result_in.get_external_eigenvectors(),
                    rtol=float_comparison_relative_tolerance,
                    atol=float_comparison_absolute_tolerance,
                )
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
            assert np.allclose(
                result.get_internal_eigenvalues(),
                result2.get_internal_eigenvalues(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                result.get_external_eigenvalues(),
                result2.get_external_eigenvalues(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

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
            assert np.allclose(
                result.get_internal_eigenvalues(),
                result3.get_internal_eigenvalues(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )
            assert np.allclose(
                result.get_external_eigenvalues(),
                result3.get_external_eigenvalues(),
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

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
        with pytest.raises(RuntimeError, match="Unable to open StabilityResult JSON file"):
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
        assert np.allclose(
            original.get_internal_eigenvalues(),
            deserialized.get_internal_eigenvalues(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            original.get_external_eigenvalues(),
            deserialized.get_external_eigenvalues(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check eigenvectors are preserved
        assert np.allclose(
            original.get_internal_eigenvectors(),
            deserialized.get_internal_eigenvectors(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.allclose(
            original.get_external_eigenvectors(),
            deserialized.get_external_eigenvectors(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
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

    def _create_scf_solver(self, backend="pyscf", scf_type="auto"):
        """Helper to create SCF solver with common settings for different backends."""
        scf_solver = algorithms.create("scf_solver", backend)
        scf_solver.settings().set("scf_type", scf_type)
        return scf_solver

    def _create_stability_checker(self, internal=True, external=True):
        """Helper to create stability checker with specified settings (always uses PySCF)."""
        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("internal", internal)
        stability_checker.settings().set("external", external)
        return stability_checker

    def _assert_basic_stability_result(self, result, is_stable, has_internal=True, has_external=True):
        """Helper to check basic stability result properties."""
        assert result is not None
        assert isinstance(result, StabilityResult)
        assert isinstance(is_stable, bool)

        internal_eigenvalues = result.get_internal_eigenvalues()
        external_eigenvalues = result.get_external_eigenvalues()

        if has_internal:
            assert len(internal_eigenvalues) > 0
        else:
            assert len(internal_eigenvalues) == 0

        if has_external:
            assert len(external_eigenvalues) > 0
        else:
            assert len(external_eigenvalues) == 0

    def _check_reference_eigenvalue(self, result, stability_checker, ref_value, is_internal=True):
        """Helper to check eigenvalue against reference with dynamic tolerance."""
        smallest = (
            result.get_smallest_internal_eigenvalue() if is_internal else result.get_smallest_external_eigenvalue()
        )
        alg_tol = stability_checker.settings().get("davidson_tolerance")
        assert abs(smallest - ref_value) < alg_tol

    def test_pyscf_stability_checker_no_analysis_requested(self):
        """Test stability checker when no analysis is requested."""
        water = create_water_structure()
        scf_solver = self._create_scf_solver()
        _, wavefunction = scf_solver.run(water, 0, 1, "sto-3g")

        stability_checker = self._create_stability_checker(internal=False, external=False)
        is_stable, result = stability_checker.run(wavefunction)

        # Should return empty result with stable defaults
        self._assert_basic_stability_result(result, is_stable, has_internal=False, has_external=False)
        assert result.is_stable() is True
        assert is_stable is True

    @pytest.mark.parametrize("scf_backend", ["pyscf", "qdk"])  # Can add "qdk" when available
    def test_pyscf_stability_checker_rhf_water_stable(self, scf_backend):
        """Test PySCF stability checker on stable RHF water molecule with different SCF backends."""
        water = create_water_structure()
        scf_solver = self._create_scf_solver(backend=scf_backend)
        _, wavefunction = scf_solver.run(water, 0, 1, "def2-svp")

        # Test full stability analysis
        stability_checker = self._create_stability_checker()
        is_stable, result = stability_checker.run(wavefunction)

        self._assert_basic_stability_result(result, is_stable, has_internal=True, has_external=True)
        assert result.is_stable() is True  # Water RHF should be stable
        assert result.is_internal_stable() is True
        assert result.is_external_stable() is True

        # Check reference values
        self._check_reference_eigenvalue(result, stability_checker, 1.1915284105596065, is_internal=True)
        self._check_reference_eigenvalue(result, stability_checker, 0.1798655099964312, is_internal=False)

        # Test internal-only analysis
        stability_checker_internal = self._create_stability_checker(internal=True, external=False)
        is_stable_internal, result_internal = stability_checker_internal.run(wavefunction)
        self._assert_basic_stability_result(result_internal, is_stable_internal, has_internal=True, has_external=False)
        assert result_internal.is_internal_stable() is True
        assert is_stable_internal is True

        # Test external-only analysis
        stability_checker_external = self._create_stability_checker(internal=False, external=True)
        is_stable_external, result_external = stability_checker_external.run(wavefunction)
        self._assert_basic_stability_result(result_external, is_stable_external, has_internal=False, has_external=True)
        assert result_external.is_external_stable() is True
        assert is_stable_external is True

    @pytest.mark.parametrize("scf_backend", ["pyscf", "qdk"])
    def test_pyscf_stability_checker_uhf_o2(self, scf_backend):
        """Test PySCF stability checker on UHF oxygen molecule with different SCF backends."""
        o2 = create_o2_structure()
        scf_solver = self._create_scf_solver(backend=scf_backend)
        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")

        # Test internal-only analysis (external not supported for UHF)
        stability_checker = self._create_stability_checker(internal=True, external=False)
        _, result = stability_checker.run(wavefunction)

        self._assert_basic_stability_result(result, True, has_internal=True, has_external=False)
        assert result.is_internal_stable() is True
        self._check_reference_eigenvalue(result, stability_checker, 0.04196462242858642, is_internal=True)

        # Test that external analysis raises error for UHF
        stability_checker_external = self._create_stability_checker(internal=False, external=True)
        with pytest.raises(ValueError, match=r"External stability analysis.*is not supported for UHF"):
            stability_checker_external.run(wavefunction)

    def test_pyscf_stability_checker_rohf_o2(self):
        """Test PySCF stability checker on ROHF oxygen molecule."""
        o2 = create_o2_structure()
        scf_solver = self._create_scf_solver(scf_type="restricted")
        _, wavefunction = scf_solver.run(o2, 0, 3, "def2-svp")

        # Test internal-only analysis (external not supported for ROHF)
        stability_checker = self._create_stability_checker(internal=True, external=False)
        _, result = stability_checker.run(wavefunction)

        self._assert_basic_stability_result(result, False, has_internal=True, has_external=False)
        assert result.is_internal_stable() is False  # ROHF O2 should be internally unstable
        self._check_reference_eigenvalue(result, stability_checker, -0.00965024239084978, is_internal=True)

        # Test that external analysis raises error for ROHF
        stability_checker_external = self._create_stability_checker(internal=False, external=True)
        with pytest.raises(ValueError, match=r"External stability analysis.*is not supported for ROHF"):
            stability_checker_external.run(wavefunction)

    @pytest.mark.parametrize("scf_backend", ["pyscf", "qdk"])
    @pytest.mark.parametrize(
        ("distance", "expected_internal_stable", "expected_external_stable", "ref_internal", "ref_external"),
        [
            (1.2, True, False, 0.6683982584414123, -0.04997491473779583),  # External instability
            (1.6, False, False, -0.26163382707539584, -0.28245222121208535),  # Internal instability
        ],
    )
    def test_pyscf_stability_checker_n2_rhf_instabilities(
        self,
        scf_backend,
        distance,
        expected_internal_stable,
        expected_external_stable,
        ref_internal,
        ref_external,
    ):
        """Test PySCF stability checker on N2 at different distances with RHF."""
        n2 = create_stretched_n2_structure(distance_angstrom=distance)
        scf_solver = self._create_scf_solver(backend=scf_backend)
        _, wavefunction = scf_solver.run(n2, 0, 1, "def2-svp")

        stability_checker = self._create_stability_checker()
        is_stable, result = stability_checker.run(wavefunction)

        self._assert_basic_stability_result(result, is_stable, has_internal=True, has_external=True)
        assert result.is_internal_stable() is expected_internal_stable
        assert result.is_external_stable() is expected_external_stable
        assert is_stable is (expected_internal_stable and expected_external_stable)

        # Check reference eigenvalues
        self._check_reference_eigenvalue(result, stability_checker, ref_internal, is_internal=True)
        self._check_reference_eigenvalue(result, stability_checker, ref_external, is_internal=False)

    @pytest.mark.parametrize("scf_backend", ["pyscf", "qdk"])
    def test_pyscf_stability_checker_bn_plus_uhf(self, scf_backend):
        """Test PySCF stability checker on BN+ cation (UHF) with different SCF backends."""
        structure = create_bn_plus_structure()
        ref_eigenvalue = -0.15872092489909065
        expected_negative_count = 2

        scf_solver = self._create_scf_solver(backend=scf_backend)
        _, wavefunction = scf_solver.run(structure, 1, 2, "def2-svp")

        # Test internal-only analysis (external not supported for UHF)
        stability_checker = self._create_stability_checker(internal=True, external=False)
        _, result = stability_checker.run(wavefunction)

        self._assert_basic_stability_result(result, False, has_internal=True, has_external=False)
        assert result.is_internal_stable() is False  # Should be internally unstable
        self._check_reference_eigenvalue(result, stability_checker, ref_eigenvalue, is_internal=True)

        # Check number of negative eigenvalues
        internal_eigenvalues = result.get_internal_eigenvalues()
        stability_tol = stability_checker.settings().get("stability_tolerance")
        num_negative = np.sum(internal_eigenvalues < stability_tol)
        assert num_negative == expected_negative_count, (
            f"Expected {expected_negative_count} negative eigenvalues, got {num_negative}"
        )

    def test_pyscf_stability_checker_c2_plus_rohf(self):
        """Test PySCF stability checker on C2+ cation (ROHF)."""
        structure = create_c2_plus_structure()
        ref_eigenvalue = -0.08256762551795531

        scf_solver = self._create_scf_solver(scf_type="restricted")
        _, wavefunction = scf_solver.run(structure, 1, 2, "def2-svp")

        # Test internal-only analysis (external not supported for ROHF)
        stability_checker = self._create_stability_checker(internal=True, external=False)
        _, result = stability_checker.run(wavefunction)

        self._assert_basic_stability_result(result, False, has_internal=True, has_external=False)
        assert result.is_internal_stable() is False  # Should be internally unstable
        self._check_reference_eigenvalue(result, stability_checker, ref_eigenvalue, is_internal=True)

        # Check number of negative eigenvalues
        internal_eigenvalues = result.get_internal_eigenvalues()
        stability_tol = stability_checker.settings().get("stability_tolerance")
        num_negative = np.sum(internal_eigenvalues < stability_tol)
        assert num_negative >= 1, f"Expected at least 1 negative eigenvalue, got {num_negative}"


@pytest.mark.skipif(not PYSCF_AVAILABLE, reason="PySCF not available")
class TestStabilityWorkflow:
    """Test class for stability workflow functionality."""

    @staticmethod
    def _run_scf_with_stability_workflow(
        structure,
        charge,
        spin_multiplicity,
        scf_solver,
        stability_checker,
        initial_guess=None,
        max_stability_iterations=5,
    ):
        """Run SCF with iterative stability checking and orbital rotation workflow.

        This is a Python-only test utility that implements the stability workflow.
        It performs iterative SCF, stability checks, and orbital rotations until convergence or max iterations.
        """
        if max_stability_iterations < 1:
            raise ValueError("max_stability_iterations must be at least 1")

        if initial_guess is None:
            # Run initial SCF calculation
            energy, wavefunction = scf_solver.run(structure, charge, spin_multiplicity, "def2-svp")
        else:
            # Run initial SCF calculation
            energy, wavefunction = scf_solver.run(structure, charge, spin_multiplicity, initial_guess)

        # Determine if calculation is restricted from initial wavefunction
        is_restricted_calculation = wavefunction.get_orbitals().is_restricted() and spin_multiplicity == 1

        # Configure stability checker based on calculation type
        if is_restricted_calculation:
            stability_checker.settings().set("external", True)
        else:
            stability_checker.settings().set("external", False)

        stability_result = None
        is_stable = False
        iteration = 0

        while iteration < max_stability_iterations:
            iteration += 1

            # Perform stability analysis
            is_stable, result = stability_checker.run(wavefunction)
            stability_result = result

            if is_stable:
                break

            # Last iteration check - don't rotate if we've reached the limit
            if iteration >= max_stability_iterations:
                break

            # Get the rotation vector corresponding to the smallest eigenvalue
            do_external = False
            if not stability_result.is_internal_stable():
                _, rotation_vector = stability_result.get_smallest_internal_eigenvalue_and_vector()
            elif not stability_result.is_external_stable() and stability_result.has_external_result():
                _, rotation_vector = stability_result.get_smallest_external_eigenvalue_and_vector()
                do_external = True
            else:
                raise RuntimeError(
                    "Stability analysis failed, but neither internal nor external instability was detected. "
                    "This is an unexpected state; rotation_vector is undefined."
                )

            # Get occupation numbers from wavefunction
            orbitals = wavefunction.get_orbitals()
            num_alpha_electrons, num_beta_electrons = wavefunction.get_total_num_electrons()

            # Rotate the orbitals
            rotated_orbitals = rotate_orbitals(
                orbitals, rotation_vector, num_alpha_electrons, num_beta_electrons, do_external
            )

            # If external instability, switch to unrestricted and disable external checks
            if do_external:
                # Create new solver instances with updated settings
                scf_solver_name = scf_solver.name()
                stability_checker_name = stability_checker.name()
                new_scf_solver = algorithms.create("scf_solver", scf_solver_name)
                new_stability_checker = algorithms.create("stability_checker", stability_checker_name)

                # Copy all settings from original solvers
                scf_settings_map = scf_solver.settings().to_dict()
                stability_settings_map = stability_checker.settings().to_dict()
                new_scf_solver.settings().from_dict(scf_settings_map)
                new_stability_checker.settings().from_dict(stability_settings_map)

                # Update specific settings for unrestricted calculation
                new_scf_solver.settings().set("scf_type", "unrestricted")
                new_stability_checker.settings().set("external", False)

                scf_solver = new_scf_solver
                stability_checker = new_stability_checker

            # Restart SCF with rotated orbitals
            energy, wavefunction = scf_solver.run(structure, charge, spin_multiplicity, rotated_orbitals)

        return energy, wavefunction, is_stable, stability_result

    def test_workflow_rohf_o2(self):
        """Test stability workflow on ROHF O2 molecule - internal stability only."""
        o2 = create_o2_structure()

        # Create and configure solvers
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")

        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("internal", True)

        # Run workflow with pyscf SCF (required for ROHF) and pyscf stability checker
        energy, _wfn, _is_stable, result = self._run_scf_with_stability_workflow(
            o2, 0, 3, scf_solver, stability_checker
        )

        # Check internal stability status
        assert result.is_internal_stable() is True, "Wavefunction should be internally stable"

        # Check energy matches reference value (internal-only stability with pyscf ROHF)
        # ROHF converges to different energy than UHF
        assert abs(energy - (-149.4705939454018)) < 1e-5, f"Energy {energy} should match reference -149.4705939454018"

    @pytest.mark.parametrize("scf_backend", ["pyscf", "qdk"])
    def test_workflow_n2_rhf_both_instability(self, scf_backend):
        """Test stability workflow on N2 at 1.6Å with both internal and external instabilities."""
        n2 = create_stretched_n2_structure(distance_angstrom=1.6)

        # Create and configure solvers
        scf_solver = algorithms.create("scf_solver", scf_backend)
        scf_solver.settings().set("scf_type", "auto")

        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("internal", True)
        stability_checker.settings().set("external", True)
        stability_checker.settings().set("stability_tolerance", -1e-4)
        stability_checker.settings().set("davidson_tolerance", 1e-4)
        stability_checker.settings().set("nroots", 3)

        # Run workflow with pyscf SCF and pyscf stability checker
        # This system may not fully converge but should achieve correct energy
        energy, _wfn, _is_stable, _result = self._run_scf_with_stability_workflow(
            n2, 0, 1, scf_solver, stability_checker
        )
        # Check energy matches reference value - workflow should achieve correct energy
        assert abs(energy - (-108.606721153932)) < 1e-6, f"Energy {energy} should match reference -108.606721153932"

    @pytest.mark.parametrize("scf_backend", ["pyscf", "qdk"])
    def test_workflow_n2_rhf_external_instability(self, scf_backend):
        """Test stability workflow on N2 at 1.2Å with external instability - should switch to UHF."""
        n2 = create_stretched_n2_structure(distance_angstrom=1.2)

        # Create and configure solvers
        scf_solver = algorithms.create("scf_solver", scf_backend)
        scf_solver.settings().set("scf_type", "auto")

        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("internal", True)
        stability_checker.settings().set("external", True)
        stability_checker.settings().set("stability_tolerance", -1e-4)
        stability_checker.settings().set("davidson_tolerance", 1e-4)
        stability_checker.settings().set("nroots", 3)

        # Run workflow - should detect external instability and switch to UHF
        energy, wfn, is_stable, result = self._run_scf_with_stability_workflow(n2, 0, 1, scf_solver, stability_checker)

        # Final wavefunction should be unrestricted (switched from RHF to UHF)
        assert not wfn.get_orbitals().is_restricted(), (
            "Final wavefunction should be unrestricted after external instability"
        )

        # Should be stable after resolving external instability
        assert is_stable is True, "Wavefunction should be stable after switching to UHF"
        assert result.is_internal_stable() is True, "Final wavefunction should be internally stable"

        # Check energy matches reference value - should converge to same UHF energy as manual rotation
        assert abs(energy - (-108.815746915896)) < 1e-6, f"Energy {energy} should match reference -108.815746915896"

    @pytest.mark.parametrize("scf_backend", ["pyscf", "qdk"])
    def test_workflow_n2_uhf_instability(self, scf_backend):
        """Test stability workflow on N2 at 1.4Å with internal instability of UHF."""
        n2 = create_stretched_n2_structure(distance_angstrom=1.4)

        # Create and configure solvers
        scf_solver = algorithms.create("scf_solver", scf_backend)
        scf_solver.settings().set("scf_type", "unrestricted")

        stability_checker = algorithms.create("stability_checker", "pyscf")
        stability_checker.settings().set("internal", True)
        stability_checker.settings().set("external", False)
        stability_checker.settings().set("stability_tolerance", -1e-4)
        stability_checker.settings().set("davidson_tolerance", 1e-4)
        stability_checker.settings().set("nroots", 3)

        # Run workflow - should detect internal instability of UHF
        # Suppress expected warning about unrestricted reference for closed-shell system
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrestricted reference requested for closed-shell system")
            energy, _wfn, _is_stable, result = self._run_scf_with_stability_workflow(
                n2, 0, 1, scf_solver, stability_checker, max_stability_iterations=10
            )
        assert result.is_internal_stable() is True, "Final wavefunction should be internally stable"

        # Check energy matches reference value - should converge to same UHF energy as manual rotation
        assert abs(energy - (-108.736487493576)) < 1e-6, f"Energy {energy} should match reference -108.815746915896"
