"""Comprehensive tests for the Wavefunction class in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import pickle

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.data import (
    CanonicalFourCenterHamiltonianContainer,
    CasWavefunctionContainer,
    Configuration,
    CoupledClusterContainer,
    Hamiltonian,
    MP2Container,
    Orbitals,
    SciWavefunctionContainer,
    SlaterDeterminantContainer,
    Structure,
    Wavefunction,
    WavefunctionType,
)

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
    rdm_tolerance,
    scf_orbital_tolerance,
)
from .test_helpers import create_test_basis_set


class TestWavefunctionType:
    """Test the WavefunctionType enum."""

    def test_wavefunction_type_values(self):
        """Test that WavefunctionType enum values are accessible."""
        wf_type = WavefunctionType

        # Test enum values exist
        assert hasattr(wf_type, "SelfDual")
        assert hasattr(wf_type, "NotSelfDual")


class TestWavefunction:
    """Test the main Wavefunction class."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
        basis_set = create_test_basis_set(3, "test-wavefunction")
        return Orbitals(coeffs, None, None, basis_set)

    @pytest.fixture
    def cas_wavefunction(self, basic_orbitals):
        """Create a SCI-based wavefunction for testing."""
        det1 = Configuration("20")
        det2 = Configuration("ud")
        dets = [det1, det2]
        coeffs = np.array([0.9, 0.436])  # Roughly normalized

        # dummy rdm
        rdm_aa = np.array([[1.0, 0.0], [0.0, 0.0]])
        rdm_bb = np.array([[1.0, 0.0], [0.0, 0.0]])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals, None, rdm_aa, rdm_bb)
        return Wavefunction(container)

    @pytest.fixture
    def slater_wavefunction(self, basic_orbitals):
        """Create a single Slater determinant wavefunction for testing."""
        det = Configuration("20")
        container = SlaterDeterminantContainer(det, basic_orbitals)
        return Wavefunction(container)

    def test_wavefunction_construction(self, cas_wavefunction):
        """Test Wavefunction construction with container."""
        wf = cas_wavefunction
        assert wf.size() == 2

    def test_wavefunction_orbitals_access(self, cas_wavefunction):
        """Test accessing orbitals from wavefunction."""
        wf = cas_wavefunction
        orbitals = wf.get_orbitals()
        assert orbitals is not None

        # Should be the same orbital object
        alpha_coeffs, _ = orbitals.get_coefficients()
        assert alpha_coeffs.shape == (3, 2)

    def test_wavefunction_electron_counts(self, cas_wavefunction):
        """Test getting electron counts."""
        wf = cas_wavefunction
        n_alpha, n_beta = wf.get_total_num_electrons()

        # But this depends on the implementation details
        assert isinstance(n_alpha, int)
        assert isinstance(n_beta, int)
        assert n_alpha >= 0
        assert n_beta >= 0

    def test_wavefunction_orbital_occupations(self, cas_wavefunction):
        """Test getting orbital occupations."""
        wf = cas_wavefunction
        alpha_occ, beta_occ = wf.get_total_orbital_occupations()

        assert len(alpha_occ) > 0
        assert len(beta_occ) > 0
        assert all(occ >= 0.0 for occ in alpha_occ)
        assert all(occ >= 0.0 for occ in beta_occ)

    def test_wavefunction_coefficient_access(self, cas_wavefunction):
        """Test accessing coefficients through wavefunction."""
        wf = cas_wavefunction

        det1 = Configuration("20")
        coeff1 = wf.get_coefficient(det1)
        assert np.isclose(
            coeff1, 0.9, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_wavefunction_get_coefficients(self, cas_wavefunction):
        wf = cas_wavefunction
        coeffs = wf.get_coefficients()
        assert np.allclose(
            coeffs, [0.9, 0.436], rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_wavefunction_determinants_access(self, cas_wavefunction):
        """Test accessing determinants from wavefunction."""
        wf = cas_wavefunction
        dets = wf.get_active_determinants()

        assert len(dets) == 2
        # Test that we can access the configurations
        for det in dets:
            assert isinstance(det, Configuration)

    def test_wavefunction_norm(self, cas_wavefunction, slater_wavefunction):
        """Test wavefunction norm calculation."""
        # SCI wavefunction
        sci_norm = cas_wavefunction.norm()
        expected_norm = np.sqrt(0.9**2 + 0.436**2)
        assert np.isclose(
            sci_norm, expected_norm, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

        # Slater determinant should have norm 1.0
        slater_norm = slater_wavefunction.norm()
        assert np.isclose(
            slater_norm, 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_wavefunction_overlap(self, basic_orbitals):
        """Test overlap calculation between wavefunctions."""
        # Create two similar wavefunctions
        det1 = Configuration("20")
        det2 = Configuration("ud")
        dets = [det1, det2]

        coeffs1 = np.array([0.9, 0.1])
        coeffs2 = np.array([0.8, 0.2])

        container1 = CasWavefunctionContainer(coeffs1, dets, basic_orbitals)
        container2 = CasWavefunctionContainer(coeffs2, dets, basic_orbitals)

        wf1 = Wavefunction(container1)
        wf2 = Wavefunction(container2)

        overlap = wf1.overlap(wf2)
        expected_overlap = 0.9 * 0.8 + 0.1 * 0.2
        assert np.isclose(
            overlap,
            expected_overlap,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_wavefunction_self_overlap(self, cas_wavefunction):
        """Test that self-overlap equals norm squared."""
        wf = cas_wavefunction

        self_overlap = wf.overlap(wf)
        norm_squared = wf.norm() ** 2

        assert np.isclose(
            self_overlap,
            norm_squared,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_wavefunction_rdm_availability_checks(self, cas_wavefunction):
        """Test RDM availability check methods."""
        wf = cas_wavefunction

        # These should not raise exceptions
        has_1rdm_spin_dep = wf.has_one_rdm_spin_dependent()
        has_1rdm_spin_traced = wf.has_one_rdm_spin_traced()
        has_2rdm_spin_dep = wf.has_two_rdm_spin_dependent()
        has_2rdm_spin_traced = wf.has_two_rdm_spin_traced()

        # Should return boolean values
        assert isinstance(has_1rdm_spin_dep, bool)
        assert isinstance(has_1rdm_spin_traced, bool)
        assert isinstance(has_2rdm_spin_dep, bool)
        assert isinstance(has_2rdm_spin_traced, bool)

    def test_wavefunction_type_access(self, cas_wavefunction):
        """Test accessing wavefunction type."""
        wf = cas_wavefunction
        wf_type = wf.get_type()
        assert wf_type == WavefunctionType.SelfDual

    def test_wavefunction_repr(self, cas_wavefunction):
        """Test string representation of wavefunction."""
        wf = cas_wavefunction
        repr_str = repr(wf)

        assert "qdk_chemistry.Wavefunction" in repr_str
        assert "size=2" in repr_str
        assert "norm=" in repr_str

    def test_repr_method(self, cas_wavefunction):
        """Test that __repr__ returns appropriate string representation."""
        repr_str = repr(cas_wavefunction)
        assert "qdk_chemistry.Wavefunction" in repr_str
        assert "size=" in repr_str
        assert "norm=" in repr_str
        assert str(cas_wavefunction.size()) in repr_str

    def test_str_method(self, cas_wavefunction):
        """Test that __str__ returns appropriate string representation."""
        str_str = str(cas_wavefunction)
        assert "qdk_chemistry.Wavefunction" in str_str
        assert "size=" in str_str
        assert "norm=" in str_str
        assert str(cas_wavefunction.size()) in str_str

    def test_get_summary(self, cas_wavefunction):
        """Test that get_summary() returns a proper summary string."""
        summary = cas_wavefunction.get_summary()
        assert isinstance(summary, str)

        # Verify it contains expected content based on C++ implementation
        assert "Wavefunction Summary:" in summary
        assert "Container type: cas" in summary
        assert "Number of determinants: 2" in summary
        assert "Wavefunction type: SelfDual" in summary
        assert "Complex: no" in summary
        assert "Norm:" in summary
        assert "Total electrons" in summary
        assert "Active electrons" in summary
        assert "1-RDM available:" in summary
        assert "2-RDM available:" in summary
        assert "Orbitals:" in summary

    def test_pickling_wavefunction(self, cas_wavefunction):
        """Test that Wavefunction can be pickled and unpickled correctly."""
        # Test pickling round-trip
        pickled_data = pickle.dumps(cas_wavefunction)
        wf_restored = pickle.loads(pickled_data)

        # Verify core properties
        assert wf_restored.size() == cas_wavefunction.size()
        assert np.isclose(
            wf_restored.norm(),
            cas_wavefunction.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert wf_restored.is_complex() == cas_wavefunction.is_complex()
        assert wf_restored.get_type() == cas_wavefunction.get_type()

        # Verify electron counts
        orig_electrons = cas_wavefunction.get_total_num_electrons()
        restored_electrons = wf_restored.get_total_num_electrons()
        assert orig_electrons == restored_electrons

        # Verify determinants
        orig_dets = cas_wavefunction.get_active_determinants()
        restored_dets = wf_restored.get_active_determinants()
        assert len(orig_dets) == len(restored_dets)

        # Verify coefficients for each determinant
        for det in orig_dets:
            orig_coeff = cas_wavefunction.get_coefficient(det)
            restored_coeff = wf_restored.get_coefficient(det)
            if isinstance(orig_coeff, complex):
                assert np.isclose(
                    orig_coeff,
                    restored_coeff,
                    rtol=float_comparison_relative_tolerance,
                    atol=float_comparison_absolute_tolerance,
                )
            else:
                assert np.isclose(
                    orig_coeff,
                    restored_coeff,
                    rtol=float_comparison_relative_tolerance,
                    atol=float_comparison_absolute_tolerance,
                )

        # Verify orbital consistency
        orig_orbs = cas_wavefunction.get_orbitals()
        restored_orbs = wf_restored.get_orbitals()
        assert orig_orbs.get_num_molecular_orbitals() == restored_orbs.get_num_molecular_orbitals()


class TestWavefunctionRDMs:
    """Test RDM-related functionality when available."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])  # Simple 2x2 identity
        basis_set = create_test_basis_set(2, "test-rdm")
        return Orbitals(coeffs, None, None, basis_set)

    @pytest.fixture
    def cas_wavefunction_with_rdms(self, basic_orbitals):
        """Create CAS wavefunction with RDMs for testing."""
        det = Configuration("20")  # Closed shell
        dets = [det]
        coeffs = np.array([1.0])

        # Simple 2x2 RDMs for 2 orbitals
        one_rdm_traced = np.array([[2.0, 0.0], [0.0, 0.0]])  # Doubly occupied first orbital
        one_rdm_aa = np.array([[1.0, 0.0], [0.0, 0.0]])
        one_rdm_bb = np.array([[1.0, 0.0], [0.0, 0.0]])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals, one_rdm_traced, one_rdm_aa, one_rdm_bb)
        return Wavefunction(container)

    def test_one_rdm_spin_traced_access(self, cas_wavefunction_with_rdms):
        """Test accessing spin-traced one-particle RDM."""
        wf = cas_wavefunction_with_rdms

        if wf.has_one_rdm_spin_traced():
            rdm = wf.get_active_one_rdm_spin_traced()
            assert rdm.shape == (2, 2)
            assert rdm[0, 0] == 2.0  # Doubly occupied
            assert rdm[1, 1] == 0.0  # Unoccupied

    def test_one_rdm_spin_dependent_access(self, cas_wavefunction_with_rdms):
        """Test accessing spin-dependent one-particle RDMs."""
        wf = cas_wavefunction_with_rdms

        if wf.has_one_rdm_spin_dependent():
            rdm_aa, rdm_bb = wf.get_active_one_rdm_spin_dependent()
            assert rdm_aa.shape == (2, 2)
            assert rdm_bb.shape == (2, 2)
            assert rdm_aa[0, 0] == 1.0  # Singly occupied alpha
            assert rdm_bb[0, 0] == 1.0  # Singly occupied beta

    def test_rdm_error_handling(self, basic_orbitals):
        """Test that appropriate errors are raised when RDMs are not available."""
        # Create simple SCI wavefunction without RDMs
        det = Configuration("20")
        dets = [det]
        coeffs = np.array([1.0])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        # These should raise RuntimeError if not available
        if not wf.has_one_rdm_spin_traced():
            with pytest.raises(RuntimeError):
                wf.get_active_one_rdm_spin_traced()

        if not wf.has_one_rdm_spin_dependent():
            with pytest.raises(RuntimeError):
                wf.get_active_one_rdm_spin_dependent()

        if not wf.has_two_rdm_spin_traced():
            with pytest.raises(RuntimeError):
                wf.get_active_two_rdm_spin_traced()

        if not wf.has_two_rdm_spin_dependent():
            with pytest.raises(RuntimeError):
                wf.get_active_two_rdm_spin_dependent()


class TestWavefunctionComplexSupport:
    """Test complex wavefunction support."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
        basis_set = create_test_basis_set(3, "test-complex")
        return Orbitals(coeffs, None, None, basis_set)

    def test_complex_wavefunction(self, basic_orbitals):
        """Test complex coefficient wavefunctions."""
        det1 = Configuration("20")
        det2 = Configuration("ud")
        dets = [det1, det2]

        # Complex coefficients
        coeffs = np.array([0.8 + 0.2j, 0.3 - 0.4j])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        # Test coefficient retrieval
        coeff1 = wf.get_coefficient(det1)
        assert np.isclose(
            coeff1, 0.8 + 0.2j, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

        # Test norm calculation with complex coefficients
        expected_norm = np.sqrt(abs(0.8 + 0.2j) ** 2 + abs(0.3 - 0.4j) ** 2)
        actual_norm = wf.norm()
        assert np.isclose(
            actual_norm,
            expected_norm,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_complex_overlap(self, basic_orbitals):
        """Test overlap with complex wavefunctions."""
        det1 = Configuration("20")
        det2 = Configuration("ud")
        dets = [det1, det2]

        coeffs1 = np.array([0.8 + 0.2j, 0.3 - 0.4j])
        coeffs2 = np.array([0.7 - 0.1j, 0.2 + 0.3j])

        container1 = CasWavefunctionContainer(coeffs1, dets, basic_orbitals)
        container2 = CasWavefunctionContainer(coeffs2, dets, basic_orbitals)

        wf1 = Wavefunction(container1)
        wf2 = Wavefunction(container2)

        overlap = wf1.overlap(wf2)
        # For complex wavefunctions: ⟨ψ₁|ψ₂⟩ = Σᵢ c₁ᵢ* c₂ᵢ
        expected = (0.8 - 0.2j) * (0.7 - 0.1j) + (0.3 + 0.4j) * (0.2 + 0.3j)
        assert np.isclose(
            overlap, expected, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )


class TestWavefunctionEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        basis_set = create_test_basis_set(2, "test-edge-cases")
        return Orbitals(coeffs, None, None, basis_set)

    def test_empty_wavefunction(self, basic_orbitals) -> None:
        """Test wavefunction with no determinants."""
        dets: list[Configuration] = []
        coeffs = np.array([])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        assert wf.size() == 0
        assert wf.norm() == 0.0

    def test_single_determinant_wavefunction(self, basic_orbitals):
        """Test wavefunction with single determinant."""
        det = Configuration("20")
        dets = [det]
        coeffs = np.array([0.7])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        assert wf.size() == 1
        assert np.isclose(
            wf.norm(), 0.7, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

        coeff = wf.get_coefficient(det)
        assert np.isclose(
            coeff, 0.7, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_zero_coefficient_handling(self, basic_orbitals):
        """Test handling of zero coefficients."""
        det1 = Configuration("20")
        det2 = Configuration("ud")
        dets = [det1, det2]
        coeffs = np.array([1.0, 0.0])  # Second coefficient is zero

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        assert wf.size() == 2  # Still contains both determinants

        coeff1 = wf.get_coefficient(det1)
        coeff2 = wf.get_coefficient(det2)

        assert np.isclose(
            coeff1, 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            coeff2, 0.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

        assert np.isclose(
            wf.norm(), 1.0, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_empty_wavefunction_bounds_checking(self, basic_orbitals) -> None:
        """Test that empty wavefunctions properly raise exceptions for determinant-dependent methods."""
        dets: list[Configuration] = []
        coeffs = np.array([])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        # These methods should raise RuntimeError for empty wavefunctions
        with pytest.raises(RuntimeError, match="No determinants available"):
            wf.get_total_num_electrons()

        with pytest.raises(RuntimeError, match="No determinants available"):
            wf.get_total_orbital_occupations()

        with pytest.raises(RuntimeError, match="No determinants available"):
            wf.get_active_orbital_occupations()

        with pytest.raises(RuntimeError, match="No determinants available"):
            wf.get_coefficient(Configuration("20"))

    def test_entropy_bounds_checking(self, basic_orbitals):
        """Test that entropy calculation fails gracefully when RDMs are missing."""
        det = Configuration("20")
        dets = [det]
        coeffs = np.array([1.0])

        # Create wavefunction without RDMs
        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        # Should raise RuntimeError when trying to calculate entropies without RDMs
        with pytest.raises(RuntimeError):
            wf.get_single_orbital_entropies()


class TestWavefunctionSerialization:
    """Test wavefunction serialization and deserialization."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0]])
        basis_set = create_test_basis_set(3, "test-wavefunction")
        return Orbitals(coeffs, None, None, basis_set)

    @pytest.fixture
    def cas_wavefunction_real(self, basic_orbitals):
        """Create a real CAS wavefunction for testing."""
        det1 = Configuration("20")
        det2 = Configuration("ud")
        dets = [det1, det2]
        coeffs = np.array([0.8, 0.6])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        return Wavefunction(container)

    @pytest.fixture
    def cas_wavefunction_complex(self, basic_orbitals):
        """Create a complex CAS wavefunction for testing."""
        det1 = Configuration("20")
        det2 = Configuration("ud")
        dets = [det1, det2]
        coeffs = np.array([0.8 + 0.1j, 0.6 - 0.2j])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        return Wavefunction(container)

    @pytest.fixture
    def sd_wavefunction(self, basic_orbitals):
        """Create a Slater determinant wavefunction for testing."""
        det = Configuration("20")
        container = SlaterDeterminantContainer(det, basic_orbitals)
        return Wavefunction(container)

    def test_json_serialization_cas_real(self, cas_wavefunction_real):
        """Test JSON serialization for real CAS wavefunction."""
        # Test to_json and from_json
        json_str = cas_wavefunction_real.to_json()

        # Parse JSON string to verify essential fields
        json_data = json.loads(json_str)
        assert "container_type" in json_data
        assert "container" in json_data
        assert json_data["container_type"] == "cas"

        # Test round-trip serialization
        wf_reconstructed = Wavefunction.from_json(json_str)

        # Verify properties are preserved
        assert wf_reconstructed.size() == cas_wavefunction_real.size()
        assert np.isclose(
            wf_reconstructed.norm(),
            cas_wavefunction_real.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert wf_reconstructed.get_type() == cas_wavefunction_real.get_type()
        assert wf_reconstructed.is_complex() == cas_wavefunction_real.is_complex()

        # Compare determinants
        orig_dets = cas_wavefunction_real.get_active_determinants()
        recon_dets = wf_reconstructed.get_active_determinants()
        assert len(recon_dets) == len(orig_dets)

        for orig_det, recon_det in zip(orig_dets, recon_dets, strict=True):
            assert orig_det.to_string() == recon_det.to_string()

    def test_json_serialization_cas_complex(self, cas_wavefunction_complex):
        """Test JSON serialization for complex CAS wavefunction."""
        json_str = cas_wavefunction_complex.to_json()
        json_data = json.loads(json_str)

        # Verify essential fields
        assert "container_type" in json_data
        assert "container" in json_data
        assert json_data["container_type"] == "cas"

        # Test round-trip serialization
        wf_reconstructed = Wavefunction.from_json(json_str)

        # Verify properties are preserved
        assert wf_reconstructed.size() == cas_wavefunction_complex.size()
        assert np.isclose(
            wf_reconstructed.norm(),
            cas_wavefunction_complex.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert wf_reconstructed.is_complex() == cas_wavefunction_complex.is_complex()

    def test_json_serialization_slater_determinant(self, sd_wavefunction):
        """Test JSON serialization for Slater determinant wavefunction."""
        json_str = sd_wavefunction.to_json()
        json_data = json.loads(json_str)

        # Verify essential fields
        assert "container_type" in json_data
        assert "container" in json_data
        assert json_data["container_type"] == "sd"

        # Test round-trip serialization
        wf_reconstructed = Wavefunction.from_json(json_str)

        # Verify properties are preserved
        assert wf_reconstructed.size() == 1
        assert np.isclose(
            wf_reconstructed.norm(),
            1.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert not wf_reconstructed.is_complex()

    def test_hdf5_serialization_cas_real(self, cas_wavefunction_real, tmp_path):
        """Test HDF5 serialization for real CAS wavefunction."""
        filename = tmp_path / "test_wavefunction_cas_real.wavefunction.h5"

        # Save to HDF5 file
        cas_wavefunction_real.to_hdf5_file(str(filename))

        # Load from HDF5 file
        wf_reconstructed = Wavefunction.from_hdf5_file(str(filename))

        # Verify properties are preserved
        assert wf_reconstructed.size() == cas_wavefunction_real.size()
        assert np.isclose(
            wf_reconstructed.norm(),
            cas_wavefunction_real.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert wf_reconstructed.get_type() == cas_wavefunction_real.get_type()

        # Compare determinants
        orig_dets = cas_wavefunction_real.get_active_determinants()
        recon_dets = wf_reconstructed.get_active_determinants()
        assert len(recon_dets) == len(orig_dets)

    def test_hdf5_serialization_cas_complex(self, cas_wavefunction_complex, tmp_path):
        """Test HDF5 serialization for complex CAS wavefunction."""
        filename = tmp_path / "test_wavefunction_cas_complex.wavefunction.h5"

        # Save to HDF5 file
        cas_wavefunction_complex.to_hdf5_file(str(filename))

        # Load from HDF5 file
        wf_reconstructed = Wavefunction.from_hdf5_file(str(filename))

        # Verify properties are preserved
        assert wf_reconstructed.size() == cas_wavefunction_complex.size()
        assert np.isclose(
            wf_reconstructed.norm(),
            cas_wavefunction_complex.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert wf_reconstructed.is_complex()

    def test_hdf5_serialization_slater_determinant(self, sd_wavefunction, tmp_path):
        """Test HDF5 serialization for Slater determinant wavefunction."""
        filename = tmp_path / "test_wavefunction_sd.wavefunction.h5"

        # Save to HDF5 file
        sd_wavefunction.to_hdf5_file(str(filename))

        # Load from HDF5 file
        wf_reconstructed = Wavefunction.from_hdf5_file(str(filename))

        # Verify properties are preserved
        assert wf_reconstructed.size() == 1
        assert np.isclose(
            wf_reconstructed.norm(),
            1.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert not wf_reconstructed.is_complex()

    def test_json_file_io(self, cas_wavefunction_real, tmp_path):
        """Test JSON file I/O."""
        filename = tmp_path / "test_wavefunction.wavefunction.json"

        # Save to JSON file
        cas_wavefunction_real.to_json_file(str(filename))

        # Load from JSON file
        wf_reconstructed = Wavefunction.from_json_file(str(filename))

        # Verify properties are preserved
        assert wf_reconstructed.size() == cas_wavefunction_real.size()
        assert np.isclose(
            wf_reconstructed.norm(),
            cas_wavefunction_real.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_generic_file_io(self, cas_wavefunction_real, tmp_path):
        """Test generic file I/O with different formats."""
        json_filename = tmp_path / "test_wavefunction_generic.wavefunction.json"
        hdf5_filename = tmp_path / "test_wavefunction_generic.wavefunction.h5"

        # Test JSON format
        cas_wavefunction_real.to_file(str(json_filename), "json")
        wf_json = Wavefunction.from_file(str(json_filename), "json")
        assert np.isclose(
            wf_json.norm(),
            cas_wavefunction_real.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Test HDF5 format
        cas_wavefunction_real.to_file(str(hdf5_filename), "hdf5")
        wf_hdf5 = Wavefunction.from_file(str(hdf5_filename), "hdf5")
        assert np.isclose(
            wf_hdf5.norm(),
            cas_wavefunction_real.norm(),
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Test invalid format
        with pytest.raises(ValueError, match="Unsupported file type"):
            cas_wavefunction_real.to_file(str(tmp_path / "test.xyz"), "xyz")

        with pytest.raises(ValueError, match="Unsupported file type"):
            Wavefunction.from_file(str(tmp_path / "test.xyz"), "xyz")

    def test_error_handling(self):
        """Test error handling for malformed data."""
        # Test malformed JSON
        bad_json = {"container_type": "unknown"}

        with pytest.raises(TypeError):
            Wavefunction.from_json(bad_json)

        # Test non-existent files
        with pytest.raises(RuntimeError):
            Wavefunction.from_json_file("non_existent.wavefunction.json")

        with pytest.raises(RuntimeError):
            Wavefunction.from_hdf5_file("non_existent.wavefunction.h5")


class TestWavefunctionRdmIntegraion:
    """Test integration of RDMs within the Wavefunction class."""

    def test_rdm_n2_singlet_6_6(self):
        """Test RDM properties for N2 singlet 6e 6o wavefunction."""
        nelec_alpha = 3
        nelec_beta = 3
        norb = 6
        ntot = nelec_alpha + nelec_beta

        mol = Structure(["N", "N"], [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        scf_solver = algorithms.create("scf_solver")
        sd_wf = scf_solver.run(mol, 0, np.abs(nelec_alpha - nelec_beta) + 1, "def2-svp")[1]

        active_space_selector = algorithms.create("active_space_selector", "qdk_valence")
        active_space_selector.settings().set("num_active_electrons", ntot)
        active_space_selector.settings().set("num_active_orbitals", norb)
        active_orbs_sd = active_space_selector.run(sd_wf)

        hamil_ctor = algorithms.create("hamiltonian_constructor")
        hamiltonian = hamil_ctor.run(active_orbs_sd.get_orbitals())

        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)
        wfn = macis_calc.run(hamiltonian, nelec_alpha, nelec_beta)[1]

        one_rdm = wfn.get_active_one_rdm_spin_traced()
        two_rdm = wfn.get_active_two_rdm_spin_traced()
        one_rdm_aa, one_rdm_bb = wfn.get_active_one_rdm_spin_dependent()
        two_rdm_ab, two_rdm_aa, two_rdm_bb = wfn.get_active_two_rdm_spin_dependent()

        two_rdm = np.reshape(two_rdm, (norb, norb, norb, norb))
        two_rdm_aa = np.reshape(two_rdm_aa, (norb, norb, norb, norb))
        two_rdm_bb = np.reshape(two_rdm_bb, (norb, norb, norb, norb))
        two_rdm_ab = np.reshape(two_rdm_ab, (norb, norb, norb, norb))
        two_rdm_ba = np.einsum("pqrs->rspq", two_rdm_ab)

        assert np.allclose(
            one_rdm, one_rdm_aa + one_rdm_bb, rtol=float_comparison_relative_tolerance, atol=rdm_tolerance
        )
        assert np.allclose(
            two_rdm,
            two_rdm_aa + two_rdm_bb + two_rdm_ab + two_rdm_ba,
            rtol=float_comparison_relative_tolerance,
            atol=rdm_tolerance,
        )
        assert np.allclose(
            one_rdm,
            np.einsum("pqrr->pq", two_rdm) / (ntot - 1),
            rtol=float_comparison_relative_tolerance,
            atol=rdm_tolerance,
        )
        assert np.allclose(
            one_rdm_aa,
            np.einsum("pqrr->pq", two_rdm_ab) / (nelec_beta),
            rtol=float_comparison_relative_tolerance,
            atol=rdm_tolerance,
        )
        assert np.allclose(
            one_rdm_bb,
            np.einsum("rrpq->pq", two_rdm_ab) / (nelec_alpha),
            rtol=float_comparison_relative_tolerance,
            atol=rdm_tolerance,
        )

        s1_entropy = wfn.get_single_orbital_entropies()
        assert np.allclose(
            s1_entropy,
            np.array([0.02040482, 0.17135976, 0.17135976, 0.17651599, 0.17651599, 0.00478807]),
            rtol=float_comparison_relative_tolerance,
            atol=scf_orbital_tolerance,
        )

    def test_rdm_o2_triplet_6_6(self):
        """Test RDM retrieval for O2 triplet 6e 6o wavefunction."""
        try:
            import pyscf  # noqa: PLC0415, F401
        except ImportError:
            pytest.skip("pyscf not available, skipping O2 triplet RDM test")
        import qdk_chemistry.plugins.pyscf as pyscf_plugin  # noqa: PLC0415

        pyscf_plugin.load()

        nelec_alpha = 5
        nelec_beta = 3
        norb = 6
        ntot = nelec_alpha + nelec_beta

        mol = Structure(["O", "O"], [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
        scf_solver = algorithms.create("scf_solver", "pyscf")
        scf_solver.settings().set("scf_type", "restricted")
        sd_wf = scf_solver.run(mol, 0, np.abs(nelec_alpha - nelec_beta) + 1, "def2-svp")[1]

        active_space_selector = algorithms.create("active_space_selector", "qdk_valence")
        active_space_selector.settings().set("num_active_electrons", ntot)
        active_space_selector.settings().set("num_active_orbitals", norb)
        active_orbs_sd = active_space_selector.run(sd_wf)

        hamil_ctor = algorithms.create("hamiltonian_constructor")
        hamiltonian = hamil_ctor.run(active_orbs_sd.get_orbitals())

        macis_calc = algorithms.create("multi_configuration_calculator", "macis_cas")
        macis_calc.settings().set("calculate_one_rdm", True)
        macis_calc.settings().set("calculate_two_rdm", True)
        wfn = macis_calc.run(hamiltonian, nelec_alpha, nelec_beta)[1]

        one_rdm = wfn.get_active_one_rdm_spin_traced()
        two_rdm = wfn.get_active_two_rdm_spin_traced()
        one_rdm_aa, one_rdm_bb = wfn.get_active_one_rdm_spin_dependent()
        two_rdm_ab, two_rdm_aa, two_rdm_bb = wfn.get_active_two_rdm_spin_dependent()

        two_rdm = np.reshape(two_rdm, (norb, norb, norb, norb))
        two_rdm_aa = np.reshape(two_rdm_aa, (norb, norb, norb, norb))
        two_rdm_bb = np.reshape(two_rdm_bb, (norb, norb, norb, norb))
        two_rdm_ab = np.reshape(two_rdm_ab, (norb, norb, norb, norb))
        two_rdm_ba = np.einsum("pqrs->rspq", two_rdm_ab)

        assert np.allclose(
            one_rdm, one_rdm_aa + one_rdm_bb, rtol=float_comparison_relative_tolerance, atol=rdm_tolerance
        )
        assert np.allclose(
            two_rdm,
            two_rdm_aa + two_rdm_bb + two_rdm_ab + two_rdm_ba,
            rtol=float_comparison_relative_tolerance,
            atol=rdm_tolerance,
        )
        assert np.allclose(
            one_rdm,
            np.einsum("pqrr->pq", two_rdm) / (ntot - 1),
            rtol=float_comparison_relative_tolerance,
            atol=rdm_tolerance,
        )
        assert np.allclose(
            one_rdm_aa,
            np.einsum("pqrr->pq", two_rdm_ab) / (nelec_beta),
            rtol=float_comparison_relative_tolerance,
            atol=rdm_tolerance,
        )
        assert np.allclose(
            one_rdm_bb,
            np.einsum("rrpq->pq", two_rdm_ab) / (nelec_alpha),
            rtol=float_comparison_relative_tolerance,
            atol=rdm_tolerance,
        )

        s1_entropy = wfn.get_single_orbital_entropies()
        assert np.allclose(
            s1_entropy,
            np.array([0.11641596, 0.11641596, 0.03304609, 0.11492634, 0.11492634, 0.03494613]),
            rtol=float_comparison_relative_tolerance,
            atol=scf_orbital_tolerance,
        )

    def test_sci_hdf5_roundtrip(self, tmp_path):
        symbols = ["C", "C", "H", "H", "H", "H"]
        coords = np.array(
            [
                [0.0, 0.0, 0.6695],
                [0.0, 0.0, -0.6695],
                [0.0, 0.9289, 1.2321],
                [0.0, -0.9289, 1.2321],
                [0.0, 0.9289, -1.2321],
                [0.0, -0.9289, -1.2321],
            ]
        )

        structure = Structure(coords, symbols)

        # Run SCF
        scf = algorithms.create("scf_solver")
        _, wfn_scf = scf.run(structure, 0, 1, "sto-3g")

        # Build Hamiltonian (12e/12o active space)
        as_selector = algorithms.create("active_space_selector", "qdk_valence")
        as_selector.settings().set("num_active_electrons", 12)
        as_selector.settings().set("num_active_orbitals", 12)
        orbitals = as_selector.run(wfn_scf).get_orbitals()

        ham_constructor = algorithms.create("hamiltonian_constructor")
        hamiltonian = ham_constructor.run(orbitals)

        # Run ASCI with RDM calculation (minimal settings)
        mc = algorithms.create("multi_configuration_calculator", "macis_asci")
        mc.settings().set("calculate_one_rdm", True)
        mc.settings().set("calculate_two_rdm", True)
        mc.settings().set("ntdets_max", 10)
        mc.settings().set("ntdets_min", 1)
        mc.settings().set("grow_factor", 2)
        mc.settings().set("max_refine_iter", 0)
        _, wfn_sci = mc.run(hamiltonian, 6, 6)

        assert wfn_sci.has_one_rdm_spin_dependent(), "one rdm is not available"

        # Verify RDMs exist before save
        original_1rdm_aa, _ = wfn_sci.get_active_one_rdm_spin_dependent()
        original_2rdm_aabb, original_2rdm_aaaa, _ = wfn_sci.get_active_two_rdm_spin_dependent()

        # Save and reload using tmp_path
        h5_file = tmp_path / "test_wavefunction.wavefunction.h5"
        wfn_sci.to_hdf5_file(str(h5_file))
        wfn_loaded = Wavefunction.from_hdf5_file(str(h5_file))

        # Check RDMs after load
        assert wfn_loaded.has_one_rdm_spin_dependent(), "one rdm is not available"
        after_1rdm_aa, _ = wfn_loaded.get_active_one_rdm_spin_dependent()

        assert wfn_loaded.has_two_rdm_spin_dependent(), "two rdm is not available"
        after_2rdm_aabb, after_2rdm_aaaa, _ = wfn_loaded.get_active_two_rdm_spin_dependent()

        assert np.allclose(original_1rdm_aa, after_1rdm_aa, atol=rdm_tolerance)
        assert np.allclose(original_2rdm_aaaa, after_2rdm_aaaa, atol=rdm_tolerance)
        assert np.allclose(original_2rdm_aabb, after_2rdm_aabb, atol=rdm_tolerance)

    def test_cas_hdf5_roundtrip(self, tmp_path):
        symbols = ["H", "H"]
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
            ]
        )

        structure = Structure(coords, symbols)

        # Run SCF
        scf = algorithms.create("scf_solver")
        _, wfn_scf = scf.run(structure, 0, 1, "def2-svp")

        # Build Hamiltonian
        ham_constructor = algorithms.create("hamiltonian_constructor")
        hamiltonian = ham_constructor.run(wfn_scf.get_orbitals())

        # Run CAS with RDM calculation
        mc = algorithms.create("multi_configuration_calculator", "macis_cas")
        mc.settings().set("calculate_one_rdm", True)
        mc.settings().set("calculate_two_rdm", True)
        _, wfn_cas = mc.run(hamiltonian, 2, 2)

        assert wfn_cas.has_one_rdm_spin_dependent(), "one rdm is not available"

        # Verify RDMs exist before save
        original_1rdm_aa, _ = wfn_cas.get_active_one_rdm_spin_dependent()
        original_2rdm_aabb, original_2rdm_aaaa, _ = wfn_cas.get_active_two_rdm_spin_dependent()

        # Save and reload using tmp_path
        h5_file = tmp_path / "test_wavefunction.wavefunction.h5"
        wfn_cas.to_hdf5_file(str(h5_file))
        wfn_loaded = Wavefunction.from_hdf5_file(str(h5_file))

        # Check RDMs after load
        assert wfn_loaded.has_one_rdm_spin_dependent(), "one rdm is not available"
        after_1rdm_aa, _ = wfn_loaded.get_active_one_rdm_spin_dependent()

        assert wfn_loaded.has_two_rdm_spin_dependent(), "two rdm is not available"
        after_2rdm_aabb, after_2rdm_aaaa, _ = wfn_loaded.get_active_two_rdm_spin_dependent()

        assert np.allclose(original_1rdm_aa, after_1rdm_aa, atol=rdm_tolerance)
        assert np.allclose(original_2rdm_aaaa, after_2rdm_aaaa, atol=rdm_tolerance)
        assert np.allclose(original_2rdm_aabb, after_2rdm_aabb, atol=rdm_tolerance)

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        energies = np.array([-1.0, -0.5, 0.5])  # Two occupied, one virtual
        basis_set = create_test_basis_set(3, "test-mp2")
        return Orbitals(coeffs, energies, None, basis_set)

    @pytest.fixture
    def basic_hamiltonian(self, basic_orbitals):
        """Create a basic Hamiltonian for testing."""
        # Create simple 1e and 2e integrals for 3 orbitals
        h1e = np.array([[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.5]])
        h2e = np.zeros((3, 3, 3, 3))
        # Add some simple repulsion integrals
        h2e[0, 0, 0, 0] = 0.5
        h2e[1, 1, 1, 1] = 0.3
        h2e[0, 1, 0, 1] = 0.2
        h2e[1, 0, 1, 0] = 0.2

        core_energy = 0.0
        inactive_fock = np.eye(0)  # Empty inactive Fock matrix
        return Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(h1e, h2e.flatten(), basic_orbitals, core_energy, inactive_fock)
        )


class TestMP2Container:
    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        energies = np.array([-1.0, -0.5, 0.5])  # Two occupied, one virtual
        basis_set = create_test_basis_set(3, "test-mp2")
        return Orbitals(coeffs, energies, None, basis_set)

    @pytest.fixture
    def basic_hamiltonian(self, basic_orbitals):
        """Create a basic Hamiltonian for testing."""
        # Create simple 1e and 2e integrals for 3 orbitals
        h1e = np.array([[-1.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.5]])
        h2e = np.zeros((3, 3, 3, 3))
        # Add some simple repulsion integrals
        h2e[0, 0, 0, 0] = 0.5
        h2e[1, 1, 1, 1] = 0.3
        h2e[0, 1, 0, 1] = 0.2
        h2e[1, 0, 1, 0] = 0.2

        core_energy = 0.0
        inactive_fock = np.eye(0)  # Empty inactive Fock matrix
        return Hamiltonian(
            CanonicalFourCenterHamiltonianContainer(h1e, h2e.flatten(), basic_orbitals, core_energy, inactive_fock)
        )

    @pytest.fixture
    def reference_wavefunction(self, basic_orbitals):
        """Create a reference wavefunction for MP2/CC tests."""
        ref = Configuration("220")  # Two electrons in first two orbitals
        sd_container = SlaterDeterminantContainer(ref, basic_orbitals)
        return Wavefunction(sd_container)

    def test_mp2_container_construction(self, basic_hamiltonian, reference_wavefunction):
        """Test MP2Container construction with lazy evaluation."""
        mp2_container = MP2Container(basic_hamiltonian, reference_wavefunction)

        assert mp2_container is not None

        # Amplitudes should not be computed initially
        assert not mp2_container.has_t1_amplitudes(), (
            "T1 amplitudes should NOT be computed until requested (lazy evaluation)"
        )
        assert not mp2_container.has_t2_amplitudes(), (
            "T2 amplitudes should NOT be computed until requested (lazy evaluation)"
        )

        # Trigger computations
        t1_aa, t1_bb = mp2_container.get_t1_amplitudes()
        t2_abab, t2_aaaa, t2_bbbb = mp2_container.get_t2_amplitudes()

        # amplitudes should now be available
        assert mp2_container.has_t1_amplitudes(), "T1 amplitudes should be cached after first access"
        assert mp2_container.has_t2_amplitudes(), "T2 amplitudes should be cached after first access"

        # Verify T1 amplitudes are zero for MP2
        assert np.allclose(t1_aa, 0.0), "T1 alpha amplitudes should be zero for MP2"
        assert np.allclose(t1_bb, 0.0), "T1 beta amplitudes should be zero for MP2"

        # Verify T2 amplitudes exist (non-zero)
        assert t2_abab is not None
        assert t2_aaaa is not None
        assert t2_bbbb is not None


class TestCCContainer:
    """Test the CoupledClusterContainer wavefunction container."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        energies = np.array([-1.0, -0.5, 0.5])  # Two occupied, one virtual
        basis_set = create_test_basis_set(3, "test-cc")
        return Orbitals(coeffs, energies, None, basis_set)

    @pytest.fixture
    def reference_wavefunction(self, basic_orbitals):
        """Create a reference wavefunction for CC tests."""
        ref = Configuration("220")  # Two electrons in first two orbitals
        sd_container = SlaterDeterminantContainer(ref, basic_orbitals)
        return Wavefunction(sd_container)

    def test_cc_container_construction(self, basic_orbitals, reference_wavefunction):
        """Test CoupledClusterContainer construction."""
        # Create dummy amplitudes for 2 occupied, 1 virtual orbital
        # T1: nocc * nvir = 2 * 1 = 2
        t1 = np.array([0.01, 0.02])
        # T2: nocc * nocc * nvir * nvir = 2 * 2 * 1 * 1 = 4
        t2 = np.array([0.001, 0.002, 0.003, 0.004])

        # Enable amplitude storage
        cc_container = CoupledClusterContainer(basic_orbitals, reference_wavefunction, t1, t2)

        assert cc_container is not None
        assert cc_container.has_t1_amplitudes()
        assert cc_container.has_t2_amplitudes()

    def test_cc_container_in_wavefunction(self, basic_orbitals, reference_wavefunction):
        """Test CoupledClusterContainer within a Wavefunction wrapper."""
        # Create dummy amplitudes
        # T1: nocc * nvir = 2 * 1 = 2
        t1 = np.array([0.01, 0.02])
        # T2: nocc * nocc * nvir * nvir = 2 * 2 * 1 * 1 = 4
        t2 = np.array([0.001, 0.002, 0.003, 0.004])

        cc_container = CoupledClusterContainer(basic_orbitals, reference_wavefunction, t1, t2)
        wf = Wavefunction(cc_container)

        # Test container type checking
        assert wf.get_container_type() == "coupled_cluster"

        # Test getting the container back
        retrieved_container = wf.get_container()
        assert retrieved_container is not None
        assert retrieved_container.has_t1_amplitudes()
        assert retrieved_container.has_t2_amplitudes()

    def test_cc_container_electron_counts(self, basic_orbitals, reference_wavefunction):
        """Test getting electron counts from CoupledClusterContainer."""
        cc_container = CoupledClusterContainer(basic_orbitals, reference_wavefunction)
        wf = Wavefunction(cc_container)

        n_alpha, n_beta = wf.get_active_num_electrons()
        assert n_alpha == 2
        assert n_beta == 2

        n_alpha_total, n_beta_total = wf.get_total_num_electrons()
        assert n_alpha_total == 2
        assert n_beta_total == 2


def test_wavefunction_data_type_name():
    """Test that Wavefunction has the correct _data_type_name class attribute."""
    assert hasattr(Wavefunction, "_data_type_name")
    assert Wavefunction._data_type_name == "wavefunction"


class TestWavefunctionTruncate:
    """Test the Wavefunction.truncate method."""

    @pytest.fixture
    def sci_wavefunction(self):
        """Create a SCI wavefunction with multiple determinants for testing."""
        coeffs = np.array([[0.9, 0.1], [0.1, -0.9], [0.0, 0.0], [0.0, 0.0]])
        basis_set = create_test_basis_set(4, "test-truncate")
        orbitals = Orbitals(coeffs, None, None, basis_set)

        dets = [
            Configuration("2200"),  # largest coeff
            Configuration("2020"),  # second largest
            Configuration("2002"),  # third largest
            Configuration("0220"),  # smallest
        ]
        coeffs = np.array([0.8, 0.4, 0.3, 0.1])  # Not normalized

        container = SciWavefunctionContainer(coeffs, dets, orbitals)
        return Wavefunction(container)

    def test_truncate_to_n_determinants(self, sci_wavefunction):
        """Test truncation to specific number of determinants."""
        truncated = sci_wavefunction.truncate(max_determinants=2)

        # Should have 2 determinants
        assert truncated.size() == 2

        # Should be normalized
        assert truncated.norm() == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)

    def test_truncate_keeps_top_determinants(self, sci_wavefunction):
        """Test that truncation keeps the top determinants by coefficient magnitude."""
        truncated = sci_wavefunction.truncate(max_determinants=2)

        # Get the determinants
        dets = truncated.get_active_determinants()
        assert len(dets) == 2
        assert str(dets[0]) == "2200"  # largest
        assert str(dets[1]) == "2020"  # second largest

    def test_truncate_renormalizes_coefficients(self, sci_wavefunction):
        """Test that truncated coefficients are properly renormalized."""
        truncated = sci_wavefunction.truncate(max_determinants=2)

        # Original top 2 coefficients: 0.8, 0.4
        # Norm of [0.8, 0.4] = sqrt(0.64 + 0.16) = sqrt(0.80)
        expected_norm = np.sqrt(0.8**2 + 0.4**2)
        coeffs = truncated.get_coefficients()

        assert coeffs[0] == pytest.approx(0.8 / expected_norm, abs=float_comparison_absolute_tolerance)
        assert coeffs[1] == pytest.approx(0.4 / expected_norm, abs=float_comparison_absolute_tolerance)

    def test_truncate_with_none_returns_all(self, sci_wavefunction):
        """Test truncation with None returns all determinants (renormalized)."""
        truncated = sci_wavefunction.truncate(max_determinants=None)

        assert truncated.size() == 4
        assert truncated.norm() == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)

    def test_truncate_more_than_exist(self, sci_wavefunction):
        """Test truncation requesting more determinants than exist."""
        truncated = sci_wavefunction.truncate(max_determinants=10)

        assert truncated.size() == 4
        assert truncated.norm() == pytest.approx(1.0, abs=float_comparison_absolute_tolerance)
