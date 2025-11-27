"""Tests for QDK Chemistry to Qiskit conversion utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import CasWavefunctionContainer, Configuration, Orbitals, Wavefunction
from qdk_chemistry.plugins.qiskit.conversion import (
    _configuration_to_statevector_index,
    create_statevector_from_wavefunction,
)
from tests.conftest import create_test_basis_set

from .reference_tolerances import (
    float_comparison_absolute_tolerance,
    float_comparison_relative_tolerance,
)


class TestConfigurationToStatevectorIndex:
    """Test the _configuration_to_statevector_index helper function."""

    def test_basic_index_calculation(self):
        """Test the canonical example from C++ documentation.

        Configuration "2ud0" with 4 orbitals:
        - Orbital 0: doubly occupied (α=1, β=1)
        - Orbital 1: alpha electron (α=1, β=0)
        - Orbital 2: beta electron (α=0, β=1)
        - Orbital 3: empty (α=0, β=0)

        Qubit layout:
        Qubits: 7 6 5 4 | 3 2 1 0
                β-orbs  | α-orbs
                3 2 1 0 | 3 2 1 0
                0 1 0 1 | 0 0 1 1
        Binary: 01010011 = 64 + 16 + 2 + 1 = 83
        """
        config = Configuration("2ud0")
        index = _configuration_to_statevector_index(config, 4)
        assert index == 83

    def test_empty_configuration(self):
        """Test that empty configuration maps to index 0."""
        config = Configuration("0000")
        index = _configuration_to_statevector_index(config, 4)
        assert index == 0

    def test_all_doubly_occupied(self):
        """Test configuration with all orbitals doubly occupied.

        All alpha bits set (0-3) and all beta bits set (4-7)
        Binary: 11111111 = 255
        """
        config = Configuration("2222")
        index = _configuration_to_statevector_index(config, 4)
        assert index == 255

    def test_all_alpha_electrons(self):
        """Test configuration with only alpha electrons.

        Alpha bits set (0-3), beta bits clear
        Binary: 00001111 = 15
        """
        config = Configuration("uuuu")
        index = _configuration_to_statevector_index(config, 4)
        assert index == 15

    def test_all_beta_electrons(self):
        """Test configuration with only beta electrons.

        Alpha bits clear, beta bits set (4-7)
        Binary: 11110000 = 240
        """
        config = Configuration("dddd")
        index = _configuration_to_statevector_index(config, 4)
        assert index == 240

    def test_partial_orbital_usage(self):
        """Test using only a subset of orbitals from configuration.

        Configuration "2ud000" using first 3 orbitals:
        - Orbital 0: doubly (α=1, β=1)
        - Orbital 1: alpha (α=1, β=0)
        - Orbital 2: beta (α=0, β=1)

        Qubits: 5 4 3 | 2 1 0
                2 1 0 | 2 1 0
                1 0 1 | 0 1 1
        Binary: 101011 = 32 + 8 + 2 + 1 = 43
        """
        config = Configuration("2ud000")
        index = _configuration_to_statevector_index(config, 3)
        assert index == 43

    def test_single_orbital_empty(self):
        """Test single empty orbital."""
        config = Configuration("0")
        index = _configuration_to_statevector_index(config, 1)
        assert index == 0

    def test_single_orbital_alpha(self):
        """Test single orbital with alpha electron.

        Bit 0 set, bit 1 clear: Binary 01 = 1
        """
        config = Configuration("u")
        index = _configuration_to_statevector_index(config, 1)
        assert index == 1

    def test_single_orbital_beta(self):
        """Test single orbital with beta electron.

        Bit 0 clear, bit 1 set: Binary 10 = 2
        """
        config = Configuration("d")
        index = _configuration_to_statevector_index(config, 1)
        assert index == 2

    def test_single_orbital_doubly(self):
        """Test single orbital doubly occupied.

        Bits 0 and 1 set: Binary 11 = 3
        """
        config = Configuration("2")
        index = _configuration_to_statevector_index(config, 1)
        assert index == 3

    def test_little_endian_ordering(self):
        """Verify little-endian ordering with specific pattern.

        "u0d0" means orbital 0 has alpha, orbital 2 has beta
        Qubits: 7 6 5 4 | 3 2 1 0
                3 2 1 0 | 3 2 1 0
                0 1 0 0 | 0 0 0 1
        Binary: 01000001 = 64 + 1 = 65
        """
        config = Configuration("u0d0")
        index = _configuration_to_statevector_index(config, 4)
        assert index == 65

    def test_error_on_too_many_orbitals(self):
        """Test that requesting more orbitals than available raises error."""
        config = Configuration("ud")
        with pytest.raises(RuntimeError):
            _configuration_to_statevector_index(config, 10)


class TestCreateStatevectorFromWavefunction:
    """Test the create_statevector_from_wavefunction function."""

    @pytest.fixture
    def basic_orbitals(self):
        """Create basic orbitals for testing."""
        coeffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        basis_set = create_test_basis_set(2, "test-conversion")
        return Orbitals(coeffs, None, None, basis_set)

    @pytest.fixture
    def simple_wavefunction(self, basic_orbitals):
        """Create a simple wavefunction with two determinants."""
        det1 = Configuration("20")  # Doubly occupied first orbital
        det2 = Configuration("ud")  # Singly occupied each orbital
        dets = [det1, det2]
        coeffs = np.array([0.9, 0.436])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        return Wavefunction(container)

    def test_statevector_basic_properties(self, simple_wavefunction):
        """Test basic properties of generated statevector."""
        sv = create_statevector_from_wavefunction(simple_wavefunction, normalize=False)

        # Check type and shape
        assert isinstance(sv, np.ndarray)
        assert sv.dtype == np.complex128
        assert sv.shape == (2**4,)  # 2 orbitals * 2 = 4 qubits

        # Check that only expected indices are non-zero
        nonzero_indices = np.nonzero(sv)[0]
        assert len(nonzero_indices) == 2

    def test_statevector_determinant_mapping(self, simple_wavefunction):
        """Test that determinants map to correct indices."""
        sv = create_statevector_from_wavefunction(simple_wavefunction, normalize=False)

        # Configuration "20" -> index for doubly occupied first orbital
        # Orbital 0: α=1, β=1; Orbital 1: α=0, β=0
        # Qubits: 3 2 1 0 -> beta1 beta0 alpha1 alpha0 -> 0 0 0 1 | 0 0 1 0 -> ... wait
        # Actually: lower 2 bits are alpha orbitals, upper 2 bits are beta orbitals
        # "20": orbital 0 doubly, orbital 1 empty
        # Alpha: 1 0, Beta: 1 0
        # Qubits [3 2 | 1 0] = [beta1 beta0 | alpha1 alpha0] = [0 1 | 0 1] = binary 0101 = 5
        # Wait, let me recalculate using the helper function
        det1_index = _configuration_to_statevector_index(Configuration("20"), 2)
        det2_index = _configuration_to_statevector_index(Configuration("ud"), 2)

        # Check coefficients are in the right places
        assert np.isclose(
            sv[det1_index], 0.9, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )
        assert np.isclose(
            sv[det2_index], 0.436, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_statevector_normalization(self, simple_wavefunction):
        """Test that normalization works correctly."""
        sv_unnormalized = create_statevector_from_wavefunction(simple_wavefunction, normalize=False)
        sv_normalized = create_statevector_from_wavefunction(simple_wavefunction, normalize=True)

        # Check unnormalized has expected norm
        expected_norm = np.sqrt(0.9**2 + 0.436**2)
        assert np.isclose(
            np.linalg.norm(sv_unnormalized),
            expected_norm,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check normalized has unit norm
        assert np.isclose(
            np.linalg.norm(sv_normalized),
            1.0,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

        # Check normalization is consistent
        expected_normalized = sv_unnormalized / expected_norm
        assert np.allclose(
            sv_normalized,
            expected_normalized,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_complex_wavefunction(self, basic_orbitals):
        """Test statevector creation with complex coefficients."""
        det1 = Configuration("20")
        det2 = Configuration("ud")
        dets = [det1, det2]
        coeffs = np.array([0.8 + 0.2j, 0.3 - 0.4j])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        sv = create_statevector_from_wavefunction(wf, normalize=False)

        # Check that complex coefficients are preserved
        det1_index = _configuration_to_statevector_index(det1, 2)
        det2_index = _configuration_to_statevector_index(det2, 2)

        assert np.isclose(
            sv[det1_index],
            0.8 + 0.2j,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )
        assert np.isclose(
            sv[det2_index],
            0.3 - 0.4j,
            rtol=float_comparison_relative_tolerance,
            atol=float_comparison_absolute_tolerance,
        )

    def test_single_determinant_wavefunction(self, basic_orbitals):
        """Test statevector for wavefunction with single determinant."""
        det = Configuration("20")
        dets = [det]
        coeffs = np.array([0.7])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        sv = create_statevector_from_wavefunction(wf, normalize=False)

        # Only one non-zero element
        nonzero_indices = np.nonzero(sv)[0]
        assert len(nonzero_indices) == 1

        det_index = _configuration_to_statevector_index(det, 2)
        assert np.isclose(
            sv[det_index], 0.7, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_statevector_dimension_scaling(self):
        """Test that statevector dimension scales correctly with orbital count."""
        # Test with different numbers of orbitals
        for num_orbs in [1, 2, 3, 4]:
            coeffs = np.eye(num_orbs)
            basis_set = create_test_basis_set(num_orbs, f"test-dim-{num_orbs}")
            orbitals = Orbitals(coeffs, None, None, basis_set)

            # Create simple wavefunction
            config_str = "2" + "0" * (num_orbs - 1)
            det = Configuration(config_str)
            dets = [det]
            coeffs_wf = np.array([1.0])

            container = CasWavefunctionContainer(coeffs_wf, dets, orbitals)
            wf = Wavefunction(container)

            sv = create_statevector_from_wavefunction(wf, normalize=False)

            # Check dimension: 2^(2*num_orbs)
            expected_dim = 2 ** (2 * num_orbs)
            assert sv.shape == (expected_dim,)

    def test_zero_norm_handling(self, basic_orbitals):
        """Test handling of zero-norm wavefunctions."""
        det = Configuration("20")
        dets = [det]
        coeffs = np.array([0.0])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        # Should not raise error, but normalization should not happen
        sv = create_statevector_from_wavefunction(wf, normalize=True)
        assert np.all(sv == 0.0)

    def test_multiple_determinants(self, basic_orbitals):
        """Test wavefunction with multiple determinants."""
        det1 = Configuration("20")
        det2 = Configuration("ud")
        det3 = Configuration("du")
        det4 = Configuration("02")
        dets = [det1, det2, det3, det4]
        coeffs = np.array([0.5, 0.3, 0.2, 0.1])

        container = CasWavefunctionContainer(coeffs, dets, basic_orbitals)
        wf = Wavefunction(container)

        sv = create_statevector_from_wavefunction(wf, normalize=False)

        # Check all determinants are present
        for i, det in enumerate(dets):
            det_index = _configuration_to_statevector_index(det, 2)
            assert np.isclose(
                sv[det_index],
                coeffs[i],
                rtol=float_comparison_relative_tolerance,
                atol=float_comparison_absolute_tolerance,
            )

        # Check that there are exactly 4 non-zero elements
        nonzero_indices = np.nonzero(sv)[0]
        assert len(nonzero_indices) == 4


class TestStatevectorIndexBinaryEncoding:
    """Test the binary encoding logic in detail."""

    def test_alpha_beta_separation(self):
        """Test that alpha and beta electrons are in separate bit ranges."""
        # Pure alpha configuration
        config_alpha = Configuration("uuuu")
        index_alpha = _configuration_to_statevector_index(config_alpha, 4)
        # Should have bits 0-3 set, bits 4-7 clear
        assert index_alpha == 0b00001111  # 15

        # Pure beta configuration
        config_beta = Configuration("dddd")
        index_beta = _configuration_to_statevector_index(config_beta, 4)
        # Should have bits 0-3 clear, bits 4-7 set
        assert index_beta == 0b11110000  # 240

        # Doubly occupied should be sum
        config_doubly = Configuration("2222")
        index_doubly = _configuration_to_statevector_index(config_doubly, 4)
        assert index_doubly == index_alpha + index_beta

    def test_bit_position_correspondence(self):
        """Test that orbital i corresponds to bit i for alpha and bit (n+i) for beta."""
        num_orbs = 4

        for orb_idx in range(num_orbs):
            # Test alpha electron in orbital i
            config_str = "0" * orb_idx + "u" + "0" * (num_orbs - orb_idx - 1)
            config = Configuration(config_str)
            index = _configuration_to_statevector_index(config, num_orbs)
            # Should have only bit orb_idx set
            assert index == (1 << orb_idx)

            # Test beta electron in orbital i
            config_str = "0" * orb_idx + "d" + "0" * (num_orbs - orb_idx - 1)
            config = Configuration(config_str)
            index = _configuration_to_statevector_index(config, num_orbs)
            # Should have only bit (num_orbs + orb_idx) set
            assert index == (1 << (num_orbs + orb_idx))

    def test_superposition_of_occupations(self):
        """Test that multiple occupied orbitals create correct bit patterns."""
        # Orbitals 0 and 2 with alpha, orbital 1 with beta
        config = Configuration("udu0")
        index = _configuration_to_statevector_index(config, 4)

        # Expected bits set:
        # Alpha: bit 0 (orbital 0), bit 2 (orbital 2) -> 0101
        # Beta: bit 5 (orbital 1) -> 0010 in beta range -> 00100000
        # Combined: 00100101 = 32 + 4 + 1 = 37
        assert index == 37
