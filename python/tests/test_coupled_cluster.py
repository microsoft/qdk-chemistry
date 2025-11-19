"""Tests for the CoupledClusterAmplitudes class in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.data import CoupledClusterAmplitudes, Orbitals

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import create_test_basis_set


def test_coupled_cluster_construction():
    """Test constructing CoupledClusterAmplitudes objects."""
    # Set up orbital data for testing using new immutable constructor
    coeffs = np.array([[0.9, 0.1], [0.1, -0.9]])
    energies = np.array([-1.0, 0.5])

    # Create AO overlap matrix (2x2 for this example)
    ao_overlap = np.eye(2)

    # Create basis set
    basis_set = create_test_basis_set(2)

    orb = Orbitals(coeffs, energies, ao_overlap, basis_set)

    # Create T1 and T2 amplitudes
    # For restricted case with 1 occupied, 1 virtual orbital:
    # T1 size should be no * nv = 1 * 1 = 1
    t1_amplitudes = np.array([0.01])
    # T2 size should be (no * nv)^2 = (1 * 1)^2 = 1
    t2_amplitudes = np.array([0.001])

    # Create CoupledClusterAmplitudes with data
    # For 1 alpha and 1 beta electron (restricted case)
    cc = CoupledClusterAmplitudes(orb, t1_amplitudes, t2_amplitudes, 1, 1)

    # Verify data is stored correctly
    assert cc.has_t1_amplitudes()
    assert cc.has_t2_amplitudes()

    # Check amplitudes
    assert np.allclose(
        cc.get_t1_amplitudes(),
        t1_amplitudes,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )
    assert np.allclose(
        cc.get_t2_amplitudes(),
        t2_amplitudes,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )

    # Test copy constructor
    cc2 = CoupledClusterAmplitudes(cc)

    assert cc2.has_t1_amplitudes()
    assert cc2.has_t2_amplitudes()

    assert np.allclose(
        cc2.get_t1_amplitudes(),
        t1_amplitudes,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )
    assert np.allclose(
        cc2.get_t2_amplitudes(),
        t2_amplitudes,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )


def test_coupled_cluster_orbital_indices():
    """Test accessing orbital counts from CoupledClusterAmplitudes."""
    # Set up orbital data with canonical occupations using new immutable constructor
    coeffs = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.1, 0.9]])
    energies = np.array([-1.0, -0.5, 0.5])

    # Create AO overlap matrix
    ao_overlap = np.eye(3)

    # Create basis set
    basis_set = create_test_basis_set(3)

    orb = Orbitals(coeffs, energies, ao_overlap, basis_set)

    # Create CoupledClusterAmplitudes object
    # For restricted case with 2 occupied, 1 virtual:
    # T1 size should be no * nv = 2 * 1 = 2
    t1_amplitudes = np.array([0.01, 0.02])
    # T2 size should be (no * nv)^2 = (2 * 1)^2 = 4
    t2_amplitudes = np.array([0.001, 0.002, 0.003, 0.004])

    cc = CoupledClusterAmplitudes(orb, t1_amplitudes, t2_amplitudes, 2, 2)

    # Test getting occupied counts
    alpha_occ_count, beta_occ_count = cc.get_num_occupied()
    assert alpha_occ_count == 2  # First two orbitals occupied for alpha
    assert beta_occ_count == 2  # First two orbitals occupied for beta (restricted case)

    # Test getting virtual counts
    alpha_virt_count, beta_virt_count = cc.get_num_virtual()
    assert alpha_virt_count == 1  # Last orbital virtual for alpha
    assert beta_virt_count == 1  # Last orbital virtual for beta (restricted case)


def test_coupled_cluster_indices_validation():
    """Test the validation of orbital indices and energies in CoupledClusterAmplitudes."""
    # Create amplitudes for case with 2 occupied, 1 virtual (for test later)
    t1_amplitudes = np.array([0.01, 0.02])
    t2_amplitudes = np.array([0.001, 0.002, 0.003, 0.004])

    # Test case 1: Non-adjacent indices
    coeffs = np.array([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.0, 0.2], [0.0, 0.1, 0.0, 0.9], [0.0, 0.0, 0.0, 0.1]])
    energies = np.array([-1.0, -0.5, 0.0, 0.5])

    # Create AO overlap matrix
    ao_overlap = np.eye(4)

    # Create basis set
    basis_set = create_test_basis_set(4)

    orb = Orbitals(coeffs, energies, ao_overlap, basis_set)

    # Should fail validation due to non-adjacent indices
    with pytest.raises(ValueError, match="Invalid T1 amplitudes dimension"):
        CoupledClusterAmplitudes(orb, t1_amplitudes, t2_amplitudes, 3, 3)

    # Test case 2: Unsorted energies
    coeffs2 = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.1, 0.9]])

    # Energy of second orbital (index 1) is higher than third (index 2)
    unsorted_energies = np.array([-1.0, 0.7, 0.5])

    # Create AO overlap matrix
    ao_overlap2 = np.eye(3)

    # Create basis set
    basis_set2 = create_test_basis_set(3)

    orb2 = Orbitals(coeffs2, unsorted_energies, ao_overlap2, basis_set2)

    # Should fail validation due to unsorted energies
    with pytest.raises(RuntimeError, match="energies"):
        CoupledClusterAmplitudes(orb2, t1_amplitudes, t2_amplitudes, 2, 2)


def test_coupled_cluster_amplitude_dimensions():
    """Test that T1/T2 amplitude dimensions are validated correctly."""
    # Set up orbital data with canonical occupations using new immutable constructor
    coeffs = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.2], [0.0, 0.1, 0.9]])
    energies = np.array([-1.0, -0.5, 0.5])

    # Create AO overlap matrix
    ao_overlap = np.eye(3)

    # Create basis set
    basis_set = create_test_basis_set(3)

    orb = Orbitals(coeffs, energies, ao_overlap, basis_set)

    # For restricted case with 2 occupied, 1 virtual orbitals:
    # T1 size should be no * nv = 2 * 1 = 2
    valid_t1 = np.array([0.01, 0.02])
    # T2 size should be (no * nv)² = (2 * 1)² = 4
    valid_t2 = np.array([0.001, 0.002, 0.003, 0.004])

    # Test that valid dimensions work
    cc = CoupledClusterAmplitudes(orb, valid_t1, valid_t2, 2, 2)
    assert cc.has_t1_amplitudes()
    assert cc.has_t2_amplitudes()

    # Test invalid T1 dimensions
    invalid_t1_large = np.array([0.01, 0.02, 0.03])  # 3 elements instead of 2
    with pytest.raises(ValueError, match="Invalid T1 amplitudes dimension"):
        CoupledClusterAmplitudes(orb, invalid_t1_large, valid_t2, 2, 2)
