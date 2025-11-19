"""Test for statevector utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import numpy as np
import pytest

from qdk_chemistry.utils.statevector import (
    _create_statevector_from_coeffs_and_dets_string,
    create_statevector_from_wavefunction,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


def test_create_statevector():
    """Test statevector creation from coefficients and bit strings."""
    # Test with valid inputs
    coeffs = [1 + 1j, 0.5 - 0.5j]
    dets = ["0001", "0010"]
    num_qubits = 4
    expected_statevector = np.zeros(2**num_qubits, dtype=complex)
    expected_statevector[1] = 1 + 1j
    expected_statevector[2] = 0.5 - 0.5j

    result = _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits, normalize=False)
    assert np.allclose(
        result, expected_statevector, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )

    # Test with empty inputs
    coeffs = []
    dets = []
    num_qubits = 4
    expected_statevector = np.zeros(2**num_qubits, dtype=complex)

    result = _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits, normalize=False)
    assert np.allclose(
        result, expected_statevector, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
    )


def test_create_statevector_mismatched_lengths():
    """Test create_statevector with mismatched coefficient and determinant lengths."""
    # Line 123: Test error for mismatched lengths
    coeffs = [1.0, 0.5, 0.3]  # 3 coefficients
    dets = ["01", "10"]  # 2 determinants
    num_qubits = 2

    with pytest.raises(ValueError, match="Number of coefficients must match number of bitstrings"):
        _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits)


def test_create_statevector_determinant_exceeds_size():
    """Test create_statevector with determinant exceeding statevector size."""
    # Line 128: Test error for determinant exceeding 2**num_qubits
    coeffs = [1.0]
    dets = ["111"]  # Binary 111 = decimal 7
    num_qubits = 2  # 2**2 = 4, but determinant 7 >= 4
    match_string = f"Determinant .* exceeds the size of 2\\*\\*{num_qubits}"

    with pytest.raises(ValueError, match=match_string):
        _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits)


def test_create_statevector_edge_cases():
    """Test additional edge cases for create_statevector."""
    # Test with zero coefficient
    coeffs = [0.0, 1.0]
    dets = ["00", "01"]
    num_qubits = 2

    result = _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits)
    expected = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    np.testing.assert_array_equal(result, expected)

    # Test with complex coefficients
    coeffs = np.array([1 + 1j, 0.5 - 0.5j], dtype=complex)
    dets = ["00", "11"]
    num_qubits = 2

    result = _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits, normalize=False)
    expected = np.array([1 + 1j, 0.0, 0.0, 0.5 - 0.5j], dtype=complex)
    np.testing.assert_array_equal(result, expected)

    # Test with numpy array coefficients
    coeffs = np.array([0.7, 0.3])
    dets = ["10", "01"]
    num_qubits = 2

    result = _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits, normalize=False)
    expected = np.array([0.0, 0.3, 0.7, 0.0], dtype=complex)
    np.testing.assert_array_equal(result, expected)


def test_create_statevector_normalization():
    """Test statevector normalization."""
    coeffs = [1 + 1j, 0.5 - 0.5j]
    dets = ["0001", "0010"]
    num_qubits = 4

    # Create unnormalized statevector
    statevector = _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits, normalize=False)

    # Normalize manually
    norm = np.linalg.norm(statevector)
    normalized_statevector = statevector / norm

    # Create normalized statevector using the function
    result = _create_statevector_from_coeffs_and_dets_string(coeffs, dets, num_qubits, normalize=True)

    assert np.allclose(
        result,
        normalized_statevector,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_absolute_tolerance,
    )

    coeffs_zero = [0.0 + 0j, 0.0 + 0j]
    with pytest.raises(ValueError, match=re.escape("Zero statevector norm; cannot normalize.")):
        _create_statevector_from_coeffs_and_dets_string(coeffs_zero, dets, num_qubits, normalize=True)


def test_create_statevector_from_wavefunction(wavefunction_10e6o):
    """Test statevector creation from a Wavefunction object."""
    result = create_statevector_from_wavefunction(wavefunction_10e6o, renormalize=False)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2**12,)
    # Check that non-zero elements match the wavefunction
    nonzero_indices = np.nonzero(result)[0]
    assert len(nonzero_indices) == 3

    truncated_result = create_statevector_from_wavefunction(wavefunction_10e6o, max_dets=2, renormalize=True)
    assert truncated_result.shape == (2**12,)
    nonzero_indices_truncated = np.nonzero(truncated_result)[0]
    assert len(nonzero_indices_truncated) == 2
