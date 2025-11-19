"""Test for bitstring manipulation utilities in QDK/Chemistry.

This module provides comprehensive tests for the bitstring utility functions
in qdk_chemistry.utils.bitstring, which are essential for quantum state preparation
algorithms, particularly for converting between classical electronic structure
representations and quantum circuit formats.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.utils.bitstring import (
    binary_to_decimal,
    bitstrings_to_binary_matrix,
    separate_alpha_beta_to_binary_string,
)


def test_separate_alpha_beta_string():
    """Test separation of alpha and beta strings."""
    bitstring = "2du0"
    alpha, beta = separate_alpha_beta_to_binary_string(bitstring)
    assert alpha == "1010"
    assert beta == "1100"


def test_separate_alpha_beta_string_invalid_format():
    """Test separation with invalid characters."""
    bitstring = "2dx0"  # 'x' is invalid
    with pytest.raises(ValueError, match=r"Invalid character 'x' in input string\."):
        separate_alpha_beta_to_binary_string(bitstring)


def test_binary_to_decimal():
    """Test function for converting binary strings/lists to decimal integers."""
    # Test with binary string
    assert binary_to_decimal("1010") == 10
    assert binary_to_decimal("1010", reverse=True) == 5

    # Test with binary list
    assert binary_to_decimal([1, 0, 1, 0]) == 10
    assert binary_to_decimal([1, 0, 1, 0], reverse=True) == 5


def test_binary_to_decimal_invalid_input():
    """Test binary_to_decimal with invalid input types."""
    with pytest.raises(ValueError, match=r"Input must be a non-empty binary string or list\."):
        binary_to_decimal(1010)  # Invalid type: int


def test_bitstrings_to_binary_matrix():
    """Test conversion of bitstrings to binary matrices."""
    # Test case 1: Simple 3-qubit, 2-determinant example
    bitstrings = ["101", "010"]  # q[2]q[1]q[0] format (Little Endian)
    result = bitstrings_to_binary_matrix(bitstrings)

    # Expected: matrix should be (3, 2) with rows q[0], q[1], q[2]
    # "101" -> reversed to [1,0,1] for column 0
    # "010" -> reversed to [0,1,0] for column 1
    expected = np.array(
        [
            [1, 0],  # q[0]
            [0, 1],  # q[1]
            [1, 0],  # q[2]
        ],
        dtype=np.int8,
    )

    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, expected)

    # Test case 2: Single bitstring
    bitstrings = ["1100"]
    result = bitstrings_to_binary_matrix(bitstrings)

    # "1100" -> reversed to [0,0,1,1] for single column
    expected = np.array(
        [
            [0],  # q[0]
            [0],  # q[1]
            [1],  # q[2]
            [1],  # q[3]
        ],
        dtype=np.int8,
    )

    assert result.shape == (4, 1)
    np.testing.assert_array_equal(result, expected)

    # Test case 3: Multiple determinants with same length
    bitstrings = ["00", "01", "10", "11"]
    result = bitstrings_to_binary_matrix(bitstrings)

    # Expected matrix (2, 4):
    # "00" -> [0,0], "01" -> [1,0], "10" -> [0,1], "11" -> [1,1]
    expected = np.array(
        [
            [0, 1, 0, 1],  # q[0]
            [0, 0, 1, 1],  # q[1]
        ],
        dtype=np.int8,
    )

    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result, expected)

    # Test case 4: All zeros and all ones
    bitstrings = ["000", "111"]
    result = bitstrings_to_binary_matrix(bitstrings)

    expected = np.array(
        [
            [0, 1],  # q[0]
            [0, 1],  # q[1]
            [0, 1],  # q[2]
        ],
        dtype=np.int8,
    )

    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result, expected)

    # Test case 5: Verify matrix properties
    bitstrings = ["1010", "0101"]
    result = bitstrings_to_binary_matrix(bitstrings)

    # Check dtype and shape
    assert result.dtype == np.int8
    assert result.shape == (4, 2)

    # Check that all values are 0 or 1
    assert np.all((result == 0) | (result == 1))


def test_bitstrings_to_binary_matrix_edge_cases():
    """Test edge cases and error conditions for bitstring-to-matrix conversion."""
    # Test case 1: Empty bitstrings list
    with pytest.raises(ValueError, match="Bitstrings list cannot be empty"):
        bitstrings_to_binary_matrix([])

    # Test case 2: Inconsistent bitstring lengths
    bitstrings = ["10", "101"]  # Different lengths
    with pytest.raises(ValueError, match="All bitstrings must have the same length"):
        bitstrings_to_binary_matrix(bitstrings)

    # Test case 3: Single character bitstrings
    bitstrings = ["0", "1"]
    result = bitstrings_to_binary_matrix(bitstrings)

    expected = np.array([[0, 1]], dtype=np.int8)  # Single row for q[0]
    assert result.shape == (1, 2)
    np.testing.assert_array_equal(result, expected)

    # Test case 4: Large number of determinants
    bitstrings = ["01", "10", "00", "11", "01"]  # 5 determinants
    result = bitstrings_to_binary_matrix(bitstrings)

    expected = np.array(
        [
            [1, 0, 0, 1, 1],  # q[0]
            [0, 1, 0, 1, 0],  # q[1]
        ],
        dtype=np.int8,
    )

    assert result.shape == (2, 5)
    np.testing.assert_array_equal(result, expected)


def test_bitstrings_to_binary_matrix_qiskit_convention():
    """Test that the function correctly handles Qiskit Little Endian convention."""
    # Test the specific example from the docstring
    bitstrings = ["101", "010"]  # q[2]q[1]q[0] format
    result = bitstrings_to_binary_matrix(bitstrings)

    # Verify the transformation:
    # Input "101" means q[2]=1, q[1]=0, q[0]=1
    # Input "010" means q[2]=0, q[1]=1, q[0]=0
    # Output matrix should have q[0] in first row, q[1] in second row, q[2] in third row

    # Column 0 from "101": q[0]=1, q[1]=0, q[2]=1
    assert result[0, 0] == 1  # q[0] from "101"
    assert result[1, 0] == 0  # q[1] from "101"
    assert result[2, 0] == 1  # q[2] from "101"

    # Column 1 from "010": q[0]=0, q[1]=1, q[2]=0
    assert result[0, 1] == 0  # q[0] from "010"
    assert result[1, 1] == 1  # q[1] from "010"
    assert result[2, 1] == 0  # q[2] from "010"

    # Test another example to ensure consistency
    bitstrings = ["1001"]  # q[3]q[2]q[1]q[0] = 1001
    result = bitstrings_to_binary_matrix(bitstrings)

    # This should give us q[0]=1, q[1]=0, q[2]=0, q[3]=1
    expected = np.array([[1], [0], [0], [1]], dtype=np.int8)
    np.testing.assert_array_equal(result, expected)


def test_binary_to_decimal_edge_cases():
    """Test additional edge cases for binary_to_decimal."""
    # Test with single bit
    assert binary_to_decimal("1") == 1
    assert binary_to_decimal("0") == 0
    assert binary_to_decimal([1]) == 1
    assert binary_to_decimal([0]) == 0

    # Test reverse with empty inputs - should raise ValueError
    with pytest.raises(ValueError, match="invalid literal for int"):
        binary_to_decimal("", reverse=True)
    with pytest.raises(ValueError, match="invalid literal for int"):
        binary_to_decimal([], reverse=True)


def test_bitstrings_to_binary_matrix_additional_validation():
    """Test additional validation scenarios for bitstrings_to_binary_matrix."""
    # Test with large valid bitstring
    large_bitstring = ["0" * 50, "1" * 50]
    result = bitstrings_to_binary_matrix(large_bitstring)
    assert result.shape == (50, 2)
    assert np.all(result[:, 0] == 0)  # First column all zeros (reversed "0"*50)
    assert np.all(result[:, 1] == 1)  # Second column all ones (reversed "1"*50)
