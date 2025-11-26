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

import pytest

from qdk_chemistry.utils.bitstring import (
    binary_to_decimal,
)


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
