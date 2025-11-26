"""Bitstring utility functions for quantum computing and electronic structure calculations.

This module provides comprehensive utilities for working with bitstrings:

    * **Quantum State Representation**:  Functions for converting between different quantum state representations
        including binary strings, decimal numbers, and statevectors.
    * **Format Conversions**: Utilities for converting between different bitstring formats.
        * Compact format (2=doubly occupied, u=up, d=down, 0=empty)
        * Binary format (1=occupied, 0=empty)
    * **Matrix Operations**: Functions for converting bitstrings to binary matrices and performing operations on them,
        particularly useful for quantum circuit optimization and state preparation.

Key Features:

    * Conversion between binary, decimal, and quantum state representations
    * Support for multiple quantum computing framework conventions
    * Comprehensive input validation and error handling

The module is particularly useful for:

    * Quantum circuit optimization and state preparation
    * Quantum chemistry and electronic structure calculations
    * Converting between different quantum computing framework formats
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging

from qdk_chemistry.data import Configuration

_LOGGER = logging.getLogger(__name__)


def binary_string_to_configuration(bitstring: str) -> Configuration:
    """Convert a binary string to a Configuration object.

    Args:
        bitstring (str): Binary string representing the configuration.

    Returns:
        Configuration object corresponding to the binary string.

    """
    if len(bitstring) % 2 != 0:
        raise ValueError("Bitstring length must be even to represent alpha and beta electrons.")
    n = len(bitstring) // 2
    alpha_string = bitstring[:n][::-1]
    beta_string = bitstring[n:][::-1]
    canonical_string = ""
    for i in range(n):
        if alpha_string[i] == "1" and beta_string[i] == "1":
            canonical_string += "2"
        elif alpha_string[i] == "1" and beta_string[i] == "0":
            canonical_string += "u"
        elif alpha_string[i] == "0" and beta_string[i] == "1":
            canonical_string += "d"
        elif alpha_string[i] == "0" and beta_string[i] == "0":
            canonical_string += "0"
        else:
            raise ValueError("Invalid bitstring format.")
    return Configuration(canonical_string)


def binary_to_decimal(binary: str | list, reverse=False) -> int:
    """Convert a binary string or list to its decimal equivalent.

    Args:
        binary (str or list): Binary string or list of bits.
        reverse (bool): If True, reverse the order of the bits before conversion.

    Returns:
        Decimal representation of the binary input.

    Raises:
        ValueError: If the input is neither a string nor a list.

    """
    if reverse:
        binary = binary[::-1]
    if isinstance(binary, str):
        return int(binary, 2)
    if isinstance(binary, list):
        return int("".join(map(str, binary)), 2)
    raise ValueError("Input must be a non-empty binary string or list.")
