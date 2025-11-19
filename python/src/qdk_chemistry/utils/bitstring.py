"""Bitstring utility functions for quantum computing and electronic structure calculations.

This module provides comprehensive utilities for working with bitstrings:

* **Quantum State Representation**:  Functions for converting between different quantum state representations
  including binary strings, decimal numbers, and statevectors.
* **Format Conversions**:
  Utilities for converting between different bitstring formats.
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

import numpy as np

from qdk_chemistry.data import Configuration

_LOGGER = logging.getLogger(__name__)


def separate_alpha_beta_to_binary_string(input_string: str) -> tuple[str, str]:
    """Separate a compact alpha-beta bitstring into separate alpha and beta parts.

    Compact format uses:

        * 2: orbital doubly occupied
        * u: orbital up electron
        * d: orbital down electron
        * 0: orbital unoccupied

    Example:
      Compact: 2du0 -> alpha: 1010, beta: 1100

    Args:
        input_string (str): The compact bitstring.

    Returns:
        A tuple containing the alpha and beta bitstrings.

    """
    n = len(input_string)
    alpha_string = ""
    beta_string = ""
    for i in range(n):
        if input_string[i] == "2":
            alpha_string += "1"
            beta_string += "1"
        elif input_string[i] == "u":
            alpha_string += "1"
            beta_string += "0"
        elif input_string[i] == "d":
            alpha_string += "0"
            beta_string += "1"
        elif input_string[i] == "0":
            alpha_string += "0"
            beta_string += "0"
        else:
            raise ValueError(f"Invalid character '{input_string[i]}' in input string.")
    return alpha_string[:n], beta_string[:n]


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


def bitstrings_to_binary_matrix(bitstrings: list[str]) -> np.ndarray:
    """Convert a list of bitstrings to a binary matrix.

    This function converts a list of bitstrings (determinants) into a binary matrix
    where each column represents a determinant and each row represents a qubit.

    Args:
        bitstrings (list[str]): List of bitstrings in Qiskit little endian order.
            Each bitstring represents a determinant where the string is ordered
            as "q[N-1]...q[0]" (most significant bit first in the string).

    Returns:
        Binary matrix M of shape (N, k) where

            * N is the number of qubits (rows)
            * k is the number of determinants (columns)

        The matrix follows Qiskit circuit top-down convention with row ordering "q[0]...q[N-1]" (qubit 0 at the top).

    Note:
        The input bitstrings are in Qiskit little endian order ("q[N-1]...q[0]"),
        but the output binary matrix follows the Qiskit circuit convention with
        row ordering "q[0]...q[N-1]". This means each bitstring is reversed
        when converting to a column in the matrix.

    Example:
        >>> bitstrings = ["101", "010"]  # q[2]q[1]q[0] format
        >>> matrix = bitstrings_to_binary_matrix(bitstrings)
        >>> print(matrix)
        [[1 0]  # q[0]
         [0 1]  # q[1]
         [1 0]] # q[2]

    """
    if not bitstrings:
        raise ValueError("Bitstrings list cannot be empty")

    n_qubits = len(bitstrings[0])
    n_dets = len(bitstrings)

    # Validate all bitstrings have the same length
    for i, bitstring in enumerate(bitstrings):
        if len(bitstring) != n_qubits:
            raise ValueError(
                f"All bitstrings must have the same length. "
                f"Bitstring {i} has length {len(bitstring)}, expected {n_qubits}"
            )

    # Create binary matrix with correct row ordering (reverse each bitstring)
    bitstring_matrix = np.zeros((n_qubits, n_dets), dtype=np.int8)
    for i, bitstring in enumerate(bitstrings):
        # Reverse the bitstring to get correct qubit ordering
        # Input: "q[N-1]...q[0]" -> Output: column with q[0] at top
        reversed_bitstring = bitstring[::-1]
        bitstring_matrix[:, i] = np.array(list(map(int, reversed_bitstring)), dtype=np.int8)

    return bitstring_matrix
