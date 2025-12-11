"""Conversion utilities for QDK Chemistry to Qiskit interoperability.

This module provides functions to convert QDK Chemistry objects into Qiskit-compatible
representations, particularly for quantum circuit simulation and state preparation.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry import data

__all__ = ["create_statevector_from_wavefunction"]


def create_statevector_from_wavefunction(wavefunction: data.Wavefunction, normalize: bool = True) -> np.ndarray:
    """Create a Qiskit-compatible statevector from a QDK Chemistry wavefunction.

    This function converts a QDK Chemistry wavefunction into a dense statevector
    representation suitable for use with Qiskit quantum circuit simulators.

    The encoding uses a little-endian qubit ordering convention where each spatial
    orbital is mapped to two qubits (one for alpha spin, one for beta spin):
    - Lower qubits (0 to num_orbitals-1): alpha spin orbitals
    - Upper qubits (num_orbitals to 2*num_orbitals-1): beta spin orbitals

    Each determinant in the wavefunction is mapped to its corresponding basis state
    index, and the wavefunction coefficient is placed at that index in the statevector.

    Args:
        wavefunction: The wavefunction to convert to statevector representation.
            Must have a defined active space with the same number of alpha and
            beta orbitals.
        normalize: Whether to normalize the resulting statevector to unit norm.
            Default is True.

    Returns:
        numpy.ndarray: Dense complex statevector of size 2^(2*num_active_orbitals).
            The dtype is always complex128, even if the wavefunction has real
            coefficients.

    Examples:
        >>> from qiskit.quantum_info import Statevector
        >>> # Assuming we have a wavefunction already
        >>> sv_array = create_statevector_from_wavefunction(wavefunction)
        >>> qiskit_sv = Statevector(sv_array)
        >>> print(f"Statevector dimension: {len(sv_array)}")

    """
    orbitals = wavefunction.get_orbitals()
    indices, _ = orbitals.get_active_space_indices()

    num_orbs = len(indices)
    num_qubits = num_orbs * 2
    dim = 1 << num_qubits  # 2^num_qubits

    # Initialize statevector as complex array
    statevector = np.zeros(dim, dtype=np.complex128)

    # Get determinants and coefficients
    determinants = wavefunction.get_active_determinants()
    coefficients = wavefunction.get_coefficients()

    coeffs_array = np.array(coefficients)

    # Fill statevector
    for i, det in enumerate(determinants):
        # Convert configuration to statevector index
        index = _configuration_to_statevector_index(det, num_orbs)
        statevector[index] = coeffs_array[i]

    # Normalize if requested
    if normalize:
        norm = np.linalg.norm(statevector)
        if norm > 1e-15:
            statevector /= norm

    return statevector


def _configuration_to_statevector_index(configuration: data.Configuration, num_orbitals: int) -> int:
    """Convert a Configuration to its corresponding integer index in the statevector array.

    This function maps an electronic configuration (orbital occupation pattern) to
    its position in a dense statevector representation. The encoding uses little-endian
    qubit ordering where alpha electrons occupy lower-indexed qubits and beta electrons
    occupy higher-indexed qubits.

    The qubit layout for n spatial orbitals is:
        Qubits: [2n-1, 2n-2, ..., n+1, n] [n-1, n-2, ..., 1, 0]
                      beta orbitals              alpha orbitals

    Example:
        Configuration "2ud0" with 4 orbitals maps to:
        - Orbital 0: doubly occupied
        - Orbital 1: alpha
        - Orbital 2: beta
        - Orbital 3: empty

        Qubit layout:
        Qubits: 7 6 5 4 | 3 2 1 0
                beta    | alpha
                3 2 1 0 | 3 2 1 0
                0 1 0 1 | 0 0 1 1

        As binary (little-endian): 01010011 = 64 + 16 + 2 + 1 = 83

    Args:
        configuration (Configuration): The electronic configuration to convert. This object
            encodes the occupation of each orbital (unoccupied, alpha, beta,
            or doubly occupied).
        num_orbitals (int): Number of spatial orbitals to use from the configuration.
            This allows extracting a subset for active space calculations.

    Returns:
        int: The statevector index (0 to 2^(2*num_orbitals) - 1) corresponding
            to this configuration in the computational basis.

    Raises:
        RuntimeError: If num_orbitals exceeds the configuration's capacity.

    """
    # Get binary strings for alpha and beta
    alpha_str, beta_str = configuration.to_binary_strings(num_orbitals)

    index = 0

    # Process alpha electrons (lower bits)
    # Little-endian: bit i corresponds to qubit i
    for i, bit in enumerate(alpha_str):
        if bit == "1":
            index |= 1 << i

    # Process beta electrons (upper bits, offset by num_orbitals)
    for i, bit in enumerate(beta_str):
        if bit == "1":
            index |= 1 << (num_orbitals + i)

    return index
