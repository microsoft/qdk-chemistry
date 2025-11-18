"""Statevector utilities in qdk_chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import Wavefunction
from qdk_chemistry.utils.bitstring import binary_to_decimal, separate_alpha_beta_to_binary_string


def _create_statevector_from_coeffs_and_dets_string(
    coeffs: list[complex | float] | np.ndarray,
    dets_string: list[str],
    num_qubits: int,
    normalize: bool = True,
) -> np.ndarray:
    """Convert coefficients and determinants into a column vector format.

    Args:
        coeffs: List of coefficients for each determinant.
        dets_string: List of binary determinants as strings or lists.
        num_qubits: Number of qubits in the system.
        normalize: Whether to normalize the statevector.

    Returns:
        A complex numpy array to represent the statevector.

    """
    if len(coeffs) != len(dets_string):
        raise ValueError("Number of coefficients must match number of bitstrings.")
    statevector = np.zeros(2**num_qubits, dtype=complex)
    for coeff, det in zip(coeffs, dets_string, strict=True):
        det_num = binary_to_decimal(det)
        if det_num >= len(statevector):
            raise ValueError(f"Determinant {det} exceeds the size of 2**{num_qubits}.")
        statevector[det_num] = coeff
    # Normalize the statevector if requested
    if normalize is True:
        norm = np.linalg.norm(statevector)
        if norm == 0:
            raise ValueError("Zero statevector norm; cannot normalize.")
        statevector = statevector / norm
    return statevector


def create_statevector_from_wavefunction(
    wavefunction: Wavefunction,
    max_dets: int | None = None,
    renormalize: bool = True,
) -> np.ndarray:
    """Convert a wavefunction (in the form of a coefficient array) into a statevector.

    Args:
        wavefunction (Wavefunction): The Wavefunction.
        max_dets: Max number of determinants to include.
        renormalize: Whether to renormalize the statevector after filtering max number of determinants.

    Returns:
        A complex numpy array to represent the statevector.

    """
    orbitals = wavefunction.get_orbitals()
    active_orbital_indices, _ = orbitals.get_active_space_indices()
    num_orbs = len(active_orbital_indices)
    det_strings = []
    coeffs = []
    for det in wavefunction.get_active_determinants():
        coeffs.append(wavefunction.get_coefficient(det))
        alpha_str, beta_str = separate_alpha_beta_to_binary_string(det.to_string()[:num_orbs])
        bitstring = beta_str[::-1] + alpha_str[::-1]  # Convert to little-endian format to match qiskit convention
        det_strings.append(bitstring)

    indices = range(len(det_strings))
    sorted_indices = sorted(indices, key=lambda i: -np.abs(coeffs[i]))
    sorted_det_strings = [det_strings[i] for i in sorted_indices]
    sorted_coeffs = [coeffs[i] for i in sorted_indices]

    num_dets = len(det_strings)
    if max_dets is not None and max_dets < num_dets:
        sorted_det_strings = sorted_det_strings[:max_dets]
        sorted_coeffs = sorted_coeffs[:max_dets]

    return _create_statevector_from_coeffs_and_dets_string(
        sorted_coeffs, sorted_det_strings, num_orbs * 2, normalize=renormalize
    )
