"""Pure-Python Pauli matrix utilities.

Provides the same functionality as the C++ pybind11 ``pauli_matrix`` module
using only NumPy (and SciPy for sparse output), preserving the original
bitmask algorithms to minimise memory cost and keep efficiency.

All functions use Little-Endian label convention: ``label[0]`` corresponds to
qubit *n*-1 (MSB).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import scipy.sparse

__all__ = [
    "pauli_expectation",
    "pauli_string_to_masks",
    "pauli_to_dense_matrix",
    "pauli_to_sparse_matrix",
]


def pauli_string_to_masks(pauli_str: str) -> tuple[int, int, complex]:
    """Decompose a Pauli label string into bitmasks and phase factor.

    X sets a bit in *x_mask*, Z sets a bit in *z_mask*, Y sets both.
    The cumulative phase factor ``(i)^(number of Y operators)`` is also
    returned.

    Args:
        pauli_str: Pauli string (characters in {I, X, Y, Z}), using
            Little-Endian label convention (``label[0]`` = qubit *n*-1).

    Returns:
        ``(x_mask, z_mask, y_phase)`` where *x_mask* and *z_mask* are
        integers and *y_phase* is a complex number.

    """
    n = len(pauli_str)
    x_mask = 0
    z_mask = 0
    y_count = 0
    for i, ch in enumerate(pauli_str):
        bit = 1 << (n - 1 - i)
        if ch in "XY":
            x_mask |= bit
        if ch in "ZY":
            z_mask |= bit
        if ch == "Y":
            y_count += 1
    return x_mask, z_mask, 1j ** (y_count & 3)


def pauli_expectation(pauli_str: str, psi: np.ndarray) -> float:
    """Compute the expectation value ``<psi|P|psi>`` for a single Pauli string.

    Uses the bitmask approach to evaluate the expectation value without
    materialising the full Pauli matrix.  Only the real part is returned
    because every Pauli string is Hermitian.

    Args:
        pauli_str: Pauli label of length *n* (characters in {I, X, Y, Z}),
            using Little-Endian convention (``label[0]`` = qubit *n*-1).
        psi: Complex state vector of length ``2**n``.

    Returns:
        Real-valued expectation value.

    """
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    x_mask, z_mask, phase = pauli_string_to_masks(pauli_str)
    dim = psi.shape[0]

    # Build the row-index array and the corresponding column-index array
    rows = np.arange(dim, dtype=np.int64)
    cols = rows ^ x_mask  # P maps |r> -> phase * |r ^ x_mask>

    # Parity of (col & z_mask) gives the sign: (-1)^popcount(col & z_mask)
    parity_bits = cols & z_mask
    signs = np.where(np.bitwise_count(parity_bits) & 1, -1.0, 1.0)

    val = np.sum(np.conj(psi[rows]) * phase * signs * psi[cols])
    return float(val.real)


def pauli_to_dense_matrix(
    pauli_strings: list[str],
    coefficients: np.ndarray,
) -> np.ndarray:
    r"""Build a dense Hamiltonian matrix from Pauli strings and coefficients.

    Computes :math:`H = \sum_t \mathrm{coeff}[t] \cdot P_t` where each
    :math:`P_t` is a Pauli string in Little-Endian convention.

    Args:
        pauli_strings: List of Pauli label strings (characters in
            {I, X, Y, Z}), all of the same length *n*.
        coefficients: Complex array of coefficients, one per Pauli term.

    Returns:
        Dense complex matrix of shape ``(2**n, 2**n)``.

    """
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    n = len(pauli_strings[0])
    dim = 1 << n
    pauli_t = len(pauli_strings)

    # Pre-compute masks for all terms
    x_masks = np.empty(pauli_t, dtype=np.int64)
    z_masks = np.empty(pauli_t, dtype=np.int64)
    scaled = np.empty(pauli_t, dtype=np.complex128)
    for t, ps in enumerate(pauli_strings):
        xm, zm, yph = pauli_string_to_masks(ps)
        x_masks[t] = xm
        z_masks[t] = zm
        scaled[t] = coefficients[t] * yph

    matrix = np.zeros((dim, dim), dtype=np.complex128)
    rows = np.arange(dim, dtype=np.int64)

    for t in range(pauli_t):
        cols = rows ^ x_masks[t]
        signs = np.where(np.bitwise_count(cols & z_masks[t]) & 1, -1.0, 1.0)
        matrix[rows, cols] += scaled[t] * signs

    return matrix


def pauli_to_sparse_matrix(
    pauli_strings: list[str],
    coefficients: np.ndarray,
) -> scipy.sparse.csr_matrix:
    r"""Build a sparse CSR Hamiltonian matrix from Pauli strings and coefficients.

    Computes :math:`H = \sum_t \mathrm{coeff}[t] \cdot P_t` and returns the
    result as a :class:`scipy.sparse.csr_matrix`.

    Args:
        pauli_strings: List of Pauli label strings (characters in
            {I, X, Y, Z}), all of the same length *n*.
        coefficients: Complex array of coefficients, one per Pauli term.

    Returns:
        Sparse complex matrix of shape ``(2**n, 2**n)``.

    Raises:
        RuntimeError: If the matrix dimensions exceed int32 index limits.

    """
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    n = len(pauli_strings[0])
    dim = 1 << n
    # Pre-compute masks for all terms
    x_masks_list: list[int] = []
    z_masks_list: list[int] = []
    scaled_list: list[complex] = []
    for t, ps in enumerate(pauli_strings):
        xm, zm, yph = pauli_string_to_masks(ps)
        x_masks_list.append(xm)
        z_masks_list.append(zm)
        scaled_list.append(complex(coefficients[t] * yph))

    # Deduplicate x_masks to find nnz per row (same for every row)
    unique_xm = sorted(set(x_masks_list))
    nnz_per_row = len(unique_xm)
    total_nnz = nnz_per_row * dim

    if dim > np.iinfo(np.int32).max or total_nnz > np.iinfo(np.int32).max:
        raise RuntimeError("Matrix too large for int32 CSR indices in pauli_to_sparse_matrix.")

    # Group terms by their x_mask
    xm_to_idx = {xm: i for i, xm in enumerate(unique_xm)}
    terms_by_x: list[list[int]] = [[] for _ in range(nnz_per_row)]
    for t, xm in enumerate(x_masks_list):
        terms_by_x[xm_to_idx[xm]].append(t)

    z_masks_arr = np.array(z_masks_list, dtype=np.int64)
    scaled_arr = np.array(scaled_list, dtype=np.complex128)

    # Allocate CSR arrays
    indptr = np.arange(0, (dim + 1) * nnz_per_row, nnz_per_row, dtype=np.int32)
    indices = np.empty(total_nnz, dtype=np.int32)
    data = np.empty(total_nnz, dtype=np.complex128)

    rows = np.arange(dim, dtype=np.int64)

    for k, xm in enumerate(unique_xm):
        cols = rows ^ xm
        group = terms_by_x[k]
        acc = np.zeros(dim, dtype=np.complex128)
        for t in group:
            signs = np.where(np.bitwise_count(cols & z_masks_arr[t]) & 1, -1.0, 1.0)
            acc += scaled_arr[t] * signs
        # Fill into the k-th slot of each row
        indices[k::nnz_per_row] = cols.astype(np.int32)
        data[k::nnz_per_row] = acc

    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(dim, dim))
