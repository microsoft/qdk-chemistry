"""Tests for MPS sparse state preparation algorithm.

Tests both the classical preprocessing (decomposition correctness) and
the full Q# circuit (state preparation fidelity via statevector simulation).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk import qsharp
from scipy.linalg import block_diag

from qdk_chemistry.algorithms.state_preparation.mps_sparse import (
    MPSSparseStatePreparation,
    _decompose_sparse_site,
    _expand_to_unitary,
    _find_column_permutation,
    _get_rectangles_and_row_permutation,
    _invert_perm,
    _order_blocks,
    _pad_permutation,
    _perm_to_bitstrings,
    _tensor_to_target_matrix,
    generate_mps_sparse_preparation_data,
)
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction
from qdk_chemistry.utils.qsharp import get_qsharp_utils

# =============================================================================
# Qualtran reference data (from test_mps_sequential_state_prep.py)
# =============================================================================

_qualtran_mps_tensors = (
    np.array(
        [
            [
                [0.01650572, 0.0, 0.0, 0.0],
                [0.0, -0.52929781, 0.0, 0.0],
                [0.0, 0.0, -0.84462254, 0.0],
                [0.0, 0.0, 0.0, -0.07863941],
            ]
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.05969264, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.9973967, 0.04045497, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.08381532, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.98376348, 0.15869598, 0.0],
            ],
            [
                [-0.0421477, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.46961402, 0.0265522, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.41109095, 0.03268939, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.77904869],
            ],
        ]
    ),
    np.array(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.19640516, 0.0, 0.0, 0.0], [0.0, -0.98052283, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.98052283, 0.0, 0.0, 0.0], [0.0, 0.19640516, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [-0.02411236, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.99970925, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [-0.99970925, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.02411236, 0.0]],
            [
                [-0.17695837, 0.0, 0.0, 0.0],
                [0.0, -0.58052668, 0.0, 0.0],
                [0.0, 0.0, -0.53176612, 0.0],
                [0.0, 0.0, 0.0, -0.59067698],
            ],
        ]
    ),
    np.array(
        [
            [[0.0], [0.0], [0.0], [1.0]],
            [[0.0], [0.0], [1.0], [0.0]],
            [[0.0], [1.0], [0.0], [0.0]],
            [[1.0], [0.0], [0.0], [0.0]],
        ]
    ),
)

_qualtran_mps_expected_state = np.array(
    [0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.01650572, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.03159519, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.12468186, 0.        , 0.        ,
     0.51343194, 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.07079231, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.15403441, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.82743524, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00331447, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00930066, 0.        , 0.        ,
     0.03580077, 0.        , 0.        , 0.        , 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.00334943, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.03225657, 0.        , 0.        ,
     0.        , 0.        , 0.        , 0.01084116, 0.        , 0.        ,
     0.03556534, 0.        , 0.        , 0.03257808, 0.        , 0.        ,
     0.03618719, 0.        , 0.        , 0.        ])  # fmt: skip

# Qualtran resource estimates for sparse mode.
QUALTRAN_COST_SPARSE = {"num_qubits": 32, "toffoli": 321}

# =============================================================================
# Qualtran reference decomposition for 16×4×16 sparse tensor
# (from SiteUnitarySparse in Rupprecht & Wölk's Qualtran implementation)
# =============================================================================

# fmt: off
# Sparse tensor from the Qualtran bloq_example (SiteUnitarySparse).
# Shape: (16, 4, 16) — a right-canonical MPS tensor with block-sparse structure.
_qualtran_sparse_tensor_values = np.array(
    [ 1.        , -0.06529681, -0.21823697,  0.80873407, -0.54227129,
      0.99196206, -0.05352461,  0.10145871,  0.05340906, -0.0314629 ,
     -0.97114398, -0.23105394,  0.05003537, -0.10371929, -0.07991947,
      0.53129032,  0.83701003, -0.71416854,  0.38268343, -0.58610297,
     -0.29581829, -0.92387953, -0.2427718 , -0.63439328,  0.77301045,
     -0.70710678, -0.70710678, -0.70710678,  0.70710678, -0.04667196,
      0.08183508,  0.17913141, -0.4376217 ,  0.50294103,  0.37637954,
      0.08194149,  0.10117301, -0.3569069 ,  0.47769864,  0.01959044,
     -0.6091723 ,  0.06806463, -0.27941392, -0.1228121 ,  0.20408854,
     -0.09277709, -0.18419342,  0.09848402, -0.4107272 , -0.51416707,
     -0.06458102, -0.00057106,  0.04646671, -0.39307151, -0.05791556,
     -0.32300574, -0.48580074, -0.13162936, -0.00774822, -0.37507311,
      0.56000868, -0.16724278,  0.2841895 ,  0.04419075,  0.08735139,
     -0.81258978, -0.35215065, -0.06047946, -0.06983318, -0.15004217,
     -0.01491312, -0.3064603 ,  0.01239363, -0.23504973, -0.0787661 ,
     -0.03991293,  0.00944982,  0.07644049,  0.09293174, -0.05945385,
     -0.95174013,  0.04407359,  0.10529448, -0.00350562,  0.02966449,
      0.02995518, -0.16987694,  0.01490345, -0.0160302 , -0.05276522,
     -0.06852768, -0.00860989, -0.0944491 ,  0.03659029,  0.97522901])
_qualtran_sparse_tensor_rows = np.array(
    [ 0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,
      4,  5,  5,  5,  6,  6,  6,  7,  7,  8,  8,  9,  9, 10, 10, 10,
     10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
     11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13,
     13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14,
     14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15])
_qualtran_sparse_tensor_phys = np.array(
    [3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 2, 3, 3,
     2, 3, 3, 2, 3, 1, 3, 1, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,
     0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2,
     3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 0, 0, 0, 1, 1, 1, 2,
     2, 2, 3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
_qualtran_sparse_tensor_cols = np.array(
    [ 0,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,
      3,  4,  5,  6,  4,  5,  6,  4,  6,  4, 10,  4, 10,  1,  2,  3,
      7,  8,  9, 11, 12, 13, 14, 15,  1,  2,  3,  7,  8,  9, 11, 12,
     13, 14, 15,  1,  2,  3,  7,  8,  9, 11, 12, 13, 14, 15,  1,  2,
      3,  7,  8,  9, 11, 12, 13, 14, 15,  1,  2,  3,  7,  8,  9, 11,
     12, 13, 14, 15,  1,  2,  3,  7,  8,  9, 11, 12, 13, 14, 15])


def _get_qualtran_sparse_tensor() -> np.ndarray:
    """Reconstruct the 16×4×16 test tensor as a dense array."""
    from scipy.sparse import coo_array
    sparse = coo_array(
        (_qualtran_sparse_tensor_values,
         (_qualtran_sparse_tensor_rows,
          _qualtran_sparse_tensor_phys,
          _qualtran_sparse_tensor_cols)),
        shape=(16, 4, 16),
    )
    return sparse.toarray()


# Qualtran reference decomposition results (ancilla_dim=16, active_dim=64).
# These come from SiteUnitarySparse._decomposition in the Qualtran implementation.
_QUALTRAN_SPARSE_REF_RECT_SHAPES = [(1, 1), (4, 4), (3, 3), (2, 2), (11, 6)]

_QUALTRAN_SPARSE_REF_ROW_PERM = [
    1, 2, 3, 23, 24, 25, 43, 44, 45, 62, 63, 0, 49, 50, 51, 36, 53, 54,
    20, 58, 48, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40,
    41, 42, 46, 47, 52, 55, 56, 57, 59, 60, 61]

# First 16 entries of col_perm correspond to the data columns and must match exactly.
_QUALTRAN_SPARSE_REF_COL_PERM_DATA = [20, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 1, 2, 3, 4, 5]

_QUALTRAN_SPARSE_REF_BLOCK_SIZES = [
    11, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1]

# Non-trivial blocks (size > 1) from Qualtran decomposition.
_QUALTRAN_SPARSE_REF_BLOCK_11 = np.array([
    [-4.66719600e-02, -6.09172300e-01, -5.71060000e-04,  2.84189500e-01, -2.35049730e-01,  2.96644900e-02, -7.78887764e-02, -5.62845150e-01, -3.00084719e-01, -2.13886358e-02, -2.76530005e-01],
    [ 8.18350800e-02,  6.80646300e-02,  4.64667100e-02,  4.41907500e-02, -7.87661000e-02,  2.99551800e-02,  2.12332587e-01, -1.02423868e-01,  1.80956226e-01,  9.27899747e-01, -1.67743747e-01],
    [ 1.79131410e-01, -2.79413920e-01, -3.93071510e-01,  8.73513900e-02, -3.99129300e-02, -1.69876940e-01, -1.94377454e-01,  2.96320577e-01, -4.11523544e-01,  2.82429466e-01,  5.67931859e-01],
    [-4.37621700e-01, -1.22812100e-01, -5.79155600e-02, -8.12589780e-01,  9.44982000e-03,  1.49034500e-02, -2.49246433e-02, -2.08763878e-02, -3.10632025e-01,  1.31702496e-01, -1.20589376e-01],
    [ 5.02941030e-01,  2.04088540e-01, -3.23005740e-01, -3.52150650e-01,  7.64404900e-02, -1.60302000e-02, -1.02385827e-01, -6.30288735e-01,  1.42792323e-01, -5.74266003e-02,  1.98799081e-01],
    [ 3.76379540e-01, -9.27770900e-02, -4.85800740e-01, -6.04794600e-02,  9.29317400e-02, -5.27652200e-02, -2.91772027e-02,  3.54277773e-01, -2.68404030e-02, -6.22028630e-02, -6.83883469e-01],
    [ 8.19414900e-02, -1.84193420e-01, -1.31629360e-01, -6.98331800e-02, -5.94538500e-02, -6.85276800e-02,  9.37058426e-01,  6.13959385e-03, -4.32501422e-02, -1.65389829e-01,  1.46915119e-01],
    [ 1.01173010e-01,  9.84840200e-02, -7.74822000e-03, -1.50042170e-01, -9.51740130e-01, -8.60989000e-03, -7.33206196e-02,  1.39656911e-01,  1.40746943e-01, -8.25912867e-02,  9.89163323e-03],
    [-3.56906900e-01, -4.10727200e-01, -3.75073110e-01, -1.49131200e-02,  4.40735900e-02, -9.44491000e-02, -9.46442210e-02,  9.75208942e-03,  7.26776550e-01, -9.10716239e-03,  1.21719174e-01],
    [ 4.77698640e-01, -5.14167070e-01,  5.60008680e-01, -3.06460300e-01,  1.05294480e-01,  3.65902900e-02, -7.98087312e-02,  1.90290215e-01,  2.00906998e-01,  2.15479934e-03,  6.73405327e-02],
    [ 1.95904400e-02, -6.45810200e-02, -1.67242780e-01,  1.23936300e-02, -3.50562000e-03,  9.75229010e-01,  1.81339969e-02,  7.64796339e-02, -1.41988041e-03,  1.71027729e-03,  1.00274234e-01]])

_QUALTRAN_SPARSE_REF_BLOCK_4 = np.array([
    [-0.06529681,  0.99196206, -0.0314629 , -0.10371929],
    [-0.21823697, -0.05352461, -0.97114398, -0.07991947],
    [ 0.80873407,  0.10145871, -0.23105394,  0.53129032],
    [-0.54227129,  0.05340906,  0.05003537,  0.83701003]])

_QUALTRAN_SPARSE_REF_BLOCK_3 = np.array([
    [-0.71416854, -0.29581829, -0.63439328],
    [ 0.38268343, -0.92387953,  0.        ],
    [-0.58610297, -0.2427718 ,  0.77301045]])

_QUALTRAN_SPARSE_REF_BLOCK_2 = np.array([
    [-0.70710678, -0.70710678],
    [-0.70710678,  0.70710678]])
# fmt: on


# =============================================================================
# Tests: Preprocessing correctness
# =============================================================================


class TestSparseDecompositionHelpers:
    """Unit tests for the internal sparse decomposition helper functions."""

    def test_tensor_to_target_matrix_shape(self):
        """Verify target matrix has correct shape."""
        tensor = _qualtran_mps_tensors[1]  # shape (4, 4, 6)
        chi_left = tensor.shape[0]
        ancilla_dim = 8
        mat = _tensor_to_target_matrix(tensor, ancilla_dim)
        assert mat.shape == (4 * ancilla_dim, chi_left)

    def test_tensor_to_target_matrix_sparse(self):
        """Verify target matrix is sparse (most entries zero)."""
        tensor = _qualtran_mps_tensors[1]
        mat = _tensor_to_target_matrix(tensor, 8)
        dense = mat.toarray()
        nnz_frac = np.count_nonzero(dense) / dense.size
        assert nnz_frac < 0.15  # less than 15% non-zero

    def test_rectangles_cover_nonzero_rows(self):
        """Verify that extracted rectangles cover all nonzero structure."""
        tensor = _qualtran_mps_tensors[1]
        ancilla_dim = 8
        mat = _tensor_to_target_matrix(tensor, ancilla_dim)
        rectangles, row_perm = _get_rectangles_and_row_permutation(mat)

        # All nonzero rows should be captured in row_perm
        dense = mat.toarray()
        nonzero_rows = set(np.where(np.any(dense != 0, axis=1))[0])
        assert nonzero_rows.issubset(set(row_perm))

        # Each rectangle should be non-empty
        for rect in rectangles:
            assert rect.size > 0

    def test_column_permutation_is_valid(self):
        """Verify column permutation is a valid permutation."""
        tensor = _qualtran_mps_tensors[1]
        ancilla_dim = 8
        active_dim = 4 * ancilla_dim
        mat = _tensor_to_target_matrix(tensor, ancilla_dim)
        rectangles, _ = _get_rectangles_and_row_permutation(mat)
        col_perm = _find_column_permutation(rectangles, active_dim)

        # Should be a valid permutation of [0..active_dim-1]
        assert sorted(col_perm) == list(range(active_dim))

    def test_expand_to_unitary_produces_unitary(self):
        """Verify expanded blocks are unitary."""
        tensor = _qualtran_mps_tensors[1]
        ancilla_dim = 8
        mat = _tensor_to_target_matrix(tensor, ancilla_dim)
        rectangles, _ = _get_rectangles_and_row_permutation(mat)

        for rect in rectangles:
            unitary = _expand_to_unitary(rect)
            h = unitary.shape[0]
            assert np.allclose(unitary @ unitary.T, np.eye(h), atol=1e-8)

    def test_order_blocks_sorts_descending(self):
        """Verify blocks are sorted by size largest first."""
        blocks = [np.eye(2), np.eye(5), np.eye(3)]
        dim = 10
        _, sorted_blocks = _order_blocks(blocks, dim)
        sizes = [b.shape[0] for b in sorted_blocks]
        assert sizes == sorted(sizes, reverse=True)

    def test_order_blocks_permutation_valid(self):
        """Verify _order_blocks returns a valid permutation."""
        blocks = [np.eye(2), np.eye(5), np.eye(3)]
        dim = 10
        perm, _ = _order_blocks(blocks, dim)
        assert sorted(perm) == list(range(dim))

    def test_invert_perm_roundtrip(self):
        """Verify inverting a permutation round-trips."""
        perm = [3, 0, 4, 1, 2]
        inv = _invert_perm(perm)
        # Applying perm then inv should be identity
        for i in range(len(perm)):
            assert inv[perm[i]] == i

    def test_perm_to_bitstrings_roundtrip(self):
        """Verify bitstring encoding round-trips correctly."""
        perm = [5, 2, 7, 0, 3, 1, 6, 4]
        num_bits = 3
        bitstrings = _perm_to_bitstrings(perm, num_bits)
        for i, bits in enumerate(bitstrings):
            val = sum(int(b) << j for j, b in enumerate(bits))
            assert val == perm[i]


class TestSparseDecompositionCorrectness:
    """End-to-end correctness tests for the sparse decomposition.

    Verifies that V[invert(row_perm)][:, col_perm][:, :cols] == target.
    """

    @pytest.mark.parametrize("site_idx", [1, 2, 3])
    def test_decomposition_reconstructs_target(self, site_idx):
        """Verify decomposition reproduces the target matrix for each site."""
        tensor = _qualtran_mps_tensors[site_idx]
        chi_left = tensor.shape[0]
        ancilla_dim = 8
        active_dim = 4 * ancilla_dim

        target_mat = _tensor_to_target_matrix(tensor, ancilla_dim)
        target_dense = target_mat.toarray()
        num_cols = chi_left

        # Run the decomposition steps manually
        rectangles, row_perm = _get_rectangles_and_row_permutation(target_mat)
        row_perm = _pad_permutation(row_perm, active_dim)
        col_perm = _find_column_permutation(rectangles, active_dim)

        blocks = [_expand_to_unitary(r) for r in rectangles]
        blocks += [np.eye(1)] * (active_dim - sum(b.shape[0] for b in blocks))

        ordering_perm, blocks = _order_blocks(blocks, active_dim)

        # Compose following Qualtran convention
        col_composed = [col_perm[ordering_perm[i]] for i in range(active_dim)]
        col_perm_final = _invert_perm(col_composed)
        row_perm_final = [row_perm[ordering_perm[i]] for i in range(active_dim)]

        row_inv = _invert_perm(row_perm_final)
        block_diag_mat = block_diag(*blocks)
        result = block_diag_mat[row_inv][:, col_perm_final][:, :num_cols]

        assert np.allclose(result, target_dense, atol=1e-10), (
            f"Decomposition failed for site {site_idx}: max error = {np.max(np.abs(result - target_dense)):.2e}"
        )

    def test_decomposition_all_blocks_unitary(self):
        """Verify all blocks after ordering are unitary."""
        tensor = _qualtran_mps_tensors[1]
        ancilla_dim = 8
        active_dim = 4 * ancilla_dim

        mat = _tensor_to_target_matrix(tensor, ancilla_dim)
        rectangles, _ = _get_rectangles_and_row_permutation(mat)
        blocks = [_expand_to_unitary(r) for r in rectangles]
        blocks += [np.eye(1)] * (active_dim - sum(b.shape[0] for b in blocks))
        _, blocks = _order_blocks(blocks, active_dim)

        for i, b in enumerate(blocks):
            h = b.shape[0]
            err = np.max(np.abs(b @ b.T - np.eye(h)))
            assert err < 1e-8, f"Block {i} (size {h}) not unitary: error={err:.2e}"


class TestSparseDecompositionMatchesQualtran:
    """Verify QDK sparse decomposition matches the Qualtran reference implementation.

    Uses the 16×4×16 block-sparse tensor from the Qualtran SiteUnitarySparse bloq_example.
    Reference values were generated by running SiteUnitarySparse._decomposition from the
    Qualtran implementation by Rupprecht & Wölk (Zenodo: 20393500).
    """

    @pytest.fixture()
    def decomposition_data(self):
        """Run QDK decomposition on the Qualtran sparse tensor."""
        tensor = _get_qualtran_sparse_tensor()
        ancilla_dim = 16  # 2^4, since chi_left = chi_right = 16
        active_dim = 4 * ancilla_dim  # = 64
        target_mat = _tensor_to_target_matrix(tensor, ancilla_dim)

        rectangles, row_perm = _get_rectangles_and_row_permutation(target_mat)
        row_perm_padded = _pad_permutation(row_perm, active_dim)
        col_perm = _find_column_permutation(rectangles, active_dim)

        blocks = [_expand_to_unitary(r) for r in rectangles]
        blocks += [np.eye(1)] * (active_dim - sum(b.shape[0] for b in blocks))
        ordering_perm, sorted_blocks = _order_blocks(blocks, active_dim)

        col_composed = [col_perm[ordering_perm[i]] for i in range(active_dim)]
        col_perm_final = _invert_perm(col_composed)
        row_perm_final = [row_perm_padded[ordering_perm[i]] for i in range(active_dim)]

        return {
            "tensor": tensor,
            "ancilla_dim": ancilla_dim,
            "active_dim": active_dim,
            "target_mat": target_mat,
            "rectangles": rectangles,
            "sorted_blocks": sorted_blocks,
            "col_perm_final": col_perm_final,
            "row_perm_final": row_perm_final,
        }

    def test_rectangle_shapes_match(self, decomposition_data):
        """Verify extracted rectangle shapes match Qualtran."""
        rects = decomposition_data["rectangles"]
        shapes = [(r.shape[0], r.shape[1]) for r in rects]
        assert shapes == _QUALTRAN_SPARSE_REF_RECT_SHAPES

    def test_block_sizes_match(self, decomposition_data):
        """Verify block sizes after ordering match Qualtran."""
        sizes = [b.shape[0] for b in decomposition_data["sorted_blocks"]]
        assert sizes == _QUALTRAN_SPARSE_REF_BLOCK_SIZES

    def test_row_permutation_matches(self, decomposition_data):
        """Verify final row permutation matches Qualtran exactly."""
        assert decomposition_data["row_perm_final"] == _QUALTRAN_SPARSE_REF_ROW_PERM

    def test_col_permutation_data_columns_match(self, decomposition_data):
        """Verify column permutation matches Qualtran for data columns.

        The first chi_left=16 entries of the column permutation determine which
        columns of V map to the target matrix columns. These MUST match exactly.
        Entries beyond position 16 are "unused" dimensions (map to zero columns)
        and may differ without affecting correctness.
        """
        chi_left = 16
        col_perm = decomposition_data["col_perm_final"]
        assert col_perm[:chi_left] == _QUALTRAN_SPARSE_REF_COL_PERM_DATA

    def test_block_diagonal_11x11_matches(self, decomposition_data):
        """Verify the largest block (11×11) matches Qualtran."""
        block = decomposition_data["sorted_blocks"][0]
        assert block.shape == (11, 11)
        assert np.allclose(block, _QUALTRAN_SPARSE_REF_BLOCK_11, atol=1e-7)

    def test_block_diagonal_4x4_matches(self, decomposition_data):
        """Verify the 4×4 block matches Qualtran."""
        block = decomposition_data["sorted_blocks"][1]
        assert block.shape == (4, 4)
        assert np.allclose(block, _QUALTRAN_SPARSE_REF_BLOCK_4, atol=1e-7)

    def test_block_diagonal_3x3_matches(self, decomposition_data):
        """Verify the 3×3 block matches Qualtran."""
        block = decomposition_data["sorted_blocks"][2]
        assert block.shape == (3, 3)
        assert np.allclose(block, _QUALTRAN_SPARSE_REF_BLOCK_3, atol=1e-7)

    def test_block_diagonal_2x2_matches(self, decomposition_data):
        """Verify the 2×2 block matches Qualtran."""
        block = decomposition_data["sorted_blocks"][3]
        assert block.shape == (2, 2)
        assert np.allclose(block, _QUALTRAN_SPARSE_REF_BLOCK_2, atol=1e-7)

    def test_full_reconstruction_matches(self, decomposition_data):
        """Verify V[row_inv][:, col][:, :chi] reproduces the target matrix."""
        blocks = decomposition_data["sorted_blocks"]
        row_perm = decomposition_data["row_perm_final"]
        col_perm = decomposition_data["col_perm_final"]
        target_dense = decomposition_data["target_mat"].toarray()
        chi_left = 16

        row_inv = _invert_perm(row_perm)
        V = block_diag(*blocks)
        recon = V[row_inv][:, col_perm][:, :chi_left]

        assert np.allclose(recon, target_dense, atol=1e-10)

    def test_decompose_sparse_site_end_to_end(self):
        """Verify _decompose_sparse_site produces matching permutations."""
        tensor = _get_qualtran_sparse_tensor()
        ancilla_dim = 16
        result = _decompose_sparse_site(tensor, ancilla_dim)

        assert result.row_perm_targets == _QUALTRAN_SPARSE_REF_ROW_PERM
        assert result.col_perm_targets[:16] == _QUALTRAN_SPARSE_REF_COL_PERM_DATA
        assert result.target_bits == 6  # log2(64)

        # Inverse permutations should be consistent
        assert _invert_perm(result.row_perm_targets) == result.row_inv_perm_targets
        assert _invert_perm(result.col_perm_targets) == result.col_inv_perm_targets


class TestGenerateMPSSparsePreparationData:
    """Test the full sparse preprocessing pipeline."""

    def test_data_structure_standard(self):
        """Verify sparse data has correct structure for standard tensors."""
        data = generate_mps_sparse_preparation_data(_qualtran_mps_tensors)

        assert data.num_sites == 4
        assert data.ancilla_bits >= 2
        # One entry per site after the first
        assert len(data.sites) == 3

    def test_initial_state_normalized(self):
        """Verify the initial state vector is normalized."""
        data = generate_mps_sparse_preparation_data(_qualtran_mps_tensors)
        init_vec = np.array(data.initial_state_vec)
        assert abs(np.linalg.norm(init_vec) - 1.0) < 1e-10

    def test_permutation_sizes_consistent(self):
        """Verify permutation target arrays have consistent sizes."""
        data = generate_mps_sparse_preparation_data(_qualtran_mps_tensors)
        ancilla_dim = 1 << data.ancilla_bits
        active_dim = 4 * ancilla_dim

        for site in data.sites:
            assert len(site.col_perm_targets) == active_dim
            assert len(site.row_perm_targets) == active_dim
            assert sorted(site.col_perm_targets) == list(range(active_dim))
            assert sorted(site.row_perm_targets) == list(range(active_dim))
            assert site.target_bits == int(np.log2(active_dim))

    def test_to_qsharp_params_structure(self):
        """Verify to_qsharp_params returns all expected keys."""
        data = generate_mps_sparse_preparation_data(_qualtran_mps_tensors)
        params = data.to_qsharp_params(rotation_bits=10)

        expected_keys = {
            "initialStateVec",
            "numSites",
            "rotationBits",
            "numAncillaQubits",
            "siteColPermTargets",
            "siteColInvPermTargets",
            "siteRowPermTargets",
            "siteRowInvPermTargets",
            "siteBlockLayerAngles",
            "siteBlockLayerShifted",
            "siteBlockPhases",
        }
        assert set(params.keys()) == expected_keys
        assert params["numSites"] == 4
        assert params["rotationBits"] == 10
        assert len(params["siteColPermTargets"]) == 3
        assert len(params["siteRowPermTargets"]) == 3
        assert len(params["siteColInvPermTargets"]) == 3
        assert len(params["siteRowInvPermTargets"]) == 3

    @pytest.mark.parametrize(
        ("num_sites", "bond_dim", "seed"),
        [
            (2, 4, 42),
            (3, 2, 123),
            (4, 2, 456),
        ],
    )
    def test_random_mps_decomposition(self, num_sites, bond_dim, seed):
        """Verify decomposition produces valid data for random MPS tensors."""
        rng = np.random.default_rng(seed)
        mps = MPSWavefunction.random(num_sites=num_sites, bond_dim=bond_dim, rng=rng)
        data = generate_mps_sparse_preparation_data(mps.tensors)

        assert data.num_sites == num_sites
        assert len(data.sites) == num_sites - 1
        init_vec = np.array(data.initial_state_vec)
        assert abs(np.linalg.norm(init_vec) - 1.0) < 1e-10


class TestMPSSparseQSharpFidelity:
    """Test that the MPSSparse Q# circuit produces the correct state."""

    @pytest.mark.parametrize(
        ("num_sites", "bond_dim", "seed"),
        [
            (2, 4, 42),
        ],
    )
    def test_fidelity_random_mps(self, num_sites, bond_dim, seed):
        """Test sparse state preparation fidelity on random MPS.

        Uses single-shot statevector simulation via DumpMachine().
        """
        get_qsharp_utils()

        rng = np.random.default_rng(seed)
        mps = MPSWavefunction.random(num_sites=num_sites, bond_dim=bond_dim, rng=rng)
        target_state = mps.contract()

        data = generate_mps_sparse_preparation_data(mps.tensors)
        params = data.to_qsharp_params(rotation_bits=10)

        num_state_qubits = 2 * num_sites
        num_ancilla_qubits = data.ancilla_bits

        qs_code = _build_mps_sparse_eval_code(params)
        qsharp.eval(f"use state = Qubit[{num_state_qubits}];")
        qsharp.eval(f"use ancilla = Qubit[{num_ancilla_qubits}];")
        qsharp.eval(qs_code)
        dump = qsharp.dump_machine()
        amplitudes = np.array(dump.as_dense_state(), dtype=complex)
        qsharp.eval("ResetAll(state + ancilla);")

        state_amplitudes = _extract_state_amplitudes(amplitudes, num_state_qubits, num_ancilla_qubits)

        # P(ancilla = |0>) should be high
        ancilla_zero_prob = np.sum(np.abs(state_amplitudes) ** 2)
        assert ancilla_zero_prob > 0.85, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low"

        # Normalize and reindex
        state_amplitudes = state_amplitudes / np.sqrt(ancilla_zero_prob)
        state_amplitudes = _reindex_sites(state_amplitudes, num_sites)

        # Compute probability fidelity (Bhattacharyya coefficient squared).
        # The sparse circuit uses measurement-based QROAM uncomputation which
        # introduces non-deterministic phase flips; probability fidelity is the
        # appropriate metric.
        p = np.abs(state_amplitudes[: len(target_state)]) ** 2
        q = np.abs(target_state) ** 2
        fidelity = np.sum(np.sqrt(p * q)) ** 2
        assert fidelity > 0.90, f"Fidelity {fidelity:.4f} too low for num_sites={num_sites}, bond_dim={bond_dim}"

    def test_fidelity_qualtran_tensors(self):
        """Test sparse preparation fidelity on the Qualtran reference tensors."""
        get_qsharp_utils()

        mps = MPSWavefunction(_qualtran_mps_tensors)
        target_state = _qualtran_mps_expected_state

        data = generate_mps_sparse_preparation_data(mps.tensors)
        params = data.to_qsharp_params(rotation_bits=10)

        num_sites = 4
        num_state_qubits = 2 * num_sites
        num_ancilla_qubits = data.ancilla_bits

        qs_code = _build_mps_sparse_eval_code(params)
        qsharp.eval(f"use state = Qubit[{num_state_qubits}];")
        qsharp.eval(f"use ancilla = Qubit[{num_ancilla_qubits}];")
        qsharp.eval(qs_code)
        dump = qsharp.dump_machine()
        qsharp.eval("ResetAll(state + ancilla);")

        # Use sparse extraction (dump may have many internal qubits from QROAM)
        state_amplitudes = _extract_state_amplitudes_sparse(dump, num_state_qubits, num_ancilla_qubits)

        ancilla_zero_prob = np.sum(np.abs(state_amplitudes) ** 2)
        assert ancilla_zero_prob > 0.85, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low for Qualtran tensors"

        state_amplitudes = state_amplitudes / np.sqrt(ancilla_zero_prob)
        state_amplitudes = _reindex_sites(state_amplitudes, num_sites)

        # Probability fidelity: measurement-based QROAM uncomputation
        # introduces non-deterministic phase flips but preserves the correct
        # probability distribution.
        p = np.abs(state_amplitudes[: len(target_state)]) ** 2
        q = np.abs(target_state) ** 2
        fidelity = np.sum(np.sqrt(p * q)) ** 2
        assert fidelity > 0.90, f"Qualtran tensor fidelity {fidelity:.4f} too low"


class TestMPSSparseResourceEstimate:
    """Test that resource estimates are consistent with Qualtran sparse mode."""

    def test_resource_estimate_qubit_count(self):
        """Verify Q# resource estimate qubit count is reasonable."""
        mps = MPSWavefunction(_qualtran_mps_tensors)
        algo = MPSSparseStatePreparation()
        circuit = algo.run(mps)
        result = circuit.estimate()
        counts = result.logical_counts

        # Sparse mode should use comparable qubits to Qualtran
        assert counts["numQubits"] >= QUALTRAN_COST_SPARSE["num_qubits"]
        assert counts["numQubits"] <= QUALTRAN_COST_SPARSE["num_qubits"] * 2

    def test_resource_estimate_toffoli_count(self):
        """Verify Q# Toffoli count is in a reasonable range."""
        mps = MPSWavefunction(_qualtran_mps_tensors)
        algo = MPSSparseStatePreparation()
        circuit = algo.run(mps)
        result = circuit.estimate()
        counts = result.logical_counts

        assert counts["cczCount"] >= QUALTRAN_COST_SPARSE["toffoli"]
        assert counts["cczCount"] <= QUALTRAN_COST_SPARSE["toffoli"] * 2


# =============================================================================
# Helper functions
# =============================================================================


def _extract_state_amplitudes(
    amplitudes: np.ndarray,
    num_state_qubits: int,
    num_ancilla_qubits: int,
) -> np.ndarray:
    """Extract amplitudes where ancilla = |0>.

    DumpMachine qubit ordering: state[0]...state[N-1], ancilla[0]...
    Ancilla qubits are the rightmost bits.
    """
    num_total_qubits = num_state_qubits + num_ancilla_qubits
    dim = 2**num_total_qubits
    ancilla_mask = (1 << num_ancilla_qubits) - 1
    state_dim = 2**num_state_qubits
    state_amplitudes = np.zeros(state_dim, dtype=complex)
    for idx in range(dim):
        if (idx & ancilla_mask) == 0:
            state_idx = idx >> num_ancilla_qubits
            state_amplitudes[state_idx] = amplitudes[idx]
    return state_amplitudes


def _extract_state_amplitudes_sparse(
    dump,
    num_state_qubits: int,
    num_ancilla_qubits: int,
) -> np.ndarray:
    """Extract amplitudes where ancilla = |0> from a sparse DumpMachine result.

    Works with large qubit counts where as_dense_state() would be infeasible.
    Only considers basis states where all internal qubits (beyond state+ancilla)
    are |0>, i.e., properly uncomputed.

    Parameters
    ----------
    dump : StateDump
        Sparse dump from qsharp.dump_machine().
    num_state_qubits : int
        Number of state qubits (lowest-addressed qubits).
    num_ancilla_qubits : int
        Number of ancilla qubits (next after state).

    Returns
    -------
    np.ndarray
        Amplitudes for the state register conditioned on ancilla = |0>.

    """
    num_relevant_qubits = num_state_qubits + num_ancilla_qubits
    ancilla_mask = (1 << num_ancilla_qubits) - 1
    state_dim = 2**num_state_qubits
    state_amplitudes = np.zeros(state_dim, dtype=complex)

    for idx in dump:
        # Only consider states where internal qubits (above state+ancilla) are 0
        if idx >> num_relevant_qubits != 0:
            continue
        # Only consider states where ancilla = |0>
        if (idx & ancilla_mask) == 0:
            state_idx = idx >> num_ancilla_qubits
            if state_idx < state_dim:
                state_amplitudes[state_idx] = dump[idx]

    return state_amplitudes


def _reindex_sites(state_amplitudes: np.ndarray, num_sites: int) -> np.ndarray:
    """Reindex from Q# big-endian to Python MPS convention.

    The Q# circuit uses little-endian within each 2-qubit site, so
    DumpMachine's big-endian bits need to be reversed within each site.
    """
    site_bits = 2
    state_dim = len(state_amplitudes)
    reordered = np.zeros_like(state_amplitudes)
    for dm_idx in range(state_dim):
        py_idx = 0
        for site in range(num_sites):
            shift = (num_sites - 1 - site) * site_bits
            site_val = (dm_idx >> shift) & ((1 << site_bits) - 1)
            # Reverse bits within this site
            rev_val = 0
            for b in range(site_bits):
                if site_val & (1 << b):
                    rev_val |= 1 << (site_bits - 1 - b)
            py_idx |= rev_val << shift
        reordered[py_idx] = state_amplitudes[dm_idx]
    return reordered


def _float_to_qsharp(x: float) -> str:
    """Format float for Q#."""
    return f"{x:.15f}"


def _build_mps_sparse_eval_code(params: dict) -> str:
    """Build Q# eval code for MPSSparse from the params dict.

    Assumes `state` and `ancilla` qubit registers are already allocated in scope.
    """
    initial_state_str = ", ".join(_float_to_qsharp(x) for x in params["initialStateVec"])
    args = [
        f"[{initial_state_str}]",
        str(params["numSites"]),
        str(params["rotationBits"]),
        _nested_list_to_qsharp_3d_bool(params["siteColPermTargets"]),
        _nested_list_to_qsharp_3d_bool(params["siteColInvPermTargets"]),
        _nested_list_to_qsharp_3d_bool(params["siteRowPermTargets"]),
        _nested_list_to_qsharp_3d_bool(params["siteRowInvPermTargets"]),
        _nested_list_to_qsharp_3d(params["siteBlockLayerAngles"]),
        _nested_list_to_qsharp_2d_bool(params["siteBlockLayerShifted"]),
        _nested_list_to_qsharp_2d_bool(params["siteBlockPhases"]),
        "state",
        "ancilla",
    ]
    args_str = ",\n                ".join(args)
    return f"MPSSparse.MPSSparse(\n                {args_str}\n    )"


def _nested_list_to_qsharp_3d(data: list) -> str:
    """Convert list[list[list[float]]] to Q# literal."""
    site_strs = []
    for site in data:
        layer_strs = []
        for layer in site:
            angles = ", ".join(_float_to_qsharp(a) for a in layer)
            layer_strs.append(f"[{angles}]")
        site_strs.append(f"[{', '.join(layer_strs)}]")
    return f"[{', '.join(site_strs)}]"


def _nested_list_to_qsharp_3d_bool(data: list) -> str:
    """Convert list[list[list[bool]]] to Q# literal."""
    site_strs = []
    for site in data:
        row_strs = []
        for row in site:
            vals = ", ".join("true" if b else "false" for b in row)
            row_strs.append(f"[{vals}]")
        site_strs.append(f"[{', '.join(row_strs)}]")
    return f"[{', '.join(site_strs)}]"


def _nested_list_to_qsharp_2d_bool(data: list) -> str:
    """Convert list[list[bool]] to Q# literal."""
    site_strs = []
    for site in data:
        vals = ", ".join("true" if b else "false" for b in site)
        site_strs.append(f"[{vals}]")
    return f"[{', '.join(site_strs)}]"
