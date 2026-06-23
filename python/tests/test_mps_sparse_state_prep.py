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
            (3, 4, 42),
            (4, 2, 42),
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

        # Compute fidelity
        fidelity = np.abs(np.dot(np.conj(state_amplitudes[: len(target_state)]), target_state)) ** 2
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

        fidelity = np.abs(np.dot(np.conj(state_amplitudes[: len(target_state)]), target_state)) ** 2
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
        assert counts["numQubits"] <= QUALTRAN_COST_SPARSE["num_qubits"] * 3

    def test_resource_estimate_toffoli_count(self):
        """Verify Q# Toffoli count is in a reasonable range."""
        mps = MPSWavefunction(_qualtran_mps_tensors)
        algo = MPSSparseStatePreparation()
        circuit = algo.run(mps)
        result = circuit.estimate()
        counts = result.logical_counts

        assert counts["cczCount"] > 0
        # Allow up to 5x Qualtran (different decomposition choices)
        assert counts["cczCount"] <= QUALTRAN_COST_SPARSE["toffoli"] * 5


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
