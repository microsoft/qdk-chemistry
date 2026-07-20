"""Tests for MPS state preparation algorithm.

Tests both the classical preprocessing (fidelity of the CSD decomposition)
and the full Q# circuit (state preparation fidelity and gate counts).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk import qsharp
from scipy.sparse import issparse

import qdk_chemistry.data as chemistry_data
from qdk_chemistry.algorithms.state_preparation.mps_sequential import (
    MPSSequentialStatePreparation,
    compute_site_unitary_dense_data,
    decompose_2d,
    decompose_unitary_to_givens,
    generate_mps_preparation_data,
)
from qdk_chemistry.data import (
    AbelianMPSContainer,
    MPSCanonicalForm,
    MPSContainer,
    MPSSite,
    Wavefunction,
    WavefunctionContainer,
)
from qdk_chemistry.utils.qsharp import get_qsharp_utils

from .mps_test_utils import contract_mps, make_mps, random_mps
from .test_helpers import create_test_orbitals


class TestAbelianMPSContainer:
    """Tests for the AbelianMPSContainer data container."""

    def test_basic_construction(self):
        """Test constructing an AbelianMPSContainer from tensors."""
        rng = np.random.default_rng(42)
        mps = random_mps(num_sites=3, bond_dim=4, rng=rng)
        assert mps.num_sites == 3
        assert mps.physical_dimension == 4
        assert mps.max_bond_dimension == 4

    def test_flattened_chemistry_properties(self):
        """Test that chemistry properties are exposed directly on the wavefunction."""
        site = MPSSite.from_dense(np.ones((1, 4, 1)))
        mps = AbelianMPSContainer([site], create_test_orbitals(1))

        assert isinstance(mps, WavefunctionContainer)
        assert isinstance(mps, MPSContainer)
        assert isinstance(mps, AbelianMPSContainer)
        assert mps.total_num_particles is None
        assert mps.active_num_particles is None
        assert Wavefunction(mps).get_container_type() == "mps"

    def test_metadata_is_stored_directly(self):
        """Test that MPS metadata is exposed directly on the container."""
        site = MPSSite.from_dense(np.ones((1, 4, 1)))
        physical_basis = ["empty", "alpha", "beta", "alpha_beta"]
        mps = AbelianMPSContainer(
            [site],
            create_test_orbitals(1),
            canonical_form=MPSCanonicalForm.Mixed,
            canonical_center=0,
            discarded_weight=1e-8,
            physical_basis=physical_basis,
        )

        assert mps.canonical_form == MPSCanonicalForm.Mixed
        assert mps.canonical_center == 0
        assert mps.discarded_weight == 1e-8
        assert mps.physical_basis == physical_basis
        assert not hasattr(chemistry_data, "MPSMetadata")

    def test_contract_normalized(self):
        """Test that contracted state vector is normalized."""
        rng = np.random.default_rng(42)
        mps = random_mps(num_sites=3, bond_dim=4, rng=rng)
        state = contract_mps(mps)
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_stores_sparse_physical_slices_and_materializes_one_site(self):
        """Dense input is stored as sparse A^p matrices with a reversible site view."""
        tensor = np.arange(24, dtype=float).reshape(2, 4, 3)
        mps = make_mps([tensor])

        assert isinstance(mps.sites[0], MPSSite)
        assert len(mps.sites[0].physical_slices) == 4
        assert all(issparse(value) for value in mps.sites[0].physical_slices)
        assert np.array_equal(mps.sites[0].to_dense(), tensor)

    def test_from_dense_preserves_complex_amplitudes(self):
        """Generic dense construction dispatches to complex storage when needed."""
        tensor = np.array([[[1.0j], [0.0], [0.0], [0.0]]])
        site = MPSSite.from_dense(tensor)

        assert site.is_complex
        assert np.array_equal(site.to_dense(), tensor)

    def test_validation_errors(self):
        """Test that invalid inputs raise ValueError."""
        with pytest.raises(ValueError, match="must contain at least one site"):
            AbelianMPSContainer([], create_test_orbitals(1))

        with pytest.raises(ValueError, match="incorrect number of dimensions"):
            MPSSite.from_dense(np.zeros((4, 4)))

    def test_bond_dim_consistency(self):
        """Test that inconsistent bond dimensions are caught."""
        t1 = np.zeros((1, 4, 3))
        t2 = np.zeros((2, 4, 1))  # chi_left=2 doesn't match t1's chi_right=3
        with pytest.raises(ValueError, match="incompatible bond spaces"):
            make_mps([t1, t2])


class TestDecomposition:
    """Test that CSD reconstructs the target matrix."""

    @pytest.mark.parametrize(
        ("chi_left", "chi_right", "seed"),
        [
            (2, 2, 42),
            (4, 2, 123),
            (2, 4, 456),
            (4, 4, 789),
        ],
    )
    def test_decomposition_factors_are_orthogonal(self, chi_left, chi_right, seed):
        """Verify every unitary factor in the 7-matrix decomposition is orthogonal."""
        rng = np.random.default_rng(seed)
        d = 4

        # Generate a random isometry
        raw = rng.standard_normal((d * chi_right, chi_left))
        q_mat, _ = np.linalg.qr(raw, mode="reduced")
        tensor = q_mat.reshape(d, chi_right, chi_left).transpose(2, 0, 1)

        ancilla_bits = int(np.ceil(np.log2(max(chi_left, d * chi_right))))
        ancilla_dim = 1 << ancilla_bits

        data = compute_site_unitary_dense_data(tensor, v_from_next=None, ancilla_dim=ancilla_dim)

        # Check that V is orthogonal
        v = data["v"]
        width = v.shape[0]
        assert np.allclose(v @ v.T, np.eye(width), atol=1e-10)

        # Check W matrices are orthogonal
        dim = ancilla_dim
        w_0 = data["w_0"]
        w_1 = data["w_1"]
        assert np.allclose(w_0 @ w_0.T, np.eye(dim), atol=1e-10)
        assert np.allclose(w_1 @ w_1.T, np.eye(dim), atol=1e-10)

        # Check U matrices are orthogonal
        for u in data["u"]:
            assert np.allclose(u @ u.T, np.eye(dim), atol=1e-10)

    def test_decompose_2d_correctness(self):
        """Verify decompose_2d produces valid decomposition."""
        rng = np.random.default_rng(42)
        dim = 4
        raw = rng.standard_normal((2 * dim, dim))
        q_mat, _ = np.linalg.qr(raw, mode="reduced")
        a = q_mat[:dim, :]
        b = q_mat[dim:, :]

        u_1, u_2, d_1, _d_2, v = decompose_2d(a, b)

        assert u_1.shape == (dim, dim)
        assert u_2.shape == (dim, dim)
        assert v.shape == (dim, dim)
        assert np.allclose(u_1 @ u_1.T, np.eye(dim), atol=1e-10)
        assert np.allclose(u_2 @ u_2.T, np.eye(dim), atol=1e-10)
        assert np.allclose(v @ v.T, np.eye(dim), atol=1e-10)
        assert np.allclose(a, u_1[:, :dim] * d_1 @ v, atol=1e-10)

    def test_givens_decomposition_reconstructs_unitary(self):
        """Verify Givens decomposition reconstructs the original unitary."""
        rng = np.random.default_rng(42)
        dim = 4
        raw = rng.standard_normal((dim, dim))
        q_mat, _ = np.linalg.qr(raw)

        layer_angles, layer_shifted, phases = decompose_unitary_to_givens(q_mat)

        # Reconstruct from layers using circuit convention:
        # Circuit applies layers[0] first, ..., layers[l-1], then D.
        # Resulting unitary = D · L_{l-1} · ... · L_0
        # Build by left-multiplying each layer in order, then D.
        reconstructed = np.eye(dim)
        for angles, shifted in zip(layer_angles, layer_shifted, strict=False):
            layer_mat = np.eye(dim)
            if shifted:
                pairs = [(2 * k + 1, 2 * k + 2) for k in range((dim - 1) // 2)]
            else:
                pairs = [(2 * k, 2 * k + 1) for k in range(dim // 2)]

            for slot_idx, (i, j) in enumerate(pairs):
                if slot_idx < len(angles) and abs(angles[slot_idx]) > 1e-15:
                    c = np.cos(angles[slot_idx])
                    s = np.sin(angles[slot_idx])
                    rot = np.eye(dim)
                    rot[i, i] = c
                    rot[i, j] = -s
                    rot[j, i] = s
                    rot[j, j] = c
                    layer_mat = layer_mat @ rot
            reconstructed = layer_mat @ reconstructed

        # Apply phase correction (D applied last = leftmost in product)
        phase_diag = np.diag([(-1.0 if p else 1.0) for p in phases[:dim]])
        reconstructed = phase_diag @ reconstructed

        assert np.allclose(np.abs(reconstructed - q_mat), 0, atol=1e-8) or np.allclose(
            np.abs(reconstructed + q_mat), 0, atol=1e-8
        )


class TestGenerateMPSPreparationData:
    """Test the full preprocessing pipeline for MPSSequential."""

    def test_two_site_data_structure(self):
        """Verify generate_mps_preparation_data returns correct structure for 2 sites."""
        rng = np.random.default_rng(42)
        mps = random_mps(num_sites=2, bond_dim=2, rng=rng)
        data = generate_mps_preparation_data(mps.sites)

        assert data.num_sites == 2
        assert data.ancilla_bits >= 1
        assert len(data.initial_state_vec) > 0
        # One site unitary (num_sites - 1)
        assert len(data.sites) == 1

    def test_three_site_data_structure(self):
        """Verify generate_mps_preparation_data returns correct structure for 3 sites."""
        rng = np.random.default_rng(42)
        mps = random_mps(num_sites=3, bond_dim=2, rng=rng)
        data = generate_mps_preparation_data(mps.sites)

        assert data.num_sites == 3
        assert len(data.sites) == 2

    def test_initial_state_normalized(self):
        """Verify the initial state vector is normalized."""
        rng = np.random.default_rng(42)
        mps = random_mps(num_sites=3, bond_dim=4, rng=rng)
        data = generate_mps_preparation_data(mps.sites)

        init_vec = np.array(data.initial_state_vec)
        assert abs(np.linalg.norm(init_vec) - 1.0) < 1e-10

    @pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
    def test_rejects_invalid_initial_state(self, value):
        """Invalid amplitudes cannot be serialized as a quantum state."""
        with pytest.raises(ValueError, match="finite amplitudes with nonzero norm"):
            generate_mps_preparation_data([np.full((1, 4, 1), value)])

    def test_rejects_complex_mps(self):
        """The real-valued synthesis path rejects complex tensors explicitly."""
        with pytest.raises(ValueError, match="only real-valued"):
            generate_mps_preparation_data([np.array([[[1.0j], [0.0], [0.0], [0.0]]])])


class TestMPSSequentialPublicApi:
    """Test public algorithm validation and circuit construction."""

    def test_run_constructs_composable_circuit(self):
        """Public run resolves both the estimator entry point and composable Q# operation."""
        mps = make_mps([np.array([[[1.0], [0.0], [0.0], [0.0]]])])
        circuit = MPSSequentialStatePreparation().run(mps)

        assert circuit._qsharp_op is not None

    def test_mps_and_base_callables_share_context(self):
        """Loading either utility namespace does not invalidate retained Q# callables."""
        from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: PLC0415

        base_callable = QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit
        mps_callable = QSHARP_UTILS.MPSSequential.MakeMPSSequentialCircuit
        _ = QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit

        assert base_callable.__dict__["_qdk_context"] is mps_callable.__dict__["_qdk_context"]

    @pytest.mark.parametrize("rotation_bits", [1, 63])
    def test_rejects_unsupported_rotation_precision(self, rotation_bits):
        """Settings reject precision values that Q# cannot execute safely."""
        algorithm = MPSSequentialStatePreparation()
        with pytest.raises(ValueError, match="out of allowed range"):
            algorithm.settings().update("rotation_bits", rotation_bits)


@pytest.mark.slow
class TestMPSSequentialFidelity:
    """Test that MPS Sequential state preparation produces high-fidelity states."""

    @pytest.mark.parametrize(
        ("num_sites", "bond_dim", "seed"),
        [
            (2, 4, 42),
            (3, 4, 42),
            (4, 2, 42),
        ],
    )
    def test_fidelity_vs_exact(self, num_sites, bond_dim, seed):
        """Test state preparation fidelity against the exact MPS state vector.

        Uses single-shot statevector simulation via DumpMachine() instead of
        statistical sampling (prohibitively slow due to the QROAM-based circuit
        allocating many internal ancillas per shot).
        """
        # Ensure MPS Q# project is loaded into the global interpreter
        get_qsharp_utils()

        rng = np.random.default_rng(seed)
        mps = random_mps(num_sites=num_sites, bond_dim=bond_dim, rng=rng)

        # Classical reference state
        target_state = contract_mps(mps)

        # Prepare gate-based data
        prep_data = generate_mps_preparation_data(mps.sites)
        params = prep_data.to_qsharp_params(rotation_bits=10)
        num_state_qubits = 2 * num_sites
        num_ancilla_qubits = prep_data.ancilla_bits

        qs_code = _build_mps_eval_code(params)
        qsharp.eval(f"use state = Qubit[{num_state_qubits}];")
        qsharp.eval(f"use ancilla = Qubit[{num_ancilla_qubits}];")
        qsharp.eval(qs_code)
        dump = qsharp.dump_machine()
        amplitudes = np.array(dump.as_dense_state(), dtype=complex)
        qsharp.eval("ResetAll(state + ancilla);")

        # Extract amplitudes where ancilla = |0⟩.
        # DumpMachine qubit ordering: state[0]...state[N-1], ancilla[0]...
        # (MSB-first in bit string). Ancilla qubits are the rightmost bits.
        num_total_qubits = num_state_qubits + num_ancilla_qubits
        dim = 2**num_total_qubits
        ancilla_mask = (1 << num_ancilla_qubits) - 1
        state_dim = 2**num_state_qubits
        state_amplitudes = np.zeros(state_dim, dtype=complex)
        for idx in range(dim):
            if (idx & ancilla_mask) == 0:
                state_idx = idx >> num_ancilla_qubits
                state_amplitudes[state_idx] = amplitudes[idx]

        # P(ancilla = |0⟩) — measures how well the ancilla is disentangled
        ancilla_zero_prob = np.sum(np.abs(state_amplitudes) ** 2)
        assert ancilla_zero_prob > 0.90, f"P(ancilla=0) = {ancilla_zero_prob:.4f} too low — ancilla not clean"

        # Normalize the post-selected state
        state_amplitudes = state_amplitudes / np.sqrt(ancilla_zero_prob)

        # Reindex: the Q# circuit uses little-endian ordering within each
        # 2-qubit site, so DumpMachine's big-endian bits need to be reversed
        # within each site to match the Python MPS convention.
        site_bits = 2  # qubits per site
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
        state_amplitudes = reordered

        # Compute quantum state fidelity |⟨target|prepared⟩|²
        fidelity = np.abs(np.dot(np.conj(state_amplitudes[: len(target_state)]), target_state)) ** 2

        # Should achieve high fidelity with rotation_bits=10
        assert fidelity > 0.95, f"Fidelity {fidelity:.4f} too low for num_sites={num_sites}, bond_dim={bond_dim}"


# Qualtran reference test data: MPS tensors and expected states

# These tensors and expected states are from the Qualtran MPSPreparation tests
# (Apache-2.0). They serve as regression fixtures for state preparation fidelity.

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

# Qualtran resource estimates (QROM mode) for cross-validation.
# From QubitCount and QECGatesCost (and_bloq + cswap).
QUALTRAN_COST_DENSE = {"num_qubits": 26, "toffoli": 600}
QUALTRAN_COST_SPARSE = {"num_qubits": 32, "toffoli": 321}

# Non-zero spin MPS tensors — a 4-site system with chi_left=3 (singlet embedding).
# From the Qualtran MPSPreparation tests (Apache-2.0).
_qualtran_mps_tensors_non_zero_spin = (
    np.array(
        [
            [
                [-0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.00316609, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054, 0.0, 0.0],
            ],
            [
                [0.0, 0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.00223876, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.00223876, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054, 0.0],
            ],
            [
                [0.0, 0.0, -0.00110206, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.00316609, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.57734054],
            ],
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.70710678, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.55872176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.82920795, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01562571, 0.0],
            ],
            [
                [0.0, -0.55872176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.82920795, -0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01562571],
            ],
            [
                [0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.70710678, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.70710678],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
    ),
    np.array(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [-0.99960484, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, 0.02810986],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, -0.70710678, 0.0, 0.0],
                [0.0, 0.0, -0.70710678, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, -0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, -0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
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

_qualtran_mps_expected_state_non_zero_spin = np.array(
    [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00110206,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00110206,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.00176896,  0.        ,  0.        ,  0.        ,
       -0.00125085,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00262431,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.00185567,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.00125085,  0.        ,  0.        ,  0.        ,  0.00176896,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.00185567,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.00262431,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.57734054,  0.        ,  0.        ,
        0.        , -0.40824141,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        , -0.40824141,  0.        ,
        0.        ,  0.        ,  0.57734054,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ])  # fmt: skip

# Qualtran resource estimates for the non-zero spin case (chi_left=3).
QUALTRAN_COST_NON_ZERO_SPIN_DENSE = {"num_qubits": 29, "toffoli": 734}
QUALTRAN_COST_NON_ZERO_SPIN_SPARSE = {"num_qubits": 29, "toffoli": 258}


class TestMPSSequentialQualtranFidelity:
    """Test that AbelianMPSContainer contraction matches Qualtran expected states."""

    @pytest.mark.parametrize(
        ("tensors", "expected_state"),
        [
            (_qualtran_mps_tensors, _qualtran_mps_expected_state),
            (_qualtran_mps_tensors_non_zero_spin, _qualtran_mps_expected_state_non_zero_spin),
        ],
        ids=["standard", "non_zero_spin"],
    )
    def test_contract_matches_expected_state(self, tensors, expected_state):
        """Verify MPS contraction produces the Qualtran expected state vector."""
        mps = make_mps(tensors)
        contracted = contract_mps(mps)
        # Use atol=1e-3: non-zero spin tensors are not perfectly canonical,
        # so raw contraction has O(1e-4) leakage. Fidelity is still > 0.9999.
        assert np.allclose(contracted, expected_state, atol=1e-3)


class TestMPSSequentialQualtranCostComparison:
    """Test that Q# resource estimates are consistent with Qualtran."""

    @pytest.mark.parametrize(
        ("tensors", "expected_num_sites", "min_ancilla_bits"),
        [
            (_qualtran_mps_tensors, 4, 3),
            (_qualtran_mps_tensors_non_zero_spin, 4, 3),
        ],
        ids=["standard", "non_zero_spin"],
    )
    def test_prepare_gate_based_data_produces_valid_output(self, tensors, expected_num_sites, min_ancilla_bits):
        """Verify generate_mps_preparation_data succeeds on Qualtran tensors."""
        mps = make_mps(tensors)
        data = generate_mps_preparation_data(mps.sites)

        assert data.num_sites == expected_num_sites
        assert data.ancilla_bits >= min_ancilla_bits
        # num_sites - 1 site unitaries
        assert len(data.sites) == expected_num_sites - 1

        # Initial state should be normalized
        init_vec = np.array(data.initial_state_vec)
        assert abs(np.linalg.norm(init_vec) - 1.0) < 1e-10

    @pytest.mark.parametrize(
        ("tensors", "qualtran_cost"),
        [
            (_qualtran_mps_tensors, QUALTRAN_COST_DENSE),
            (_qualtran_mps_tensors_non_zero_spin, QUALTRAN_COST_NON_ZERO_SPIN_DENSE),
        ],
        ids=["standard", "non_zero_spin"],
    )
    @pytest.mark.slow
    def test_resource_estimate_qubit_count(self, tensors, qualtran_cost):
        """Verify Q# resource estimate qubit count is consistent with Qualtran."""
        mps = make_mps(tensors)
        algo = MPSSequentialStatePreparation()
        circuit = algo.run(mps)
        result = circuit.estimate()
        counts = result.logical_counts

        # Q# estimate should use a comparable number of qubits to Qualtran dense mode
        assert counts["numQubits"] >= qualtran_cost["num_qubits"]
        assert counts["numQubits"] <= qualtran_cost["num_qubits"] * 2

    @pytest.mark.parametrize(
        ("tensors", "qualtran_cost"),
        [
            (_qualtran_mps_tensors, QUALTRAN_COST_SPARSE),
            (_qualtran_mps_tensors_non_zero_spin, QUALTRAN_COST_NON_ZERO_SPIN_SPARSE),
        ],
        ids=["standard", "non_zero_spin"],
    )
    @pytest.mark.slow
    def test_resource_estimate_toffoli_count(self, tensors, qualtran_cost):
        """Verify Q# resource estimate Toffoli count is consistent with Qualtran."""
        mps = make_mps(tensors)
        algo = MPSSequentialStatePreparation()
        circuit = algo.run(mps)
        result = circuit.estimate()
        counts = result.logical_counts

        # Q# logical_counts reports all CCZ gates including internal QROAM/Select
        # decompositions, so the count is larger than Qualtran's sparse Toffoli count.
        assert counts["cczCount"] > 0
        assert counts["cczCount"] <= qualtran_cost["toffoli"] * 10


@pytest.mark.slow
class TestMPSSequentialFastEstimation:
    """Test that fast resource estimation mode produces similar results to normal mode."""

    @pytest.mark.parametrize(
        "tensors",
        [_qualtran_mps_tensors, _qualtran_mps_tensors_non_zero_spin],
        ids=["standard", "non_zero_spin"],
    )
    def test_fast_vs_normal_resource_estimates(self, tensors):
        """Verify fast estimation produces similar qubit and Toffoli counts as normal mode."""
        mps = make_mps(tensors)

        # Normal mode (full CSD decomposition per site)
        algo_normal = MPSSequentialStatePreparation()
        circuit_normal = algo_normal.run(mps)
        result_normal = circuit_normal.estimate()
        counts_normal = result_normal.logical_counts

        # Fast mode (one representative per shape group)
        algo_fast = MPSSequentialStatePreparation()
        algo_fast.settings().update("fast_resource_estimation", True)
        circuit_fast = algo_fast.run(mps)
        result_fast = circuit_fast.estimate()
        counts_fast = result_fast.logical_counts

        # Qubit counts should be identical (same register layout)
        assert counts_fast["numQubits"] == counts_normal["numQubits"]

        # Toffoli counts should be close — fast mode uses dummy angles of the
        # correct array dimensions, so QROAM table sizes and rotation counts match.
        # Allow up to 30% deviation since shape grouping may merge sites with
        # slightly different effective dimensions.
        normal_ccz = counts_normal["cczCount"]
        fast_ccz = counts_fast["cczCount"]
        assert fast_ccz > 0
        ratio = fast_ccz / normal_ccz
        assert 0.9 <= ratio <= 1.1, (
            f"Fast/normal CCZ ratio {ratio:.3f} outside [0.9, 1.1]: fast={fast_ccz}, normal={normal_ccz}"
        )

    @pytest.mark.parametrize(
        ("num_sites", "bond_dim", "seed"),
        [
            (3, 2, 42),
            (3, 4, 99),
            (4, 2, 7),
        ],
    )
    def test_fast_vs_normal_small_random_mps(self, num_sites, bond_dim, seed):
        """Verify fast estimation agrees with normal mode on small random MPS circuits."""
        rng = np.random.default_rng(seed)
        mps = random_mps(num_sites=num_sites, bond_dim=bond_dim, rng=rng)

        algo_normal = MPSSequentialStatePreparation()
        circuit_normal = algo_normal.run(mps)
        result_normal = circuit_normal.estimate()
        counts_normal = result_normal.logical_counts

        algo_fast = MPSSequentialStatePreparation()
        algo_fast.settings().update("fast_resource_estimation", True)
        circuit_fast = algo_fast.run(mps)
        result_fast = circuit_fast.estimate()
        counts_fast = result_fast.logical_counts

        assert counts_fast["numQubits"] == counts_normal["numQubits"]

        normal_ccz = counts_normal["cczCount"]
        fast_ccz = counts_fast["cczCount"]
        assert fast_ccz > 0
        ratio = fast_ccz / normal_ccz
        assert 0.9 <= ratio <= 1.1, (
            f"Fast/normal CCZ ratio {ratio:.3f} outside [0.9, 1.1]: fast={fast_ccz}, normal={normal_ccz}"
        )


# =============================================================================
# Helper functions for Q# literal generation
# =============================================================================


def _float_to_qsharp(x: float) -> str:
    """Format float for Q#."""
    return f"{x:.15f}"


def _build_mps_eval_code(params: dict) -> str:
    """Build Q# eval code for MPSSequential from the params dict.

    Assumes `state` and `ancilla` qubit registers are already allocated in scope.
    """
    initial_state_str = ", ".join(_float_to_qsharp(x) for x in params["initialStateVec"])
    args = [
        f"[{initial_state_str}]",
        str(params["numSites"]),
        str(params["rotationBits"]),
        _nested_list_to_qsharp_3d(params["siteVLayerAngles"]),
        _nested_list_to_qsharp_2d_bool(params["siteVLayerShifted"]),
        _nested_list_to_qsharp_2d_bool(params["siteVPhases"]),
        _nested_list_to_qsharp_2d_float(params["siteRot0Angles"]),
        _nested_list_to_qsharp_2d_float(params["siteRot1Angles"]),
        _nested_list_to_qsharp_2d_float(params["siteRot2Angles"]),
        _nested_list_to_qsharp_3d(params["siteW0LayerAngles"]),
        _nested_list_to_qsharp_2d_bool(params["siteW0LayerShifted"]),
        _nested_list_to_qsharp_2d_bool(params["siteW0Phases"]),
        _nested_list_to_qsharp_3d(params["siteW1LayerAngles"]),
        _nested_list_to_qsharp_2d_bool(params["siteW1LayerShifted"]),
        _nested_list_to_qsharp_2d_bool(params["siteW1Phases"]),
        _nested_list_to_qsharp_3d(params["siteULayerAngles"]),
        _nested_list_to_qsharp_2d_bool(params["siteULayerShifted"]),
        _nested_list_to_qsharp_2d_bool(params["siteUPhases"]),
        "state",
        "ancilla",
    ]
    args_str = ",\n                ".join(args)
    return f"MPSSequential.MPSSequential(\n                {args_str}\n    )"


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


def _nested_list_to_qsharp_2d_float(data: list) -> str:
    """Convert list[list[float]] to Q# literal."""
    site_strs = []
    for site in data:
        vals = ", ".join(_float_to_qsharp(v) for v in site)
        site_strs.append(f"[{vals}]")
    return f"[{', '.join(site_strs)}]"


def _nested_list_to_qsharp_2d_bool(data: list) -> str:
    """Convert list[list[bool]] to Q# literal."""
    site_strs = []
    for site in data:
        vals = ", ".join("true" if b else "false" for b in site)
        site_strs.append(f"[{vals}]")
    return f"[{', '.join(site_strs)}]"
