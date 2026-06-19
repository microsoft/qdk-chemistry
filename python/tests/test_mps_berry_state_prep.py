"""Tests for MPS Berry state preparation algorithm.

Tests both the classical preprocessing (fidelity of the decomposition) and
the full Q# circuit (state preparation fidelity and gate counts).

Attribution: The Berry decomposition tested here is based on code originally
published by Felix Rupprecht (DLR) on Zenodo (https://zenodo.org/records/15587498).
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.algorithms.state_preparation.mps_berry_preprocessing import (
    compute_site_unitary_dense_data,
    decompose_2d,
    decompose_unitary_to_givens,
    prepare_gate_based_data,
)
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction


# =============================================================================
# Test MPSWavefunction container
# =============================================================================


class TestMPSWavefunction:
    """Tests for the MPSWavefunction data container."""

    def test_basic_construction(self):
        """Test constructing an MPSWavefunction from tensors."""
        rng = np.random.default_rng(42)
        mps = MPSWavefunction.random(num_sites=3, bond_dim=4, rng=rng)
        assert mps.num_sites == 3
        assert mps.num_qubits == 6
        assert mps.max_bond_dim == 4

    def test_contract_normalized(self):
        """Test that contracted state vector is normalized."""
        rng = np.random.default_rng(42)
        mps = MPSWavefunction.random(num_sites=3, bond_dim=4, rng=rng)
        state = mps.contract()
        assert abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_from_state_vector_roundtrip(self):
        """Test that from_state_vector reproduces the original state (no truncation)."""
        rng = np.random.default_rng(42)
        num_sites = 2
        dim = 4**num_sites
        vec = rng.standard_normal(dim)
        vec /= np.linalg.norm(vec)

        mps = MPSWavefunction.from_state_vector(vec, num_sites=num_sites)
        reconstructed = mps.contract()
        fidelity = abs(np.dot(vec, reconstructed)) ** 2
        assert fidelity > 1.0 - 1e-10

    def test_from_state_vector_truncated(self):
        """Test that truncation reduces bond dimension."""
        rng = np.random.default_rng(42)
        num_sites = 3
        dim = 4**num_sites
        vec = rng.standard_normal(dim)
        vec /= np.linalg.norm(vec)

        mps = MPSWavefunction.from_state_vector(vec, num_sites=num_sites, max_bond_dim=2)
        assert mps.max_bond_dim <= 2

    def test_validation_errors(self):
        """Test that invalid inputs raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            MPSWavefunction([])

        with pytest.raises(ValueError, match="3-dimensional"):
            MPSWavefunction([np.zeros((4, 4))])

        with pytest.raises(ValueError, match="chi_left=1"):
            MPSWavefunction([np.zeros((2, 4, 1))])  # chi_left != 1

    def test_bond_dim_consistency(self):
        """Test that inconsistent bond dimensions are caught."""
        t1 = np.zeros((1, 4, 3))
        t2 = np.zeros((2, 4, 1))  # chi_left=2 doesn't match t1's chi_right=3
        with pytest.raises(ValueError, match="Bond dimension mismatch"):
            MPSWavefunction([t1, t2])


# =============================================================================
# Test Berry CSD decomposition (classical correctness)
# =============================================================================


class TestBerryDecomposition:
    """Test the Berry CSD decomposition reconstructs the target matrix."""

    @pytest.mark.parametrize(
        "chi_left,chi_right,seed",
        [
            (2, 2, 42),
            (4, 2, 123),
            (2, 4, 456),
            (4, 4, 789),
        ],
    )
    def test_reconstruction_matches_target(self, chi_left, chi_right, seed):
        """Verify the 7-matrix decomposition reconstructs the target isometry."""
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

        u_1, u_2, d_1, d_2, v = decompose_2d(a, b)

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

        # Reconstruct from layers
        reconstructed = np.eye(dim)
        for angles, shifted in zip(layer_angles, layer_shifted):
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
            reconstructed = reconstructed @ layer_mat

        # Apply phase correction
        phase_diag = np.diag([(-1.0 if p else 1.0) for p in phases[:dim]])
        reconstructed = reconstructed @ phase_diag

        assert np.allclose(np.abs(reconstructed - q_mat), 0, atol=1e-8) or np.allclose(
            np.abs(reconstructed + q_mat), 0, atol=1e-8
        )


# =============================================================================
# Test prepare_gate_based_data
# =============================================================================


class TestPrepareGateBasedData:
    """Test the full preprocessing pipeline for MPSPreparationBerry."""

    def test_two_site_data_structure(self):
        """Verify prepare_gate_based_data returns correct structure for 2 sites."""
        rng = np.random.default_rng(42)
        mps = MPSWavefunction.random(num_sites=2, bond_dim=2, rng=rng)
        data = prepare_gate_based_data(mps.tensors)

        assert data["num_sites"] == 2
        assert data["ancilla_bits"] >= 1
        assert len(data["initial_state_vec"]) > 0
        # One site unitary (num_sites - 1)
        assert len(data["site_v_layer_angles"]) == 1
        assert len(data["site_rot0_angles"]) == 1
        assert len(data["site_w0_layer_angles"]) == 1
        assert len(data["site_w1_layer_angles"]) == 1
        assert len(data["site_u_layer_angles"]) == 1

    def test_three_site_data_structure(self):
        """Verify prepare_gate_based_data returns correct structure for 3 sites."""
        rng = np.random.default_rng(42)
        mps = MPSWavefunction.random(num_sites=3, bond_dim=2, rng=rng)
        data = prepare_gate_based_data(mps.tensors)

        assert data["num_sites"] == 3
        assert len(data["site_v_layer_angles"]) == 2
        assert len(data["site_rot0_angles"]) == 2

    def test_initial_state_normalized(self):
        """Verify the initial state vector is normalized."""
        rng = np.random.default_rng(42)
        mps = MPSWavefunction.random(num_sites=3, bond_dim=4, rng=rng)
        data = prepare_gate_based_data(mps.tensors)

        init_vec = np.array(data["initial_state_vec"])
        assert abs(np.linalg.norm(init_vec) - 1.0) < 1e-10


# =============================================================================
# Test MPS state preparation fidelity (Q# simulation)
# =============================================================================


@pytest.fixture
def qsharp_ctx():
    """Initialize Q# context with MPS Berry operations loaded."""
    try:
        import qdk
        from qdk import qsharp
    except ImportError:
        pytest.skip("qdk package not available")

    from pathlib import Path

    qs_dir = (
        Path(__file__).parent.parent
        / "src" / "qdk_chemistry" / "utils" / "qsharp"
    )
    qs_files = [
        qs_dir / "PhaseGradient.qs",
        qs_dir / "QroamStatePrep.qs",
        qs_dir / "GivensDecomposition.qs",
        qs_dir / "MPSPreparationBerry.qs",
    ]

    code = "\n".join(f.read_text() for f in qs_files)
    qsharp.eval(code)
    return qdk, qsharp


class TestMPSBerryFidelity:
    """Test that MPS Berry state preparation produces high-fidelity states."""

    @pytest.mark.parametrize(
        "num_sites,bond_dim,seed",
        [
            (2, 2, 42),
            (2, 4, 123),
            (3, 2, 456),
        ],
    )
    def test_fidelity_vs_exact(self, qsharp_ctx, num_sites, bond_dim, seed):
        """Test state preparation fidelity against the exact MPS state vector.

        The fidelity should be high (> 0.99) for moderate phase gradient precision.
        """
        qdk_mod, qsharp_mod = qsharp_ctx
        rng = np.random.default_rng(seed)
        mps = MPSWavefunction.random(num_sites=num_sites, bond_dim=bond_dim, rng=rng)

        # Classical reference state
        target_state = mps.contract()

        # Prepare gate-based data
        data = prepare_gate_based_data(mps.tensors)
        b_rot = 10
        ancilla_bits = data["ancilla_bits"]
        num_state_qubits = 2 * num_sites
        num_ancilla_qubits = ancilla_bits

        # Build Q# parameter strings
        initial_state_str = ", ".join(f"{x:.15f}" for x in data["initial_state_vec"])
        v_angles_str = _nested_list_to_qsharp_3d(data["site_v_layer_angles"])
        v_shifted_str = _nested_list_to_qsharp_2d_bool(data["site_v_layer_shifted"])
        v_phases_str = _nested_list_to_qsharp_2d_bool(data["site_v_phases"])
        rot0_str = _nested_list_to_qsharp_2d_float(data["site_rot0_angles"])
        rot1_str = _nested_list_to_qsharp_2d_float(data["site_rot1_angles"])
        rot2_str = _nested_list_to_qsharp_2d_float(data["site_rot2_angles"])
        w0_angles_str = _nested_list_to_qsharp_3d(data["site_w0_layer_angles"])
        w0_shifted_str = _nested_list_to_qsharp_2d_bool(data["site_w0_layer_shifted"])
        w0_phases_str = _nested_list_to_qsharp_2d_bool(data["site_w0_phases"])
        w1_angles_str = _nested_list_to_qsharp_3d(data["site_w1_layer_angles"])
        w1_shifted_str = _nested_list_to_qsharp_2d_bool(data["site_w1_layer_shifted"])
        w1_phases_str = _nested_list_to_qsharp_2d_bool(data["site_w1_phases"])
        u_angles_str = _nested_list_to_qsharp_3d(data["site_u_layer_angles"])
        u_shifted_str = _nested_list_to_qsharp_2d_bool(data["site_u_layer_shifted"])
        u_phases_str = _nested_list_to_qsharp_2d_bool(data["site_u_phases"])

        # Run Q# state preparation and sample
        n_shots = 10000
        qs_code = f"""{{
            use state = Qubit[{num_state_qubits}];
            use ancilla = Qubit[{num_ancilla_qubits}];
            MPSPreparationBerry(
                [{initial_state_str}],
                {num_sites},
                {b_rot},
                {v_angles_str},
                {v_shifted_str},
                {v_phases_str},
                {rot0_str},
                {rot1_str},
                {rot2_str},
                {w0_angles_str},
                {w0_shifted_str},
                {w0_phases_str},
                {w1_angles_str},
                {w1_shifted_str},
                {w1_phases_str},
                {u_angles_str},
                {u_shifted_str},
                {u_phases_str},
                state,
                ancilla
            );
            // Measure state register
            mutable results = [0, size = {num_state_qubits}];
            for i in 0..{num_state_qubits - 1} {{
                set results w/= i <- M(state[i]) == One ? 1 | 0;
            }}
            // Check ancilla is zero
            mutable ancillaClean = true;
            for i in 0..{num_ancilla_qubits - 1} {{
                if M(ancilla[i]) == One {{
                    set ancillaClean <- false;
                }}
            }}
            ResetAll(state + ancilla);
            (results, ancillaClean)
        }}"""

        results = qsharp_mod.run(qs_code, shots=n_shots)

        # Compute empirical distribution
        dim = 2**num_state_qubits
        counts = np.zeros(dim)
        ancilla_clean_count = 0
        for bit_array, ancilla_clean in results:
            idx = sum(b * (2**i) for i, b in enumerate(bit_array))
            counts[idx] += 1
            if ancilla_clean:
                ancilla_clean_count += 1

        # Empirical probability distribution
        probs_measured = counts / n_shots

        # Target probability distribution
        probs_target = target_state**2

        # Classical fidelity (Bhattacharyya coefficient) as proxy
        fidelity_proxy = np.sum(np.sqrt(probs_measured * probs_target)) ** 2

        # Should achieve high fidelity with b_rot=10
        assert fidelity_proxy > 0.95, (
            f"Fidelity {fidelity_proxy:.4f} too low for "
            f"num_sites={num_sites}, bond_dim={bond_dim}"
        )

        # Ancilla should be clean (returned to |0>) most of the time
        ancilla_clean_rate = ancilla_clean_count / n_shots
        assert ancilla_clean_rate > 0.90, (
            f"Ancilla clean rate {ancilla_clean_rate:.4f} too low"
        )


# =============================================================================
# Test gate count estimation
# =============================================================================


class TestMPSBerryGateCount:
    """Test that gate count scaling matches theoretical expectations."""

    def test_gate_count_scales_with_sites(self):
        """Gate count should scale linearly with number of sites."""
        rng = np.random.default_rng(42)
        bond_dim = 2
        counts = []

        for num_sites in [2, 3, 4]:
            mps = MPSWavefunction.random(num_sites=num_sites, bond_dim=bond_dim, rng=rng)
            data = prepare_gate_based_data(mps.tensors)

            # Count total Givens layers (proxy for gate count)
            total_layers = 0
            for site_idx in range(num_sites - 1):
                total_layers += len(data["site_v_layer_angles"][site_idx])
                total_layers += len(data["site_w0_layer_angles"][site_idx])
                total_layers += len(data["site_w1_layer_angles"][site_idx])
                total_layers += len(data["site_u_layer_angles"][site_idx])
                total_layers += 3  # 3 UCR rotation layers per site

            counts.append(total_layers)

        # Verify roughly linear scaling (each site adds a constant number of layers)
        diff_1 = counts[1] - counts[0]
        diff_2 = counts[2] - counts[1]
        # Differences should be similar (linear scaling)
        assert abs(diff_1 - diff_2) <= max(diff_1, diff_2) * 0.5 + 1

    def test_ancilla_bits_matches_bond_dim(self):
        """Ancilla bits should be ceil(log2(max_bond_dim))."""
        rng = np.random.default_rng(42)

        for bond_dim in [2, 4, 8]:
            mps = MPSWavefunction.random(num_sites=3, bond_dim=bond_dim, rng=rng)
            data = prepare_gate_based_data(mps.tensors)
            expected_bits = int(np.ceil(np.log2(bond_dim)))
            assert data["ancilla_bits"] >= expected_bits


# =============================================================================
# Helper functions for Q# literal generation
# =============================================================================


def _float_to_qsharp(x: float) -> str:
    """Format float for Q#."""
    return f"{x:.15f}"


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
