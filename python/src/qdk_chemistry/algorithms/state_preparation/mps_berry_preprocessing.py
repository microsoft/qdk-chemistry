"""Berry CSD preprocessing for MPS state preparation.

Computes the Berry et al. (arXiv:2409.11748, Appendix B) 7-matrix CSD decomposition
of MPS site unitaries, then decomposes each unitary into Givens rotation layers
for circuit synthesis.

Attribution
-----------
This module is based on code originally published by Felix Rupprecht (DLR) on Zenodo
(https://zenodo.org/records/15587498). The implementation has been rewritten and adapted
for integration into the QDK Chemistry library.

References
----------
- Berry, Tong, et al. arXiv:2409.11748 (Eq. 30, Appendix B)
- Rupprecht & Wölk (2025), Zenodo: https://zenodo.org/records/15587498
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def decompose_2d(a: np.ndarray, b: np.ndarray):
    """Decompose [a; b] (orthonormal columns) as in Berry et al. Eq. 30.

    Returns (u_1, u_2, d_1, d_2, v).
    """
    u_1, d_1, vt = np.linalg.svd(a, full_matrices=True)
    v = vt

    bv = b @ vt.conj().T
    w, s, vt2 = np.linalg.svd(bv, full_matrices=True)
    width = a.shape[1]
    u_2 = w.copy()
    u_2[:width, :width] = w[:width, :width] @ vt2
    d_2_matrix = (vt2.T.conj() * s) @ vt2
    d_2 = np.diag(d_2_matrix).real

    return u_1, u_2, d_1, d_2, v


def _pad_to_power_of_2(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad array with zeros to target length."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate([arr, np.zeros(target_len - len(arr))])


def compute_site_unitary_dense_data(
    tensor: np.ndarray,
    v_from_next: np.ndarray | None,
    ancilla_dim: int,
) -> dict:
    """Compute the Berry 7-matrix CSD decomposition for one MPS site.

    Parameters
    ----------
    tensor : np.ndarray of shape (left, d, right)
        The MPS tensor for this site (d=4 for two-qubit sites).
    v_from_next : np.ndarray or None
        The V matrix from the next site's decomposition.
    ancilla_dim : int
        The ancilla register dimension (power of 2).

    Returns
    -------
    dict with keys: 'u', 'd_prime', 'w_0', 'w_1', 'v', 'ancilla_dim'
    """
    left, site_dim, right = tensor.shape
    dim = ancilla_dim
    full_dim = site_dim * dim

    target = tensor.transpose(1, 2, 0)  # (d, right, left)
    if v_from_next is not None:
        target = np.einsum("ij,djk->dik", v_from_next, target)

    padded = np.pad(target, ((0, 0), (0, dim - target.shape[1]), (0, 0)))
    matrix = padded.reshape(full_dim, left)

    # Berry decomposition via QR + CSD
    b_full, r = np.linalg.qr(matrix[dim:, :], mode="complete")
    c, s = np.linalg.qr(b_full[dim:, :dim], mode="complete")

    u_2, u_3, d_2, d_2_, v__ = decompose_2d(c[:dim, :dim], c[dim:, :dim])
    u_1, u_dummy, d_1, d_1_, v_ = decompose_2d(b_full[:dim, :dim], s[:dim, :])
    u_0, u_top, d_0, d_0_, v = decompose_2d(matrix[:dim, :], r[:dim, :])

    d_0_ = _pad_to_power_of_2(np.asarray(d_0_).real, dim)
    d_1_ = np.asarray(d_1_).real
    d_2_ = np.asarray(d_2_).real

    w_0 = v_ @ u_top
    w_1 = v__ @ u_dummy

    return {
        "u": (u_0, u_1, u_2, u_3),
        "d_prime": (d_0_, d_1_, d_2_),
        "w_0": w_0,
        "w_1": w_1,
        "v": v,
        "ancilla_dim": dim,
    }


def _decompose_unitary_to_givens_python(matrix: np.ndarray):
    """Decompose a real unitary into Givens rotation layers (pure Python fallback).

    Uses column-by-column QR elimination, then organizes into parallel layers.

    Parameters
    ----------
    matrix : np.ndarray
        Real orthogonal matrix (dim x dim).

    Returns
    -------
    layer_angles : list[list[float]]
    layer_shifted : list[bool]
    phases : list[bool]
    """
    dim = matrix.shape[0]
    m = matrix.copy().astype(float)

    rotations: list[tuple[int, float]] = []

    for col in range(dim - 1):
        for row in range(dim - 1, col, -1):
            a_val = m[row - 1, col]
            b_val = m[row, col]
            if abs(b_val) < 1e-15:
                continue
            theta = np.arctan2(b_val, a_val)
            c, s = np.cos(theta), np.sin(theta)
            row_i = m[row - 1, :].copy()
            row_j = m[row, :].copy()
            m[row - 1, :] = c * row_i + s * row_j
            m[row, :] = -s * row_i + c * row_j
            rotations.append((row - 1, theta))

    diag_vals = np.diag(m)
    phases = [bool(d < 0) for d in diag_vals]

    # Organize into parallel layers
    layer_angles, layer_shifted = _organize_into_layers(rotations, dim)
    return layer_angles, layer_shifted, phases


def _organize_into_layers(
    rotations: list[tuple[int, float]], dim: int
) -> tuple[list[list[float]], list[bool]]:
    """Organize Givens rotations into parallel layers."""
    if not rotations:
        return [], []

    num_even_slots = dim // 2
    num_odd_slots = (dim - 1) // 2

    pair_latest: list[int] = [-1] * (dim - 1)
    layers_slots: list[set] = []
    assignments: list[int] = []

    for pair, _angle in rotations:
        is_odd = pair % 2 == 1
        min_layer = 0
        for p in range(max(0, pair - 1), min(dim - 1, pair + 2)):
            if pair_latest[p] >= 0:
                min_layer = max(min_layer, pair_latest[p] + 1)

        if is_odd and min_layer % 2 == 0:
            min_layer += 1
        elif not is_odd and min_layer % 2 == 1:
            min_layer += 1

        layer_idx = min_layer
        while True:
            while layer_idx >= len(layers_slots):
                layers_slots.append(set())
            if pair not in layers_slots[layer_idx]:
                break
            layer_idx += 2

        while layer_idx >= len(layers_slots):
            layers_slots.append(set())

        layers_slots[layer_idx].add(pair)
        assignments.append(layer_idx)
        pair_latest[pair] = layer_idx

    num_layers = len(layers_slots)
    layer_rotations: list[list[tuple[int, float]]] = [[] for _ in range(num_layers)]
    for idx, (pair, angle) in enumerate(rotations):
        layer_rotations[assignments[idx]].append((pair, angle))

    result_angles: list[list[float]] = []
    result_shifted: list[bool] = []

    for layer_idx in range(num_layers):
        if not layer_rotations[layer_idx]:
            continue

        shifted = layer_idx % 2 == 1
        if shifted:
            num_slots = num_odd_slots
            slot_pairs = [2 * k + 1 for k in range(num_slots)]
        else:
            num_slots = num_even_slots
            slot_pairs = [2 * k for k in range(num_slots)]

        pair_to_slot = {p: i for i, p in enumerate(slot_pairs)}
        angles = [0.0] * num_slots

        for pair, angle in layer_rotations[layer_idx]:
            slot = pair_to_slot[pair]
            angles[slot] = angle

        result_angles.append(angles)
        result_shifted.append(shifted)

    return result_angles, result_shifted


def decompose_unitary_to_givens(matrix: np.ndarray):
    """Decompose a real unitary matrix into Givens rotation layers.

    Attempts to use the Rust-based decomposition if available, falls back
    to a pure Python implementation.

    Parameters
    ----------
    matrix : np.ndarray
        Real orthogonal matrix (dim x dim), dim must be a power of 2.

    Returns
    -------
    layer_angles : list[list[float]]
    layer_shifted : list[bool]
    phases : list[bool]
    """
    try:
        from unitary_synthesis._givens_decomposition import decompose_real  # noqa: PLC0415

        dim = matrix.shape[0]
        phases_raw, layers = decompose_real([matrix.astype(np.float64)])

        layer_angles = []
        layer_shifted = []
        for layer in layers:
            layer_angles.append(list(layer.angles_ry))
            layer_shifted.append(layer.shifted)

        phases = [bool(p) for p in phases_raw]
        phases += [False] * (dim - len(phases))
        return layer_angles, layer_shifted, phases
    except ImportError:
        return _decompose_unitary_to_givens_python(matrix)


def decompose_block_diagonal_to_givens(blocks: list[np.ndarray]):
    """Decompose a block-diagonal real unitary into merged Givens rotation layers.

    Parameters
    ----------
    blocks : list[np.ndarray]
        List of real orthogonal matrices.

    Returns
    -------
    layer_angles : list[list[float]]
    layer_shifted : list[bool]
    phases : list[bool]
    """
    try:
        from unitary_synthesis._givens_decomposition import decompose_real  # noqa: PLC0415

        block_mats = tuple(b.astype(np.float64) for b in blocks)
        total_dim = sum(b.shape[0] for b in blocks)
        phases_raw, layers = decompose_real(block_mats)

        layer_angles = []
        layer_shifted = []
        for layer in layers:
            layer_angles.append(list(layer.angles_ry))
            layer_shifted.append(layer.shifted)

        phases = [bool(p) for p in phases_raw]
        phases += [False] * (total_dim - len(phases))
        return layer_angles, layer_shifted, phases
    except ImportError:
        # Fallback: decompose full block-diagonal matrix
        from scipy.linalg import block_diag as scipy_block_diag  # noqa: PLC0415

        full_matrix = scipy_block_diag(*[b.astype(np.float64) for b in blocks])
        return _decompose_unitary_to_givens_python(full_matrix)


def prepare_gate_based_data(tensors: Sequence[np.ndarray]) -> dict:
    """Compute all data needed for MPSPreparationBerry Q# operation.

    Performs CSD decomposition + Givens layer decomposition for each site.
    Returns raw angles (Double) and phases (Bool) — Q# handles angle
    quantization internally.

    Parameters
    ----------
    tensors : sequence of np.ndarray
        MPS tensors. tensors[i] has shape (chi_left, d, chi_right).

    Returns
    -------
    dict with all parameters needed by MPSPreparationBerry.
    """
    num_sites = len(tensors)
    d = tensors[0].shape[1]

    # Determine consistent ancilla size across all sites
    max_ancilla_dim = 1
    for i in range(1, num_sites):
        chi_left, _, chi_right = tensors[i].shape
        local_bits = int(np.ceil(np.log2(max(chi_left, chi_right)))) if max(chi_left, chi_right) > 1 else 1
        max_ancilla_dim = max(max_ancilla_dim, 1 << local_bits)
    chi_1 = tensors[0].shape[2]
    init_bits = int(np.ceil(np.log2(max(1, chi_1)))) if chi_1 > 1 else 1
    max_ancilla_dim = max(max_ancilla_dim, 1 << init_bits)
    ancilla_bits = int(np.ceil(np.log2(max_ancilla_dim))) if max_ancilla_dim > 1 else 1
    ancilla_dim = 1 << ancilla_bits

    # Per-site decomposition data
    site_v_layer_angles = []
    site_v_layer_shifted = []
    site_v_phases = []
    site_rot0_angles = []
    site_rot1_angles = []
    site_rot2_angles = []
    site_w0_layer_angles = []
    site_w0_layer_shifted = []
    site_w0_phases = []
    site_w1_layer_angles = []
    site_w1_layer_shifted = []
    site_w1_phases = []
    site_u_layer_angles = []
    site_u_layer_shifted = []
    site_u_phases = []

    for i in range(1, num_sites):
        tensor = tensors[i]
        chi_left = tensor.shape[0]
        dim = ancilla_dim

        data = compute_site_unitary_dense_data(tensor, v_from_next=None, ancilla_dim=dim)
        d_0_, d_1_, d_2_ = data["d_prime"]
        w_0 = data["w_0"]
        w_1 = data["w_1"]
        u_0, u_1, u_2, u_3 = data["u"]
        v = data["v"]

        # V: Givens decomposition
        v_pad = np.eye(dim, dtype=np.float64)
        v_pad[: v.shape[0], : v.shape[1]] = np.asarray(v).real
        v_angles, v_shifted, v_phases_arr = decompose_unitary_to_givens(v_pad)
        site_v_layer_angles.append(v_angles)
        site_v_layer_shifted.append(v_shifted)
        site_v_phases.append(v_phases_arr)

        # UCR rotation angles: 2*arcsin(d')
        rot0_angles = [2.0 * float(np.arcsin(np.clip(d_0_[k], -1, 1))) for k in range(dim)]
        rot1_angles = [
            2.0 * float(np.arcsin(np.clip(d_1_[k] if k < len(d_1_) else 0.0, -1, 1)))
            for k in range(dim)
        ]
        rot2_angles = [
            2.0 * float(np.arcsin(np.clip(d_2_[k] if k < len(d_2_) else 0.0, -1, 1)))
            for k in range(dim)
        ]
        site_rot0_angles.append(rot0_angles)
        site_rot1_angles.append(rot1_angles)
        site_rot2_angles.append(rot2_angles)

        # W0: Givens decomposition
        w0_pad = np.eye(dim, dtype=np.float64)
        w0_pad[: w_0.shape[0], : w_0.shape[1]] = np.asarray(w_0).real
        w0_angles, w0_shifted, w0_phases = decompose_unitary_to_givens(w0_pad)
        site_w0_layer_angles.append(w0_angles)
        site_w0_layer_shifted.append(w0_shifted)
        site_w0_phases.append(w0_phases)

        # W1: Givens decomposition
        w1_pad = np.eye(dim, dtype=np.float64)
        w1_pad[: w_1.shape[0], : w_1.shape[1]] = np.asarray(w_1).real
        w1_angles, w1_shifted, w1_phases = decompose_unitary_to_givens(w1_pad)
        site_w1_layer_angles.append(w1_angles)
        site_w1_layer_shifted.append(w1_shifted)
        site_w1_phases.append(w1_phases)

        # U (block-diagonal): decompose blocks
        u_blocks = [u_0, u_1, u_2, u_3]
        u_block_mats = []
        for u_b in u_blocks:
            u_block_pad = np.eye(dim, dtype=np.float64)
            u_block_pad[: u_b.shape[0], : u_b.shape[1]] = np.asarray(u_b).real
            u_block_mats.append(u_block_pad)
        u_angles, u_shifted, u_phases_arr = decompose_block_diagonal_to_givens(u_block_mats)
        site_u_layer_angles.append(u_angles)
        site_u_layer_shifted.append(u_shifted)
        site_u_phases.append(u_phases_arr)

    # Initial state from first tensor
    first_tensor = tensors[0]
    chi_1 = first_tensor.shape[2]
    init_state = first_tensor[0]  # (d, chi_right)
    init_padded = np.zeros((d, ancilla_dim))
    init_padded[:, :chi_1] = init_state
    initial_state_vec = init_padded.flatten()
    norm = np.linalg.norm(initial_state_vec)
    if norm > 1e-15:
        initial_state_vec = initial_state_vec / norm

    return {
        "initial_state_vec": initial_state_vec.tolist(),
        "num_sites": num_sites,
        "ancilla_bits": ancilla_bits,
        "site_v_layer_angles": site_v_layer_angles,
        "site_v_layer_shifted": site_v_layer_shifted,
        "site_v_phases": site_v_phases,
        "site_rot0_angles": site_rot0_angles,
        "site_rot1_angles": site_rot1_angles,
        "site_rot2_angles": site_rot2_angles,
        "site_w0_layer_angles": site_w0_layer_angles,
        "site_w0_layer_shifted": site_w0_layer_shifted,
        "site_w0_phases": site_w0_phases,
        "site_w1_layer_angles": site_w1_layer_angles,
        "site_w1_layer_shifted": site_w1_layer_shifted,
        "site_w1_phases": site_w1_phases,
        "site_u_layer_angles": site_u_layer_angles,
        "site_u_layer_shifted": site_u_layer_shifted,
        "site_u_phases": site_u_phases,
    }
