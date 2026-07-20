"""Matrix Product State (MPS) state preparation exploiting block sparsity.

Implements the sparse MPS preparation method from :cite:`Rupprecht2026`.
Each site unitary is decomposed as ``U = P_row · V_blockdiag · P_col``
where ``P_row``, ``P_col`` are permutations (implemented via QROAM + SWAP +
X-measure) and ``V_blockdiag`` is block-diagonal (synthesized via Givens
rotation layers per block). This exploits U(1) symmetries (particle number,
spin) that make MPS tensors block-sparse, yielding 10-30x Toffoli savings
over the dense method.

Attribution
-----------
Based on the method described in :cite:`Rupprecht2026` and the Qualtran
implementation by Felix Rupprecht (DLR) published on Zenodo
:cite:`Rupprecht2026Zenodo` under Apache 2.0 license. The implementation
has been rewritten for integration into QDK Chemistry.

References
----------
    Felix Rupprecht and Sabine Wölk. (2026). Faster matrix product state preparation by
    exploiting symmetry-induced block-sparsity.
    https://arxiv.org/pdf/2605.28489. Zenodo: https://zenodo.org/records/20393500.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
from scipy.sparse import csc_array, vstack

from qdk_chemistry.data import AbelianMPSContainer, MPSSite
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .mps_sequential import (
    GivensLayerData,
    decompose_block_diagonal_to_givens,
)
from .state_preparation import StatePreparation, StatePreparationSettings

__all__: list[str] = [
    "MPSSparseStatePreparation",
]


class MPSSparseStatePreparationSettings(StatePreparationSettings):
    """Settings for MPS sparse state preparation."""

    def __init__(self):
        """Initialize the MPSSparseStatePreparationSettings."""
        super().__init__()
        self._set_default("rotation_bits", "int", 10, "Phase gradient precision.")


class MPSSparseStatePreparation(StatePreparation):
    r"""MPS state preparation exploiting block sparsity.

    Prepare the state using permutation-based decomposition. Each site unitary
    is factored as ``U = P_row · V_blockdiag · P_col``, where permutations are
    implemented via QROAM and the block-diagonal unitary is synthesized via
    Givens rotation layers. This exploits the block-sparse structure of MPS
    tensors arising from U(1) symmetries (particle number, spin conservation).

    Attribution
    -----------
    Based on the method in :cite:`Rupprecht2026` and code originally published by
    Felix Rupprecht on Zenodo :cite:`Rupprecht2026Zenodo` under Apache 2.0 license.
    """

    def __init__(self):
        """Initialize the MPS sparse state preparation algorithm."""
        super().__init__()
        self._settings = MPSSparseStatePreparationSettings()

    def name(self) -> str:
        """Return the algorithm name."""
        return "mps_sparse"

    def _run_impl(self, wavefunction: AbelianMPSContainer) -> Circuit:
        """Return a circuit to prepare an MPS state using block-sparsity.

        Args:
            wavefunction: An AbelianMPSContainer containing the tensors.

        Returns:
            A Circuit object implementing the MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an AbelianMPSContainer instance.

        """
        if not isinstance(wavefunction, AbelianMPSContainer):
            raise TypeError(f"MPSSparseStatePreparation requires an AbelianMPSContainer, got {type(wavefunction)}.")

        if wavefunction.physical_dimension != 4:
            raise ValueError("Sparse MPS state preparation requires four physical states per site.")
        data = generate_mps_sparse_preparation_data(wavefunction.sites)
        rotation_bits = self._settings.get("rotation_bits")
        params = data.to_qsharp_params(rotation_bits)
        program = QSHARP_UTILS.MPSSparse.MakeMPSSparseCircuit

        qsharp_factory = QsharpFactoryData(
            program=program,
            parameter=params,
        )

        op_params = QSHARP_UTILS.MPSSparse.MPSSparseParams(**params)
        qsharp_op = QSHARP_UTILS.MPSSparse.MakeMPSSparseOp(op_params)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op, encoding="jordan-wigner")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SparseSiteUnitaryData:
    r"""Decomposition data for a single sparse MPS site unitary.

    Each site unitary is decomposed as U = P_row · V_blockdiag · P_col.

    The permutations are stored as target mappings: perm_targets[i] gives the
    target index for basis state |i>. The block-diagonal unitary is stored
    as Givens layer data.
    """

    col_perm_targets: list[int]
    """Column permutation targets: col_perm_targets[i] = P_col(i)."""

    col_inv_perm_targets: list[int]
    """Inverse column permutation targets: col_inv_perm_targets[j] = P_col^{-1}(j)."""

    row_perm_targets: list[int]
    """Row permutation targets: row_perm_targets[i] = P_row(i)."""

    row_inv_perm_targets: list[int]
    """Inverse row permutation targets: row_inv_perm_targets[j] = P_row^{-1}(j)."""

    block_givens: GivensLayerData
    """Givens layers for the block-diagonal unitary V."""

    target_bits: int
    """Number of bits in the target register (site + ancilla)."""


@dataclass
class MPSSparsePreparationData:
    """All data needed to drive the MPSSparse Q# operation."""

    initial_state_vec: list[float]
    """Flattened initial state vector for the first site."""

    num_sites: int
    """Number of MPS sites."""

    ancilla_bits: int
    """Number of ancilla qubits (log2 of ancilla dimension)."""

    sites: list[SparseSiteUnitaryData] = field(default_factory=list)
    """Per-site decomposition data (one entry per site 1..num_sites-1)."""

    def to_qsharp_params(self, rotation_bits: int) -> dict:
        """Flatten into the dict expected by the MakeMPSSparseCircuit Q# operation."""
        d = 4  # physical dimension (2-qubit site register)
        ancilla_dim = 1 << self.ancilla_bits
        return {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteColPermTargets": [
                _perm_to_bitstrings(
                    _remap_perm_to_qsharp_order(s.col_perm_targets, d, ancilla_dim),
                    s.target_bits,
                )
                for s in self.sites
            ],
            "siteColInvPermTargets": [
                _perm_to_bitstrings(
                    _remap_perm_to_qsharp_order(s.col_inv_perm_targets, d, ancilla_dim),
                    s.target_bits,
                )
                for s in self.sites
            ],
            "siteRowPermTargets": [
                _perm_to_bitstrings(
                    _remap_perm_to_qsharp_order(s.row_perm_targets, d, ancilla_dim),
                    s.target_bits,
                )
                for s in self.sites
            ],
            "siteRowInvPermTargets": [
                _perm_to_bitstrings(
                    _remap_perm_to_qsharp_order(s.row_inv_perm_targets, d, ancilla_dim),
                    s.target_bits,
                )
                for s in self.sites
            ],
            "siteBlockLayerAngles": [s.block_givens.layer_angles for s in self.sites],
            "siteBlockLayerShifted": [s.block_givens.layer_shifted for s in self.sites],
            "siteBlockPhases": [s.block_givens.phases for s in self.sites],
        }


# ---------------------------------------------------------------------------
# Sparse decomposition algorithm
# ---------------------------------------------------------------------------


def generate_mps_sparse_preparation_data(
    tensors: Sequence[np.ndarray | MPSSite],
) -> MPSSparsePreparationData:
    """Compute all data needed for the MPSSparse Q# operation.

    Performs the permutation + block-diagonal decomposition for each site.

    Parameters
    ----------
    tensors : sequence of np.ndarray
        MPS tensors. ``tensors[i]`` has shape ``(chi_left, d, chi_right)``.

    Returns
    -------
    MPSSparsePreparationData
        Structured preparation data.

    """
    mps_sites = [tensor if isinstance(tensor, MPSSite) else MPSSite.from_dense(tensor) for tensor in tensors]
    num_sites = len(mps_sites)
    d = mps_sites[0].shape[1]
    if d != 4:
        raise ValueError("Sparse MPS state preparation requires four physical states per site.")

    # Determine consistent ancilla size
    max_ancilla_dim = 1
    for i in range(1, num_sites):
        chi_left, _, chi_right = mps_sites[i].shape
        local_bits = int(np.ceil(np.log2(max(chi_left, chi_right)))) if max(chi_left, chi_right) > 1 else 1
        max_ancilla_dim = max(max_ancilla_dim, 1 << local_bits)
    chi_1 = mps_sites[0].shape[2]
    init_bits = int(np.ceil(np.log2(max(1, chi_1)))) if chi_1 > 1 else 1
    max_ancilla_dim = max(max_ancilla_dim, 1 << init_bits)
    ancilla_bits = int(np.ceil(np.log2(max_ancilla_dim))) if max_ancilla_dim > 1 else 1
    ancilla_dim = 1 << ancilla_bits

    # Per-site decomposition
    sites: list[SparseSiteUnitaryData] = []
    for i in range(1, num_sites):
        site_data = _decompose_sparse_site(mps_sites[i], ancilla_dim)
        sites.append(site_data)

    # Initial state from first tensor
    first_tensor = mps_sites[0].to_dense()
    chi_1 = first_tensor.shape[2]
    init_state = first_tensor.transpose(1, 2, 0).sum(axis=2)  # (d, chi_1)
    init_padded = np.zeros((d, ancilla_dim))
    init_padded[:, :chi_1] = init_state
    initial_state_vec_arr = init_padded.flatten()
    norm = np.linalg.norm(initial_state_vec_arr)
    if norm > 1e-15:
        initial_state_vec_arr = initial_state_vec_arr / norm
    initial_state_vec = initial_state_vec_arr.tolist()

    return MPSSparsePreparationData(
        initial_state_vec=initial_state_vec,
        num_sites=num_sites,
        ancilla_bits=ancilla_bits,
        sites=sites,
    )


def _decompose_sparse_site(tensor: np.ndarray | MPSSite, ancilla_dim: int) -> SparseSiteUnitaryData:
    """Decompose one MPS site tensor using the sparse permutation method.

    Parameters
    ----------
    tensor : np.ndarray of shape (chi_left, 4, chi_right)
        The MPS tensor for this site.
    ancilla_dim : int
        The ancilla register dimension (power of 2).

    Returns
    -------
    SparseSiteUnitaryData
        The decomposition data for this site.

    """
    active_dim = 4 * ancilla_dim

    # Build the target matrix: transpose tensor indices and form CSC sparse matrix
    # Target matrix has shape (4*dim, chi_left) with columns = bond states
    target_matrix = _tensor_to_target_matrix(tensor, ancilla_dim)

    # Step 1: Extract rectangles and row permutation
    rectangles, row_perm = _get_rectangles_and_row_permutation(target_matrix)
    row_perm = _pad_permutation(row_perm, active_dim)

    # Step 2: Find column permutation to make rectangles square
    col_perm = _find_column_permutation(rectangles, active_dim)

    # Step 3: Expand rectangles to unitaries
    blocks = [_expand_to_unitary(rect) for rect in rectangles]

    # Pad with 1x1 identity blocks to fill remaining space
    used_dim = sum(b.shape[0] for b in blocks)
    remaining = active_dim - used_dim
    blocks += [np.eye(1)] * remaining

    # Step 4: Order blocks by size (largest first)
    # Returns inverted ordering permutation and sorted blocks
    ordering_perm, blocks = _order_blocks(blocks, active_dim)

    col_composed = [col_perm[ordering_perm[i]] for i in range(active_dim)]
    col_perm_final = _invert_perm(col_composed)
    row_perm_final = [row_perm[ordering_perm[i]] for i in range(active_dim)]

    # Step 5: Decompose block-diagonal into Givens layers
    block_angles, block_shifted, block_phases = decompose_block_diagonal_to_givens(blocks)
    block_givens = GivensLayerData(
        layer_angles=block_angles,
        layer_shifted=block_shifted,
        phases=block_phases,
    )

    # Compute inverse permutations for measurement-based unlookup
    col_inv_perm_final = _invert_perm(col_perm_final)
    row_inv_perm_final = _invert_perm(row_perm_final)

    # Number of bits in the target register (site + ancilla)
    target_bits = int(np.log2(active_dim))

    return SparseSiteUnitaryData(
        col_perm_targets=col_perm_final,
        col_inv_perm_targets=col_inv_perm_final,
        row_perm_targets=row_perm_final,
        row_inv_perm_targets=row_inv_perm_final,
        block_givens=block_givens,
        target_bits=target_bits,
    )


# ---------------------------------------------------------------------------
# Sparse decomposition helpers
# ---------------------------------------------------------------------------


def _tensor_to_target_matrix(tensor: np.ndarray | MPSSite, ancilla_dim: int) -> csc_array:
    """Build the sparse target matrix from an MPS tensor.

    The target matrix has shape (4 * ancilla_dim, chi_left), where each column
    corresponds to a left-bond index and the 4 blocks of ancilla_dim rows
    correspond to the 4 physical states.

    Parameters
    ----------
    tensor : np.ndarray of shape (chi_left, 4, chi_right)
        The MPS tensor.
    ancilla_dim : int
        The ancilla dimension (padded, power of 2).

    Returns
    -------
    csc_array
        The sparse target matrix of shape (4 * ancilla_dim, chi_left).

    """
    site = tensor if isinstance(tensor, MPSSite) else MPSSite.from_dense(tensor)
    chi_left, _, _ = site.shape
    # Reshape: for each physical index p, take the slice tensor[:, p, :] of shape
    # (chi_left, chi_right) -> transpose to (chi_right, chi_left).
    # Stack 4 such slices vertically to get (4*chi_right, chi_left), then pad rows.
    slices: list[csc_array] = []
    for physical_slice in site.physical_slices:
        matrix = physical_slice.T.tocsc()  # shape (chi_right, chi_left)
        # Pad rows to ancilla_dim
        if matrix.shape[0] < ancilla_dim:
            padding = csc_array((ancilla_dim - matrix.shape[0], chi_left))
            slices.append(csc_array(vstack((matrix, padding), format="csc")))
        else:
            slices.append(matrix[:ancilla_dim, :])
    return csc_array(vstack(slices, format="csc"))


def _get_rectangles_and_row_permutation(
    matrix: csc_array,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract chained rectangular blocks and row permutation from a sparse matrix.

    Walks columns left-to-right, grouping them by shared nonzero rows into
    contiguous rectangular blocks. Returns the blocks (as dense arrays) and
    the row permutation that chains them.

    Parameters
    ----------
    matrix : csc_array
        The sparse target matrix.

    Returns
    -------
    blocks : list[np.ndarray]
        Dense rectangular blocks with orthonormal (or near-orthonormal) columns.
    row_permutation : list[int]
        Row permutation mapping original row indices to chained order.

    """
    num_rows, num_cols = matrix.shape

    permutation: list[int] = []
    blocks: list[np.ndarray] = []
    seen_rows = np.zeros(num_rows, dtype=bool)

    current_rectangle_rows: list[int] = []
    current_rectangle_cols: list[int] = []

    for col_idx in range(num_cols):
        non_zero_rows = matrix.indices[matrix.indptr[col_idx] : matrix.indptr[col_idx + 1]].tolist()
        new_rows = [r for r in non_zero_rows if not seen_rows[r]]

        if len(new_rows) == len(non_zero_rows) and len(non_zero_rows) > 0:
            # New block starts
            if current_rectangle_rows:
                block = matrix[current_rectangle_rows, :][:, current_rectangle_cols].toarray()
                blocks.append(block)
            current_rectangle_rows = list(new_rows)
            current_rectangle_cols = [col_idx]
            permutation.extend(new_rows)
        else:
            # Continue current block
            current_rectangle_cols.append(col_idx)
            for r in new_rows:
                current_rectangle_rows.append(r)
                permutation.append(r)

        for r in new_rows:
            seen_rows[r] = True

    # Flush last block
    if current_rectangle_rows:
        block = matrix[current_rectangle_rows, :][:, current_rectangle_cols].toarray()
        blocks.append(block)

    # Add unseen rows at the end
    zero_rows = np.where(~seen_rows)[0].tolist()
    permutation.extend(zero_rows)

    return blocks, permutation


def _find_column_permutation(rectangles: list[np.ndarray], dim: int) -> list[int]:
    """Compute column permutation to transform chained rectangles into squares.

    For each rectangle that is taller than wide, permutes in columns from
    the right side so each block becomes square (enabling block-diagonal form).

    Parameters
    ----------
    rectangles : list[np.ndarray]
        Chained rectangular blocks.
    dim : int
        Total dimension of the active register.

    Returns
    -------
    list[int]
        Column permutation.

    """
    mapping = list(range(dim))

    col_left = 0
    col_right = dim
    diag = 0

    for rect in rectangles:
        height, width = rect.shape
        diff = height - width

        if diff > 0:
            # Move 'diff' columns from the right to fill the gap
            for i in range(width, col_right - col_left):
                mapping[col_left + i] += diff
            for i in range(diff):
                mapping[col_right - diff + i] = diag + width + i
            col_left += width
            col_right -= diff
        else:
            col_left += width

        diag += height

    return _invert_perm(mapping)


def _expand_to_unitary(rectangle: np.ndarray) -> np.ndarray:
    """Expand a rectangle with orthonormal columns into a full unitary.

    Uses null-space completion when the rectangle is taller than wide.

    Parameters
    ----------
    rectangle : np.ndarray
        Matrix with orthonormal columns (height >= width).

    Returns
    -------
    np.ndarray
        Square unitary matrix.

    """
    h, w = rectangle.shape
    if h == w:
        return rectangle
    # Null space of rectangle^T gives the orthogonal complement
    _, _, vt = np.linalg.svd(rectangle.conj().T)
    null_space = vt[w:, :].T.conj()
    return np.hstack([rectangle, null_space])


def _order_blocks(blocks: list[np.ndarray], dim: int) -> tuple[list[int], list[np.ndarray]]:
    """Sort blocks by size (largest first) and return the reordering permutation.

    Parameters
    ----------
    blocks : list[np.ndarray]
        Block matrices forming a block-diagonal.
    dim : int
        Total dimension.

    Returns
    -------
    permutation : list[int]
        Inverted ordering permutation.
    sorted_blocks : list[np.ndarray]
        Blocks sorted largest-first.

    """
    mapping = list(range(dim))

    # Sort indices by block size (descending)
    sorted_indices = sorted(range(len(blocks)), key=lambda i: -blocks[i].shape[0])

    # Compute cumulative offsets
    offsets = []
    offset = 0
    for block in blocks:
        offsets.append(offset)
        offset += block.shape[0]

    # Build forward mapping: mapping[old_pos] = new_pos
    new_offset = 0
    for idx in sorted_indices:
        block_size = blocks[idx].shape[0]
        old_start = offsets[idx]
        for k in range(block_size):
            mapping[old_start + k] = new_offset + k
        new_offset += block_size

    # Return inverted mapping
    sorted_blocks = [blocks[i] for i in sorted_indices]
    return _invert_perm(mapping), sorted_blocks


# ---------------------------------------------------------------------------
# Permutation utilities
# ---------------------------------------------------------------------------


def _invert_perm(perm: list[int]) -> list[int]:
    """Invert a permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _pad_permutation(perm: list[int], target_len: int) -> list[int]:
    """Pad a permutation to target_len with identity mapping for extra indices."""
    if len(perm) >= target_len:
        return perm[:target_len]
    return perm + list(range(len(perm), target_len))


# ---------------------------------------------------------------------------
# Q# encoding utilities
# ---------------------------------------------------------------------------


def _remap_perm_to_qsharp_order(perm_targets: list[int], d: int, ancilla_dim: int) -> list[int]:
    """Remap permutation indices from target-matrix order to Q# register order.

    The target matrix uses row = physical_state * ancilla_dim + ancilla_state,
    but the Q# register (target = newSite + ancilla) with little-endian
    convention gives value = physical_state + ancilla_state * d.

    This function conjugates the permutation by the reindexing so that
    SelectSwap (which uses Q# little-endian addressing) applies the correct
    permutation.

    Parameters
    ----------
    perm_targets : list[int]
        Permutation targets in target-matrix row ordering.
    d : int
        Physical dimension (always 4 for 2-qubit site register).
    ancilla_dim : int
        Ancilla dimension (2^ancilla_bits).

    Returns
    -------
    list[int]
        Permutation targets reindexed for Q# register ordering.

    """
    active_dim = d * ancilla_dim
    qs_perm = [0] * active_dim
    for v in range(active_dim):
        # Register value v encodes physical=v%d, ancilla=v//d
        p = v % d
        a = v // d
        # Convert to target matrix row
        r = p * ancilla_dim + a
        # Apply permutation in target matrix space
        r_out = perm_targets[r]
        # Convert result back to Q# register value
        p_out = r_out // ancilla_dim
        a_out = r_out % ancilla_dim
        v_out = p_out + a_out * d
        qs_perm[v] = v_out
    return qs_perm


def _perm_to_bitstrings(perm_targets: list[int], num_bits: int) -> list[list[bool]]:
    """Encode permutation targets as Bool[][] for Q# SelectSwap.

    Each target integer is encoded as a little-endian bit string of length num_bits.

    Parameters
    ----------
    perm_targets : list[int]
        Permutation target indices.
    num_bits : int
        Number of bits for each target encoding.

    Returns
    -------
    list[list[bool]]
        Bool[N][num_bits] encoding for Q#.

    """
    result = []
    for target in perm_targets:
        bits = [(target >> b) & 1 == 1 for b in range(num_bits)]
        result.append(bits)
    return result
