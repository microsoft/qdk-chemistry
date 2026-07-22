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

from qdk_chemistry._core.utils import decompose_sparse_site
from qdk_chemistry.data import AbelianMPSSite, MPSContainer, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .mps_sequential import GivensLayerData
from .state_preparation import StatePreparation, StatePreparationSettings

__all__: list[str] = [
    "MPSSparseStatePreparation",
]


class MPSSparseStatePreparationSettings(StatePreparationSettings):
    """Settings for MPS sparse state preparation."""

    def __init__(self):
        """Initialize the MPSSparseStatePreparationSettings."""
        super().__init__()
        self._set_default("rotation_bits", "int", 10, "Phase gradient precision.", (2, 62))


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

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Return a circuit to prepare an MPS state using block-sparsity.

        Args:
            wavefunction: The wavefunction to prepare.

        Returns:
            A Circuit object implementing the MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an MPSContainer instance.

        """
        container = wavefunction.get_container()
        if not isinstance(container, MPSContainer):
            raise TypeError(f"MPSSparseStatePreparation requires an MPSContainer, got {type(container)}.")
        if container.is_complex:
            raise ValueError("Sparse MPS state preparation currently supports only real-valued MPS tensors.")

        if container.physical_dimension != 4:
            raise ValueError("Sparse MPS state preparation requires four physical states per site.")
        if container.orthogonality_center != 0:
            raise ValueError("Sparse MPS state preparation requires a right-canonical MPS with center zero.")

        num_orbitals = container.orbitals.get_num_molecular_orbitals()
        if container.num_sites != num_orbitals:
            raise ValueError("Sparse MPS state preparation requires exactly one MPS site per molecular orbital.")

        data = generate_mps_sparse_preparation_data(container.sites, container.site_to_orbital_order)
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

    row_perm_targets: list[int]
    """Row permutation targets: row_perm_targets[i] = P_row(i)."""

    block_givens: GivensLayerData
    """Givens layers for the block-diagonal unitary V."""


@dataclass
class MPSSparsePreparationData:
    """All data needed to drive the MPSSparse Q# operation."""

    initial_state_vec: list[float]
    """Flattened initial state vector for the first site."""

    num_sites: int
    """Number of MPS sites."""

    ancilla_bits: int
    """Number of ancilla qubits (log2 of ancilla dimension)."""

    site_to_orbital_order: list[int]
    """Molecular-orbital index for each site in MPS chain order."""

    sites: list[SparseSiteUnitaryData] = field(default_factory=list)
    """Per-site decomposition data (one entry per site 1..num_sites-1)."""

    def to_qsharp_params(
        self,
        rotation_bits: int,
    ) -> dict:
        """Flatten into the dict expected by the MakeMPSSparseCircuit Q# operation."""
        d = 4  # physical dimension (2-qubit site register)
        ancilla_dim = 1 << self.ancilla_bits
        target_bits = self.ancilla_bits + 2
        return {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "siteToOrbitalOrder": self.site_to_orbital_order,
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteColPermTargets": [
                _perm_to_bitstrings(
                    _remap_perm_to_qsharp_order(s.col_perm_targets, d, ancilla_dim),
                    target_bits,
                )
                for s in self.sites
            ],
            "siteColInvPermTargets": [
                _perm_to_bitstrings(
                    _invert_perm(_remap_perm_to_qsharp_order(s.col_perm_targets, d, ancilla_dim)),
                    target_bits,
                )
                for s in self.sites
            ],
            "siteRowPermTargets": [
                _perm_to_bitstrings(
                    _remap_perm_to_qsharp_order(s.row_perm_targets, d, ancilla_dim),
                    target_bits,
                )
                for s in self.sites
            ],
            "siteRowInvPermTargets": [
                _perm_to_bitstrings(
                    _invert_perm(_remap_perm_to_qsharp_order(s.row_perm_targets, d, ancilla_dim)),
                    target_bits,
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
    tensors: Sequence[np.ndarray | AbelianMPSSite],
    site_to_orbital_order: Sequence[int] | None = None,
) -> MPSSparsePreparationData:
    """Compute all data needed for the MPSSparse Q# operation.

    Performs the permutation + block-diagonal decomposition for each site.

    Parameters
    ----------
    tensors : sequence of np.ndarray or AbelianMPSSite
        MPS sites. Array inputs have shape ``(chi_left, d, chi_right)`` and are
        wrapped as unsymmetrized sites. AbelianMPSSite inputs preserve their numerical
        sparsity when converted to per-physical-state CSC matrices.
    site_to_orbital_order : sequence of int, optional
        Molecular-orbital index for each tensor in MPS chain order. Defaults to
        the identity mapping.

    Returns
    -------
    MPSSparsePreparationData
        Structured preparation data.

    """
    mps_sites = [
        tensor if isinstance(tensor, AbelianMPSSite) else AbelianMPSSite.from_dense(tensor) for tensor in tensors
    ]
    if not mps_sites:
        raise ValueError("Sparse MPS state preparation requires at least one site.")
    if any(site.is_complex for site in mps_sites):
        raise ValueError("Sparse MPS state preparation currently supports only real-valued MPS tensors.")
    if any(not np.isfinite(site.to_dense()).all() or np.linalg.norm(site.to_dense()) <= 1e-15 for site in mps_sites):
        raise ValueError("MPS sites must contain finite amplitudes with nonzero norm.")
    num_sites = len(mps_sites)
    orbital_order = list(range(num_sites)) if site_to_orbital_order is None else list(site_to_orbital_order)
    if len(orbital_order) != num_sites or len(set(orbital_order)) != num_sites or any(i < 0 for i in orbital_order):
        raise ValueError("site_to_orbital_order must contain one unique nonnegative index per MPS site.")
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
    if not np.isfinite(initial_state_vec_arr).all() or not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError("MPS initial state must contain finite amplitudes with nonzero norm.")
    initial_state_vec_arr = initial_state_vec_arr / norm
    initial_state_vec = initial_state_vec_arr.tolist()

    return MPSSparsePreparationData(
        initial_state_vec=initial_state_vec,
        num_sites=num_sites,
        ancilla_bits=ancilla_bits,
        site_to_orbital_order=orbital_order,
        sites=sites,
    )


def _decompose_sparse_site(tensor: np.ndarray | AbelianMPSSite, ancilla_dim: int) -> SparseSiteUnitaryData:
    """Decompose one MPS site tensor using the sparse permutation method.

    Parameters
    ----------
    tensor : np.ndarray of shape (chi_left, 4, chi_right) or AbelianMPSSite
        The MPS site to decompose. An AbelianMPSSite is read through its sparse
        physical slices; an array is first wrapped as an unsymmetrized site.
    ancilla_dim : int
        The ancilla register dimension (power of 2).

    Returns
    -------
    SparseSiteUnitaryData
        The decomposition data for this site.

    """
    # Build the target matrix: transpose tensor indices and form CSC sparse matrix
    # Target matrix has shape (4*dim, chi_left) with columns = bond states
    target_matrix = _tensor_to_target_matrix(tensor, ancilla_dim)

    (
        col_perm_final,
        row_perm_final,
        block_angles,
        block_shifted,
        block_phases,
    ) = decompose_sparse_site(target_matrix.toarray())
    block_givens = GivensLayerData(
        layer_angles=block_angles,
        layer_shifted=[bool(value) for value in block_shifted],
        phases=[bool(value) for value in block_phases],
    )

    return SparseSiteUnitaryData(
        col_perm_targets=list(col_perm_final),
        row_perm_targets=list(row_perm_final),
        block_givens=block_givens,
    )


# ---------------------------------------------------------------------------
# Sparse decomposition helpers
# ---------------------------------------------------------------------------


def _tensor_to_target_matrix(tensor: np.ndarray | AbelianMPSSite, ancilla_dim: int) -> csc_array:
    """Build the sparse target matrix from an MPS tensor.

    The target matrix has shape (4 * ancilla_dim, chi_left), where each column
    corresponds to a left-bond index and the 4 blocks of ancilla_dim rows
    correspond to the 4 physical states.

    Parameters
    ----------
    tensor : np.ndarray of shape (chi_left, 4, chi_right) or AbelianMPSSite
        The MPS site. Its four physical matrices are transposed, row-padded to
        ``ancilla_dim``, and stacked vertically.
    ancilla_dim : int
        The ancilla dimension (padded, power of 2).

    Returns
    -------
    csc_array
        The sparse target matrix of shape (4 * ancilla_dim, chi_left).

    """
    site = tensor if isinstance(tensor, AbelianMPSSite) else AbelianMPSSite.from_dense(tensor)
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


# ---------------------------------------------------------------------------
# Permutation utilities
# ---------------------------------------------------------------------------


def _invert_perm(perm: list[int]) -> list[int]:
    """Invert a permutation."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


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
