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
from qdk._native import TargetProfile

from qdk_chemistry.data import AbelianMPSSite, Configuration, MPSContainer, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, use_qsharp_profile
from qdk_chemistry.utils.unitary_synthesis import decompose_sparse_site

from .state_preparation import StatePreparation, StatePreparationSettings

__all__: list[str] = [
    "MPSSparseStatePreparation",
]


def validate_mps_physical_basis(container: MPSContainer) -> None:
    """Require the physical-slice order assumed by the Q# operations."""
    canonical_basis = [Configuration.from_spin_half_string(state) for state in ("0", "u", "d", "2")]
    if container.physical_basis != canonical_basis:
        raise ValueError("MPS state preparation requires physical basis ordering ('0', 'u', 'd', '2').")


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
        validate_mps_physical_basis(container)
        if container.orthogonality_center != 0:
            raise ValueError("Sparse MPS state preparation requires a right-canonical MPS with center zero.")

        num_orbitals = container.orbitals.get_num_molecular_orbitals()
        if container.num_sites != num_orbitals:
            raise ValueError("Sparse MPS state preparation requires exactly one MPS site per molecular orbital.")

        data = generate_mps_sparse_preparation_data(container.sites)
        rotation_bits = self._settings.get("rotation_bits")
        params = data.to_qsharp_params(rotation_bits, container.site_to_orbital_order)
        # MPS sparse state preparation requires the Adaptive profile; temporarily switch
        # the shared global QSHARP_UTILS so the resolved callable carries the Adaptive context.
        with use_qsharp_profile(TargetProfile.Adaptive_RIF):
            program = QSHARP_UTILS.MPSSparse.MakeMPSSparseCircuit

        qsharp_factory = QsharpFactoryData(
            program=program,
            parameter=params,
        )

        return Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SparseSiteUnitaryData:
    r"""Decomposition data for a single sparse MPS site unitary.

    Each site unitary is decomposed as U = P_row · V_blockdiag · P_col.

    The permutations are stored as target mappings: perm_targets[i] gives the
    target index for basis state |i>. The block-diagonal unitary is stored as
    Givens rotation layers.
    """

    col_perm_targets: list[int]
    """Column permutation targets: col_perm_targets[i] = P_col(i)."""

    row_perm_targets: list[int]
    """Row permutation targets: row_perm_targets[i] = P_row(i)."""

    layer_angles: list[list[float]]
    """Givens angles per layer for the block-diagonal unitary V."""

    layer_shifted: list[bool]
    """Whether each Givens layer is shifted."""

    phases: list[bool]
    """Phase corrections for the block-diagonal unitary V."""


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

    def to_qsharp_params(
        self,
        rotation_bits: int,
        site_to_orbital_order: Sequence[int] | None = None,
    ) -> dict:
        """Flatten into the dict expected by the MakeMPSSparseCircuit Q# operation."""
        d = 4  # physical dimension (2-qubit site register)
        ancilla_dim = 1 << self.ancilla_bits
        orbital_order = list(range(self.num_sites)) if site_to_orbital_order is None else list(site_to_orbital_order)
        if (
            len(orbital_order) != self.num_sites
            or len(set(orbital_order)) != self.num_sites
            or any(i < 0 for i in orbital_order)
        ):
            raise ValueError("site_to_orbital_order must contain one unique nonnegative index per MPS site.")
        return {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "siteToOrbitalOrder": orbital_order,
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteDecompositions": [
                {
                    "colPermTargets": _remap_perm_to_qsharp_order(site.col_perm_targets, d, ancilla_dim),
                    "rowPermTargets": _remap_perm_to_qsharp_order(site.row_perm_targets, d, ancilla_dim),
                    "blockLayerAngles": site.layer_angles,
                    "blockLayerShifted": site.layer_shifted,
                    "blockPhases": site.phases,
                }
                for site in self.sites
            ],
        }


# ---------------------------------------------------------------------------
# Sparse decomposition algorithm
# ---------------------------------------------------------------------------


def generate_mps_sparse_preparation_data(
    tensors: Sequence[np.ndarray | AbelianMPSSite],
) -> MPSSparsePreparationData:
    """Compute all data needed for the MPSSparse Q# operation.

    Performs the permutation + block-diagonal decomposition for each site.

    Parameters
    ----------
    tensors : sequence of np.ndarray or AbelianMPSSite
        MPS sites. Array inputs have shape ``(chi_left, d, chi_right)`` and are
        wrapped as unsymmetrized sites. AbelianMPSSite inputs preserve their numerical
        sparsity when converted to per-physical-state CSC matrices.

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
    if mps_sites[0].shape[0] != 1 or mps_sites[-1].shape[2] != 1:
        raise ValueError("Sparse MPS state preparation requires open boundary bonds of dimension one.")
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
        col_perm, row_perm, angles, shifted, phases = decompose_sparse_site(mps_sites[i], ancilla_dim)
        sites.append(
            SparseSiteUnitaryData(
                col_perm_targets=list(col_perm),
                row_perm_targets=list(row_perm),
                layer_angles=angles,
                layer_shifted=shifted,
                phases=phases,
            )
        )

    # Initial state from first tensor
    first_tensor = mps_sites[0].to_dense()
    chi_1 = first_tensor.shape[2]
    init_state = first_tensor[0]  # (d, chi_1)
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
        sites=sites,
    )


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
