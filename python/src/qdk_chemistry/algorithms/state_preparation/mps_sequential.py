"""Matrix Product State (MPS) state preparation via sequential site unitary synthesis.

Implements the MPS state preparation algorithm based on
:cite:`Berry2025`. Each site unitary is decomposed based on Appendix B in
:cite:`Rupprecht2026`.

Attribution
-----------
The unitary synthesis is based on code originally published by Felix Rupprecht
on Zenodo :cite:`Rupprecht2026Zenodo` under Apache 2.0 license.
The implementation has been rewritten for integration into QDK Chemistry.

References
----------
    Felix Rupprecht and Sabine Wölk. (2026). Faster matrix product state preparation by
    exploiting symmetry-induced block-sparsity.
    https://arxiv.org/pdf/2605.28489. Zenodo: https://zenodo.org/records/20393500.

    Dominic W. Berry et al. (2025). Rapid Initial-State Preparation for the Quantum Simulation of
    Strongly Correlated Molecules. PRX Quantum 6, 020327.
    https://doi.org/10.1103/PRXQuantum.6.020327.

    William R. Clements et al. (2017). An Optimal Design for Universal Multiport Interferometers.
    https://arxiv.org/abs/1603.08788.

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

from qdk_chemistry._core.utils import (
    decompose_block_diagonal_to_givens,
    decompose_site_csd,
    decompose_unitary_to_givens,
)
from qdk_chemistry.data import AbelianMPSContainer, MPSSite
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation, StatePreparationSettings

__all__: list[str] = [
    "MPSSequentialStatePreparation",
]


class MPSSequentialStatePreparationSettings(StatePreparationSettings):
    """Settings for MPS sequential state preparation."""

    def __init__(self):
        """Initialize the MPSSequentialStatePreparationSettings."""
        super().__init__()
        self._set_default("rotation_bits", "int", 10, "Phase gradient precision.", (2, 62))
        self._set_default(
            "fast_resource_estimation",
            "bool",
            False,
            "Synthesize one representative per shape group and use BeginEstimateCaching to "
            "replicate cost. Valid for resource estimation only — not simulation.",
        )
        self._set_default(
            "fast_grouped_resource_estimation",
            "bool",
            False,
            "Compatibility alias for fast_resource_estimation.",
        )


class MPSSequentialStatePreparation(StatePreparation):
    r"""Matrix Product State (MPS) state preparation using sequential unitary synthesis.

    Prepare the state sequentially, qubit-by-qubit (2 qubits per site), using an
    ancilla register that stores the virtual bond dimension. Each site unitary
    is decomposed based on Appendix B in :cite:`Rupprecht2026`, and synthesized
    into a quantum circuit via Givens rotation layers with QROAM (Quantum
    Read-Only Access Memory) angle loading and phase gradient rotations.

    Attribution
    -----------
    The unitary synthesis is based on code originally published by Felix Rupprecht
    on Zenodo :cite:`Rupprecht2026Zenodo` under Apache 2.0 license.
    The implementation has been rewritten for integration into QDK Chemistry.
    """

    def __init__(self):
        """Initialize the MPS sequential state preparation algorithm."""
        super().__init__()
        self._settings = MPSSequentialStatePreparationSettings()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"mps_sequential"``

        """
        return "mps_sequential"

    def _run_impl(self, wavefunction: AbelianMPSContainer) -> Circuit:
        """Return a circuit to prepare an MPS state.

        Args:
            wavefunction: An AbelianMPSContainer containing the tensors.

        Returns:
            A Circuit object implementing the MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an AbelianMPSContainer instance.

        """
        if not isinstance(wavefunction, AbelianMPSContainer):
            raise TypeError(f"MPSSequentialStatePreparation requires an AbelianMPSContainer, got {type(wavefunction)}.")
        if wavefunction.is_complex:
            raise ValueError("MPS sequential state preparation currently supports only real-valued MPS tensors.")

        fast_re = self._settings.get("fast_resource_estimation")
        fast_grouped_re = self._settings.get("fast_grouped_resource_estimation")

        if wavefunction.physical_dimension != 4:
            raise ValueError("MPS sequential state preparation requires four physical states per site.")
        if wavefunction.orthogonality_center != 0:
            raise ValueError("MPS sequential state preparation requires a right-canonical MPS with center zero.")
        num_orbitals = wavefunction.orbitals.get_num_molecular_orbitals()
        if wavefunction.num_sites != num_orbitals:
            raise ValueError("MPS sequential state preparation requires exactly one MPS site per molecular orbital.")
        data = generate_mps_preparation_data(wavefunction.sites, fast_resource_estimation=fast_re or fast_grouped_re)

        rotation_bits = self._settings.get("rotation_bits")

        site_to_orbital_order = wavefunction.site_to_orbital_order
        if fast_re or fast_grouped_re:
            params = data.to_qsharp_params_grouped(rotation_bits, site_to_orbital_order)
            program = QSHARP_UTILS.MPSSequential.MakeMPSSequentialCircuitGrouped
        else:
            params = data.to_qsharp_params(rotation_bits, site_to_orbital_order)
            program = QSHARP_UTILS.MPSSequential.MakeMPSSequentialCircuit

        qsharp_factory = QsharpFactoryData(
            program=program,
            parameter=params,
        )

        # Build composable op for QPE composition
        if fast_re or fast_grouped_re:
            op_params = QSHARP_UTILS.MPSSequential.MPSSequentialGroupedParams(**params)
            qsharp_op = QSHARP_UTILS.MPSSequential.MakeMPSSequentialOpGrouped(op_params)
        else:
            op_params = QSHARP_UTILS.MPSSequential.MPSSequentialParams(**params)
            qsharp_op = QSHARP_UTILS.MPSSequential.MakeMPSSequentialOp(op_params)

        return Circuit(qsharp_factory=qsharp_factory, qsharp_op=qsharp_op, encoding="jordan-wigner")


# ---------------------------------------------------------------------------
# Data containers for unitary decomposition
# ---------------------------------------------------------------------------


@dataclass
class GivensLayerData:
    """Result of decomposing a unitary into Givens rotation layers.

    Stores the factorization ``U = D · L_l · ... · L_1`` where each ``L_j``
    is a layer of parallel R_y rotations and ``D`` is a ±1 sign matrix.
    """

    layer_angles: list[list[float]]
    """Per-layer R_y rotation angles for each parallel slot."""

    layer_shifted: list[bool]
    """Whether each layer uses odd-indexed pairs (True) or even (False)."""

    phases: list[bool]
    """Diagonal sign flips (True where entry is -1)."""


@dataclass
class SiteUnitaryData:
    r"""Decomposition data for a single MPS site unitary.

    Holds the 7-matrix Cosine-Sine Decomposition(CSD) from Appendix B of
    :cite:`Rupprecht2026` and the Givens-layer synthesis of each component.

    The circuit applies these components in order (see Fig. 5 of the paper)::

        V -> UCR(d_0') -> CNOT -> W_0 -> UCR(d_1') -> CNOT -> W_1 -> UCR(d_2') -> U

    where each UCR (Uniformly Controlled Rotation) is a multiplexed R_y
    rotation addressed by the ancilla register, CNOT is a Controlled-NOT
    gate, and U = diag(u_0, u_1, u_2, u_3) is block-diagonal.
    """

    v: GivensLayerData
    """Givens layers for V (right unitary, pushed from next site)."""

    rot_angles: list[list[float]]
    """UCR (Uniformly Controlled Rotation) angles for each of the 3 rotation steps.

    Format: ``[rot0, rot1, rot2]``.
    """

    w0: GivensLayerData
    """Givens layers for W_0 (mixing unitary, controlled by site[0])."""

    w1: GivensLayerData
    """Givens layers for W_1 (mixing unitary, controlled by site[1])."""

    u: GivensLayerData
    """Givens layers for U (block-diagonal unitary on ancilla+site)."""


@dataclass
class MPSPreparationData:
    """All data needed to drive the MPSSequential Q# operation.

    Produced by :func:`generate_mps_preparation_data` and consumed by
    :meth:`MPSSequentialStatePreparation._run_impl`.
    """

    num_sites: int
    """Number of MPS sites."""

    ancilla_bits: int
    """Number of ancilla qubits."""

    initial_state_vec: list[float]
    """Flattened initial state vector for the first site."""

    sites: list[SiteUnitaryData] = field(default_factory=list)
    """Per-site decomposition data.

    In resource estimation mode, this contains one entry per unique unitary
    shape rather than per site.
    """

    site_shape_indices: list[int] | None = None
    """In resource estimation mode, this maps each site to its corresponding shape group.
    None means ungrouped."""

    shape_effective_bits: list[int] | None = None
    """In resource estimation mode, effective ancilla bits per shape group.
    None means ungrouped."""

    def to_qsharp_params(
        self,
        rotation_bits: int,
        site_to_orbital_order: list[int] | None = None,
    ) -> dict:
        """Flatten into the dict expected by the MakeMPSSequentialCircuit Q# operation."""
        site_to_orbital_order = list(range(self.num_sites)) if site_to_orbital_order is None else site_to_orbital_order
        params: dict = {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "siteToOrbitalOrder": site_to_orbital_order,
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteVLayerAngles": [s.v.layer_angles for s in self.sites],
            "siteVLayerShifted": [s.v.layer_shifted for s in self.sites],
            "siteVPhases": [s.v.phases for s in self.sites],
            "siteRot0Angles": [s.rot_angles[0] for s in self.sites],
            "siteRot1Angles": [s.rot_angles[1] for s in self.sites],
            "siteRot2Angles": [s.rot_angles[2] for s in self.sites],
            "siteW0LayerAngles": [s.w0.layer_angles for s in self.sites],
            "siteW0LayerShifted": [s.w0.layer_shifted for s in self.sites],
            "siteW0Phases": [s.w0.phases for s in self.sites],
            "siteW1LayerAngles": [s.w1.layer_angles for s in self.sites],
            "siteW1LayerShifted": [s.w1.layer_shifted for s in self.sites],
            "siteW1Phases": [s.w1.phases for s in self.sites],
            "siteULayerAngles": [s.u.layer_angles for s in self.sites],
            "siteULayerShifted": [s.u.layer_shifted for s in self.sites],
            "siteUPhases": [s.u.phases for s in self.sites],
        }
        return params

    def to_qsharp_params_grouped(
        self,
        rotation_bits: int,
        site_to_orbital_order: list[int] | None = None,
    ) -> dict:
        """Flatten into the dict expected by MakeMPSSequentialCircuitGrouped.

        Passes one representative per unique shape + a site-to-shape mapping.
        This minimizes data transfer while enabling accurate per-shape caching
        in the Q# resource estimator.
        """
        assert self.site_shape_indices is not None, "Grouped mode requires site_shape_indices"
        assert self.shape_effective_bits is not None, "Grouped mode requires shape_effective_bits"
        site_to_orbital_order = list(range(self.num_sites)) if site_to_orbital_order is None else site_to_orbital_order

        params: dict = {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "siteToOrbitalOrder": site_to_orbital_order,
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteShapeIndices": self.site_shape_indices,
            "shapeEffectiveBits": self.shape_effective_bits,
            "shapeVLayerAngles": [s.v.layer_angles for s in self.sites],
            "shapeVLayerShifted": [s.v.layer_shifted for s in self.sites],
            "shapeVPhases": [s.v.phases for s in self.sites],
            "shapeRot0Angles": [s.rot_angles[0] for s in self.sites],
            "shapeRot1Angles": [s.rot_angles[1] for s in self.sites],
            "shapeRot2Angles": [s.rot_angles[2] for s in self.sites],
            "shapeW0LayerAngles": [s.w0.layer_angles for s in self.sites],
            "shapeW0LayerShifted": [s.w0.layer_shifted for s in self.sites],
            "shapeW0Phases": [s.w0.phases for s in self.sites],
            "shapeW1LayerAngles": [s.w1.layer_angles for s in self.sites],
            "shapeW1LayerShifted": [s.w1.layer_shifted for s in self.sites],
            "shapeW1Phases": [s.w1.phases for s in self.sites],
            "shapeULayerAngles": [s.u.layer_angles for s in self.sites],
            "shapeULayerShifted": [s.u.layer_shifted for s in self.sites],
            "shapeUPhases": [s.u.phases for s in self.sites],
        }
        return params

    def to_qsharp_params_grouped_fast(
        self,
        rotation_bits: int,
        site_to_orbital_order: list[int] | None = None,
    ) -> dict:
        """Flatten into the dict expected by MakeMPSSequentialCircuitGroupedFast.

        Passes only 2 representative layers per Givens matrix (one non-shifted,
        one shifted) plus the total layer count. This reduces serialization from
        O(dim^2) to O(dim) and enables RepeatEstimates on the Q# side.
        """
        assert self.site_shape_indices is not None, "Grouped mode requires site_shape_indices"
        assert self.shape_effective_bits is not None, "Grouped mode requires shape_effective_bits"
        site_to_orbital_order = list(range(self.num_sites)) if site_to_orbital_order is None else site_to_orbital_order

        def _rep_layers(givens: GivensLayerData) -> list[list[float]]:
            """Extract at most 2 representative layers (non-shifted, shifted)."""
            if len(givens.layer_angles) == 0:
                return []
            if len(givens.layer_angles) == 1:
                return [givens.layer_angles[0]]
            return [givens.layer_angles[0], givens.layer_angles[1]]

        params: dict = {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "siteToOrbitalOrder": site_to_orbital_order,
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteShapeIndices": self.site_shape_indices,
            "shapeEffectiveBits": self.shape_effective_bits,
            "shapeVRepLayerAngles": [_rep_layers(s.v) for s in self.sites],
            "shapeVNumLayers": [len(s.v.layer_angles) for s in self.sites],
            "shapeVPhases": [s.v.phases for s in self.sites],
            "shapeRot0Angles": [s.rot_angles[0] for s in self.sites],
            "shapeRot1Angles": [s.rot_angles[1] for s in self.sites],
            "shapeRot2Angles": [s.rot_angles[2] for s in self.sites],
            "shapeW0RepLayerAngles": [_rep_layers(s.w0) for s in self.sites],
            "shapeW0NumLayers": [len(s.w0.layer_angles) for s in self.sites],
            "shapeW0Phases": [s.w0.phases for s in self.sites],
            "shapeW1RepLayerAngles": [_rep_layers(s.w1) for s in self.sites],
            "shapeW1NumLayers": [len(s.w1.layer_angles) for s in self.sites],
            "shapeW1Phases": [s.w1.phases for s in self.sites],
            "shapeURepLayerAngles": [_rep_layers(s.u) for s in self.sites],
            "shapeUNumLayers": [len(s.u.layer_angles) for s in self.sites],
            "shapeUPhases": [s.u.phases for s in self.sites],
        }
        return params


# ---------------------------------------------------------------------------
# Unitary synthesis helpers
# ---------------------------------------------------------------------------


def generate_mps_preparation_data(
    tensors: Sequence[np.ndarray | MPSSite],
    fast_resource_estimation: bool = False,
) -> MPSPreparationData:
    """Compute all data needed for the MPSSequential Q# operation.

    Performs CSD + Givens layer decomposition for each site.
    Returns structured data with raw angles (Double) and phases (Bool) --
    Q# handles angle quantization internally.

    Parameters
    ----------
    tensors : sequence of np.ndarray
        MPS tensors. ``tensors[i]`` has shape ``(chi_left, d, chi_right)``.
    fast_resource_estimation : bool
        If True, for each unitary of a certain shapes, only decompose a single
        representative site unitary and replicate its data for all sites.
        This is valid for resource estimation but NOT for simulation.

    Returns
    -------
    MPSPreparationData
        Structured preparation data. Call ``.to_qsharp_params(rotation_bits)``
        to flatten into the dict expected by the Q# operation.

    """
    mps_sites = [tensor if isinstance(tensor, MPSSite) else MPSSite.from_dense(tensor) for tensor in tensors]
    if not mps_sites:
        raise ValueError("MPS sequential state preparation requires at least one site.")
    if any(site.is_complex for site in mps_sites):
        raise ValueError("MPS sequential state preparation currently supports only real-valued MPS tensors.")
    if any(not np.isfinite(site.to_dense()).all() or np.linalg.norm(site.to_dense()) <= 1e-15 for site in mps_sites):
        raise ValueError("MPS sites must contain finite amplitudes with nonzero norm.")
    num_sites = len(mps_sites)
    d = mps_sites[0].shape[1]
    if d != 4:
        raise ValueError("MPS sequential state preparation requires four physical states per site.")

    # Determine consistent ancilla size across all sites
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

    # Per-site decomposition — process in reverse order to propagate V matrices.
    # Each site's V is absorbed into the previous site's target matrix (via v_from_next),
    # eliminating the explicit V unitary application.
    sites: list[SiteUnitaryData] = []
    v_from_next: np.ndarray | None = None
    site_shape_indices: list[int] | None = None
    unique_bits: list[int] | None = None
    if fast_resource_estimation and num_sites > 2:
        # Group sites by shape.
        # Sites with the same max(chi_left, chi_right) have the similar circuit cost.
        site_effective_bits: list[int] = []
        for i in range(1, num_sites):
            chi_left, _, chi_right = mps_sites[i].shape
            eff = max(chi_left, chi_right)
            bits = int(np.ceil(np.log2(eff))) if eff > 1 else 1
            site_effective_bits.append(bits)

        unique_bits = sorted(set(site_effective_bits))
        shape_to_idx = {b: idx for idx, b in enumerate(unique_bits)}
        site_shape_indices = [shape_to_idx[b] for b in site_effective_bits]

        # Generate dummy data at the padded ancilla dimension for all shapes.
        # Normal mode pads all sites to ancilla_dim, so fast mode must match.
        shape_representatives: list[SiteUnitaryData] = []
        for _ in unique_bits:
            site_data = _make_dummy_site_data(ancilla_dim)
            shape_representatives.append(site_data)
        unique_bits = [ancilla_bits] * len(unique_bits)

        sites = shape_representatives
    else:
        # Reverse-order decomposition: last site first, propagating V backwards.
        sites_reversed: list[SiteUnitaryData] = []
        for i in range(num_sites - 1, 0, -1):
            site_data, v_mat = _decompose_site_with_v(mps_sites[i].to_dense(), ancilla_dim, v_from_next=v_from_next)
            sites_reversed.append(site_data)
            v_from_next = v_mat
        sites = list(reversed(sites_reversed))

    # Initial state from first tensor, absorbing V from site 1's decomposition.
    if fast_resource_estimation:
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(d * ancilla_dim)
        vec /= np.linalg.norm(vec)
        initial_state_vec = vec.tolist()
    else:
        first_tensor = mps_sites[0].to_dense()
        chi_1 = first_tensor.shape[2]
        init_state = first_tensor.transpose(1, 2, 0).sum(axis=2)  # (d, chi_1)
        init_padded = np.zeros((d, ancilla_dim))
        init_padded[:, :chi_1] = init_state
        # Absorb the V matrix from site 1 into the initial state
        if v_from_next is not None:
            v_pad = np.eye(ancilla_dim, dtype=np.float64)
            v_pad[: v_from_next.shape[0], : v_from_next.shape[1]] = np.asarray(v_from_next).real
            # V acts on the ancilla (right) dimension of init_padded
            init_padded = init_padded @ v_pad.T
        initial_state_vec_arr = init_padded.flatten()
        norm = np.linalg.norm(initial_state_vec_arr)
        if not np.isfinite(initial_state_vec_arr).all() or not np.isfinite(norm) or norm <= 1e-15:
            raise ValueError("MPS initial state must contain finite amplitudes with nonzero norm.")
        initial_state_vec_arr = initial_state_vec_arr / norm
        initial_state_vec = initial_state_vec_arr.tolist()

    return MPSPreparationData(
        initial_state_vec=initial_state_vec,
        num_sites=num_sites,
        ancilla_bits=ancilla_bits,
        sites=sites,
        site_shape_indices=site_shape_indices,
        shape_effective_bits=unique_bits,
    )


def _make_dummy_site_data(effective_dim: int) -> SiteUnitaryData:
    """Generate representative site unitary data for resource estimation.

    Creates arrays with the correct shapes (matching the Clements Givens
    decomposition output) without performing the expensive O(n³) decomposition.
    The Q# resource estimator cost depends only on array dimensions (number of
    layers, angles per layer, QROAM table sizes), not on actual angle values.
    """
    dim = effective_dim

    # V is always empty (absorbed into previous site)
    v_givens = GivensLayerData(layer_angles=[], layer_shifted=[], phases=[False] * dim)

    # W0, W1: Clements decomposition of dim*dim unitary produces exactly `dim` layers
    w0_givens = _make_dummy_givens_layers(dim)
    w1_givens = _make_dummy_givens_layers(dim)

    # UCR rotation angles: array length = dim (resource cost depends on length)
    rot_angles = [[0.1] * dim for _ in range(3)]

    # U: block-diagonal of 4 dim*dim blocks, merged into global layers
    u_givens = _make_dummy_block_diagonal_layers(dim, num_blocks=4)

    return SiteUnitaryData(v=v_givens, rot_angles=rot_angles, w0=w0_givens, w1=w1_givens, u=u_givens)


def _make_dummy_givens_layers(dim: int) -> GivensLayerData:
    """Create dummy Givens layer structure for a dim*dim Clements decomposition.

    The Clements decomposition always produces exactly `dim` layers with
    alternating even/odd pair structure. This runs in O(dim) time.
    """
    if dim <= 1:
        return GivensLayerData(layer_angles=[], layer_shifted=[], phases=[False] * max(dim, 0))

    num_layers = 1 if dim == 2 else dim
    layer_angles: list[list[float]] = []
    layer_shifted: list[bool] = []

    for i in range(num_layers):
        shifted = i % 2 == 1
        num_slots = (dim - 1) // 2 if shifted else dim // 2
        layer_angles.append([0.1] * num_slots)
        layer_shifted.append(shifted)

    phases = [False] * dim
    return GivensLayerData(layer_angles=layer_angles, layer_shifted=layer_shifted, phases=phases)


def _make_dummy_block_diagonal_layers(dim: int, num_blocks: int = 4) -> GivensLayerData:
    """Create dummy merged Givens layers for a block-diagonal unitary.

    Simulates the merge of `num_blocks` Clements decompositions (each dim*dim)
    into global layers of the full (num_blocks*dim)*(num_blocks*dim) register.
    Runs in O(num_blocks * dim) time instead of O(dim³).
    """
    total_dim = num_blocks * dim
    num_even_slots = total_dim // 2
    num_odd_slots = (total_dim - 1) // 2

    if dim <= 1:
        return GivensLayerData(layer_angles=[], layer_shifted=[], phases=[False] * total_dim)

    # Each block has `dim` layers. Simulate merge by computing how many global
    # layers are needed. In practice, blocks at different offsets can share
    # global layers when parity allows. The worst case is `dim` global layers
    # (all blocks fit in parallel). We use `dim` layers as accurate estimate.
    num_layers = 1 if dim == 2 else dim
    layer_angles: list[list[float]] = []
    layer_shifted: list[bool] = []

    # Determine global_shifted from offset of first block (offset=0 → no flip)
    for i in range(num_layers):
        shifted = i % 2 == 1
        num_slots = num_odd_slots if shifted else num_even_slots
        layer_angles.append([0.1] * num_slots)
        layer_shifted.append(shifted)

    phases = [False] * total_dim
    return GivensLayerData(layer_angles=layer_angles, layer_shifted=layer_shifted, phases=phases)


def _random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix via QR decomposition."""
    raw = rng.standard_normal((dim, dim))
    q, _ = np.linalg.qr(raw)
    return q


def _pad_and_givens(mat: np.ndarray, dim: int) -> GivensLayerData:
    """Pad a real unitary to ``dim x dim`` and decompose into Givens layers."""
    padded = np.eye(dim, dtype=np.float64)
    padded[: mat.shape[0], : mat.shape[1]] = np.asarray(mat).real
    angles, shifted, phases = decompose_unitary_to_givens(padded)
    return GivensLayerData(layer_angles=angles, layer_shifted=shifted, phases=phases)


def _d_prime_to_ucr_angles(d_prime: np.ndarray, dim: int) -> list[float]:
    """Convert a sin-component diagonal to UCR R_y angles.

    Each angle is ``2*arcsin(d'[k])``.
    """
    return [2.0 * float(np.arcsin(np.clip(d_prime[k] if k < len(d_prime) else 0.0, -1, 1))) for k in range(dim)]


def _decompose_site_with_v(
    tensor: np.ndarray, ancilla_dim: int, v_from_next: np.ndarray | None = None
) -> tuple[SiteUnitaryData, np.ndarray]:
    """CSD-decompose one MPS site tensor, returning both SiteUnitaryData and raw V.

    When ``v_from_next`` is provided (from the next site's decomposition), it is
    absorbed into this site's target matrix. The returned V is for propagation to
    the previous site (or initial state). The V in the returned SiteUnitaryData
    is always set to identity since V is never applied in the circuit — it's
    always absorbed by the previous entity (another site or the initial state).
    """
    dim = ancilla_dim

    data = compute_site_unitary_dense_data(tensor, v_from_next=v_from_next, ancilla_dim=dim)

    d_0_, d_1_, d_2_ = data["d_prime"]
    u_0, u_1, u_2, u_3 = data["u"]

    # V is never applied in the circuit — always propagated to the previous site.
    v_givens = GivensLayerData(layer_angles=[], layer_shifted=[], phases=[False] * dim)
    w0_givens = _pad_and_givens(data["w_0"], dim)
    w1_givens = _pad_and_givens(data["w_1"], dim)

    # UCR (Uniformly Controlled Rotation) angles from the 3 sin-component diagonals
    rot_angles = [
        _d_prime_to_ucr_angles(d_0_, dim),
        _d_prime_to_ucr_angles(d_1_, dim),
        _d_prime_to_ucr_angles(d_2_, dim),
    ]

    # U: block-diagonal Givens decomposition
    u_block_mats = []
    for u_b in [u_0, u_1, u_2, u_3]:
        u_pad = np.eye(dim, dtype=np.float64)
        u_pad[: u_b.shape[0], : u_b.shape[1]] = np.asarray(u_b).real
        u_block_mats.append(u_pad)
    u_angles, u_shifted, u_phases = decompose_block_diagonal_to_givens(u_block_mats)
    u_givens = GivensLayerData(layer_angles=u_angles, layer_shifted=u_shifted, phases=u_phases)

    site_data = SiteUnitaryData(
        v=v_givens,
        rot_angles=rot_angles,
        w0=w0_givens,
        w1=w1_givens,
        u=u_givens,
    )
    return site_data, data["v"]


def compute_site_unitary_dense_data(
    tensor: np.ndarray,
    v_from_next: np.ndarray | None,
    ancilla_dim: int,
) -> dict:
    r"""Compute the 7-matrix CSD of a site unitary.

    Given the MPS tensor ``M_i`` of shape (left, 4, right), the target
    isometry is ``U' = [A_0; A_1; A_2; A_3]`` where ``A_j = (M_i^j)^T``
    (each block is dim x width). This function decomposes ``U'`` into the
    7-matrix factorization from Appendix B of :cite:`Rupprecht2026` (Fig. 5)::

        U' = diag(U_0..U_3)              -- block-diagonal unitary
           · [[I,0,0,0],[0,I,0,0],       -- rotation with D_2/D_2'
              [0,0,D_2,·],[0,0,D_2',·]]
           · [[I,0,0,0],[0,I,0,0],       -- W_1 (controlled by site[0])
              [0,0,W_1,0],[0,0,0,W_1]]
           · [[I,0,0,0],[0,D_1,·,0],     -- rotation with D_1/D_1'
              [0,D_1',·,0],[0,0,0,I]]
           · [[I,0,0,0],[0,W_0,0,0],     -- W_0 (controlled by site[1])
              [0,0,W_0,0],[0,0,0,I]]
           · [[D_0,·,0,0],[D_0',·,0,0],  -- rotation with D_0/D_0'
              [0,0,D_0,·],[0,0,D_0',·]]
           · diag(V, V, V, V)            -- V (pushed to prior site)

    The circuit (Fig. 5) applies these right-to-left::

        V -> UCR(D_0') -> CNOT -> W_0 -> UCR(D_1') -> CNOT -> W_1 -> UCR(D_2') -> U

    **Algorithm (three peeling steps):**

    1. QR-decompose the lower 3
       blocks ``[A_1; A_2; A_3]`` = ``[B_2; B_3; B_4] * R``.
    2. QR-decompose the lower 2 of those: ``[B_3; B_4]`` = ``[C_3; C_4] * S``.
    3. Apply ``decompose_2d`` three times (bottom-to-top) to peel off pairs:

       - ``[C_3; C_4]`` → ``(U_2, U_3, D_2, D_2', V'')``
       - ``[B_2; S]``   → ``(U_1, _, D_1, D_1', V')``
       - ``[A_0; R]``   → ``(U_0, _, D_0, D_0', V)``

    Each ``decompose_2d`` call returns *both* diagonals ``(D, D')`` of the
    rotation block ``[[D, -D'], [D', D]]`` (see ``decompose_2d`` docs). Only
    the ``D'`` values (the sine components) are needed for the circuit's
    R_y rotation angles ``theta = 2·arcsin(D'[k])``. The mixing unitaries
    are ``W_0 = V' @ _`` and ``W_1 = V'' @ _`` (products of the intermediate
    V and U factors from adjacent peeling steps).

    Parameters
    ----------
    tensor : np.ndarray of shape (left, d, right)
        The tensor for this site (d=4 for two-qubit sites).
    v_from_next : np.ndarray or None
        The V matrix from the next site's decomposition (applied to the
        incoming ancilla register). None for the last site.
    ancilla_dim : int
        The ancilla register dimension (must be a power of 2).

    Returns
    -------
    dict
        - ``'u'``: tuple ``(u_0, u_1, u_2, u_3)`` — block-diagonal unitaries.
        - ``'d_prime'``: tuple ``(d_0', d_1', d_2')`` — sin-component diagonals
          for the 3 UCR layers.
        - ``'w_0'``, ``'w_1'``: mixing unitaries on the ancilla register.
        - ``'v'``: right unitary to be pushed to the previous site's circuit.
        - ``'ancilla_dim'``: the ancilla dimension used.

    """
    left, site_dim, _ = tensor.shape
    dim = ancilla_dim

    # Build target isometry U' = [A_0; A_1; A_2; A_3] of shape (4·dim, left).
    # Each A_j block has shape (dim, left).
    target = tensor.transpose(1, 2, 0)  # (d, right, left)
    if v_from_next is not None:
        target = np.einsum("ij,djk->dik", v_from_next, target)
    padded = np.pad(target, ((0, 0), (0, dim - target.shape[1]), (0, 0)))
    matrix = padded.reshape(site_dim * dim, left)

    u, d_prime, w_0, w_1, v = decompose_site_csd(matrix, dim)
    u_0, u_1, u_2, u_3 = u
    d_0_, d_1_, d_2_ = d_prime
    d_0_ = _pad_to_power_of_2(np.asarray(d_0_).real, dim)

    return {
        "u": (u_0, u_1, u_2, u_3),
        "d_prime": (d_0_, d_1_, d_2_),
        "w_0": w_0,
        "w_1": w_1,
        "v": v,
        "ancilla_dim": dim,
    }


def _pad_to_power_of_2(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad or truncate a 1-D array to ``target_len`` (zero-padding)."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate([arr, np.zeros(target_len - len(arr))])
