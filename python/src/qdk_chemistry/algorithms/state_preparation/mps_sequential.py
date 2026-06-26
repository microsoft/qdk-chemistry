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

from collections import deque

import numpy as np

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction
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
        self._set_default("rotation_bits", "int", 10, "Phase gradient precision.")
        self._set_default(
            "fast_resource_estimation",
            "bool",
            False,
            "Only synthesize one site unitary to reduce classical preprocessing overhead. "
            "Valid for resource estimation (with BeginEstimateCaching) but not simulation.",
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

    def _run_impl(self, wavefunction: MPSWavefunction) -> Circuit:
        """Return a circuit to prepare an MPS state.

        Args:
            wavefunction: An MPSWavefunction containing the tensors.

        Returns:
            A Circuit object implementing the MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an MPSWavefunction instance.

        """
        if not isinstance(wavefunction, MPSWavefunction):
            raise TypeError(f"MPSSequentialStatePreparation requires an MPSWavefunction, got {type(wavefunction)}.")

        fast_re = self._settings.get("fast_resource_estimation")

        data = generate_mps_preparation_data(wavefunction.tensors, fast_resource_estimation=fast_re)

        rotation_bits = self._settings.get("rotation_bits")

        if fast_re:
            params = data.to_qsharp_params_grouped(rotation_bits)
            program = QSHARP_UTILS.MPSSequential.MakeMPSSequentialCircuitGrouped
        else:
            params = data.to_qsharp_params(rotation_bits)
            program = QSHARP_UTILS.MPSSequential.MakeMPSSequentialCircuit

        qsharp_factory = QsharpFactoryData(
            program=program,
            parameter=params,
        )

        # Build composable op for QPE composition
        if fast_re:
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

    def to_qsharp_params(self, rotation_bits: int) -> dict:
        """Flatten into the dict expected by the MakeMPSSequentialCircuit Q# operation."""
        params: dict = {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
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

    def to_qsharp_params_grouped(self, rotation_bits: int) -> dict:
        """Flatten into the dict expected by MakeMPSSequentialCircuitGrouped.

        Passes one representative per unique shape + a site-to-shape mapping.
        This minimizes data transfer while enabling accurate per-shape caching
        in the Q# resource estimator.
        """
        assert self.site_shape_indices is not None, "Grouped mode requires site_shape_indices"
        assert self.shape_effective_bits is not None, "Grouped mode requires shape_effective_bits"

        params: dict = {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
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


# ---------------------------------------------------------------------------
# Unitary synthesis helpers
# ---------------------------------------------------------------------------


def generate_mps_preparation_data(
    tensors: Sequence[np.ndarray],
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
            chi_left, _, chi_right = tensors[i].shape
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
            site_data, v_mat = _decompose_site_with_v(tensors[i], ancilla_dim, v_from_next=v_from_next)
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
        first_tensor = tensors[0]
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
        if norm > 1e-15:
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

    Uses random orthogonal matrices to produce realistic Givens layer counts
    and array dimensions, matching what the full CSD pipeline would produce
    for a generic (non-trivial) unitary. This is much cheaper than the full
    pipeline (skips tensor reshaping, multi-block QR, and 3* SVD) while
    giving accurate resource estimates.
    """
    dim = effective_dim
    rng = np.random.default_rng(0)  # fixed seed for reproducibility

    # V is always empty (absorbed into previous site)
    v_givens = GivensLayerData(layer_angles=[], layer_shifted=[], phases=[False] * dim)

    # W0, W1: decompose random dim x dim orthogonal matrices
    w0_angles, w0_shifted, w0_phases = decompose_unitary_to_givens(_random_orthogonal(dim, rng))
    w0_givens = GivensLayerData(layer_angles=w0_angles, layer_shifted=w0_shifted, phases=w0_phases)

    w1_angles, w1_shifted, w1_phases = decompose_unitary_to_givens(_random_orthogonal(dim, rng))
    w1_givens = GivensLayerData(layer_angles=w1_angles, layer_shifted=w1_shifted, phases=w1_phases)

    # UCR rotation angles: random values (resource cost depends on array length, not values)
    rot_angles = [rng.uniform(-1, 1, size=dim).tolist() for _ in range(3)]

    # U: block-diagonal of 4 random dim x dim orthogonal blocks
    blocks = [_random_orthogonal(dim, rng) for _ in range(4)]
    u_angles, u_shifted, u_phases = decompose_block_diagonal_to_givens(blocks)
    u_givens = GivensLayerData(layer_angles=u_angles, layer_shifted=u_shifted, phases=u_phases)

    return SiteUnitaryData(v=v_givens, rot_angles=rot_angles, w0=w0_givens, w1=w1_givens, u=u_givens)


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

    # --- Step 1: QR on lower 3 blocks [A_1; A_2; A_3] ---
    # b_full (3*dim x 3*dim) has orthonormal columns, r (3*dim x left) is upper triangular.
    # Split b_full into B_2 = b_full[:dim], and [B_3; B_4] = b_full[dim:].
    b_full, r = np.linalg.qr(matrix[dim:, :], mode="complete")

    # --- Step 2: QR on lower 2 of B: [B_3; B_4] ---
    # c (2*dim x 2*dim) -> C_3 = c[:dim], C_4 = c[dim:].
    # s (2*dim x dim) is upper triangular.
    c, s = np.linalg.qr(b_full[dim:, :dim], mode="complete")

    # --- Step 3: Three peeling decompositions (bottom → top) ---
    # Peel [C_3; C_4] → rotation D_2/D_2' and unitaries U_2, U_3
    u_2, u_3, _d_2, d_2_, v__ = decompose_2d(c[:dim, :dim], c[dim:, :dim])

    # Peel [B_2; S] → rotation D_1/D_1' and unitary U_1
    u_1, u_dummy, _d_1, d_1_, v_ = decompose_2d(b_full[:dim, :dim], s[:dim, :])

    # Peel [A_0; R] → rotation D_0/D_0' and unitary U_0
    u_0, u_top, _d_0, d_0_, v = decompose_2d(matrix[:dim, :], r[:dim, :])

    d_0_ = _pad_to_power_of_2(np.asarray(d_0_).real, dim)
    d_1_ = np.asarray(d_1_).real
    d_2_ = np.asarray(d_2_).real

    # Mixing unitaries: W_0 = V' @ U_top,  W_1 = V'' @ U_dummy
    # (products of the intermediate right-unitary and leftover factor from
    # the adjacent peeling step)
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


def decompose_2d(a: np.ndarray, b: np.ndarray):
    r"""Decompose a 2-block column matrix via SVD + polar decomposition.

    Given matrices ``a`` (shape m, k) and ``b`` (shape m, k) whose vertical
    stack ``[a; b]`` has orthonormal columns, compute the factorization from
    Eq. (30) of :cite:`Berry2025` and the Lemma in
    Appendix B of :cite:`Rupprecht2026`::

        [a]   [u_1   0 ] [D_1] [v]
        [b] = [ 0   u_2] [D_2] [v]

    where ``u_1``, ``u_2`` are unitary (m x m), ``v`` is unitary (k x k),
    and ``D_1``, ``D_2`` are real diagonal (m x k) matrices satisfying
    ``D_1^2 + D_2^2 = I``.

    **Why two diagonal vectors are returned:** The full middle block
    ``[[D_1, -D_2], [D_2, D_1]]`` has orthonormal columns and encodes a
    rotation: each pair ``(D_1[j], D_2[j])`` satisfies ``cos^2 + sin^2 = 1``.
    On the quantum circuit, only ``D_2`` (= sin component) is needed to set
    the R_y rotation angle ``theta = 2*arcsin(D_2[j])``; ``D_1`` (= cos) is
    implied. Both are returned so callers can verify the decomposition.

    **Algorithm:** ``D_1`` and ``v`` come from the SVD of ``a``. Then ``D_2``
    and ``u_2`` come from the polar decomposition of ``b @ v^H``, which
    yields a guaranteed-diagonal ``D_2`` (unlike a QR decomposition which
    can fail when ``b`` is rank-deficient).

    Parameters
    ----------
    a : np.ndarray of shape (m, k)
        Upper block of the column matrix.
    b : np.ndarray of shape (m, k)
        Lower block of the column matrix.

    Returns
    -------
    u_1 : np.ndarray of shape (m, m)
        Left unitary for the upper block (from SVD of ``a``).
    u_2 : np.ndarray of shape (m, m)
        Left unitary for the lower block (from polar decomposition).
    d_1 : np.ndarray of shape (k,)
        Singular values of ``a``; the cos-like diagonal.
    d_2 : np.ndarray of shape (k,)
        Polar factor diagonal of ``b @ v^H``; the sin-like diagonal.
    v : np.ndarray of shape (k, k)
        Right unitary (shared by both blocks).

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
    """Pad or truncate a 1-D array to ``target_len`` (zero-padding)."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.concatenate([arr, np.zeros(target_len - len(arr))])


def decompose_unitary_to_givens(matrix: np.ndarray):
    """Decompose a real orthogonal matrix into parallel Givens rotation layers.

    Implements the Clements double-sided decomposition
    (:cite:`Clements2017`) which guarantees exactly ``dim`` layers for a
    ``dim x dim`` orthogonal matrix by alternating right-column and
    left-row Givens eliminations.

    The matrix is factored as ``U = D · L_d · ... · L_1`` where each layer
    ``L_j`` consists of parallel 2x2 R_y(theta) (Y-axis rotation) Givens
    rotations acting on neighboring pairs of columns, and ``D`` is a diagonal
    sign matrix (±1).
    Layers alternate between "even" (pairs 0-1, 2-3, ...) and "odd"
    (pairs 1-2, 3-4, ...) forms as in Eq. (3) of the paper.

    Produces the factorization ``U = D · L_d · ... · L_1`` (Eq. 7 in
    :cite:`Rupprecht2026`) where each ``L_j`` is a layer of
    parallel R_y rotations and ``D`` is a diagonal ±1 sign matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Real orthogonal matrix of shape (dim, dim). dim must be a power of 2.

    Returns
    -------
    layer_angles : list[list[float]]
        Per-layer R_y rotation angles for each parallel slot.
    layer_shifted : list[bool]
        Whether each layer uses odd-indexed pairs (True) or even (False).
    phases : list[bool]
        Diagonal sign flips (True where entry is -1).

    """
    dim = matrix.shape[0]
    m = matrix.copy().astype(float)

    if dim <= 1:
        return [], [], [bool(m[0, 0] < 0)] if dim == 1 else []

    # Clements double-sided elimination.
    # Layer i is shifted=(i%2==1): even layers have even pairs, odd layers odd pairs.
    num_layers = 1 if dim == 2 else dim
    # upper_rots[i] collects right-side rotations for layer i
    upper_rots: list[list[tuple[int, float]]] = [[] for _ in range(num_layers)]
    # lower_rots[col_idx] collects left-side rotations (combined into layers later)
    lower_rots: list[list[tuple[int, float]]] = [[] for _ in range(num_layers)]

    for k in range(dim - 1):
        if k % 2 == 0:
            # Right column elimination: zero M[row_idx, col_idx] for decreasing col_idx
            for i, col_idx in enumerate(range(k, -1, -1)):
                row_idx = dim - 1 - i
                a_val = m[row_idx, col_idx + 1]
                b_val = m[row_idx, col_idx]
                if abs(b_val) < 1e-15:
                    continue
                theta = np.arctan2(b_val, a_val)
                c, s = np.cos(theta), np.sin(theta)
                col_i = m[:, col_idx].copy()
                col_j = m[:, col_idx + 1].copy()
                # Zero col_idx: [[c, s], [-s, c]] right-multiplied on cols (col_idx, col_idx+1)
                m[:, col_idx] = c * col_i - s * col_j
                m[:, col_idx + 1] = s * col_i + c * col_j
                upper_rots[i].append((col_idx, theta))
        else:
            # Left row elimination: zero M[row_idx, col_idx] for increasing row_idx
            for col_idx, row_idx in enumerate(range(dim - k - 1, dim)):
                a_val = m[row_idx - 1, col_idx]
                b_val = m[row_idx, col_idx]
                if abs(b_val) < 1e-15:
                    continue
                theta = np.arctan2(b_val, a_val)
                c, s = np.cos(theta), np.sin(theta)
                row_i = m[row_idx - 1, :].copy()
                row_j = m[row_idx, :].copy()
                m[row_idx - 1, :] = c * row_i + s * row_j
                m[row_idx, :] = -s * row_i + c * row_j
                lower_rots[col_idx].append((row_idx - 1, theta))

    # Extract diagonal (±1 for orthogonal matrices)
    diag_vals = np.diag(m)
    phases = [bool(d < 0) for d in diag_vals]

    # Combine upper and lower rotations into exactly dim layers.
    # Lower rotations (left-side) are commuted past the diagonal D to become
    # right-side rotations: angle is adjusted by sign(D[p]*D[p+1]).
    # lower_rots[col_idx] maps to layer (num_layers - 1 - col_idx).
    num_even_slots = dim // 2
    num_odd_slots = (dim - 1) // 2

    result_angles: list[list[float]] = []
    result_shifted: list[bool] = []

    for layer_idx in range(num_layers):
        shifted = layer_idx % 2 == 1
        if shifted:
            num_slots = num_odd_slots
            slot_pairs = [2 * k + 1 for k in range(num_slots)]
        else:
            num_slots = num_even_slots
            slot_pairs = [2 * k for k in range(num_slots)]

        pair_to_slot = {p: i for i, p in enumerate(slot_pairs)}
        angles = [0.0] * num_slots

        # Add upper (right-side) rotations for this layer
        for pair, angle in upper_rots[layer_idx]:
            if pair in pair_to_slot:
                angles[pair_to_slot[pair]] = angle

        # Add lower (left-side) rotations, adjusted by diagonal signs.
        # lower_rots[col_idx] → combined into layer (num_layers - 1 - col_idx)
        lower_col_idx = num_layers - 1 - layer_idx
        if lower_col_idx < len(lower_rots):
            for pair, angle in reversed(lower_rots[lower_col_idx]):
                sign = 1.0 if diag_vals[pair] * diag_vals[pair + 1] > 0 else -1.0
                adjusted_angle = angle * sign
                if pair in pair_to_slot:
                    angles[pair_to_slot[pair]] = adjusted_angle

        # Only include non-trivial layers
        if any(abs(a) > 1e-15 for a in angles):
            result_angles.append(angles)
            result_shifted.append(shifted)

    return result_angles, result_shifted, phases


def decompose_block_diagonal_to_givens(blocks: list[np.ndarray]):
    """Decompose a block-diagonal real unitary into merged Givens rotation layers.

    For the block-diagonal matrix ``diag(u_0, u_1, u_2, u_3)`` from the CSD
    decomposition, this exploits the block structure so that rotations within
    each block can be scheduled in parallel across blocks, reducing the total
    number of Givens layers compared to treating the full matrix as dense.
    See Sec. 3 and Fig. 3 of :cite:`Rupprecht2026`.

    Each block is decomposed independently using the Clements double-sided
    algorithm (guaranteed ``block_dim`` layers per block), then layers from
    different blocks are merged into global layers of the full register.
    This matches the Qualtran Rust implementation's ``normal_form_blocks``.

    Parameters
    ----------
    blocks : list[np.ndarray]
        List of real orthogonal matrices forming the diagonal blocks.

    Returns
    -------
    layer_angles : list[list[float]]
        Per-layer R_y rotation angles for each parallel slot.
    layer_shifted : list[bool]
        Whether each layer uses odd-indexed pairs (True) or even (False).
    phases : list[bool]
        Diagonal sign flips (True where entry is -1).

    """
    total_dim = sum(b.shape[0] for b in blocks)
    num_even_slots = total_dim // 2
    num_odd_slots = (total_dim - 1) // 2

    # Decompose each block independently using Clements decomposition.
    block_starts: list[int] = []
    block_decomps: list[list[tuple[bool, list[tuple[int, float]]]]] = []
    all_phases: list[bool] = [False] * total_dim

    offset = 0
    for block in blocks:
        block_starts.append(offset)
        block_dim = block.shape[0]
        angles, shifted, phases = decompose_unitary_to_givens(block)

        # Convert layer data back to (pair, angle) tuples remapped to full register
        layers: list[tuple[bool, list[tuple[int, float]]]] = []
        for layer_angles_l, layer_shifted_l in zip(angles, shifted, strict=False):
            rots: list[tuple[int, float]] = []
            if layer_shifted_l:
                slot_pairs = [2 * k + 1 for k in range((block_dim - 1) // 2)]
            else:
                slot_pairs = [2 * k for k in range(block_dim // 2)]
            for slot_idx, angle in enumerate(layer_angles_l):
                if abs(angle) > 1e-15:
                    rots.append((offset + slot_pairs[slot_idx], angle))
            if rots:
                layers.append((layer_shifted_l, rots))

        block_decomps.append(layers)

        # Store phases remapped to full register
        for k, p in enumerate(phases):
            all_phases[offset + k] = p

        offset += block_dim

    # Merge layers across blocks into global layers.
    # A block's layer with local shifted=S at block offset O fits in a global
    # layer with global_shifted=G if (O + S) and G have the same parity
    # (i.e., the local pair indices map to global pairs of the correct type).
    # Determine initial global_shifted from the largest block's first layer.
    max_block_idx = max(range(len(blocks)), key=lambda i: blocks[i].shape[0])
    if block_decomps[max_block_idx]:
        first_local_shifted = block_decomps[max_block_idx][0][0]
        # A locally shifted layer at offset O produces pairs O+1, O+3, ...
        # These fit in a global shifted layer if O is even, or unshifted if O is odd.
        global_shifted = first_local_shifted ^ (block_starts[max_block_idx] % 2 == 1)
    else:
        global_shifted = False

    # Convert block_decomps to deques for popping from front
    block_queues = [deque(layers) for layers in block_decomps]

    result_angles: list[list[float]] = []
    result_shifted: list[bool] = []

    while any(q for q in block_queues):
        current_offset = 1 if global_shifted else 0

        # Collect rotations from blocks whose front layer aligns with current global layer
        layer_rots: list[tuple[int, float]] = []
        for blk_idx, q in enumerate(block_queues):
            if not q:
                continue
            local_shifted, rots = q[0]
            # Check alignment: a locally shifted layer at block start O
            # produces global pairs at O+1, O+3, ... which are all odd if O is even.
            # These fit in global_shifted=True layer.
            block_offset = block_starts[blk_idx]
            local_parity = (block_offset + (1 if local_shifted else 0)) % 2
            global_parity = current_offset % 2
            if local_parity == global_parity:
                q.popleft()
                layer_rots.extend(rots)

        # Build the global layer
        if global_shifted:
            num_slots = num_odd_slots
            slot_pairs = [2 * k + 1 for k in range(num_slots)]
        else:
            num_slots = num_even_slots
            slot_pairs = [2 * k for k in range(num_slots)]

        pair_to_slot = {p: i for i, p in enumerate(slot_pairs)}
        angles_layer = [0.0] * num_slots

        for pair, angle in layer_rots:
            if pair in pair_to_slot:
                angles_layer[pair_to_slot[pair]] = angle

        if any(abs(a) > 1e-15 for a in angles_layer):
            result_angles.append(angles_layer)
            result_shifted.append(global_shifted)

        global_shifted = not global_shifted

    return result_angles, result_shifted, all_phases
