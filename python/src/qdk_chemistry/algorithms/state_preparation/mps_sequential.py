"""MPS (Matrix Product State) state preparation via sequential site unitary synthesis.

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

from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.mps_wavefunction import MPSWavefunction
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation, StatePreparationSettings

__all__: list[str] = [
    "MPSPreparationData",
    "MPSSequentialStatePreparation",
]


class MPSSequentialStatePreparationSettings(StatePreparationSettings):
    """Settings for MPS sequential state preparation."""

    def __init__(self):
        """Initialize the MPSSequentialStatePreparationSettings."""
        super().__init__()
        self._set_default("rotation_bits", "int", 10, "Phase gradient precision (number of bits).")
        self._set_default(
            "fast_resource_estimation",
            "bool",
            False,
            "Only synthesize one site unitary and replicate it for all sites. "
            "Valid for resource estimation (with BeginEstimateCaching) but not simulation.",
        )


class MPSSequentialStatePreparation(StatePreparation):
    r"""MPS (Matrix Product State) state preparation using sequential unitary synthesis.

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
        params = data.to_qsharp_params(self._settings.get("rotation_bits"))

        from qdk_chemistry.utils.qsharp import _get_mps_context

        mps_ctx = _get_mps_context()
        mps_ops = mps_ctx.code.MPSSequential

        num_state_qubits = 2 * len(wavefunction.tensors)
        num_ancilla_qubits = data.ancilla_bits
        estimate_expr = _build_mps_estimate_expr(params, num_state_qubits, num_ancilla_qubits)

        qsharp_factory = QsharpFactoryData(
            program=mps_ops.MPSSequential,
            parameter=params,
            estimate_expr=estimate_expr,
            context=mps_ctx,
        )

        return Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")

# ---------------------------------------------------------------------------
# Data containers for structured decomposition results
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

    Holds the 7-matrix CSD (Cosine-Sine Decomposition) from Appendix B of
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

    initial_state_vec: list[float]
    """Flattened initial state vector for the first site (state-preparation)."""

    num_sites: int
    """Number of MPS sites."""

    ancilla_bits: int
    """Number of ancilla qubits (log2 of ancilla dimension)."""

    sites: list[SiteUnitaryData] = field(default_factory=list)
    """Per-site decomposition data (one entry per site 1..num_sites-1)."""

    def to_qsharp_params(self, rotation_bits: int) -> dict:
        """Flatten into the dict expected by the MPSSequential Q# operation."""
        params: dict = {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "rotationBits": rotation_bits,
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
        If True, only decompose a single representative site unitary and
        replicate its data for all sites.  This is valid for resource
        estimation (where ``BeginEstimateCaching`` caches the first site
        and reuses its cost) but NOT for simulation.

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

    # Per-site decomposition
    sites: list[SiteUnitaryData] = []
    if fast_resource_estimation and num_sites > 2:
        # Only decompose one site; replicate for all others.
        # Pick the site with the largest tensor (most representative cost).
        representative_idx = max(range(1, num_sites), key=lambda i: tensors[i].size)
        site_data = _decompose_site(tensors[representative_idx], ancilla_dim)
        sites = [site_data] * (num_sites - 1)
    else:
        for i in range(1, num_sites):
            site_data = _decompose_site(tensors[i], ancilla_dim)
            sites.append(site_data)

    # Initial state from first tensor
    first_tensor = tensors[0]
    chi_left_0 = first_tensor.shape[0]
    chi_1 = first_tensor.shape[2]
    # Transpose to (d, chi_right, chi_left) then sum over chi_left → (d, chi_right)
    init_state = first_tensor.transpose(1, 2, 0).sum(axis=2)  # (d, chi_1)
    init_padded = np.zeros((d, ancilla_dim))
    init_padded[:, :chi_1] = init_state
    initial_state_vec = init_padded.flatten()
    norm = np.linalg.norm(initial_state_vec)
    if norm > 1e-15:
        initial_state_vec = initial_state_vec / norm

    return MPSPreparationData(
        initial_state_vec=initial_state_vec.tolist(),
        num_sites=num_sites,
        ancilla_bits=ancilla_bits,
        sites=sites,
    )


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


def _decompose_site(tensor: np.ndarray, ancilla_dim: int) -> SiteUnitaryData:
    """CSD-decompose one MPS site tensor and Givens-decompose all components.

    Combines ``compute_site_unitary_dense_data``
    (CSD = Cosine-Sine Decomposition) with Givens decomposition of each
    resulting unitary, returning a single ``SiteUnitaryData`` ready for
    circuit synthesis.
    """
    dim = ancilla_dim
    data = compute_site_unitary_dense_data(tensor, v_from_next=None, ancilla_dim=dim)

    d_0_, d_1_, d_2_ = data["d_prime"]
    u_0, u_1, u_2, u_3 = data["u"]

    # Givens decompositions of V, W_0, W_1
    v_givens = _pad_and_givens(data["v"], dim)
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

    return SiteUnitaryData(
        v=v_givens,
        rot_angles=rot_angles,
        w0=w0_givens,
        w1=w1_givens,
        u=u_givens,
    )


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


def _decompose_unitary_to_givens_python(matrix: np.ndarray):
    """Decompose a real orthogonal matrix into parallel Givens rotation layers.

    Implements the :cite:`Clements2017` decomposition used
    for fault-tolerant unitary synthesis via phase-gradient rotations.

    The matrix is factored as ``U = D · L_l · ... · L_1`` where each layer
    ``L_j`` consists of parallel 2x2 R_y(theta) (Y-axis rotation) Givens
    rotations acting on neighboring pairs of columns, and ``D`` is a diagonal
    sign matrix (±1).
    Layers alternate between "even" (pairs 0-1, 2-3, ...) and "odd"
    (pairs 1-2, 3-4, ...) forms as in Eq. (3) of the paper.

    This is a pure-Python fallback; the Rust ``unitary_synthesis`` library
    is preferred for performance.

    The decomposition uses right-multiplication (column elimination):
    ``M · G†_1 · ... · G†_N = D``, giving ``M = D · G_N · ... · G_1``.
    Layers are stored in decomposition order so that the Q# circuit
    (which applies layers sequentially then D) produces the correct unitary.

    Parameters
    ----------
    matrix : np.ndarray
        Real orthogonal matrix of shape (dim, dim).

    Returns
    -------
    layer_angles : list[list[float]]
        Per-layer R_y rotation angles for each parallel slot.
    layer_shifted : list[bool]
        Whether each layer uses odd-indexed pairs (shifted=True) or
        even-indexed pairs (shifted=False).
    phases : list[bool]
        Diagonal sign flips: True where the residual diagonal entry is -1.

    """
    dim = matrix.shape[0]
    m = matrix.copy().astype(float)

    rotations: list[tuple[int, float]] = []

    # Right-multiplication: zero off-diagonal elements column by column.
    # For each row, eliminate elements to the right of the diagonal using
    # Givens rotations on neighboring column pairs.
    # M · G(θ_1) · ... · G(θ_N) = D  =>  M = D · G(-θ_N) · ... · G(-θ_1)
    # We store -θ so the circuit can directly apply G(stored_angle).
    for row in range(dim - 1):
        for col in range(dim - 1, row, -1):
            a_val = m[row, col - 1]
            b_val = m[row, col]
            if abs(b_val) < 1e-15:
                continue
            theta = np.arctan2(b_val, a_val)
            c, s = np.cos(theta), np.sin(theta)
            col_i = m[:, col - 1].copy()
            col_j = m[:, col].copy()
            m[:, col - 1] = c * col_i + s * col_j
            m[:, col] = -s * col_i + c * col_j
            rotations.append((col - 1, -theta))

    diag_vals = np.diag(m)
    phases = [bool(d < 0) for d in diag_vals]

    # Organize into parallel layers
    layer_angles, layer_shifted = _organize_into_layers(rotations, dim)
    return layer_angles, layer_shifted, phases


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


def _organize_into_layers(rotations: list[tuple[int, float]], dim: int) -> tuple[list[list[float]], list[bool]]:
    """Assign Givens rotations to parallel layers (Clements layout).

    Maps a sequence of (pair_index, angle) rotations into layers that
    alternate between even and odd pair placements, ensuring no two
    rotations in the same layer share a qubit index.

    Parameters
    ----------
    rotations : list of (int, float)
        Each entry is (pair_index, angle) where pair_index indicates the
        rotation acts on rows pair_index and pair_index+1.
    dim : int
        Dimension of the unitary (determines number of slots per layer).

    Returns
    -------
    layer_angles : list[list[float]]
        Angles for each slot in each layer.
    layer_shifted : list[bool]
        Whether the layer uses odd-indexed pairs.

    """
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

        if (is_odd and min_layer % 2 == 0) or (not is_odd and min_layer % 2 == 1):
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
    """Decompose a real orthogonal matrix into parallel Givens rotation layers.

    Produces the factorization ``U = D · L_l · ... · L_1`` (Eq. 7 in
    :cite:`Rupprecht2026`) where each ``L_j`` is a layer of
    parallel R_y rotations and ``D`` is a diagonal ±1 sign matrix.

    Uses the Rust ``unitary_synthesis`` library when available for
    performance; falls back to a pure-Python implementation.

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

    For the block-diagonal matrix ``diag(u_0, u_1, u_2, u_3)`` from the CSD
    decomposition, this exploits the block structure so that rotations within
    each block can be scheduled in parallel across blocks, reducing the total
    number of Givens layers compared to treating the full matrix as dense.
    See Sec. 3 and Fig. 3 of :cite:`Rupprecht2026`.

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