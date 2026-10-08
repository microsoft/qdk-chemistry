"""Matrix Product State (MPS) state preparation.

Implements the MPS state preparation algorithm based on :cite:`Berry2025`, which prepares the
state one site at a time using an ancilla register that stores the virtual bond. The
``unitary_synthesis`` setting selects how each site unitary is synthesized:

``"general"``
    Each site unitary is decomposed based on Appendix B in :cite:`Rupprecht2026`, with orthogonal
    factors synthesized as parallel Givens rotation layers using the elimination schedule of
    :cite:`Clements2016`. The cost depends only on the bond dimensions.

``"block_sparse"``
    Each site unitary is decomposed as ``U = P_row · V_blockdiag · P_col`` following
    :cite:`Rupprecht2026`, where ``P_row`` and ``P_col`` are permutations (a table lookup of the
    permuted index, a SWAP, and erasure of the old index by X-basis measurement with a phase
    fixup) and ``V_blockdiag`` is block diagonal, with each block synthesized from Givens rotation
    layers. This exploits U(1) symmetries (particle number, spin) that make MPS tensors block
    sparse, yielding 10-30x Toffoli savings over ``"general"``.

Attribution
-----------
The unitary synthesis is based on the method described in :cite:`Rupprecht2026` and the Qualtran
implementation by Felix Rupprecht (DLR) published on Zenodo :cite:`Rupprecht2026Zenodo` under
Apache 2.0 license. The implementation has been rewritten for integration into QDK Chemistry.

References
----------
    Felix Rupprecht and Sabine Wölk. (2026). Faster matrix product state preparation by
    exploiting symmetry-induced block-sparsity.
    https://arxiv.org/pdf/2605.28489. Zenodo: https://zenodo.org/records/20393500.

    Dominic W. Berry et al. (2025). Rapid Initial-State Preparation for the Quantum Simulation of
    Strongly Correlated Molecules. PRX Quantum 6, 020327.
    https://doi.org/10.1103/PRXQuantum.6.020327.

    William R. Clements et al. (2016). Optimal design for universal multiport interferometers.
    Optica 3, 1460-1465. https://doi.org/10.1364/OPTICA.3.001460.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.data import Configuration, MPSContainer, MPSSite, Settings, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.data.symmetry import (
    SymmetryBlockedTensorRank3,
    SymmetryBlockedTensorRank3Complex,
    SymmetryLabel,
    SymmetryProduct,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS
from qdk_chemistry.utils.unitary_synthesis import (
    block_sparse_unitary_synthesis,
    decompose_mps,
    dense_unitary_synthesis,
)

from .state_preparation import StatePreparation

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__: list[str] = [
    "MatrixProductStatePreparation",
]

_DESCRIPTION = "MPS state preparation"
_UNITARY_SYNTHESIS_METHODS = ["general", "block_sparse"]

# Supported physical bases by the number of physical states per site. A site with two states
# is one spinless mode on one qubit; a site with four states is one spatial orbital on two
# qubits, its alpha and beta modes.
_PHYSICAL_BASIS_LABELS = {2: ("0", "1"), 4: ("0", "u", "d", "2")}
_PHYSICAL_BASES = {
    2: [Configuration.from_bitstring(state) for state in _PHYSICAL_BASIS_LABELS[2]],
    4: [Configuration.from_spin_half_string(state) for state in _PHYSICAL_BASIS_LABELS[4]],
}
_DIMENSION_ERROR = "MPS state preparation requires two or four physical states per site."


class MatrixProductStatePreparationSettings(Settings):
    """Settings for matrix product state preparation."""

    def __init__(self):
        """Initialize the MatrixProductStatePreparationSettings."""
        super().__init__()
        self._set_default("rotation_bits", "int", 10, "Phase gradient precision.", (2, 62))
        self._set_default(
            "unitary_synthesis",
            "string",
            "general",
            "Site unitary synthesis: 'general' factors every site unitary over the full bond, and "
            "'block_sparse' factors it into permutations and a block-diagonal unitary.",
            _UNITARY_SYNTHESIS_METHODS,
        )
        self._set_default(
            "fast_resource_estimation",
            "bool",
            False,
            "Replace the site decompositions by placeholder data with the same gate structure. "
            "The circuit is valid for resource estimation only and does not prepare the state. "
            "The qubit count is exact; with a ('0', '1') physical basis the Toffoli count is an "
            "upper bound that is tight for sites whose bonds fill the ancilla register. "
            "Requires unitary_synthesis='general'.",
        )


class MatrixProductStatePreparation(StatePreparation):
    r"""Matrix Product State (MPS) state preparation.

    Prepare the state sequentially, one site at a time, using an ancilla
    register that stores the virtual bond. The ``unitary_synthesis`` setting
    selects how each site unitary is synthesized:

    * ``"general"`` decomposes the site unitary based on Appendix B in
      :cite:`Rupprecht2026` and synthesizes it from Givens rotation layers with
      QROM-loaded angles and phase gradient rotations. Every site unitary acts
      on the full ancilla register, so with ``fast_resource_estimation`` the
      cost can be estimated from the bond dimensions alone. The placeholder
      gives every block unitary a full set of Givens layers. For the
      ``('0', 'u', 'd', '2')`` basis the mixing unitaries make that the typical
      case. For the ``('0', '1')`` basis, sites whose bonds are smaller than the
      ancilla register, typically near the chain ends, need fewer layers, so
      the Toffoli estimate is an upper bound.
    * ``"block_sparse"`` factors the site unitary as
      ``U = P_row · V_blockdiag · P_col``, where permutations are implemented
      by a table lookup and a SWAP whose leftover register is erased by
      measurement, and the block-diagonal unitary is synthesized via Givens
      rotation layers. This exploits the block-sparse structure of MPS tensors
      arising from U(1) symmetries (particle number, spin conservation).

    Sites with the ``('0', 'u', 'd', '2')`` physical basis are spatial
    orbitals in the blocked Jordan-Wigner layout: qubit ``o`` holds the alpha
    mode of orbital ``o`` and qubit ``num_orbitals + o`` its beta mode. Sites
    with the ``('0', '1')`` physical basis are spinless modes in the
    Jordan-Wigner layout, with qubit ``o`` holding mode ``o``. MPS basis states
    create the modes of each site in chain order, alpha before beta; a final
    layer of CZ gates applies the fermionic signs of reordering these modes
    into qubit order.

    Attribution
    -----------
    The unitary synthesis is based on the method in :cite:`Rupprecht2026` and code
    originally published by Felix Rupprecht on Zenodo :cite:`Rupprecht2026Zenodo`
    under Apache 2.0 license. The implementation has been rewritten for integration
    into QDK Chemistry.
    """

    def __init__(self):
        """Initialize the matrix product state preparation algorithm."""
        super().__init__()
        self._settings = MatrixProductStatePreparationSettings()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"matrix_product_state"``

        """
        return "matrix_product_state"

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Return a circuit to prepare an MPS state.

        Args:
            wavefunction: The wavefunction to prepare.

        Returns:
            A Circuit object implementing the MPS state preparation. With
            ``fast_resource_estimation`` the circuit only supports resource estimation.

        Raises:
            TypeError: If wavefunction is not an MPSContainer instance.
            ValueError: If ``fast_resource_estimation`` is combined with ``"block_sparse"``
                synthesis, or if the MPS is complex, not right-canonical with orthogonality
                center zero, does not use one of the ``('0', '1')`` and ``('0', 'u', 'd', '2')``
                physical bases on every site, or does not have exactly one site per molecular
                orbital.

        """
        rotation_bits = self._settings.get("rotation_bits")
        unitary_synthesis = self._settings.get("unitary_synthesis")
        fast_resource_estimation = self._settings.get("fast_resource_estimation")
        if fast_resource_estimation and unitary_synthesis != "general":
            raise ValueError("fast_resource_estimation requires unitary_synthesis='general'.")

        container = wavefunction.get_container()
        if not isinstance(container, MPSContainer):
            raise TypeError(f"MatrixProductStatePreparation requires an MPSContainer, got {type(container)}.")
        if container.orthogonality_center != 0:
            raise ValueError(f"{_DESCRIPTION} requires a right-canonical MPS with center zero.")
        if container.num_sites != container.orbitals.get_num_molecular_orbitals():
            raise ValueError(f"{_DESCRIPTION} requires exactly one MPS site per molecular orbital.")

        if fast_resource_estimation:
            # Only the chain length and the bond dimensions are read, so no site is densified or
            # decomposed. The initial state is a fixed pseudo-random vector of the right size.
            sites = container.sites
            if container.is_complex:
                raise ValueError(f"{_DESCRIPTION} currently supports only real-valued MPS tensors.")
            _validate_physical_basis(sites)
            ancilla_bits = _ancilla_bits(container.max_bond_dimension)
            vector = np.random.default_rng(0).standard_normal(sites[0].physical_dimension << ancilla_bits)
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.MPSSequential.MakeMPSSequentialPlaceholderCircuit,
                parameter={
                    "initialStateVec": (vector / np.linalg.norm(vector)).tolist(),
                    "numSites": len(sites),
                    "numQubitsPerSite": _qubits_per_site(sites),
                    "siteToOrbitalOrder": container.site_to_orbital_order,
                    "rotationBits": rotation_bits,
                    "numAncillaQubits": ancilla_bits,
                },
            )
            return Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")

        data = generate_matrix_product_state_preparation_data(container, unitary_synthesis)
        params = data.to_qsharp_params(rotation_bits)
        if unitary_synthesis == "block_sparse":
            # MakeMPSSparseCircuit is Adaptive-only; the default shared context targets Adaptive_RIF.
            qsharp_factory = QsharpFactoryData(program=QSHARP_UTILS.MPSSparse.MakeMPSSparseCircuit, parameter=params)
            return Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")

        # MakeMPSSequentialCircuit is Adaptive-only; the default shared context targets Adaptive_RIF.
        qsharp_factory = QsharpFactoryData(
            program=QSHARP_UTILS.MPSSequential.MakeMPSSequentialCircuit,
            parameter=params,
        )
        op_params = QSHARP_UTILS.MPSSequential.MPSSequentialParams(**params)
        qsharp_op = QSHARP_UTILS.MPSSequential.MakeMPSSequentialOp(op_params)
        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            encoding="jordan-wigner",
            num_qubits=data.num_qubits_per_site * data.num_sites,
        )


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
class GeneralSiteUnitaryData:
    r"""Decomposition data for a single MPS site unitary with ``"general"`` synthesis.

    Holds the Cosine-Sine Decomposition (CSD) from Appendix B of
    :cite:`Rupprecht2026` and the Givens-layer synthesis of each component.

    A site with four physical states applies these components in order (see
    Fig. 5 of the paper)::

        UCR(d_0') -> CNOT -> W_0 -> UCR(d_1') -> CNOT -> W_1 -> UCR(d_2') -> U

    where each UCR (Uniformly Controlled Rotation) is a multiplexed R_y
    rotation addressed by the ancilla register, CNOT is a Controlled-NOT
    gate, and U = diag(u_0, u_1, u_2, u_3) is block-diagonal. A site with two
    physical states applies ``UCR(d_0') -> U`` with U = diag(u_0, u_1). The
    right factor V of the decomposition is absorbed into the preceding site or
    the initial state.
    """

    rot_angles: list[list[float]]
    """UCR (Uniformly Controlled Rotation) angles of each rotation step.

    Format: ``[rot0, rot1, rot2]`` for four physical states and ``[rot0]`` for two.
    """

    u: GivensLayerData
    """Givens layers for U (block-diagonal unitary on ancilla+site)."""

    w0: GivensLayerData | None = None
    """Givens layers for W_0 (mixing unitary, controlled by site[0]); None for two physical states."""

    w1: GivensLayerData | None = None
    """Givens layers for W_1 (mixing unitary, controlled by site[1]); None for two physical states."""

    def to_qsharp(self) -> dict:
        """Return the fields of the Q# ``SequentialSiteDecomposition`` struct."""
        rotations = [*self.rot_angles, [], []][:3]
        empty = GivensLayerData(layer_angles=[], layer_shifted=[], phases=[])
        w0 = self.w0 or empty
        w1 = self.w1 or empty
        return {
            "rot0Angles": rotations[0],
            "rot1Angles": rotations[1],
            "rot2Angles": rotations[2],
            "w0LayerAngles": w0.layer_angles,
            "w0LayerShifted": w0.layer_shifted,
            "w0Phases": w0.phases,
            "w1LayerAngles": w1.layer_angles,
            "w1LayerShifted": w1.layer_shifted,
            "w1Phases": w1.phases,
            "uLayerAngles": self.u.layer_angles,
            "uLayerShifted": self.u.layer_shifted,
            "uPhases": self.u.phases,
        }


@dataclass
class BlockSparseSiteUnitaryData:
    r"""Decomposition data for a single MPS site unitary with ``"block_sparse"`` synthesis.

    Each site unitary is decomposed as U = P_row · V_blockdiag · P_col.

    The permutations are stored as target mappings: perm_targets[i] gives the
    target index for basis state |i>, where basis state
    ``physical * ancilla_dim + bond`` holds physical state ``physical`` and
    bond state ``bond``. The block-diagonal unitary is stored as Givens
    rotation layers.
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

    def to_qsharp(self) -> dict:
        """Return the fields of the Q# ``SparseUnitaryDecomposition`` struct."""
        return {
            "colPermTargets": self.col_perm_targets,
            "rowPermTargets": self.row_perm_targets,
            "blockLayerAngles": self.layer_angles,
            "blockLayerShifted": self.layer_shifted,
            "blockPhases": self.phases,
        }


@dataclass
class MatrixProductStatePreparationData:
    """All data needed to drive the Q# MPS preparation operations.

    Produced by :func:`generate_matrix_product_state_preparation_data` and consumed by
    :meth:`MatrixProductStatePreparation._run_impl`. The sites hold
    :class:`GeneralSiteUnitaryData` for the ``MPSSequential`` Q# operations or
    :class:`BlockSparseSiteUnitaryData` for the ``MPSSparse`` Q# operation.
    """

    num_sites: int
    """Number of MPS sites."""

    num_qubits_per_site: int
    """Physical qubits per site: one for the ``('0', '1')`` basis and two for ``('0', 'u', 'd', '2')``."""

    ancilla_bits: int
    """Number of ancilla qubits."""

    initial_state_vec: list[float]
    """Flattened initial state vector for the first site."""

    site_to_orbital_order: list[int]
    """Orbital that holds each chain site, from the MPS container or the identity for bare sites."""

    sites: list[GeneralSiteUnitaryData] | list[BlockSparseSiteUnitaryData] = field(default_factory=list)
    """Per-site decomposition data (one entry per site 1..num_sites-1)."""

    def to_qsharp_params(self, rotation_bits: int) -> dict:
        """Flatten into the dict expected by ``MakeMPSSequentialCircuit`` or ``MakeMPSSparseCircuit``."""
        return {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "numQubitsPerSite": self.num_qubits_per_site,
            "siteToOrbitalOrder": self.site_to_orbital_order,
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteDecompositions": [site.to_qsharp() for site in self.sites],
        }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def generate_matrix_product_state_preparation_data(
    tensors: MPSContainer | Sequence[np.ndarray | MPSSite],
    unitary_synthesis: str = "general",
) -> MatrixProductStatePreparationData:
    """Compute all data needed for the Q# MPS preparation operations.

    Decomposes every site after the first. Returns structured data with
    raw angles (Double) and phases (Bool); Q# handles angle quantization internally.

    Parameters
    ----------
    tensors : MPSContainer or sequence of np.ndarray or MPSSite
        An MPS container or sites in chain order. Array inputs must be real with shape ``(chi_left, d, chi_right)``
        and are wrapped as trivial-symmetry :class:`~qdk_chemistry.data.MPSSite` objects.
        Every site must use the ``('0', '1')`` physical basis or every site the
        ``('0', 'u', 'd', '2')`` physical basis. A container supplies its validated site-to-orbital
        order; bare sites map chain site ``k`` to orbital ``k``.
    unitary_synthesis : str
        ``"general"`` for the cosine-sine and Givens-layer decomposition of each site, or
        ``"block_sparse"`` for the permutation and block-diagonal decomposition, which reads the
        stored symmetry blocks directly so absent blocks are structural zeros.

    Returns
    -------
    MatrixProductStatePreparationData
        Structured preparation data. Call ``.to_qsharp_params(rotation_bits)``
        to flatten into the dict expected by the Q# operation.

    Raises
    ------
    ValueError
        If ``unitary_synthesis`` is unknown, or the sites cannot be prepared.

    """
    if unitary_synthesis not in _UNITARY_SYNTHESIS_METHODS:
        raise ValueError(f"unitary_synthesis must be one of {_UNITARY_SYNTHESIS_METHODS}, got {unitary_synthesis!r}.")
    if isinstance(tensors, MPSContainer):
        container = tensors
        mps_sites = container.sites
    else:
        container = None
        mps_sites = _as_mps_sites(tensors)
    if mps_sites[0].is_complex:
        raise ValueError(f"{_DESCRIPTION} currently supports only real-valued MPS tensors.")
    _validate_physical_basis(mps_sites)
    # With a left bond of dimension one the packed (left * d, right) matrix is (d, chi_1).
    first_site = mps_sites[0].to_dense()
    max_bond = (
        container.max_bond_dimension if container is not None else max(site.right_bond_dimension for site in mps_sites)
    )
    ancilla_bits = _ancilla_bits(max_bond)
    ancilla_dim = 1 << ancilla_bits

    if container is not None:
        syntheses = decompose_mps(container, ancilla_dim, unitary_synthesis)
    elif unitary_synthesis == "general":
        syntheses = []
        following_factor = np.empty((0, 0))
        for site in reversed(mps_sites[1:]):
            synthesis = dense_unitary_synthesis(site, ancilla_dim, following_factor)
            syntheses.append(synthesis)
            following_factor = synthesis[3]
        syntheses.reverse()
    else:
        syntheses = [block_sparse_unitary_synthesis(site, ancilla_dim) for site in mps_sites[1:]]

    sites: list[GeneralSiteUnitaryData] | list[BlockSparseSiteUnitaryData]
    if unitary_synthesis == "general":
        # Each site absorbs the right factor V of its successor, so V never appears in the
        # circuit; the first synthesized site's V is absorbed into the initial state.
        # Sites with two physical states have no mixing unitaries, so w0 and w1 stay None.
        sites = [
            GeneralSiteUnitaryData(rotation_angles, GivensLayerData(*block), *(GivensLayerData(*g) for g in mixing))
            for rotation_angles, mixing, block, _ in syntheses
        ]
        if syntheses:
            first_site = first_site @ syntheses[0][3].T
    else:
        sites = [
            BlockSparseSiteUnitaryData(
                col_perm_targets=col_perm,
                row_perm_targets=row_perm,
                layer_angles=angles,
                layer_shifted=shifted,
                phases=phases,
            )
            for col_perm, row_perm, (angles, shifted, phases) in syntheses
        ]

    # The initial state is indexed by physical * ancilla_dim + bond.
    padded = np.zeros((first_site.shape[0], ancilla_dim))
    padded[:, : first_site.shape[1]] = first_site
    vector = padded.reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError("MPS initial state must contain finite amplitudes with nonzero norm.")

    return MatrixProductStatePreparationData(
        initial_state_vec=(vector / norm).tolist(),
        site_to_orbital_order=(
            container.site_to_orbital_order if container is not None else list(range(len(mps_sites)))
        ),
        num_sites=len(mps_sites),
        num_qubits_per_site=_qubits_per_site(mps_sites),
        ancilla_bits=ancilla_bits,
        sites=sites,
    )


# ---------------------------------------------------------------------------
# Validation and encoding helpers
# ---------------------------------------------------------------------------


def _validate_physical_basis(sites: Sequence[MPSSite]) -> None:
    """Require a physical dimension and basis order supported by the Q# operations.

    Args:
        sites: MPS sites to check.

    Raises:
        ValueError: If the sites do not share one physical dimension, the dimension is not two or
            four, or a site does not order its basis as ``('0', '1')`` or ``('0', 'u', 'd', '2')``.

    """
    dimensions = {site.physical_dimension for site in sites}
    if len(dimensions) > 1:
        raise ValueError("MPS state preparation requires the same physical dimension on every site.")
    (dimension,) = dimensions
    if dimension not in _PHYSICAL_BASES:
        raise ValueError(_DIMENSION_ERROR)
    if any(site.physical_basis != _PHYSICAL_BASES[dimension] for site in sites):
        labels = ", ".join(f"'{label}'" for label in _PHYSICAL_BASIS_LABELS[dimension])
        raise ValueError(f"MPS state preparation requires physical basis ordering ({labels}).")


def _as_mps_sites(tensors: Sequence[np.ndarray | MPSSite]) -> list[MPSSite]:
    """Wrap dense inputs as MPS sites and check the chain structure without densifying it.

    Args:
        tensors: MPS sites or dense ``(chi_left, d, chi_right)`` arrays in chain order.

    Returns:
        The MPS sites.

    Raises:
        ValueError: If a dense input is not rank three, does not have two or four physical
            states, or is not finite, or if the chain is empty or has incompatible scalar types
            or bond spaces.

    """
    sites = []
    for tensor in tensors:
        if isinstance(tensor, MPSSite):
            sites.append(tensor)
            continue
        # Dense arrays become single-block trivial-symmetry sites with the default basis.
        values = np.asarray(tensor)
        if values.ndim != 3:
            raise ValueError("Dense MPS site tensors must have shape (chi_left, d, chi_right).")
        left, physical, right = values.shape
        if physical not in _PHYSICAL_BASES:
            raise ValueError(_DIMENSION_ERROR)
        product, label = SymmetryProduct([]), SymmetryLabel([])
        tensor_type = SymmetryBlockedTensorRank3Complex if np.iscomplexobj(values) else SymmetryBlockedTensorRank3
        blocked = tensor_type(
            [product] * 3,
            [{label: left}, {label: physical}, {label: right}],
            [((label,) * 3, values.reshape(left * physical, right))],
        )
        sites.append(MPSSite(blocked, [label], [label], [label]))
    MPSContainer.validate_sites(sites)
    return sites


def _qubits_per_site(sites: Sequence[MPSSite]) -> int:
    """Return one for the ``('0', '1')`` basis and two for the ``('0', 'u', 'd', '2')`` basis."""
    return (sites[0].physical_dimension - 1).bit_length()


def _ancilla_bits(max_bond: int) -> int:
    """Return ``ceil(log2(max bond dimension))``, and at least one, the width of the bond register."""
    return max(1, (max_bond - 1).bit_length())
