"""Matrix Product State (MPS) state preparation via sequential site unitary synthesis.

Implements the MPS state preparation algorithm based on
:cite:`Berry2025`. Each site unitary is decomposed based on Appendix B in
:cite:`Rupprecht2026`, with orthogonal factors synthesized as parallel Givens
rotation layers using the elimination schedule of :cite:`Clements2016`.

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

from qdk_chemistry.data import Settings, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS
from qdk_chemistry.utils.unitary_synthesis import decompose_dense_sites

from ._mps_utils import (
    ancilla_bits_for,
    as_mps_sites,
    first_site_amplitudes,
    initial_state_vector,
    qubits_per_site,
    validate_mps_wavefunction,
    validate_site_to_orbital_order,
)
from .state_preparation import StatePreparation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qdk_chemistry.data import MPSSite

__all__: list[str] = [
    "MPSSequentialStatePreparation",
]

_DESCRIPTION = "MPS sequential state preparation"


class MPSSequentialStatePreparationSettings(Settings):
    """Settings for MPS sequential state preparation."""

    def __init__(self):
        """Initialize the MPSSequentialStatePreparationSettings."""
        super().__init__()
        self._set_default("rotation_bits", "int", 10, "Phase gradient precision.", (2, 62))
        self._set_default(
            "fast_resource_estimation",
            "bool",
            False,
            "Replace the site decompositions by placeholder data with the same gate structure. "
            "The circuit is valid for resource estimation only and does not prepare the state. "
            "The qubit count is exact; with a ('0', '1') physical basis the Toffoli count is an "
            "upper bound that is tight for sites whose bonds fill the ancilla register.",
        )


class MPSSequentialStatePreparation(StatePreparation):
    r"""Matrix Product State (MPS) state preparation using sequential unitary synthesis.

    Prepare the state sequentially, one site at a time, using an ancilla
    register that stores the virtual bond. Each site unitary is decomposed
    based on Appendix B in :cite:`Rupprecht2026` and synthesized from Givens
    rotation layers with QROM-loaded angles and phase gradient rotations.
    Every site unitary acts on the full ancilla register, so with
    ``fast_resource_estimation`` the cost can be estimated from the bond
    dimensions alone. The placeholder gives every block unitary a full set of
    Givens layers. For the ``('0', 'u', 'd', '2')`` basis the mixing unitaries
    make that the typical case. For the ``('0', '1')`` basis, sites whose bonds
    are smaller than the ancilla register, typically near the chain ends, need
    fewer layers, so the Toffoli estimate is an upper bound.

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

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Return a circuit to prepare an MPS state.

        Args:
            wavefunction: The wavefunction to prepare.

        Returns:
            A Circuit object implementing the MPS state preparation. With
            ``fast_resource_estimation`` the circuit only supports resource estimation.

        Raises:
            TypeError: If wavefunction is not an MPSContainer instance.
            ValueError: If the MPS is complex, not right-canonical with orthogonality center zero,
                does not use one of the ``('0', '1')`` and ``('0', 'u', 'd', '2')`` physical bases
                on every site, or does not have exactly one site per molecular orbital.

        """
        container = validate_mps_wavefunction(wavefunction, "MPSSequentialStatePreparation", _DESCRIPTION)
        rotation_bits = self._settings.get("rotation_bits")
        site_to_orbital_order = container.site_to_orbital_order

        if self._settings.get("fast_resource_estimation"):
            # Only the chain length and the bond dimensions are read, so no site is densified or
            # decomposed. The initial state is a fixed pseudo-random vector of the right size.
            sites = as_mps_sites(container.sites, _DESCRIPTION)
            ancilla_bits = ancilla_bits_for(sites)
            vector = np.random.default_rng(0).standard_normal(sites[0].physical_dimension << ancilla_bits)
            qsharp_factory = QsharpFactoryData(
                program=QSHARP_UTILS.MPSSequential.MakeMPSSequentialPlaceholderCircuit,
                parameter={
                    "initialStateVec": (vector / np.linalg.norm(vector)).tolist(),
                    "numSites": len(sites),
                    "numQubitsPerSite": qubits_per_site(sites),
                    "siteToOrbitalOrder": validate_site_to_orbital_order(site_to_orbital_order, len(sites)),
                    "rotationBits": rotation_bits,
                    "numAncillaQubits": ancilla_bits,
                },
            )
            return Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")

        data = generate_mps_preparation_data(container.sites)
        params = data.to_qsharp_params(rotation_bits, site_to_orbital_order)
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
class SiteUnitaryData:
    r"""Decomposition data for a single MPS site unitary.

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
class MPSPreparationData:
    """All data needed to drive the MPSSequential Q# operation.

    Produced by :func:`generate_mps_preparation_data` and consumed by
    :meth:`MPSSequentialStatePreparation._run_impl`.
    """

    num_sites: int
    """Number of MPS sites."""

    num_qubits_per_site: int
    """Physical qubits per site: one for the ``('0', '1')`` basis and two for ``('0', 'u', 'd', '2')``."""

    ancilla_bits: int
    """Number of ancilla qubits."""

    initial_state_vec: list[float]
    """Flattened initial state vector for the first site."""

    sites: list[SiteUnitaryData] = field(default_factory=list)
    """Per-site decomposition data (one entry per site 1..num_sites-1)."""

    def to_qsharp_params(
        self,
        rotation_bits: int,
        site_to_orbital_order: Sequence[int] | None = None,
    ) -> dict:
        """Flatten into the dict expected by the MakeMPSSequentialCircuit Q# operation."""
        return {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "numQubitsPerSite": self.num_qubits_per_site,
            "siteToOrbitalOrder": validate_site_to_orbital_order(site_to_orbital_order, self.num_sites),
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteDecompositions": [site.to_qsharp() for site in self.sites],
        }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def generate_mps_preparation_data(tensors: Sequence[np.ndarray | MPSSite]) -> MPSPreparationData:
    """Compute all data needed for the MPSSequential Q# operation.

    Performs CSD + Givens layer decomposition for each site.
    Returns structured data with raw angles (Double) and phases (Bool) --
    Q# handles angle quantization internally.

    Parameters
    ----------
    tensors : sequence of np.ndarray or MPSSite
        MPS sites in chain order. Array inputs must be real with shape ``(chi_left, d, chi_right)``
        and are wrapped as trivial-symmetry :class:`~qdk_chemistry.data.MPSSite` objects.
        Every site must use the ``('0', '1')`` physical basis or every site the
        ``('0', 'u', 'd', '2')`` physical basis.

    Returns
    -------
    MPSPreparationData
        Structured preparation data. Call ``.to_qsharp_params(rotation_bits)``
        to flatten into the dict expected by the Q# operation.

    """
    mps_sites = as_mps_sites(tensors, _DESCRIPTION)
    first_site = first_site_amplitudes(mps_sites)  # (d, chi_1)
    num_sites = len(mps_sites)
    ancilla_bits = ancilla_bits_for(mps_sites)
    ancilla_dim = 1 << ancilla_bits

    # Each site absorbs the right factor V of its successor, so V never appears in the
    # circuit; the first synthesized site's V is absorbed into the initial state.
    syntheses = decompose_dense_sites(mps_sites[1:], ancilla_dim)
    sites: list[SiteUnitaryData] = []
    for rotation_angles, mixing, block, _ in syntheses:
        mixing_data = [GivensLayerData(*givens) for givens in mixing]
        w0, w1 = mixing_data if mixing_data else (None, None)
        sites.append(SiteUnitaryData(rot_angles=rotation_angles, u=GivensLayerData(*block), w0=w0, w1=w1))

    if syntheses:
        first_site = first_site @ syntheses[0][3].T

    return MPSPreparationData(
        initial_state_vec=initial_state_vector(first_site, ancilla_dim),
        num_sites=num_sites,
        num_qubits_per_site=qubits_per_site(mps_sites),
        ancilla_bits=ancilla_bits,
        sites=sites,
    )
