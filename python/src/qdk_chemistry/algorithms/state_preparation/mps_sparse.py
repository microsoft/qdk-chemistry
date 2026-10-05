"""Matrix Product State (MPS) state preparation exploiting block sparsity.

Implements the sparse MPS preparation method from :cite:`Rupprecht2026`.
Each site unitary is decomposed as ``U = P_row · V_blockdiag · P_col``
where ``P_row``, ``P_col`` are permutations (a table lookup of the permuted index,
a SWAP, and erasure of the old index by X-basis measurement with a phase fixup) and
``V_blockdiag`` is block-diagonal (synthesized via Givens rotation layers per block,
using the elimination schedule of :cite:`Clements2016`).
This exploits U(1) symmetries (particle number,
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

from qdk_chemistry.data import Settings, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS
from qdk_chemistry.utils.unitary_synthesis import decompose_sparse_sites

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

    import numpy as np

    from qdk_chemistry.data import MPSSite

__all__: list[str] = [
    "MPSSparseStatePreparation",
]

_DESCRIPTION = "Sparse MPS state preparation"


class MPSSparseStatePreparationSettings(Settings):
    """Settings for MPS sparse state preparation."""

    def __init__(self):
        """Initialize the MPSSparseStatePreparationSettings."""
        super().__init__()
        self._set_default("rotation_bits", "int", 10, "Phase gradient precision.", (2, 62))


class MPSSparseStatePreparation(StatePreparation):
    r"""MPS state preparation exploiting block sparsity.

    Prepare the state using permutation-based decomposition. Each site unitary
    is factored as ``U = P_row · V_blockdiag · P_col``, where permutations are
    implemented by a table lookup and a SWAP whose leftover register is erased
    by measurement, and the block-diagonal unitary is synthesized via Givens
    rotation layers. This exploits the block-sparse structure of MPS
    tensors arising from U(1) symmetries (particle number, spin conservation).

    The circuit uses the same qubit layouts and fermionic sign convention as
    :class:`~qdk_chemistry.algorithms.state_preparation.mps_sequential.MPSSequentialStatePreparation`.

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
            ValueError: If the MPS is complex, not right-canonical with orthogonality center zero,
                does not use one of the ``('0', '1')`` and ``('0', 'u', 'd', '2')`` physical bases
                on every site, or does not have exactly one site per molecular orbital.

        """
        container = validate_mps_wavefunction(wavefunction, "MPSSparseStatePreparation", _DESCRIPTION)
        data = generate_mps_sparse_preparation_data(container.sites)
        rotation_bits = self._settings.get("rotation_bits")
        params = data.to_qsharp_params(rotation_bits, container.site_to_orbital_order)
        # MakeMPSSparseCircuit is Adaptive-only; the default shared context targets Adaptive_RIF.
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


@dataclass
class MPSSparsePreparationData:
    """All data needed to drive the MPSSparse Q# operation."""

    initial_state_vec: list[float]
    """Flattened initial state vector for the first site."""

    num_sites: int
    """Number of MPS sites."""

    num_qubits_per_site: int
    """Physical qubits per site: one for the ``('0', '1')`` basis and two for ``('0', 'u', 'd', '2')``."""

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
        return {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "numQubitsPerSite": self.num_qubits_per_site,
            "siteToOrbitalOrder": validate_site_to_orbital_order(site_to_orbital_order, self.num_sites),
            "rotationBits": rotation_bits,
            "numAncillaQubits": self.ancilla_bits,
            "siteDecompositions": [
                {
                    "colPermTargets": site.col_perm_targets,
                    "rowPermTargets": site.row_perm_targets,
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
    tensors: Sequence[np.ndarray | MPSSite],
) -> MPSSparsePreparationData:
    """Compute all data needed for the MPSSparse Q# operation.

    Performs the permutation + block-diagonal decomposition of all sites in one concurrent call.

    Parameters
    ----------
    tensors : sequence of np.ndarray or MPSSite
        MPS sites in chain order. Array inputs must be real with shape ``(chi_left, d, chi_right)``
        and are wrapped as trivial-symmetry :class:`~qdk_chemistry.data.MPSSite` objects.
        Every site must use the ``('0', '1')`` physical basis or every site the
        ``('0', 'u', 'd', '2')`` physical basis. The decomposition reads the stored symmetry blocks
        directly, so absent blocks are structural zeros.

    Returns
    -------
    MPSSparsePreparationData
        Structured preparation data.

    """
    mps_sites = as_mps_sites(tensors, _DESCRIPTION)
    first_site = first_site_amplitudes(mps_sites)
    num_sites = len(mps_sites)
    ancilla_bits = ancilla_bits_for(mps_sites)
    ancilla_dim = 1 << ancilla_bits

    sites = [
        SparseSiteUnitaryData(
            col_perm_targets=list(col_perm),
            row_perm_targets=list(row_perm),
            layer_angles=angles,
            layer_shifted=shifted,
            phases=phases,
        )
        for col_perm, row_perm, (angles, shifted, phases) in decompose_sparse_sites(mps_sites[1:], ancilla_dim)
    ]

    return MPSSparsePreparationData(
        initial_state_vec=initial_state_vector(first_site, ancilla_dim),
        num_sites=num_sites,
        num_qubits_per_site=qubits_per_site(mps_sites),
        ancilla_bits=ancilla_bits,
        sites=sites,
    )
