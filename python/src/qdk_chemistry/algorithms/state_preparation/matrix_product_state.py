"""Matrix Product State (MPS) state preparation.

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
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS
from qdk_chemistry.utils.unitary_synthesis import DenseSiteSynthesis, matrix_product_state_synthesis

from .state_preparation import StatePreparation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qdk_chemistry.utils.unitary_synthesis import GivensDecomposition, SparseSiteSynthesis

__all__: list[str] = [
    "MatrixProductStatePreparation",
    "MatrixProductStatePreparationData",
]

_DESCRIPTION = "MPS state preparation"
_UNITARY_SYNTHESIS_METHODS = ["dense", "block_sparse"]

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
            "dense",
            "Site unitary synthesis: 'dense' factors every site unitary over the full bond, and "
            "'block_sparse' factors it into permutations and a block-diagonal unitary.",
            _UNITARY_SYNTHESIS_METHODS,
        )
        self._set_default(
            "allocate_phase_gradient",
            "bool",
            True,
            "Whether to allocate and initialize the phase gradient register internally.",
        )


class MatrixProductStatePreparation(StatePreparation):
    r"""Matrix Product State (MPS) state preparation.

    Implements the MPS state preparation algorithm based on :cite:`Berry2025` and :cite:`Rupprecht2026`,
    which prepares the state one site at a time using an ancilla register that stores the virtual bond.
    The ``unitary_synthesis`` setting selects how each site unitary is synthesized:

    ``"dense"``
        Each site unitary is decomposed based on Appendix B in :cite:`Rupprecht2026`, with orthogonal
        factors synthesized as parallel Givens rotation layers using the elimination schedule of
        :cite:`Clements2016`. The cost depends only on the bond dimensions.

    ``"block_sparse"``
        Each site unitary is decomposed as ``U = P_row · V_blockdiag · P_col`` following
        :cite:`Rupprecht2026`, where ``P_row`` and ``P_col`` are permutations (a table lookup of the
        permuted index, a SWAP, and erasure of the old index by X-basis measurement with a phase
        fixup) and ``V_blockdiag`` is block diagonal, with each block synthesized from Givens rotation
        layers. This exploits U(1) symmetries (particle number, spin) that make MPS tensors block
        sparse, yielding 10-30x Toffoli savings over ``"dense"``.

    Sites with the ``('0', 'u', 'd', '2')`` physical basis are spatial
    orbitals in the blocked Jordan-Wigner layout. Sites
    with the ``('0', '1')`` physical basis are spinless modes in the
    Jordan-Wigner layout.

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

    def generate_matrix_product_state_preparation_data(self, mps: MPSContainer) -> MatrixProductStatePreparationData:
        """Compute the site decompositions using this algorithm's synthesis setting.

        Args:
            mps: Real MPS container with the same supported physical basis on every site.
                Its validated site-to-orbital order is preserved.

        Returns:
            Preparation data with raw angles and phases for Q# angle quantization.
            Site zero is the initial state; every following site is synthesized.

        Raises:
            TypeError: If the input is not an MPSContainer.
            ValueError: If the physical basis is unsupported, the MPS is complex,
                the initial state has zero norm, or a synthesized site is not isometric.

        """
        if not isinstance(mps, MPSContainer):
            raise TypeError(f"MatrixProductStatePreparation requires an MPSContainer, got {type(mps)}.")
        if mps.is_complex:
            raise ValueError(f"{_DESCRIPTION} currently supports only real-valued MPS tensors.")
        mps_sites = mps.sites
        self._validate_physical_basis(mps_sites)
        unitary_synthesis = self._settings.get("unitary_synthesis")
        # With a left bond of dimension one the packed matrix is (physical, right).
        first_site = mps_sites[0].to_dense()
        ancilla_bits = self._ancilla_bits(mps.max_bond_dimension)
        ancilla_dim = 1 << ancilla_bits
        syntheses = matrix_product_state_synthesis(mps, ancilla_dim, unitary_synthesis)
        if unitary_synthesis == "dense" and syntheses:
            # The successor's right factor is absorbed into each site, leaving only
            # site one's incoming factor to absorb into the initial state.
            first_site = first_site @ syntheses[0].right_factor.T

        # The initial state is indexed by physical * ancilla_dim + bond.
        padded = np.zeros((first_site.shape[0], ancilla_dim))
        padded[:, : first_site.shape[1]] = first_site
        vector = padded.reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 1e-15:
            raise ValueError("MPS initial state must contain finite amplitudes with nonzero norm.")

        return MatrixProductStatePreparationData(
            initial_state_vec=(vector / norm).tolist(),
            site_to_orbital_order=mps.site_to_orbital_order,
            num_sites=mps.num_sites,
            num_qubits_per_site=self._qubits_per_site(mps_sites),
            ancilla_bits=ancilla_bits,
            sites=syntheses,
        )

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Return a circuit to prepare an MPS state.

        Args:
            wavefunction: The wavefunction to prepare.

        Returns:
            A Circuit object implementing the MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an MPSContainer instance.
            ValueError: If the MPS is complex, not right-canonical with orthogonality
                center zero, does not use one of the ``('0', '1')`` and ``('0', 'u', 'd', '2')``
                physical bases on every site, or does not have exactly one site per molecular
                orbital.

        """
        rotation_bits = self._settings.get("rotation_bits")
        unitary_synthesis = self._settings.get("unitary_synthesis")

        container = wavefunction.get_container()
        if not isinstance(container, MPSContainer):
            raise TypeError(f"MatrixProductStatePreparation requires an MPSContainer, got {type(container)}.")
        if container.orthogonality_center != 0:
            raise ValueError(f"{_DESCRIPTION} requires a right-canonical MPS with center zero.")
        if container.num_sites != container.orbitals.get_num_molecular_orbitals():
            raise ValueError(f"{_DESCRIPTION} requires exactly one MPS site per molecular orbital.")

        data = self.generate_matrix_product_state_preparation_data(container)
        params = data.to_qsharp_params(rotation_bits)
        # The MPS circuits are Adaptive-only; the default shared context targets Adaptive_RIF.
        if unitary_synthesis == "block_sparse":
            sparse = QSHARP_UTILS.MPSSparse
            program, params_type = sparse.MakeMPSSparseCircuit, sparse.MPSSparseParams
            make_op, make_op_with_phase_gradient = sparse.MakeMPSSparseOp, sparse.MakeMPSSparseOpWithPhaseGradient
        else:
            sequential = QSHARP_UTILS.MPSSequential
            program, params_type = sequential.MakeMPSSequentialCircuit, sequential.MPSSequentialParams
            make_op = sequential.MakeMPSSequentialOp
            make_op_with_phase_gradient = sequential.MakeMPSSequentialOpWithPhaseGradient
        num_qubits = data.num_qubits_per_site * data.num_sites
        if self._settings.get("allocate_phase_gradient"):
            qsharp_op = make_op(params_type(**params))
            num_gradient_ancillas = 0
        else:
            # The caller owns the trailing gradient register, must leave the gradient in it, and
            # must exclude it from any reflection about |0>.
            qsharp_op = make_op_with_phase_gradient(params_type(**params))
            num_gradient_ancillas = rotation_bits
        # An exported circuit has no caller to own the gradient, so it always allocates its own.
        qsharp_factory = QsharpFactoryData(program=program, parameter=params)
        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            encoding="jordan-wigner",
            num_qubits=num_qubits + num_gradient_ancillas,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=num_gradient_ancillas),
        )

    @staticmethod
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

    @staticmethod
    def _qubits_per_site(sites: Sequence[MPSSite]) -> int:
        """Return one for the ``('0', '1')`` basis and two for the ``('0', 'u', 'd', '2')`` basis."""
        return (sites[0].physical_dimension - 1).bit_length()

    @staticmethod
    def _ancilla_bits(max_bond: int) -> int:
        """Return ``ceil(log2(max bond dimension))``, and at least one, the width of the bond register."""
        return max(1, (max_bond - 1).bit_length())


# ---------------------------------------------------------------------------
# Preparation data passed to Q#
# ---------------------------------------------------------------------------


@dataclass
class MatrixProductStatePreparationData:
    """All data needed to drive the Q# MPS preparation operations.

    Produced by :meth:`MatrixProductStatePreparation.generate_matrix_product_state_preparation_data` and consumed by
    :meth:`MatrixProductStatePreparation._run_impl`. The sites are the C++ synthesis results:
    :class:`~qdk_chemistry.utils.unitary_synthesis.DenseSiteSynthesis` for the ``MPSSequential``
    Q# operations or :class:`~qdk_chemistry.utils.unitary_synthesis.SparseSiteSynthesis` for the
    ``MPSSparse`` Q# operation. Q# structs of the same names mirror them field by field.
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
    """Orbital that holds each chain site, from the MPS container."""

    sites: list[DenseSiteSynthesis] | list[SparseSiteSynthesis] = field(default_factory=list)
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
            "siteDecompositions": [_site_to_qsharp(site) for site in self.sites],
        }


def _givens_to_qsharp(givens: GivensDecomposition) -> dict:
    """Return the fields of the Q# ``GivensDecomposition`` struct."""
    return {"layerAngles": givens.layer_angles, "layerShifted": givens.layer_shifted, "phases": givens.phases}


def _site_to_qsharp(site: DenseSiteSynthesis | SparseSiteSynthesis) -> dict:
    """Return the fields of the Q# ``DenseSiteSynthesis`` or ``SparseSiteSynthesis`` struct.

    The dense right factor is absorbed classically into the preceding site, so it has no Q# field.
    """
    if isinstance(site, DenseSiteSynthesis):
        return {
            "rotationAngles": site.rotation_angles,
            "mixingGivens": [_givens_to_qsharp(givens) for givens in site.mixing_givens],
            "blockGivens": _givens_to_qsharp(site.block_givens),
        }
    return {
        "columnPermutation": site.column_permutation,
        "rowPermutation": site.row_permutation,
        "blockGivens": _givens_to_qsharp(site.block_givens),
    }
