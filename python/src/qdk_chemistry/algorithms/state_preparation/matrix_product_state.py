"""Matrix Product State (MPS) state preparation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# Portions of this file are adapted from code by Felix Rupprecht published at
# https://zenodo.org/records/20393500, Copyright 2026 German Aerospace Center (DLR),
# licensed under the Apache License, Version 2.0, and modified for QDK Chemistry.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import sparse

from qdk_chemistry.algorithms.circuit_mapper.unitary_synthesis_mapper import UnitarySynthesisMapper
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    MPSContainer,
    MPSSite,
    RealUnitaryContainer,
    Settings,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.circuit import Circuit, CircuitMetadata, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .state_preparation import StatePreparation

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__: list[str] = [
    "MatrixProductStatePreparation",
    "MatrixProductStatePreparationData",
]

_DESCRIPTION = "MPS state preparation"

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
        self._set_default(
            "rotation_bit_precision",
            "int",
            10,
            "Size of the phase gradient register, which sets the precision of every rotation "
            "angle. The upper bound of 30 is a sanity limit as 2^-30 is far below chemical "
            "accuracy.",
            (1, 30),
        )
        self._set_default(
            "unitary_synthesis",
            "algorithm_ref",
            AlgorithmRef("circuit_mapper", "unitary_synthesis"),
            "unitary_synthesis circuit mapper that synthesizes every site isometry; its method is "
            "'dense' or 'block_sparse'. Its num_input_qubits is set to the bond register width and "
            "its omit_right_factor to whether the method is 'dense'. Its rotation_bit_precision and "
            "allocate_phase_gradient are unused, as all sites share this algorithm's phase gradient.",
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
    Site zero is loaded as an initial state, and every later site is applied as an isometry from its
    left bond, held in the bond register, to its physical state and right bond. The nested
    ``unitary_synthesis`` circuit mapper synthesizes each isometry, and its ``method`` setting
    selects how:

    ``"dense"``
        Each site isometry is decomposed based on Appendix B in :cite:`Rupprecht2026`, with orthogonal
        factors synthesized as parallel Givens rotation layers using the elimination schedule of
        :cite:`Clements2016`. The right factor of each decomposition, an orthogonal matrix on the left
        bond, is absorbed classically into the preceding site or the initial state. The cost depends
        only on the bond dimensions.

    ``"block_sparse"``
        Each site isometry is decomposed as ``U = P_row · V_blockdiag · P_col`` following
        :cite:`Rupprecht2026`, where ``P_row`` and ``P_col`` are permutations (a table lookup of the
        permuted index, a SWAP, and erasure of the old index by X-basis measurement with a phase
        fixup) and ``V_blockdiag`` is block diagonal, with each block synthesized from Givens rotation
        layers. This exploits U(1) symmetries (particle number, spin) that make MPS tensors block
        sparse. :cite:`Rupprecht2026` reports 10-30x Toffoli savings over the dense state of the art
        for large molecular MPS; the savings shrink with the bond dimension and the block sparsity,
        and small bonds can cost more than ``"dense"``.

    For example, ``create("state_prep", "matrix_product_state", unitary_synthesis=AlgorithmRef(
    "circuit_mapper", "unitary_synthesis", method="block_sparse"))`` selects the block-sparse
    synthesis.

    Sites with the ``('0', 'u', 'd', '2')`` physical basis are spatial
    orbitals in the blocked Jordan-Wigner layout. Sites
    with the ``('0', '1')`` physical basis are spinless modes in the
    Jordan-Wigner layout.

    The unitary synthesis is adapted from the Qualtran implementation by Felix Rupprecht
    published on Zenodo :cite:`Rupprecht2026Zenodo` under the Apache License 2.0.
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
        """Synthesize the site isometries with the nested ``unitary_synthesis`` circuit mapper.

        Args:
            mps: Real MPS container with the same supported physical basis on every site.
                Its validated site-to-orbital order is preserved.

        Returns:
            Preparation data with raw angles and phases for Q# angle quantization.
            Site zero is the initial state; every following site is synthesized.

        Raises:
            TypeError: If the input is not an MPSContainer, or the ``unitary_synthesis`` setting
                does not refer to a :class:`UnitarySynthesisMapper`.
            ValueError: If the physical basis is unsupported, the MPS is complex,
                the initial state has zero norm, or a synthesized site is not isometric.

        """
        if not isinstance(mps, MPSContainer):
            raise TypeError(f"MatrixProductStatePreparation requires an MPSContainer, got {type(mps)}.")
        if mps.is_complex:
            raise ValueError(f"{_DESCRIPTION} currently supports only real-valued MPS tensors.")
        mps_sites = mps.sites
        self._validate_physical_basis(mps_sites)
        ancilla_bits = self._ancilla_bits(mps.max_bond_dimension)
        ancilla_dim = 1 << ancilla_bits
        mapper, method = self._create_site_mapper(ancilla_bits)

        # Sites are synthesized from the last one so that each dense right factor V, which the
        # circuit of its site leaves out, can be absorbed into the right bond of the site before.
        syntheses: list[dict[str, Any]] = []
        right_factor: np.ndarray | None = None
        for index in range(len(mps_sites) - 1, 0, -1):
            isometry = self._site_isometry(mps_sites[index], ancilla_dim)
            if right_factor is not None:
                bond = right_factor.shape[0]
                isometry[:, :bond] = np.einsum("bc,pca->pba", right_factor, isometry[:, :bond])
            matrix = isometry.reshape(-1, isometry.shape[2])
            try:
                container = RealUnitaryContainer(sparse.csc_array(matrix) if method == "block_sparse" else matrix)
            except ValueError as error:
                raise ValueError(f"{_DESCRIPTION} requires isometric sites; site {index} is not: {error}") from error
            unitary = UnitaryRepresentation(container)
            syntheses.append(mapper.run(unitary)._qsharp_factory.parameter["synthesis"])  # noqa: SLF001
            right_factor = mapper.right_factor(unitary) if method == "dense" else None
        syntheses.reverse()

        # With a left bond of dimension one the packed matrix is (physical, right).
        first_site = np.asarray(mps_sites[0].to_dense())
        if right_factor is not None:
            first_site = first_site @ right_factor.T
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
            method=method,
            sites=syntheses,
        )

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Return a circuit to prepare an MPS state.

        Args:
            wavefunction: The wavefunction to prepare.

        Returns:
            A Circuit object implementing the MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an MPSContainer instance, or the
                ``unitary_synthesis`` setting does not refer to a :class:`UnitarySynthesisMapper`.
            ValueError: If the MPS is complex, not right-canonical with orthogonality
                center zero, does not use one of the ``('0', '1')`` and ``('0', 'u', 'd', '2')``
                physical bases on every site, or does not have exactly one site per molecular
                orbital.

        """
        rotation_bit_precision = self._settings.get("rotation_bit_precision")

        container = wavefunction.get_container()
        if not isinstance(container, MPSContainer):
            raise TypeError(f"MatrixProductStatePreparation requires an MPSContainer, got {type(container)}.")
        if container.orthogonality_center != 0:
            raise ValueError(f"{_DESCRIPTION} requires a right-canonical MPS with center zero.")
        if container.num_sites != container.orbitals.get_num_molecular_orbitals():
            raise ValueError(f"{_DESCRIPTION} requires exactly one MPS site per molecular orbital.")

        data = self.generate_matrix_product_state_preparation_data(container)
        params = data.to_qsharp_params(rotation_bit_precision)
        # The MPS circuits are Adaptive-only; the default shared context targets Adaptive_RIF.
        if data.method == "block_sparse":
            sparse_module = QSHARP_UTILS.MPSSparse
            program, params_type = sparse_module.MakeMPSSparseCircuit, sparse_module.MPSSparseParams
            make_op = sparse_module.MakeMPSSparseOp
            make_op_with_phase_gradient = sparse_module.MakeMPSSparseOpWithPhaseGradient
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
            num_gradient_ancillas = rotation_bit_precision
        # An exported circuit has no caller to own the gradient, so it always allocates its own.
        qsharp_factory = QsharpFactoryData(program=program, parameter=params)
        return Circuit(
            qsharp_factory=qsharp_factory,
            qsharp_op=qsharp_op,
            encoding="jordan-wigner",
            num_qubits=num_qubits + num_gradient_ancillas,
            metadata=CircuitMetadata(num_phase_gradient_ancillas=num_gradient_ancillas),
        )

    def _create_site_mapper(self, ancilla_bits: int) -> tuple[UnitarySynthesisMapper, str]:
        """Return the nested site mapper, configured for the bond register, and its method.

        Raises:
            TypeError: If the ``unitary_synthesis`` setting does not refer to a
                :class:`UnitarySynthesisMapper`.

        """
        mapper = self._create_nested("unitary_synthesis")
        if not isinstance(mapper, UnitarySynthesisMapper):
            raise TypeError(
                f"{_DESCRIPTION} requires the unitary_synthesis circuit mapper, got "
                f"{mapper.type_name()} '{mapper.name()}'."
            )
        settings = mapper.settings()
        method = settings.get("method")
        # The bond register is the input register; the site qubits are the added qubits.
        settings.update("num_input_qubits", ancilla_bits)
        settings.update("omit_right_factor", method == "dense")
        return mapper, method

    @staticmethod
    def _site_isometry(site: MPSSite, ancilla_dim: int) -> np.ndarray:
        """Return ``isometry[p, b, a]`` = A^p_{ab}, zero-padded to ``ancilla_dim`` right bond states.

        Reshaped to ``(physical * ancilla_dim, left)``, row ``p * ancilla_dim + b`` is site
        qubit state ``p`` and bond register state ``b``, and column ``a`` is the left bond state.
        """
        left, physical, right = site.shape
        tensor = np.asarray(site.to_dense()).reshape(left, physical, right)
        isometry = np.zeros((physical, ancilla_dim, left))
        isometry[:, :right] = tensor.transpose(1, 2, 0)
        return isometry

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
    :meth:`~qdk_chemistry.algorithms.StatePreparation.run`. The sites are the fields of the Q#
    ``DenseIsometrySynthesis`` structs for the ``MPSSequential`` Q# operations or of the Q#
    ``SparseIsometrySynthesis`` structs for the ``MPSSparse`` Q# operations, as produced by the
    ``unitary_synthesis`` circuit mapper. Dense sites carry no right factor, which is absorbed into
    the preceding site.
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

    method: str = "dense"
    """Synthesis method of the sites: ``"dense"`` or ``"block_sparse"``."""

    sites: list[dict[str, Any]] = field(default_factory=list)
    """Per-site Q# synthesis data (one entry per site 1..num_sites-1)."""

    def to_qsharp_params(self, rotation_bit_precision: int) -> dict:
        """Flatten into the dict expected by ``MakeMPSSequentialCircuit`` or ``MakeMPSSparseCircuit``."""
        return {
            "initialStateVec": self.initial_state_vec,
            "numSites": self.num_sites,
            "numQubitsPerSite": self.num_qubits_per_site,
            "siteToOrbitalOrder": self.site_to_orbital_order,
            "rotationBitPrecision": rotation_bit_precision,
            "numAncillaQubits": self.ancilla_bits,
            "siteDecompositions": self.sites,
        }
