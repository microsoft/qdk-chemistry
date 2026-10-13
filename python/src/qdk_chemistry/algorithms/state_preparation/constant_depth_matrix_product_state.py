"""Constant-depth matrix product state (MPS) preparation with adaptive circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.algorithms.circuit_mapper.unitary_synthesis_mapper import UnitarySynthesisMapper
from qdk_chemistry.data import (
    AlgorithmRef,
    MPSContainer,
    RealUnitaryContainer,
    Settings,
    UnitaryRepresentation,
    Wavefunction,
)
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .matrix_product_state import MatrixProductStatePreparation
from .state_preparation import StatePreparation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qdk_chemistry.data import MPSSite

__all__: list[str] = [
    "ConstantDepthMPSFragment",
    "ConstantDepthMatrixProductStatePreparation",
    "ConstantDepthMatrixProductStatePreparationData",
]

_DESCRIPTION = "Constant-depth MPS state preparation"
# Relative tolerance of the pushing condition and of the rank of a fragment tensor.
_PUSHING_TOLERANCE = 1e-8


class ConstantDepthMatrixProductStatePreparationSettings(Settings):
    """Settings for constant-depth matrix product state preparation."""

    def __init__(self):
        """Initialize the ConstantDepthMatrixProductStatePreparationSettings."""
        super().__init__()
        self._set_default(
            "rotation_bit_precision",
            "int",
            10,
            "Size of each phase gradient register, which sets the precision of every rotation "
            "angle. The upper bound of 30 is a sanity limit as 2^-30 is far below chemical "
            "accuracy.",
            (1, 30),
        )
        self._set_default(
            "fragment_size",
            "int",
            2,
            "Number of consecutive sites in each fragment, at least one. The last fragment holds "
            "the remaining sites. The circuit depth grows with the fragment size but not with the "
            "number of sites.",
        )
        self._set_default(
            "unitary_synthesis",
            "algorithm_ref",
            AlgorithmRef("circuit_mapper", "unitary_synthesis"),
            "unitary_synthesis circuit mapper that synthesizes every site isometry and correction; its "
            "method must be 'dense'. Its num_input_qubits is set to the width of the register each "
            "matrix acts on and its omit_right_factor to True. Its rotation_bit_precision and "
            "allocate_phase_gradient are unused, as every fragment allocates a phase gradient register "
            "of this algorithm's rotation_bit_precision.",
        )


class ConstantDepthMatrixProductStatePreparation(StatePreparation):
    r"""Constant-depth matrix product state preparation with measurements and feed-forward.

    Implements Protocol 1 of :cite:`Smith2024`. The chain is split into fragments of
    ``fragment_size`` consecutive sites. Every fragment is prepared in parallel by the ``"dense"`` site
    isometries of :class:`MatrixProductStatePreparation`, starting from a Bell pair on its left bond.
    The nested ``unitary_synthesis`` circuit mapper synthesizes the site isometries and the
    corrections, and its ``method`` must be ``"dense"``.
    Bell measurements then fuse each pair of adjacent bonds, which leaves a random Pauli defect
    :math:`B` on every fused bond. Each defect is pushed through the fragments to its right,
    :math:`B\,T^m = \sum_n W_{mn} T^n H`, where :math:`T` is the fragment tensor and :math:`H` the
    defect on its right bond, and is absorbed by the last fragment, whose right bond is trivial.
    The physical corrections :math:`W^\mathsf{T}` are applied in one layer chosen from the
    measurement record, so the quantum circuit depth is independent of the number of sites when
    classical feed-forward is free.

    Defects are pushed toward the end of the chain, the mirror image of the convention in
    :cite:`Smith2024`. A tensor :math:`A` from the paper corresponds to the right-canonical tensor
    with the left and right bonds of :math:`A` swapped.

    A unitary :math:`W` exists for a defect if and only if
    :math:`[B^\mathsf{T} \otimes H^{-1}, \bar{T}^\mathsf{T} \bar{T}] = 0` for some Pauli :math:`H`,
    where :math:`\bar{T}` has rows :math:`m` and columns :math:`(a, b)`. Every Pauli on every fused
    bond must be pushable, which is Theorem 2 of :cite:`Smith2024`; otherwise the MPS is rejected.
    Examples are the GHZ, cluster and AKLT states and the :math:`\mathbb{Z}_2` family of
    :cite:`Smith2024`, but not generic MPS. Fused bonds must have power-of-two dimension. Protocol 2
    of :cite:`Smith2024` for non-normal MPS is not implemented.

    The operation measures qubits, so it is not adjointable. It allocates one phase gradient
    register per fragment so that the fragments run in parallel. The fermionic sign layer for
    ``('0', 'u', 'd', '2')`` sites or a non-identity site-to-orbital order is the same CZ network as
    in :class:`MatrixProductStatePreparation`, whose depth is not constant.
    """

    def __init__(self):
        """Initialize the constant-depth matrix product state preparation algorithm."""
        super().__init__()
        self._settings = ConstantDepthMatrixProductStatePreparationSettings()

    def name(self) -> str:
        """Return the algorithm name.

        Returns:
            str: The name ``"constant_depth_matrix_product_state"``

        """
        return "constant_depth_matrix_product_state"

    def generate_constant_depth_preparation_data(
        self, mps: MPSContainer
    ) -> ConstantDepthMatrixProductStatePreparationData:
        """Compute the fragment preparations and their pushing relations.

        Args:
            mps: Real right-canonical MPS container with the same supported physical basis on every site.
                Its validated site-to-orbital order is preserved.

        Returns:
            Preparation data with raw angles and phases for Q# angle quantization.

        Raises:
            TypeError: If the input is not an MPSContainer, or the ``unitary_synthesis`` setting does
                not refer to a :class:`UnitarySynthesisMapper`.
            ValueError: If the ``unitary_synthesis`` method is not ``"dense"``, the physical basis is
                unsupported, the MPS is complex, the fragment size is below one, a fused bond is not
                a power of two, a defect cannot be pushed through a fragment, the initial state has
                zero norm, or a synthesized site is not isometric.

        """
        if not isinstance(mps, MPSContainer):
            raise TypeError(f"{_DESCRIPTION} requires an MPSContainer, got {type(mps)}.")
        if mps.is_complex:
            raise ValueError(f"{_DESCRIPTION} currently supports only real-valued MPS tensors.")
        fragment_size = self._settings.get("fragment_size")
        if fragment_size < 1:
            raise ValueError(f"{_DESCRIPTION} requires a fragment size of at least one, got {fragment_size}.")
        mps_sites = mps.sites
        MatrixProductStatePreparation._validate_physical_basis(mps_sites)  # noqa: SLF001
        bond_bits = MatrixProductStatePreparation._ancilla_bits(mps.max_bond_dimension)  # noqa: SLF001
        bond_dim = 1 << bond_bits
        # The bond register is the input register of every site isometry. A correction is square, so
        # the default width of -1 takes all its qubits as input and adds none.
        site_mapper = self._create_mapper(bond_bits)
        correction_mapper = self._create_mapper(-1)
        tensors = [site.to_dense().reshape(site.shape) for site in mps_sites]
        bounds = [*range(0, mps.num_sites, fragment_size), mps.num_sites]
        for start in bounds[1:-1]:
            fused_bond = tensors[start].shape[0]
            if fused_bond & (fused_bond - 1):
                raise ValueError(
                    f"{_DESCRIPTION} requires power-of-two bond dimensions between fragments, "
                    f"but the bond before site {start} has dimension {fused_bond}."
                )

        fragments = [self._first_fragment(tensors[0], mps_sites[1 : bounds[1]], bond_dim, site_mapper)]
        for start, stop in itertools.pairwise(bounds[1:]):
            fragments.append(
                self._fused_fragment(
                    tensors[start:stop], mps_sites[start:stop], start, bond_dim, site_mapper, correction_mapper
                )
            )

        return ConstantDepthMatrixProductStatePreparationData(
            num_sites=mps.num_sites,
            num_qubits_per_site=MatrixProductStatePreparation._qubits_per_site(mps_sites),  # noqa: SLF001
            bond_bits=bond_bits,
            site_to_orbital_order=mps.site_to_orbital_order,
            fragments=fragments,
        )

    def _run_impl(self, wavefunction: Wavefunction) -> Circuit:
        """Return a circuit to prepare an MPS state in constant depth.

        Args:
            wavefunction: The wavefunction to prepare.

        Returns:
            A Circuit object implementing the constant-depth MPS state preparation.

        Raises:
            TypeError: If wavefunction is not an MPSContainer instance, or the ``unitary_synthesis``
                setting does not refer to a :class:`UnitarySynthesisMapper`.
            ValueError: If the MPS is not right-canonical with orthogonality center zero, does not
                have exactly one site per molecular orbital, or is rejected by
                :meth:`generate_constant_depth_preparation_data`.

        """
        container = wavefunction.get_container()
        if not isinstance(container, MPSContainer):
            raise TypeError(f"{_DESCRIPTION} requires an MPSContainer, got {type(container)}.")
        if container.orthogonality_center != 0:
            raise ValueError(f"{_DESCRIPTION} requires a right-canonical MPS with center zero.")
        if container.num_sites != container.orbitals.get_num_molecular_orbitals():
            raise ValueError(f"{_DESCRIPTION} requires exactly one MPS site per molecular orbital.")

        data = self.generate_constant_depth_preparation_data(container)
        params = data.to_qsharp_params(self._settings.get("rotation_bit_precision"))
        constant_depth = QSHARP_UTILS.MPSConstantDepth
        # Mid-circuit measurements make the operation Adaptive-only and not adjointable.
        return Circuit(
            qsharp_factory=QsharpFactoryData(program=constant_depth.MakeMPSConstantDepthCircuit, parameter=params),
            qsharp_op=constant_depth.MakeMPSConstantDepthOp(constant_depth.MPSConstantDepthParams(**params)),
            encoding="jordan-wigner",
            num_qubits=data.num_qubits_per_site * data.num_sites,
        )

    def _create_mapper(self, num_input_qubits: int) -> UnitarySynthesisMapper:
        """Return a new nested dense mapper that omits right factors and has the given input register width.

        Raises:
            TypeError: If the ``unitary_synthesis`` setting does not refer to a
                :class:`UnitarySynthesisMapper`.
            ValueError: If the method of the mapper is not ``"dense"``.

        """
        mapper = self._create_nested("unitary_synthesis")
        if not isinstance(mapper, UnitarySynthesisMapper):
            raise TypeError(
                f"{_DESCRIPTION} requires the unitary_synthesis circuit mapper, got "
                f"{mapper.type_name()} '{mapper.name()}'."
            )
        settings = mapper.settings()
        method = settings.get("method")
        if method != "dense":
            raise ValueError(f"{_DESCRIPTION} supports only the 'dense' unitary_synthesis method, got '{method}'.")
        settings.update("num_input_qubits", num_input_qubits)
        settings.update("omit_right_factor", True)
        return mapper

    @staticmethod
    def _first_fragment(
        first_site: np.ndarray, later_sites: Sequence[MPSSite], bond_dim: int, site_mapper: UnitarySynthesisMapper
    ) -> ConstantDepthMPSFragment:
        """Return the first fragment, whose first site is prepared directly with its right bond.

        Args:
            first_site: Dense tensor of the first site with shape (1, physical, right).
            later_sites: Sites of the fragment after the first.
            bond_dim: Dimension of the bond register.
            site_mapper: Dense mapper of the site isometries.

        Returns:
            The fragment with its initial state indexed by physical * bond_dim + bond.

        Raises:
            ValueError: If the initial state has zero norm or a site is not isometric.

        """
        syntheses, right_factor = _synthesize_sites(later_sites, 1, first_site.shape[2], bond_dim, site_mapper)
        initial = first_site[0] @ right_factor.T
        padded = np.zeros((initial.shape[0], bond_dim))
        padded[:, : initial.shape[1]] = initial
        vector = padded.reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 1e-15:
            raise ValueError("MPS initial state must contain finite amplitudes with nonzero norm.")
        return ConstantDepthMPSFragment(
            sites=list(range(len(later_sites) + 1)),
            initial_state_vec=(vector / norm).tolist(),
            num_fusion_qubits=0,
            site_syntheses=syntheses,
        )

    @staticmethod
    def _fused_fragment(
        tensors: Sequence[np.ndarray],
        sites: Sequence[MPSSite],
        start: int,
        bond_dim: int,
        site_mapper: UnitarySynthesisMapper,
        correction_mapper: UnitarySynthesisMapper,
    ) -> ConstantDepthMPSFragment:
        """Return a fragment that starts from a Bell pair on its left bond and is fused with its predecessor.

        Args:
            tensors: Dense site tensors of the fragment with shape (left, physical, right).
            sites: MPS sites of the fragment.
            start: Chain index of the fragment's first site.
            bond_dim: Dimension of the bond register.
            site_mapper: Dense mapper of the site isometries.
            correction_mapper: Dense mapper of the square corrections.

        Returns:
            The fragment with its Bell pair indexed by fusion * bond_dim + bond.

        Raises:
            ValueError: If a Pauli defect on the left bond cannot be pushed through the fragment, or
                a site is not isometric.

        """
        stop = start + len(sites)
        try:
            pushed_labels, rotations = _pushing_relations(_fragment_tensor(tensors))
        except ValueError as error:
            raise ValueError(
                f"{_DESCRIPTION} requires every Pauli defect on a fused bond to be pushable "
                f"(Theorem 2 of Smith et al., PRX Quantum 5, 030344), but a defect on the bond "
                f"before site {start} cannot be pushed through sites {start}..{stop - 1}: {error}"
            ) from error
        left_bond = tensors[0].shape[0]
        syntheses, right_factor = _synthesize_sites(sites, start, left_bond, bond_dim, site_mapper)
        # The left bond starts in the Bell pair sum_a |a> V|a> / sqrt(D), where V is the first
        # site's right factor.
        bell_pair = np.zeros((left_bond, bond_dim))
        bell_pair[:, :left_bond] = right_factor.T / np.sqrt(left_bond)
        # The pushing relation leaves W|n> on the fragment, which W^T undoes.
        corrections = [_synthesize_correction(rotation.T, correction_mapper) for rotation in rotations]
        return ConstantDepthMPSFragment(
            sites=list(range(start, stop)),
            initial_state_vec=bell_pair.reshape(-1).tolist(),
            num_fusion_qubits=(left_bond - 1).bit_length(),
            site_syntheses=syntheses,
            corrections=corrections,
            pushed_labels=pushed_labels,
        )


def _synthesize_sites(
    sites: Sequence[MPSSite], start: int, left_bond: int, bond_dim: int, mapper: UnitarySynthesisMapper
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Synthesize consecutive site isometries whose last one leaves the raw right bond on the bond register.

    As in the ``"dense"`` method of :class:`MatrixProductStatePreparation`, every site circuit omits
    its right factor, which is absorbed into the right bond of the site before.

    Args:
        sites: Consecutive MPS sites.
        start: Chain index of the first site.
        left_bond: Dimension of the left bond of the first site.
        bond_dim: Dimension of the bond register.
        mapper: Dense mapper whose input register is the bond register.

    Returns:
        The Q# ``DenseIsometrySynthesis`` struct of every site, and the right factor of the first
        site, which its predecessor must absorb. The right factor is the identity without sites.

    Raises:
        ValueError: If a site is not isometric.

    """
    syntheses: list[dict[str, Any]] = []
    right_factor: np.ndarray | None = None
    for index in range(len(sites) - 1, -1, -1):
        isometry = MatrixProductStatePreparation._site_isometry(sites[index], bond_dim)  # noqa: SLF001
        if right_factor is not None:
            bond = right_factor.shape[0]
            isometry[:, :bond] = np.einsum("bc,pca->pba", right_factor, isometry[:, :bond])
        try:
            container = RealUnitaryContainer(isometry.reshape(-1, isometry.shape[2]))
        except ValueError as error:
            raise ValueError(
                f"{_DESCRIPTION} requires isometric sites; site {start + index} is not: {error}"
            ) from error
        unitary = UnitaryRepresentation(container)
        syntheses.append(mapper.run(unitary)._qsharp_factory.parameter["synthesis"])  # noqa: SLF001
        right_factor = mapper.right_factor(unitary)
    syntheses.reverse()
    return syntheses, np.eye(left_bond) if right_factor is None else right_factor


def _synthesize_correction(correction: np.ndarray, mapper: UnitarySynthesisMapper) -> dict[str, Any]:
    """Return the Q# ``GivensDecomposition`` struct of a square orthogonal correction.

    A square matrix adds no qubits, so its dense synthesis is the block Givens network alone.
    """
    circuit = mapper.run(UnitaryRepresentation(RealUnitaryContainer(correction)))
    return circuit._qsharp_factory.parameter["synthesis"]["blockGivens"]  # noqa: SLF001


def _pauli_transpose(label: int, num_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Return :math:`(X^x Z^z)^\mathsf{T}` for the label ``x | (z << num_qubits)`` as a signed permutation.

    Row :math:`a` of the transpose holds :math:`(-1)^{|a \wedge z|}` in column :math:`a \oplus x`, as
    :math:`(X^x Z^z)_{ba} = \delta_{b, a \oplus x} (-1)^{|a \wedge z|}`.
    """
    rows = np.arange(1 << num_qubits)
    x, z = label & ((1 << num_qubits) - 1), label >> num_qubits
    signs = np.array([1.0 - 2.0 * (int(row & z).bit_count() & 1) for row in rows])
    return rows ^ x, signs


def _pushing_relations(tensor: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    r"""Find how every Pauli defect on the left bond of a fragment is pushed to its right bond.

    For each defect :math:`B` with label ``x | (z << k)`` for :math:`X^x Z^z`, finds a Pauli
    :math:`H`, the identity first, and an orthogonal :math:`W` with
    :math:`B\,T^m = \sum_n W_{mn} T^n H` (Theorem 1 and Appendix A of :cite:`Smith2024`). Such a
    :math:`W` exists if and only if :math:`O = B^\mathsf{T} \otimes H^{-1}` commutes with
    :math:`\bar{T}^\mathsf{T} \bar{T}`, where :math:`\bar{T}` has rows :math:`m` and columns
    :math:`(a, b)`. Then :math:`W = U (V^\mathsf{T} O V) U^\mathsf{T} + 1 - U U^\mathsf{T}` from the thin
    singular value decomposition :math:`\bar{T} = U S V^\mathsf{T}`.

    Args:
        tensor: Fragment tensor with shape (left, physical, right), both bonds powers of two.

    Returns:
        The label of :math:`H` and the matrix :math:`W` for each defect label.

    Raises:
        ValueError: If a defect cannot be pushed.

    """
    left, physical, right = tensor.shape
    left_qubits, right_qubits = (left - 1).bit_length(), (right - 1).bit_length()
    matrix = tensor.transpose(1, 0, 2).reshape(physical, left * right)
    gram = matrix.T @ matrix
    gram_tolerance = _PUSHING_TOLERANCE * max(1.0, np.linalg.norm(gram))
    left_vectors, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular_values > _PUSHING_TOLERANCE * max(1.0, singular_values[0])))
    left_vectors, right_vectors = left_vectors[:, :rank], right_vectors[:rank].T
    projector = np.eye(physical) - left_vectors @ left_vectors.T
    pushed_paulis = [_pauli_transpose(pushed, right_qubits) for pushed in range(1 << (2 * right_qubits))]

    labels: list[int] = []
    rotations: list[np.ndarray] = []
    for label in range(1 << (2 * left_qubits)):
        defect_columns, defect_signs = _pauli_transpose(label, left_qubits)
        for pushed, (pushed_columns, pushed_signs) in enumerate(pushed_paulis):
            # O = B^T ⊗ H^T is a signed permutation, and O commutes with the Gram matrix exactly
            # when O G O^T = G. The inverse of a real Pauli is its transpose.
            columns = (defect_columns[:, None] * right + pushed_columns[None, :]).reshape(-1)
            signs = np.outer(defect_signs, pushed_signs).reshape(-1)
            conjugated = signs[:, None] * signs[None, :] * gram[np.ix_(columns, columns)]
            if np.linalg.norm(conjugated - gram) <= gram_tolerance:
                core = right_vectors.T @ (signs[:, None] * right_vectors[columns])
                labels.append(pushed)
                rotations.append(left_vectors @ core @ left_vectors.T + projector)
                break
        else:
            raise ValueError(f"no Pauli on the right bond pushes the defect with label {label}.")
    return labels, rotations


def _fragment_tensor(tensors: Sequence[np.ndarray]) -> np.ndarray:
    """Contract site tensors (left, physical, right) into one fragment tensor, first site least significant."""
    fragment = tensors[0]
    for tensor in tensors[1:]:
        left, physical, _ = fragment.shape
        fragment = np.einsum("apb,bqc->aqpc", fragment, tensor).reshape(left, physical * tensor.shape[1], -1)
    return fragment


# ---------------------------------------------------------------------------
# Preparation data passed to Q#
# ---------------------------------------------------------------------------


@dataclass
class ConstantDepthMPSFragment:
    """Preparation and correction data of one fragment, mirroring the Q# ``MPSFragment`` struct."""

    sites: list[int]
    """Chain sites of the fragment, in order."""

    initial_state_vec: list[float]
    """Initial amplitudes on the bond and fusion registers, and the first site for the first fragment."""

    num_fusion_qubits: int
    """Qubits of the left bond fused with the previous fragment; zero for the first fragment."""

    site_syntheses: list[dict[str, Any]] = field(default_factory=list)
    """Q# ``DenseIsometrySynthesis`` struct of every site after the initial state, each without its right factor."""

    corrections: list[dict[str, Any]] = field(default_factory=list)
    """Q# ``GivensDecomposition`` struct of the physical correction of each incoming Pauli defect
    ``x | (z << num_fusion_qubits)``."""

    pushed_labels: list[int] = field(default_factory=list)
    """Outgoing Pauli defect on the right bond for each incoming defect."""

    def to_qsharp(self) -> dict:
        """Return the fields of the Q# ``MPSFragment`` struct."""
        return {
            "initialStateVec": self.initial_state_vec,
            "sites": self.sites,
            "numFusionQubits": self.num_fusion_qubits,
            "siteDecompositions": self.site_syntheses,
            "corrections": self.corrections,
            "pushedLabels": self.pushed_labels,
        }


@dataclass
class ConstantDepthMatrixProductStatePreparationData:
    """All data needed to drive the Q# constant-depth MPS preparation operations.

    Produced by :meth:`ConstantDepthMatrixProductStatePreparation.generate_constant_depth_preparation_data`.
    """

    num_sites: int
    """Number of MPS sites."""

    num_qubits_per_site: int
    """Physical qubits per site: one for the ``('0', '1')`` basis and two for ``('0', 'u', 'd', '2')``."""

    bond_bits: int
    """Number of qubits of each fragment's bond register."""

    site_to_orbital_order: list[int]
    """Orbital that holds each chain site, from the MPS container."""

    fragments: list[ConstantDepthMPSFragment] = field(default_factory=list)
    """Fragments of consecutive sites, in chain order."""

    def to_qsharp_params(self, rotation_bit_precision: int) -> dict:
        """Flatten into the dict expected by ``MakeMPSConstantDepthCircuit``."""
        return {
            "fragments": [fragment.to_qsharp() for fragment in self.fragments],
            "numSites": self.num_sites,
            "numQubitsPerSite": self.num_qubits_per_site,
            "siteToOrbitalOrder": self.site_to_orbital_order,
            "rotationBitPrecision": rotation_bit_precision,
            "numBondQubits": self.bond_bits,
        }
