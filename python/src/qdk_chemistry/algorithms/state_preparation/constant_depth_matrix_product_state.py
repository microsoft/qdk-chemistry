"""Constant-depth matrix product state (MPS) preparation with adaptive circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry._core.utils.unitary_synthesis import dense_unitary_synthesis, orthogonal_givens_decomposition
from qdk_chemistry.data import MPSContainer, Settings, Wavefunction
from qdk_chemistry.data.circuit import Circuit, QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .matrix_product_state import MatrixProductStatePreparation, _givens_to_qsharp, _site_to_qsharp
from .state_preparation import StatePreparation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qdk_chemistry._core.utils.unitary_synthesis import DenseSiteSynthesis, GivensDecomposition
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


class ConstantDepthMatrixProductStatePreparation(StatePreparation):
    r"""Constant-depth matrix product state preparation with measurements and feed-forward.

    Implements Protocol 1 of :cite:`Smith2024`. The chain is split into fragments of
    ``fragment_size`` consecutive sites. Every fragment is prepared in parallel by the dense site
    unitaries of :class:`MatrixProductStatePreparation`, starting from a Bell pair on its left bond.
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
            TypeError: If the input is not an MPSContainer.
            ValueError: If the physical basis is unsupported, the MPS is complex, the fragment size is
                below one, a fused bond is not a power of two, a defect cannot be pushed through a
                fragment, the initial state has zero norm, or a synthesized site is not isometric.

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
        tensors = [site.to_dense().reshape(site.shape) for site in mps_sites]
        bounds = [*range(0, mps.num_sites, fragment_size), mps.num_sites]
        for start in bounds[1:-1]:
            fused_bond = tensors[start].shape[0]
            if fused_bond & (fused_bond - 1):
                raise ValueError(
                    f"{_DESCRIPTION} requires power-of-two bond dimensions between fragments, "
                    f"but the bond before site {start} has dimension {fused_bond}."
                )

        fragments = [self._first_fragment(tensors[0], mps_sites[1 : bounds[1]], bond_dim)]
        for start, stop in itertools.pairwise(bounds[1:]):
            fragments.append(self._fused_fragment(tensors[start:stop], mps_sites[start:stop], start, bond_dim))

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
            TypeError: If wavefunction is not an MPSContainer instance.
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

    @staticmethod
    def _first_fragment(
        first_site: np.ndarray, later_sites: Sequence[MPSSite], bond_dim: int
    ) -> ConstantDepthMPSFragment:
        """Return the first fragment, whose first site is prepared directly with its right bond.

        Args:
            first_site: Dense tensor of the first site with shape (1, physical, right).
            later_sites: Sites of the fragment after the first.
            bond_dim: Dimension of the bond register.

        Returns:
            The fragment with its initial state indexed by physical * bond_dim + bond.

        Raises:
            ValueError: If the initial state has zero norm.

        """
        syntheses = _synthesize_sites(later_sites, bond_dim)
        initial = first_site[0]
        if syntheses:
            initial = initial @ syntheses[0].right_factor.T
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
        tensors: Sequence[np.ndarray], sites: Sequence[MPSSite], start: int, bond_dim: int
    ) -> ConstantDepthMPSFragment:
        """Return a fragment that starts from a Bell pair on its left bond and is fused with its predecessor.

        Args:
            tensors: Dense site tensors of the fragment with shape (left, physical, right).
            sites: MPS sites of the fragment.
            start: Chain index of the fragment's first site.
            bond_dim: Dimension of the bond register.

        Returns:
            The fragment with its Bell pair indexed by fusion * bond_dim + bond.

        Raises:
            ValueError: If a Pauli defect on the left bond cannot be pushed through the fragment.

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
        syntheses = _synthesize_sites(sites, bond_dim)
        # The left bond starts in the Bell pair sum_a |a> V|a> / sqrt(D), where V is the first
        # site's right factor.
        left_bond = tensors[0].shape[0]
        bell_pair = np.zeros((left_bond, bond_dim))
        bell_pair[:, :left_bond] = syntheses[0].right_factor.T / np.sqrt(left_bond)
        return ConstantDepthMPSFragment(
            sites=list(range(start, stop)),
            initial_state_vec=bell_pair.reshape(-1).tolist(),
            num_fusion_qubits=(left_bond - 1).bit_length(),
            site_syntheses=syntheses,
            # The pushing relation leaves W|n> on the fragment, which W^T undoes.
            corrections=[orthogonal_givens_decomposition(rotation.T) for rotation in rotations],
            pushed_labels=pushed_labels,
        )


def _synthesize_sites(sites: Sequence[MPSSite], bond_dim: int) -> list[DenseSiteSynthesis]:
    """Synthesize consecutive site unitaries whose last one leaves the raw right bond on the bond register.

    Every other site unitary rotates its right bond by the next site's right factor, as in
    :class:`MatrixProductStatePreparation`.
    """
    syntheses: list[DenseSiteSynthesis] = []
    following = np.zeros((0, 0))
    for site in reversed(sites):
        synthesis = dense_unitary_synthesis(site, bond_dim, following)
        syntheses.insert(0, synthesis)
        following = synthesis.right_factor
    return syntheses


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

    site_syntheses: list[DenseSiteSynthesis] = field(default_factory=list)
    """Site unitaries after the initial state."""

    corrections: list[GivensDecomposition] = field(default_factory=list)
    """Physical correction of each incoming Pauli defect ``x | (z << num_fusion_qubits)``."""

    pushed_labels: list[int] = field(default_factory=list)
    """Outgoing Pauli defect on the right bond for each incoming defect."""

    def to_qsharp(self) -> dict:
        """Return the fields of the Q# ``MPSFragment`` struct."""
        return {
            "initialStateVec": self.initial_state_vec,
            "sites": self.sites,
            "numFusionQubits": self.num_fusion_qubits,
            "siteDecompositions": [_site_to_qsharp(site) for site in self.site_syntheses],
            "corrections": [_givens_to_qsharp(givens) for givens in self.corrections],
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
