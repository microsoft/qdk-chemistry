"""Choose and condition an active space for stretched N2.

The workflow progresses from Hartree--Fock orbitals through valence-space CASCI,
natural orbitals, entropy-based autoCAS refinement, and a deterministic choice
of coordinates inside occupation-degenerate natural-orbital subspaces. The last
step coordinate-minimizes the mapped Hamiltonian coefficient norm
``lambda = sum(abs(h_l))`` while preserving the selected orbital subspace and its
CASCI energy.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import cast

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import MajoranaMapping, Orbitals, Structure, Wavefunction
from qdk_chemistry.data.symmetry import SymmetryLabel, axes
from qdk_chemistry.utils import Logger, compute_valence_space_parameters
from qdk_chemistry.utils.cubegen import generate_cubefiles_from_orbitals
from scipy.optimize import minimize_scalar


@dataclass
class ActiveSpaceResult:
    """Results shared by later tutorial chapters and the visualization notebook.

    Attributes:
        structure: Molecular geometry used by every calculation.
        hartree_fock_energy: Restricted Hartree--Fock energy in Hartree.
        valence_wavefunction: Hartree--Fock wavefunction with initial valence
            active-space metadata attached.
        valence_indices: Spatial-orbital indices in the initial valence space.
        num_valence_electrons: Total active electrons in the valence space.
        num_valence_orbitals: Spatial orbitals in the valence space.
        valence_energy: Initial valence-space CASCI energy in Hartree.
        valence_casci_wavefunction: Correlated valence-space wavefunction whose
            reduced density matrices define the natural orbitals.
        num_valence_determinants: Determinants in the complete valence-space CI.
        natural_orbital_energy: CASCI energy after the natural-orbital rotation.
        natural_orbital_casci_wavefunction: Correlated wavefunction recomputed in
            the natural-orbital representation.
        orbital_entropies: One entropy per valence spatial orbital, ordered like
            ``valence_indices``.
        refined_wavefunction: autoCAS wavefunction carrying the selected orbital
            partition before coordinate minimization.
        refined_orbitals: Selected orbitals after coordinate minimization inside
            occupation-degenerate blocks.
        inactive_indices: Frozen doubly occupied spatial-orbital indices.
        refined_indices: Spatial-orbital indices retained as active by autoCAS.
        num_refined_electrons: Electrons in the selected active space.
        num_virtual_orbitals: Spatial orbitals held empty outside the active space.
        refined_energy: Selected-space CASCI energy in Hartree and the later
            algorithmic reference.
        refined_casci_wavefunction: Complete selected-space CASCI wavefunction in
            the coordinate-minimized orbital representation.
        num_refined_determinants: Determinants in the complete selected-space CI.
        natural_orbital_coordinate_minimization: Coordinate-minimization diagnostics.
    """

    structure: Structure
    hartree_fock_energy: float
    valence_wavefunction: Wavefunction
    valence_indices: list[int]
    num_valence_electrons: int
    num_valence_orbitals: int
    valence_energy: float
    valence_casci_wavefunction: Wavefunction
    num_valence_determinants: int
    natural_orbital_energy: float
    natural_orbital_casci_wavefunction: Wavefunction
    orbital_entropies: list[float]
    refined_wavefunction: Wavefunction
    refined_orbitals: Orbitals
    inactive_indices: list[int]
    refined_indices: list[int]
    num_refined_electrons: int
    num_virtual_orbitals: int
    refined_energy: float
    refined_casci_wavefunction: Wavefunction
    num_refined_determinants: int
    natural_orbital_coordinate_minimization: (
        "NaturalOrbitalCoordinateMinimizationResult"
    )


@dataclass(frozen=True)
class NaturalOrbitalCoordinateMinimizationResult:
    """Orbitals and diagnostics from deterministic coordinate minimization.

    Here *gauge* means a choice of orbital vectors used as coordinates for a
    fixed occupation-degenerate subspace. Different gauges span the same
    subspace and have the same exact CASCI energy, but can produce different
    mapped Hamiltonian representations.

    Attributes:
        orbitals: Selected orbitals after AO anchoring and coordinate minimization.
        selected_blocks: Natural-orbital index blocks present in the selected
            active space. Singleton blocks require no rotational search.
        coefficient_norm_before: Mapped coefficient norm ``lambda`` in Hartree
            before deterministic gauge selection.
        coefficient_norm_after: Mapped coefficient norm in Hartree after accepted
            coordinate rotations.
        effective_pauli_terms_before: Pauli terms above the diagnostic threshold
            before gauge selection.
        effective_pauli_terms_after: Corresponding effective term count after
            gauge selection; this is reported but not optimized directly.
    """

    orbitals: Orbitals
    selected_blocks: tuple[tuple[int, ...], ...]
    coefficient_norm_before: float
    coefficient_norm_after: float
    effective_pauli_terms_before: int
    effective_pauli_terms_after: int


def _natural_occupation_blocks(
    reference_wavefunction: Wavefunction,
    natural_indices: list[int],
    *,
    degeneracy_tolerance: float = 1e-6,
) -> tuple[tuple[int, ...], ...]:
    """Group consecutive sorted natural occupations into degeneracy blocks.

    Natural occupations are eigenvalues of the spin-traced active-space
    one-particle RDM and are returned in descending order. Consecutive values
    separated by less than ``degeneracy_tolerance`` define one block whose
    eigenvectors can rotate without changing the RDM.

    Args:
        reference_wavefunction: Correlated wavefunction containing the active
            spin-traced one-particle RDM.
        natural_indices: Orbital indices ordered like the descending natural
            occupations.
        degeneracy_tolerance: Maximum dimensionless occupation-number gap within
            one block.

    Returns:
        Non-overlapping ordered blocks that cover ``natural_indices``.
    """
    one_rdm = np.asarray(reference_wavefunction.get_active_one_rdm_spin_traced())
    occupations = np.linalg.eigvalsh(0.5 * (one_rdm + one_rdm.T))[::-1]

    occupation_blocks: list[tuple[int, ...]] = []
    block_start = 0
    for block_stop in range(1, len(occupations) + 1):
        if block_stop == len(occupations) or (
            abs(occupations[block_stop] - occupations[block_stop - 1])
            >= degeneracy_tolerance
        ):
            occupation_blocks.append(tuple(natural_indices[block_start:block_stop]))
            block_start = block_stop
    return tuple(occupation_blocks)


def _orbitals_with_coefficients(
    template: Orbitals, coefficients: np.ndarray
) -> Orbitals:
    """Copy an orbital object while replacing its coefficient matrix.

    Args:
        template: Source of overlap, basis-set, and active/inactive metadata.
        coefficients: Replacement AO-by-MO coefficient matrix.

    Returns:
        Restricted orbitals with replacement coefficients and metadata copied
        from ``template``. Orbital energies are omitted because rotations inside
        a correlated active space do not assign unique one-electron energies.
    """
    return Orbitals(
        coefficients,
        None,
        template.get_overlap_matrix(),
        template.get_basis_set(),
        template.active_indices(),
        template.inactive_indices(),
    )


def _ao_anchor_block(block: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Orient one natural-orbital block reproducibly using atomic-orbital anchors.

    Degenerate eigenvectors can arrive in any orthonormal orientation. The AO
    projections depend only on the subspace projector, so pivoting those
    projections and symmetrically orthogonalizing them produces the same block
    after arbitrary rotations or sign changes of the input eigenvectors.

    Args:
        block: AO-by-orbital coefficient matrix for one occupation-degenerate block.
        overlap: Atomic-orbital overlap matrix.

    Returns:
        An AO-by-orbital matrix spanning the same subspace, with a deterministic
        orientation and columns orthonormal in the AO metric.

    Raises:
        RuntimeError: If independent AO anchors cannot be found.
    """
    projected_ao = overlap @ block
    residuals = projected_ao.copy()
    anchors = []

    # Pivoted Gram-Schmidt chooses independent AO rows. Rounding the projection
    # norms makes near-ties reproducible; np.argmax then selects the lowest AO
    # index among equal rounded values. The coefficients themselves are not rounded.
    for _ in range(block.shape[1]):
        projection_norms = np.einsum("ij,ij->i", residuals, residuals)
        anchor = int(np.argmax(np.round(projection_norms, decimals=14)))
        anchors.append(anchor)
        anchor_vector = residuals[anchor].copy()
        anchor_norm = np.linalg.norm(anchor_vector)
        if anchor_norm <= np.finfo(float).eps:
            raise RuntimeError(
                "Unable to find independent AO anchors for a natural-orbital block"
            )
        anchor_vector /= anchor_norm
        residuals -= np.outer(residuals @ anchor_vector, anchor_vector)

    # Löwdin symmetric orthogonalization turns the projected AO anchors into an
    # orthonormal rotation without privileging their processing order.
    anchor_coefficients = projected_ao[anchors].T
    gram_values, gram_vectors = np.linalg.eigh(
        anchor_coefficients.T @ anchor_coefficients
    )
    if np.min(gram_values) <= np.finfo(float).eps:
        raise RuntimeError("Natural-orbital AO anchors are linearly dependent")
    orthogonalizer = (
        anchor_coefficients @ gram_vectors @ np.diag(gram_values**-0.5) @ gram_vectors.T
    )
    return block @ orthogonalizer


def _givens_rotation(size: int, left: int, right: int, angle: float) -> np.ndarray:
    """Construct an orthogonal plane rotation inside an orbital block.

    Args:
        size: Dimension of the degenerate orbital block.
        left: First zero-based block-column index to rotate.
        right: Second zero-based block-column index to rotate.
        angle: Rotation angle in radians.

    Returns:
        A ``size``-by-``size`` identity matrix with the selected plane replaced
        by a Givens rotation.

    Raises:
        IndexError: If either selected column lies outside the matrix.
    """
    rotation = np.eye(size)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation[left, left] = cosine
    rotation[left, right] = -sine
    rotation[right, left] = sine
    rotation[right, right] = cosine
    return rotation


def coordinate_minimize_natural_orbital_coefficient_norm(
    reference_wavefunction: Wavefunction,
    selected_orbitals: Orbitals,
    natural_indices: list[int],
    *,
    degeneracy_tolerance: float = 1e-6,
    angle_samples: int = 32,
    max_sweeps: int = 3,
    improvement_tolerance: float = 1e-10,
    effective_pauli_threshold: float = 1e-10,
) -> NaturalOrbitalCoordinateMinimizationResult:
    r"""Coordinate-minimize :math:`\lambda` inside degenerate orbital blocks.

    Natural orbitals with equal occupations define a subspace but not unique
    orbital vectors. First, AO anchoring gives every selected occupation block
    deterministic coordinates. Then deterministic coordinate sweeps search
    plane rotations within degenerate blocks and accept only reductions in the
    mapped coefficient norm :math:`\lambda=\sum_\ell |h_\ell|`.

    Rotations stay inside occupation-degenerate blocks, so they preserve the
    selected orbital subspace and its exact CASCI energy. Minimizing lambda is
    useful for phase-estimation time bounds and future Hamiltonian-simulation
    cost analysis. Effective Pauli counts are diagnostics rather than the
    optimization objective because raw counts are sensitive to pruning noise.

    The search is deterministic coordinate descent, not a proof of the global
    minimum for arbitrary high-dimensional blocks. Each sweep visits every
    Givens plane in a fixed order. A coarse angular grid identifies a candidate
    basin and bounded scalar minimization refines it. Only reductions larger
    than ``improvement_tolerance`` are accepted. For the two-dimensional pi
    blocks in the diatomic examples, each coordinate is a one-angle search;
    larger blocks can have coupled local minima.

    Args:
        reference_wavefunction: Correlated valence-space wavefunction whose
            spin-traced one-particle RDM defines occupation degeneracies.
        selected_orbitals: autoCAS-selected natural orbitals to canonicalize.
        natural_indices: Valence orbital indices ordered like the natural
            occupations in ``reference_wavefunction``.
        degeneracy_tolerance: Dimensionless occupation-number gap below which
            consecutive natural orbitals belong to one degenerate block.
        angle_samples: Uniform samples over :math:`[0,\pi)` used to locate a
            candidate basin for each plane rotation. Must be at least four.
        max_sweeps: Maximum deterministic passes over all block-coordinate planes.
        improvement_tolerance: Minimum absolute reduction in :math:`\lambda`, in
            Hartree, required to accept a rotation or begin another sweep.
        effective_pauli_threshold: Absolute Pauli-coefficient threshold, in
            Hartree, used only for before/after diagnostic term counts.

    Returns:
        Coordinate-minimized orbitals and before/after mapping diagnostics.

    Raises:
        ValueError: If fewer than four angular samples are requested.
        RuntimeError: If autoCAS splits an occupation-degenerate block or AO
            anchoring cannot construct deterministic coordinates.
    """
    if angle_samples < 4:
        raise ValueError("angle_samples must be at least four")

    alpha_channel = SymmetryLabel([axes.alpha()])
    selected_indices = tuple(selected_orbitals.active_indices().indices(alpha_channel))
    selected_index_set = set(selected_indices)
    selected_blocks = []
    for block in _natural_occupation_blocks(
        reference_wavefunction,
        natural_indices,
        degeneracy_tolerance=degeneracy_tolerance,
    ):
        selected_members = selected_index_set.intersection(block)
        if selected_members and len(selected_members) != len(block):
            raise RuntimeError(
                f"The selected active space splits degenerate natural-orbital block {block}"
            )
        if selected_members:
            selected_blocks.append(block)

    coefficients = np.array(
        selected_orbitals.coefficients().block((alpha_channel, alpha_channel)),
        copy=True,
    )
    overlap = selected_orbitals.get_overlap_matrix()
    original_coefficients = coefficients.copy()

    # AO anchoring fixes signs, permutations, and common rotations before the
    # lambda search. If lambda has equivalent minima, keeping a zero rotation
    # below preserves this deterministic AO-based tie-break.
    for block in selected_blocks:
        coefficients[:, block] = _ao_anchor_block(coefficients[:, block], overlap)

    hamiltonian_constructor = create("hamiltonian_constructor", "qdk")
    qubit_mapper = create(
        "qubit_mapper",
        "qdk",
        threshold=1e-14,
        integral_threshold=1e-14,
    )
    mapping = MajoranaMapping.jordan_wigner(num_modes=2 * len(selected_indices))

    def mapped_metrics(candidate_coefficients: np.ndarray) -> tuple[float, int]:
        """Map one candidate and return ``(lambda, effective Pauli terms)``.

        Mapping uses tight ``1e-14`` integral and Pauli thresholds so pruning
        does not create artificial minima. The separately reported effective
        count applies ``effective_pauli_threshold`` for a stable diagnostic.
        """
        candidate_orbitals = _orbitals_with_coefficients(
            selected_orbitals, candidate_coefficients
        )
        hamiltonian = hamiltonian_constructor.run(candidate_orbitals)
        qubit_hamiltonian = qubit_mapper.run(hamiltonian, mapping)
        effective_terms = int(
            np.count_nonzero(
                np.abs(qubit_hamiltonian.coefficients) >= effective_pauli_threshold
            )
        )
        return float(qubit_hamiltonian.schatten_norm), effective_terms

    def rotated_norm(
        angle: float,
        *,
        base_coefficients: np.ndarray,
        block_coefficients: np.ndarray,
        block: tuple[int, ...],
        left: int,
        right: int,
    ) -> float:
        """Return lambda after one candidate Givens rotation.

        ``partial`` binds the current block, coordinate plane, and coefficient
        snapshots, leaving a one-argument objective for scalar minimization.
        """
        candidate_coefficients = base_coefficients.copy()
        candidate_coefficients[:, block] = block_coefficients @ _givens_rotation(
            len(block), left, right, angle
        )
        return mapped_metrics(candidate_coefficients)[0]

    norm_before, terms_before = mapped_metrics(original_coefficients)
    current_norm, _ = mapped_metrics(coefficients)
    angle_grid = np.linspace(0.0, np.pi, angle_samples, endpoint=False)
    angle_step = np.pi / angle_samples

    # Coordinate descent supports blocks larger than two through Givens plane
    # rotations. Diatomic pi blocks are two-dimensional, so each sweep reduces
    # to a transparent one-angle search.
    for _ in range(max_sweeps):
        sweep_start_norm = current_norm
        for block in selected_blocks:
            if len(block) < 2:
                continue
            for left, right in combinations(range(len(block)), 2):
                block_coefficients = coefficients[:, block].copy()
                norm_at_angle = partial(
                    rotated_norm,
                    base_coefficients=coefficients,
                    block_coefficients=block_coefficients,
                    block=block,
                    left=left,
                    right=right,
                )

                grid_norms = np.asarray(
                    [norm_at_angle(float(angle)) for angle in angle_grid]
                )
                best_index = int(np.argmin(grid_norms))
                if grid_norms[best_index] >= current_norm - improvement_tolerance:
                    continue

                best_angle = float(angle_grid[best_index])
                refinement = minimize_scalar(
                    norm_at_angle,
                    bounds=(best_angle - angle_step, best_angle + angle_step),
                    method="bounded",
                    options={"xatol": 1e-13},
                )
                if refinement.fun < current_norm - improvement_tolerance:
                    coefficients[:, block] = block_coefficients @ _givens_rotation(
                        len(block), left, right, float(refinement.x % np.pi)
                    )
                    current_norm = float(refinement.fun)

        if current_norm >= sweep_start_norm - improvement_tolerance:
            break

    norm_after, terms_after = mapped_metrics(coefficients)
    return NaturalOrbitalCoordinateMinimizationResult(
        orbitals=_orbitals_with_coefficients(selected_orbitals, coefficients),
        selected_blocks=tuple(selected_blocks),
        coefficient_norm_before=norm_before,
        coefficient_norm_after=norm_after,
        effective_pauli_terms_before=terms_before,
        effective_pauli_terms_after=terms_after,
    )


def run_active_space_workflow() -> ActiveSpaceResult:
    """Build the correlated molecular model used by later tutorial chapters.

    The workflow computes Hartree--Fock orbitals, forms a complete valence-space
    CASCI reference, diagonalizes its one-particle RDM to obtain natural
    orbitals, applies entropy-based autoCAS refinement, coordinate-minimizes the
    selected degenerate-orbital gauge, and solves the final selected-space CASCI
    problem.

    Returns:
        Energies, wavefunctions, orbital partitions, mapping diagnostics, and
        coordinate-minimized selected orbitals needed by mapping, state
        preparation, and visualization.
    """
    ################################################################################
    # docs:xyz ../data/tutorial_stretched_n2.structure.xyz
    # start-cell-hartree-fock
    structure = Structure.from_xyz("""\
2
Stretched N2 molecule for the ground-state QPE tutorial
N    0.000000    0.000000    0.000000
N    0.000000    0.000000    1.850000
""")
    charge = 0
    spin_multiplicity = 1
    basis_set = "cc-pvdz"

    scf_solver = create("scf_solver", "qdk")
    hartree_fock_energy, hartree_fock_wavefunction = scf_solver.run(
        structure,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        basis_or_guess=basis_set,
    )
    # end-cell-hartree-fock
    ################################################################################

    ################################################################################
    # start-cell-valence-space
    num_valence_electrons, num_valence_orbitals = compute_valence_space_parameters(
        hartree_fock_wavefunction, charge
    )
    valence_selector = create(
        "active_space_selector",
        "qdk_valence",
        num_active_electrons=num_valence_electrons,
        num_active_orbitals=num_valence_orbitals,
    )
    valence_wavefunction = valence_selector.run(hartree_fock_wavefunction)

    # Restricted alpha and beta channels contain the same spatial-orbital indices,
    # so read one channel to count each spatial orbital once.
    alpha_channel = SymmetryLabel([axes.alpha()])
    valence_indices = list(
        valence_wavefunction.get_orbitals().active_indices().indices(alpha_channel)
    )
    num_valence_alpha, num_valence_beta = (
        valence_wavefunction.get_active_num_electrons()
    )
    # end-cell-valence-space
    ################################################################################

    ################################################################################
    # start-cell-initial-casci
    hamiltonian_constructor = create("hamiltonian_constructor")
    casci_solver = create(
        "multi_configuration_calculator",
        "macis_cas",
        # autoCAS entropies require both one- and two-particle RDMs.
        calculate_one_rdm=True,
        calculate_two_rdm=True,
    )

    valence_hamiltonian = hamiltonian_constructor.run(
        valence_wavefunction.get_orbitals()
    )
    valence_energy, valence_casci_wavefunction = casci_solver.run(
        valence_hamiltonian,
        num_valence_alpha,
        num_valence_beta,
    )
    num_valence_determinants = len(valence_casci_wavefunction.get_coefficients())
    # end-cell-initial-casci
    ################################################################################

    ################################################################################
    # start-cell-natural-orbitals
    # Rotate the valence orbitals using the CASCI one-particle RDM so each
    # natural orbital has a well-defined correlated occupation.
    natural_orbital_localizer = create("orbital_localizer", "qdk_natural_orbitals")
    natural_orbital_wavefunction = natural_orbital_localizer.run(
        valence_casci_wavefunction,
        valence_indices,
        valence_indices,
    )

    # Rebuild and solve in the rotated basis so the RDMs and orbital entropies
    # describe the same natural-orbital representation.
    natural_orbital_hamiltonian = hamiltonian_constructor.run(
        natural_orbital_wavefunction.get_orbitals()
    )
    natural_orbital_energy, natural_orbital_casci_wavefunction = casci_solver.run(
        natural_orbital_hamiltonian,
        num_valence_alpha,
        num_valence_beta,
    )
    # Store ordinary Python floats rather than library scalar types so the
    # values can be printed and passed to the visualization notebook directly.
    orbital_entropies = [
        float(value)
        for value in natural_orbital_casci_wavefunction.get_single_orbital_entropies()
    ]
    # end-cell-natural-orbitals
    ################################################################################

    ################################################################################
    # start-cell-refine
    # autoCAS uses the RDM-derived orbital entropies to retain the orbitals that
    # carry the strongest correlation in a smaller active space.
    autocas_selector = create("active_space_selector", "qdk_autocas_eos")
    refined_wavefunction = autocas_selector.run(natural_orbital_casci_wavefunction)
    refined_orbitals = refined_wavefunction.get_orbitals()

    # Summarize the inactive, selected active, and virtual spatial-orbital spaces.
    alpha_channel = SymmetryLabel([axes.alpha()])
    refined_indices = list(refined_orbitals.active_indices().indices(alpha_channel))
    inactive_indices = list(refined_orbitals.inactive_indices().indices(alpha_channel))
    num_refined_alpha, num_refined_beta = (
        refined_wavefunction.get_active_num_electrons()
    )
    num_refined_electrons = num_refined_alpha + num_refined_beta
    num_refined_orbitals = len(refined_indices)
    num_virtual_orbitals = (
        refined_orbitals.get_num_molecular_orbitals()
        - len(inactive_indices)
        - num_refined_orbitals
    )

    # Natural occupations can be degenerate, leaving the corresponding orbital
    # vectors free to rotate. Choose that gauge only after autoCAS has selected
    # the final active subspace, because lambda belongs to its mapped Hamiltonian.
    coordinate_minimization = coordinate_minimize_natural_orbital_coefficient_norm(
        valence_casci_wavefunction,
        refined_orbitals,
        valence_indices,
    )
    refined_orbitals = coordinate_minimization.orbitals
    # end-cell-refine
    ################################################################################

    ################################################################################
    # start-cell-final-casci
    refined_hamiltonian = hamiltonian_constructor.run(refined_orbitals)
    refined_energy, refined_casci_wavefunction = casci_solver.run(
        refined_hamiltonian,
        num_refined_alpha,
        num_refined_beta,
    )
    num_refined_determinants = len(refined_casci_wavefunction.get_coefficients())
    # end-cell-final-casci
    ################################################################################

    return ActiveSpaceResult(
        structure=structure,
        hartree_fock_energy=hartree_fock_energy,
        valence_wavefunction=valence_wavefunction,
        valence_indices=valence_indices,
        num_valence_electrons=num_valence_electrons,
        num_valence_orbitals=num_valence_orbitals,
        valence_energy=valence_energy,
        valence_casci_wavefunction=valence_casci_wavefunction,
        num_valence_determinants=num_valence_determinants,
        natural_orbital_energy=natural_orbital_energy,
        natural_orbital_casci_wavefunction=natural_orbital_casci_wavefunction,
        orbital_entropies=orbital_entropies,
        refined_wavefunction=refined_wavefunction,
        refined_orbitals=refined_orbitals,
        inactive_indices=inactive_indices,
        refined_indices=refined_indices,
        num_refined_electrons=num_refined_electrons,
        num_virtual_orbitals=num_virtual_orbitals,
        refined_energy=refined_energy,
        refined_casci_wavefunction=refined_casci_wavefunction,
        num_refined_determinants=num_refined_determinants,
        natural_orbital_coordinate_minimization=coordinate_minimization,
    )


def print_active_space_results(result: ActiveSpaceResult) -> None:
    """Print active-space evidence for the cumulative lab notebook.

    The output distinguishes energy invariance under the natural-orbital
    rotation, entropy-based orbital selection, coordinate-minimization
    diagnostics, and the energy cost of reducing the active space.

    Args:
        result: Completed active-space workflow.
    """
    print(f"Hartree-Fock energy: {result.hartree_fock_energy:.12f} Hartree")
    print(
        f"Initial valence space: CAS({result.num_valence_electrons}e, "
        f"{result.num_valence_orbitals}o)"
    )
    print(f"Initial active orbital indices: {result.valence_indices}")
    print(f"Initial CASCI energy: {result.valence_energy:.12f} Hartree")
    print(f"Initial CASCI determinants: {result.num_valence_determinants}")
    print(f"Natural-orbital CASCI energy: {result.natural_orbital_energy:.12f} Hartree")
    orbital_transformation_energy_change = (
        result.natural_orbital_energy - result.valence_energy
    )
    print(
        "Energy change after the natural-orbital transformation: "
        f"{orbital_transformation_energy_change:.12e} Hartree"
    )
    print("Single-orbital entropies:")
    for orbital_index, entropy in zip(
        result.valence_indices, result.orbital_entropies, strict=True
    ):
        selection_marker = "*" if orbital_index in result.refined_indices else " "
        print(f" {selection_marker} orbital {orbital_index}: {entropy:.9f}")
    print(
        f"Refined active space: CAS({result.num_refined_electrons}e, {len(result.refined_indices)}o)"
    )
    print(f"Inactive orbital indices: {result.inactive_indices}")
    print(f"Active orbital indices: {result.refined_indices}")
    print(f"Virtual orbitals: {result.num_virtual_orbitals}")
    print(
        "Degenerate selected orbital blocks: "
        f"{result.natural_orbital_coordinate_minimization.selected_blocks}"
    )
    print(
        "Mapped coefficient norm before/after coordinate minimization: "
        f"{result.natural_orbital_coordinate_minimization.coefficient_norm_before:.12f} / "
        f"{result.natural_orbital_coordinate_minimization.coefficient_norm_after:.12f} Hartree"
    )
    print(
        "Effective Pauli terms before/after coordinate minimization: "
        f"{result.natural_orbital_coordinate_minimization.effective_pauli_terms_before} / "
        f"{result.natural_orbital_coordinate_minimization.effective_pauli_terms_after}"
    )
    print(f"Final CASCI energy: {result.refined_energy:.12f} Hartree")
    print(f"Final CASCI determinants: {result.num_refined_determinants}")
    energy_increase = result.refined_energy - result.natural_orbital_energy
    print(
        f"Energy increase from reducing the active space: {energy_increase:.12f} Hartree"
    )


def generate_active_orbital_cube_data(
    result: ActiveSpaceResult,
    grid_size: tuple[int, int, int] = (30, 30, 30),
    margin: float = 10.0,
) -> dict[str, dict]:
    """Generate molecular-viewer data for every candidate natural orbital.

    Args:
        result: Completed workflow containing valence natural orbitals,
            occupations, entropies, and selected indices.
        grid_size: Cube-grid points along each Cartesian direction. Tests use a
            smaller grid to reduce runtime; the default supports smooth viewing.
        margin: Additional cube-grid extent around the molecule in Bohr.

    Returns:
        A mapping from display labels to cube-file text and an ``info`` mapping
        containing total spatial-orbital occupation, entropy, and autoCAS
        selection status. Total occupation is alpha plus beta occupation.
    """
    wavefunction = result.natural_orbital_casci_wavefunction
    orbitals = wavefunction.get_orbitals()
    occupation_alpha, occupation_beta = wavefunction.get_active_orbital_occupations()

    # Occupation arrays use active-space positions, while cube-file labels use
    # the original molecular-orbital indices; this dictionary connects them.
    active_position = {
        orbital_index: position
        for position, orbital_index in enumerate(result.valence_indices)
    }
    raw_cube_data = cast(
        dict[str, str],
        generate_cubefiles_from_orbitals(
            orbitals=orbitals,
            grid_size=grid_size,
            margin=margin,
            indices=result.valence_indices,
        ),
    )

    cube_data = {}
    for raw_label, cube_file in raw_cube_data.items():
        # Cube labels number orbitals from one, while QDK/Chemistry indices start
        # from zero, so convert before looking up occupations and entropies.
        orbital_index = int(raw_label.split("_")[1]) - 1
        position = active_position[orbital_index]
        # Add the alpha and beta occupations to report the total occupation of
        # each spatial orbital in the viewer.
        occupation = float(occupation_alpha[position]) + float(
            occupation_beta[position]
        )
        cube_data[f"Orbital {orbital_index}"] = {
            "data": cube_file,
            "info": {
                "Occupation": f"{occupation:.3f}",
                "Entropy": f"{result.orbital_entropies[position]:.3f}",
                "Selected by autoCAS": "yes"
                if orbital_index in result.refined_indices
                else "no",
            },
        }
    return cube_data


def main() -> None:
    """Run the command-line workflow and print its lab-notebook evidence."""
    Logger.set_global_level(Logger.LogLevel.off)
    # Earlier tutorial scripts execute one linear calculation. This chapter also
    # provides an interactive notebook, so functions keep both versions on the
    # same tested chemistry workflow instead of duplicating the calculation.
    result = run_active_space_workflow()
    print_active_space_results(result)


if __name__ == "__main__":
    main()
