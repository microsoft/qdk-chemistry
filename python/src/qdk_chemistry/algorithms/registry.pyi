"""Type stubs for registry.create() with all algorithm overloads."""

from typing import Literal, overload, Union
from .base import Algorithm

import qdk_chemistry.algorithms.active_space_selector
import qdk_chemistry.algorithms.circuit_executor.qdk
import qdk_chemistry.algorithms.dynamical_correlation_calculator
import qdk_chemistry.algorithms.energy_estimator.qdk
import qdk_chemistry.algorithms.hamiltonian_constructor
import qdk_chemistry.algorithms.multi_configuration_calculator
import qdk_chemistry.algorithms.orbital_localizer
import qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation
import qdk_chemistry.algorithms.projected_multi_configuration_calculator
import qdk_chemistry.algorithms.qubit_hamiltonian_solver
import qdk_chemistry.algorithms.qubit_mapper.qdk_qubit_mapper
import qdk_chemistry.algorithms.scf_solver
import qdk_chemistry.algorithms.stability_checker
import qdk_chemistry.algorithms.state_preparation.sparse_isometry
import qdk_chemistry.algorithms.time_evolution.builder.partially_randomized
import qdk_chemistry.algorithms.time_evolution.builder.qdrift
import qdk_chemistry.algorithms.time_evolution.builder.trotter
import qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.pauli_sequence_mapper

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_occupation'] | None = None,
    occupation_threshold: float = 0.1,
) -> qdk_chemistry.algorithms.active_space_selector.QdkOccupationActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_autocas_eos'] | None = None,
    diff_threshold: float = 0.1,
    entropy_threshold: float = 0.14,
    normalize_entropies: bool = True,
) -> qdk_chemistry.algorithms.active_space_selector.QdkAutocasEosActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_autocas'] | None = None,
    entropy_threshold: float = 0.14,
    min_plateau_size: unknown = 10,
    normalize_entropies: bool = True,
    num_bins: unknown = 100,
) -> qdk_chemistry.algorithms.active_space_selector.QdkAutocasActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_valence'] | None = None,
    num_active_electrons: unknown = -1,
    num_active_orbitals: unknown = -1,
) -> qdk_chemistry.algorithms.active_space_selector.QdkValenceActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['hamiltonian_constructor'],
    algorithm_name: Literal['qdk_cholesky'] | None = None,
    cholesky_tolerance: float = 1e-08,
    eri_threshold: float = 1e-12,
    scf_type: str = "auto",
    store_cholesky_vectors: bool = False,
) -> qdk_chemistry.algorithms.hamiltonian_constructor.QdkCholeskyHamiltonianConstructor: ...

@overload
def create(
    algorithm_type: Literal['hamiltonian_constructor'],
    algorithm_name: Literal['qdk'] | None = None,
    eri_method: str = "direct",
    scf_type: str = "auto",
) -> qdk_chemistry.algorithms.hamiltonian_constructor.QdkHamiltonianConstructor: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_vvhv'] | None = None,
    max_iterations: unknown = 10000,
    minimal_basis: str = "sto-3g",
    small_rotation_tolerance: float = 1e-12,
    tolerance: float = 1e-06,
    weighted_orthogonalization: bool = True,
) -> qdk_chemistry.algorithms.orbital_localizer.QdkVVHVLocalizer: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_mp2_natural_orbitals'] | None = None,
) -> qdk_chemistry.algorithms.orbital_localizer.QdkMP2NaturalOrbitalLocalizer: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_pipek_mezey'] | None = None,
    max_iterations: unknown = 10000,
    small_rotation_tolerance: float = 1e-12,
    tolerance: float = 1e-06,
) -> qdk_chemistry.algorithms.orbital_localizer.QdkPipekMezeyLocalizer: ...

@overload
def create(
    algorithm_type: Literal['multi_configuration_calculator'],
    algorithm_name: Literal['macis_asci'] | None = None,
    calculate_mutual_information: bool = False,
    calculate_one_rdm: bool = False,
    calculate_single_orbital_entropies: bool = False,
    calculate_two_orbital_entropies: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    constraint_level: unknown = 2,
    core_selection_strategy: str = "percentage",
    core_selection_threshold: float = 0.95,
    grow_factor: float = 8.0,
    grow_with_rot: bool = False,
    growth_backoff_rate: float = 0.5,
    growth_recovery_rate: float = 1.1,
    h_el_tol: float = 1e-08,
    just_singles: bool = False,
    max_refine_iter: unknown = 6,
    max_solver_iterations: unknown = 200,
    min_grow_factor: float = 1.01,
    ncdets_max: unknown = 100,
    ntdets_max: unknown = 100000,
    ntdets_min: unknown = 100,
    nxtval_bcount_inc: unknown = 10,
    nxtval_bcount_thresh: unknown = 1000,
    pair_size_max: unknown = 500000000,
    pt2_bigcon_thresh: unknown = 250,
    pt2_constraint_refine_force: unknown = 0,
    pt2_max_constraint_level: unknown = 5,
    pt2_min_constraint_level: unknown = 0,
    pt2_precompute_eps: bool = False,
    pt2_precompute_idx: bool = False,
    pt2_print_progress: bool = False,
    pt2_prune: bool = False,
    pt2_reserve_count: unknown = 70000000,
    pt2_tol: float = 1e-16,
    refine_energy_tol: float = 1e-06,
    rot_size_start: unknown = 1000,
    rv_prune_tol: float = 1e-08,
) -> qdk_chemistry.algorithms.multi_configuration_calculator.QdkMacisAsci: ...

@overload
def create(
    algorithm_type: Literal['multi_configuration_calculator'],
    algorithm_name: Literal['macis_cas'] | None = None,
    calculate_mutual_information: bool = False,
    calculate_one_rdm: bool = False,
    calculate_single_orbital_entropies: bool = False,
    calculate_two_orbital_entropies: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    max_solver_iterations: unknown = 200,
) -> qdk_chemistry.algorithms.multi_configuration_calculator.QdkMacisCas: ...

@overload
def create(
    algorithm_type: Literal['projected_multi_configuration_calculator'],
    algorithm_name: Literal['macis_pmc'] | None = None,
    H_thresh: float = 1e-16,
    calculate_mutual_information: bool = False,
    calculate_one_rdm: bool = False,
    calculate_single_orbital_entropies: bool = False,
    calculate_two_orbital_entropies: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    davidson_max_m: unknown = 200,
    davidson_res_tol: float = 1e-08,
    h_el_tol: float = 1e-08,
    iterative_solver_dimension_cutoff: unknown = 100,
    max_solver_iterations: unknown = 200,
) -> qdk_chemistry.algorithms.projected_multi_configuration_calculator.QdkMacisPmc: ...

@overload
def create(
    algorithm_type: Literal['dynamical_correlation_calculator'],
    algorithm_name: Literal['qdk_mp2_calculator'] | None = None,
) -> qdk_chemistry.algorithms.dynamical_correlation_calculator.DynamicalCorrelationCalculator: ...

@overload
def create(
    algorithm_type: Literal['scf_solver'],
    algorithm_name: Literal['qdk'] | None = None,
    convergence_threshold: float = 1e-07,
    enable_gdm: bool = True,
    energy_thresh_diis_switch: float = 0.001,
    eri_method: str = "direct",
    eri_threshold: float = -1.0,
    eri_use_atomics: bool = False,
    fock_reset_steps: unknown = 1073741824,
    gdm_bfgs_history_size_limit: unknown = 50,
    gdm_max_diis_iteration: unknown = 50,
    level_shift: float = -1.0,
    max_iterations: unknown = 50,
    method: str = "hf",
    nthreads: unknown = -1,
    scf_type: str = "auto",
    shell_pair_threshold: float = 1e-12,
) -> qdk_chemistry.algorithms.scf_solver.QdkScfSolver: ...

@overload
def create(
    algorithm_type: Literal['stability_checker'],
    algorithm_name: Literal['qdk'] | None = None,
    davidson_tolerance: float = 1e-08,
    external: bool = False,
    internal: bool = True,
    max_subspace: unknown = 80,
    method: str = "hf",
    stability_tolerance: float = -0.0001,
) -> qdk_chemistry.algorithms.stability_checker.QdkStabilityChecker: ...

@overload
def create(
    algorithm_type: Literal['energy_estimator'],
    algorithm_name: Literal['qdk'] | None = None,
) -> qdk_chemistry.algorithms.energy_estimator.qdk.QdkEnergyEstimator: ...

@overload
def create(
    algorithm_type: Literal['state_prep'],
    algorithm_name: Literal['sparse_isometry_gf2x'] | None = None,
    basis_gates: list[str] = ['x', 'y', 'z', 'cx', 'cz', 'id', 'h', 's', 'sdg', 'rz'],
    dense_preparation_method: str = "qdk",
    transpile: bool = True,
    transpile_optimization_level: unknown = 0,
) -> qdk_chemistry.algorithms.state_preparation.sparse_isometry.SparseIsometryGF2XStatePreparation: ...

@overload
def create(
    algorithm_type: Literal['qubit_mapper'],
    algorithm_name: Literal['qdk'] | None = None,
    encoding: str = "jordan-wigner",
    integral_threshold: float = 1e-12,
    threshold: float = 1e-12,
) -> qdk_chemistry.algorithms.qubit_mapper.qdk_qubit_mapper.QdkQubitMapper: ...

@overload
def create(
    algorithm_type: Literal['qubit_hamiltonian_solver'],
    algorithm_name: Literal['qdk_dense_matrix_solver'] | None = None,
) -> qdk_chemistry.algorithms.qubit_hamiltonian_solver.DenseMatrixSolver: ...

@overload
def create(
    algorithm_type: Literal['qubit_hamiltonian_solver'],
    algorithm_name: Literal['qdk_sparse_matrix_solver'] | None = None,
    max_m: unknown = 20,
    tol: float = 1e-08,
) -> qdk_chemistry.algorithms.qubit_hamiltonian_solver.SparseMatrixSolver: ...

@overload
def create(
    algorithm_type: Literal['time_evolution_builder'],
    algorithm_name: Literal['trotter'] | None = None,
    error_bound: str = "commutator",
    num_divisions: unknown = 0,
    order: unknown = 1,
    target_accuracy: float = 0.0,
    weight_threshold: float = 1e-12,
) -> qdk_chemistry.algorithms.time_evolution.builder.trotter.Trotter: ...

@overload
def create(
    algorithm_type: Literal['time_evolution_builder'],
    algorithm_name: Literal['qdrift'] | None = None,
    commutation_type: str = "general",
    merge_duplicate_terms: bool = True,
    num_samples: unknown = 100,
    seed: unknown = -1,
) -> qdk_chemistry.algorithms.time_evolution.builder.qdrift.QDrift: ...

@overload
def create(
    algorithm_type: Literal['time_evolution_builder'],
    algorithm_name: Literal['partially_randomized'] | None = None,
    commutation_type: str = "general",
    merge_duplicate_terms: bool = True,
    num_random_samples: unknown = 100,
    seed: unknown = -1,
    tolerance: float = 1e-12,
    trotter_order: unknown = 2,
    weight_threshold: float = -1.0,
) -> qdk_chemistry.algorithms.time_evolution.builder.partially_randomized.PartiallyRandomized: ...

@overload
def create(
    algorithm_type: Literal['controlled_evolution_circuit_mapper'],
    algorithm_name: Literal['pauli_sequence'] | None = None,
    power: unknown = 1,
) -> qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.pauli_sequence_mapper.PauliSequenceMapper: ...

@overload
def create(
    algorithm_type: Literal['circuit_executor'],
    algorithm_name: Literal['qdk_full_state_simulator'] | None = None,
    seed: unknown = 42,
    type: str = "cpu",
) -> qdk_chemistry.algorithms.circuit_executor.qdk.QdkFullStateSimulator: ...

@overload
def create(
    algorithm_type: Literal['circuit_executor'],
    algorithm_name: Literal['qdk_sparse_state_simulator'] | None = None,
    seed: unknown = 42,
) -> qdk_chemistry.algorithms.circuit_executor.qdk.QdkSparseStateSimulator: ...

@overload
def create(
    algorithm_type: Literal['phase_estimation'],
    algorithm_name: Literal['iterative'] | None = None,
    evolution_time: float = 0.0,
    num_bits: unknown = -1,
    shots_per_bit: unknown = 3,
) -> qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation.IterativePhaseEstimation: ...

def create(
    algorithm_type: str,
    algorithm_name: str | None = None,
    **kwargs,
) -> Union[Algorithm | qdk_chemistry.algorithms.active_space_selector.QdkAutocasActiveSpaceSelector | qdk_chemistry.algorithms.active_space_selector.QdkAutocasEosActiveSpaceSelector | qdk_chemistry.algorithms.active_space_selector.QdkOccupationActiveSpaceSelector | qdk_chemistry.algorithms.active_space_selector.QdkValenceActiveSpaceSelector | qdk_chemistry.algorithms.circuit_executor.qdk.QdkFullStateSimulator | qdk_chemistry.algorithms.circuit_executor.qdk.QdkSparseStateSimulator | qdk_chemistry.algorithms.dynamical_correlation_calculator.DynamicalCorrelationCalculator | qdk_chemistry.algorithms.energy_estimator.qdk.QdkEnergyEstimator | qdk_chemistry.algorithms.hamiltonian_constructor.QdkCholeskyHamiltonianConstructor | qdk_chemistry.algorithms.hamiltonian_constructor.QdkHamiltonianConstructor | qdk_chemistry.algorithms.multi_configuration_calculator.QdkMacisAsci | qdk_chemistry.algorithms.multi_configuration_calculator.QdkMacisCas | qdk_chemistry.algorithms.orbital_localizer.QdkMP2NaturalOrbitalLocalizer | qdk_chemistry.algorithms.orbital_localizer.QdkPipekMezeyLocalizer | qdk_chemistry.algorithms.orbital_localizer.QdkVVHVLocalizer | qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation.IterativePhaseEstimation | qdk_chemistry.algorithms.projected_multi_configuration_calculator.QdkMacisPmc | qdk_chemistry.algorithms.qubit_hamiltonian_solver.DenseMatrixSolver | qdk_chemistry.algorithms.qubit_hamiltonian_solver.SparseMatrixSolver | qdk_chemistry.algorithms.qubit_mapper.qdk_qubit_mapper.QdkQubitMapper | qdk_chemistry.algorithms.scf_solver.QdkScfSolver | qdk_chemistry.algorithms.stability_checker.QdkStabilityChecker | qdk_chemistry.algorithms.state_preparation.sparse_isometry.SparseIsometryGF2XStatePreparation | qdk_chemistry.algorithms.time_evolution.builder.partially_randomized.PartiallyRandomized | qdk_chemistry.algorithms.time_evolution.builder.qdrift.QDrift | qdk_chemistry.algorithms.time_evolution.builder.trotter.Trotter | qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.pauli_sequence_mapper.PauliSequenceMapper]: ...