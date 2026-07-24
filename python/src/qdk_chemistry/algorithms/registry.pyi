"""Type stubs for registry.create() with all algorithm overloads."""

from typing import Literal, overload

import qdk_chemistry.algorithms.registry

from .base import Algorithm

@overload
def create(
    algorithm_type: Literal["active_space_selector"],
    algorithm_name: Literal["pyscf_avas"] | None = None,
    ao_labels: list[str] = [],
    canonicalize: bool = False,
    openshell_option: unknown = 2,
    threshold: float = 0.2,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["active_space_selector"],
    algorithm_name: Literal["qdk_occupation"] | None = None,
    occupation_threshold: float = 0.1,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["active_space_selector"],
    algorithm_name: Literal["qdk_autocas_eos"] | None = None,
    diff_threshold: float = 0.1,
    entropy_threshold: float = 0.14,
    normalize_entropies: bool = True,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["active_space_selector"],
    algorithm_name: Literal["qdk_autocas"] | None = None,
    entropy_threshold: float = 0.14,
    min_plateau_size: unknown = 10,
    normalize_entropies: bool = True,
    num_bins: unknown = 100,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["active_space_selector"],
    algorithm_name: Literal["qdk_valence"] | None = None,
    num_active_electrons: unknown = -1,
    num_active_orbitals: unknown = -1,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_constructor"],
    algorithm_name: Literal["qdk_cholesky"] | None = None,
    cholesky_tolerance: float = 1e-08,
    eri_threshold: float = 1e-12,
    store_ao_cholesky_vectors: bool = False,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_constructor"],
    algorithm_name: Literal["qdk"] | None = None,
    eri_method: str = "direct",
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["orbital_localizer"],
    algorithm_name: Literal["pyscf_multi"] | None = None,
    method: str = "pipek-mezey",
    occupation_threshold: float = 1e-10,
    population_method: str = "mulliken",
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["orbital_localizer"],
    algorithm_name: Literal["qdk_vvhv"] | None = None,
    max_iterations: unknown = 10000,
    minimal_basis: str = "sto-3g",
    small_rotation_tolerance: float = 1e-12,
    tolerance: float = 1e-06,
    weighted_orthogonalization: bool = True,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["orbital_localizer"],
    algorithm_name: Literal["qdk_mp2_natural_orbitals"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["orbital_localizer"],
    algorithm_name: Literal["qdk_pipek_mezey"] | None = None,
    max_iterations: unknown = 10000,
    small_rotation_tolerance: float = 1e-12,
    tolerance: float = 1e-06,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["multi_configuration_calculator"],
    algorithm_name: Literal["macis_asci"] | None = None,
    calculate_mutual_information: bool = False,
    calculate_one_rdm: bool = False,
    calculate_single_orbital_entropies: bool = False,
    calculate_two_orbital_entropies: bool = False,
    calculate_two_rdm: bool = False,
    ci_matel_tol: float = 2.220446049250313e-16,
    ci_residual_tolerance: float = 1e-06,
    constraint_level: unknown = 2,
    core_selection_strategy: str = "percentage",
    core_selection_threshold: float = 0.95,
    grow_factor: float = 8.0,
    grow_with_rot: bool = False,
    growth_backoff_rate: float = 0.5,
    growth_recovery_rate: float = 1.1,
    iterative_solver_dimension_cutoff: unknown = 2000,
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
    search_matel_tol: float = 1e-08,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["multi_configuration_calculator"],
    algorithm_name: Literal["macis_cas"] | None = None,
    calculate_mutual_information: bool = False,
    calculate_one_rdm: bool = False,
    calculate_single_orbital_entropies: bool = False,
    calculate_two_orbital_entropies: bool = False,
    calculate_two_rdm: bool = False,
    ci_matel_tol: float = 2.220446049250313e-16,
    ci_residual_tolerance: float = 1e-06,
    iterative_solver_dimension_cutoff: unknown = 2000,
    max_solver_iterations: unknown = 200,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["multi_configuration_scf"],
    algorithm_name: Literal["pyscf"] | None = None,
    max_cycle_macro: unknown = 50,
    multi_configuration_calculator: AlgorithmRef = AlgorithmRef(
        type="multi_configuration_calculator", name="macis_cas"
    ),
    verbose: unknown = 0,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["projected_multi_configuration_calculator"],
    algorithm_name: Literal["macis_pmc"] | None = None,
    calculate_mutual_information: bool = False,
    calculate_one_rdm: bool = False,
    calculate_single_orbital_entropies: bool = False,
    calculate_two_orbital_entropies: bool = False,
    calculate_two_rdm: bool = False,
    ci_matel_tol: float = 2.220446049250313e-16,
    ci_residual_tolerance: float = 1e-06,
    iterative_solver_dimension_cutoff: unknown = 2000,
    max_solver_iterations: unknown = 200,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["dynamical_correlation_calculator"],
    algorithm_name: Literal["pyscf_coupled_cluster"] | None = None,
    async_io: bool = True,
    compute_bra: bool = False,
    conv_tol: float = 1e-07,
    conv_tol_normt: float = 1e-05,
    diis_space: unknown = 6,
    diis_start_cycle: unknown = 0,
    direct: bool = False,
    incore_complete: bool = True,
    max_cycle: unknown = 50,
    store_amplitudes: bool = False,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["dynamical_correlation_calculator"],
    algorithm_name: Literal["qdk_mp2_calculator"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["scf_solver"],
    algorithm_name: Literal["pyscf"] | None = None,
    convergence_threshold: float = 1e-07,
    max_iterations: unknown = 50,
    method: str = "hf",
    scf_type: str = "auto",
    xc_grid: unknown = 3,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["scf_solver"],
    algorithm_name: Literal["qdk"] | None = None,
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
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["stability_checker"],
    algorithm_name: Literal["pyscf"] | None = None,
    davidson_tolerance: float = 1e-08,
    external: bool = True,
    internal: bool = True,
    method: str = "hf",
    nroots: unknown = 3,
    pyscf_verbose: unknown = 4,
    stability_tolerance: float = -0.0001,
    with_symmetry: bool = False,
    xc_grid: unknown = 3,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["stability_checker"],
    algorithm_name: Literal["qdk"] | None = None,
    davidson_tolerance: float = 1e-08,
    external: bool = False,
    internal: bool = True,
    max_subspace: unknown = 80,
    method: str = "hf",
    stability_tolerance: float = -0.0001,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["energy_estimator"],
    algorithm_name: Literal["qdk"] | None = None,
    circuit_executor: AlgorithmRef = AlgorithmRef(type="circuit_executor", name="qdk_sparse_state_simulator"),
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["circuit_mapper"],
    algorithm_name: Literal["pauli_sequence"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_simulation"],
    algorithm_name: Literal["euler_integrator"] | None = None,
    circuit_executor: AlgorithmRef = AlgorithmRef(type="circuit_executor", name="qdk_sparse_state_simulator"),
    circuit_mapper: AlgorithmRef = AlgorithmRef(type="circuit_mapper", name="pauli_sequence"),
    dt: float = 0.0,
    evolution_builder: AlgorithmRef = AlgorithmRef(type="hamiltonian_unitary_builder", name="trotter"),
    observable_estimator: AlgorithmRef = AlgorithmRef(type="energy_estimator", name="qdk"),
    propagator: AlgorithmRef = AlgorithmRef(type="propagator", name="magnus"),
    total_time: float = 1.0,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["state_prep"],
    algorithm_name: Literal["sparse_isometry_gf2x"] | None = None,
    basis_gates: list[str] = ["x", "y", "z", "cx", "cz", "id", "h", "s", "sdg", "rz"],
    dense_preparation_method: str = "qdk",
    transpile: bool = True,
    transpile_optimization_level: unknown = 0,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["state_prep"],
    algorithm_name: Literal["mps_sparse"] | None = None,
    basis_gates: list[str] = ["x", "y", "z", "cx", "cz", "id", "h", "s", "sdg", "rz"],
    rotation_bits: unknown = 10,
    transpile: bool = True,
    transpile_optimization_level: unknown = 0,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["state_prep"],
    algorithm_name: Literal["dense_pure_state"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["state_prep"],
    algorithm_name: Literal["alias_sampling"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["state_prep"],
    algorithm_name: Literal["qrom"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["state_prep"],
    algorithm_name: Literal["qiskit_regular_isometry"] | None = None,
    basis_gates: list[str] = ["x", "y", "z", "cx", "cz", "id", "h", "s", "sdg", "rz"],
    transpile: bool = True,
    transpile_optimization_level: unknown = 0,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["term_grouper"],
    algorithm_name: Literal["commuting"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["term_grouper"],
    algorithm_name: Literal["qubit_wise_commuting"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["term_grouper"],
    algorithm_name: Literal["identity"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["term_grouper"],
    algorithm_name: Literal["nx_commuting"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["term_grouper"],
    algorithm_name: Literal["nx_qubit_wise_commuting"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qubit_mapper"],
    algorithm_name: Literal["qdk"] | None = None,
    integral_threshold: float = 1e-12,
    threshold: float = 1e-12,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qubit_mapper"],
    algorithm_name: Literal["qiskit"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qubit_mapper"],
    algorithm_name: Literal["openfermion"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qubit_hamiltonian_solver"],
    algorithm_name: Literal["qdk_dense_matrix_solver"] | None = None,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qubit_hamiltonian_solver"],
    algorithm_name: Literal["qdk_sparse_matrix_solver"] | None = None,
    max_m: unknown = 20,
    tol: float = 1e-08,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_unitary_builder"],
    algorithm_name: Literal["trotter"] | None = None,
    error_bound: str = "commutator",
    num_divisions: unknown = 0,
    order: unknown = 1,
    power: unknown = 1,
    power_strategy: str = "repeat",
    target_accuracy: float = 0.0,
    time: float = 0.0,
    weight_threshold: float = 1e-12,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_unitary_builder"],
    algorithm_name: Literal["zassenhaus"] | None = None,
    error_bound: str = "commutator",
    num_divisions: unknown = 0,
    order: unknown = 2,
    power: unknown = 1,
    power_strategy: str = "repeat",
    target_accuracy: float = 0.0,
    term_grouper: AlgorithmRef = AlgorithmRef(type="term_grouper", name="commuting"),
    time: float = 0.0,
    weight_threshold: float = 1e-12,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_unitary_builder"],
    algorithm_name: Literal["qdrift"] | None = None,
    commutation_type: str = "general",
    error_bound: str = "campbell",
    merge_duplicate_terms: bool = True,
    num_samples: unknown = 100,
    power: unknown = 1,
    power_strategy: str = "repeat",
    seed: unknown = -1,
    target_accuracy: float = 0.0,
    time: float = 0.0,
    weight_threshold: float = 1e-12,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_unitary_builder"],
    algorithm_name: Literal["partially_randomized"] | None = None,
    commutation_type: str = "general",
    merge_duplicate_terms: bool = True,
    num_random_samples: unknown = 100,
    power: unknown = 1,
    power_strategy: str = "repeat",
    seed: unknown = -1,
    time: float = 0.0,
    tolerance: float = 1e-12,
    trotter_order: unknown = 2,
    weight_threshold: float = -1.0,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_unitary_builder"],
    algorithm_name: Literal["lcu"] | None = None,
    power: unknown = 1,
    quantum_walk: bool = False,
    tolerance: float = 9.999999960041972e-13,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hamiltonian_unitary_builder"],
    algorithm_name: Literal["sossa"] | None = None,
    power: unknown = 1,
    tolerance: float = 9.999999960041972e-13,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["controlled_circuit_mapper"],
    algorithm_name: Literal["prepare_select_prepare"] | None = None,
    control_indices: unknown = [0],
    prepare: AlgorithmRef = AlgorithmRef(type="state_prep", name="dense_pure_state"),
    target_indices: unknown = [],
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["controlled_circuit_mapper"],
    algorithm_name: Literal["pauli_sequence"] | None = None,
    control_indices: unknown = [0],
    target_indices: unknown = [],
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["controlled_circuit_mapper"],
    algorithm_name: Literal["sossa"] | None = None,
    coefficient_bit_precision: unknown = 10,
    control_indices: unknown = [0],
    inner_prepare_algorithm: str = "controlled_alias_sampling",
    outer_prepare: AlgorithmRef = AlgorithmRef(type="state_prep", name="alias_sampling"),
    rotation_bit_precision: unknown = 10,
    select_algorithm: str = "qrom_phase_gradient",
    target_indices: unknown = [],
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["circuit_executor"],
    algorithm_name: Literal["qdk_full_state_simulator"] | None = None,
    seed: unknown = 42,
    type: str = "cpu",
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["circuit_executor"],
    algorithm_name: Literal["qdk_sparse_state_simulator"] | None = None,
    seed: unknown = 42,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["circuit_executor"],
    algorithm_name: Literal["qiskit_aer_simulator"] | None = None,
    device_backend_name: str = "",
    method: str = "statevector",
    post_transpilation_passes: list[str] = [],
    pre_transpilation_passes: list[str] = [],
    seed: unknown = 42,
    transpile_optimization_level: unknown = 0,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qpe_circuit_builder"],
    algorithm_name: Literal["qdk_iterative"] | None = None,
    controlled_circuit_mapper: AlgorithmRef = AlgorithmRef(type="controlled_circuit_mapper", name="pauli_sequence"),
    num_bits: unknown = -1,
    num_iteration: unknown = -1,
    phase_correction: float = 0.0,
    unitary_builder: AlgorithmRef = AlgorithmRef(type="hamiltonian_unitary_builder", name="trotter"),
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qpe_circuit_builder"],
    algorithm_name: Literal["qdk_standard"] | None = None,
    controlled_circuit_mapper: AlgorithmRef = AlgorithmRef(type="controlled_circuit_mapper", name="pauli_sequence"),
    num_bits: unknown = -1,
    unitary_builder: AlgorithmRef = AlgorithmRef(type="hamiltonian_unitary_builder", name="trotter"),
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qpe_circuit_builder"],
    algorithm_name: Literal["qiskit_iterative"] | None = None,
    controlled_circuit_mapper: AlgorithmRef = AlgorithmRef(type="controlled_circuit_mapper", name="pauli_sequence"),
    num_bits: unknown = -1,
    num_iteration: unknown = -1,
    phase_correction: float = 0.0,
    unitary_builder: AlgorithmRef = AlgorithmRef(type="hamiltonian_unitary_builder", name="trotter"),
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["qpe_circuit_builder"],
    algorithm_name: Literal["qiskit_standard"] | None = None,
    controlled_circuit_mapper: AlgorithmRef = AlgorithmRef(type="controlled_circuit_mapper", name="pauli_sequence"),
    num_bits: unknown = -1,
    qft_do_swaps: bool = True,
    unitary_builder: AlgorithmRef = AlgorithmRef(type="hamiltonian_unitary_builder", name="trotter"),
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["phase_estimation"],
    algorithm_name: Literal["qdk_iterative"] | None = None,
    circuit_executor: AlgorithmRef = AlgorithmRef(type="circuit_executor", name="qdk_sparse_state_simulator"),
    qpe_circuit_builder: AlgorithmRef = AlgorithmRef(type="qpe_circuit_builder", name="qdk_iterative"),
    shots_per_bit: unknown = 3,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["phase_estimation"],
    algorithm_name: Literal["qdk_standard"] | None = None,
    circuit_executor: AlgorithmRef = AlgorithmRef(type="circuit_executor", name="qdk_sparse_state_simulator"),
    qpe_circuit_builder: AlgorithmRef = AlgorithmRef(type="qpe_circuit_builder", name="qdk_iterative"),
    shots: unknown = 3,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hadamard_test"],
    algorithm_name: Literal["qdk"] | None = None,
    circuit_executor: AlgorithmRef = AlgorithmRef(type="circuit_executor", name="qdk_full_state_simulator"),
    controlled_circuit_mapper: AlgorithmRef = AlgorithmRef(type="controlled_circuit_mapper", name="pauli_sequence"),
    test_basis: str = "X",
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["hadamard_test_circuit_builder"],
    algorithm_name: Literal["qdk"] | None = None,
    controlled_circuit_mapper: AlgorithmRef = AlgorithmRef(type="controlled_circuit_mapper", name="pauli_sequence"),
    test_basis: str = "X",
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
@overload
def create(
    algorithm_type: Literal["propagator"],
    algorithm_name: Literal["magnus"] | None = None,
    order: unknown = 1,
) -> qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
def create(
    algorithm_type: str,
    algorithm_name: str | None = None,
    **kwargs,
) -> Algorithm | qdk_chemistry.algorithms.registry._AlgorithmWrapper: ...
