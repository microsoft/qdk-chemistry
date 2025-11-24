"""Type stubs for registry.create() with all algorithm overloads."""

from typing import Literal, overload, Union
from .base import Algorithm

import qdk_chemistry.algorithms
import qdk_chemistry.algorithms.energy_estimator.qsharp
import qdk_chemistry.algorithms.state_preparation.sparse_isometry
import qdk_chemistry.plugins.qiskit.energy_estimator
import qdk_chemistry.plugins.qiskit.qubit_mapper
import qdk_chemistry.plugins.qiskit.regular_isometry

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_occupation'] | None = None,
    occupation_threshold: float = 0.1,
) -> qdk_chemistry.algorithms.ActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_autocas_eos'] | None = None,
    diff_threshold: float = 0.1,
    entropy_threshold: float = 0.14,
    normalize_entropies: bool = True,
) -> qdk_chemistry.algorithms.ActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_autocas'] | None = None,
    entropy_threshold: float = 0.14,
    min_plateau_size: unknown = 10,
    normalize_entropies: bool = True,
    num_bins: unknown = 100,
) -> qdk_chemistry.algorithms.ActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_valence'] | None = None,
    num_active_electrons: int = -1,
    num_active_orbitals: int = -1,
) -> qdk_chemistry.algorithms.ActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['hamiltonian_constructor'],
    algorithm_name: Literal['qdk'] | None = None,
    eri_method: str = direct,
) -> qdk_chemistry.algorithms.HamiltonianConstructor: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_vvhv'] | None = None,
    max_iterations: unknown = 10000,
    minimal_basis: str = sto-3g,
    small_rotation_tolerance: float = 1e-12,
    tolerance: float = 1e-06,
    weighted_orthogonalization: bool = True,
) -> qdk_chemistry.algorithms.Localizer: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_mp2_natural_orbitals'] | None = None,
) -> qdk_chemistry.algorithms.Localizer: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_pipek_mezey'] | None = None,
    max_iterations: unknown = 10000,
    small_rotation_tolerance: float = 1e-12,
    tolerance: float = 1e-06,
) -> qdk_chemistry.algorithms.Localizer: ...

@overload
def create(
    algorithm_type: Literal['multi_configuration_calculator'],
    algorithm_name: Literal['macis_asci'] | None = None,
    calculate_one_rdm: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    constraint_level: int = 2,
    davidson_iterations: unknown = 200,
    grow_factor: unknown = 8,
    grow_with_rot: bool = False,
    h_el_tol: float = 1e-08,
    just_singles: bool = False,
    max_refine_iter: unknown = 6,
    ncdets_max: unknown = 100,
    ntdets_max: unknown = 100000,
    ntdets_min: unknown = 100,
    num_roots: unknown = 1,
    nxtval_bcount_inc: unknown = 10,
    nxtval_bcount_thresh: unknown = 1000,
    pair_size_max: unknown = 500000000,
    pt2_bigcon_thresh: unknown = 250,
    pt2_constraint_refine_force: int = 0,
    pt2_max_constraint_level: int = 5,
    pt2_min_constraint_level: int = 0,
    pt2_precompute_eps: bool = False,
    pt2_precompute_idx: bool = False,
    pt2_print_progress: bool = False,
    pt2_prune: bool = False,
    pt2_reserve_count: unknown = 70000000,
    pt2_tol: float = 1e-16,
    refine_energy_tol: float = 1e-06,
    rot_size_start: unknown = 1000,
    rv_prune_tol: float = 1e-08,
) -> qdk_chemistry.algorithms.MultiConfigurationCalculator: ...

@overload
def create(
    algorithm_type: Literal['multi_configuration_calculator'],
    algorithm_name: Literal['macis_cas'] | None = None,
    calculate_one_rdm: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    davidson_iterations: unknown = 200,
    num_roots: unknown = 1,
) -> qdk_chemistry.algorithms.MultiConfigurationCalculator: ...

@overload
def create(
    algorithm_type: Literal['projected_multi_configuration_calculator'],
    algorithm_name: Literal['macis_pmc'] | None = None,
    H_thresh: float = 1e-16,
    calculate_one_rdm: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    davidson_iterations: unknown = 200,
    davidson_max_m: unknown = 200,
    davidson_res_tol: float = 1e-08,
    h_el_tol: float = 1e-08,
    iterative_solver_dimension_cutoff: unknown = 100,
    num_roots: unknown = 1,
) -> qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator: ...

@overload
def create(
    algorithm_type: Literal['scf_solver'],
    algorithm_name: Literal['qdk'] | None = None,
    basis_set: str = def2-svp,
    max_iterations: int = 50,
    method: str = hf,
    tolerance: float = 1e-08,
) -> qdk_chemistry.algorithms.ScfSolver: ...

@overload
def create(
    algorithm_type: Literal['energy_estimator'],
    algorithm_name: Literal['qdk_base_simulator'] | None = None,
) -> qdk_chemistry.algorithms.energy_estimator.qsharp.QDKEnergyEstimator: ...

@overload
def create(
    algorithm_type: Literal['energy_estimator'],
    algorithm_name: Literal['qiskit_aer_simulator'] | None = None,
) -> qdk_chemistry.plugins.qiskit.energy_estimator.QiskitEnergyEstimator: ...

@overload
def create(
    algorithm_type: Literal['state_prep'],
    algorithm_name: Literal['sparse_isometry_gf2x'] | None = None,
    basis_gates: list[str] = ['x', 'y', 'z', 'cx', 'cz', 'id', 'h', 's', 'sdg', 'rz'],
    transpile: bool = True,
    transpile_optimization_level: int = 0,
) -> qdk_chemistry.algorithms.state_preparation.sparse_isometry.SparseIsometryGF2XStatePreparation: ...

@overload
def create(
    algorithm_type: Literal['state_prep'],
    algorithm_name: Literal['regular_isometry'] | None = None,
    basis_gates: list[str] = ['x', 'y', 'z', 'cx', 'cz', 'id', 'h', 's', 'sdg', 'rz'],
    transpile: bool = True,
    transpile_optimization_level: int = 0,
) -> qdk_chemistry.plugins.qiskit.regular_isometry.RegularIsometryStatePreparation: ...

@overload
def create(
    algorithm_type: Literal['qubit_mapper'],
    algorithm_name: Literal['qiskit'] | None = None,
    encoding: str = jordan-wigner,
) -> qdk_chemistry.plugins.qiskit.qubit_mapper.QiskitQubitMapper: ...

def create(
    algorithm_type: str,
    algorithm_name: str | None = None,
    **kwargs,
) -> Union[Algorithm | qdk_chemistry.algorithms.ActiveSpaceSelector | qdk_chemistry.algorithms.HamiltonianConstructor | qdk_chemistry.algorithms.Localizer | qdk_chemistry.algorithms.MultiConfigurationCalculator | qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator | qdk_chemistry.algorithms.ScfSolver | qdk_chemistry.algorithms.energy_estimator.qsharp.QDKEnergyEstimator | qdk_chemistry.algorithms.state_preparation.sparse_isometry.SparseIsometryGF2XStatePreparation | qdk_chemistry.plugins.qiskit.energy_estimator.QiskitEnergyEstimator | qdk_chemistry.plugins.qiskit.qubit_mapper.QiskitQubitMapper | qdk_chemistry.plugins.qiskit.regular_isometry.RegularIsometryStatePreparation]: ...
