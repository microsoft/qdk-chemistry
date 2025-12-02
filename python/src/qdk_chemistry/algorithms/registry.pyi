"""Type stubs for registry.create() with all algorithm overloads."""

from typing import Literal, overload, Union
from .base import Algorithm

import qdk.chemistry.algorithms
import qdk.chemistry.algorithms.energy_estimator.qsharp
import qdk.chemistry.plugins.qiskit.energy_estimator
import qdk.chemistry.plugins.qiskit.qubit_mapper

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_occupation'] | None = None,
    occupation_threshold: float = 0.1,
) -> qdk.chemistry.algorithms.ActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_autocas_eos'] | None = None,
    diff_threshold: float = 0.1,
    entropy_threshold: float = 0.14,
    normalize_entropies: bool = True,
) -> qdk.chemistry.algorithms.ActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_autocas'] | None = None,
    entropy_threshold: float = 0.14,
    min_plateau_size: int = 10,
    normalize_entropies: bool = True,
    num_bins: int = 100,
) -> qdk.chemistry.algorithms.ActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['active_space_selector'],
    algorithm_name: Literal['qdk_valence'] | None = None,
    num_active_electrons: int = -1,
    num_active_orbitals: int = -1,
) -> qdk.chemistry.algorithms.ActiveSpaceSelector: ...

@overload
def create(
    algorithm_type: Literal['hamiltonian_constructor'],
    algorithm_name: Literal['qdk'] | None = None,
    eri_method: str = direct,
) -> qdk.chemistry.algorithms.HamiltonianConstructor: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_vvhv'] | None = None,
    max_iterations: int = 10000,
    minimal_basis: str = sto-3g,
    small_rotation_tolerance: float = 1e-12,
    tolerance: float = 1e-06,
    weighted_orthogonalization: bool = True,
) -> qdk.chemistry.algorithms.Localizer: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_mp2_natural_orbitals'] | None = None,
) -> qdk.chemistry.algorithms.Localizer: ...

@overload
def create(
    algorithm_type: Literal['orbital_localizer'],
    algorithm_name: Literal['qdk_pipek_mezey'] | None = None,
    max_iterations: int = 10000,
    small_rotation_tolerance: float = 1e-12,
    tolerance: float = 1e-06,
) -> qdk.chemistry.algorithms.Localizer: ...

@overload
def create(
    algorithm_type: Literal['multi_configuration_calculator'],
    algorithm_name: Literal['macis_asci'] | None = None,
    calculate_one_rdm: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    constraint_level: int = 2,
    davidson_iterations: int = 200,
    grow_factor: int = 8,
    grow_with_rot: bool = False,
    h_el_tol: float = 1e-08,
    just_singles: bool = False,
    max_refine_iter: int = 6,
    ncdets_max: int = 100,
    ntdets_max: int = 100000,
    ntdets_min: int = 100,
    num_roots: int = 1,
    nxtval_bcount_inc: int = 10,
    nxtval_bcount_thresh: int = 1000,
    pair_size_max: int = 500000000,
    pt2_bigcon_thresh: int = 250,
    pt2_constraint_refine_force: int = 0,
    pt2_max_constraint_level: int = 5,
    pt2_min_constraint_level: int = 0,
    pt2_precompute_eps: bool = False,
    pt2_precompute_idx: bool = False,
    pt2_print_progress: bool = False,
    pt2_prune: bool = False,
    pt2_reserve_count: int = 70000000,
    pt2_tol: float = 1e-16,
    refine_energy_tol: float = 1e-06,
    rot_size_start: int = 1000,
    rv_prune_tol: float = 1e-08,
) -> qdk.chemistry.algorithms.MultiConfigurationCalculator: ...

@overload
def create(
    algorithm_type: Literal['multi_configuration_calculator'],
    algorithm_name: Literal['macis_cas'] | None = None,
    calculate_one_rdm: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    davidson_iterations: int = 200,
    num_roots: int = 1,
) -> qdk.chemistry.algorithms.MultiConfigurationCalculator: ...

@overload
def create(
    algorithm_type: Literal['projected_multi_configuration_calculator'],
    algorithm_name: Literal['macis_pmc'] | None = None,
    calculate_one_rdm: bool = False,
    calculate_two_rdm: bool = False,
    ci_residual_tolerance: float = 1e-06,
    davidson_iterations: int = 200,
    h_el_tol: float = 1e-08,
    num_roots: int = 1,
) -> qdk.chemistry.algorithms.ProjectedMultiConfigurationCalculator: ...

@overload
def create(
    algorithm_type: Literal['scf_solver'],
    algorithm_name: Literal['qdk'] | None = None,
    basis_set: str = def2-svp,
    max_iterations: int = 50,
    method: str = hf,
    tolerance: float = 1e-08,
) -> qdk.chemistry.algorithms.ScfSolver: ...

@overload
def create(
    algorithm_type: Literal['energy_estimator'],
    algorithm_name: Literal['qdk_base_simulator'] | None = None,
) -> qdk.chemistry.algorithms.energy_estimator.qsharp.QDKEnergyEstimator: ...

@overload
def create(
    algorithm_type: Literal['energy_estimator'],
    algorithm_name: Literal['qiskit_aer_simulator'] | None = None,
) -> qdk.chemistry.plugins.qiskit.energy_estimator.QiskitEnergyEstimator: ...

@overload
def create(
    algorithm_type: Literal['qubit_mapper'],
    algorithm_name: Literal['qiskit'] | None = None,
    encoding: str = jordan-wigner,
) -> qdk.chemistry.plugins.qiskit.qubit_mapper.QiskitQubitMapper: ...

def create(
    algorithm_type: str,
    algorithm_name: str | None = None,
    **kwargs,
) -> Union[Algorithm | qdk.chemistry.algorithms.ActiveSpaceSelector | qdk.chemistry.algorithms.HamiltonianConstructor | qdk.chemistry.algorithms.Localizer | qdk.chemistry.algorithms.MultiConfigurationCalculator | qdk.chemistry.algorithms.ProjectedMultiConfigurationCalculator | qdk.chemistry.algorithms.ScfSolver | qdk.chemistry.algorithms.energy_estimator.qsharp.QDKEnergyEstimator | qdk.chemistry.plugins.qiskit.energy_estimator.QiskitEnergyEstimator | qdk.chemistry.plugins.qiskit.qubit_mapper.QiskitQubitMapper]: ...
