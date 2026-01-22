"""QDK/Chemistry-algorithms registry.

This module contains a registration mechanism for Hamiltonian constructor, SCF solver, localizer,
active space selector, coupled cluster calculator, (projected) multi configuration calculator,
and MultiConfigurationScf. The user should be able to use the registration mechanism to easily
add and remove custom algorithms at runtime.

Algorithm Lifecycle Management
------------------------------

The registry system handles the lifecycle of algorithm instances to prevent memory issues. When
Python-implemented algorithms are registered, they need special handling during Python interpreter
shutdown to avoid "double-free" errors that can occur when C++ static deinitialization runs after
Python's garbage collection.

This module automatically registers cleanup handlers (via `atexit`) that unregister all custom
algorithms before Python shuts down. Users should never need to call the cleanup functions directly.

Important Notes
---------------

* All registration functions in this module provide automatic cleanup.
* Custom Python algorithms are automatically unregistered during interpreter shutdown.
* The cleanup functions are **not** intended to be called by users.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory

__all__ = [
    "available",
    "create",
    "inspect_settings",
    "print_settings",
    "register",
    "register_factory",
    "show_default",
    "unregister",
    "unregister_factory",
]

# Universal cleanup solution for all algorithms
__cleanup_registered: bool = False

__factories: list[AlgorithmFactory] = []


def create(algorithm_type: str, algorithm_name: str | None = None, **kwargs) -> Algorithm:
    """Create an algorithm instance by type and name.

    This function creates an algorithm instance from the registry using the specified
    algorithm type and name. If no name is provided, the default implementation for that
    type is created.

    Available algorithm types depend on the installed plugins and any user-registered
    algorithms. Use :func:`available` to inspect what is currently loaded before calling
    :func:`create`.

    Args:
        algorithm_type (str): The type of algorithm to create.

            (e.g., "scf_solver", "active_space_selector", "coupled_cluster_calculator").

        algorithm_name (str | None): The specific name of the algorithm implementation to create.

            If None or empty string, creates the default algorithm for that type.

        kwargs: Optional keyword arguments (passed via ``**kwargs``).

            These configure the algorithm's settings. These are forwarded directly to the algorithm's settings
            via `settings().update()`. Available settings depend on the specific algorithm
            type and implementation and can be looked up with inspect_settings() or print_settings().

    Returns:
        Algorithm: The created algorithm instance.

    Raises:
        KeyError: If the specified algorithm type is not registered in the system.

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> # Create the default SCF solver
        >>> scf = registry.create("scf_solver")
        >>> # Create a specific SCF solver by name
        >>> pyscf_solver = registry.create("scf_solver", "pyscf")
        >>> # Create an SCF solver with custom settings
        >>> scf = registry.create("scf_solver", "pyscf", max_iterations=100, convergence_threshold=1e-8)
        >>> # Create an MP2 calculator
        >>> mp2_calc = registry.create("dynamical_correlation_calculator", "qdk_mp2_calculator")
        >>> # Create the default reference-derived calculator (MP2)
        >>> default_calc = registry.create("dynamical_correlation_calculator")

    """
    if algorithm_name is None:
        algorithm_name = ""
    for factory in __factories:
        if factory.algorithm_type_name() == algorithm_type:
            try:
                instance = factory.create(algorithm_name)
                instance.settings().update(kwargs or {})
                return instance
            except (KeyError, RuntimeError, ValueError) as e:
                available_algorithms = factory.available()
                if not available_algorithms:
                    raise KeyError(
                        f"No algorithms available for type '{algorithm_type}'. "
                        "This may indicate that no plugins providing this algorithm type are loaded or registered."
                    ) from e
                raise KeyError(
                    f"Algorithm '{algorithm_name}' not found for type '{algorithm_type}'. "
                    f"Available algorithms for this type: {', '.join(available_algorithms)}. "
                    "Available algorithms are influenced by loaded plugins and registered custom algorithms. "
                    "Please ensure the relevant plugins are loaded or custom algorithms are registered "
                    "ahead of calling create()."
                ) from e
    available_types = [factory.algorithm_type_name() for factory in __factories]
    raise KeyError(
        f"Algorithm type '{algorithm_type}' is not registered. Available algorithm types: {', '.join(available_types)}."
        "Available algorithm types are influenced by loaded plugins and registered custom algorithms. "
        "Please ensure the relevant plugins are loaded or custom algorithms are registered ahead of calling create()."
    )


def print_settings(algorithm_type: str, algorithm_name: str, characters: int = 120) -> None:
    """Print the settings table for a specific algorithm.

    This function retrieves and prints the settings schema for a given algorithm type and name
    as a formatted table. The table displays configurable parameters including their names,
    current values, allowed values/ranges, and descriptions.

    Args:
        algorithm_type (str): The type of algorithm.

            (e.g., "scf_solver", "active_space_selector", "coupled_cluster_calculator").

        algorithm_name (str): The specific name of the algorithm implementation.

        characters (int): Maximum width of the table in characters (default: 120).

    Raises:
        KeyError: If the specified algorithm type is not registered in the system.

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> # Print settings table for the PySCF SCF solver
        >>> registry.print_settings("scf_solver", "pyscf")
        ------------------------------------------------------------------------------------------------------------------------
        Key                  | Value           | Allowed              | Description
        ------------------------------------------------------------------------------------------------------------------------
        charge               | 0               | -                    | Total molecular charge
        convergence_thresh...| 1.00e-06        | 1.00e-12 <= x        | Energy convergence threshold
                             |                 | x <= 1.00e-02        |
        force_restricted     | false           | -                    | Force restricted calculation
        max_iterations       | 50              | 1 <= x <= 1000       | Maximum SCF iterations
        method               | "hf"            | ["hf", "dft"]        | SCF method to use
        spin_multiplicity    | 1               | 1 <= x <= 10         | Spin multiplicity (2S+1)
        ------------------------------------------------------------------------------------------------------------------------
        >>> # Print with custom width
        >>> registry.print_settings("scf_solver", "pyscf", characters=100)

    """
    for factory in __factories:
        if factory.algorithm_type_name() == algorithm_type:
            instance = factory.create(algorithm_name)
            print(instance.settings().as_table(characters))
            return
    available_types = [factory.algorithm_type_name() for factory in __factories]
    raise KeyError(
        f"Algorithm type '{algorithm_type}' is not registered. "
        f"Available algorithm types: {', '.join(available_types)}. "
        "Available algorithm types are influenced by loaded plugins and registered custom algorithms. "
        "Please ensure the relevant plugins are loaded or custom algorithms are registered ahead of calling create()."
    )


def inspect_settings(algorithm_type: str, algorithm_name: str) -> list[tuple[str, str, Any, str | None, Any | None]]:
    """Inspect the settings schema for a specific algorithm.

    This function retrieves the settings schema for a given algorithm type and name.
    The settings schema provides information about configurable parameters for the
    algorithm, including their names, expected Python types, default values, descriptions,
    and allowed values/ranges.

    Args:
        algorithm_type (str): The type of algorithm.

            (e.g., "scf_solver", "active_space_selector", "coupled_cluster_calculator").

        algorithm_name (str): The specific name of the algorithm implementation.

    Returns:
        list[tuple[str, str, Any, str | None, Any | None]]: A list of tuples where each tuple contains:
            - Setting name (str): The name/key of the setting
            - Expected Python type (str): The Python type expected for this setting
              (e.g., int, float, str, bool, list[int], list[float])
            - Default value (Any): The default value for this setting
            - Description (str | None): Human-readable description of the setting, or None if not available
            - Limits (Any | None): Allowed values or range, or None if not constrained.
              For numeric types: tuple of (min, max). For strings/lists: list of allowed values.

    Raises:
        KeyError: If the specified algorithm type is not registered in the system.

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> # Show settings for the PySCF SCF solver
        >>> settings_info = registry.inspect_settings("scf_solver", "pyscf")
        >>> for name, python_type, default, description, limits in settings_info:
        ...     limit_str = f" (allowed: {limits})" if limits else ""
        ...     desc_str = f"  # {description}" if description else ""
        ...     print(f"{name}: {python_type} = {default}{limit_str}{desc_str}")
        method: str = hf (allowed: ['hf', 'dft'])  # SCF method to use
        basis_set: str = def2-svp  # Basis set for the calculation
        charge: int = 0  # Total molecular charge
        spin_multiplicity: int = 1 (allowed: (1, 10))  # Spin multiplicity (2S+1)
        tolerance: float = 1e-06 (allowed: (1e-12, 0.01))  # Convergence threshold
        max_iterations: int = 50 (allowed: (1, 1000))  # Maximum SCF iterations
        force_restricted: bool = False  # Force restricted calculation

    """
    for factory in __factories:
        if factory.algorithm_type_name() == algorithm_type:
            instance = factory.create(algorithm_name)
            settings = instance.settings().to_dict()
            result = []
            for name, default in settings.items():
                python_type = instance.settings().get_expected_python_type(name)
                description = (
                    instance.settings().get_description(name) if instance.settings().has_description(name) else None
                )
                limits = instance.settings().get_limits(name) if instance.settings().has_limits(name) else None
                result.append((name, python_type, default, description, limits))
            return result
    available_types = [factory.algorithm_type_name() for factory in __factories]
    raise KeyError(
        f"Algorithm type '{algorithm_type}' is not registered. Available algorithm types: {', '.join(available_types)}"
        "Available algorithm types are influenced by loaded plugins and registered custom algorithms. "
        "Please ensure the relevant plugins are loaded or custom algorithms are registered ahead of calling create()."
    )


def register(generator: Callable[[], Algorithm]) -> None:
    """Register a custom algorithm implementation.

    This function registers a custom algorithm implementation (typically written in Python)
    into the registry system. The generator function should return a new instance of the
    algorithm each time it's called. The algorithm's type is automatically detected from
    the returned instance.

    Args:
        generator (Callable[[], Algorithm]): A callable that returns a new instance.

            Need to return an instance of the custom algorithm.
            This will be called each time the algorithm is created
            from the factory.

    Raises:
        KeyError: If the algorithm's type is not a recognized algorithm type in the system.

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> from qdk_chemistry.algorithms import ScfSolver
        >>> class MyCustomScf(ScfSolver):
        ...     def name(self):
        ...         return "my_custom_scf"
        ...     def _run_impl(self, structure, charge, spin_multiplicity):
        ...         # Custom implementation
        ...         pass
        >>> # Register the custom algorithm
        >>> registry.register(lambda: MyCustomScf())
        >>> # Now it can be created from the registry
        >>> scf = registry.create("scf_solver", "my_custom_scf")

    """
    _ensure_cleanup_registered()
    tmp = generator()
    algorithm_type = tmp.type_name()
    for factory in __factories:
        if factory.algorithm_type_name() == algorithm_type:
            factory.register_instance(generator)
            return
    available_types = [factory.algorithm_type_name() for factory in __factories]
    raise KeyError(
        f"Algorithm type '{algorithm_type}' is not registered. Available algorithm types: {', '.join(available_types)}"
    )


def available(algorithm_type: str | None = None) -> dict[str, list[str]] | list[str]:
    """List all available algorithms by type.

    This function returns information about available algorithms in the registry.
    When called without arguments, it returns a dictionary mapping all algorithm
    types to their available implementations. When called with a specific algorithm
    type, it returns only the list of available algorithms for that type.

    Args:
        algorithm_type (str | None): If provided, only list algorithms of this type.

            If None, list all algorithms across all types.

    Returns:
        dict[str, list[str]] | list[str]: Information on available algorithms.

            When algorithm_type is None, returns a dictionary
            where keys are algorithm type names and values are lists of available algorithm
            names for that type. When algorithm_type is specified, returns a list of available
            algorithm names for that specific type (empty list if type not found or no
            algorithms are available).

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> # List all available algorithms across all types
        >>> all_algorithms = registry.available()
        >>> print(all_algorithms)
        {'scf_solver': ['pyscf', 'qdk'], 'active_space_selector': ['pyscf_avas', 'qdk_occupation', ...], ...}
        >>> # List only SCF solvers
        >>> scf_solvers = registry.available("scf_solver")
        >>> print(scf_solvers)
        ['pyscf', 'qdk']
        >>> # Check what active space selectors are available
        >>> selectors = registry.available("active_space_selector")
        >>> print(selectors)
        ['pyscf_avas', 'qdk_occupation', 'qdk_autocas_eos', 'qdk_autocas', 'qdk_valence']

    """
    if algorithm_type is None:
        result: dict[str, list[str]] = {}
        for factory in __factories:
            result[factory.algorithm_type_name()] = factory.available()
        return result
    for factory in __factories:
        if factory.algorithm_type_name() == algorithm_type:
            return factory.available()
    return []


def show_default(algorithm_type: str | None = None) -> dict[str, str] | str:
    """List the default algorithm by type.

    This function returns information about the default algorithms configured
    for each algorithm type. When called without arguments, it returns a dictionary
    mapping all algorithm types to their default algorithm names. When called with
    a specific algorithm type, it returns only the default algorithm name for that type.

    Args:
        algorithm_type (str | None): If provided, only return the default algorithm
            for this type. If None, return default algorithms for all types.

    Returns:
        dict[str, str] | str: When algorithm_type is None, returns a dictionary where
            keys are algorithm type names and values are the default algorithm names for
            each type. When algorithm_type is specified, returns the default algorithm
            name for that specific type (empty string if type not found).

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> # List the default algorithms across all types
        >>> default_algorithms = registry.show_default()
        >>> print(default_algorithms)
        {'scf_solver': 'qdk', 'active_space_selector': 'qdk_autocas_eos', ...}
        >>> # Get the default SCF solver
        >>> default_scf = registry.show_default("scf_solver")
        >>> print(default_scf)
        'qdk'

    """
    if algorithm_type is None:
        result: dict[str, str] = {}
        for factory in __factories:
            result[factory.algorithm_type_name()] = factory.default_algorithm_name()
        return result
    for factory in __factories:
        if factory.algorithm_type_name() == algorithm_type:
            return factory.default_algorithm_name()
    return ""


def unregister(algorithm_type: str, algorithm_name: str) -> None:
    """Unregister a custom algorithm implementation.

    This function removes a previously registered algorithm from the registry.

    Args:
        algorithm_type (str): The type of algorithm to unregister.

            (e.g., "scf_solver", "active_space_selector").

        algorithm_name (str): The name of the specific algorithm implementation to unregister.

    Raises:
        KeyError: If the specified algorithm type is not registered in the system.

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> # Assuming you previously registered a custom algorithm
        >>> registry.unregister("scf_solver", "my_custom_scf")

    """
    for factory in __factories:
        if factory.algorithm_type_name() == algorithm_type:
            factory.unregister_instance(algorithm_name)
            return
    available_types = [factory.algorithm_type_name() for factory in __factories]
    raise KeyError(
        f"Algorithm type '{algorithm_type}' is not registered. Available types: {', '.join(available_types)}"
    )


def register_factory(factory: AlgorithmFactory) -> None:
    """Register a new algorithm factory.

    This function allows adding new algorithm factories to the registry,
    thus adding entire algorithm types.

    Args:
        factory (AlgorithmFactory): The factory instance to register.

    Raises:
        ValueError: If a factory with the same algorithm type name is already registered.

    """
    algorithm_type = factory.algorithm_type_name()
    for existing_factory in __factories:
        if existing_factory.algorithm_type_name() == algorithm_type:
            raise ValueError(f"Factory for algorithm type '{algorithm_type}' is already registered.")
    __factories.append(factory)


def unregister_factory(algorithm_type: str) -> None:
    """Unregister an existing algorithm factory.

    This function allows removing algorithm factories from the registry,
    thus removing entire algorithm types.

    Args:
        algorithm_type (str): The type name of the factory to unregister.

    Raises:
        KeyError: If no factory with the specified algorithm type name is found.

    """
    for existing_factory in __factories:
        if existing_factory.algorithm_type_name() == algorithm_type:
            __factories.remove(existing_factory)
            return
    raise KeyError(f"Factory for algorithm type '{algorithm_type}' is not registered.")


def _register_cpp_factories():
    """Register all built-in C++ algorithm factories.

    This internal initialization function registers all the C++-implemented
    algorithm factories provided by the core library. This includes factories
    for SCF solvers, active space selectors, coupled cluster calculators,
    localizers, multi-configuration calculators, and other core algorithm types.

    This function is automatically called during module import and should not
    be called by users.
    """
    from qdk_chemistry._core._algorithms import (  # noqa: PLC0415
        ActiveSpaceSelectorFactory,
        DynamicalCorrelationCalculatorFactory,
        HamiltonianConstructorFactory,
        LocalizerFactory,
        MultiConfigurationCalculatorFactory,
        MultiConfigurationScfFactory,
        ProjectedMultiConfigurationCalculatorFactory,
        ScfSolverFactory,
        StabilityCheckerFactory,
    )

    register_factory(ActiveSpaceSelectorFactory)
    register_factory(HamiltonianConstructorFactory)
    register_factory(LocalizerFactory)
    register_factory(MultiConfigurationCalculatorFactory)
    register_factory(MultiConfigurationScfFactory)
    register_factory(ProjectedMultiConfigurationCalculatorFactory)
    register_factory(DynamicalCorrelationCalculatorFactory)
    register_factory(ScfSolverFactory)
    register_factory(StabilityCheckerFactory)


def _register_python_factories():
    """Register all built-in Python algorithm factories.

    This internal initialization function registers all the Python-implemented
    algorithm factories. This includes factories for energy estimators, phase estimation algorithms,
    qubit Hamiltonian solvers, qubit mappers, time evolution algorithms, and state preparation algorithms
    that are implemented in Python.

    This function is automatically called during module import and should not
    be called by users.
    """
    from qdk_chemistry.algorithms.circuit_executor import CircuitExecutorFactory  # noqa: PLC0415
    from qdk_chemistry.algorithms.energy_estimator import EnergyEstimatorFactory  # noqa: PLC0415
    from qdk_chemistry.algorithms.phase_estimation import PhaseEstimationFactory  # noqa: PLC0415
    from qdk_chemistry.algorithms.qubit_hamiltonian_solver import QubitHamiltonianSolverFactory  # noqa: PLC0415
    from qdk_chemistry.algorithms.qubit_mapper import QubitMapperFactory  # noqa: PLC0415
    from qdk_chemistry.algorithms.state_preparation import StatePreparationFactory  # noqa: PLC0415
    from qdk_chemistry.algorithms.time_evolution.builder import TimeEvolutionBuilderFactory  # noqa: PLC0415
    from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper import (  # noqa: PLC0415
        ControlledEvolutionCircuitMapperFactory,
    )

    register_factory(EnergyEstimatorFactory())
    register_factory(StatePreparationFactory())
    register_factory(QubitMapperFactory())
    register_factory(QubitHamiltonianSolverFactory())
    register_factory(TimeEvolutionBuilderFactory())
    register_factory(ControlledEvolutionCircuitMapperFactory())
    register_factory(CircuitExecutorFactory())
    register_factory(PhaseEstimationFactory())


_ = _register_cpp_factories()
_ = _register_python_factories()


def _ensure_cleanup_registered():
    """Ensure cleanup is registered exactly once per module import.

    This internal function makes sure that the atexit handler for cleaning up
    algorithm registrations is set up exactly once. This prevents redundant
    registrations of the cleanup function when multiple registration functions
    are called.

    The atexit handler is critical for preventing memory issues that can occur when
    C++ static deinitialization happens after Python garbage collection. Without this
    automatic cleanup, Python-implemented algorithms could cause double-free errors
    during interpreter shutdown.
    """
    global __cleanup_registered  # noqa: PLW0603
    if not __cleanup_registered:
        atexit.register(_cleanup_algorithms)
        __cleanup_registered = True


def _cleanup_algorithms():
    """Clean up all registered algorithms to prevent segfaults.

    This function is automatically called during Python interpreter shutdown
    through the atexit module. It ensures that all algorithm instances registered
    from Python are properly unregistered before Python garbage collection occurs.

    This prevents double-free errors that can happen when C++ static deinitialization
    runs after Python has already garbage-collected the Python objects. Users should
    never need to call this function directly as it's handled automatically by the
    registry system.
    """
    for factory in __factories:
        factory.clear()


def _register_python_algorithms():
    """Register all built-in Python algorithm instances.

    This internal initialization function registers specific Python-implemented
    algorithm instances as built-in algorithms. This includes the default QDK energy estimator,
    phase estimation algorithms, qubit Hamiltonian solvers, time evolution algorithms, and state preparation algorithms.

    This function is automatically called during module import and should not
    be called by users.
    """
    from qdk_chemistry.algorithms.circuit_executor.qdk import QdkFullStateSimulator  # noqa: PLC0415
    from qdk_chemistry.algorithms.energy_estimator import QDKEnergyEstimator  # noqa: PLC0415
    from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import (  # noqa: PLC0415
        IterativePhaseEstimation,
    )
    from qdk_chemistry.algorithms.qubit_hamiltonian_solver import DenseMatrixSolver, SparseMatrixSolver  # noqa: PLC0415
    from qdk_chemistry.algorithms.qubit_mapper import QdkQubitMapper  # noqa: PLC0415
    from qdk_chemistry.algorithms.state_preparation import SparseIsometryGF2XStatePreparation  # noqa: PLC0415
    from qdk_chemistry.algorithms.time_evolution.builder.trotter import (  # noqa: PLC0415
        Trotter,
    )
    from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper import (  # noqa: PLC0415
        PauliSequenceMapper,
    )

    register(lambda: QDKEnergyEstimator())
    register(lambda: SparseIsometryGF2XStatePreparation())
    register(lambda: DenseMatrixSolver())
    register(lambda: SparseMatrixSolver())
    register(lambda: QdkQubitMapper())
    register(lambda: Trotter())
    register(lambda: PauliSequenceMapper())
    register(lambda: QdkFullStateSimulator())
    register(lambda: IterativePhaseEstimation())


_register_python_algorithms()
