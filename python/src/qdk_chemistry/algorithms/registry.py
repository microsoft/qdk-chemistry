"""QDK/Chemistry-algorithms registry.

This module contains a registration mechanism for Hamiltonian constructor, SCF solver, localizer,
active space selector, coupled cluster calculator, (projected) multi configuration calculator,
and MultiConfigurationScf. The user should be able to use the registration mechanism to easily
add and remove custom algorithms at runtime.

Algorithm Lifecycle Management
-----------------------------
The registry system handles the lifecycle of algorithm instances to prevent memory issues. When
Python-implemented algorithms are registered, they need special handling during Python interpreter
shutdown to avoid "double-free" errors that can occur when C++ static deinitialization runs after
Python's garbage collection.

This module automatically registers cleanup handlers (via `atexit`) that unregister all custom
algorithms before Python shuts down. Users should never need to call the cleanup functions directly.

Important Notes:
- All registration functions in this module provide automatic cleanup
- Custom Python algorithms are automatically unregistered during interpreter shutdown
- The cleanup functions are NOT intended to be called by users
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import atexit
from collections.abc import Callable
from typing import Any

from qdk_chemistry._core._algorithms import (
    ActiveSpaceSelectorFactory,
    CoupledClusterCalculatorFactory,
    HamiltonianConstructorFactory,
    LocalizerFactory,
    MultiConfigurationCalculatorFactory,
    MultiConfigurationScfFactory,
    ProjectedMultiConfigurationCalculatorFactory,
    ScfSolverFactory,
    StabilityCheckerFactory,
)
from qdk_chemistry.algorithms.energy_estimator import EnergyEstimatorFactory, QDKEnergyEstimator
from qdk_chemistry.algorithms.qubit_mapper import QubitMapperFactory
from qdk_chemistry.algorithms.state_preparation import SparseIsometryGF2XStatePreparation, StatePreparationFactory

from .base import Algorithm, AlgorithmFactory

# Universal cleanup solution for all algorithms
__cleanup_registered: bool = False

__factories: list[AlgorithmFactory] = []


def create(algorithm_type: str, algorithm_name: str | None = None, **kwargs) -> Algorithm:
    """Create an algorithm instance by type and name.

    This function creates an algorithm instance from the registry using the specified
    algorithm type and name. If no name is provided, the default algorithm for that
    type is created.

    Available algorithm types and algorithms are dependent on the QDK/Chemistry installation
    but importantly also the loaded plugins and registered custom algorithms. If an
    expected algorithm is not found, ensure that the corresponding plugin is loaded
    or that the custom algorithm is registered.

    The currently loaded algorithms can be queried using the `available()` function.
    Which lists the available algorithms by type for all currently loaded plugins and
    registered custom algorithms.

    Args:
        algorithm_type (str): The type of algorithm to create (e.g., "scf_solver",
            "active_space_selector", "coupled_cluster_calculator").
        algorithm_name (Optional[str]): The specific name of the algorithm implementation
            to create. If None or empty string, creates the default algorithm for that type.
        **kwargs: Optional keyword arguments to configure the algorithm's settings.
            These are passed directly to the algorithm's settings via `settings().update()`.
            Available settings depend on the specific algorithm type and implementation
            and can be looked up with show_settings() method in .

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


def show_settings(algorithm_type: str, algorithm_name: str) -> list[tuple[str, str, Any]]:
    """Show the settings schema for a specific algorithm.

    This function retrieves the settings schema for a given algorithm type and name.
    The settings schema provides information about configurable parameters for the
    algorithm, including their names, expected Python types, and default values.

    Args:
        algorithm_type (str): The type of algorithm (e.g., "scf_solver",
            "active_space_selector", "coupled_cluster_calculator").
        algorithm_name (str): The specific name of the algorithm implementation.

    Returns:
        list[tuple[str, str, Any]]: A list of tuples where each tuple contains:
            - Setting name (str): The name/key of the setting
            - Expected Python type (str): The Python type expected for this setting
              (e.g., int, float, str, bool, list[int], list[float])
            - Default value (Any): The default value for this setting

    Raises:
        KeyError: If the specified algorithm type is not registered in the system.

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> # Show settings for the PySCF SCF solver
        >>> settings_info = registry.show_settings("scf_solver", "pyscf")
        >>> for name, python_type, default in settings_info:
        ...     print(f"{name}: {python_type} = {default}")
        method: str = hf
        basis_set: str = def2-svp
        charge: int = 0
        spin_multiplicity: int = 1
        tolerance: float = 1e-06
        max_iterations: int = 50
        force_restricted: bool = False

    """
    for factory in __factories:
        if factory.algorithm_type_name() == algorithm_type:
            instance = factory.create(algorithm_name)
            settings = instance.settings().get_all_settings()
            return [
                (name, instance.settings().get_expected_python_type(name), default)
                for name, default in settings.items()
            ]
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
        generator (Callable[[], Algorithm]): A callable that returns a new instance of
            the custom algorithm. This will be called each time the algorithm is created
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
        algorithm_type (Optional[str]): If provided, only list algorithms of this type.
            If None, list all algorithms across all types.

    Returns:
        dict[str, list[str]] | list[str]: When algorithm_type is None, returns a dictionary
            where keys are algorithm type names and values are lists of available algorithm
            names for that type. When algorithm_type is specified, returns a list of available
            algorithm names for that specific type (empty list if type not found or no
            algorithms are available).

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> # List all available algorithms across all types
        >>> all_algorithms = registry.available()
        >>> print(all_algorithms)
        {'scf_solver': ['pyscf', 'qdk'], 'active_space_selector': ['occupation', 'avas'], ...}
        >>> # List only SCF solvers
        >>> scf_solvers = registry.available("scf_solver")
        >>> print(scf_solvers)
        ['pyscf', 'qdk']
        >>> # Check what active space selectors are available
        >>> selectors = registry.available("active_space_selector")
        >>> print(selectors)
        ['occupation', 'avas', 'valence']

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


def unregister(algorithm_type: str, algorithm_name: str) -> None:
    """Unregister a custom algorithm implementation.

    This function removes a previously registered algorithm from the registry.

    Args:
        algorithm_type (str): The type of algorithm to unregister
            (e.g., "scf_solver", "active_space_selector").
        algorithm_name (str): The name of the specific algorithm implementation
            to unregister.

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
    register_factory(ActiveSpaceSelectorFactory)
    register_factory(CoupledClusterCalculatorFactory)
    register_factory(HamiltonianConstructorFactory)
    register_factory(LocalizerFactory)
    register_factory(MultiConfigurationCalculatorFactory)
    register_factory(MultiConfigurationScfFactory)
    register_factory(ProjectedMultiConfigurationCalculatorFactory)
    register_factory(ScfSolverFactory)
    register_factory(StabilityCheckerFactory)


def _register_python_factories():
    register_factory(EnergyEstimatorFactory())
    register_factory(StatePreparationFactory())
    register_factory(QubitMapperFactory())


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


# Register built-in Python energy estimators
register(lambda: QDKEnergyEstimator())
register(lambda: SparseIsometryGF2XStatePreparation())
