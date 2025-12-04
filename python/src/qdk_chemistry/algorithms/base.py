"""QDK Chemistry Algorithms Base Class.

This module defines the base class for custom algorithms that can be
integrated into the QDK/Chemistry framework.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from collections.abc import Callable

from qdk_chemistry.data import Settings


class Algorithm(ABC):
    """Base class for custom algorithms in QDK/Chemistry.

    In derived classes, ensure to call super().__init__() to properly
    initialize the base class and override the _settings attribute if
    custom settings are needed.
    Furthermore, derived classes must implement the abstract methods
    defined in this base class.

    Examples:
        >>> # Creating a custom SCF solver algorithm
        >>> from qdk_chemistry.algorithms import Algorithm, registry
        >>> from qdk_chemistry.data import Structure, Wavefunction, ElectronicStructureSettings
        >>>
        >>> class MyCustomScfSolver(Algorithm):
        ...     def __init__(self):
        ...         super().__init__()
        ...         # Replace with specialized settings
        ...         self._settings = ElectronicStructureSettings()
        ...         self._settings.set("max_iterations", 50)
        ...         self._settings.set("convergence_threshold", 1e-6)
        ...
        ...     def name(self) -> str:
        ...         return "my_custom_scf"
        ...
        ...     def type_name(self) -> str:
        ...         return "scf_solver"
        ...
        ...     def aliases(self) -> list[str]:
        ...         return ["my_custom_scf", "custom_scf"]
        ...
        ...     def _run_impl(self, structure: Structure, charge: int,
        ...                   spin_multiplicity: int) -> tuple[float, Wavefunction]:
        ...         # Custom SCF implementation
        ...         max_iter = self.settings().get("max_iterations")
        ...         threshold = self.settings().get("convergence_threshold")
        ...
        ...         # ... perform SCF calculation ...
        ...         energy = -1.0  # placeholder
        ...         wavefunction = Wavefunction()  # placeholder
        ...
        ...         return energy, wavefunction
        >>>
        >>> # Register the custom algorithm
        >>> registry.register(lambda: MyCustomScfSolver())
        >>>
        >>> # Use it like any built-in algorithm
        >>> scf = registry.create("scf_solver", "my_custom_scf")
        >>> # Or using the alias
        >>> scf = registry.create("scf_solver", "custom_scf")
        >>>
        >>> # Configure and run
        >>> scf.settings().set("max_iterations", 100)
        >>> energy, wfn = scf.run(structure, charge=0, spin_multiplicity=1)

    """

    def __init__(self):
        """Initialize the base algorithm."""
        super().__init__()
        self._settings = Settings()

    @abstractmethod
    def _run_impl(self, *args, **kwargs):
        """The implementation of the algorithm.

        Derived classes must implement this method.

        Args:
            args: The arguments required to run the algorithm.
            kwargs: The keyword arguments required to run the algorithm.

        Returns:
            Any: The results of the algorithm

        """

    def run(self, *args, **kwargs):
        """Run the algorithm with the provided arguments.

        This method wraps the internal _run_impl method to provide a
        consistent interface for executing the algorithm.

        Args:
            args: The arguments required to run the algorithm.
            kwargs: The keyword arguments required to run the algorithm.

        Returns:
            Any: The results of the algorithm

        """
        self._settings.lock()
        return self._run_impl(*args, **kwargs)

    def settings(self) -> Settings:
        """Get the settings for this algorithm.

        Returns:
            Settings:The settings object associated with this algorithm.

        """
        return self._settings

    @abstractmethod
    def type_name(self) -> str:
        """Return the name of the algorithm type.

        Derived classes must implement this method.

        An example of an algorithm type is "scf_solver".
        An example of an algorithm name is "pyscf" indicating
        the origin of the specific implementation of the algorithm type.
        Or in the case of an active space selector,
        an example of an algorithm type is "active_space_selector".
        and an example of an algorithm name is "qdk_valence",
        indicating the specific algorithm name and origin.

        Returns:
            str: The main name of the algorithm type.

        """

    @abstractmethod
    def name(self) -> str:
        """Return the main name of the algorithm.

        Derived classes must implement this method.

        An example of an algorithm type is "scf_solver".
        An example of an algorithm name is "pyscf" indicating
        the origin of the specific implementation of the algorithm type.
        Or in the case of an active space selector,
        an example of an algorithm type is "active_space_selector".
        and an example of an algorithm name is "qdk_valence",
        indicating the specific algorithm name and origin.

        Returns:
            str: The main name of the algorithm

        """

    def aliases(self) -> list[str]:
        """Return all aliases of the algorithm's name.

        Derived classes can override this method.
        The aliases must include the main name returned by name().

        By default, this method returns a list containing only the main name.

        Returns:
            list[str]: All aliases of the algorithm's name including the main name.

        """
        return [self.name()]


class AlgorithmFactory(ABC):
    """Base class for algorithm factories in QDK/Chemistry.

    Algorithm factories are responsible for creating and managing algorithm instances
    of a specific type. Each factory maintains a registry of algorithm implementations
    that can be instantiated by name. Factories handle both built-in C++ implementations
    and custom Python implementations.

    The factory pattern allows for dynamic algorithm selection at runtime and provides
    a centralized mechanism for registering and discovering available implementations.

    Note:
        This class is typically not used directly by end users. Instead, use the
        higher-level registry functions in `qdk_chemistry.algorithms` for
        creating and registering algorithms.

    Examples:
        >>> # Creating a custom factory for a new algorithm type
        >>> from qdk_chemistry.algorithms.base import AlgorithmFactory, Algorithm
        >>> import qdk_chemistry.algorithms.registry as registry
        >>> import qdk_chemistry.algorithms as algorithms
        >>> from qdk_chemistry.data import Structure
        >>>
        >>> # Example custom algorithm type
        >>> class GeometryOptimizer(Algorithm):
        ...     def type_name(self) -> str:
        ...         return "geometry_optimizer"
        >>>
        >>> # Example factory for this algorithm type
        >>> class GeometryOptimizerFactory(AlgorithmFactory):
        ...     def algorithm_type_name(self) -> str:
        ...         return "geometry_optimizer"
        ...
        ...     def default_algorithm_name(self) -> str:
        ...         return "bfgs"  # Default algorithm
        >>>
        >>> # Register a custom implementation
        >>> class BfgsOptimizer(GeometryOptimizer):
        ...     def name(self) -> str:
        ...         return "bfgs"
        ...     def _run_impl(self, structure: Structure):
        ...         # Implementation here
        ...         pass
        >>>
        >>> # Register the factory with the registry system
        >>> factory = GeometryOptimizerFactory()
        >>> registry.register_factory(factory)
        >>>
        >>> # Register algorithm implementation
        >>> algorithms.register(lambda: BfgsOptimizer())
        >>>
        >>> # Now use via the top-level API
        >>> optimizer = algorithms.create("geometry_optimizer", "bfgs")
        >>> available_opts = algorithms.available("geometry_optimizer")
        >>> print(available_opts)
        {'geometry_optimizer': ['bfgs']}

    See Also:
        qdk_chemistry.algorithms.registry:

            Higher-level registry functions for creating and managing algorithms across all types.

    """

    def __init__(self) -> None:
        """Initialize the algorithm factory with an empty registry."""
        self._registry: dict[str, Callable[[], Algorithm]] = {}

    @abstractmethod
    def algorithm_type_name(self) -> str:
        """Return the type name of algorithms this factory creates.

        Derived classes must implement this method to specify the algorithm
        type they manage (e.g., "scf_solver", "active_space_selector").

        Returns:
            str: The algorithm type name.

        """

    @abstractmethod
    def default_algorithm_name(self) -> str:
        """Return the name of the default algorithm for this type.

        Derived classes must implement this method to specify which algorithm
        should be created when no specific name is provided to `create()`.

        Returns:
            str: The name of the default algorithm implementation.

        """

    def create(self, name: str | None = None) -> Algorithm:
        """Create an algorithm instance by name.

        Creates and returns a new instance of the requested algorithm. If no name
        is provided, creates an instance of the default algorithm for this type.

        Args:
            name (Optional[str]): The name of the algorithm to create.

                If None or empty, creates the default algorithm.

        Returns:
            Algorithm: A new instance of the requested algorithm.

        Raises:
            RuntimeError: If the requested algorithm name is not registered in this factory.

        Examples:
            >>> factory = ScfSolverFactory()
            >>> # Create default SCF solver
            >>> default_scf = factory.create()
            >>> # Create specific implementation
            >>> pyscf_solver = factory.create("pyscf")

        """
        if name is None or name == "":
            name = self.default_algorithm_name()
        if name not in self._registry:
            raise RuntimeError(
                f"Algorithm '{name}' of type '{self.algorithm_type_name()}' is not registered. "
                f"Available algorithms: {list(self._registry.keys())}"
            )
        return self._registry[name]()

    def register_instance(self, generator: Callable[[], Algorithm]) -> None:
        """Register a new algorithm implementation in this factory.

        Adds a new algorithm to the factory's registry. The generator function
        will be called each time an instance of this algorithm is requested.

        Args:
            generator (Callable[[], Algorithm]): A callable that returns a new instance of the algorithm.

                Must return an Algorithm whose name()
                will be used as the registration key.

        Examples:
            >>> factory = ScfSolverFactory()
            >>> factory.register_instance(lambda: MyCustomScf())

        """
        self._registry[generator().name()] = generator

    def unregister_instance(self, name: str) -> bool:
        """Remove an algorithm implementation from this factory.

        Args:
            name (str): The name of the algorithm to unregister.

        Returns:
            bool: True if the algorithm was found and removed, False otherwise.

        Examples:
            >>> factory = ScfSolverFactory()
            >>> success = factory.unregister_instance("my_custom_scf")

        """
        return self._registry.pop(name, None) is not None

    def available(self) -> list[str]:
        """Get a list of all available algorithm names in this factory.

        Returns:
            list[str]: Names of all registered algorithms.

        Examples:
            >>> factory = ScfSolverFactory()
            >>> algos = factory.available()
            >>> print(algos)
            ['pyscf', 'qdk', 'my_custom_scf']

        """
        return list(self._registry.keys())

    def has(self, key: str) -> bool:
        """Check if an algorithm is registered in this factory.

        Args:
            key (str): The algorithm name to check.

        Returns:
            bool: True if the algorithm is registered, False otherwise.

        Examples:
            >>> factory = ScfSolverFactory()
            >>> if factory.has("pyscf"):
            ...     scf = factory.create("pyscf")

        """
        return key in self._registry

    def clear(self) -> None:
        """Remove all registered algorithms from this factory.

        This method clears the entire registry. Use with caution as it will
        remove all algorithm implementations, including built-in ones.

        Note:
            This is typically used internally for cleanup during Python
            interpreter shutdown. Users rarely need to call this directly.

        """
        self._registry.clear()
