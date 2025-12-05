.. _extending:

Extending QDK/Chemistry
=======================

This document describes how to extend QDK/Chemistry's functionality through its plugin system.
QDK/Chemistry is designed with an extensible plugin architecture that allows algorithms to be implemented either natively within QDK/Chemistry or as interfaces to established third-party quantum chemistry packages.
This approach combines the benefits of a consistent API with the specialized capabilities of different software packages, following QDK/Chemistry's core :doc:`design principles <design/index>` of extensibility and interoperability.

.. _plugin-system:

Plugin system
-------------

QDK/Chemistry's plugin system provides unified access to both native implementations and third-party quantum chemistry packages.
This design allows researchers to leverage the unique strengths of various quantum chemistry programs while maintaining a consistent workflow.

.. _algorithm-plugin-relationship:

Algorithm classes are the foundation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the heart of the plugin system are QDK/Chemistry's algorithm classes.
Each algorithm class defines a contract—a set of inputs, outputs, and behaviors—that all implementations must follow.
The plugin system then enables multiple implementations of each algorithm class, whether they are native QDK/Chemistry implementations or interfaces to third-party packages.

The ``Algorithm`` base class provides common functionality for all algorithms:

- **Settings management**: All algorithms inherit the ability to be configured through the :doc:`Settings <algorithms/settings>` system
- **Type identification**: Each algorithm class declares a ``type_name()`` that identifies its category (e.g., "scf_solver", "localizer")
- **Execution interface**: A common pattern for running the algorithm and returning results

Concrete algorithm classes like :doc:`ScfSolver <algorithms/scf_solver>`, :doc:`Localizer <algorithms/localizer>`, and :doc:`MCCalculator <algorithms/mc_calculator>` inherit from ``Algorithm`` and define the specific interface for their computational task:

.. code-block:: text

   Algorithm (base)
   ├── ScfSolver
   │   ├── QdkScfSolver      (native implementation)
   │   └── PyscfScfSolver    (PySCF plugin)
   ├── OrbitalLocalizer
   │   ├── QdkLocalizer      (native implementation)
   │   └── PyscfLocalizer    (PySCF plugin)
   └── MCCalculator
       ├── MacisCalculator   (native MACIS implementation)
       └── PyscfMCCalculator (PySCF plugin)

.. _plugin-architecture:

Factories connect algorithms to plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each algorithm class has an associated factory (e.g., ``ScfSolverFactory``) that maintains a registry of available implementations.
The :doc:`factory pattern <algorithms/factory_pattern>` allows algorithms to be instantiated by name without the user needing to know the specific implementation details.

When you call ``create_scf_solver("pyscf")``, the factory:

1. Looks up "pyscf" in its registry
2. Instantiates the corresponding ``PyscfScfSolver`` class
3. Returns it as an ``ScfSolver`` reference, hiding the implementation details

This separation means your code works with the abstract ``ScfSolver`` interface, while the plugin system handles the concrete implementation behind the scenes.

.. graphviz:: /_static/diagrams/interface_architecture.dot

Design principles
~~~~~~~~~~~~~~~~~

The plugin system is built on the following principles:

1. **Unified API**: All implementations of an algorithm share the same interface, regardless of the underlying implementation
2. **Runtime Selection**: Users can select implementations at runtime without changing their code
3. **Transparent Delegation**: QDK/Chemistry handles all data format conversions between the QDK/Chemistry data model and external packages
4. **Consistent Configuration**: All implementations are configured through the same :doc:`Settings <algorithms/settings>` interface

This design enables several powerful workflows:

- **Benchmarking**: Compare native QDK/Chemistry implementations against established packages like PySCF using identical input data and settings
- **Gradual adoption**: Start with familiar third-party packages, then switch to native implementations as needed
- **Best-of-breed selection**: Use PySCF for SCF calculations but MACIS for multi-configurational methods, all within a single workflow
- **Custom extensions**: Add your own implementations that seamlessly integrate with existing algorithms

For details on creating your own implementations, see :ref:`adding-plugins` below.

.. _third-party-plugins:

Third-party plugins
~~~~~~~~~~~~~~~~~~~

QDK/Chemistry provides plugins for popular quantum chemistry packages, each carefully integrated to preserve their strengths while presenting a unified API to the user.
These include:

- **PySCF**: Python-based Simulations of Chemistry Framework

Each plugin is implemented as a derived class that inherits from the appropriate algorithm base class (e.g., :doc:`ScfSolver <algorithms/scf_solver>`, :doc:`Localizer <algorithms/localizer>`), ensuring type safety and consistent behavior across implementations.

Using plugins
~~~~~~~~~~~~~

Plugins are accessed through the standard algorithm factory pattern, which provides a consistent way to instantiate
any algorithm regardless of its implementation.
This pattern is implemented across all major algorithm types in QDK/Chemistry.

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/interfaces.cpp
      :language: cpp
      :start-after: // start-cell-scf
      :end-before: // end-cell-scf

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/interfaces.py
      :language: python
      :start-after: # start-cell-scf
      :end-before: # end-cell-scf

.. _listing-implementations:

Listing available implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can discover what implementations are available for each algorithm type:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/interfaces.cpp
      :language: cpp
      :start-after: // start-cell-list-methods
      :end-before: // end-cell-list-methods

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/interfaces.py
      :language: python
      :start-after: # start-cell-list-methods
      :end-before: # end-cell-list-methods

.. _plugin-specific-settings:

Plugin-specific settings
~~~~~~~~~~~~~~~~~~~~~~~~

While QDK/Chemistry provides a unified API for all implementations, each backend may support additional options specific to that package.
These package-specific settings are accessed through the same :doc:`Settings <algorithms/settings>` interface but are typically prefixed with the package name to avoid namespace collisions.
This approach leverages the flexibility of QDK/Chemistry's :doc:`settings system <algorithms/settings>` to accommodate package-specific options while maintaining a consistent configuration experience.

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/interfaces.cpp
      :language: cpp
      :start-after: // start-cell-settings
      :end-before: // end-cell-settings

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/interfaces.py
      :language: python
      :start-after: # start-cell-settings
      :end-before: # end-cell-settings

Each interface implementation typically documents its specific settings, including both the common settings that are
translated to the backend and the backend-specific settings that are passed through directly.

Data conversion
~~~~~~~~~~~~~~~

QDK/Chemistry handles the conversion of data between its own format and third-party packages automatically.
This internal conversion process is transparent to the user, allowing you to work exclusively with QDK/Chemistry data structures regardless of which backend implementation is used.
This capability is built on QDK/Chemistry's robust :doc:`serialization <data/serialization>` system, which provides standardized methods for data conversion between different formats.

The data types that are automatically converted include:

:doc:`Molecular structures <data/structure>`:
   Atoms, coordinates, charges, and multiplicity
:doc:`Basis sets <data/basis_set>`:
   Basis set specifications, primitive and contracted functions
:doc:`Orbitals and wavefunctions <data/orbitals>`:
   Coefficients, occupations, and energies
:doc:`Hamiltonians <data/hamiltonian>`:
   One and two-electron integrals, core Hamiltonians
Calculation results (see :class:`~qdk_chemistry.data.Wavefunction`):
   Energies, gradients, properties

The conversion process is optimized to minimize data copying when possible, especially for large data structures like electron repulsion integrals (:term:`ERIs`).
When working with large systems, QDK/Chemistry may use direct algorithms or disk-based approaches to manage memory usage efficiently.

Performance considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Using an interface to a third-party package may involve some overhead for data conversion.
However, this overhead is typically negligible compared to the computational cost of the quantum chemical calculations themselves, especially for larger systems and more computationally intensive methods.

QDK/Chemistry implements several optimizations to minimize this overhead:

1. **Lazy evaluation**: Some data conversions are only performed when actually needed
2. **Caching**: Converted data may be cached to avoid repeated conversions
3. **Direct interfaces**: For some packages, QDK/Chemistry can use direct memory interfaces instead of file-based interfaces

Different backend implementations may have different performance characteristics depending on the system size, method,
and hardware environment.

.. _available-plugins:

Available plugins by algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   PySCF and other third-party plugins may not be fully implemented for all algorithm classes mentioned.

The following table provides an overview of the available plugins for each algorithm class in QDK/Chemistry.
Each algorithm class is implemented through the factory pattern, allowing you to select different implementations at runtime.

.. list-table::
   :header-rows: 1

   * - Algorithm class
     - QDK/Chemistry implementations
     - Third-party plugins
   * - :class:`~qdk_chemistry.algorithms.ScfSolver`
     - "qdk"
     - "pyscf"
   * - :class:`~qdk_chemistry.algorithms.OrbitalLocalizer`
     - "qdk"
     - "pyscf"
   * - :class:`~qdk_chemistry.algorithms.MultiConfigurationCalculator`
     - "macis_asci", "macis_cas"
     - "pyscf"
   * - :class:`~qdk_chemistry.algorithms.ActiveSpaceSelector`
     - "qdk", "autocas"
     - "pyscf"

.. _plugins:

Included plugins
----------------

Besides the core functionality provided by QDK/Chemistry, included and compiled into the library by default, such as :term:`MACIS`, there is also a plugin system that allows for extending the functionality of QDK/Chemistry by adding custom algorithms, data structures, and interfaces.
Some plugins for popular quantum chemistry packages and quantum computing packages are provided with QDK/Chemistry, while others can be found in community repositories.

.. _core-plugins:

Core plugins
~~~~~~~~~~~~

The following lists included plugins that are available in QDK/Chemistry, developed and maintained by the QDK/Chemistry team.
These plugins are shipped and installed along with QDK/Chemistry and are enabled once the corresponding external packages are installed by the user.

- `Qiskit <https://www.ibm.com/quantum/qiskit>`_
- `PySCF <https://pyscf.org/>`_

.. _community-plugins:

Community plugins
~~~~~~~~~~~~~~~~~

We welcome the addition of community-developed plugins to enhance the capabilities of QDK/Chemistry.

.. The following lists plugins that are available in addition to the above list, these plugins are developed and maintained by the community, for installation instructions please refer to the respective repositories and documentation.
.. (Note: This is likely an incomplete list.
.. If you are aware of other community plugins, please consider contributing to the documentation.)

.. - t.b.d.

.. _adding-plugins:

Creating custom plugins
-----------------------

QDK/Chemistry's plugin architecture makes it straightforward to extend the toolkit's capabilities.
There are two main extension scenarios:

1. **Adding implementations for existing algorithm types**: Create a new backend for an existing algorithm class like ``ScfSolver`` or ``Localizer``
2. **Creating entirely new algorithm types**: Define new algorithm classes with their own factories and implementations

.. _adding-implementations:

Adding implementations for existing algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new implementation for an existing algorithm type (e.g., integrating a new quantum chemistry package):

1. Create a new implementation class that inherits from the algorithm's base class (e.g., ``ScfSolver``, ``Localizer``)
2. Implement the required methods, translating between QDK/Chemistry data structures and the external package's format
3. Register the implementation with the algorithm's factory

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      #include "custom_chemistry_package.hpp"

      namespace qdk::chemistry {
      namespace algorithms {

      class CustomScfSolver : public ScfSolver {
      public:
      CustomScfSolver() = default;

      std::tuple<double, data::Orbitals> solve(
            const data::Structure& structure) override {
         // Convert QDK/Chemistry structure to custom package format
         auto custom_mol = convert_to_custom_format(structure);

         // Run calculation with custom package
         auto result = custom_chemistry::run_scf(
            custom_mol, settings().get<std::string>("basis_set"),
            settings().get<std::string>("method"));

         // Convert results back to QDK/Chemistry format
         double energy = result.energy;
         data::Orbitals orbitals = convert_from_custom_format(result.orbitals);

         return {energy, orbitals};
      }

      private:
      custom_chemistry::Molecule convert_to_custom_format(
            const data::Structure& structure);
      data::Orbitals convert_from_custom_format(
            const custom_chemistry::Orbitals& orbitals);
      };

      // Register in a static initializer block
      namespace {
      bool registered = ScfSolverFactory::register_implementation(
         "custom", []() { return std::make_unique<CustomScfSolver>(); },
         "Interface to Custom Chemistry Package");
      }  // anonymous namespace

      }  // namespace algorithms
      }  // namespace qdk::chemistry

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import ScfSolver, register_scf_solver
      import custom_chemistry_package as ccp

      class CustomScfSolver(ScfSolver):
          def __init__(self):
              super().__init__()

          def solve(self, structure):
              # Convert QDK/Chemistry structure to custom package format
              custom_mol = self._convert_structure(structure)

              # Run calculation with custom package
              custom_energy, custom_orbs = ccp.run_scf(
                  custom_mol,
                  basis=self.settings().get("basis_set"),
                  method=self.settings().get("method")
              )

              # Convert results back to QDK/Chemistry format
              energy = custom_energy
              orbitals = self._convert_orbitals(custom_orbs)

              return energy, orbitals

      # Register the new solver with the factory
      register_scf_solver("custom", CustomScfSolver,
                         "Interface to Custom Chemistry Package")

.. _custom-algorithm-types:

Creating new algorithm types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create an entirely new algorithm type (e.g., a geometry optimizer that doesn't yet exist in QDK/Chemistry):

1. Define a new algorithm class that inherits from ``Algorithm``
2. Create a factory class that inherits from ``AlgorithmFactory``
3. Implement a concrete algorithm
4. Register both the factory and implementation with the registry system

.. code-block:: python

    from qdk_chemistry.algorithms.base import AlgorithmFactory, Algorithm
    import qdk_chemistry.algorithms.registry as registry
    import qdk_chemistry.algorithms as algorithms
    from qdk_chemistry.data import Structure

    # Step 1: Define the custom algorithm type
    class GeometryOptimizer(Algorithm):
        def type_name(self) -> str:
            return "geometry_optimizer"

    # Step 2: Create a factory for this algorithm type
    class GeometryOptimizerFactory(AlgorithmFactory):
        def algorithm_type_name(self) -> str:
            return "geometry_optimizer"

        def default_algorithm_name(self) -> str:
            return "bfgs"  # Default algorithm

    # Step 3: Implement a concrete algorithm
    class BfgsOptimizer(GeometryOptimizer):
        def name(self) -> str:
            return "bfgs"

        def _run_impl(self, structure: Structure) -> Structure:
            # Implementation here
            ...
            return new_structure  # Return optimized structure

    # Step 4: Register the factory with the registry system.
    #         (Done in the initialization phase of your plugin
    #         when shipped as a package.)
    factory = GeometryOptimizerFactory()
    registry.register_factory(factory)

    # Step 5: Register algorithm implementation
    #         (Done in the initialization phase of your plugin
    #         when shipped as a package.)
    algorithms.register(lambda: BfgsOptimizer())


    # Now use via the top-level API
    optimizer = algorithms.create("geometry_optimizer", "bfgs")
    available_opts = algorithms.available("geometry_optimizer")
    print(available_opts)
    # Output: {'geometry_optimizer': ['bfgs']}

Key components
~~~~~~~~~~~~~~

**Algorithm Base Class**
    Your custom algorithm must inherit from ``Algorithm`` and implement the ``type_name()`` method to identify the algorithm type.

**Factory Class**
    The factory manages creation and registration of algorithm instances.
    It must implement ``algorithm_type_name()`` and ``default_algorithm_name()`` methods.

**Registry System**
    The registry (``qdk_chemistry.algorithms.registry``) maintains all available algorithm types and their implementations, enabling discovery and instantiation at runtime.

**Top-Level API**
    Once registered, your custom algorithms are accessible through the standard ``algorithms.create()`` and ``algorithms.available()`` functions, maintaining consistency with built-in algorithms.

For more detailed guidance on the factory pattern, see the :doc:`Factory pattern <algorithms/factory_pattern>` documentation.

Further reading
---------------

- Some of the above examples can be downloaded as complete `C++ <../../_static/examples/cpp/interfaces.cpp>`_ and `Python <../../_static/examples/python/interfaces.py>`_ scripts.
- :doc:`Design principles <design/index>`: Core architectural principles of QDK/Chemistry
- :doc:`Factory pattern <algorithms/factory_pattern>`: How to extend QDK/Chemistry with new algorithms and interfaces
- :doc:`Settings <algorithms/settings>`: Configuring algorithm behavior consistently across implementations
- :doc:`Serialization <data/serialization>`: Data persistence and conversion in QDK/Chemistry
