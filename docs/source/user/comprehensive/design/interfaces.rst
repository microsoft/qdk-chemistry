Interface system
================

This document describes QDK/Chemistry's interface system, which provides unified access to various quantum chemistry packages.
This design allows researchers to leverage the unique strengths of various quantum chemistry programs while maintaining a consistent workflow.

Overview
--------

QDK/Chemistry is designed with an extensible plugin architecture that allows algorithms to be implemented either natively within QDK/Chemistry or as interfaces to established third-party quantum chemistry packages.
This approach combines the benefits of a consistent API with the specialized capabilities of different software packages, following QDK/Chemistry's core :doc:`design principles <index>` of extensibility and interoperability.

The interface system implements the :doc:`factory pattern <factory_pattern>`, which allows algorithms to be instantiated by name without the user needing to know the specific implementation details.
This abstraction enables seamless switching between different backends without changing your code, making it easier to benchmark and compare different approaches.

Interface architecture
----------------------

The interface system is built on the following principles:

1. **Unified API**: All implementations of an algorithm share the same interface, regardless of the underlying
   implementation
2. **Runtime Selection**: Users can select implementations at runtime without changing their code
3. **Transparent Delegation**: QDK/Chemistry handles all data format conversions between the QDK/Chemistry data model and external
   packages
4. **Consistent Configuration**: All implementations are configured through the same :doc:`Settings <settings>` interface

.. graphviz:: /_static/diagrams/interface_architecture.dot

Supported interfaces
--------------------

QDK/Chemistry provides interfaces to many popular quantum chemistry packages, each carefully integrated to preserve their strengths while presenting a unified API to the user.
These include:

- **PySCF**: Python-based Simulations of Chemistry Framework

Each interface is implemented as a derived class that inherits from the appropriate algorithm base class (e.g.,
:doc:`ScfSolver <../algorithms/scf_solver>`, :doc:`Localizer <../algorithms/localizer>`), ensuring type safety and consistent behavior across implementations.

Using interfaces
----------------

Interfaces are accessed through the standard algorithm factory pattern, which provides a consistent way to instantiate
any algorithm regardless of its implementation.
This pattern is implemented across all major algorithm types in QDK/Chemistry.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      // Create an SCF solver that uses the QDK/Chemistry library as solver
      auto scf = ScfSolverFactory::create();

      // Configure it using the standard settings interface
      scf->settings().set("basis_set", "cc-pvdz");
      scf->settings().set("method", "hf");

      // Run calculation with the same API as native implementations
      auto [energy, orbitals] = scf->solve(structure);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/interfaces.py
      :language: python
      :lines: 1-14

Listing available implementations
---------------------------------

You can discover what implementations are available for each algorithm type:

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      #include <iostream>

      // Get a list of available SCF solver implementations
      auto available_solvers = ScfSolverFactory::available();
      for (const auto& solver : available_solvers) {
        std::cout << solver << std::endl;
      }

      // Get documentation for a specific implementation
      std::cout << ScfSolverFactory::get_docstring("default") << std::endl;

.. tab:: Python API

   .. literalinclude:: ../../../../examples/factory_pattern.py
      :language: python
      :lines: 1-13

Adding new interfaces
---------------------

QDK/Chemistry's plugin architecture makes it straightforward to add interfaces to new packages.
This extensibility is a core design principle of QDK/Chemistry, allowing the community to expand the toolkit's capabilities.

To create a new interface:

1. Create a new implementation class that inherits from the algorithm's base class (e.g., ``ScfSolver``, ``Localizer``)
2. Register the implementation with the algorithm's factory using the registration methods
3. Implement the required methods, translating between QDK/Chemistry data structures and the external package's format

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
              custom_mol,
              settings().get<std::string>("basis_set"),
              settings().get<std::string>("method")
          );

          // Convert results back to QDK/Chemistry format
          double energy = result.energy;
          data::Orbitals orbitals = convert_from_custom_format(result.orbitals);

          return {energy, orbitals};
        }

      private:
        custom_chemistry::Molecule convert_to_custom_format(const data::Structure& structure);
        data::Orbitals convert_from_custom_format(const custom_chemistry::Orbitals& orbitals);
      };

      // Register in a static initializer block
      namespace {
      bool registered = ScfSolverFactory::register_implementation(
          "custom",
          []() { return std::make_unique<CustomScfSolver>(); },
          "Interface to Custom Chemistry Package");
      }  // anonymous namespace

      }  // namespace algorithms
      }  // namespace qdk::chemistry

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import register_scf_solver
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

For developers interested in adding new interfaces, please refer to the :doc:`Factory pattern <factory_pattern>` documentation for more detailed guidance.

Interface-specific settings
---------------------------

While QDK/Chemistry provides a unified API for all implementations, each backend may support additional options specific to that package.
These package-specific settings are accessed through the same :doc:`Settings <settings>` interface but are typically prefixed with the package name to avoid namespace collisions.
This approach leverages the flexibility of QDK/Chemistry's :doc:`settings system <settings>` to accommodate package-specific options while maintaining a consistent configuration experience.

.. tab:: C++ API

   .. code-block:: cpp

      // Set general options that work across all backends
      scf->settings().set("basis_set", "cc-pvdz");
      scf->settings().set("max_iterations", 100);

.. tab:: Python API

   .. code-block:: python

      # Set general options that work across all backends
      scf.settings().set("basis_set", "cc-pvdz")
      scf.settings().set("max_iterations", 100)

Each interface implementation typically documents its specific settings, including both the common settings that are
translated to the backend and the backend-specific settings that are passed through directly.

Data conversion
---------------

QDK/Chemistry handles the conversion of data between its own format and third-party packages automatically.
This internal conversion process is transparent to the user, allowing you to work exclusively with QDK/Chemistry data structures regardless of which backend implementation is used.
This capability is built on QDK/Chemistry's robust :doc:`serialization <../data/serialization>` system, which provides standardized methods for data conversion between different formats.

The data types that are automatically converted include:

- **:doc:`Molecular structures <../data/structure>`**: Atoms, coordinates, charges, and multiplicity
- **:doc:`Basis sets <../data/basis_set>`**: Basis set specifications, primitive and contracted functions
- **:doc:`Orbitals and wavefunctions <../data/orbitals>`**: Coefficients, occupations, and energies
- **:doc:`Hamiltonians <../data/hamiltonian>`**: One and two-electron integrals, core Hamiltonians
- **Calculation results** (see :class:`~qdk_chemistry.data.Wavefunction`): Energies, gradients, properties

The conversion process is optimized to minimize data copying when possible, especially for large data structures like electron repulsion integrals (:term:`ERIs`).
When working with large systems, QDK/Chemistry may use direct algorithms or disk-based approaches to manage memory usage efficiently.

Performance considerations
--------------------------

Using an interface to a third-party package may involve some overhead for data conversion.
However, this overhead is typically negligible compared to the computational cost of the quantum chemical calculations themselves, especially for larger systems and more computationally intensive methods.

QDK/Chemistry implements several optimizations to minimize this overhead:

1. **Lazy evaluation**: Some data conversions are only performed when actually needed
2. **Caching**: Converted data may be cached to avoid repeated conversions
3. **Direct interfaces**: For some packages, QDK/Chemistry can use direct memory interfaces instead of file-based interfaces

Different backend implementations may have different performance characteristics depending on the system size, method,
and hardware environment.

Available interfaces by algorithm
---------------------------------

.. note::
   PySCF and other third-party interfaces may not be fully implemented for all algorithm classes mentioned.

The following table provides an overview of the available interfaces for each algorithm class in QDK/Chemistry.
Each algorithm class is implemented through the factory pattern, allowing you to select different implementations at runtime.

.. list-table::
   :header-rows: 1

   * - Algorithm class
     - QDK/Chemistry implementations
     - Third-party interfaces
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

.. TODO: fix the function names below in this commented-out text.
.. You can discover all available implementations for a particular algorithm using the appropriate listing function (e.g., ``list_scf_solvers()`` in Python or ``ScfSolverFactory::available()`` in C++).

Related topics
--------------

- :doc:`Design principles <index>`: Core architectural principles of QDK/Chemistry
- :doc:`Factory pattern <factory_pattern>`: How to extend QDK/Chemistry with new algorithms and interfaces
- :doc:`Settings <settings>`: Configuring algorithm behavior consistently across implementations
- :doc:`Serialization <../data/serialization>`: Data persistence and conversion in QDK/Chemistry
