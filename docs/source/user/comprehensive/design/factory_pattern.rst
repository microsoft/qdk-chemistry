Factory pattern
===============

QDK/Chemistry extensively uses the Factory pattern, a creational design pattern that provides an interface for creating objects without specifying their concrete classes.
This document explains how and why QDK/Chemistry uses this pattern for algorithm instantiation.

Overview of the factory pattern
-------------------------------

The factory pattern is a design pattern that encapsulates object creation logic.
Instead of directly instantiating objects using constructors, clients request objects from a factory, typically by specifying a type identifier.
This approach has several advantages:

1. **Abstraction**: Clients work with abstract interfaces rather than concrete implementations
2. **Flexibility**: The concrete implementation can be changed without affecting client code
3. **Configuration**: Objects can be configured based on runtime parameters
4. **Extension**: New implementations can be added without modifying existing code

Factory pattern in QDK/Chemistry
--------------------------------

In QDK/Chemistry, algorithm classes like ``ScfSolver``, ``Localizer``, and ``MCCalculator`` are instantiated through factory classes rather than direct constructors.
This design allows QDK/Chemistry to:

- Support multiple implementations of the same algorithm interface
- Configure algorithm instances based on settings
- Load algorithm implementations dynamically at runtime
- Isolate algorithm implementation details from client code

Factory classes in QDK/Chemistry
--------------------------------

QDK/Chemistry provides factory classes for each algorithm type:

.. list-table:: QDK/Chemistry Algorithm Factories
   :header-rows: 1
   :widths: auto

   * - Algorithm
     - Factory Class
     - Creation Function (Python)
   * - :doc:`ScfSolver <../algorithms/scf_solver>`
     - ``ScfSolverFactory``
     - ``create_scf_solver()``
   * - :doc:`Localizer <../algorithms/localizer>`
     - ``LocalizerFactory``
     - ``create_localizer()``
   * - :doc:`ActiveSpaceSelector <../algorithms/active_space>`
     - ``ActiveSpaceSelectorFactory``
     - ``create_active_space_selector()``
   * - :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`
     - ``HamiltonianConstructorFactory``
     - ``create_hamiltonian_constructor()``
   * - :doc:`MCCalculator <../algorithms/mc_calculator>`
     - ``MCCalculatorFactory``
     - ``create_mc_calculator()``
..   * - :doc:`DynamicalCorrelation <../algorithms/dynamical_correlation>`
..     - ``DynamicalCorrelationFactory``
..     - ``create_dynamical_correlation()``


Using factories
---------------

QDK/Chemistry provides factory methods to create algorithm instances in both C++ and Python.
These factory methods allow users to create default implementations or specific implementations by name.

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::algorithms;

      // Create default implementation
      auto scf_solver = ScfSolverFactory::create();

      // Create specific implementation by name
      auto localizer = LocalizerFactory::create("pipek-mezey");

      // Configure and use the instance
      scf_solver->settings().set("basis_set", "def2-tzvp");
      auto [E_scf, orbitals] = scf_solver->solve(structure);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/factory_pattern.py
      :language: python
      :lines: 4,19-29

Extending QDK/Chemistry
-----------------------

QDK/Chemistry is designed with extensibility in mind, allowing developers to add new algorithms, data formats, and functionality.
The factory pattern is the primary mechanism for extending QDK/Chemistry with custom implementations, especially for interfacing with external quantum chemistry programs.

Interface implementation patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QDK/Chemistry can be extended in several ways:

1. **External Program Interfaces**: Create implementations that connect QDK/Chemistry to external quantum chemistry packages
2. **Custom Algorithm Implementations**: Develop your own implementations of QDK/Chemistry's algorithm interfaces
3. **Data Format Bridges**: Create bridges between QDK/Chemistry's data structures and external formats

Implementing program interfaces in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a new interface to an external program in C++:

1. **Inherit from the algorithm base class**:

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry/algorithms/scf_solver.hpp>
      #include <external_program/api.h>  // Your external program's API

      namespace my_namespace {

      class ExternalProgramScfSolver : public qdk::chemistry::algorithms::ScfSolver {
      public:
          ExternalProgramScfSolver() = default;
          ~ExternalProgramScfSolver() override = default;

          // Implement the interface method that connects to your external program
          std::tuple<double, qdk::chemistry::data::Orbitals> solve(const qdk::chemistry::data::Structure& structure) override {
              // Convert QDK/Chemistry structure to external program format
              auto ext_molecule = convert_to_external_format(structure);

              // Run calculation using external program's API
              auto ext_results = external_program::run_scf(ext_molecule, settings().get_all());

              // Convert results back to QDK/Chemistry format
              double energy = ext_results.energy;
              qdk::chemistry::data::Orbitals orbitals = convert_from_external_format(ext_results.orbitals);

              return {energy, orbitals};
          }

      private:
          // Helper functions for format conversion
          external_program::Molecule convert_to_external_format(const qdk::chemistry::data::Structure& structure);
          qdk::chemistry::data::Orbitals convert_from_external_format(const external_program::Orbitals& ext_orbitals);
      };

      } // namespace my_namespace

1. **Register with the factory**:

.. code-block:: cpp

   #include <qdk/chemistry/algorithms/scf_solver_factory.hpp>

   // Register the implementation
   bool registered = qdk::chemistry::algorithms::ScfSolverFactory::register_implementation(
       "external-program",  // Name for this implementation
       []() { return std::make_unique<my_namespace::ExternalProgramScfSolver>(); }
   );

3. **Use your implementation**:

.. code-block:: cpp

   auto solver = qdk::chemistry::algorithms::ScfSolverFactory::create("external-program");

Implementing program interfaces in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a new interface to an external program in Python:

1. **Inherit from the algorithm base class**:

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import ScfSolver
      from qdk_chemistry.data import Structure, Orbitals
      import external_program  # Your external program's Python package

      class ExternalProgramScfSolver(ScfSolver):
          def __init__(self):
              super().__init__()

          def solve(self, structure):
              # Convert QDK/Chemistry structure to external program format
              ext_molecule = self._convert_to_external_format(structure)

              # Set up calculation with external program
              ext_calc = external_program.SCF(ext_molecule)

              # Apply settings
              for key, value in self.settings().get_all().items():
                  if key.startswith("external_program."):
                      option_name = key.split(".", 1)[1]
                      ext_calc.set_option(option_name, value)

              # Run calculation
              ext_results = ext_calc.run()

              # Convert results back to QDK/Chemistry format
              energy = ext_results.energy
              orbitals = self._convert_from_external_format(ext_results.orbitals)

              return energy, orbitals

          def _convert_to_external_format(self, structure):
              # Implementation of conversion from QDK/Chemistry to external format
              # ...
              return ext_molecule

          def _convert_from_external_format(self, ext_orbitals):
              # Implementation of conversion from external format to QDK/Chemistry
              # ...
              return orbitals

2. **Register with the factory**:

.. code-block:: python

   from qdk_chemistry.algorithms import register_scf_solver

   # Register the implementation
   register_scf_solver("external-program", ExternalProgramScfSolver)

3. **Use your implementation**:

.. code-block:: python

   from qdk_chemistry.algorithms import create_scf_solver
   solver = create_scf_solver("external-program")

Connection to the interface system
----------------------------------

The factory pattern serves as the foundation for QDK/Chemistry's :doc:`Interface System <interfaces>`.
In QDK/Chemistry, factories enable the registration and instantiation of interface implementations that connect to external quantum chemistry programs.

Internally, QDK/Chemistry's factories maintain a registry of creator functions mapped to implementation names.
When a client requests an implementation by name, the factory looks up the appropriate creator function and instantiates the object with the necessary setup.

This design enables several key capabilities:

- Seamless integration with external quantum chemistry packages
- Runtime selection of specific implementations
- Decoupling of interface usage from implementation details

For detailed information about implementing interfaces to external programs, including data conversion, error handling, resource management, and settings translation, see the :doc:`Interfaces <interfaces>` documentation.

Related topics
--------------

- :doc:`Settings <settings>`: Configuration of algorithm instances
- :doc:`Interfaces <interfaces>`: QDK/Chemistry's interface system to external packages
