Design Principles of QDK/Chemistry
==================================

This document outlines the core architectural design principles of QDK/Chemistry, explaining the conceptual framework that guides the library's organization and implementation.

For a complete overview of QDK/Chemistry's documentation, see the :doc:`comprehensive documentation index <../index>`.

Overview
--------

QDK/Chemistry is designed with a clear separation between **data containers** and **algorithms**.
This fundamental design choice enables flexibility, extensibility, and maintainability of the codebase, while providing
users with a consistent and intuitive API.

Core Design Principles
----------------------

Separation of Data and Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QDK/Chemistry follows a design pattern that strictly separates:

1. **Data Classes**: Immutable containers that store and manage quantum chemical data
2. **Algorithm Classes**: Processors that operate on data objects to produce new data objects

This separation follows the principle of single responsibility and creates a clear flow of data through computational
workflows.

.. graphviz:: /_static/diagrams/data_flow.dot

Data Classes
~~~~~~~~~~~~

Data classes in QDK/Chemistry are designed to be:

- Immutable: Once created, the core data cannot be modified (except through specific methods)
- Self-contained: Include all information necessary to represent the concept
- :doc:`Serializable <serialization>`: Can be easily saved to and loaded from files
- Language-agnostic: Accessible through identical APIs in both C++ and Python

QDK/Chemistry includes the following data classes:

- :doc:`Structure <../data/structure>`: Molecular geometry and composition
- :doc:`BasisSet <../data/basis_set>`: Quantum chemistry basis set definitions
- :doc:`Orbitals <../data/orbitals>`: Molecular orbital coefficients and energies
- :doc:`Hamiltonian <../data/hamiltonian>`: Molecular Hamiltonian
- :doc:`Wavefunction <../data/wavefunction>`: Wavefunction representation

Algorithm Classes
~~~~~~~~~~~~~~~~~

Algorithm classes in QDK/Chemistry are designed to be:

- **Stateless**: Their behavior depends only on their input data and configuration
- **Configurable**: Through a standardized ``Settings`` interface
- **Factory-constructed**: Created through factory methods for flexibility and extensibility
- **Consistent**: Follow a uniform interface pattern
- **Interoperable**: Provide unified interfaces to both native implementations and third-party packages

QDK/Chemistry includes the following algorithm classes:

- :doc:`ScfSolver <../algorithms/scf_solver>`: Self-consistent field calculations
- :doc:`Localizer <../algorithms/localizer>`: Orbital localization methods
- :doc:`ActiveSpaceSelector <../algorithms/active_space>`: Active space selection methods
- :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`: Hamiltonian construction
- :doc:`MCCalculator <../algorithms/mc_calculator>`: Multi-configuration calculations
- :doc:`DynamicalCorrelation <../algorithms/dynamical_correlation>`: Dynamical correlation methods

Each algorithm class can leverage both Microsoft-developed implementations (developed within QDK/Chemistry) and
:doc:`interfaces <interfaces>` to established third-party electronic structure packages. This design allows users to
benefit from specialized capabilities of external software while maintaining a consistent API.

Factory Pattern
---------------

QDK/Chemistry implements the :doc:`factory pattern <factory_pattern>` for algorithm creation:

.. tab:: C++ API

   .. code-block:: cpp

      auto scf = ScfSolverFactory::create();

.. tab:: Python API

   .. code-block:: python

      scf = create_scf_solver()

This pattern allows:

- Runtime selection of the most appropriate implementation
- Extension with new implementations without changing client code
- Centralized management of dependencies and resources

Settings Pattern
----------------

Algorithm configuration is managed through a consistent :doc:`Settings <settings>` interface:

.. tab:: C++ API

   .. code-block:: cpp

      scf_solver->settings().set("max_iterations", 100);

.. tab:: Python API

   .. code-block:: python

      scf_solver.settings().set("max_iterations", 100)

This approach provides:

- Uniform configuration across all algorithms
- Type safety and validation
- Default values with explicit overrides
- Documentation of available options

Data Flow Example
-----------------

A typical workflow in QDK/Chemistry demonstrates the data-algorithm separation:

.. todo::

   TODO (NAB):  David needs to check if these code examples are still correct
   https://dev.azure.com/ms-azurequantum/AzureQuantum/_workitems/edit/41366

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::data;
      using namespace qdk::chemistry::algorithms;

      int main() {
          // Create molecular structure from an XYZ file
          Structure molecule;
          molecule.from_xyz_file("molecule.xyz");

          // Configure and run SCF calculation
          auto scf_solver = ScfSolverFactory::create();
          scf_solver->settings().set("basis_set", "cc-pvdz");
          auto [scf_energy, orbitals] = scf_solver->solve(molecule);

          // Select active space orbitals
          auto active_selector = ActiveSpaceSelectorFactory::create();
          active_selector->settings().set("num_active_orbitals", 6);
          active_selector->settings().set("num_active_electrons", 6);
          auto active_indices = active_selector->select(orbitals);

          // Create Hamiltonian with active space
          auto ham_constructor = HamiltonianConstructorFactory::create();
          ham_constructor->settings().set("active_orbitals", active_indices);
          auto hamiltonian = ham_constructor->run(orbitals);

          // Run multi-configuration calculation
          auto mc_solver = MCCalculatorFactory::create();
          auto [mc_energy, wave_function] = mc_solver->solve(hamiltonian);

          return 0;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk.chemistry.data import Structure
      from qdk.chemistry.algorithms import (
         create_scf_solver, create_active_space_selector, create_hamiltonian_constructor,
         create_mc_calculator
      )

      # Create molecular structure from an XYZ file
      molecule = Structure()
      molecule.from_xyz_file("molecule.structure.xyz")

      # Configure and run SCF calculation
      scf_solver = create_scf_solver()
      scf_solver.settings().set("basis_set", "cc-pvdz")
      scf_energy, orbitals = scf_solver.solve(molecule)

      # Select active space orbitals
      active_selector = create_active_space_selector()
      active_selector.settings().set("num_active_orbitals", 6)
      active_selector.settings().set("num_active_electrons", 6)
      active_indices = active_selector.select(orbitals)

      # Create Hamiltonian with active space
      ham_constructor = create_hamiltonian_constructor()
      ham_constructor.settings().set("active_orbitals", active_indices)
      hamiltonian = ham_constructor.run(orbitals)

      # Run multi-configuration calculation
      mc_solver = create_mc_calculator()
      mc_energy, wave_function = mc_solver.solve(hamiltonian)

Interface Architecture
----------------------

QDK/Chemistry is designed with a plugin architecture that allows for consistent
:doc:`interfaces <interfaces>` to various external packages:

.. graphviz:: /_static/diagrams/plugin_architecture.dot

This design provides several advantages:

1. **Unified API**: Users interact with a consistent interface regardless of the underlying implementation
2. **Implementation Flexibility**: Algorithms can be implemented natively or delegate to specialized external packages
3. **Best-of-Breed Approach**: Leverage strengths of different packages while maintaining consistent data structures
4. **Future-Proofing**: New implementations can be added without changing the user-facing API

Benefits of This Design
-----------------------

1. **Maintainability**: Clear separation of concerns makes the code easier to understand and modify
2. **Testability**: Data objects and algorithms can be tested independently
3. **Extensibility**: New algorithms can be added without changing existing code
4. **Composability**: Algorithms can be combined into complex workflows
5. **Consistency**: Uniform interfaces make the library easier to learn and use
6. **Interoperability**: Seamless integration with established quantum chemistry packages

Related Topics
--------------

- :doc:`Factory Pattern <factory_pattern>`: Details on QDK/Chemistry's implementation of the factory pattern
- :doc:`Settings <settings>`: How to configure algorithms through the Settings interface
- :doc:`Serialization <serialization>`: Data serialization and deserialization
- :doc:`Interfaces <interfaces>`: QDK/Chemistry's interface system to external packages
