.. _plugins:

Plugins
=======

QDK/Chemistry uses a plugin system to support multiple implementations of each of the available :doc:`algorithm <algorithms/index>` type.
This allows switching between native QDK implementations and third-party packages (e.g., PySCF, Qiskit) without modifying application code.

.. _plugin-system:

Plugin system
-------------

.. _algorithm-plugin-relationship:

Architecture
~~~~~~~~~~~~

Each :doc:`algorithm <algorithms/index>` in QDK/Chemistry can have multiple implementations.
All implementations inherit from the same base class and conform to the same interface:

.. graphviz:: /_static/diagrams/interface_architecture.dot

This design supports several workflows:

- Benchmarking native implementations against established packages
- Mixing backends (e.g., PySCF for SCF, MACIS for multi-configurational methods)
- Adding custom implementations

The implementations for each algorithm type are managed by a :doc:`factory class <algorithms/factory_pattern>`, which provides a consistent interface for creating instances and listing available implementations.
We refer the reader to the :doc:`factory pattern <algorithms/factory_pattern>` and :doc:`algorithm <algorithms/index>` documentation pages for more details on this design pattern.





Using plugins
~~~~~~~~~~~~~

To select an implementation, specify it by name:

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

To list available implementations:

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

Documentation pertaining to the availability and configuration of each algorithm implementation provided within QDK/Chemistry can be found on the :doc:`algorithm <algorithms/index>` documentation pages.



Included third-party plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the native implementations packaged within QDK/Chemistry, plugins are included for the following packages:

- `PySCF <https://pyscf.org/>`_ — Python-based quantum chemistry
- `Qiskit <https://www.ibm.com/quantum/qiskit>`_ — Quantum computing

These plugins are enabled automatically when the corresponding package is installed.

.. _community-plugins:

Community-developed plugins are also welcome. See :ref:`adding-plugins` for guidance on creating new plugins.

.. _adding-plugins:

Creating plugins
----------------

QDK/Chemistry supports two extension mechanisms:

1. Implementing a new backend for an existing algorithm type (e.g., integrating an external quantum chemistry package)
2. Defining an entirely new algorithm type with its own factory and implementations

The following sections provide comprehensive examples of each approach.

.. _adding-implementations:

Implementing a new algorithm backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section demonstrates how to integrate an external SCF solver as a QDK/Chemistry plugin, enabling access through the standard API.

**Interface requirements**

Each algorithm type in QDK/Chemistry defines an abstract base class specifying the interface that all implementations must satisfy:

- A ``name()`` method that returns a unique identifier for the implementation
- A ``_run_impl()`` method containing the computational logic
- A ``settings()`` object for runtime configuration

**Defining custom settings**

When an implementation requires configuration options beyond those provided by the base settings class, a derived settings class can be defined:

.. tab:: C++ API

   .. code-block:: cpp

      class CustomScfSettings : public qdk::chemistry::algorithms::ElectronicStructureSettings {
       public:
        CustomScfSettings() : ElectronicStructureSettings() {
          // Define additional settings beyond the inherited defaults
          set_default("custom_option", "default_value");
        }
      };

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.data import ElectronicStructureSettings

      class CustomScfSettings(ElectronicStructureSettings):
          def __init__(self):
              super().__init__()
              # Define additional settings beyond the inherited defaults
              self._set_default("custom_option", "string", "default_value",
                               "Description of the custom option")

**Implementation structure**

The implementation class inherits from the algorithm base class and overrides the required methods.
The ``_run_impl()`` method is responsible for:

1. Converting QDK/Chemistry data structures to the external package's format
2. Invoking the external computation
3. Converting results back to QDK/Chemistry data structures

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry/algorithms/scf.hpp>
      #include "external_chemistry_package.hpp"

      class CustomScfSolver : public qdk::chemistry::algorithms::ScfSolver {
       public:
        CustomScfSolver() {
          _settings = std::make_unique<CustomScfSettings>();
        }

        std::string name() const override { return "custom"; }

       protected:
        std::pair<double, std::shared_ptr<qdk::chemistry::data::Wavefunction>>
        _run_impl(std::shared_ptr<qdk::chemistry::data::Structure> structure,
                  int charge, int spin_multiplicity,
                  std::optional<std::shared_ptr<qdk::chemistry::data::Orbitals>> initial_guess) override {

          // Convert to external format
          auto external_mol = convert_to_external_format(structure);

          // Execute external calculation
          auto basis = _settings->get<std::string>("basis_set");
          auto [energy, external_orbitals] = external_package::run_scf(external_mol, basis);

          // Convert results to QDK format
          auto wavefunction = convert_to_qdk_wavefunction(external_orbitals);

          return {energy, wavefunction};
        }
      };

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import ScfSolver
      from qdk_chemistry.data import Structure, Wavefunction, Orbitals

      class CustomScfSolver(ScfSolver):
          def __init__(self):
              super().__init__()
              self._settings = CustomScfSettings()

          def name(self) -> str:
              return "custom"

          def _run_impl(
              self,
              structure: Structure,
              charge: int,
              spin_multiplicity: int,
              initial_guess: Orbitals | None = None,
          ) -> tuple[float, Wavefunction]:
              # Convert to external format
              # external_mol = external_package.Molecule(structure.positions, structure.elements)

              # Execute external calculation
              basis_set = self.settings().get("basis_set")
              # energy, external_orbs = external_package.run_scf(external_mol, basis=basis_set)

              # Convert results to QDK format
              # wavefunction = self._convert_to_wavefunction(external_orbs)

              return energy, wavefunction

**Registration**

Implementations are registered with the algorithm factory to enable discovery and instantiation by name.
Registration is typically performed during module initialization:

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry/algorithms/scf.hpp>

      // Static registration during library initialization
      static auto registration = qdk::chemistry::algorithms::ScfSolver::register_implementation(
          []() { return std::make_unique<CustomScfSolver>(); }
      );

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms.registry import register

      # Registration during module import
      register(lambda: CustomScfSolver())

Following registration, the implementation is accessible through the standard API:

.. code-block:: python

   from qdk_chemistry.algorithms import ScfSolver

   # Instantiate the custom solver
   solver = ScfSolver.create("custom")
   solver.settings()["basis_set"] = "cc-pvdz"
   energy, wavefunction = solver.run(molecule, charge=0, spin_multiplicity=1)

   # Verify registration
   print(ScfSolver.available())  # [..., 'custom']

.. _custom-algorithm-types:

Defining a new algorithm type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the required functionality does not correspond to an existing algorithm category, a new algorithm type can be defined.
This section demonstrates the complete process using a geometry optimizer as an example.

**Interface design**

The first step is to specify the algorithm's interface:

- **Input type**: The data the algorithm operates on (e.g., ``Structure``)
- **Output type**: The data the algorithm produces (e.g., optimized ``Structure``)
- **Configuration**: Required settings (e.g., convergence thresholds, iteration limits)

**Settings class definition**

Define a settings class containing all configuration parameters:

.. tab:: C++ API

   .. code-block:: cpp

      class GeometryOptimizerSettings : public qdk::chemistry::data::Settings {
       public:
        GeometryOptimizerSettings() {
          set_default<int64_t>("max_steps", 100, "Maximum optimization steps",
                               qdk::chemistry::data::BoundConstraint<int64_t>{1, 10000});
          set_default<double>("convergence_threshold", 1e-5,
                              "Gradient convergence threshold");
          set_default<double>("step_size", 0.1, "Initial optimization step size");
        }
      };

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.data import Settings

      class GeometryOptimizerSettings(Settings):
          def __init__(self):
              super().__init__()
              self._set_default("max_steps", "int", 100,
                               "Maximum optimization steps", (1, 10000))
              self._set_default("convergence_threshold", "double", 1e-5,
                               "Gradient convergence threshold")
              self._set_default("step_size", "double", 0.1,
                               "Initial optimization step size")

**Base class definition**

Define an abstract base class specifying the interface for all implementations:

.. tab:: C++ API

   .. code-block:: cpp

      class GeometryOptimizer
          : public qdk::chemistry::algorithms::Algorithm<
                GeometryOptimizer,
                std::shared_ptr<qdk::chemistry::data::Structure>,  // Return type
                std::shared_ptr<qdk::chemistry::data::Structure>>  // Input type
      {
       public:
        static std::string type_name() { return "geometry_optimizer"; }
      };

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms.base import Algorithm

      class GeometryOptimizer(Algorithm):
          """Abstract base class for geometry optimization algorithms."""

          def type_name(self) -> str:
              return "geometry_optimizer"

**Factory definition**

The factory manages implementation registration and provides instance creation:

.. tab:: C++ API

   .. code-block:: cpp

      // The Algorithm base class template provides factory functionality automatically

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms.base import AlgorithmFactory

      class GeometryOptimizerFactory(AlgorithmFactory):
          def algorithm_type_name(self) -> str:
              return "geometry_optimizer"

          def default_algorithm_name(self) -> str:
              return "bfgs"

**Concrete implementations**

Implement the algorithm by inheriting from the base class:

.. tab:: C++ API

   .. code-block:: cpp

      class BfgsOptimizer : public GeometryOptimizer {
       public:
        BfgsOptimizer() {
          _settings = std::make_unique<GeometryOptimizerSettings>();
        }

        std::string name() const override { return "bfgs"; }

       protected:
        std::shared_ptr<qdk::chemistry::data::Structure>
        _run_impl(std::shared_ptr<qdk::chemistry::data::Structure> structure) override {
          auto max_steps = _settings->get<int64_t>("max_steps");
          auto threshold = _settings->get<double>("convergence_threshold");

          // BFGS optimization implementation
          return optimized_structure;
        }
      };

.. tab:: Python API

   .. code-block:: python

      class BfgsOptimizer(GeometryOptimizer):
          """BFGS quasi-Newton geometry optimizer."""

          def __init__(self):
              super().__init__()
              self._settings = GeometryOptimizerSettings()

          def name(self) -> str:
              return "bfgs"

          def _run_impl(self, structure: Structure) -> Structure:
              max_steps = self.settings().get("max_steps")
              threshold = self.settings().get("convergence_threshold")

              # BFGS optimization implementation
              return optimized_structure

Additional implementations follow the same pattern:

.. code-block:: python

   class SteepestDescentOptimizer(GeometryOptimizer):
       """Steepest descent geometry optimizer."""

       def __init__(self):
           super().__init__()
           self._settings = GeometryOptimizerSettings()

       def name(self) -> str:
           return "steepest_descent"

       def _run_impl(self, structure: Structure) -> Structure:
           # Steepest descent implementation
           return optimized_structure

**Registration**

Register the factory and all implementations:

.. tab:: C++ API

   .. code-block:: cpp

      // During library initialization
      static auto factory_reg = register_factory<GeometryOptimizer>();
      static auto bfgs_reg = GeometryOptimizer::register_implementation(
          []() { return std::make_unique<BfgsOptimizer>(); }
      );

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import registry

      # Register the factory
      registry.register_factory(GeometryOptimizerFactory())

      # Register implementations
      registry.register(lambda: BfgsOptimizer())
      registry.register(lambda: SteepestDescentOptimizer())

**Usage**

Following registration, the new algorithm type is accessible through the standard API:

.. code-block:: python

   from qdk_chemistry.algorithms import create, available

   # List available implementations
   print(available("geometry_optimizer"))  # ['bfgs', 'steepest_descent']

   # Instantiate and configure
   optimizer = create("geometry_optimizer", "bfgs")
   optimizer.settings().set("max_steps", 200)
   optimizer.settings().set("convergence_threshold", 1e-6)

   # Execute
   optimized_structure = optimizer.run(initial_structure)

For additional information on the factory pattern and settings system, refer to the
:doc:`factory pattern <algorithms/factory_pattern>` and :doc:`settings <algorithms/settings>` documentation.


Further reading
---------------

- Custom plugin examples: `C++ source <../../_static/examples/cpp/custom_plugin.cpp>`__ | `Python source <../../_static/examples/python/custom_plugin.py>`__
- Plugin usage examples: `C++ example <../../_static/examples/cpp/interfaces.cpp>`__ | `Python example <../../_static/examples/python/interfaces.py>`__
- :doc:`Factory pattern <algorithms/factory_pattern>`
- :doc:`Settings <algorithms/settings>`
- :doc:`Serialization <data/serialization>`
