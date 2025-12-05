.. _extending:

Extending QDK/Chemistry
=======================

QDK/Chemistry uses a plugin system to support multiple implementations of each algorithm type.
This allows switching between native implementations and third-party packages like PySCF without modifying application code.

.. _plugin-system:

Plugin system
-------------

.. _algorithm-plugin-relationship:

Architecture
~~~~~~~~~~~~

Each algorithm in QDK/Chemistry—:doc:`ScfSolver <algorithms/scf_solver>`, :doc:`Localizer <algorithms/localizer>`, :doc:`MCCalculator <algorithms/mc_calculator>`, and others—can have multiple implementations.
All implementations inherit from the same base class and conform to the same interface:

.. code-block:: text

   Algorithm (base)
   ├── ScfSolver
   │   ├── QdkScfSolver      (native)
   │   └── PyscfScfSolver    (PySCF)
   ├── OrbitalLocalizer
   │   ├── QdkLocalizer      (native)
   │   └── PyscfLocalizer    (PySCF)
   └── MCCalculator
       ├── MacisCalculator   (native MACIS)
       └── PyscfMCCalculator (PySCF)

.. _plugin-architecture:

The :doc:`factory pattern <algorithms/factory_pattern>` manages instantiation.
A call to ``create_scf_solver("pyscf")`` looks up "pyscf" in the factory registry, instantiates the corresponding class, and returns it as an ``ScfSolver``.
Application code interacts with the abstract interface while the plugin system manages the concrete implementation.

.. graphviz:: /_static/diagrams/interface_architecture.dot

This design supports several workflows:

- Benchmarking native implementations against established packages
- Mixing backends (e.g., PySCF for SCF, MACIS for multi-configurational methods)
- Adding custom implementations

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

.. _plugin-specific-settings:

Backend-specific settings
~~~~~~~~~~~~~~~~~~~~~~~~~

Each backend may support additional options beyond the common :doc:`Settings <algorithms/settings>`.
These are typically prefixed with the package name to avoid collisions:

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

.. _available-plugins:

Available implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Third-party plugins may not be available for all algorithm classes.

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Native
     - Third-party
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

Included third-party plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QDK/Chemistry includes plugins for the following packages:

- `PySCF <https://pyscf.org/>`_ — Python-based quantum chemistry
- `Qiskit <https://www.ibm.com/quantum/qiskit>`_ — Quantum computing

These plugins are enabled automatically when the corresponding package is installed.

.. _community-plugins:

Community-developed plugins are also welcome. See :ref:`adding-plugins` for guidance on creating new plugins.

Data conversion
~~~~~~~~~~~~~~~

QDK/Chemistry handles data conversion between its internal format and third-party packages automatically through its :doc:`serialization <data/serialization>` system.
Application code works exclusively with QDK/Chemistry types; each plugin manages the translation internally.

Converted types include :doc:`structures <data/structure>`, :doc:`basis sets <data/basis_set>`, :doc:`orbitals <data/orbitals>`, :doc:`Hamiltonians <data/hamiltonian>`, and calculation results.
For large data such as :term:`ERIs`, conversions are optimized to minimize memory copying.

.. _adding-plugins:

Creating plugins
----------------

QDK/Chemistry can be extended in two ways:

1. Adding a new backend for an existing algorithm type (e.g., wrapping another SCF package)
2. Defining an entirely new algorithm type

.. _adding-implementations:

Adding a backend for an existing algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new backend, inherit from the algorithm's base class, implement the required methods, and register with the factory:

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry.hpp>
      #include "custom_chemistry_package.hpp"

      namespace qdk::chemistry::algorithms {

      class CustomScfSolver : public ScfSolver {
      public:
         std::tuple<double, data::Orbitals> solve(
               const data::Structure& structure) override {
            // Convert to custom format
            auto custom_mol = convert_to_custom_format(structure);

            // Run calculation
            auto result = custom_chemistry::run_scf(
               custom_mol,
               settings().get<std::string>("basis_set"),
               settings().get<std::string>("method"));

            // Convert back
            return {result.energy, convert_from_custom_format(result.orbitals)};
         }

      private:
         custom_chemistry::Molecule convert_to_custom_format(
               const data::Structure& structure);
         data::Orbitals convert_from_custom_format(
               const custom_chemistry::Orbitals& orbitals);
      };

      // Register the implementation
      namespace {
      bool registered = ScfSolverFactory::register_implementation(
         "custom",
         []() { return std::make_unique<CustomScfSolver>(); },
         "Custom Chemistry Package");
      }

      }  // namespace qdk::chemistry::algorithms

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import ScfSolver, register_scf_solver
      import custom_chemistry_package as ccp

      class CustomScfSolver(ScfSolver):
          def solve(self, structure):
              custom_mol = self._convert_structure(structure)
              energy, orbs = ccp.run_scf(
                  custom_mol,
                  basis=self.settings().get("basis_set"),
                  method=self.settings().get("method")
              )
              return energy, self._convert_orbitals(orbs)

      register_scf_solver("custom", CustomScfSolver, "Custom Chemistry Package")

.. _custom-algorithm-types:

Defining a new algorithm type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To define a new algorithm type, create an algorithm class, its corresponding factory, and register both with the system:

.. code-block:: python

    from qdk_chemistry.algorithms.base import AlgorithmFactory, Algorithm
    import qdk_chemistry.algorithms.registry as registry
    import qdk_chemistry.algorithms as algorithms
    from qdk_chemistry.data import Structure

    # Define the algorithm type
    class GeometryOptimizer(Algorithm):
        def type_name(self) -> str:
            return "geometry_optimizer"

    # Create its factory
    class GeometryOptimizerFactory(AlgorithmFactory):
        def algorithm_type_name(self) -> str:
            return "geometry_optimizer"

        def default_algorithm_name(self) -> str:
            return "bfgs"

    # Implement a concrete version
    class BfgsOptimizer(GeometryOptimizer):
        def name(self) -> str:
            return "bfgs"

        def _run_impl(self, structure: Structure) -> Structure:
            # ... optimization logic ...
            return optimized_structure

    # Register (typically in your package's __init__)
    registry.register_factory(GeometryOptimizerFactory())
    algorithms.register(lambda: BfgsOptimizer())

    # Usage
    optimizer = algorithms.create("geometry_optimizer", "bfgs")

For additional details, see the :doc:`factory pattern <algorithms/factory_pattern>` documentation.

Further reading
---------------

- Example files: `C++ <../../_static/examples/cpp/interfaces.cpp>`_ | `Python <../../_static/examples/python/interfaces.py>`_
- :doc:`Factory pattern <algorithms/factory_pattern>`
- :doc:`Settings <algorithms/settings>`
- :doc:`Serialization <data/serialization>`
