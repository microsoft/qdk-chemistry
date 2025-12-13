Factory pattern
===============

QDK/Chemistry extensively uses the Factory pattern, a creational design pattern that provides an interface for creating objects without specifying their concrete classes.
This document explains how and why QDK/Chemistry uses this pattern for :doc:`algorithm <index>` instantiation.

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

In QDK/Chemistry, :doc:`algorithm <index>` classes are instantiated through factory classes rather than direct constructors.
This design allows QDK/Chemistry to:

- Support multiple implementations of the same algorithm interface
- Configure algorithm instances based on settings
- Load algorithm implementations dynamically at runtime
- Isolate algorithm implementation details from client code

Factory classes in QDK/Chemistry
--------------------------------

QDK/Chemistry provides factory infrastructure for each algorithm type.
In Python, algorithm instantiation is managed through a centralized registry module rather than individual factory classes.

.. list-table:: QDK/Chemistry Algorithm Factories
   :header-rows: 1
   :widths: auto

   * - Algorithm
     - Algorithm Type (Python)
     - Factory Class (C++)
   * - :doc:`ScfSolver <../algorithms/scf_solver>`
     - ``"scf_solver"``
     - ``ScfSolverFactory``
   * - :doc:`Localizer <../algorithms/localizer>`
     - ``"orbital_localizer"``
     - ``LocalizerFactory``
   * - :doc:`ActiveSpaceSelector <../algorithms/active_space>`
     - ``"active_space_selector"``
     - ``ActiveSpaceSelectorFactory``
   * - :doc:`HamiltonianConstructor <../algorithms/hamiltonian_constructor>`
     - ``"hamiltonian_constructor"``
     - ``HamiltonianConstructorFactory``
   * - :doc:`MCCalculator <../algorithms/mc_calculator>`
     - ``"multi_configuration_calculator"``
     - ``MultiConfigurationCalculatorFactory``
   * - :doc:`MultiConfigurationScf <../algorithms/mcscf>`
     - ``"multi_configuration_scf"``
     - ``MultiConfigurationScfFactory``
   * - :doc:`ProjectedMultiConfigurationCalculator <../algorithms/pmc>`
     - ``"projected_multi_configuration_calculator"``
     - ``ProjectedMultiConfigurationCalculatorFactory``
   * - :doc:`DynamicalCorrelationCalculator <../algorithms/dynamical_correlation>`
     - ``"dynamical_correlation_calculator"``
     - ``DynamicalCorrelationCalculatorFactory``
   * - :doc:`StabilityChecker <../algorithms/stability_checker>`
     - ``"stability_checker"``
     - ``StabilityCheckerFactory``
   * - :doc:`EnergyEstimator <../algorithms/energy_estimator>`
     - ``"energy_estimator"``
     - Python only
   * - :doc:`StatePreparation <../algorithms/state_preparation>`
     - ``"state_prep"``
     - Python only
   * - :doc:`QubitMapper <../algorithms/qubit_mapper>`
     - ``"qubit_mapper"``
     - Python only


Using factories
---------------

To create an algorithm instance, call the appropriate factory method with an optional implementation name.
If no name is provided, the default implementation is used.
See :ref:`discovering-implementations` below for how to list available implementations programmatically.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/factory_pattern.cpp
      :language: cpp
      :start-after: // start-cell-scf-localizer
      :end-before: // end-cell-scf-localizer

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/factory_pattern.py
      :language: python
      :start-after: # start-cell-scf-localizer
      :end-before: # end-cell-scf-localizer

.. _discovering-implementations:

Discovering implementations
---------------------------

QDK/Chemistry provides programmatic discovery of available algorithm types and their implementations.
This is useful for exploring what's available at runtime, building dynamic UIs, or debugging plugin loading.

Listing algorithm types and implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab:: C++ API

   .. code-block:: cpp

      #include <iostream>
      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::algorithms;

      // List available SCF solver implementations
      auto scf_methods = ScfSolverFactory::available();
      std::cout << "Available SCF solvers:" << std::endl;
      for (const auto& name : scf_methods) {
          std::cout << "  - " << name << std::endl;
      }

      // List available localizer implementations
      auto localizer_methods = LocalizerFactory::available();
      std::cout << "Available localizers:" << std::endl;
      for (const auto& name : localizer_methods) {
          std::cout << "  - " << name << std::endl;
      }

      // List available Hamiltonian constructor implementations
      auto ham_methods = HamiltonianConstructorFactory::available();
      std::cout << "Available Hamiltonian constructors:" << std::endl;
      for (const auto& name : ham_methods) {
          std::cout << "  - " << name << std::endl;
      }

      // List available multi-configuration calculator implementations
      auto mc_methods = MultiConfigurationCalculatorFactory::available();
      std::cout << "Available MC calculators:" << std::endl;
      for (const auto& name : mc_methods) {
          std::cout << "  - " << name << std::endl;
      }

      // Show default implementation for each factory type
      std::cout << "Default SCF solver: " << ScfSolverFactory::default_name() << std::endl;
      std::cout << "Default localizer: " << LocalizerFactory::default_name() << std::endl;
      std::cout << "Default Hamiltonian constructor: " << HamiltonianConstructorFactory::default_name() << std::endl;
      std::cout << "Default MC calculator: " << MultiConfigurationCalculatorFactory::default_name() << std::endl;

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import registry

      # List all algorithm types and their implementations
      all_algorithms = registry.available()
      print(all_algorithms)
      # Output: {'scf_solver': ['qdk', 'pyscf'], 'orbital_localizer': ['qdk_pipek_mezey', 'pyscf'], ...}

      # List implementations for a specific algorithm type
      scf_methods = registry.available("scf_solver")
      print("Available SCF solvers:", scf_methods)
      # Output: ['qdk', 'pyscf']

      localizer_methods = registry.available("orbital_localizer")
      print("Available localizers:", localizer_methods)
      # Output: ['qdk_pipek_mezey', 'pyscf', ...]

      # Show default implementations for each algorithm type
      defaults = registry.show_default()
      print("Defaults:", defaults)
      # Output: {'scf_solver': 'qdk', 'orbital_localizer': 'qdk_pipek_mezey', ...}

Inspecting settings
~~~~~~~~~~~~~~~~~~~

Each algorithm implementation has configurable settings.
You can discover available settings programmatically as shown below.
For comprehensive documentation on working with settings, see :doc:`settings`.

.. tab:: C++ API

   .. code-block:: cpp

      #include <iostream>
      #include <qdk/chemistry.hpp>

      using namespace qdk::chemistry::algorithms;

      // Create a SCF solver and inspect its settings
      auto scf = ScfSolverFactory::create("qdk");

      // Print settings as a formatted table
      std::cout << scf->settings().as_table() << std::endl;

      // Or iterate over individual settings
      for (const auto& key : scf->settings().keys()) {
          std::cout << key << ": " << scf->settings().get_as_string(key) << std::endl;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import registry

      # Create a SCF solver and inspect its settings
      scf = registry.create("scf_solver", "qdk")

      # Print settings as a formatted table
      registry.print_settings("scf_solver", "qdk")

      # Or iterate over individual settings
      for key in scf.settings().keys():
          print(f"{key}: {scf.settings().get(key)}")

Connection to the plugin system
-------------------------------

The factory pattern serves as the foundation for QDK/Chemistry's :doc:`plugin system <../plugins>`.
Factories enable the registration and instantiation of plugin implementations that connect to external quantum chemistry programs.

Internally, QDK/Chemistry's factories maintain a registry of creator functions mapped to implementation names.
When a client requests an implementation by name, the factory looks up the appropriate creator function and instantiates the object with the necessary setup.

This design enables several key capabilities:

- Seamless integration with external quantum chemistry packages
- Runtime selection of specific implementations
- Decoupling of plugin usage from implementation details

For detailed information about implementing custom plugins see the :doc:`plugin documentation <../plugins>`.

Further reading
---------------

- Factory usage examples: `C++ <../../../_static/examples/cpp/factory_pattern.cpp>`_ | `Python <../../../_static/examples/python/factory_pattern.py>`_
- :doc:`Settings <settings>`: Configuration of algorithm instances
- :doc:`Plugins <../plugins>`: Extending QDK/Chemistry with custom implementations
