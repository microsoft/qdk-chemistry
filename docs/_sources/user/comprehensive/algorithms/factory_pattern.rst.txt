Factory pattern
===============

QDK/Chemistry extensively uses the Factory pattern, a creational design pattern that provides an interface for creating objects without specifying their concrete classes.
This document explains how and why QDK/Chemistry uses this pattern for :doc:`algorithm <index>` instantiation.

Overview of the factory pattern
-------------------------------

The factory pattern is a design pattern that encapsulates object creation logic.
Instead of directly instantiating objects using constructors, clients request objects from a factory, typically by specifying a type identifier.
This approach has several advantages:

Abstraction
  Clients work with abstract interfaces rather than concrete implementations
Flexibility
  The concrete implementation can be changed without affecting client code
Configuration
  Objects can be configured based on runtime parameters
Extension
  New implementations can be added without modifying existing code

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

   .. literalinclude:: ../../../_static/examples/cpp/factory_pattern.cpp
      :language: cpp
      :start-after: // start-cell-list-algorithms
      :end-before: // end-cell-list-algorithms

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/factory_pattern.py
      :language: python
      :start-after: # start-cell-list-algorithms
      :end-before: # end-cell-list-algorithms

Inspecting settings
~~~~~~~~~~~~~~~~~~~

Each algorithm implementation has configurable settings.
You can discover available settings programmatically as shown below.
For comprehensive documentation on working with settings, see :doc:`settings`.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/factory_pattern.cpp
      :language: cpp
      :start-after: // start-cell-inspect-settings
      :end-before: // end-cell-inspect-settings

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/factory_pattern.py
      :language: python
      :start-after: # start-cell-inspect-settings
      :end-before: # end-cell-inspect-settings

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
