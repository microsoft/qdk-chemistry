High-level design
#################

This document outlines the core architectural design principles of QDK/Chemistry, explaining the conceptual framework that guides the library's organization and implementation.
For a complete overview of QDK/Chemistry's documentation, see the :doc:`in-depth documentation index <../index>`.

QDK/Chemistry is designed with a clear separation between **data classes** and **algorithms**.
This design choice enables flexibility, extensibility, and maintainability of the codebase, while providing users with a consistent and intuitive API.

Separation of Data and Algorithms
=================================

QDK/Chemistry follows a design pattern that strictly separates:

1. **Data Classes**: Immutable containers that store and manage quantum chemical data
2. **Algorithm Classes**: Processors that operate on data objects to produce new data objects

This separation follows the principle of single responsibility and creates a clear flow of data through computational workflows.

.. graphviz:: /_static/diagrams/data_flow.dot

|

.. _hl_data_class:

Data classes
------------

Data classes in QDK/Chemistry concretely represent intermediate quantities commonly encountered in quantum applications workflows. These classes are designed to be:

Immutable
   Once created, the core data cannot be modified

Self-contained
   Include all information necessary to represent the underlying quantum chemical quantity

:doc:`Serializable <../data/serialization>`
   Can be easily saved to and loaded from files

Language-agnostic
   Accessible through identical APIs in both C++ and Python

See the :doc:`Data Classes <../data/index>` documentation for further details on the availability and usage of QDK/Chemistry's data classes.

.. _hl_algorithm_class:

Algorithm classes
-----------------

Algorithm classes represent mutations on data, such as the execution of quantum chemical methods and generation of circuit components commonly found in quantum applications workflows.
These classes are designed to be:

Stateless
   Their behavior depends only on their input data and configuration

Configurable
   Through a standardized ``Settings`` interface

Conforming
   Exposing a common interface for disparate implementations to enable a unified user experience.

Extensible
   Allowing new implementations to be added without modifying existing code

Programatically, Algorithms are specified as abstract interfaces which can be specialized downstream through concrete implementations.
This allows QDK/Chemistry to be expressed as a :doc:`plugin architecture <../plugins>`, for which algorithm implementations may be specified either natively within QDK/Chemistry or through established third-party quantum chemistry packages:

.. graphviz:: /_static/diagrams/plugin_architecture.dot

|

This design allows users to benefit from specialized capabilities of "best-in-breed" software while maintaining a consistent user experience.
See the :doc:`Plugin System <../plugins>` documentation for further details on how to contribute new algorithm implementations.

Further details on the availablity and usage of QDK/Chemistry's algorithm implementations can be found in the :doc:`Algorithms <../algorithms/index>` documentation.

.. _hl_factory_pattertn:

Factory pattern
===============

QDK/Chemistry's :doc:`plugin architecture <../plugins>` leverages a :doc:`factory pattern <../algorithms/factory_pattern>` for algorithm creation:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/design_principles.cpp
      :language: cpp
      :start-after: // start-cell-scf-create
      :end-before: // end-cell-scf-create

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/design_principles.py
      :language: python
      :start-after: # start-cell-scf-create
      :end-before: # end-cell-scf-create

This pattern allows:

- Extension of workflows by new Algorithm implementations without changing client code
- Centralized management of dependencies and resources

Read more on QDK/Chemistry's usage of this pattern in the :doc:`Factory Pattern <../algorithms/factory_pattern>` documentation.

.. _hl_settings:

Runtime configuration with settings
===================================

Algorithm configuration is managed through instances of :doc:`Settings <../algorithms/settings>` objects, which contain a type-safe data store of configuration parameters consistent between the python and C++ APIs:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/design_principles.cpp
      :language: cpp
      :start-after: // start-cell-scf-settings
      :end-before: // end-cell-scf-settings

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/design_principles.py
      :language: python
      :start-after: # start-cell-scf-settings
      :end-before: # end-cell-scf-settings

Read more on how one can configure, discover, and extend instances of Settings objects in the
:doc:`Settings <../algorithms/settings>` documentation.

.. _hl_dataflow_example:

A complete dataflow example
===========================

A typical workflow in QDK/Chemistry demonstrates the data-algorithm separation:

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/design_principles.cpp
      :language: cpp
      :start-after: // start-cell-data-flow
      :end-before: // end-cell-data-flow

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/design_principles.py
      :language: python
      :start-after: # start-cell-data-flow
      :end-before: # end-cell-data-flow

Further reading
===============

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/design_principles.cpp>`_ and `Python <../../../_static/examples/python/design_principles.py>`_ scripts.
- :doc:`Factory Pattern <../algorithms/factory_pattern>`: Details on QDK/Chemistry's implementation of the factory pattern
- :doc:`Settings <../algorithms/settings>`: How to configure the execution behavior of algorithms through the Settings interface
- :ref:`Plugin system <plugin-system>`: QDK/Chemistry's plugin system for extending functionality
