Design Principles of QDK/Chemistry
##################################

This document outlines the core architectural design principles of QDK/Chemistry, explaining the conceptual framework that guides the library's organization and implementation.
For a complete overview of QDK/Chemistry's documentation, see the :doc:`in-depth documentation index <../index>`.

QDK/Chemistry is designed with a clear separation between **data containers** and **algorithms**.
This design choice enables flexibility, extensibility, and maintainability of the codebase, while providing users with a consistent and intuitive API.

Separation of data and algorithms
=================================

QDK/Chemistry follows a design pattern that strictly separates:

1. **Data Classes**: Immutable containers that store and manage quantum chemical data
2. **Algorithm Classes**: Processors that operate on data objects to produce new data objects

This separation follows the principle of single responsibility and creates a clear flow of data through computational
workflows.

.. graphviz:: /_static/diagrams/data_flow.dot

|

Data classes
------------

Data classes in QDK/Chemistry represent quantities that represent intermediate data commonly encounters in quantum applications workflows. These classes are designed to be:

- **Immutable**: Once created, the core data cannot be modified
- **Self-contained**: Include all information necessary to represent the underlying quantum chemical quantity
- :doc:`Serializable <../data/serialization>`: Can be easily saved to and loaded from files
- **Language-agnostic**: Accessible through identical APIs in both C++ and Python

See the :doc:`Data <../data/index>` documentation for further details on the availability and usage of
QDK/Chemistry's data classes.

Algorithm classes
-----------------

Algorithm classes represent mutations on data, such as the execution of quantum chemical methods and generation of circuit components commonly found in quantum applications workflows.
These classes are designed to be:

- **Stateless**: Their behavior depends only on their input data and configuration
- **Configurable**: Through a standardized ``Settings`` interface
- **Consistent**: Follow a uniform interface pattern
- **Interoperable**: Provide unified interfaces to both native implementations and third-party packages


Each algorithm class can leverage both QDK-developed implementations and :doc:`interfaces <interfaces>` to established third-party electronic structure packages.
This design allows users to benefit from specialized capabilities of external software while maintaining a consistent API.

See the :doc:`Algorithms <../algorithms/index>` documentation for further details on the availablity and usage of
 QDK/Chemistry's algorithm implementations.


Factory pattern
===============

QDK/Chemistry leverages a :doc:`factory pattern <../algorithms/factory_pattern>` for algorithm creation:

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

- Runtime selection of the most appropriate implementation without changing calls to factory functions
- Extension by new implementations without changing client code
- Centralized management of dependencies and resources

Read more on QDK/Chemistry's usage of this pattern in the :doc:`Factory Pattern <../algorithms/factory_pattern>` documentation.

Runtime Algororithm Configuration via Settings
==============================================

Algorithm configuration is managed through instances of :doc:`Settings <../settings>` objects:

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

Read more on how one can configure, discover and extend instances of Settings objects in the
:doc:`Settings <../settings>` documentation.

A Complete Dataflow Example
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

Interface architecture (plugins)
================================

QDK/Chemistry is designed with a plugin architecture that allows for consistent :doc:`interfaces <interfaces>` to various external packages
as well as methods developed within QDK/Chemistry itself:

.. graphviz:: /_static/diagrams/plugin_architecture.dot

This design provides several advantages:

1. **Unified API**: Users interact with a consistent interface regardless of the underlying implementation
2. **Implementation Flexibility**: Algorithms can be implemented natively or delegate to specialized external packages
3. **Best-of-Breed Approach**: Leverage strengths of different packages while maintaining consistent data structures
4. **Resilient**: New implementations can be added without changing the user-facing API

Further reading
===============

- The above examples can be downloaded as complete `C++ <../../../_static/examples/cpp/design_principles.cpp>`_ and `Python <../../../_static/examples/python/design_principles.py>`_ scripts.
- :doc:`Factory Pattern <../algorithms/../algorithms/factory_pattern>`: Details on QDK/Chemistry's implementation of the factory pattern
- :doc:`Settings <../settings>`: How to configure the execution behavior of algorithms through the Settings interface
- :doc:`Interfaces <interfaces>`: QDK/Chemistry's interface system to external packages

.. toctree::
   :maxdepth: 1
   :hidden:

   ../algorithms/factory_pattern
   interfaces
