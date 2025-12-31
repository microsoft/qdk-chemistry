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
- Mixing backends (e.g., PySCF for :term:`SCF`, :term:`MACIS` for multi-configurational methods)
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

This section demonstrates how to integrate an external :term:`SCF` solver as a QDK/Chemistry plugin, enabling access through the standard API.

.. rubric:: Interface requirements

Each algorithm type in QDK/Chemistry defines an abstract base class specifying the interface that all implementations must satisfy:

- A ``name()`` method that returns a unique identifier for the implementation
- A ``_run_impl()`` method containing the computational logic
- A ``settings()`` object for runtime configuration

.. rubric:: Defining custom settings

When an implementation requires configuration options beyond those provided by the base settings class, a derived settings class can be defined:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-custom-settings
      :end-before: // end-cell-custom-settings

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-custom-settings
      :end-before: # end-cell-custom-settings

.. rubric:: Implementation structure

The implementation class inherits from the algorithm base class and overrides the required methods.
The ``_run_impl()`` method is responsible for:

1. Converting QDK/Chemistry data structures to the external package's format
2. Invoking the external computation
3. Converting results back to QDK/Chemistry data structures

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-custom-scf-solver
      :end-before: // end-cell-custom-scf-solver

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-custom-scf-solver
      :end-before: # end-cell-custom-scf-solver

.. rubric:: Registration

Implementations are registered with the algorithm factory to enable discovery and instantiation by name.
Registration is typically performed during module initialization:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-registration
      :end-before: // end-cell-registration

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-registration
      :end-before: # end-cell-registration

Following registration, the implementation is accessible through the standard API:

.. literalinclude:: ../../_static/examples/python/custom_plugin.py
   :language: python
   :start-after: # start-cell-usage-after-registration
   :end-before: # end-cell-usage-after-registration

.. _custom-algorithm-types:

Defining a new algorithm type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the required functionality does not correspond to an existing algorithm category, a new algorithm type can be defined.
This section demonstrates the complete process using a geometry optimizer as an example.

.. rubric:: Interface design

The first step is to specify the algorithm's interface:

Input type
   The data the algorithm operates on (e.g., ``Structure``)
Output type
   The data the algorithm produces (e.g., optimized ``Structure``)
Configuration
   Required settings (e.g., convergence thresholds, iteration limits)

.. rubric:: Settings class definition

Define a settings class containing all configuration parameters:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-geometry-settings
      :end-before: // end-cell-geometry-settings

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-geometry-settings
      :end-before: # end-cell-geometry-settings

.. rubric:: Base class definition

Define an abstract base class specifying the interface for all implementations:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-geometry-base-class
      :end-before: // end-cell-geometry-base-class

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-geometry-base-class
      :end-before: # end-cell-geometry-base-class

.. rubric:: Factory definition

The factory manages implementation registration and provides instance creation:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-geometry-factory
      :end-before: // end-cell-geometry-factory

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-geometry-factory
      :end-before: # end-cell-geometry-factory

.. rubric:: Concrete implementations

Implement the algorithm by inheriting from the base class:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-geometry-implementations
      :end-before: // end-cell-geometry-implementations

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-geometry-implementations
      :end-before: # end-cell-geometry-implementations

Additional implementations follow the same pattern:

.. literalinclude:: ../../_static/examples/python/custom_plugin.py
   :language: python
   :start-after: # start-cell-steepest-descent
   :end-before: # end-cell-steepest-descent

.. rubric:: Registration

Register the factory and all implementations:

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-geometry-registration
      :end-before: // end-cell-geometry-registration

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-geometry-registration
      :end-before: # end-cell-geometry-registration

.. rubric:: Usage

Following registration, the new algorithm type is accessible through the standard API:

.. literalinclude:: ../../_static/examples/python/custom_plugin.py
   :language: python
   :start-after: # start-cell-geometry-usage
   :end-before: # end-cell-geometry-usage

For additional information on the factory pattern and settings system, refer to the
:doc:`factory pattern <algorithms/factory_pattern>` and :doc:`settings <algorithms/settings>` documentation.


Further reading
---------------

- Custom plugin examples: `C++ source <../../_static/examples/cpp/custom_plugin.cpp>`__ | `Python source <../../_static/examples/python/custom_plugin.py>`__
- Plugin usage examples: `C++ example <../../_static/examples/cpp/interfaces.cpp>`__ | `Python example <../../_static/examples/python/interfaces.py>`__
- :doc:`Factory pattern <algorithms/factory_pattern>`
- :doc:`Settings <algorithms/settings>`
- :doc:`Serialization <data/serialization>`
