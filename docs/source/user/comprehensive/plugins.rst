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

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/interfaces.py
      :language: python
      :start-after: # start-cell-scf
      :end-before: # end-cell-scf

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/interfaces.cpp
      :language: cpp
      :start-after: // start-cell-scf
      :end-before: // end-cell-scf

.. _listing-implementations:

To list available implementations:

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/interfaces.py
      :language: python
      :start-after: # start-cell-list-methods
      :end-before: # end-cell-list-methods

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/interfaces.cpp
      :language: cpp
      :start-after: // start-cell-list-methods
      :end-before: // end-cell-list-methods

Documentation pertaining to the availability and configuration of each algorithm implementation provided within QDK/Chemistry can be found on the :doc:`algorithm <algorithms/index>` documentation pages.



Included third-party plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the native implementations packaged within QDK/Chemistry, plugins are included for the following packages:

- `PySCF <https://pyscf.org/>`_ — Python-based quantum chemistry
- `Qiskit <https://www.ibm.com/quantum/qiskit>`_ — Quantum algorithm primitives
- `OpenFermion <https://quantumai.google/openfermion>`_ — Quantum algorithm primitives
- `geomeTRIC <https://github.com/leeping/geomeTRIC>`_ — Molecular geometry optimization

These plugins are enabled automatically when the corresponding package is installed.

.. _pyscf-plugin-details:

PySCF plugin details
^^^^^^^^^^^^^^^^^^^^

The PySCF plugin is installed via the ``plugins`` extra:

.. code-block:: bash

   pip install 'qdk-chemistry[plugins]'

.. note::

   PySCF is the only package in the ``plugins`` extra and publishes no Windows wheels, so the PySCF
   plugin is unavailable on native Windows. Because ``jupyter``, ``test``, and ``all`` depend on
   ``plugins``, this applies to those extras as well — they install successfully on Windows, but
   without PySCF.

   The native QDK/Chemistry implementations are unaffected and remain available on Windows.
   To use the PySCF plugin on a Windows machine, work inside
   `WSL <https://learn.microsoft.com/windows/wsl/install>`_.

.. _qiskit-plugin-details:

Qiskit plugin details
^^^^^^^^^^^^^^^^^^^^^

The Qiskit plugin uses **opportunistic loading** to maximize compatibility across different installation configurations.
When Qiskit is installed, the plugin will load and register the available algorithms. The optional ecosystem packages (Qiskit Aer and Qiskit Nature) are loaded based on their availability in your environment.

**Loading behavior:**

- **Qiskit (core)**: Loaded when the plugin is initialized and Qiskit is installed.
- **Qiskit Nature**: Loaded if ``qiskit-nature`` is installed.
- **Qiskit Aer**: Loaded if ``qiskit-aer`` is installed.

**Installing optional Qiskit packages:**

To install the optional Qiskit ecosystem packages, use the ``qiskit-extras`` extra when installing QDK/Chemistry:

.. code-block:: bash

   pip install 'qdk-chemistry[qiskit-extras]'

Alternatively, you can install them directly:

.. code-block:: bash

   pip install qiskit-aer qiskit-nature

.. note::

   On Python 3.14, ``qiskit-aer`` is omitted from ``qiskit-extras`` on Linux ARM64
   (aarch64), because Qiskit does not yet publish a Python 3.14 wheel for that
   platform. All other platforms install the full set.

**Checking what is loaded:**

To determine which Qiskit components are available in your environment, you can check the following module-level variables:

.. code-block:: python

   from qdk_chemistry.plugins.qiskit import (
       QDK_CHEMISTRY_HAS_QISKIT,
       QDK_CHEMISTRY_HAS_QISKIT_NATURE,
       QDK_CHEMISTRY_HAS_QISKIT_AER,
   )

   print(f"Qiskit core available: {QDK_CHEMISTRY_HAS_QISKIT}")
   print(f"Qiskit Nature available: {QDK_CHEMISTRY_HAS_QISKIT_NATURE}")
   print(f"Qiskit Aer available: {QDK_CHEMISTRY_HAS_QISKIT_AER}")

These boolean variables are set at module load time and reflect the actual availability of each package in your Python environment.

.. warning::

   If you attempt to use an algorithm that requires an optional Qiskit package that is not installed,
   the algorithm will not be available in the factory. Use the :ref:`listing-implementations` pattern
   to see which implementations are currently available.

.. _openfermion-plugin-details:

OpenFermion plugin details
^^^^^^^^^^^^^^^^^^^^^^^^^^
The OpenFermion plugin integrates QDK/Chemistry with `OpenFermion <https://quantumai.google/openfermion>`_.
Like the Qiskit plugin, it uses **opportunistic loading**: the plugin loads when OpenFermion is installed.

**Loading behavior:**

- **OpenFermion**: Loaded when the plugin is initialized and OpenFermion is installed.

**Installing OpenFermion packages:**

To install OpenFermion, use the ``openfermion-extras`` extra when installing QDK/Chemistry:

.. code-block:: bash

   pip install 'qdk-chemistry[openfermion-extras]'

**Checking what is loaded:**

To determine which OpenFermion components are available in your environment, you can check the following module-level variables:

.. code-block:: python

   from qdk_chemistry.plugins.openfermion import (
       QDK_CHEMISTRY_HAS_OPENFERMION,
   )

   print(f"OpenFermion available: {QDK_CHEMISTRY_HAS_OPENFERMION}")

This boolean variable is set at module load time and reflects the actual availability of each package in your Python environment.

.. warning::

   If you attempt to use an algorithm that requires OpenFermion but the package is not installed,
   the algorithm will not be available in the factory. Use the :ref:`listing-implementations` pattern
   to see which implementations are currently available.

.. _community-plugins:

Community-developed plugins are also welcome. See :ref:`adding-plugins` for guidance on creating new plugins.

.. _adding-plugins:

Creating plugins
----------------

An installed Python package can contribute any combination of:

- Implementations of existing algorithm types
- New algorithm types and their implementations
- :class:`~qdk_chemistry.data.DataClass` types used in algorithm inputs or outputs
- Remote execution backends
- Cache backends

The following sections provide complete remote backend and algorithm examples. The same plugin object can register multiple capabilities through :class:`~qdk_chemistry.plugins.PluginRegistrar`.

Registration names must be unique within their registry. Registering a second algorithm, algorithm type, data class, remote backend, or cache backend under an existing name raises :class:`~qdk_chemistry.plugins.DuplicateRegistrationError`, a :class:`ValueError` subclass. The rejected registration does not replace the existing implementation.

Automatic discovery
~~~~~~~~~~~~~~~~~~~

The plugin contract is a :class:`~qdk_chemistry.plugins.base.QdkChemistryPlugin` subclass exposed through the ``qdk_chemistry.plugins`` entry-point group. A plugin package declares this in its ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."qdk_chemistry.plugins"]
   custom = "custom_package.plugin:CustomScfPlugin"

Importing ``qdk_chemistry`` discovers the installed package and calls its ``register`` method; users do not need to import the plugin module themselves. A plugin registers its capabilities through the supplied registrar:

.. code-block:: python

   from qdk_chemistry.plugins import PluginRegistrar, QdkChemistryPlugin

   class CustomPlugin(QdkChemistryPlugin):
       def register(self, registrar: PluginRegistrar) -> None:
           registrar.register_algorithm(lambda: CustomAlgorithm())
           registrar.register_dataclass(CustomResult)
           registrar.register_remote_backend("custom", CustomRemoteBackend)
           registrar.register_cache_backend("custom", CustomCacheBackend)

Each plugin-defined ``DataClass`` used in algorithm inputs or outputs must declare a non-empty wire-format identifier in its own class body:

.. code-block:: python

   from qdk_chemistry.data import DataClass

   class CustomResult(DataClass):
       @staticmethod
       def data_type_name() -> str:
           return "custom_result"
       ...

The value returned by ``data_type_name()`` identifies the serialized format during remote and cache deserialization. A canonical loader must declare this static method directly and return a non-empty string. A subclass must declare a unique identifier and register as its own loader. Registration raises ``TypeError`` when the declaration is missing or empty, and :class:`~qdk_chemistry.plugins.DuplicateRegistrationError` when another loader already owns the identifier.

Register these classes with :meth:`~qdk_chemistry.plugins.base.PluginRegistrar.register_dataclass` or pass them through the ``data_classes`` argument of :meth:`~qdk_chemistry.plugins.base.PluginRegistrar.register_algorithm`. Python return annotations are not used for discovery.

Remote backend MCP configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remote backends expose no constructor options to MCP clients by default. To allow a client-controlled option, declare ``mcp_safe_config_options`` directly on the concrete backend class. Registration validates that it is a ``frozenset`` of non-empty constructor parameter names.

.. code-block:: python

   class CustomRemoteBackend(RemoteBackend):
      mcp_safe_config_options = frozenset({"poll_interval", "timeout"})

      def __init__(self, *, endpoint, poll_interval=5.0, timeout=3600.0):
         ...

Do not declare executable paths, credentials, endpoint selection, storage locations, or other options that can redirect execution or access. These remain backend- or server-owned.

.. rubric:: Naming and call order

An algorithm implementation name must be unique within its algorithm type; remote backend and cache backend names must be unique within their respective registries. Third-party plugins should use package- or organization-prefixed names to avoid collisions with built-in implementations and other plugins.

Registration is first come, first served and does not override an existing name. Core built-ins are registered before external registrations reach each registry. Unified plugin entry points are called in the order returned by Python's entry-point discovery, followed by bundled optional integrations. If a plugin reuses an existing name, registration raises :class:`~qdk_chemistry.plugins.DuplicateRegistrationError` and keeps the earlier implementation unchanged. During entry-point discovery, QDK/Chemistry catches that exception, emits a ``UserWarning`` identifying the plugin that failed to register, and continues loading other plugins. Because entry-point order can vary between environments, plugins must not rely on discovery order to override another implementation.

.. _adding-remote-backends:

Implementing a remote backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A remote backend implements the transport and job lifecycle required to execute a serialized QDK/Chemistry request outside the calling process. The following example uses the system ``ssh`` and ``scp`` commands to transfer files and launch a background process.

.. note::

   This example targets a directly SSH-accessible machine with ``python3`` and QDK/Chemistry available in its default environment. It does not submit through a queue scheduler such as SLURM or PBS. Queue-managed systems should provide a backend designed for their scheduler and site policy.

   The example retains each remote job directory, including its inputs, outputs,
   PID file, and logs, after the job reaches a terminal state. Applications are
   responsible for removing these artifacts through job cleanup. That cleanup
   does not remove caller-owned local job records or result directories.

.. rubric:: Backend implementation

Implement :class:`~qdk_chemistry.remote.backends.base.RemoteBackend` for the target transport and execution environment:

.. literalinclude:: ../../_static/examples/python/custom_remote_backend.py
   :language: python
   :start-after: # start-cell-custom-remote-backend
   :end-before: # end-cell-custom-remote-backend

.. rubric:: Registration and discovery

Register the backend through :class:`~qdk_chemistry.plugins.base.PluginRegistrar` from a :class:`~qdk_chemistry.plugins.base.QdkChemistryPlugin`:

.. literalinclude:: ../../_static/examples/python/custom_remote_backend.py
   :language: python
   :start-after: # start-cell-custom-remote-registration
   :end-before: # end-cell-custom-remote-registration

Expose that plugin class through the unified entry-point group in the plugin package's ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."qdk_chemistry.plugins"]
   ssh = "custom_package.ssh_backend:SSHRemoteBackendPlugin"

After the package is installed, importing ``qdk_chemistry`` discovers the entry point and registers the backend. Application code does not import the plugin module explicitly.

.. rubric:: Usage

The discovered backend is available through the standard remote backend registry:

.. literalinclude:: ../../_static/examples/python/custom_remote_backend.py
   :language: python
   :start-after: # start-cell-custom-remote-usage
   :end-before: # end-cell-custom-remote-usage

Algorithms created through :func:`qdk_chemistry.algorithms.create` accept ``remote`` and ``cache`` keyword arguments on ``run``. :func:`~qdk_chemistry.remote.backends.base.create_remote` returns a connected backend, which the caller disconnects when it is no longer needed:

.. literalinclude:: ../../_static/examples/python/custom_remote_backend.py
   :language: python
   :start-after: # start-cell-custom-remote-run
   :end-before: # end-cell-custom-remote-run

Remote argument and result values support QDK Chemistry data classes, NumPy arrays with non-object and non-structured data types, ``None``, booleans, integers, floats, strings, NumPy scalar equivalents, and lists or tuples recursively containing supported values. :class:`~qdk_chemistry.data.AlgorithmRef` values are also supported in arguments and settings, including nested algorithm-reference settings. Generic dictionaries are not supported as argument or result values. Keyword arguments and algorithm settings remain mappings because the protocol serializes their entries separately; use a QDK Chemistry data class for other structured values.

Disconnecting closes connection-scoped resources but does not remove artifacts belonging to submitted jobs. For an asynchronous :class:`~qdk_chemistry.remote.job.Job`, pass ``cleanup=True`` to :meth:`~qdk_chemistry.remote.job.Job.fetch` to remove backend artifacts after the result is successfully retrieved and persisted. Call :meth:`~qdk_chemistry.remote.job.Job.cleanup` to remove artifacts separately for any terminal job. Cleanup is idempotent; failed retrieval leaves artifacts available for inspection or retry.

Passing a path as ``cache`` creates a local :class:`~qdk_chemistry.remote.cache.folder.FolderCache`. On a completed cache hit, ``run`` reconstructs and returns the result without submitting another remote job. If the cache contains an in-flight job for the same algorithm, settings, and inputs, polling resumes instead of creating a duplicate. Pass ``force_rerun=True`` to bypass the lookup and execute again.

By default, a cache is local to the calling machine. Set ``is_shared=True`` only when the same backing store is reachable from both the calling machine and remote compute node, such as a network-mounted directory:

.. literalinclude:: ../../_static/examples/python/custom_remote_backend.py
   :language: python
   :start-after: # start-cell-custom-remote-shared-cache
   :end-before: # end-cell-custom-remote-shared-cache

A shared cache lets the remote worker reuse content-addressed inputs already present there and publish results without transferring those files through the backend. Do not mark a caller-local directory as shared; the remote worker must be able to recreate and access the configured cache.

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

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-custom-settings
      :end-before: # end-cell-custom-settings

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-custom-settings
      :end-before: // end-cell-custom-settings

.. rubric:: Implementation structure

The implementation class inherits from the algorithm base class and overrides the required methods.
The ``_run_impl()`` method is responsible for:

1. Converting QDK/Chemistry data structures to the external package's format
2. Invoking the external computation
3. Converting results back to QDK/Chemistry data structures

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-custom-scf-solver
      :end-before: # end-cell-custom-scf-solver

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-custom-scf-solver
      :end-before: // end-cell-custom-scf-solver

.. rubric:: Registration

Implementations are registered with the algorithm factory to enable discovery and instantiation by name.
The plugin registrar delegates to that existing factory registry:

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-registration
      :end-before: # end-cell-registration

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-registration
      :end-before: // end-cell-registration

Following registration, the implementation is accessible through the standard API:

.. literalinclude:: ../../_static/examples/python/custom_plugin.py
   :language: python
   :start-after: # start-cell-usage-after-registration
   :end-before: # end-cell-usage-after-registration

.. _custom-algorithm-types:

Defining a new algorithm type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the required functionality does not correspond to an existing algorithm category, a new algorithm type can be defined.
This section demonstrates the complete process using a molecular descriptor calculator as an example.

.. rubric:: Interface design

The first step is to specify the algorithm's interface:

Input type
   The data the algorithm operates on (e.g., ``Structure``)
Output type
   The data the algorithm produces (e.g., a floating-point molecular descriptor)
Configuration
   Required settings (e.g., whether to normalize the descriptor)

.. rubric:: Settings class definition

Define a settings class containing all configuration parameters:

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-descriptor-settings
      :end-before: # end-cell-descriptor-settings

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-descriptor-settings
      :end-before: // end-cell-descriptor-settings

.. rubric:: Base class definition

Define an abstract base class specifying the interface for all implementations:

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-descriptor-base-class
      :end-before: # end-cell-descriptor-base-class

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-descriptor-base-class
      :end-before: // end-cell-descriptor-base-class

.. rubric:: Factory definition

The factory manages implementation registration and provides instance creation:

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-descriptor-factory
      :end-before: # end-cell-descriptor-factory

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-descriptor-factory
      :end-before: // end-cell-descriptor-factory

.. rubric:: Concrete implementations

Implement the algorithm by inheriting from the base class:

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-descriptor-implementations
      :end-before: # end-cell-descriptor-implementations

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-descriptor-implementations
      :end-before: // end-cell-descriptor-implementations

Additional implementations follow the same pattern:

.. literalinclude:: ../../_static/examples/python/custom_plugin.py
   :language: python
   :start-after: # start-cell-mass-descriptor
   :end-before: # end-cell-mass-descriptor

.. rubric:: Registration

Register the factory and all implementations:

.. tab:: Python API

   .. literalinclude:: ../../_static/examples/python/custom_plugin.py
      :language: python
      :start-after: # start-cell-descriptor-registration
      :end-before: # end-cell-descriptor-registration

.. tab:: C++ API

   .. literalinclude:: ../../_static/examples/cpp/custom_plugin.cpp
      :language: cpp
      :start-after: // start-cell-descriptor-registration
      :end-before: // end-cell-descriptor-registration

.. rubric:: Usage

Following registration, the new algorithm type is accessible through the standard API:

.. literalinclude:: ../../_static/examples/python/custom_plugin.py
   :language: python
   :start-after: # start-cell-descriptor-usage
   :end-before: # end-cell-descriptor-usage

For additional information on the factory pattern and settings system, refer to the
:doc:`factory pattern <algorithms/factory_pattern>` and :doc:`settings <algorithms/settings>` documentation.


Further reading
---------------

- Custom plugin examples: `C++ source <../../_static/examples/cpp/custom_plugin.cpp>`__ | `Python source <../../_static/examples/python/custom_plugin.py>`__
- `SSH remote backend plugin example <../../_static/examples/python/custom_remote_backend.py>`__
- Plugin usage examples: `C++ example <../../_static/examples/cpp/interfaces.cpp>`__ | `Python example <../../_static/examples/python/interfaces.py>`__
- :doc:`Factory pattern <algorithms/factory_pattern>`
- :doc:`Settings <algorithms/settings>`
- :doc:`Serialization <data/serialization>`
