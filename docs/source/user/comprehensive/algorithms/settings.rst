Settings
========

Every :doc:`algorithm <index>` in QDK/Chemistry is configured through a :class:`~qdk_chemistry.data.Settings` object—a type-safe key-value store for customizing algorithm behavior.
This unified configuration system simplifies saving, loading, and sharing settings across workflows.

.. _discovering-settings:

Discovering available settings
------------------------------

When working with an algorithm, it is often necessary to determine which settings are available and their current values.
QDK/Chemistry provides several methods for discovering this information:

**Documentation**
   Each algorithm's documentation page includes an "Available settings" section that lists all supported parameters, their types, default values, and descriptions.
   For examples, see :doc:`ScfSolver <scf_solver>` or :doc:`MultiConfigurationCalculator <mc_calculator>`.

**Runtime inspection**
   Settings can be inspected programmatically using methods such as ``keys()``, ``items()``, or ``print_settings()``.
   This approach is particularly useful when exploring unfamiliar implementations or verifying configurations.

The following example demonstrates how to discover available algorithms and inspect their settings:

.. tab:: C++ API

   .. code-block:: cpp

      #include <qdk/chemistry/algorithms/scf.hpp>

      // List available implementations for a specific algorithm type
      auto scf_names = qdk::chemistry::algorithms::ScfSolverFactory::available();
      for (const auto& name : scf_names) {
          std::cout << "SCF solver: " << name << std::endl;
      }

      // Create an instance and inspect its settings
      auto scf = qdk::chemistry::algorithms::ScfSolverFactory::create();
      for (const auto& key : scf->settings().keys()) {
          std::cout << "  " << key << ": " << scf->settings().get_as_string(key) << std::endl;
      }

.. tab:: Python API

   .. code-block:: python

      from qdk_chemistry.algorithms import available, create, print_settings, inspect_settings

      # List all algorithm types and their implementations
      for algo_type, implementations in available().items():
          print(f"{algo_type}: {implementations}")

      # Display a formatted settings table for a specific implementation
      print_settings("scf_solver", "qdk")

      # Inspect settings programmatically
      for name, type_name, default, description, limits in inspect_settings("scf_solver", "qdk"):
          print(f"{name} ({type_name}): {default}")

      # Create an instance and iterate over its settings
      scf = create("scf_solver")
      for key, value in scf.settings().items():
          print(f"  {key}: {value}")

Working with settings
---------------------

All algorithm classes expose their configuration through the ``settings()`` method, which returns a reference to the algorithm's internal :class:`~qdk_chemistry.data.Settings` object.
This object supports both read and write operations.
Settings are validated at execution time, allowing modifications at any point before calling ``run()``.

.. important::

   **Settings are locked after execution.**
   When ``run()`` is invoked on an algorithm, its settings are automatically locked to ensure reproducibility.
   Any subsequent attempt to modify the settings raises a ``SettingsAreLocked`` exception.
   To run the same algorithm with different parameters, create a new algorithm instance.

   .. code-block:: python

      scf = create("scf_solver")
      scf.settings().set("basis_set", "sto-3g")
      energy, wfn = scf.run(structure, charge=0, spin_multiplicity=1)

      # Settings are now locked - this raises SettingsAreLocked:
      # scf.settings().set("basis_set", "cc-pvdz")

      # Create a new instance for different settings
      scf2 = create("scf_solver")
      scf2.settings().set("basis_set", "cc-pvdz")
      energy2, wfn2 = scf2.run(structure, charge=0, spin_multiplicity=1)

**Accessing and modifying settings:**

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/settings.cpp
      :language: cpp
      :start-after: // start-cell-get-settings
      :end-before: // end-cell-get-settings

   .. literalinclude:: ../../../_static/examples/cpp/settings.cpp
      :language: cpp
      :start-after: // start-cell-set-settings
      :end-before: // end-cell-set-settings

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/settings.py
      :language: python
      :start-after: # start-cell-get-settings
      :end-before: # end-cell-get-settings

   .. literalinclude:: ../../../_static/examples/python/settings.py
      :language: python
      :start-after: # start-cell-set-settings
      :end-before: # end-cell-set-settings

**Passing settings at creation time (Python only):**

The Python registry's ``create()`` function accepts keyword arguments that are automatically applied to the algorithm's settings.
This provides a convenient shorthand for configuring algorithms in a single line:

.. literalinclude:: ../../../_static/examples/python/settings.py
   :language: python
   :start-after: # start-cell-factory-settings
   :end-before: # end-cell-factory-settings

This is equivalent to creating the algorithm and then calling ``settings().set()`` for each parameter, but is more concise for common use cases.
The C++ API does not provide this shorthand; settings must be configured explicitly after algorithm creation.

**Checking and retrieving values:**

The :class:`~qdk_chemistry.data.Settings` class provides methods for safely checking and retrieving values.
Use ``has()`` to verify existence, ``get()`` for direct access (raises an exception if the key is missing), or ``get_or_default()`` to specify a fallback value.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/settings.cpp
      :language: cpp
      :start-after: // start-cell-misc-settings
      :end-before: // end-cell-misc-settings

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/settings.py
      :language: python
      :start-after: # start-cell-misc-settings
      :end-before: # end-cell-misc-settings

Serialization
-------------

Settings can be serialized to JSON (human-readable) or HDF5 (efficient, type-preserving) formats to support reproducibility and configuration sharing.
For additional information on data persistence, see :doc:`Serialization <../data/serialization>`.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/settings.cpp
      :language: cpp
      :start-after: // start-cell-serialization
      :end-before: // end-cell-serialization

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/settings.py
      :language: python
      :start-after: # start-cell-serialization
      :end-before: # end-cell-serialization

**Example formats:**

.. code-block:: json
   :caption: JSON format

   {
     "basis_set": "sto-3g",
     "convergence_threshold": 1e-08,
     "max_iterations": 200,
     "method": "hf"
   }

.. code-block:: text
   :caption: HDF5 structure

   /settings
     ├── basis_set              # String
     ├── convergence_threshold  # Double
     ├── max_iterations         # Integer
     └── method                 # String

Extending settings
------------------

Algorithm developers can create specialized settings classes by extending :class:`~qdk_chemistry.data.Settings`.
Default values are established during construction using the ``set_default`` method, ensuring that configurations are discoverable and well-documented.
This pattern integrates with the :doc:`Factory Pattern <factory_pattern>` and :ref:`plugin system <plugin-system>`.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/settings.cpp
      :language: cpp
      :start-after: // start-cell-extend-settings
      :end-before: // end-cell-extend-settings

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/settings.py
      :language: python
      :start-after: # start-cell-extend-settings
      :end-before: # end-cell-extend-settings

Supported types
---------------

The settings system uses a variant-based type system that provides flexibility while maintaining type safety.
The following types are supported:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - C++ type
     - Python type
     - Description
   * - ``bool``
     - ``bool``
     - Boolean flags
   * - ``int64_t``
     - ``int``
     - 64-bit signed integers
   * - ``double``
     - ``float``
     - Double-precision floating-point
   * - ``std::string``
     - ``str``
     - Text values
   * - ``std::vector<int64_t>``
     - ``list[int]``
     - Integer arrays
   * - ``std::vector<double>``
     - ``list[float]``
     - Floating-point arrays
   * - ``std::vector<std::string>``
     - ``list[str]``
     - String arrays

**C++ implementation:**

The C++ layer uses a ``std::variant``-based storage that provides compile-time type checking through templates.

**Python implementation:**

The Python bindings automatically convert between native Python types and the underlying C++ types.
When setting values, Python objects are validated against the expected type and converted appropriately—for example, Python ``float`` values become C++ ``double``, and Python ``list`` objects become ``std::vector`` containers.
When retrieving values, the reverse conversion occurs transparently.

The ``get_expected_python_type()`` method can be used to query the expected Python type for any setting:

.. code-block:: python

   expected_type = settings.get_expected_python_type("convergence_threshold")
   print(expected_type)  # "float"

   expected_type = settings.get_expected_python_type("active_orbitals")
   print(expected_type)  # "list[int]"

This is particularly useful when building dynamic configuration interfaces or validating user input before applying it to settings.

.. note::

   **Integer type handling.**
   All integer types are stored internally as ``int64_t`` (signed 64-bit).
   In C++, integer types such as ``int``, ``long``, ``size_t``, and ``unsigned`` are automatically converted to ``int64_t`` when setting values.
   Retrieval supports conversion back to other integer types via the templated ``get<T>()`` method, with range checking to prevent overflow.
   In Python, native ``int`` values are converted to ``int64_t`` automatically, and boolean values are explicitly rejected to prevent accidental type confusion (since ``bool`` is a subclass of ``int`` in Python).

Constraints
-----------

Settings can define constraints that specify valid ranges or allowed values.
Constraints are established when algorithm developers define settings using ``set_default()``, and they serve two purposes: documentation and validation guidance.

**Constraint types:**

- **Bound constraints** — Define minimum and maximum values for numeric settings (e.g., ``max_iterations`` must be between 1 and 1000).
- **List constraints** — Define an explicit set of allowed values for string or integer settings (e.g., ``method`` must be one of ``["hf", "dft"]``).

**Inspecting constraints:**

Constraints can be queried using ``has_limits()`` and ``get_limits()``:

.. tab:: C++ API

   .. code-block:: cpp

      if (settings.has_limits("max_iterations")) {
          auto limits = settings.get_limits("max_iterations");
          // limits is a Constraint variant (BoundConstraint or ListConstraint)
      }

.. tab:: Python API

   .. code-block:: python

      if settings.has_limits("max_iterations"):
          limits = settings.get_limits("max_iterations")
          # Returns (min, max) tuple for bounds, or list for allowed values
          print(f"Allowed range: {limits}")  # e.g., (1, 1000)

      if settings.has_limits("method"):
          allowed = settings.get_limits("method")
          print(f"Allowed values: {allowed}")  # e.g., ['hf', 'dft']

The ``print_settings()`` and ``inspect_settings()`` functions include constraint information in their output, making it easy to understand the valid configuration options for any algorithm.

Error handling
--------------

The :class:`~qdk_chemistry.data.Settings` class raises descriptive exceptions to facilitate early detection of configuration issues:

- ``SettingNotFound``: Raised when the requested key does not exist.
- ``SettingTypeMismatch``: Raised when the key exists but the requested type does not match the stored type.
- ``SettingsAreLocked``: Raised when attempting to modify settings after ``run()`` has been called.

.. tab:: C++ API

   .. literalinclude:: ../../../_static/examples/cpp/settings.cpp
      :language: cpp
      :start-after: // start-cell-settings-errors
      :end-before: // end-cell-settings-errors

.. tab:: Python API

   .. literalinclude:: ../../../_static/examples/python/settings.py
      :language: python
      :start-after: # start-cell-settings-errors
      :end-before: # end-cell-settings-errors

See also
--------

- Complete example scripts: `C++ <../../../_static/examples/cpp/settings.cpp>`_ | `Python <../../../_static/examples/python/settings.py>`_
- :doc:`Factory Pattern <factory_pattern>` — Algorithm creation and customization
- :doc:`Serialization <../data/serialization>` — Data persistence in QDK/Chemistry
- :ref:`Plugin system <plugin-system>` — Extending QDK/Chemistry with custom implementations
