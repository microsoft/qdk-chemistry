Settings
========

The :class:`~qdk_chemistry.data.Settings` class in QDK/Chemistry provides a flexible configuration mechanism for all algorithms and data structures in the quantum chemistry toolkit.
It allows users to customize the behavior of algorithms by setting various parameters with type safety and extensibility.
This unified settings system ensures consistency across the toolkit and makes it easy to save, load, and share configurations between different parts of your application.

Overview
--------

The :class:`~qdk_chemistry.data.Settings` class is a type-safe key-value store that provides a unified interface for configuring algorithms and data structures.
Each algorithm type (such as :doc:`../algorithms/scf_solver`, :doc:`../algorithms/localizer`, etc.) extends the base :class:`~qdk_chemistry.data.Settings` class to create its own specialized settings with appropriate default values.
The class follows a design philosophy where default settings are established during class initialization.
This approach aligns with QDK/Chemistry's broader :doc:`design principles <index>` of type safety and flexibility.

The class supports:

- **Type-safe storage**: Stores different value types (bool, int, long, size_t, float, double, string, and vectors) in a
  type-safe variant system
- **Convenient accessor methods**: Get values with or without default fallbacks, check existence, and validate types
- **Multiple serialization formats**: Save and load from JSON and HDF5 files for interoperability (see
  :doc:`Serialization <../data/serialization>` for more details)
- **Default settings protection**: Default settings are defined during initialization through the protected
  ``set_default`` method, values can be alterd before algorithm execution but are locked afterwasrds to ensure consistency

Accessing settings
------------------

All QDK/Chemistry algorithm classes that use configurable parameters provide access to their settings through the ``settings()`` method, which returns a reference to their internal :class:`~qdk_chemistry.data.Settings` object.
This consistent interface makes it easy to configure any algorithm in the toolkit using the same pattern.

The settings object acts as a bridge between the user interface and the internal algorithm implementation, allowing you to modify algorithm behavior without changing its code.
Most algorithms validate their settings only at execution time, so you can adjust parameters anytime before running the algorithm.

.. tab:: C++ API

   .. code-block:: cpp

      // Get the settings object
      auto& settings = algorithm->settings();

      // Set a parameter
      settings.set("parameter_name", value);

      // Get a parameter
      auto value = settings.get<ValueType>("parameter_name");

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 8-15

Common settings operations
--------------------------

The :class:`~qdk_chemistry.data.Settings` class provides a rich set of methods for manipulating and accessing configuration parameters.
Each method is designed to be intuitive while providing robust error handling and type safety.

Setting values
~~~~~~~~~~~~~~

Values can be set for any key in the settings map.
If the key already exists, the value will be updated.
If the key doesn't exist, it will be created.
The ``set`` method is overloaded to handle various types including C-style strings, which are automatically converted to ``std::string``.

.. tab:: C++ API

   .. code-block:: cpp

      // Set a string value
      settings.set("basis_set", "def2-tzvp");

      // Set a numeric value
      settings.set("convergence_threshold", 1.0e-8);

      // Set a boolean value
      settings.set("density_fitting", true);

      // Set an array value
      settings.set("active_orbitals", std::vector<int>{4, 5, 6, 7});

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 19-29

Getting values
~~~~~~~~~~~~~~

Values can be retrieved with type checking to ensure the correct data type is returned.
The templated ``get`` method throws exceptions if the key doesn't exist or if the requested type doesn't match the stored type.
For cases where you want to provide a fallback value if the key doesn't exist, use the ``get_or_default`` method.

.. tab:: C++ API

   .. code-block:: cpp

      // Get a string value
      std::string basis = settings.get<std::string>("basis_set");

      // Get a numeric value
      double threshold = settings.get<double>("convergence_threshold");

      // Get a boolean value
      bool use_df = settings.get<bool>("density_fitting");

      // Get an array value
      auto active_orbitals = settings.get<std::vector<int>>("active_orbitals");

      // Get a value with default fallback
      auto max_iter = settings.get_or_default<int>("max_iterations", 100);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 31-44

Checking for settings
~~~~~~~~~~~~~~~~~~~~~

Before accessing a setting, you might want to check if it exists or if it has the expected type.
The :class:`~qdk_chemistry.data.Settings` class provides methods for both checks.
Additionally, the ``try_get`` method returns an ``std::optional`` that contains the value if it exists and has the correct type, or is empty otherwise.

.. tab:: C++ API

   .. code-block:: cpp

      // Check if a setting exists
      if (settings.has("basis_set")) {
          // Use the setting
      }

      // Check if a setting exists with the expected type
      if (settings.has_type<double>("convergence_threshold")) {
          // Use the setting
      }

      // Try to get a value (returns std::optional)
      auto maybe_value = settings.try_get<double>("convergence_threshold");
      if (maybe_value) {
          double value = *maybe_value;
          // Use the value
      }

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 49-65

Other operations
~~~~~~~~~~~~~~~~

The :class:`~qdk_chemistry.data.Settings` class provides numerous utility methods for working with the settings collection as a whole.
These include methods for introspection (examining what settings exist), validation (checking that required settings are
present), manipulation (merging with other settings objects), and more.

.. tab:: C++ API

   .. code-block:: cpp

      // Get all setting keys
      auto keys = settings.keys();

      // Get the number of settings
      size_t count = settings.size();

      // Check if settings are empty
      bool is_empty = settings.empty();

      // Clear all settings
      settings.clear();

      // Validate that required settings exist
      settings.validate_required({"basis_set", "convergence_threshold"});

      // Get a setting as a string representation
      std::string value_str = settings.get_as_string("convergence_threshold");

      // Merge settings from another settings object
      Settings other_settings;
      settings.merge(other_settings, true); // true to overwrite existing

      // Update an existing setting (throws if key doesn't exist)
      settings.update("convergence_threshold", 1.0e-9);

      // Get the type name of a setting
      std::string type = settings.get_type_name("convergence_threshold");

.. tab:: Python API

   .. Literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 75-101

Serialization
-------------

Configuration persistence is important for reproducibility in scientific computing.
The :class:`~qdk_chemistry.data.Settings` class provides methods to serialize and deserialize settings to both JSON and HDF5 formats.
This allows you to save algorithm configurations, share them with colleagues, or use them in future runs to ensure consistent results.

JSON is a human-readable format ideal for manual editing and inspection, while HDF5 offers better performance and type preservation for large datasets.
For more information on serialization throughout QDK/Chemistry, see the :doc:`Serialization <../data/serialization>` documentation.

.. tab:: C++ API

   .. code-block:: cpp

      // Save settings to JSON file
      settings.to_json_file("configuration.settings.json");

      // Load settings from JSON file
      auto settings_from_json_file = Settings::from_json_file("configuration.settings.json");

      // Save settings to HDF5 file
      settings.to_hdf5("configuration.settings.h5");

      // Load settings from HDF5 file
      auto settings_from_hdf5 = Settings::from_hdf5("configuration.settings.h5");

      // Generic file I/O with specified format
      settings.to_file("configuration", "json");
      auto settings_from_file = Settings::from_file("configuration", "hdf5");

      // Convert to JSON object
      auto json_data = settings.to_json();

      // Load from JSON object
      auto settings_from_json = Settings::from_json(json_data);

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 110-130

Serialization format
~~~~~~~~~~~~~~~~~~~~

When settings are serialized, the format preserves both the keys and the associated values with their types.
Here are examples of how settings are serialized in both JSON and HDF5 formats, using an :term:`SCF` solver configuration as an example.

JSON format
^^^^^^^^^^^

.. code-block:: json

   {
     "basis_set": "sto-3g",
     "convergence_threshold": 1e-08,
     "max_iterations": 200,
     "method": "hf"
   }

HDF5 format
^^^^^^^^^^^

.. code-block:: text

   /settings  # HDF5 group for SCF solver settings
     ├── basis_set              # String dataset, the basis set
     ├── convergence_threshold  # Double dataset, the energy convergence threshold
     ├── max_iterations         # Integer dataset, the maximum number of iterations
     └── method                 # String dataset, the method, e.g. hf

Extending settings
------------------

The :class:`~qdk_chemistry.data.Settings` class is designed with inheritance in mind, allowing algorithm developers to create specialized settings classes with predefined parameters and defaults.
This design pattern ensures that algorithm settings are well-defined and discoverable.

The key aspect of this design is that default values are established during construction using the protected ``set_default`` method, which ensures baseline functionality.
While new settings can be added at runtime through the public ``set`` method, defining defaults during construction helps with documentation and discoverability.

This extensibility model is part of QDK/Chemistry's broader :doc:`Factory Pattern <factory_pattern>` design, which allows for flexible algorithm implementations while maintaining a consistent API.
The pattern is used throughout QDK/Chemistry, such as in the :doc:`Interface System <interfaces>` for integrating third-party packages.

Here's how to extend the :class:`~qdk_chemistry.data.Settings` class for a custom algorithm:

.. tab:: C++ API

   .. code-block:: cpp

      class MySettings : public Settings {
      public:
          MySettings() {
              // Can only call set_default during construction
              set_default("max_iterations", 100);
              set_default("tolerance", 1e-6);
              set_default("method", std::string("default"));
          }
      };

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 134-140


Supported types
---------------

The :class:`~qdk_chemistry.data.Settings` class uses a variant-based type system to store different types of values in a type-safe manner.
This system balances flexibility with strong typing, allowing settings to hold various types of data while still providing compile-time type checking when accessed.

Currently, the following value types are supported:

- ``bool`` - For binary settings (true/false flags)
- ``int`` - For integer settings with moderate range
- ``long`` - For integer settings with larger range
- ``size_t`` - For non-negative counts and indices
- ``float`` - For single-precision floating-point values
- ``double`` - For double-precision floating-point values (recommended for most numerical parameters)
- ``std::string`` - For text, identifiers, file paths, and other string data
- ``std::vector<int>`` - For collections of integers (e.g., orbital indices)
- ``std::vector<double>`` - For numerical arrays (e.g., coordinates, coefficients)
- ``std::vector<std::string>`` - For collections of strings (e.g., atom labels)

This type system can be extended in the future to support additional types as needed.
The implementation ensures type safety through C++ templates and variant handling.

Error handling
--------------

The :class:`~qdk_chemistry.data.Settings` class uses exceptions to provide clear error messages when operations fail.
This exception-based approach makes errors explicit and helps catch configuration issues early in development.
The class defines two specific exception types:

- ``SettingNotFound``: Thrown when attempting to access a setting that doesn't exist in the map.
  The exception message includes the key that was requested to help with debugging.
- ``SettingTypeMismatch``: Thrown when attempting to access a setting with the wrong type.
  For example, trying to get an ``int`` when the setting actually holds a ``string``.
  The exception message includes both the key and the expected type.

These exceptions can be caught and handled to provide graceful error recovery:

.. tab:: C++ API

   .. code-block:: cpp

      try {
          auto value = settings.get<double>("non_existent_setting");
      } catch (const qdk::chemistry::data::SettingNotFound& e) {
          std::cerr << e.what() << std::endl; // "Setting not found: non_existent_setting"
      }

      try {
          auto value = settings.get<int>("string_setting"); // where string_setting is a string
      } catch (const qdk::chemistry::data::SettingTypeMismatch& e) {
          std::cerr << e.what() << std::endl; // "Type mismatch for setting 'string_setting'. Expected: int"
      }

.. tab:: Python API

   .. literalinclude:: ../../../../examples/settings.py
      :language: python
      :lines: 144-147

Related topics
--------------

- :doc:`Design Principles <index>`: Core architectural principles of QDK/Chemistry
- :doc:`Factory Pattern <factory_pattern>`: Understanding the factory pattern and extending QDK/Chemistry
- :doc:`Interfaces <interfaces>`: QDK/Chemistry's interface system to external packages
- :doc:`Serialization <../data/serialization>`: Data persistence in QDK/Chemistry
