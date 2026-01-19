// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>

#include "path_utils.hpp"

namespace py = pybind11;

using namespace qdk::chemistry::data;  // Import the qdk::chemistry::data
                                       // namespace for SettingValue and
                                       // Settings

// Helper function to check if a Python object is an integer-like type
// (including numpy integer types)
inline bool is_integer_like(const py::object &obj) {
  if (py::isinstance<py::bool_>(obj)) {
    return false;  // Exclude booleans
  }
  if (py::isinstance<py::int_>(obj)) {
    return true;
  }
  // Check for numpy integer types
  try {
    py::module_ np = py::module_::import("numpy");
    py::object integer_type = np.attr("integer");
    return py::isinstance(obj, integer_type);
  } catch (...) {
    // numpy not available or import failed
    return false;
  }
}

// Helper function to check if a Python object is a float-like type (including
// numpy float types)
inline bool is_float_like(const py::object &obj) {
  if (py::isinstance<py::float_>(obj)) {
    return true;
  }
  // Check for numpy floating types
  try {
    py::module_ np = py::module_::import("numpy");
    py::object floating_type = np.attr("floating");
    return py::isinstance(obj, floating_type);
  } catch (...) {
    // numpy not available or import failed
    return false;
  }
}

// Helper function to check if a Python object is a numeric type (int or float,
// including numpy types)
inline bool is_numeric_like(const py::object &obj) {
  return is_integer_like(obj) || is_float_like(obj);
}

// Helper function to convert SettingValue to Python objects
py::object setting_value_to_python(const SettingValue &value) {
  return std::visit(
      [](const auto &variant_value) -> py::object {
        using ValueType = std::decay_t<decltype(variant_value)>;

        if constexpr (std::is_same_v<ValueType, bool>) {
          return py::bool_(variant_value);
        } else if constexpr (std::is_integral_v<ValueType>) {
          return py::int_(variant_value);
        } else if constexpr (std::is_same_v<ValueType, float> ||
                             std::is_same_v<ValueType, double>) {
          return py::float_(variant_value);
        } else if constexpr (std::is_same_v<ValueType, std::string>) {
          return py::str(variant_value);
        } else if constexpr (std::is_same_v<ValueType, std::vector<int64_t>>) {
          return py::cast(variant_value);
        } else if constexpr (std::is_same_v<ValueType, std::vector<uint64_t>>) {
          return py::cast(variant_value);
        } else if constexpr (std::is_same_v<ValueType, std::vector<double>>) {
          return py::cast(variant_value);
        } else if constexpr (std::is_same_v<ValueType,
                                            std::vector<std::string>>) {
          return py::cast(variant_value);
        } else {
          static_assert(sizeof(ValueType) == 0,
                        "Unsupported type in SettingValue");
        }
      },
      value);
}

// Helper function to convert Python objects to SettingValue based on expected
// type This function looks up the expected type and attempts to cast the Python
// object to it
SettingValue python_to_setting_value_with_type(const py::object &obj,
                                               const std::string &expected_type,
                                               const std::string &key) {
  try {
    if (expected_type == "bool") {
      if (!py::isinstance<py::bool_>(obj)) {
        throw SettingTypeMismatch(
            key, "bool (got " + std::string(py::str(py::type::of(obj))) + ")");
      }
      return obj.cast<bool>();
    } else if (expected_type == "int" || expected_type == "int64_t" ||
               expected_type == "uint64_t") {
      if (!py::isinstance<py::int_>(obj) || py::isinstance<py::bool_>(obj)) {
        throw SettingTypeMismatch(
            key, "int (got " + std::string(py::str(py::type::of(obj))) + ")");
      }
      return obj.cast<int64_t>();
    } else if (expected_type == "float") {
      if (!py::isinstance<py::float_>(obj) && !py::isinstance<py::int_>(obj)) {
        throw SettingTypeMismatch(
            key, "float (got " + std::string(py::str(py::type::of(obj))) + ")");
      }
      return obj.cast<float>();
    } else if (expected_type == "double") {
      if (!py::isinstance<py::float_>(obj) && !py::isinstance<py::int_>(obj)) {
        throw SettingTypeMismatch(
            key,
            "double (got " + std::string(py::str(py::type::of(obj))) + ")");
      }
      return obj.cast<double>();
    } else if (expected_type == "string") {
      if (!py::isinstance<py::str>(obj)) {
        throw SettingTypeMismatch(
            key,
            "string (got " + std::string(py::str(py::type::of(obj))) + ")");
      }
      return obj.cast<std::string>();
    } else if (expected_type == "vector<int>" ||
               expected_type == "vector<int64_t>" ||
               expected_type == "vector<uint64_t>") {
      std::vector<int64_t> result;
      if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::sequence seq = obj.cast<py::sequence>();
        result.reserve(seq.size());
        for (size_t i = 0; i < seq.size(); ++i) {
          py::object elem = seq[i];
          if (py::isinstance<py::bool_>(elem)) {
            throw SettingTypeMismatch(key, "vector<int> (element " +
                                               std::to_string(i) +
                                               " is bool, expected int)");
          }
          if (!is_integer_like(elem)) {
            throw SettingTypeMismatch(
                key, "vector<int> (element " + std::to_string(i) + " is " +
                         std::string(py::str(py::type::of(elem))) +
                         ", expected int)");
          }
          try {
            int64_t value = elem.cast<int64_t>();
            result.push_back(value);
          } catch (const py::cast_error &e) {
            throw SettingTypeMismatch(
                key, "vector<int> (element " + std::to_string(i) +
                         " cast failed: " + std::string(e.what()) + ")");
          }
        }
        return result;
      } else if (py::isinstance<py::array>(obj)) {
        py::array arr = obj.cast<py::array>();
        result.reserve(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
          py::object elem = arr[py::int_(i)];
          if (py::isinstance<py::bool_>(elem)) {
            throw SettingTypeMismatch(key, "vector<int> (array element " +
                                               std::to_string(i) +
                                               " is bool, expected int)");
          }
          if (!is_integer_like(elem)) {
            throw SettingTypeMismatch(
                key, "vector<int> (array element " + std::to_string(i) +
                         " is " + std::string(py::str(py::type::of(elem))) +
                         ", expected int)");
          }
          try {
            int64_t value = elem.cast<int64_t>();
            result.push_back(value);
          } catch (const py::cast_error &e) {
            throw SettingTypeMismatch(
                key, "vector<int> (array element " + std::to_string(i) +
                         " cast failed: " + std::string(e.what()) + ")");
          }
        }
        return result;
      } else {
        throw SettingTypeMismatch(
            key, "vector<int> (got " + std::string(py::str(py::type::of(obj))) +
                     ", expected list, tuple, or numpy array)");
      }
    } else if (expected_type == "vector<double>") {
      std::vector<double> result;
      if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::sequence seq = obj.cast<py::sequence>();
        result.reserve(seq.size());
        for (size_t i = 0; i < seq.size(); ++i) {
          py::object elem = seq[i];
          if (!is_numeric_like(elem)) {
            throw SettingTypeMismatch(
                key, "vector<double> (element " + std::to_string(i) + " is " +
                         std::string(py::str(py::type::of(elem))) +
                         ", expected float or int)");
          }
          try {
            result.push_back(elem.cast<double>());
          } catch (const py::cast_error &e) {
            throw SettingTypeMismatch(
                key, "vector<double> (element " + std::to_string(i) +
                         " cast failed: " + std::string(e.what()) + ")");
          }
        }
        return result;
      } else if (py::isinstance<py::array>(obj)) {
        py::array arr = obj.cast<py::array>();
        result.reserve(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
          py::object elem = arr[py::int_(i)];
          if (!is_numeric_like(elem)) {
            throw SettingTypeMismatch(
                key, "vector<double> (array element " + std::to_string(i) +
                         " is " + std::string(py::str(py::type::of(elem))) +
                         ", expected float or int)");
          }
          try {
            result.push_back(elem.cast<double>());
          } catch (const py::cast_error &e) {
            throw SettingTypeMismatch(
                key, "vector<double> (array element " + std::to_string(i) +
                         " type " + std::string(py::str(py::type::of(elem))) +
                         " cast failed: " + std::string(e.what()) + ")");
          }
        }
        return result;
      } else {
        throw SettingTypeMismatch(
            key, "vector<double> (got " +
                     std::string(py::str(py::type::of(obj))) +
                     ", expected list, tuple, or numpy array)");
      }
    } else if (expected_type == "vector<string>") {
      std::vector<std::string> result;
      if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::sequence seq = obj.cast<py::sequence>();
        result.reserve(seq.size());
        for (size_t i = 0; i < seq.size(); ++i) {
          py::object elem = seq[i];
          if (!py::isinstance<py::str>(elem)) {
            throw SettingTypeMismatch(
                key, "vector<string> (element " + std::to_string(i) + " is " +
                         std::string(py::str(py::type::of(elem))) +
                         ", expected string)");
          }
          try {
            result.push_back(elem.cast<std::string>());
          } catch (const py::cast_error &e) {
            throw SettingTypeMismatch(
                key, "vector<string> (element " + std::to_string(i) +
                         " cast failed: " + std::string(e.what()) + ")");
          }
        }
        return result;
      } else {
        throw SettingTypeMismatch(key,
                                  "vector<string> (got " +
                                      std::string(py::str(py::type::of(obj))) +
                                      ", expected list or tuple)");
      }
    } else if (expected_type == "vector<float>") {
      // Map vector<float> to vector<double>
      std::vector<double> result;
      if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::sequence seq = obj.cast<py::sequence>();
        result.reserve(seq.size());
        for (size_t i = 0; i < seq.size(); ++i) {
          py::object elem = seq[i];
          if (!is_numeric_like(elem)) {
            throw SettingTypeMismatch(
                key, "vector<float> (element " + std::to_string(i) + " is " +
                         std::string(py::str(py::type::of(elem))) +
                         ", expected float or int)");
          }
          try {
            result.push_back(elem.cast<double>());
          } catch (const py::cast_error &e) {
            throw SettingTypeMismatch(
                key, "vector<float> (element " + std::to_string(i) +
                         " cast failed: " + std::string(e.what()) + ")");
          }
        }
        return result;
      } else if (py::isinstance<py::array>(obj)) {
        py::array arr = obj.cast<py::array>();
        result.reserve(arr.size());
        for (size_t i = 0; i < arr.size(); ++i) {
          py::object elem = arr[py::int_(i)];
          if (!is_numeric_like(elem)) {
            throw SettingTypeMismatch(
                key, "vector<float> (array element " + std::to_string(i) +
                         " is " + std::string(py::str(py::type::of(elem))) +
                         ", expected float or int)");
          }
          try {
            result.push_back(elem.cast<double>());
          } catch (const py::cast_error &e) {
            throw SettingTypeMismatch(
                key, "vector<float> (array element " + std::to_string(i) +
                         " type " + std::string(py::str(py::type::of(elem))) +
                         " cast failed: " + std::string(e.what()) + ")");
          }
        }
        return result;
      } else {
        throw SettingTypeMismatch(
            key, "vector<float> (got " +
                     std::string(py::str(py::type::of(obj))) +
                     ", expected list, tuple, or numpy array)");
      }
    } else {
      throw std::runtime_error("Unknown expected type '" + expected_type +
                               "' for setting '" + key +
                               "'. Supported types are: bool, int, float, "
                               "double, string, vector<int>, "
                               "vector<float>, vector<double>, "
                               "vector<string>");
    }
  } catch (const py::cast_error &e) {
    throw SettingTypeMismatch(
        key, expected_type + " (cast failed: " + std::string(e.what()) + ")");
  }
}

// Python trampoline class for Settings to enable Python-based implementations
class PySettings : public Settings, public py::trampoline_self_life_support {
 public:
  using Settings::Settings;

  // Expose set_default for Python derived classes to use in __init__
  // Now requires the expected type to ensure type-safe defaults
  void _set_default(const std::string &key, const std::string &expected_type,
                    const py::object &value,
                    const py::object &description = py::none(),
                    const py::object &limit = py::none(),
                    bool documented = true) {
    SettingValue setting_value =
        python_to_setting_value_with_type(value, expected_type, key);

    std::optional<std::string> desc;
    if (!description.is_none()) {
      desc = py::cast<std::string>(description);
    }

    std::optional<Constraint> limit_val;
    if (!limit.is_none()) {
      // Convert Python limit to Constraint
      if (py::isinstance<py::tuple>(limit) && py::len(limit) == 2) {
        // Range limit (tuple of 2 elements) - convert to BoundConstraint
        py::tuple limit_tuple = py::cast<py::tuple>(limit);
        if (expected_type == "int" || expected_type == "vector<int>") {
          BoundConstraint<int64_t> bound;
          bound.min = py::cast<int64_t>(limit_tuple[0]);
          bound.max = py::cast<int64_t>(limit_tuple[1]);
          limit_val = bound;
        } else if (expected_type == "double" || expected_type == "float" ||
                   expected_type == "vector<double>" ||
                   expected_type == "vector<float>") {
          BoundConstraint<double> bound;
          bound.min = py::cast<double>(limit_tuple[0]);
          bound.max = py::cast<double>(limit_tuple[1]);
          limit_val = bound;
        }
      } else if (py::isinstance<py::list>(limit)) {
        // Enumeration limit (list of allowed values) - convert to
        // ListConstraint
        py::list limit_list = py::cast<py::list>(limit);
        if (expected_type == "int" || expected_type == "vector<int>") {
          ListConstraint<int64_t> list_constraint;
          for (auto item : limit_list) {
            list_constraint.allowed_values.push_back(py::cast<int64_t>(item));
          }
          limit_val = list_constraint;
        } else if (expected_type == "string" ||
                   expected_type == "vector<string>") {
          ListConstraint<std::string> list_constraint;
          for (auto item : limit_list) {
            list_constraint.allowed_values.push_back(
                py::cast<std::string>(item));
          }
          limit_val = list_constraint;
        }
      }
    }

    set_default(key, setting_value, desc, limit_val, documented);
  }

  void _set_default_setting_value(const std::string &key,
                                  const SettingValue &value) {
    set_default(key, value);
  }
};

// Wrapper functions for file I/O methods that accept both strings and pathlib
// Path objects
void settings_to_json_file_wrapper(Settings &self, const py::object &filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<Settings> settings_from_json_file_wrapper(
    const py::object &filename) {
  return Settings::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void settings_to_file_wrapper(Settings &self, const py::object &filename,
                              const std::string &format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

std::shared_ptr<Settings> settings_from_file_wrapper(
    const py::object &filename, const std::string &format_type) {
  return Settings::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

std::shared_ptr<Settings> settings_from_hdf5_file_wrapper(
    const py::object &filename) {
  return Settings::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void settings_to_hdf5_file_wrapper(qdk::chemistry::data::Settings &self,
                                   const py::object &filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

void bind_settings(pybind11::module &data) {
  using namespace qdk::chemistry::algorithms;

  // Bind SettingValue variant
  py::class_<SettingValue> setting_value(data, "SettingValue", R"(
Type-safe variant for storing different setting value types.

This variant can hold common types used in settings configurations:
bool, int, long, size_t, float, double, string, vector<int>, vector<double>, vector<string>
)");

  // Bind exception classes
  py::register_exception<SettingNotFound>(data, "SettingNotFound");
  py::register_exception<SettingTypeMismatch>(data, "SettingTypeMismatch");
  py::register_exception<SettingsAreLocked>(data, "SettingsAreLocked");

  // Utility functions for conversion
  data.def("setting_value_to_python", &setting_value_to_python,
           R"(
Convert a SettingValue to the appropriate Python object.

This utility function converts a C++ SettingValue variant to its corresponding Python object type, preserving the original type and value.

Args:
    value (SettingValue): The SettingValue variant to convert

Returns:
    object: Python object with appropriate type (bool, int, float, str, list)

Examples:
    >>> import qdk_chemistry.data as data
    >>> # Assuming you have a SettingValue from C++
    >>> py_value = data.setting_value_to_python(setting_val)
)",
           py::arg("value"));

  py::class_<Settings, DataClass, PySettings, py::smart_holder> settings(
      data, "Settings", R"(
Base class for extensible settings objects.

This class provides a flexible settings system that can:

* Store arbitrary typed values using a variant system
* Be easily extended by derived classes during construction only
* Map seamlessly to Python dictionaries via pybind11
* Provide type-safe access to settings with default values
* Support nested settings structures
* Prevent extension of the settings map after class initialization

The settings map can only be populated during construction using _set_default.

Examples:

    To create a custom settings class in Python:
    >>> class MySettings(qdk_chemistry.data.Settings):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self._set_default("method", "string", "default")
    ...         self._set_default("max_iterations", "int", 100)
    ...         self._set_default("convergence_threshold", "double", 1e-6)
    ...
    >>> settings = MySettings()
    >>> settings["method"] = "hf"
    >>> settings.method = "hf"     # Alternative access
    >>> settings["max_iterations"] = 200
    >>> settings["convergence_threshold"] = 1e-8
    >>> value = settings["method"]
    >>> print("convergence_threshold" in settings)
    >>> print(len(settings))
    >>>
    >>> # Iterator functionality
    >>> for key in settings:
    ...     print(key, settings[key])
    >>> for key in settings.keys():
    ...     print(key)
    >>> for value in settings.values():
    ...     print(value)
    >>> for key, value in settings.items():
    ...     print(f"{key}: {value}")
    >>>
    >>> print(settings.to_dict())  # Convert to dict

    Alternative: If you have an existing Settings object

        >>> settings = get_settings_from_somewhere()  # Already has keys defined
        >>> settings["method"] = "hf"  # This works if "method" key exists
        >>> settings.method = "hf"     # This also works
)");

  // Constructors
  settings.def(py::init<>(),
               R"(
Default constructor.

Creates an empty Settings object with no key-value pairs.

Examples:
    >>> settings = Settings()
)");

  settings.def(
      "set",
      [](Settings &self, const std::string &key, const py::object &value) {
        // Look up the expected type for this key
        if (!self.has(key)) {
          throw SettingNotFound(key);
        }
        std::string expected_type = self.get_type_name(key);

        // Convert Python object to SettingValue based on expected type
        SettingValue setting_value =
            python_to_setting_value_with_type(value, expected_type, key);
        self.set(key, setting_value);
      },
      R"(
Set a setting value (accepts any Python object).

This method allows setting values using any supported Python type, which will be automatically converted to the appropriate SettingValue variant.
The value must match the expected type for the setting key.

Args:
    key (str): The setting key name
    value (object): The value to set (bool, int, float, str, list, tuple, numpy array)

Raises:
    SettingNotFound: If the key does not exist in settings
    SettingTypeMismatch: If the value type does not match the expected type for this key
    RuntimeError: If the value type is not supported or if the key cannot be set

Examples:
    >>> settings.set("method", "hf")
    >>> settings.set("max_iterations", 100)
    >>> settings.set("convergence_threshold", 1e-6)
    >>> settings.set("parameters", [1.0, 2.0, 3.0])
)",
      py::arg("key"), py::arg("value"));

  // Keep the original SettingValue version for internal use
  settings.def("set_raw",
               static_cast<void (Settings::*)(
                   const std::string &, const SettingValue &)>(&Settings::set),
               R"(
Set a setting value using a SettingValue variant.

This is the raw interface that accepts a SettingValue directly, primarily for internal use or advanced scenarios.

Args:
    key (str): The setting key name
    value (SettingValue): The SettingValue variant to set

Raises:
    RuntimeError: If the key cannot be set
)",
               py::arg("key"), py::arg("value"));

  settings.def(
      "get",
      [](const Settings &self, const std::string &key) {
        return setting_value_to_python(self.get(key));
      },
      R"(
Get a setting value as a Python object.

Retrieves the value associated with the given key and converts it to the appropriate Python type.

Args:
    key (str): The setting key name

Returns:
    object: The setting value as a Python object (bool, int, float, str, list)

Raises:
    SettingNotFound: If the key is not found in the settings

Examples:
    >>> method = settings.get("method")
    >>> max_iter = settings.get("max_iterations")
    >>> convergence_threshold = settings.get("convergence_threshold")
)",
      py::arg("key"));

  settings.def(
      "get_raw",
      static_cast<SettingValue (Settings::*)(const std::string &) const>(
          &Settings::get),
      R"(
Get a setting value as a ``SettingValue`` variant.

This is the raw interface that returns a ``SettingValue`` directly, primarily for internal use or advanced scenarios.

Args:
    key (str): The setting key name

Returns:
    SettingValue: The SettingValue variant

Raises:
    SettingNotFound: If the key is not found in the settings
)",
      py::arg("key"));

  settings.def(
      "get_or_default",
      [](const Settings &self, const std::string &key,
         const py::object &default_value) {
        // If key exists, return the existing value
        if (self.has(key)) {
          return setting_value_to_python(self.get(key));
        } else {
          // Key doesn't exist, just return the default value as-is
          return default_value;
        }
      },
      R"(
Get a setting value with a default if not found (accepts any Python object).

Retrieves the value for the given key if it exists, or returns the default value if the key is not found.

Args:
    key (str): The setting key name
    default_value (object): Default value to return if key not found

Returns:
    object: The setting value or default value as a Python object

Examples:
    >>> method = settings.get_or_default("method", "default_method")
    >>> max_iter = settings.get_or_default("max_iterations", 1000)
    >>> params = settings.get_or_default("parameters", [])
)",
      py::arg("key"), py::arg("default_value"));

  // Keep the original SettingValue version for internal use
  settings.def("get_or_default_raw",
               static_cast<SettingValue (Settings::*)(
                   const std::string &, const SettingValue &) const>(
                   &Settings::get_or_default),
               R"(
Get a setting value with a default if not found (SettingValue).

Raw interface that works with SettingValue variants directly.

Args:
    key (str): The setting key name
    default_value (SettingValue): Default ``SettingValue`` to return if key not found

Returns:
    SettingValue: The ``SettingValue`` variant or default value
)",
               py::arg("key"), py::arg("default_value"));

  settings.def("has", &Settings::has,
               R"(
Check if a setting exists.

Args:
    key (str): The setting key name to check

Returns:
    bool: True if the setting exists, False otherwise

Examples:
    >>> if settings.has("method"):
    ...     method = settings.get("method")
    >>> exists = settings.has("nonexistent_key")  # False
)",
               py::arg("key"));

  settings.def("keys", &Settings::keys,
               R"(
Get all setting keys.

Returns:
    list[str]: List of all setting key names

Examples:
    >>> all_keys = settings.keys()
    >>> for key in all_keys:
    ...     print(f"{key}: {settings[key]}")
)");

  settings.def("size", &Settings::size,
               R"(
Get the number of settings.

Returns:
    int: Number of setting key-value pairs

Examples:
    >>> count = settings.size()
    >>> print(f"Settings contain {count} entries")
)");

  settings.def("empty", &Settings::empty,
               R"(
Check if settings are empty.

Returns:
    bool: True if no settings are stored, False otherwise

Examples:
    >>> if settings.empty():
    ...     print("No settings configured")
)");

  settings.def("get_as_string", &Settings::get_as_string,
               R"(
Get a setting value as a string representation.

Converts any setting value to its string representation, regardless of the original type.

Args:
    key (str): The setting key name

Returns:
    str: String representation of the setting value

Raises:
    SettingNotFound: If the key is not found

Examples:
    >>> str_val = settings.get_as_string("convergence_threshold")  # "1e-06"
    >>> str_val = settings.get_as_string("max_iterations")  # "100"
)",
               py::arg("key"));

  // JSON serialization
  settings.def(
      "to_json",
      [](const Settings &self) -> std::string {
        return self.to_json().dump(2);
      },
      R"(
Convert settings to JSON format.

Serializes all settings to a JSON string with pretty formatting.
The JSON format preserves all type information and can be used to reconstruct the settings later.

Returns:
    str: JSON string representation of all settings

Examples:
    >>> json_str = settings.to_json()
    >>> print(json_str)
    {
        "method": "hf",
        "max_iterations": 100,
        "convergence_threshold": 1e-06
    }
)");

  settings.def_static(
      "from_json",
      [](const std::string &json_str) {
        return Settings::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load settings from JSON format.

Parses a JSON string and loads all contained settings, replacing any existing settings.
The JSON format should match the output of ``to_json()``.

Args:
    json_str (str): JSON string containing settings data

Raises:
    RuntimeError: If the JSON string is malformed or contains unsupported types

Examples:
    >>> json_data = '{"method": "hf", "max_iterations": 100}'
    >>> settings.from_json(json_data)
    >>> assert settings["method"] == "hf"
)",
      py::arg("json_str"));

  settings.def(
      "to_json_string",
      [](const Settings &self) -> std::string {
        return self.to_json().dump(2);
      },
      R"(
Convert settings to JSON string format.

This is an alias for ``to_json()`` provided for backward compatibility.
Both methods return the same JSON string representation.

Returns:
    str: JSON string representation of all settings

Examples:
    >>> json_str = settings.to_json_string()
    >>> # Equivalent to:
    >>> json_str = settings.to_json()
)");

  settings.def_static(
      "from_json_string",
      [](const std::string &json_str) {
        return Settings::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load settings from JSON string format.

This is an alias for ``from_json()`` provided for backward compatibility.
Both methods accept the same JSON string format.

Args:
    json_str (str): JSON string containing settings data

Raises:
    RuntimeError: If the JSON string is malformed or contains unsupported types

Examples:
    >>> settings.from_json_string('{"method": "hf"}')
    >>> # Equivalent to:
    >>> settings.from_json('{"method": "hf"}')
)",
      py::arg("json_str"));

  // JSON file serialization
  settings.def("to_json_file", settings_to_json_file_wrapper,
               R"(
Save settings to JSON file.

Writes all settings to a JSON file with pretty formatting.
The file will be created or overwritten if it already exists.

Args:
    filename (str | pathlib.Path): Path to the JSON file to write.
        Must have '.settings' before the file extension (e.g., ``config.settings.json``, ``params.settings.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened or written

Examples:
    >>> settings.to_json_file("config.settings.json")
    >>> settings.to_json_file("params.settings.json")
    >>> from pathlib import Path
    >>> settings.to_json_file(Path("config.settings.json"))
)",
               py::arg("filename"));

  settings.def_static("from_json_file", settings_from_json_file_wrapper,
                      R"(
Load settings from JSON file.

Reads settings from a JSON file, replacing any existing settings.
The file should contain JSON data in the format produced by ``to_json_file()``.

Args:
    filename (str | pathlib.Path): Path to the JSON file to read.
        Must have '.settings' before the file extension (e.g., ``config.settings.json``, ``params.settings.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains malformed JSON

Examples:
    >>> settings.from_json_file("config.settings.json")
    >>> settings.from_json_file("params.settings.json")
    >>> from pathlib import Path
    >>> settings.from_json_file(Path("config.settings.json"))
)",
                      py::arg("filename"));

  // Note: set_default methods are not exposed as they are protected in C++
  // Default values should be set in derived class constructors

  settings.def("validate_required", &Settings::validate_required,
               R"(
Validate that all required settings are present.

Checks that all keys in the provided list exist in the settings.
This is useful for ensuring that all necessary configuration parameters have been set before proceeding with computations.

Args:
    required_keys (list[str]): List of setting keys that must be present

Raises:
    SettingNotFound: If any required key is missing

Examples:
    >>> required = ["method", "max_iter"]
    >>> settings.validate_required(required)
    >>> # Raises SettingNotFound if any key is missing
)",
               py::arg("required_keys"));

  settings.def("lock", &Settings::lock,
               R"(
Lock the settings to prevent further modifications.

Once settings are locked, any attempt to modify them will raise an exception.
This is useful to ensure that settings remain unchanged after they have been validated or after a computation has started.

Notes:
    Locking is irreversible for the lifetime of the Settings object.
    To modify settings after locking, create a copy of the Settings object.

Examples:
    >>> settings["method"] = "hf"
    >>> settings["max_iterations"] = 100
    >>> settings.lock()
    >>> # Any further modifications will raise SettingsAreLocked exception
    >>> settings["method"] = "dft"  # Raises SettingsAreLocked
)");

  settings.def(
      "update",
      [](Settings &self, const std::string &key, const py::object &value) {
        if (!self.has(key)) {
          throw SettingNotFound(key);
        }
        std::string expected_type = self.get_type_name(key);
        SettingValue setting_value =
            python_to_setting_value_with_type(value, expected_type, key);
        self.update(key, setting_value);
      },
      R"(
Update a setting value, throwing if key doesn't exist.

Unlike ``set()``, this method will only update existing settings and will raise an exception if the key is not already present.
Accepts any Python object for the value, which must match the expected type.

Args:
    dict (dict): Python dictionary containing settings to update, or instead
    key (str): The setting key name (must already exist)
    value (object): The new value to set

Raises:
    SettingNotFound: If the key doesn't exist in the settings
    SettingTypeMismatch: If the value type does not match the expected type for this key
    RuntimeError: If the value type is not supported

Examples:
    >>> settings.update("max_iterations", 200)  # OK if key exists
    >>> settings.update("new_key", "value")     # Raises SettingNotFound

    >>> updates = {
    ...     'max_iterations': 200,
    ...     'tolerance': 1e-8
    ... }
    >>> settings.update(updates)  # OK if both keys exist
    >>> settings.update({'new_key': 'value'})  # Raises SettingNotFound
)",
      py::arg("key"), py::arg("value"));

  // Update from dictionary
  settings.def(
      "update",
      [](Settings &self, const py::dict &dict) {
        for (auto item : dict) {
          std::string key = py::str(item.first);
          py::object value = py::reinterpret_borrow<py::object>(item.second);
          if (!self.has(key)) {
            throw SettingNotFound(key);
          }
          std::string expected_type = self.get_type_name(key);
          SettingValue setting_value =
              python_to_setting_value_with_type(value, expected_type, key);
          self.update(key, setting_value);
        }
      },
      py::arg("dict"));

  // Keep the original SettingValue version for internal use
  settings.def(
      "update_raw",
      static_cast<void (Settings::*)(const std::string &,
                                     const SettingValue &)>(&Settings::update),
      R"(
Update a setting value using ``SettingValue``, throwing if key doesn't exist.

Raw interface that works with ``SettingValue`` variants directly.

Args:
    key (str): The setting key name (must already exist)
    value (SettingValue): The new SettingValue to set

Raises:
    SettingNotFound: If the key doesn't exist in the settings
)",
      py::arg("key"), py::arg("value"));

  settings.def("get_type_name", &Settings::get_type_name,
               R"(
Get the type name of a setting value.

Returns a string describing the current type of the value stored for the given key.

Args:
    key (str): The setting key name

Returns:
    str: Type name (e.g., "int", "double", "string", "vector<int>")

Raises:
    SettingNotFound: If the key is not found

Examples:
    >>> type_name = settings.get_type_name("max_iterations")  # "int"
    >>> type_name = settings.get_type_name("convergence_threshold")  # "double"
)",
               py::arg("key"));

  settings.def("has_description", &Settings::has_description,
               R"(
Check if a setting has a description.

Args:
    key (str): The setting key name

Returns:
    bool: True if the setting has a description, False otherwise

Examples:
    >>> if settings.has_description("max_iterations"):
    ...     desc = settings.get_description("max_iterations")
)",
               py::arg("key"));

  settings.def("get_description", &Settings::get_description,
               R"(
Get the description of a setting.

Args:
    key (str): The setting key name

Returns:
    str: The description string

Raises:
    SettingNotFound: If the key doesn't exist or has no description

Examples:
    >>> desc = settings.get_description("max_iterations")
    >>> print(desc)  # "Maximum number of iterations"
)",
               py::arg("key"));

  settings.def("has_limits", &Settings::has_limits,
               R"(
Check if a setting has defined limits.

Args:
    key (str): The setting key name

Returns:
    bool: True if the setting has limits defined, False otherwise

Examples:
    >>> if settings.has_limits("max_iterations"):
    ...     limits = settings.get_limits("max_iterations")
)",
               py::arg("key"));

  settings.def(
      "get_limits",
      [](const Settings &self, const std::string &key) -> py::object {
        Constraint limits = self.get_limits(key);
        return std::visit(
            [](const auto &variant_value) -> py::object {
              using LimitType = std::decay_t<decltype(variant_value)>;
              if constexpr (std::is_same_v<LimitType,
                                           BoundConstraint<int64_t>>) {
                return py::cast(
                    std::make_tuple(variant_value.min, variant_value.max));
              } else if constexpr (std::is_same_v<LimitType,
                                                  BoundConstraint<double>>) {
                return py::cast(
                    std::make_tuple(variant_value.min, variant_value.max));
              } else if constexpr (std::is_same_v<LimitType,
                                                  ListConstraint<int64_t>>) {
                return py::cast(variant_value.allowed_values);
              } else if constexpr (std::is_same_v<
                                       LimitType,
                                       ListConstraint<std::string>>) {
                return py::cast(variant_value.allowed_values);
              } else {
                return py::none();
              }
            },
            limits);
      },
      R"(
Get the limits of a setting.

Returns the limit value which can be a range (tuple of min, max) or
an enumeration (list of allowed values).

Args:
    key (str): The setting key name

Returns:
    tuple[int, int] | tuple[float, float] | list[int] | list[str]:

        The limit value - either a range tuple or a list of allowed values

Raises:
    SettingNotFound: If the key doesn't exist or has no limits

Examples:
    >>> limits = settings.get_limits("max_iterations")
    >>> print(limits)  # (1, 1000) for a range
    >>>
    >>> limits = settings.get_limits("method")
    >>> print(limits)  # ['hf', 'dft', 'mp2'] for allowed values
)",
      py::arg("key"));

  settings.def("is_documented", &Settings::is_documented,
               R"(
Check if a setting is documented.

Args:
    key (str): The setting key name

Returns:
    bool: True if the setting is marked as documented, False otherwise

Raises:
    SettingNotFound: If the key doesn't exist

Examples:
    >>> if settings.is_documented("max_iterations"):
    ...     print("This setting is documented")
)",
               py::arg("key"));

  settings.def("as_table", &Settings::as_table,
               R"(
Print settings as a formatted table.

Prints all documented settings in a table format with columns:
Key, Value, Limits, Description

The table fits within the specified width with multi-line descriptions
as needed. Non-integer numeric values are displayed in scientific notation.

Args:
    max_width (int): Maximum total width of the table (default: 120)
    show_undocumented (bool): Whether to show undocumented settings (default: False)

Returns:
    str: Formatted table string

Examples:
    >>> print(settings.as_table())
    ------------------------------------------------------------
    Key                  | Value           | Limits              | Description
    ------------------------------------------------------------
    max_iterations       | 100             | [1, 1000]           | Maximum number of iterations
    method               | "hf"            | ["hf", "dft"...]    | Electronic structure method
    tolerance            | 1.00e-06        | [1.00e-08, 1.00...  | Convergence tolerance
    ------------------------------------------------------------

)",
               py::arg("max_width") = 120,
               py::arg("show_undocumented") = false);

  settings.def(
      "get_expected_python_type",
      [](const Settings &self, const std::string &key) -> std::string {
        if (!self.has(key)) {
          throw SettingNotFound(key);
        }
        std::string type_name = self.get_type_name(key);

        // Map C++ type names to Python type descriptions
        if (type_name == "bool") {
          return "bool";
        } else if (type_name == "int") {
          return "int";
        } else if (type_name == "float" || type_name == "double") {
          return "float";
        } else if (type_name == "string") {
          return "str";
        } else if (type_name == "vector<int>") {
          return "list[int]";
        } else if (type_name == "vector<float>" ||
                   type_name == "vector<double>") {
          return "list[float]";
        } else if (type_name == "vector<string>") {
          return "list[str]";
        } else {
          return "unknown";
        }
      },
      R"(
Get the expected Python type for a setting key.

Returns a string describing what Python type should be provided when setting the value for the given key.
This is useful for understanding what type of value is expected before attempting to set it.

Args:
    key (str): The setting key name

Returns:
    str: Expected Python type (e.g., "int", "float", "str", "bool", "list[int]", "list[float]", "list[str]", "list[bool]")

Raises:
    SettingNotFound: If the key is not found

Examples:
    >>> expected = settings.get_expected_python_type("max_iterations")
    >>> print(expected)  # "int"
    >>> expected = settings.get_expected_python_type("convergence_threshold")
    >>> print(expected)  # "float"
    >>> expected = settings.get_expected_python_type("basis_set")
    >>> print(expected)  # "str"
    >>> expected = settings.get_expected_python_type("active_orbitals")
    >>> print(expected)  # "list[int]"
)",
      py::arg("key"));

  // Expose _set_default for Python derived classes to use in __init__
  settings.def(
      "_set_default",
      [](Settings &self, const std::string &key,
         const std::string &expected_type, const py::object &value,
         const py::object &description = py::none(),
         const py::object &limit = py::none(), bool documented = true) {
        // Cast to PySettings to access the protected method
        static_cast<PySettings &>(self)._set_default(
            key, expected_type, value, description, limit, documented);
      },
      R"(
Set a default value (for use in derived class __init__ only).

This method is used internally by derived classes during construction to establish default values for settings.
It should only be called during the ``__init__`` method of derived classes.

Args:
    key (str): The setting key name
    expected_type (str): The expected type name (e.g., "int", "double", "string", "vector<int>")
    value (object): The default value to set
    description (str, optional): Human-readable description of the setting
    limit (tuple | list, optional): Limits for the setting value.

        For numeric types: tuple of (min, max)
        For string types: list of allowed values

    documented (bool): Whether this setting should be included in documentation (default: True)

Notes:
    This method is intended for internal use by derived classes only.
    Regular users should use the ``set()`` method or dictionary-style access.

Examples:
    >>> class MySettings(qdk_chemistry.data.Settings):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self._set_default("method", "string", "hf",
    ...                          "Electronic structure method",
    ...                          ["hf", "dft", "mp2"])
    ...         self._set_default("max_iter", "int", 1000,
    ...                          "Maximum iterations", (1, 10000))
    ...         self._set_default("convergence_threshold", "double", 1e-6,
    ...                          "Convergence threshold", (1e-12, 1e-3))
    ...         self._set_default("debug_mode", "bool", False,
    ...                          documented=False)

)",
      py::arg("key"), py::arg("expected_type"), py::arg("value"),
      py::arg("description") = py::none(), py::arg("limit") = py::none(),
      py::arg("documented") = true);

  // Python dictionary-like interface
  settings.def(
      "__getitem__",
      [](const Settings &self, const std::string &key) {
        return setting_value_to_python(self.get(key));
      },
      R"(
Get setting using ``[]`` operator.

Provides dictionary-style access for retrieving setting values.

Args:
    key (str): The setting key name

Returns:
    object: The setting value as a Python object

Raises:
    SettingNotFound: If the key is not found

Examples:
    >>> method = settings["method"]
    >>> convergence_threshold = settings["convergence_threshold"]
)",
      py::arg("key"));

  settings.def(
      "__setitem__",
      [](Settings &self, const std::string &key, const py::object &value) {
        // Look up the expected type for this key
        if (!self.has(key)) {
          throw SettingNotFound(key);
        }
        std::string expected_type = self.get_type_name(key);

        // Convert Python object to SettingValue based on expected type
        SettingValue setting_value =
            python_to_setting_value_with_type(value, expected_type, key);
        self.set(key, setting_value);
      },
      R"(
Set setting using ``[]`` operator (accepts any Python object).

Provides dictionary-style access for setting values.
The value must match the expected type for the setting key.

Args:
    key (str): The setting key name
    value (object): The value to set

Raises:
    SettingNotFound: If the key does not exist in settings
    SettingTypeMismatch: If the value type does not match the expected type for this key

Examples:
    >>> settings["method"] = "hf"
    >>> settings["max_iterations"] = 100
    >>> settings["parameters"] = [1.0, 2.0, 3.0]
)",
      py::arg("key"), py::arg("value"));

  // Keep original SettingValue version for internal use
  settings.def(
      "__setitem__",
      [](Settings &self, const std::string &key, const SettingValue &value) {
        self.set(key, value);
      },
      R"(
Set setting using ``[]`` operator (SettingValue).

Raw interface for setting values using SettingValue directly.
)",
      py::arg("key"), py::arg("value"));

  settings.def("__contains__", &Settings::has,
               R"(
Check if setting exists using ``in`` operator.

Args:
    key (str): The setting key name to check

Returns:
    bool: True if the setting exists

Examples:
    >>> if "method" in settings:
    ...     print("Method is configured")
    >>> exists = "nonexistent" in settings  # False
)",
               py::arg("key"));

  settings.def("__len__", &Settings::size,
               R"(
Get number of settings using ``len()``.

Returns:
    int: Number of setting key-value pairs

Examples:
    >>> count = len(settings)
    >>> print(f"Settings has {count} entries")
)");

  // Attribute-style access (obj.key and obj.key = value)
  settings.def(
      "__getattr__",
      [](const Settings &self, const std::string &key) {
        if (self.has(key)) {
          return setting_value_to_python(self.get(key));
        } else {
          throw py::attribute_error("Settings object has no attribute '" + key +
                                    "'");
        }
      },
      R"(
Get setting using attribute access (``obj.key``).

Provides object-style attribute access for retrieving setting values.
This is an alternative to dictionary-style access.

Args:
    key (str): The setting key name (as an attribute)

Returns:
    object: The setting value as a Python object

Raises:
    AttributeError: If the key is not found

Examples:
    >>> method = settings.method
    >>> tolerance = settings.tolerance
    >>> max_iter = settings.max_iterations
)",
      py::arg("key"));

  settings.def(
      "__setattr__",
      [](Settings &self, const std::string &key, const py::object &value) {
        // Allow setting any attribute that doesn't conflict with class
        // methods/members This enables both settings.key = value and internal
        // Python attribute setting
        if (!self.has(key)) {
          throw SettingNotFound(key);
        }
        std::string expected_type = self.get_type_name(key);

        // Convert Python object to SettingValue based on expected type
        SettingValue setting_value =
            python_to_setting_value_with_type(value, expected_type, key);
        self.set(key, setting_value);
      },
      R"(
Set setting using attribute access (``obj.key = value``).

Provides object-style attribute access for setting values.
This is an alternative to dictionary-style access.
The value must match the expected type for the setting key.

Args:
        key (str): The setting key name (as an attribute)
        value (object): The value to set

Raises:
    SettingNotFound: If the key does not exist in settings
    SettingTypeMismatch: If the value type does not match the expected type for this key

Examples:
    >>> settings.method = "hf"
    >>> settings.tolerance = 1e-6
    >>> settings.max_iterations = 100
)",
      py::arg("key"), py::arg("value"));

  // Iterator support - iterate over keys by default (like Python dict)
  settings.def(
      "__iter__",
      [](const Settings &self) {
        auto keys = self.keys();  // Get a copy of the keys vector
        return py::iter(py::cast(keys));
      },
      R"(
Iterate over setting keys.

Enables for-loop iteration over the settings keys, similar to iterating over a Python dictionary.

Returns:
    iterator: Iterator over setting key names

Examples:
    >>> for key in settings:
    ...     print(f"{key}: {settings[key]}")
    >>> # Equivalent to:
    >>> for key in settings.keys():
    ...     print(f"{key}: {settings[key]}")
)");

  // Dictionary-style methods for values and items (keys already defined above)
  settings.def(
      "values",
      [](const Settings &self) {
        py::list result;
        for (const auto &[key, value] : self.get_all_settings()) {
          result.append(setting_value_to_python(value));
        }
        return result;
      },
      R"(
Get all setting values as a list.

Returns:
    list: List of all setting values

Examples:
    >>> all_values = settings.values()
    >>> for value in all_values:
    ...     print(value)
    >>> # Or iterate directly:
    >>> for value in settings.values():
    ...     print(value)
)");

  settings.def(
      "items",
      [](const Settings &self) {
        py::list result;
        for (const auto &[key, value] : self.get_all_settings()) {
          py::tuple item = py::make_tuple(key, setting_value_to_python(value));
          result.append(item);
        }
        return result;
      },
      R"(
Get key-value pairs as a list of tuples.

Returns:
    list[tuple]: List of (key, value) tuples

Examples:
    >>> all_items = settings.items()
    >>> for key, value in all_items:
    ...     print(f"{key}: {value}")
    >>> # Or iterate directly:
    >>> for key, value in settings.items():
    ...     print(f"{key}: {value}")
)");

  // Dictionary conversion methods
  settings.def(
      "to_dict",
      [](const Settings &self) {
        py::dict result;
        for (const auto &[key, value] : self.get_all_settings()) {
          result[key.c_str()] = setting_value_to_python(value);
        }
        return result;
      },
      R"(
Convert settings to Python dictionary.

Creates a Python dictionary containing all settings with keys as strings and values converted to appropriate Python types.

Returns:
    dict: Python dictionary with all settings

Examples:
    >>> settings_dict = settings.to_dict()
    >>> print(settings_dict)
    {'method': 'hf', 'max_iterations': 100, 'tolerance': 1e-06}
    >>>
    >>> # Can be used with JSON, pickle, etc.
    >>> import json
    >>> json_str = json.dumps(settings_dict)
)");

  settings.def(
      "from_dict",
      [](Settings &self, const py::dict &dict) {
        for (auto item : dict) {
          std::string key = py::str(item.first);
          py::object value = py::reinterpret_borrow<py::object>(item.second);
          if (!self.has(key)) {
            throw SettingNotFound(key);
          }
          std::string expected_type = self.get_type_name(key);
          SettingValue setting_value =
              python_to_setting_value_with_type(value, expected_type, key);
          self.set(key, setting_value);
        }
      },
      R"(
Load settings from Python dictionary.

Updates existing settings with values from the provided dictionary.
Keys must be strings or convertible to strings.
Only predefined settings keys can be updated.
Values must match expected types.

Args:
    dict (dict): Python dictionary containing settings

Raises:
    SettingNotFound: If any key does not exist in settings
    SettingTypeMismatch: If any value type does not match the expected type for its key

Examples:
    >>> config = {
    ...     'method': 'hf',
    ...     'max_iterations': 100,
    ...     'tolerance': 1e-6,
    ...     'parameters': [1.0, 2.0, 3.0]
    ... }
    >>> settings.from_dict(config)
    >>> assert settings['method'] == 'hf'
)",
      py::arg("dict"));

  // String representation
  settings.def(
      "__repr__",
      [](const Settings &self) {
        return "<qdk_chemistry.Settings size=" + std::to_string(self.size()) +
               ">";
      },
      R"(
Return string representation of the Settings object.

Returns a brief representation showing the number of settings.

Returns:
    str: String representation

Examples:
    >>> repr(settings)
    '<qdk_chemistry.Settings size=3>'
)");

  settings.def(
      "__str__",
      [](const Settings &self) {
        std::string result = "Settings {\n";
        for (const auto &key : self.keys()) {
          result += "  " + key + ": " + self.get_as_string(key) + "\n";
        }
        result += "}";
        return result;
      },
      R"(
Return human-readable string representation of all settings.

Returns a formatted string showing all key-value pairs.

Returns:
    str: Formatted string with all settings

Examples:
    >>> print(settings)
    Settings {
    method: hf
    max_iterations: 100
    tolerance: 1e-06
    }
)");

  // File I/O methods
  settings.def("to_json_file", settings_to_json_file_wrapper,
               R"(
Save settings to JSON file.

Args:
    filename (str | pathlib.Path): Path to output file.
        Must have '.settings.json' extension (e.g., ``config.settings.json``, ``parameters.settings.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened or written

Examples:
    >>> settings.to_json_file("config.settings.json")
    >>> settings.to_json_file("parameters.settings.json")
    >>> from pathlib import Path
    >>> settings.to_json_file(Path("config.settings.json"))
)",
               py::arg("filename"));

  settings.def_static("from_json_file", settings_from_json_file_wrapper,
                      R"(
Load settings from JSON file.

Reads settings from a JSON file, replacing any existing settings.
The file should contain JSON data in the format produced by ``to_json_file()``.

Args:
    filename (str | pathlib.Path): Path to the JSON file to read.
        Must have '.settings' before the file extension (e.g., ``config.settings.json``, ``params.settings.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains malformed JSON

Examples:
    >>> settings.from_json_file("config.settings.json")
    >>> settings.from_json_file("parameters.settings.json")
    >>> from pathlib import Path
    >>> settings.from_json_file(Path("config.settings.json"))
)",
                      py::arg("filename"));

  // Note: set_default methods are not exposed as they are protected in C++
  // Default values should be set in derived class constructors

  settings.def("validate_required", &Settings::validate_required,
               R"(
Validate that all required settings are present.

Checks that all keys in the provided list exist in the settings.
This is useful for ensuring that all necessary configuration parameters have been set before proceeding with computations.

Args:
    required_keys (list[str]): List of setting keys that must be present

Raises:
    SettingNotFound: If any required key is missing

Examples:
    >>> required = ["method", "max_iter"]
    >>> settings.validate_required(required)
    >>> # Raises SettingNotFound if any key is missing
)",
               py::arg("required_keys"));

  settings.def(
      "update",
      [](Settings &self, const std::string &key, const py::object &value) {
        if (!self.has(key)) {
          throw SettingNotFound(key);
        }
        std::string expected_type = self.get_type_name(key);
        SettingValue setting_value =
            python_to_setting_value_with_type(value, expected_type, key);
        self.update(key, setting_value);
      },
      R"(
Update a setting value, throwing if key doesn't exist.

Unlike ``set()``, this method will only update existing settings and will raise an exception if the key is not already present.
Accepts any Python object for the value, which must match the expected type.

Args:
    dict (dict): Python dictionary containing settings to update, or instead
    key (str): The setting key name (must already exist)
    value (object): The new value to set

Raises:
    SettingNotFound: If the key doesn't exist in the settings
    SettingTypeMismatch: If the value type does not match the expected type for this key
    RuntimeError: If the value type is not supported

Examples:
    >>> settings.update("max_iterations", 200)  # OK if key exists
    >>> settings.update("new_key", "value")     # Raises SettingNotFound

    >>> updates = {
    ...     'max_iterations': 200,
    ...     'tolerance': 1e-8
    ... }
    >>> settings.update(updates)  # OK if both keys exist
    >>> settings.update({'new_key': 'value'})  # Raises SettingNotFound
)",
      py::arg("key"), py::arg("value"));

  // Update from dictionary
  settings.def(
      "update",
      [](Settings &self, const py::dict &dict) {
        for (auto item : dict) {
          std::string key = py::str(item.first);
          py::object value = py::reinterpret_borrow<py::object>(item.second);
          if (!self.has(key)) {
            throw SettingNotFound(key);
          }
          std::string expected_type = self.get_type_name(key);
          SettingValue setting_value =
              python_to_setting_value_with_type(value, expected_type, key);
          self.update(key, setting_value);
        }
      },
      py::arg("dict"));

  // Keep the original SettingValue version for internal use
  settings.def(
      "update_raw",
      static_cast<void (Settings::*)(const std::string &,
                                     const SettingValue &)>(&Settings::update),
      R"(
Update a setting value using ``SettingValue``, throwing if key doesn't exist.

Raw interface that works with ``SettingValue`` variants directly.

Args:
    key (str): The setting key name (must already exist)
    value (SettingValue): The new SettingValue to set

Raises:
    SettingNotFound: If the key doesn't exist in the settings
)",
      py::arg("key"), py::arg("value"));

  settings.def("get_type_name", &Settings::get_type_name,
               R"(
Get the type name of a setting value.

Returns a string describing the current type of the value stored for the given key.

Args:
    key (str): The setting key name

Returns:
    str: Type name (e.g., "int", "double", "string", "vector<int>")

Raises:
    SettingNotFound: If the key is not found

Examples:
    >>> type_name = settings.get_type_name("max_iterations")  # "int"
    >>> type_name = settings.get_type_name("convergence_threshold")  # "double"
)",
               py::arg("key"));

  // Expose _set_default for Python derived classes to use in __init__
  settings.def(
      "_set_default",
      [](Settings &self, const std::string &key,
         const std::string &expected_type, const py::object &value,
         const py::object &description, const py::object &limit,
         bool documented) {
        // Cast to PySettings to access the protected method
        static_cast<PySettings &>(self)._set_default(
            key, expected_type, value, description, limit, documented);
      },
      R"(
Set a default value (for use in derived class __init__ only).

This method is used internally by derived classes during construction to establish default values for settings.
It should only be called during the ``__init__`` method of derived classes.

Args:
    key (str): The setting key name
    expected_type (str): The expected type name (e.g., "int", "double", "string", "vector<int>")
    value (object): The default value to set
    description (str, optional): Human-readable description of the setting
    limit (tuple or list, optional): Allowed values or numeric range.
        - For numeric types: tuple of (min, max) e.g., (0, 100) or (0.0, 1.0)
        - For strings: list of allowed values e.g., ["option1", "option2", "option3"]
        - For int vectors: list of allowed integer values
    documented (bool, optional): Whether this setting should appear in as_table() output (default: True)

Notes:
    This method is intended for internal use by derived classes only.
    Regular users should use the ``set()`` method or dictionary-style access.

Examples:
    >>> class MySettings(qdk_chemistry.data.Settings):
    ...     def __init__(self):
    ...         super().__init__()
    ...         # Basic usage
    ...         self._set_default("method", "string", "hf")
    ...         self._set_default("max_iter", "int", 1000)
    ...         self._set_default("tolerance", "double", 1e-6)
    ...
    ...         # With description and numeric limits
    ...         self._set_default("convergence_threshold", "double", 1e-8,
    ...                          description="Convergence threshold",
    ...                          limit=(1e-12, 1e-4))
    ...
    ...         # With string enumeration limits
    ...         self._set_default("encoding", "string", "jordan-wigner",
    ...                          description="Qubit encoding method",
    ...                          limit=["jordan-wigner", "bravyi-kitaev", "parity"])
    ...
    ...         # Undocumented internal setting
    ...         self._set_default("internal_flag", "bool", False, documented=False)
)",
      py::arg("key"), py::arg("expected_type"), py::arg("value"),
      py::arg("description") = py::none(), py::arg("limit") = py::none(),
      py::arg("documented") = true);

  // Python dictionary-like interface
  settings.def(
      "__getitem__",
      [](const Settings &self, const std::string &key) {
        return setting_value_to_python(self.get(key));
      },
      R"(
Get setting using ``[]`` operator.

Provides dictionary-style access for retrieving setting values.

Args:
    key (str): The setting key name

Returns:
    object: The setting value as a Python object

Raises:
    SettingNotFound: If the key is not found

Examples:
    >>> method = settings["method"]
    >>> convergence_threshold = settings["convergence_threshold"]
)",
      py::arg("key"));

  settings.def(
      "__setitem__",
      [](Settings &self, const std::string &key, const py::object &value) {
        // Look up the expected type for this key
        if (!self.has(key)) {
          throw SettingNotFound(key);
        }
        std::string expected_type = self.get_type_name(key);

        // Convert Python object to SettingValue based on expected type
        SettingValue setting_value =
            python_to_setting_value_with_type(value, expected_type, key);
        self.set(key, setting_value);
      },
      R"(
Set setting using ``[]`` operator (accepts any Python object).

Provides dictionary-style access for setting values.
The value must match the expected type for the setting key.

Args:
    key (str): The setting key name
    value (object): The value to set

Raises:
    SettingNotFound: If the key does not exist in settings
    SettingTypeMismatch: If the value type does not match the expected type for this key

Examples:
    >>> settings["method"] = "hf"
    >>> settings["max_iterations"] = 100
    >>> settings["parameters"] = [1.0, 2.0, 3.0]
)",
      py::arg("key"), py::arg("value"));

  // Keep original SettingValue version for internal use
  settings.def(
      "__setitem__",
      [](Settings &self, const std::string &key, const SettingValue &value) {
        self.set(key, value);
      },
      R"(
Set setting using ``[]`` operator (SettingValue).

Raw interface for setting values using SettingValue directly.
)",
      py::arg("key"), py::arg("value"));

  settings.def("__contains__", &Settings::has,
               R"(
Check if setting exists using ``in`` operator.

Args:
    key (str): The setting key name to check

Returns:
    bool: True if the setting exists

Examples:
    >>> if "method" in settings:
    ...     print("Method is configured")
    >>> exists = "nonexistent" in settings  # False
)",
               py::arg("key"));

  settings.def("__len__", &Settings::size,
               R"(
Get number of settings using ``len()``.

Returns:
    int: Number of setting key-value pairs

Examples:
    >>> count = len(settings)
    >>> print(f"Settings has {count} entries")
)");

  // Attribute-style access (obj.key and obj.key = value)
  settings.def(
      "__getattr__",
      [](const Settings &self, const std::string &key) {
        if (self.has(key)) {
          return setting_value_to_python(self.get(key));
        } else {
          throw py::attribute_error("Settings object has no attribute '" + key +
                                    "'");
        }
      },
      R"(
Get setting using attribute access (``obj.key``).

Provides object-style attribute access for retrieving setting values.
This is an alternative to dictionary-style access.

Args:
    key (str): The setting key name (as an attribute)

Returns:
    object: The setting value as a Python object

Raises:
    AttributeError: If the key is not found

Examples:
    >>> method = settings.method
    >>> tolerance = settings.tolerance
    >>> max_iter = settings.max_iterations
)",
      py::arg("key"));

  settings.def(
      "__setattr__",
      [](Settings &self, const std::string &key, const py::object &value) {
        // Allow setting any attribute that doesn't conflict with class
        // methods/members This enables both settings.key = value and internal
        // Python attribute setting
        if (!self.has(key)) {
          throw SettingNotFound(key);
        }
        std::string expected_type = self.get_type_name(key);

        // Convert Python object to SettingValue based on expected type
        SettingValue setting_value =
            python_to_setting_value_with_type(value, expected_type, key);
        self.set(key, setting_value);
      },
      R"(
Set setting using attribute access (``obj.key = value``).

Provides object-style attribute access for setting values.
This is an alternative to dictionary-style access.
The value must match the expected type for the setting key.

Args:
    key (str): The setting key name (as an attribute)
    value (object): The value to set

Raises:
    SettingNotFound: If the key does not exist in settings
    SettingTypeMismatch: If the value type does not match the expected type for this key

Examples:
    >>> settings.method = "hf"
    >>> settings.tolerance = 1e-6
    >>> settings.max_iterations = 100
)",
      py::arg("key"), py::arg("value"));

  // Iterator support - iterate over keys by default (like Python dict)
  settings.def(
      "__iter__",
      [](const Settings &self) {
        auto keys = self.keys();  // Get a copy of the keys vector
        return py::iter(py::cast(keys));
      },
      R"(
Iterate over setting keys.

Enables for-loop iteration over the settings keys, similar to iterating over a Python dictionary.

Returns:
    iterator: Iterator over setting key names

Examples:
    >>> for key in settings:
    ...     print(f"{key}: {settings[key]}")
    >>> # Equivalent to:
    >>> for key in settings.keys():
    ...     print(f"{key}: {settings[key]}")
)");

  // Dictionary-style methods for values and items (keys already defined above)
  settings.def(
      "values",
      [](const Settings &self) {
        py::list result;
        for (const auto &[key, value] : self.get_all_settings()) {
          result.append(setting_value_to_python(value));
        }
        return result;
      },
      R"(
Get all setting values as a list.

Returns:
    list: List of all setting values

Examples:
    >>> all_values = settings.values()
    >>> for value in all_values:
    ...     print(value)
    >>> # Or iterate directly:
    >>> for value in settings.values():
    ...     print(value)
)");

  settings.def(
      "items",
      [](const Settings &self) {
        py::list result;
        for (const auto &[key, value] : self.get_all_settings()) {
          py::tuple item = py::make_tuple(key, setting_value_to_python(value));
          result.append(item);
        }
        return result;
      },
      R"(
Get key-value pairs as a list of tuples.

Returns:
    list[tuple]: List of (key, value) tuples

Examples:
    >>> all_items = settings.items()
    >>> for key, value in all_items:
    ...     print(f"{key}: {value}")
    >>> # Or iterate directly:
    >>> for key, value in settings.items():
    ...     print(f"{key}: {value}")
)");

  // Dictionary conversion methods
  settings.def(
      "to_dict",
      [](const Settings &self) {
        py::dict result;
        for (const auto &[key, value] : self.get_all_settings()) {
          result[key.c_str()] = setting_value_to_python(value);
        }
        return result;
      },
      R"(
Convert settings to Python dictionary.

Creates a Python dictionary containing all settings with keys as strings and values converted to appropriate Python types.

Returns:
    dict: Python dictionary with all settings

Examples:
    >>> settings_dict = settings.to_dict()
    >>> print(settings_dict)
    {'method': 'hf', 'max_iterations': 100, 'tolerance': 1e-06}
    >>>
    >>> # Can be used with JSON, pickle, etc.
    >>> import json
    >>> json_str = json.dumps(settings_dict)
)");

  settings.def(
      "from_dict",
      [](Settings &self, const py::dict &dict) {
        for (auto item : dict) {
          std::string key = py::str(item.first);
          py::object value = py::reinterpret_borrow<py::object>(item.second);
          if (!self.has(key)) {
            throw SettingNotFound(key);
          }
          std::string expected_type = self.get_type_name(key);
          SettingValue setting_value =
              python_to_setting_value_with_type(value, expected_type, key);
          self.set(key, setting_value);
        }
      },
      R"(
Load settings from Python dictionary.

Updates existing settings with values from the provided dictionary.
Keys must be strings or convertible to strings.
Only predefined settings keys can be updated. Values must match expected types.

Args:
    dict (dict): Python dictionary containing settings

Raises:
    SettingNotFound: If any key does not exist in settings
    SettingTypeMismatch: If any value type does not match the expected type for its key

Examples:
    >>> config = {
    ...     'method': 'hf',
    ...     'max_iterations': 100,
    ...     'tolerance': 1e-6,
    ...     'parameters': [1.0, 2.0, 3.0]
    ... }
    >>> settings.from_dict(config)
    >>> assert settings['method'] == 'hf'
)",
      py::arg("dict"));

  // String representation
  settings.def(
      "__repr__",
      [](const Settings &self) {
        return "<qdk_chemistry.Settings size=" + std::to_string(self.size()) +
               ">";
      },
      R"(
Return string representation of the Settings object.

Returns a brief representation showing the number of settings.

Returns:
    str: String representation

Examples:
    >>> repr(settings)
    '<qdk_chemistry.Settings size=3>'
)");

  settings.def(
      "__str__",
      [](const Settings &self) {
        std::string result = "Settings {\n";
        for (const auto &key : self.keys()) {
          result += "  " + key + ": " + self.get_as_string(key) + "\n";
        }
        result += "}";
        return result;
      },
      R"(
Return human-readable string representation of all settings.

Returns a formatted string showing all key-value pairs.

Returns:
    str: Formatted string with all settings

Examples:
    >>> print(settings)
    Settings {
    method: hf
    max_iterations: 100
    tolerance: 1e-06
    }
)");

  // String representation
  settings.def(
      "__repr__",
      [](const Settings &self) {
        return "<qdk_chemistry.Settings size=" + std::to_string(self.size()) +
               ">";
      },
      R"(
Return string representation of the Settings object.

Returns a brief representation showing the number of settings.

Returns:
    str: String representation

Examples:
    >>> repr(settings)
    '<qdk_chemistry.Settings size=3>'
)");

  settings.def(
      "__str__",
      [](const Settings &self) {
        std::string result = "Settings {\n";
        for (const auto &key : self.keys()) {
          result += "  " + key + ": " + self.get_as_string(key) + "\n";
        }
        result += "}";
        return result;
      },
      R"(
Return human-readable string representation of all settings.

Returns a formatted string showing all key-value pairs.

Returns:
    str: Formatted string with all settings

Examples:
    >>> print(settings)
    Settings {
    method: hf
    max_iterations: 100
    tolerance: 1e-06
    }
)");

  // File I/O methods
  settings.def("to_json_file", settings_to_json_file_wrapper,
               R"(
Save settings to JSON file.

Args:
    filename (str | pathlib.Path): Path to output file.
        Must have '.settings.json' extension (e.g., ``config.settings.json``, ``parameters.settings.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened or written

Examples:
    >>> settings.to_json_file("config.settings.json")
    >>> settings.to_json_file("parameters.settings.json")
    >>> from pathlib import Path
    >>> settings.to_json_file(Path("config.settings.json"))
)",
               py::arg("filename"));

  settings.def_static("from_json_file", settings_from_json_file_wrapper,
                      R"(
Load settings from JSON file.

Args:
    filename (str | pathlib.Path): Path to input file.
        Must have '.settings.json' extension (e.g., ``config.settings.json``, ``parameters.settings.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid settings data

Examples:
    >>> settings.from_json_file("config.settings.json")
    >>> settings.from_json_file("parameters.settings.json")
    >>> from pathlib import Path
    >>> settings.from_json_file(Path("config.settings.json"))
)",
                      py::arg("filename"));

  settings.def("to_hdf5_file", &Settings::to_hdf5_file,
               R"(
Save settings to HDF5 file.

Args:
    filename (str): Path to output file. Must have '.settings.h5' extension (e.g., ``config.settings.h5``, ``parameters.settings.h5``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened or written

Examples:
    >>> settings.to_hdf5_file("config.settings.h5")
    >>> settings.to_hdf5_file("parameters.settings.h5")
)",
               py::arg("filename"));

  settings.def_static("from_hdf5_file", settings_from_hdf5_file_wrapper,
                      R"(
Load settings from HDF5 file.

Args:
    filename (str): Path to input file.
        Must have '.settings.h5' extension (e.g., ``config.settings.h5``, ``parameters.settings.h5``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid settings data

Examples:
    >>> settings.from_hdf5_file("config.settings.h5")
    >>> settings.from_hdf5_file("parameters.settings.h5")
)",
                      py::arg("filename"));

  settings.def("to_file", settings_to_file_wrapper,
               R"(
Save settings to file in specified format.

Args:
    filename (str | pathlib.Path): Path to output file.
    format_type (str): Format type ("json" or "hdf5")

Raises:
    RuntimeError: If the file cannot be opened or written

Examples:
    >>> settings.to_file("config.settings.json", "json")
    >>> settings.to_file("config.settings.h5", "hdf5")
    >>> from pathlib import Path
    >>> settings.to_file(Path("config.settings.json"), "json")
)",
               py::arg("filename"), py::arg("format_type"));

  settings.def_static("from_file", settings_from_file_wrapper,
                      R"(
Load settings from file in specified format.

Args:
    filename (str | pathlib.Path): Path to input file.
    format_type (str): Format type ("json" or "hdf5")

Raises:
    RuntimeError: If the file cannot be opened, read, or contains invalid settings data

Examples:
    >>> settings.from_file("config.settings.json", "json")
    >>> settings.from_file("config.settings.h5", "hdf5")
    >>> from pathlib import Path
    >>> settings.from_file(Path("config.settings.json"), "json")
)",
                      py::arg("filename"), py::arg("format_type"));

  // Data type name class attribute
  settings.attr("_data_type_name") = DATACLASS_TO_SNAKE_CASE(Settings);

  // Bind ElectronicStructureSettings class
  py::class_<qdk::chemistry::algorithms::ElectronicStructureSettings, Settings,
             py::smart_holder>(data, "ElectronicStructureSettings", R"(
Base class for electronic structure algorithms settings.

This class extends the base Settings class with default values commonly used in electronic structure calculations such as basis sets, molecular charge, spin multiplicity, and convergence parameters.

The default settings include:

- method: "hf" - The electronic structure method (Hartree-Fock)
- charge: 0 - Molecular charge
- spin_multiplicity: 1 - Spin multiplicity (2S+1)
- basis_set: "def2-svp" - Default basis set
- tolerance: 1e-6 - Convergence tolerance
- max_iterations: 50 - Maximum number of iterations

These defaults can be overridden by setting new values after instantiation.

Examples:
    >>> import qdk_chemistry.data as data
    >>> settings = data.ElectronicStructureSettings()
    >>> print(settings.method)  # "hf"
    >>> print(settings.basis_set)  # "def2-svp"
    >>> settings.basis_set = "6-31G*"  # Override default
    >>> settings.charge = -1  # Set molecular charge
    >>> settings.max_iterations = 100  # Increase max iterations
)")
      .def(py::init<>(), R"(
Create ElectronicStructureSettings with default values.

Initializes settings with sensible defaults for electronic structure
calculations. All defaults can be modified after construction.
)");
}
