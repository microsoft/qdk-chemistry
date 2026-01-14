// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry.hpp>
#include <qdk/chemistry/data/configuration_set.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;

// Wrapper functions that accept both strings and pathlib.Path objects
namespace {

void configuration_set_to_file_wrapper(
    qdk::chemistry::data::ConfigurationSet& self, const py::object& filename,
    const std::string& format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

qdk::chemistry::data::ConfigurationSet configuration_set_from_file_wrapper(
    const py::object& filename, const std::string& format_type) {
  return qdk::chemistry::data::ConfigurationSet::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

void configuration_set_to_json_file_wrapper(
    qdk::chemistry::data::ConfigurationSet& self, const py::object& filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

qdk::chemistry::data::ConfigurationSet configuration_set_from_json_file_wrapper(
    const py::object& filename) {
  return qdk::chemistry::data::ConfigurationSet::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void configuration_set_to_hdf5_file_wrapper(
    qdk::chemistry::data::ConfigurationSet& self, const py::object& filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

qdk::chemistry::data::ConfigurationSet configuration_set_from_hdf5_file_wrapper(
    const py::object& filename) {
  return qdk::chemistry::data::ConfigurationSet::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

}  // namespace

void bind_configuration_set(pybind11::module& data) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  py::class_<ConfigurationSet, DataClass, py::smart_holder> configuration_set(
      data, "ConfigurationSet", R"(
Represents a collection of electronic configurations with associated orbital information.

This class manages a set of Configuration objects that share the same single-particle
basis (specifically the active space of an Orbitals object). By storing the orbital
information at the set level rather than in each configuration, redundant storage
of orbital information is eliminated.

Key design points:

- Configurations represent only the active space, not the full orbital set
- Inactive and virtual orbitals are not included in the configuration representation
- All configurations in the set must be consistent with the active space size
- Provides iteration and access methods similar to a Python list
- Immutable after construction to ensure consistency

Common use cases:

- Storing selected CI configurations
- Representing determinant spaces for multi-reference calculations
- Managing configuration lists for projected multi-configuration calculations

Examples:
    Create a ConfigurationSet from configurations and orbitals:

    >>> import qdk_chemistry.data as data
    >>> configs = [data.Configuration("2200"), data.Configuration("22ud")]
    >>> config_set = data.ConfigurationSet(configs, orbitals)

    Access configurations:

    >>> len(config_set)  # Number of configurations
    2
    >>> config_set[0]  # First configuration
    >>> for config in config_set:
    ...     print(config.to_string())

    Get orbital information:

    >>> orbs = config_set.get_orbitals()

    File I/O:

    >>> config_set.to_json_file("configs.json")
    >>> config_set.to_hdf5_file("configs.h5")
    >>> loaded = data.ConfigurationSet.from_json_file("configs.json")

)");

  // Constructors
  configuration_set.def(
      py::init<const std::vector<Configuration>&, std::shared_ptr<Orbitals>>(),
      R"(
Construct a ConfigurationSet from configurations and orbital information.

Args:
    configurations (list[Configuration]): List of Configuration objects representing the active space
    orbitals (Orbitals): Orbitals object defining the single-particle basis

Raises:
    ValueError: If configurations are inconsistent with active space or if orbitals is None

Note:
    All configurations must have the same number of orbitals and sufficient capacity
    to represent the active space defined in the orbitals object. Configurations only
    represent the active space; inactive and virtual orbitals are not included.

Examples:
    >>> configs = [Configuration("2200"), Configuration("22ud")]
    >>> config_set = ConfigurationSet(configs, orbitals)

)",
      py::arg("configurations"), py::arg("orbitals"));

  // Copy constructor
  configuration_set.def(py::init<const ConfigurationSet&>(), "Copy constructor",
                        py::arg("other"));

  // Configurations access
  bind_getter_as_property(configuration_set, "get_configurations",
                          &ConfigurationSet::get_configurations,
                          R"(
Get the list of configurations.

Returns:
    list[Configuration]: List of Configuration objects in the set

)",
                          py::return_value_policy::reference_internal);

  // Orbitals access
  bind_getter_as_property(configuration_set, "get_orbitals",
                          &ConfigurationSet::get_orbitals,
                          R"(
Get the orbital information.

Returns:
    Orbitals: Shared pointer to the Orbitals object

)",
                          py::return_value_policy::reference_internal);

  // Size methods
  configuration_set.def("__len__", &ConfigurationSet::size,
                        R"(
Get the number of configurations in the set.

Returns:
    int: Number of configurations

)");

  configuration_set.def("empty", &ConfigurationSet::empty,
                        R"(
Check if the set is empty.

Returns:
    bool: True if the set contains no configurations

)");

  // Item access
  configuration_set.def(
      "__getitem__",
      [](const ConfigurationSet& self, size_t idx) -> const Configuration& {
        if (idx >= self.size()) {
          throw py::index_error("ConfigurationSet index out of range");
        }
        return self[idx];
      },
      R"(
Access configuration by index.

Args:
    idx (int): Index of the configuration

Returns:
    Configuration: The configuration at the given index

Raises:
    IndexError: If index is out of range

)",
      py::arg("idx"), py::return_value_policy::reference_internal);

  configuration_set.def("at", &ConfigurationSet::at,
                        R"(
Access configuration by index with bounds checking.

Args:
    idx (int): Index of the configuration

Returns:
    Configuration: The configuration at the given index

Raises:
    IndexError: If index is out of range

)",
                        py::arg("idx"),
                        py::return_value_policy::reference_internal);

  // Iterator support
  configuration_set.def(
      "__iter__",
      [](const ConfigurationSet& self) {
        return py::make_iterator(self.begin(), self.end());
      },
      py::keep_alive<0, 1>(),
      R"(
Iterate over configurations in the set.

Returns:
    iterator: Iterator over Configuration objects

)");

  // Equality operators
  configuration_set.def("__eq__", &ConfigurationSet::operator==,
                        R"(
Check equality with another ConfigurationSet.

Args:
    other (ConfigurationSet): Another ConfigurationSet to compare

Returns:
    bool: True if the sets are equal (same configurations and orbitals)

)",
                        py::arg("other"));

  configuration_set.def("__ne__", &ConfigurationSet::operator!=,
                        R"(
Check inequality with another ConfigurationSet.

Args:
    other (ConfigurationSet): Another ConfigurationSet to compare

Returns:
    bool: True if the sets are not equal

)",
                        py::arg("other"));

  // Summary
  bind_getter_as_property(configuration_set, "get_summary",
                          &ConfigurationSet::get_summary,
                          R"(
Get a summary string describing the ConfigurationSet.

Returns:
    str: Human-readable summary of the ConfigurationSet

)");

  // Generic File I/O
  configuration_set.def("to_file", &configuration_set_to_file_wrapper,
                        R"(
Save to file based on type parameter.

Args:
    filename (str | pathlib.Path): Path to file to create/overwrite
    type (str): File format type ("json" or "hdf5")

Raises:
    RuntimeError: If unsupported type or I/O error occurs

)",
                        py::arg("filename"), py::arg("type"));

  configuration_set.def_static("from_file",
                               &configuration_set_from_file_wrapper,
                               R"(
Load from file based on type parameter.

Args:
    filename (str | pathlib.Path): Path to file to read
    type (str): File format type ("json" or "hdf5")

Returns:
    ConfigurationSet: New ConfigurationSet loaded from file

Raises:
    RuntimeError: If file doesn't exist, unsupported type, or I/O error occurs

)",
                               py::arg("filename"), py::arg("type"));

  // JSON File I/O
  configuration_set.def("to_json_file", &configuration_set_to_json_file_wrapper,
                        R"(
Save ConfigurationSet to JSON file.

Args:
    filename (str | pathlib.Path): Path to JSON file to create/overwrite

Raises:
    RuntimeError: If I/O error occurs

)",
                        py::arg("filename"));

  configuration_set.def_static("from_json_file",
                               &configuration_set_from_json_file_wrapper,
                               R"(
Load ConfigurationSet from JSON file.

Args:
    filename (str | pathlib.Path): Path to JSON file to read

Returns:
    ConfigurationSet: ConfigurationSet loaded from file

Raises:
    RuntimeError: If file doesn't exist or I/O error occurs

)",
                               py::arg("filename"));

  // HDF5 File I/O
  configuration_set.def("to_hdf5_file", &configuration_set_to_hdf5_file_wrapper,
                        R"(
Save ConfigurationSet to HDF5 file.

Args:
    filename (str | pathlib.Path): Path to HDF5 file to create/overwrite

Raises:
    RuntimeError: If I/O error occurs

)",
                        py::arg("filename"));

  configuration_set.def_static("from_hdf5_file",
                               &configuration_set_from_hdf5_file_wrapper,
                               R"(
Load ConfigurationSet from HDF5 file.

Args:
    filename (str | pathlib.Path): Path to HDF5 file to read

Returns:
    ConfigurationSet: ConfigurationSet loaded from file

Raises:
    RuntimeError: If file doesn't exist or I/O error occurs

)",
                               py::arg("filename"));

  // JSON serialization (in-memory)
  configuration_set.def(
      "to_json",
      [](const ConfigurationSet& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert ConfigurationSet to JSON string format.

Returns:
    str: JSON string containing ConfigurationSet data

)");

  configuration_set.def_static(
      "from_json",
      [](const std::string& json_str) -> ConfigurationSet {
        return ConfigurationSet::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load ConfigurationSet from JSON string format.

Args:
    json_str (str): JSON string containing ConfigurationSet data

Returns:
    ConfigurationSet: ConfigurationSet loaded from JSON string

Raises:
    RuntimeError: If JSON string is malformed

)",
      py::arg("json_str"));

  // String representation
  configuration_set.def("__repr__", [](const ConfigurationSet& self) {
    return self.get_summary();
  });

  configuration_set.def("__str__", [](const ConfigurationSet& self) {
    return self.get_summary();
  });

  // Pickling support using JSON serialization
  configuration_set.def(py::pickle(
      [](const ConfigurationSet& cs) {
        // __getstate__ - serialize to JSON string
        return cs.to_json().dump();
      },
      [](const std::string& json_str) {
        // __setstate__ - deserialize from JSON string
        return ConfigurationSet::from_json(nlohmann::json::parse(json_str));
      }));
}
