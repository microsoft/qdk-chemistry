// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <qdk/chemistry/scf/config.h>

#include <filesystem>

namespace py = pybind11;
namespace qcs = qdk::chemistry::scf;

void bind_qdk_chemistry_config(py::module& m) {
  // Bind QDKChemistryConfig as a class
  py::class_<qcs::QDKChemistryConfig>(m, "QDKChemistryConfig", R"(
QDKChemistry Configuration class.

This class provides access to QDKChemistry configuration functionality, including resource directory management and other configuration options.

Examples:
    >>> import qdk_chemistry._core as core
    >>> config = core.QDKChemistryConfig()
    >>>
    >>> # Get current resources directory
    >>> current_dir = config.get_resources_dir()
    >>> print(f"Current resources directory: {current_dir}")
    >>>
    >>> # Set custom resources directory
    >>> config.set_resources_dir("/path/to/custom/resources")
    >>>
    >>> # Verify the change
    >>> new_dir = config.get_resources_dir()
    >>> print(f"New resources directory: {new_dir}")
    >>>
    >>> # Or use static methods directly
    >>> core.QDKChemistryConfig.get_resources_dir()
    >>> core.QDKChemistryConfig.set_resources_dir("/another/path")
)")
      .def_static(
          "get_resources_dir",
          []() {
            return qcs::QDKChemistryConfig::get_resources_dir().string();
          },
          R"(
Get the current resources directory path.

Returns:
    str: The current resources directory path as a string.

Examples:
    >>> import qdk_chemistry._core as core
    >>> resources_path = core.QDKChemistryConfig.get_resources_dir()
    >>> print(f"Resources directory: {resources_path}")
)")
      .def_static(
          "set_resources_dir",
          [](const std::string& path) {
            qcs::QDKChemistryConfig::set_resources_dir(
                std::filesystem::path(path));
          },
          py::arg("path"),
          R"(
Set the resources directory path.

Parameters:
    path (str): The new resources directory path.

Examples:
    >>> import qdk_chemistry._core as core
    >>> core.QDKChemistryConfig.set_resources_dir("/path/to/custom/resources")
    >>> # Verify the change
    >>> print(core.QDKChemistryConfig.get_resources_dir())
)");
}
