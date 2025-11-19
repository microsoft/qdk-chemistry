// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace qdk::chemistry::python::utils {

/**
 * @brief Convert a Python path-like object to a string path
 *
 * This function accepts both string objects and pathlib Path objects,
 * converting them to std::string for use with C++ file operations.
 *
 * @param path_obj Python object that should be a string or pathlib Path
 * @return std::string The path as a string
 * @throws std::runtime_error If the object is not a valid path-like object
 */
inline std::string to_string_path(const pybind11::object& path_obj) {
  namespace py = pybind11;

  // Try to cast directly to string first (works for strings)
  try {
    return path_obj.cast<std::string>();
  } catch (const py::cast_error&) {
    // If that fails, try to use __fspath__() method (for pathlib objects)
    try {
      if (py::hasattr(path_obj, "__fspath__")) {
        py::object fspath_result = path_obj.attr("__fspath__")();
        return fspath_result.cast<std::string>();
      }
    } catch (const py::cast_error&) {
      // Ignore and continue
    }

    // Try str() conversion as final fallback
    try {
      return py::str(path_obj).cast<std::string>();
    } catch (const py::cast_error&) {
      throw std::runtime_error(
          "Path argument must be a string or pathlib Path object");
    }
  }
}

}  // namespace qdk::chemistry::python::utils
