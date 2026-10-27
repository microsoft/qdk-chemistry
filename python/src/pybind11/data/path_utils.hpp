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

  // Dispatch on the object's type rather than by letting a failed cast throw.
  // The exception-driven form relied on py::cast_error propagating out of
  // pybind11 for every pathlib.Path argument, which crashed with an access
  // violation under clang-cl on Windows ARM64. Testing the type first is also
  // cheaper: the common paths no longer throw and catch an exception at all.
  if (py::isinstance<py::str>(path_obj) ||
      py::isinstance<py::bytes>(path_obj)) {
    return path_obj.cast<std::string>();
  }

  // os.PathLike (pathlib.Path and friends) expose __fspath__().
  if (py::hasattr(path_obj, "__fspath__")) {
    py::object fspath_result = path_obj.attr("__fspath__")();
    if (py::isinstance<py::str>(fspath_result) ||
        py::isinstance<py::bytes>(fspath_result)) {
      return fspath_result.cast<std::string>();
    }
  }

  throw std::runtime_error(
      "Path argument must be a string or pathlib Path object");
}

}  // namespace qdk::chemistry::python::utils
