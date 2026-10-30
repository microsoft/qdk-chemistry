// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <nanobind/nanobind.h>

#include <string>

#include "stl.hpp"

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
inline std::string to_string_path(const nanobind::object& path_obj) {
  namespace nb = nanobind;

  // Dispatch on the object's type rather than by letting a failed cast throw.
  // The exception-driven form relied on nb::cast_error propagating out of
  // nanobind for every pathlib.Path argument, which crashed with an access
  // violation under clang-cl on Windows ARM64. Testing the type first is also
  // cheaper: the common paths no longer throw and catch an exception at all.
  if (nb::isinstance<nb::str>(path_obj) ||
      nb::isinstance<nb::bytes>(path_obj)) {
    return nb::cast<std::string>(path_obj);
  }

  // os.PathLike (pathlib.Path and friends) expose __fspath__().
  if (nb::hasattr(path_obj, "__fspath__")) {
    nb::object fspath_result = path_obj.attr("__fspath__")();
    if (nb::isinstance<nb::str>(fspath_result) ||
        nb::isinstance<nb::bytes>(fspath_result)) {
      return nb::cast<std::string>(fspath_result);
    }
  }

  throw std::runtime_error(
      "Path argument must be a string or pathlib Path object");
}

}  // namespace qdk::chemistry::python::utils
