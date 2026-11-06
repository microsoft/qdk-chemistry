// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>

#include <qdk/chemistry/data/json_serialization.hpp>

#include "stl.hpp"

namespace nb = nanobind;

void bind_serialization(nb::module_& data) {
  data.def("_validate_serialization_version",
           &qdk::chemistry::data::validate_serialization_version,
           R"(
Validate serialization version compatibility.

This function checks that the major and minor versions match exactly.
Patch version differences are allowed for backward compatibility.

Args:
    expected_version: The version string this code expects (e.g., "0.1.0")
    found_version: The version string found in the serialized data

Raises:
    RuntimeError: If major or minor version mismatch

Examples:
    >>> validate_serialization_version("0.1.0", "0.1.0")  # OK
    >>> validate_serialization_version("0.1.0", "0.1.1")  # OK (patch differs)
    >>> validate_serialization_version("0.1.0", "0.2.0")  # Raises RuntimeError
)",
           nb::arg("expected_version"), nb::arg("found_version"));
}
