// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <qdk/chemistry/data/shell.hpp>
#include <vector>

namespace qdk::chemistry::python::utils {

// Materialize the iterable to a list and use index-based access so no Python
// list_iterator is created while extracting items: list_iterator does not
// support weak references, which py::smart_holder tries to install, and that
// crashes during overload probing of constructors taking several
// std::vector<Shell> parameters.
inline std::vector<data::Shell> to_shell_vec(const pybind11::iterable& items) {
  pybind11::list list(items);
  const pybind11::ssize_t size = pybind11::len(list);
  std::vector<data::Shell> result;
  result.reserve(static_cast<size_t>(size));
  for (pybind11::ssize_t index = 0; index < size; ++index) {
    result.push_back(pybind11::reinterpret_borrow<pybind11::object>(
                         PyList_GET_ITEM(list.ptr(), index))
                         .cast<data::Shell>());
  }
  return result;
}

}  // namespace qdk::chemistry::python::utils
