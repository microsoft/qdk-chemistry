// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/unitary_synthesis.hpp>

namespace py = pybind11;

namespace detail {

py::list to_bool_list(const std::vector<std::uint8_t>& values) {
  py::list result;
  for (const auto value : values) {
    result.append(value != 0);
  }
  return result;
}

}  // namespace detail

void bind_unitary_synthesis(py::module& module) {
  auto unitary_synthesis = module.def_submodule(
      "unitary_synthesis", "Unitary synthesis utilities.");

  unitary_synthesis.def(
      "decompose_sparse_site",
      [](const Eigen::Ref<const Eigen::MatrixXd>& target) {
        qdk::chemistry::utils::unitary_synthesis::detail::SparseSiteSynthesis
            result;
        {
          py::gil_scoped_release release;
          result = qdk::chemistry::utils::unitary_synthesis::detail::
              decompose_sparse_site(target);
        }
        return py::make_tuple(
            result.column_permutation, result.row_permutation,
            result.block_givens.layer_angles,
            detail::to_bool_list(result.block_givens.layer_shifted),
            detail::to_bool_list(result.block_givens.phases));
      },
      py::arg("target"));
}