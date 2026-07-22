// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/unitary_synthesis.hpp>

namespace py = pybind11;

namespace {

py::list to_bool_list(const std::vector<std::uint8_t>& values) {
  py::list result;
  for (const auto value : values) {
    result.append(value != 0);
  }
  return result;
}

}  // namespace

void bind_unitary_synthesis(py::module& module) {
  module.def(
      "decompose_2d",
      [](const Eigen::Ref<const Eigen::MatrixXd>& a,
         const Eigen::Ref<const Eigen::MatrixXd>& b) {
        qdk::chemistry::utils::detail::TwoBlockCsd result;
        {
          py::gil_scoped_release release;
          result = qdk::chemistry::utils::detail::decompose_2d(a, b);
        }
        return py::make_tuple(result.u_1, result.u_2, result.d_1, result.d_2,
                              result.v);
      },
      py::arg("a"), py::arg("b"));

  module.def(
      "decompose_site_csd",
      [](const Eigen::Ref<const Eigen::MatrixXd>& matrix,
         Eigen::Index ancilla_dim) {
        qdk::chemistry::utils::detail::SiteCsd result;
        {
          py::gil_scoped_release release;
          result = qdk::chemistry::utils::detail::decompose_site_csd(
              matrix, ancilla_dim);
        }
        return py::make_tuple(result.u, result.d_prime, result.w_0, result.w_1,
                              result.v);
      },
      py::arg("matrix"), py::arg("ancilla_dim"));

  module.def(
      "decompose_unitary_to_givens",
      [](const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
        qdk::chemistry::utils::detail::GivensDecomposition result;
        {
          py::gil_scoped_release release;
          result = qdk::chemistry::utils::detail::decompose_unitary_to_givens(
              matrix);
        }
        return py::make_tuple(result.layer_angles,
                              to_bool_list(result.layer_shifted),
                              to_bool_list(result.phases));
      },
      py::arg("matrix"));

  module.def(
      "decompose_block_diagonal_to_givens",
      [](const std::vector<Eigen::MatrixXd>& blocks) {
        qdk::chemistry::utils::detail::GivensDecomposition result;
        {
          py::gil_scoped_release release;
          result =
              qdk::chemistry::utils::detail::decompose_block_diagonal_to_givens(
                  blocks);
        }
        return py::make_tuple(result.layer_angles,
                              to_bool_list(result.layer_shifted),
                              to_bool_list(result.phases));
      },
      py::arg("blocks"));

  module.def(
      "decompose_sparse_site",
      [](const Eigen::Ref<const Eigen::MatrixXd>& target) {
        qdk::chemistry::utils::detail::SparseSiteSynthesis result;
        {
          py::gil_scoped_release release;
          result = qdk::chemistry::utils::detail::decompose_sparse_site(target);
        }
        return py::make_tuple(
            result.column_permutation, result.inverse_column_permutation,
            result.row_permutation, result.inverse_row_permutation,
            result.block_givens.layer_angles,
            to_bool_list(result.block_givens.layer_shifted),
            to_bool_list(result.block_givens.phases));
      },
      py::arg("target"));
}