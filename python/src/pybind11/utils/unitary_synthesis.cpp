// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/wavefunction_containers/abelian_mps_wavefunction.hpp>
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

py::tuple sparse_site_synthesis_to_tuple(
    const qdk::chemistry::utils::unitary_synthesis::detail::SparseSiteSynthesis&
        result) {
  return py::make_tuple(result.column_permutation, result.row_permutation,
                        result.block_givens.layer_angles,
                        to_bool_list(result.block_givens.layer_shifted),
                        to_bool_list(result.block_givens.phases));
}

}  // namespace detail

void bind_unitary_synthesis(py::module& module) {
  auto unitary_synthesis =
      module.def_submodule("unitary_synthesis", "Unitary synthesis utilities.");

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
        return detail::sparse_site_synthesis_to_tuple(result);
      },
      py::arg("target"));

  unitary_synthesis.def(
      "decompose_sparse_site",
      [](const qdk::chemistry::data::AbelianMPSSite& site,
         std::size_t ancilla_dimension) {
        if (site.is_complex()) {
          throw std::invalid_argument(
              "Sparse site decomposition requires a real-valued MPS site.");
        }
        if (site.right_bond_dimension() > ancilla_dimension) {
          throw std::invalid_argument(
              "Ancilla dimension must be at least the right-bond dimension.");
        }

        const auto packed_variant = site.to_dense();
        const auto& packed = std::get<Eigen::MatrixXd>(packed_variant);
        const auto left_dimension =
            static_cast<Eigen::Index>(site.left_bond_dimension());
        const auto physical_dimension =
            static_cast<Eigen::Index>(site.physical_dimension());
        Eigen::MatrixXd target = Eigen::MatrixXd::Zero(
            physical_dimension * static_cast<Eigen::Index>(ancilla_dimension),
            left_dimension);
        for (Eigen::Index left = 0; left < left_dimension; ++left) {
          for (Eigen::Index physical = 0; physical < physical_dimension;
               ++physical) {
            target.block(
                physical * static_cast<Eigen::Index>(ancilla_dimension), left,
                packed.cols(), 1) =
                packed.row(left * physical_dimension + physical).transpose();
          }
        }

        qdk::chemistry::utils::unitary_synthesis::detail::SparseSiteSynthesis
            result;
        {
          py::gil_scoped_release release;
          result = qdk::chemistry::utils::unitary_synthesis::detail::
              decompose_sparse_site(target);
        }
        return detail::sparse_site_synthesis_to_tuple(result);
      },
      py::arg("site"), py::arg("ancilla_dimension"));
}
