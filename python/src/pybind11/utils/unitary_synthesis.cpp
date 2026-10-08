// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <string>
#include <vector>

namespace py = pybind11;

namespace detail {

namespace synthesis = qdk::chemistry::utils::detail;

py::list to_bool_list(const std::vector<std::uint8_t>& values) {
  py::list result;
  for (const auto value : values) {
    result.append(value != 0);
  }
  return result;
}

py::tuple givens_to_tuple(const synthesis::GivensDecomposition& result) {
  return py::make_tuple(result.layer_angles, to_bool_list(result.layer_shifted),
                        to_bool_list(result.phases));
}

py::tuple synthesis_to_tuple(const synthesis::DenseSiteSynthesis& result) {
  py::list mixing;
  for (const auto& givens : result.mixing_givens) {
    mixing.append(givens_to_tuple(givens));
  }
  return py::make_tuple(result.rotation_angles, mixing,
                        givens_to_tuple(result.block_givens),
                        result.right_factor);
}

py::tuple synthesis_to_tuple(const synthesis::SparseSiteSynthesis& result) {
  return py::make_tuple(result.column_permutation, result.row_permutation,
                        givens_to_tuple(result.block_givens));
}

}  // namespace detail

void bind_unitary_synthesis(py::module& module) {
  namespace synthesis = qdk::chemistry::utils::detail;
  namespace data = qdk::chemistry::data;
  auto unitary_synthesis =
      module.def_submodule("unitary_synthesis", "Unitary synthesis utilities.");

  unitary_synthesis.def(
      "dense_unitary_synthesis",
      [](const data::MPSSite& site, Eigen::Index ancilla_dimension,
         const Eigen::MatrixXd& following_right_factor) {
        synthesis::DenseSiteSynthesis result;
        {
          py::gil_scoped_release release;
          result = synthesis::dense_unitary_synthesis(site, ancilla_dimension,
                                                      following_right_factor);
        }
        return detail::synthesis_to_tuple(result);
      },
      R"(
Synthesize one validated MPS site with the general decomposition.

Args:
    site (MPSSite): Real site with two or four physical states.
    ancilla_dimension (int): Dimension of the bond register.
    following_right_factor (numpy.ndarray): Optional successor factor to absorb
        into the right bond. An empty matrix selects the identity.

Returns:
    tuple: ``(rotation_angles, mixing_givens, block_givens, right_factor)``.
        Each Givens value holds ``(layer_angles, layer_shifted, phases)``.

Raises:
    ValueError: If the site is complex, has an unsupported physical dimension,
        is nonisometric, or is incompatible with the register or successor factor.
)",
      py::arg("site"), py::arg("ancilla_dimension"),
      py::arg("following_right_factor") = Eigen::MatrixXd{});

  unitary_synthesis.def(
      "block_sparse_unitary_synthesis",
      [](const data::MPSSite& site, Eigen::Index ancilla_dimension) {
        synthesis::SparseSiteSynthesis result;
        {
          py::gil_scoped_release release;
          result = synthesis::block_sparse_unitary_synthesis(site,
                                                             ancilla_dimension);
        }
        return detail::synthesis_to_tuple(result);
      },
      R"(
Synthesize one symmetry-blocked site without forming its dense matrix.

Args:
    site (MPSSite): Validated real site whose stored sector offsets are reused.
    ancilla_dimension (int): Dimension of the bond register.

Returns:
    tuple: ``(column_permutation, row_permutation, block_givens)``. The site
        unitary applies the column permutation, the block-diagonal Givens
        network, and then the row permutation. Basis index
        ``physical * ancilla_dimension + bond`` holds the physical and bond states.

Raises:
    ValueError: If the site is complex, does not fit the register, or is not
        right-orthonormal.
)",
      py::arg("site"), py::arg("ancilla_dimension"));

  unitary_synthesis.def(
      "decompose_mps",
      [](const data::MPSContainer& mps, Eigen::Index ancilla_dimension,
         const std::string& method) {
        synthesis::MPSSynthesis results;
        {
          py::gil_scoped_release release;
          results = synthesis::decompose_mps(mps, ancilla_dimension, method);
        }
        py::list output;
        std::visit(
            [&](const auto& sites) {
              for (const auto& site : sites) {
                output.append(detail::synthesis_to_tuple(site));
              }
            },
            results);
        return output;
      },
      R"(
Synthesize sites 1 onward of an MPS container, in chain order.

The container validates adjacent bond spaces. Site zero is prepared separately
as the initial state. General synthesis absorbs every successor's right factor
into its predecessor; the first returned factor must be absorbed into site zero.
Block-sparse synthesis reads the stored tensor blocks directly.

Args:
    mps (MPSContainer): Real MPS with right-orthonormal sites after site zero.
    ancilla_dimension (int): Dimension of the bond register.
    unitary_synthesis (str): ``"general"`` (default) or ``"block_sparse"``.

Returns:
    list[tuple]: One tensor-level synthesis result per site after site zero.
        A single-site container returns an empty list.

Raises:
    ValueError: If the method is unknown, the MPS is complex, a bond does not
        fit the register, or a synthesized site is not right-orthonormal.
)",
      py::arg("mps"), py::arg("ancilla_dimension"),
      py::arg("unitary_synthesis") = "general");
}
