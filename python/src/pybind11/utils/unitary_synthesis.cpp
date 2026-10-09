// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace detail {

std::vector<bool> to_bool_vector(const std::vector<std::uint8_t>& values) {
  return {values.begin(), values.end()};
}

}  // namespace detail

void bind_unitary_synthesis(py::module& module) {
  namespace synthesis = qdk::chemistry::utils::detail;
  namespace data = qdk::chemistry::data;
  using synthesis::DenseSiteSynthesis;
  using synthesis::GivensDecomposition;
  using synthesis::SparseSiteSynthesis;
  auto unitary_synthesis =
      module.def_submodule("unitary_synthesis", "Unitary synthesis utilities.");

  py::class_<GivensDecomposition>(unitary_synthesis, "GivensDecomposition",
                                  R"(
Givens representation ``U = D L[m-1] ... L[0]`` of a real orthogonal matrix.

Layer ``L[j]`` rotates adjacent basis states by
``G(theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]``. It acts on
the pairs ``(0, 1), (2, 3), ...``, or on ``(1, 2), (3, 4), ...`` when it is
shifted. ``D`` is a diagonal sign matrix.
)")
      .def_property_readonly(
          "layer_angles",
          [](const GivensDecomposition& self) { return self.layer_angles; },
          "list[list[float]]: Rotation angles of each layer, by increasing "
          "pair index.")
      .def_property_readonly(
          "layer_shifted",
          [](const GivensDecomposition& self) {
            return detail::to_bool_vector(self.layer_shifted);
          },
          "list[bool]: Whether each layer acts on the odd-starting pairs.")
      .def_property_readonly(
          "phases",
          [](const GivensDecomposition& self) {
            return detail::to_bool_vector(self.phases);
          },
          "list[bool]: Whether each basis state receives a minus sign from "
          "``D``.");

  py::class_<DenseSiteSynthesis>(unitary_synthesis, "DenseSiteSynthesis",
                                 R"(
Circuit data for one dense site unitary of the sequential MPS preparation.

A site with four physical states is applied as
``UCR_0 CNOT W_0 UCR_1 CNOT W_1 UCR_2 U`` (Fig. 5 of Rupprecht and Wölk,
arXiv:2605.28489), and a site with two physical states as ``UCR_0 U`` (Eq. 6).
Each ``UCR_k`` is a uniformly controlled Ry rotation addressed by the bond
register, and ``U`` is block diagonal. The right factor ``V`` is absorbed into
the preceding site or the initial state.
)")
      .def_property_readonly(
          "rotation_angles",
          [](const DenseSiteSynthesis& self) { return self.rotation_angles; },
          "list[list[float]]: Ry angles of each uniformly controlled rotation, "
          "one per bond state. Three rotations for four physical states, one "
          "for two.")
      .def_property_readonly(
          "mixing_givens",
          [](const DenseSiteSynthesis& self) { return self.mixing_givens; },
          "list[GivensDecomposition]: Givens data of ``W_0`` and ``W_1``; "
          "empty for two physical states.")
      .def_property_readonly(
          "block_givens",
          [](const DenseSiteSynthesis& self) { return self.block_givens; },
          "GivensDecomposition: Merged Givens data of the block-diagonal "
          "``U``.")
      .def_property_readonly(
          "right_factor",
          [](const DenseSiteSynthesis& self) { return self.right_factor; },
          "numpy.ndarray: Right factor ``V`` acting on the incoming bond.");

  py::class_<SparseSiteSynthesis>(unitary_synthesis, "SparseSiteSynthesis",
                                  R"(
Permutations and block synthesis data for one sparse MPS site.

The site unitary ``U = P_r B P_c`` applies the column permutation
``P_c|c> = |column_permutation[c]>``, the block-diagonal orthogonal matrix ``B``,
and then the row permutation ``P_r|v> = |row_permutation[v]>``. Basis index
``physical * ancilla_dimension + bond`` holds the physical and bond states.
)")
      .def_property_readonly(
          "column_permutation",
          [](const SparseSiteSynthesis& self) {
            return self.column_permutation;
          },
          "list[int]: Column of ``B`` that holds each target column.")
      .def_property_readonly(
          "row_permutation",
          [](const SparseSiteSynthesis& self) { return self.row_permutation; },
          "list[int]: Target row held by each row of ``B``.")
      .def_property_readonly(
          "block_givens",
          [](const SparseSiteSynthesis& self) { return self.block_givens; },
          "GivensDecomposition: Merged Givens data of ``B``, largest diagonal "
          "block first.");

  unitary_synthesis.def(
      "dense_unitary_synthesis",
      [](const data::MPSSite& site, Eigen::Index ancilla_dimension,
         const Eigen::MatrixXd& following_right_factor) {
        py::gil_scoped_release release;
        return synthesis::dense_unitary_synthesis(site, ancilla_dimension,
                                                  following_right_factor);
      },
      R"(
Synthesize one MPS site with the general decomposition.

Args:
    site (MPSSite): Real site with two or four physical states.
    ancilla_dimension (int): Dimension of the bond register.
    following_right_factor (numpy.ndarray): Optional successor factor to absorb
        into the right bond. An empty matrix selects the identity.

Returns:
    DenseSiteSynthesis: Rotation angles, Givens data, and right factor of the site.

Raises:
    ValueError: If the site is complex, has an unsupported physical dimension,
        is nonisometric, or is incompatible with the register or successor factor.
)",
      py::arg("site"), py::arg("ancilla_dimension"),
      py::arg("following_right_factor") = Eigen::MatrixXd{});

  unitary_synthesis.def(
      "block_sparse_unitary_synthesis",
      [](const data::MPSSite& site, Eigen::Index ancilla_dimension) {
        py::gil_scoped_release release;
        return synthesis::block_sparse_unitary_synthesis(site,
                                                         ancilla_dimension);
      },
      R"(
Synthesize one symmetry-blocked site without forming its dense matrix.

Args:
    site (MPSSite): Validated real site whose stored sector offsets are reused.
    ancilla_dimension (int): Dimension of the bond register.

Returns:
    SparseSiteSynthesis: Permutations and block-diagonal Givens data of the site.

Raises:
    ValueError: If the site is complex, does not fit the register, or is not
        right-orthonormal.
)",
      py::arg("site"), py::arg("ancilla_dimension"));

  unitary_synthesis.def(
      "matrix_product_state_synthesis",
      [](const data::MPSContainer& mps, Eigen::Index ancilla_dimension,
         const std::string& method) {
        synthesis::MPSSynthesis results;
        {
          py::gil_scoped_release release;
          results = synthesis::matrix_product_state_synthesis(
              mps, ancilla_dimension, method);
        }
        return std::visit(
            [](auto&& sites) { return py::cast(std::move(sites)); },
            std::move(results));
      },
      R"(
Synthesize sites 1 onward of an MPS container, in chain order.

The container validates adjacent bond spaces. Site zero is prepared separately
as the initial state. Dense synthesis absorbs every successor's right factor
into its predecessor; the first returned factor must be absorbed into site zero.
Block-sparse synthesis reads the stored tensor blocks directly.

Args:
    mps (MPSContainer): Real MPS with right-orthonormal sites after site zero.
    ancilla_dimension (int): Dimension of the bond register.
    unitary_synthesis (str): ``"dense"`` (default) or ``"block_sparse"``.

Returns:
    list[DenseSiteSynthesis] | list[SparseSiteSynthesis]: One result per site
        after site zero. A single-site container returns an empty list.

Raises:
    ValueError: If the method is unknown, the MPS is complex, a bond does not
        fit the register, or a synthesized site is not right-orthonormal.
)",
      py::arg("mps"), py::arg("ancilla_dimension"),
      py::arg("unitary_synthesis") = "dense");
}
