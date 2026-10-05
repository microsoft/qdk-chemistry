// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
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

}  // namespace detail

void bind_unitary_synthesis(py::module& module) {
  namespace synthesis = qdk::chemistry::utils::detail;
  auto unitary_synthesis =
      module.def_submodule("unitary_synthesis", "Unitary synthesis utilities.");

  unitary_synthesis.def(
      "decompose_dense_sites",
      [](const std::vector<qdk::chemistry::data::MPSSite>& sites,
         Eigen::Index ancilla_dimension) {
        std::vector<synthesis::DenseSiteSynthesis> results;
        {
          py::gil_scoped_release release;
          results = synthesis::decompose_dense_sites(sites, ancilla_dimension);
        }
        py::list output;
        for (const auto& result : results) {
          py::list mixing;
          for (const auto& givens : result.mixing_givens) {
            mixing.append(detail::givens_to_tuple(givens));
          }
          output.append(
              py::make_tuple(result.rotation_angles, mixing,
                             detail::givens_to_tuple(result.block_givens),
                             result.right_factor));
        }
        return output;
      },
      R"(
Synthesize the dense site unitaries of the sequential MPS preparation.

A site with physical basis ``('0', 'u', 'd', '2')`` is factored into three
uniformly controlled Ry rotations, two mixing unitaries, and a block-diagonal
unitary. A site with physical basis ``('0', '1')`` is factored into one
uniformly controlled Ry rotation and a block-diagonal unitary. Each site absorbs
the right factor of the following site, so only the right factor of the first
site remains to be absorbed by the caller. All sites are synthesized
concurrently.

Args:
    sites (Sequence[MPSSite]): Consecutive real sites with two or four physical
        states. The right bond dimension of each site must equal the left bond
        dimension of the following site.
    ancilla_dimension (int): Dimension of the bond register. Both bond
        dimensions of every site must be at most this value.

Returns:
    list[tuple]: One ``(rotation_angles, mixing_givens, block_givens,
    right_factor)`` tuple per site, in input order. ``rotation_angles`` holds
    one list of Ry angles per rotation, each of length ``ancilla_dimension``.
    ``mixing_givens`` holds the Givens data ``(layer_angles, layer_shifted,
    phases)`` of the two mixing unitaries for four physical states and is empty
    for two. ``block_givens`` is the Givens data of the block-diagonal unitary
    on the physical and bond registers. ``right_factor`` acts on the left bond
    of the site and is absorbed into the preceding site of the sequence.

Raises:
    ValueError: If a site is complex, does not fit the bond register, has an
        unsupported physical dimension, is not right-orthonormal, or the bonds
        of consecutive sites do not match.
)",
      py::arg("sites"), py::arg("ancilla_dimension"));

  unitary_synthesis.def(
      "decompose_sparse_sites",
      [](const std::vector<qdk::chemistry::data::MPSSite>& sites,
         Eigen::Index ancilla_dimension) {
        std::vector<synthesis::SparseSiteSynthesis> results;
        {
          py::gil_scoped_release release;
          results = synthesis::decompose_sparse_sites(sites, ancilla_dimension);
        }
        py::list output;
        for (const auto& result : results) {
          output.append(
              py::make_tuple(result.column_permutation, result.row_permutation,
                             detail::givens_to_tuple(result.block_givens)));
        }
        return output;
      },
      R"(
Decompose sparse MPS sites into permutations around block-diagonal unitaries.

Basis state ``b + ancilla_dimension * p`` of the joint register holds bond state
``b`` and physical state ``p``. Sites are independent and are decomposed
concurrently.

Args:
    sites (Sequence[MPSSite]): Real right-orthonormal sites.
    ancilla_dimension (int): Dimension of the bond register. Both bond
        dimensions of every site must be at most this value.

Returns:
    list[tuple]: One ``(column_permutation, row_permutation, block_givens)``
    tuple per site, in input order. The site unitary maps basis state ``i`` to
    ``column_permutation[i]``, applies the block-diagonal unitary with Givens
    data ``(layer_angles, layer_shifted, phases)``, and maps basis state ``j``
    to ``row_permutation[j]``.

Raises:
    ValueError: If a site is complex, does not fit the bond register, or is not
        right-orthonormal.
)",
      py::arg("sites"), py::arg("ancilla_dimension"));
}
