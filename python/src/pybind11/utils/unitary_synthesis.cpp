// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <vector>

namespace py = pybind11;

namespace detail {

std::vector<bool> to_bool_vector(const std::vector<std::uint8_t>& values) {
  return {values.begin(), values.end()};
}

}  // namespace detail

void bind_unitary_synthesis(py::module& module) {
  namespace synthesis = qdk::chemistry::utils;
  using synthesis::DenseIsometrySynthesis;
  using synthesis::GivensDecomposition;
  using synthesis::SparseIsometrySynthesis;
  auto unitary_synthesis =
      module.def_submodule("unitary_synthesis", "Unitary synthesis utilities.");

  py::class_<GivensDecomposition>(unitary_synthesis, "GivensDecomposition",
                                  R"(
Givens representation ``U = D L[m-1] ... L[0]`` of a real orthogonal matrix,
where layer ``L[j]`` applies
``G(theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]`` to the
pairs ``(0, 1), (2, 3), ...``, or ``(1, 2), (3, 4), ...`` when shifted, and
``D`` is a diagonal sign matrix.
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

  py::class_<DenseIsometrySynthesis>(unitary_synthesis,
                                     "DenseIsometrySynthesis",
                                     R"(
Circuit data of a dense real isometry ``M`` with ``2**k * chi`` rows, whose row
``q * chi + b`` is added-qubit state ``q`` and input state ``b``, realized as
``B UCR_0 V`` for one added qubit, ``B UCR_2 W_1 CNOT UCR_1 W_0 CNOT UCR_0 V``
for two, and ``B`` for none (Rupprecht and Wölk, arXiv:2605.28489).
)")
      .def_property_readonly(
          "rotation_angles",
          [](const DenseIsometrySynthesis& self) {
            return self.rotation_angles;
          },
          "list[list[float]]: Ry angles of each multiplexed rotation in "
          "``(-2 pi, 2 pi]``, one per input state, with the folded signs.")
      .def_property_readonly(
          "mixing_givens",
          [](const DenseIsometrySynthesis& self) { return self.mixing_givens; },
          "list[GivensDecomposition]: Givens data of ``W_0`` and ``W_1``; "
          "empty unless two qubits are added.")
      .def_property_readonly(
          "block_givens",
          [](const DenseIsometrySynthesis& self) { return self.block_givens; },
          "GivensDecomposition: Merged Givens data of the block-diagonal "
          "``B``.")
      .def_property_readonly(
          "right_factor",
          [](const DenseIsometrySynthesis& self) { return self.right_factor; },
          "numpy.ndarray: Right factor ``V`` on the isometry columns, reported "
          "even when omitted and the identity when no qubit is added.")
      .def_property_readonly(
          "right_factor_givens",
          [](const DenseIsometrySynthesis& self) {
            return self.right_factor_givens;
          },
          "GivensDecomposition | None: Givens data of ``V`` padded to the "
          "input register up to row signs folded into ``UCR_0``, or ``None`` "
          "when ``V`` is omitted or no qubit is added.");

  py::class_<SparseIsometrySynthesis>(unitary_synthesis,
                                      "SparseIsometrySynthesis",
                                      R"(
Unitary completion ``U = P_r B P_c`` of a sparse real isometry, with
``P_c|c> = |column_permutation[c]>`` and ``P_r|v> = |row_permutation[v]>``,
whose first columns equal the isometry.
)")
      .def_property_readonly(
          "column_permutation",
          [](const SparseIsometrySynthesis& self) {
            return self.column_permutation;
          },
          "list[int]: Column of ``B`` that holds each target column.")
      .def_property_readonly(
          "row_permutation",
          [](const SparseIsometrySynthesis& self) {
            return self.row_permutation;
          },
          "list[int]: Target row held by each row of ``B``.")
      .def_property_readonly(
          "block_givens",
          [](const SparseIsometrySynthesis& self) { return self.block_givens; },
          "GivensDecomposition: Merged Givens data of ``B``, largest diagonal "
          "block first.");

  unitary_synthesis.def(
      "dense_isometry_synthesis",
      [](const Eigen::MatrixXd& isometry, Eigen::Index num_input_states,
         bool omit_right_factor) {
        py::gil_scoped_release release;
        return synthesis::dense_isometry_synthesis(isometry, num_input_states,
                                                   omit_right_factor);
      },
      R"(
Synthesize a dense real isometry with cosine-sine decompositions.

Args:
    isometry (numpy.ndarray): Matrix with 1, 2, or 4 times ``num_input_states``
        rows and at most that many orthonormal columns, realized as its polar
        factor.
    num_input_states (int): Dimension ``chi`` of the input register.
    omit_right_factor (bool): Whether to leave ``V`` out, so the circuit
        realizes ``isometry @ V.T`` and the caller applies ``V`` first.

Returns:
    DenseIsometrySynthesis: Rotation angles, Givens data, and right factor.

Raises:
    ValueError: If the shape is unsupported or the matrix is not a finite
        isometry.
)",
      py::arg("isometry"), py::arg("num_input_states"),
      py::arg("omit_right_factor") = false);

  unitary_synthesis.def(
      "dense_isometry_right_factor",
      [](const Eigen::MatrixXd& isometry, Eigen::Index num_input_states) {
        py::gil_scoped_release release;
        return synthesis::dense_isometry_right_factor(isometry,
                                                      num_input_states);
      },
      R"(
Return the right factor ``V`` that :func:`dense_isometry_synthesis` can omit.

Args:
    isometry (numpy.ndarray): See :func:`dense_isometry_synthesis`.
    num_input_states (int): See :func:`dense_isometry_synthesis`.

Returns:
    numpy.ndarray: Orthogonal matrix on the isometry columns, equal to
    ``DenseIsometrySynthesis.right_factor``.

Raises:
    ValueError: If the shape is unsupported or the matrix is not a finite
        isometry.
)",
      py::arg("isometry"), py::arg("num_input_states"));

  constexpr const char* block_sparse_doc = R"(
Decompose a sparse real isometry into permutations around a block-diagonal
orthogonal matrix, joining columns that share a nonzero row.

Args:
    isometry (numpy.ndarray | scipy.sparse.sparray | scipy.sparse.spmatrix):
        Matrix with orthonormal columns and at least as many rows as columns,
        whose duplicate entries are summed.

Returns:
    SparseIsometrySynthesis: Permutations and block-diagonal Givens data.

Raises:
    ValueError: If the matrix is empty, wider than tall, or not a finite
        isometry.
)";
  unitary_synthesis.def(
      "block_sparse_isometry_synthesis",
      [](const Eigen::MatrixXd& isometry) {
        py::gil_scoped_release release;
        return synthesis::block_sparse_isometry_synthesis(isometry);
      },
      block_sparse_doc, py::arg("isometry"));
  unitary_synthesis.def(
      "block_sparse_isometry_synthesis",
      [](const Eigen::SparseMatrix<double>& isometry) {
        py::gil_scoped_release release;
        return synthesis::block_sparse_isometry_synthesis(isometry);
      },
      block_sparse_doc, py::arg("isometry"));
}
