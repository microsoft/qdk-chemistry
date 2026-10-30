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

  py::class_<DenseIsometrySynthesis>(unitary_synthesis,
                                     "DenseIsometrySynthesis",
                                     R"(
Circuit data of a dense real isometry ``M`` with ``2**k * chi`` rows.

Row ``q * chi + b`` is the output state with ``k`` added qubits in state ``q``
and input register state ``b``; column ``a`` is input state ``a`` with the added
qubits in zero. With one added qubit the circuit is ``B UCR_0 V`` (Eq. 6 of
Rupprecht and Wölk, arXiv:2605.28489), and with two it is
``B UCR_2 W_1 CNOT UCR_1 W_0 CNOT UCR_0 V`` (Fig. 5), in operator order. Each
``UCR_j`` is an Ry rotation of an added qubit multiplexed by the input register,
and ``B`` is block diagonal over the added qubits. With no added qubit the
completed isometry is ``B`` alone.
)")
      .def_property_readonly(
          "rotation_angles",
          [](const DenseIsometrySynthesis& self) {
            return self.rotation_angles;
          },
          "list[list[float]]: Ry angles of each multiplexed rotation, one per "
          "input state. Three rotations for two added qubits, one for one, "
          "none for zero.")
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
          "numpy.ndarray: Right factor ``V`` on the input states, one row and "
          "column per isometry column. Reported even when omitted from the "
          "circuit, and the identity when no qubit is added.")
      .def_property_readonly(
          "right_factor_givens",
          [](const DenseIsometrySynthesis& self) {
            return self.right_factor_givens;
          },
          "GivensDecomposition | None: Givens data of ``V`` padded with the "
          "identity to the input register, or ``None`` when ``V`` is omitted "
          "or no qubit is added.");

  py::class_<SparseIsometrySynthesis>(unitary_synthesis,
                                      "SparseIsometrySynthesis",
                                      R"(
Permutations and block synthesis data of a sparse real isometry.

The unitary ``U = P_r B P_c`` applies the column permutation
``P_c|c> = |column_permutation[c]>``, the block-diagonal orthogonal matrix ``B``,
and then the row permutation ``P_r|v> = |row_permutation[v]>``. Its first
columns equal the isometry.
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
    isometry (numpy.ndarray): Real matrix with one, two, or four times
        ``num_input_states`` rows and at most ``num_input_states`` orthonormal
        columns.
    num_input_states (int): Dimension ``chi`` of the input register.
    omit_right_factor (bool): Whether to leave the right factor ``V`` out of the
        circuit. The circuit then realizes ``isometry @ V.T`` on its first
        columns, and the caller applies ``V`` from
        :func:`dense_isometry_right_factor` beforehand.

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

The factorization is the synthesis's own, so the result equals
``DenseIsometrySynthesis.right_factor`` for the same arguments.

Args:
    isometry (numpy.ndarray): See :func:`dense_isometry_synthesis`.
    num_input_states (int): See :func:`dense_isometry_synthesis`.

Returns:
    numpy.ndarray: Orthogonal matrix with one row and column per isometry
    column, the identity when no qubit is added.

Raises:
    ValueError: If the shape is unsupported or the matrix is not a finite
        isometry.
)",
      py::arg("isometry"), py::arg("num_input_states"));

  constexpr const char* block_sparse_doc = R"(
Decompose a sparse real isometry into permutations around a block-diagonal
orthogonal matrix.

Columns that share a nonzero row form one block. Only stored nonzero entries
are read.

Args:
    isometry (numpy.ndarray | scipy.sparse.sparray | scipy.sparse.spmatrix):
        Real matrix with orthonormal columns and at least as many rows as
        columns.

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
