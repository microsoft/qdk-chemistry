// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/unitary_synthesis.hpp>

namespace py = pybind11;

void bind_unitary_synthesis(py::module& module) {
  auto unitary_synthesis =
      module.def_submodule("unitary_synthesis", "Unitary synthesis utilities.");
  unitary_synthesis.def("dense_isometry_synthesis",
                        &qdk::chemistry::utils::dense_isometry_synthesis,
                        R"(
Synthesize a dense real isometry with cosine-sine decompositions.

Row ``q * chi + b`` of the isometry ``M`` is added-qubit state ``q`` and input
state ``b``. The circuit is ``B`` for no added qubit, ``B UCR_0 V`` for one,
and ``B UCR_2 W_1 CNOT UCR_1 W_0 CNOT UCR_0 V`` for two (Rupprecht and Wölk,
arXiv:2605.28489). Each orthogonal factor is ``D L[m-1] ... L[0]``: entry
``(j, k)`` of its angle matrix is the angle with which ``L[j]`` applies
``[[cos, -sin], [sin, cos]]`` to the pair ``(k, k + 1)``, zero if idle, and
``D`` is the diagonal of its sign vector.

Args:
    isometry (numpy.ndarray): Matrix with 1, 2, or 4 times ``chi`` rows and at most ``chi`` orthonormal columns.
    num_input_states (int): Dimension ``chi`` of the input register.
    omit_right_factor (bool): Whether to leave ``V`` out, so that the circuit realizes ``isometry @ V.T``.

Returns:
    tuple: Ry angles of ``UCR_0``, ``UCR_1``, and ``UCR_2`` by row; Givens angles and signs of ``B``, ``W_0``, ``W_1``, and ``V`` padded to ``chi``, as present; and ``V``.

Raises:
    ValueError: If the shape is unsupported or the matrix is not a finite isometry.
)",
                        py::arg("isometry"), py::arg("num_input_states"),
                        py::arg("omit_right_factor") = false,
                        py::call_guard<py::gil_scoped_release>());
  unitary_synthesis.def("block_sparse_isometry_synthesis",
                        &qdk::chemistry::utils::block_sparse_isometry_synthesis,
                        R"(
Decompose a sparse real isometry into ``P_r B P_c`` with block-diagonal ``B``.

Args:
    isometry (scipy.sparse.sparray | numpy.ndarray): Matrix with orthonormal columns and at least as many rows as columns.

Returns:
    tuple: Target row held by each row of ``B``, column of ``B`` holding each target column, and the Givens angles and signs of ``B``.

Raises:
    ValueError: If the matrix is empty, wider than tall, or not a finite isometry.
)",
                        py::arg("isometry"),
                        py::call_guard<py::gil_scoped_release>());
}
