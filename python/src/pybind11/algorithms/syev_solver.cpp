// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <lapack.hh>
#include <vector>

namespace py = pybind11;

/**
 * @brief Solve dense matrix eigenvalue problems using LAPACK's syev.
 *
 * @param matrix Input dense matrix (NumPy array).
 * @return A tuple containing the eigenvalues and eigenvectors of the matrix.
 */
py::tuple dense_matrix_solver(const py::array_t<double>& matrix) {
  // Validate input matrix
  if (matrix.ndim() != 2) {
    throw std::invalid_argument("Input matrix must be 2-dimensional.");
  }

  auto shape = matrix.shape();
  if (shape[0] != shape[1]) {
    throw std::invalid_argument("Input matrix must be square.");
  }

  int n = shape[0];

  // Use Eigen types for NumPy conversion
  Eigen::MatrixXd a = Eigen::Map<const Eigen::MatrixXd>(matrix.data(), n, n);
  Eigen::VectorXd w(n);

  int info = lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, n, a.data(), n,
                          w.data());

  if (info != 0) {
    throw std::runtime_error("LAPACK syev failed to compute eigenvalues.");
  }

  return py::make_tuple(w, a);
}

/**
 * @brief Bind syev solver to Python module.
 *
 */
void bind_syev_solver(py::module& m) {
  m.def("syev_solver", &dense_matrix_solver,
        R"(
            Solve a dense matrix eigenvalue problem using LAPACK's syev.

            This function computes all eigenvalues and eigenvectors of a dense matrix.

            Parameters
            ----------
            matrix : numpy.ndarray
                Dense matrix.

            Returns
            -------
            tuple of (eigenvalues, eigenvectors)
                - eigenvalues: numpy.ndarray, the eigenvalues of the matrix in ascending order
                - eigenvectors: numpy.ndarray, the eigenvectors of the matrix, where each column corresponds to an eigenvector

            Raises
            ------
            ValueError
                If the input matrix is not square or symmetric.
            RuntimeError
                If LAPACK fails to compute the eigenvalues.
        )",
        py::arg("matrix"));
}
