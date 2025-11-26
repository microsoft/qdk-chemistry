// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <macis/solvers/davidson.hpp>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <sparsexx/util/submatrix.hpp>

namespace py = pybind11;

namespace qdk::chemistry::python::utils {

using sparse_csr = sparsexx::csr_matrix<double, int64_t>;

/**
 * @brief Convert a SciPy CSR matrix to a sparsexx CSR matrix.
 *
 * @param py_csr Python object representing a SciPy CSR matrix
 * @return sparsexx CSR matrix
 */
inline sparse_csr from_scipy_csr(const py::object& py_csr) {
  // Add basic validation
  if (py_csr.is_none()) {
    throw std::invalid_argument("CSR matrix cannot be None");
  }

  auto data = py_csr.attr("data").cast<py::array_t<double>>();
  auto indices = py_csr.attr("indices").cast<py::array_t<int64_t>>();
  auto indptr = py_csr.attr("indptr").cast<py::array_t<int64_t>>();
  auto shape = py_csr.attr("shape").cast<std::pair<int64_t, int64_t>>();
  int64_t n_rows = shape.first;
  int64_t n_cols = shape.second;

  std::vector<int64_t> rowptr(indptr.size());
  std::vector<int64_t> colind(indices.size());
  std::vector<double> values(data.size());

  std::copy_n(indptr.data(), indptr.size(), rowptr.begin());
  std::copy_n(indices.data(), indices.size(), colind.begin());
  std::copy_n(data.data(), data.size(), values.begin());

  return sparse_csr(n_rows, n_cols, std::move(rowptr), std::move(colind),
                    std::move(values));
}

}  // namespace qdk::chemistry::python::utils

/**
 * @brief Bind Davidson solver utilities to Python module.
 *
 */
void bind_davidson_utils(py::module& m) {
  using qdk::chemistry::python::utils::from_scipy_csr;
  using qdk::chemistry::python::utils::sparse_csr;

  m.def(
      "davidson_solver",
      [](const py::object& csr_matrix, double tol, int max_m) -> py::tuple {
        // Convert input CSR matrix and set up SparseMatrixOperator
        sparse_csr H = from_scipy_csr(csr_matrix);
        macis::SparseMatrixOperator<sparse_csr> op(H);

        // Prepare initial guess using diagonal guess
        std::vector<double> X(H.m());
        macis::diagonal_guess(H.m(), H, X.data());

        auto D = sparsexx::extract_diagonal_elements(H);

        // Run Davidson
        auto [iters, eigval] =
            macis::davidson(H.m(), max_m, op, D.data(), tol, X.data());

        // Create a copy of the eigenvector data for Python
        py::array_t<double> eigvec = py::cast(X);
        return py::make_tuple(eigval, eigvec);
      },
      R"(
            Diagonalize a sparse matrix using the Davidson eigensolver.

            This function finds the lowest eigenvalue and corresponding eigenvector
            of a sparse matrix.

            Parameters
            ----------
            csr_matrix : scipy.sparse.csr_matrix
                Sparse matrix representation of the Hamiltonian.
            tol : float, optional
                Convergence tolerance for the eigenvalue (default: 1e-8).
            max_m : int, optional
                Maximum subspace size for the Davidson method (default: 20).

            Returns
            -------
            tuple of (eigenvalue, eigenvector)
                - eigenvalue: float, the lowest eigenvalue found
                - eigenvector: numpy.ndarray, corresponding eigenvector
            Raises
            ------
            RuntimeError
                If the Davidson solver fails to converge or encounters numerical issues
        )",
      py::arg("csr_matrix"), py::arg("tol") = 1e-8, py::arg("max_m") = 20);
}
