// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/double_factorization.hpp>

namespace py = pybind11;

void bind_double_factorization(py::module& m) {
  py::enum_<qdk::chemistry::utils::DoubleFactorizationMethod>(
      m, "DoubleFactorizationMethod", R"(
Selects how the reshaped (ij),(kl) two-electron supermatrix is decomposed.

Both methods reconstruct g_ijkl identically, but writing the supermatrix as
M = X X^T fixes X only up to X -> X Q for orthogonal Q, and ``lambda_df`` is
not invariant under that gauge freedom. Choose deliberately when the 1-norm
itself matters.
)")
      .value("CHOLESKY",
             qdk::chemistry::utils::DoubleFactorizationMethod::Cholesky,
             "Pivoted Cholesky, O(R * norb^4) for rank R. Requires a positive "
             "semi-definite supermatrix, which holds for physical two-electron "
             "integrals; all fragments have sign = +1. Falls back to EIGEN "
             "with a warning if the supermatrix is indefinite.")
      .value("EIGEN", qdk::chemistry::utils::DoubleFactorizationMethod::Eigen,
             "Eigendecomposition via LAPACK syev, O(norb^6). Handles "
             "indefinite input (producing sign = -1 fragments) and yields the "
             "conventional ordering by decreasing eigenvalue magnitude.");

  py::class_<qdk::chemistry::utils::TwoBodyFragment>(m, "TwoBodyFragment", R"(
A single low-rank ("perfect square") two-electron fragment produced by
double factorization of the two-electron integral tensor.
)")
      .def_readonly("U", &qdk::chemistry::utils::TwoBodyFragment::U,
                    "Orbital rotation matrix (norb x norb).")
      .def_readonly("eps", &qdk::chemistry::utils::TwoBodyFragment::eps,
                    "Eigenvalues of the fragment (norb).")
      .def_readonly("sign", &qdk::chemistry::utils::TwoBodyFragment::sign,
                    "Fragment sign (+1.0 or -1.0).")
      .def_readonly("lambda_df",
                    &qdk::chemistry::utils::TwoBodyFragment::lambda_df,
                    "Baseline fermionic 1-norm contribution of this "
                    "fragment, before any BLISS shift.");

  m.def("double_factorize", &qdk::chemistry::utils::double_factorize,
        R"(
            Double-factorize the spin-free two-electron integral tensor.

            This is a standalone diagnostic/analysis utility: it does not require
            an Algorithm/Settings/Factory instance and can be called directly.

            Args
            ----
            two_body_integrals : numpy.ndarray
                Flattened g_ijkl tensor, size norb^4 (chemist notation,
                index = i*norb^3 + j*norb^2 + k*norb + l).
            norb : int
                Number of (spatial) orbitals.
            truncation_threshold : float, optional
                Cutoff below which fragment candidates are dropped. The units
                are method-dependent, so the same numeric value does not give
                the same rank for both methods: ``EIGEN`` compares against the
                supermatrix eigenvalue magnitude, ``CHOLESKY`` against the
                largest remaining residual diagonal. Defaults to 0.0, meaning
                lossless for both. (A literal 0.0 is unreachable in floating
                point for Cholesky, so that path floors the cutoff at machine
                epsilon to stop at the true numerical rank rather than emit
                roundoff fragments.)
            method : DoubleFactorizationMethod, optional
                Which decomposition to use. Defaults to ``CHOLESKY``. This
                affects the reported ``lambda_df`` values, not just the cost --
                see ``DoubleFactorizationMethod``.

            Returns
            -------
            list[TwoBodyFragment]
                The list of retained fragments, sorted by decreasing
                contribution.
        )",
        py::arg("two_body_integrals"), py::arg("norb"),
        py::arg("truncation_threshold") = 0.0,
        py::arg("method") =
            qdk::chemistry::utils::DoubleFactorizationMethod::Cholesky);
}
