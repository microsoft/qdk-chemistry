// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/double_factorization.hpp>

namespace py = pybind11;

void bind_double_factorization(py::module& m) {
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
                Fragments whose eigenvalue magnitude of the reshaped
                supermatrix falls below this threshold are dropped. Defaults
                to 0.0 (no truncation -- the factorization is exact/lossless
                unless the caller explicitly opts into compression).

            Returns
            -------
            list[TwoBodyFragment]
                The list of retained fragments, sorted by decreasing
                eigenvalue magnitude.
        )",
        py::arg("two_body_integrals"), py::arg("norb"),
        py::arg("truncation_threshold") = 0.0);
}
