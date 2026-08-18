// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/hamiltonian_one_norm.hpp>

namespace py = pybind11;

void bind_hamiltonian_one_norm(py::module& m) {
  py::class_<qdk::chemistry::utils::HamiltonianOneNorm>(m, "HamiltonianOneNorm",
                                                        R"(
Breakdown of the double-factorization (DF) fermionic LCU 1-norm of a
restricted electronic Hamiltonian.
)")
      .def_readonly("one_body",
                    &qdk::chemistry::utils::HamiltonianOneNorm::one_body,
                    "One-electron contribution to the 1-norm (lambda_1e).")
      .def_readonly("two_body",
                    &qdk::chemistry::utils::HamiltonianOneNorm::two_body,
                    "Two-electron contribution to the 1-norm (lambda_2e).")
      .def_readonly("total", &qdk::chemistry::utils::HamiltonianOneNorm::total,
                    "Total 1-norm (lambda = lambda_1e + lambda_2e).");

  m.def("hamiltonian_one_norm", &qdk::chemistry::utils::hamiltonian_one_norm,
        R"(
            Compute the double-factorization fermionic LCU 1-norm of a
            restricted Hamiltonian.

            This is a standalone diagnostic utility: it can be called directly
            on any qdk_chemistry.data.Hamiltonian without creating or
            configuring an Algorithm (e.g.
            qdk_chemistry.algorithms.HamiltonianRegularizer). Users who only
            want to inspect the fermionic 1-norm of a Hamiltonian -- without
            necessarily also computing/applying a BLISS-style shift -- should
            call this function directly.

            Args
            ----
            hamiltonian : qdk_chemistry.data.Hamiltonian
                Restricted Hamiltonian to analyze.
            df_truncation_threshold : float, optional
                Fragments whose reshaped-supermatrix eigenvalue magnitude is
                below this are dropped from the two-body 1-norm. Defaults to
                0.0 (no truncation -- exact/lossless factorization unless the
                caller explicitly opts into compression).

            Returns
            -------
            HamiltonianOneNorm
                The one-body, two-body, and total DF fermionic 1-norm.

            Raises
            ------
            RuntimeError
                If the Hamiltonian is not restricted.
        )",
        py::arg("hamiltonian"), py::arg("df_truncation_threshold") = 0.0);
}
