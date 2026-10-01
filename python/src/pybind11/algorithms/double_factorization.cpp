// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::python;

void bind_double_factorization(py::module &m) {
  py::class_<DoubleFactorization, HamiltonianFactorization, py::smart_holder>
      double_factorization(m, "DoubleFactorization", R"(
  Double-factorize a restricted Cholesky Hamiltonian into low-rank two-electron fragments.

  The input must be a restricted :class:`qdk_chemistry.data.Hamiltonian`
  backed by a :class:`qdk_chemistry.data.CholeskyHamiltonianContainer`. Its MO
  Cholesky vectors ``L`` (``norb**2`` rows in pair order ``p*norb+q``, one
  column per vector) give the two-electron integrals as
  ``(pq|rs) = sum_l L[pq, l] * L[rs, l]``. This is the first factorization; its
  accuracy is set by whatever produced the vectors (for example
  ``cholesky_tolerance`` on the ``qdk_cholesky`` Hamiltonian constructor), not
  by this algorithm. Dense four-index integrals must be converted to Cholesky
  form first.

  The second factorization diagonalizes each Cholesky vector, read as a
  symmetric ``norb`` by ``norb`` matrix: ``L_l = U_l diag(w_l) U_l^T``. Each
  two-electron fragment is then the square of a one-body operator that is
  diagonal in the orbitals rotated by ``U_l``, with weights ``w_l``. Every
  vector and every eigenpair is kept, so the factorized integrals reproduce
  the input Cholesky product up to floating-point error.

  The result is backed by a :class:`qdk_chemistry.data.DFTHCHamiltonianContainer`
  with one rank per Cholesky vector. One-body integrals, core energy, orbitals,
  inactive Fock data, and Hamiltonian type are carried over unchanged.

See Also:
    :class:`qdk_chemistry.data.DFTHCHamiltonianContainer`

References:
    :cite:`vonBurg2021`
)");

  double_factorization.def(py::init<>(), R"(
Create a double factorization algorithm with default settings.
)");

  double_factorization.def("run", &DoubleFactorization::run, R"(
Double-factorize the given Hamiltonian.

Args:
  hamiltonian (qdk_chemistry.data.Hamiltonian): Restricted Cholesky Hamiltonian.

Returns:
  qdk_chemistry.data.Hamiltonian: The double-factorized Hamiltonian.

Raises:
  ValueError: If the input Hamiltonian or its Cholesky factors are invalid.
  RuntimeError: If an eigendecomposition fails.
)",
                           py::arg("hamiltonian"));

  double_factorization.def("settings", &DoubleFactorization::settings, R"(
Return this factorizer's settings.

Returns:
  qdk_chemistry.data.Settings: Mutable settings, locked after the first call to :meth:`run`.
)",
                           py::return_value_policy::reference_internal);

  double_factorization.def("name", &DoubleFactorization::name, R"(
Return the implementation name.

Returns:
  str: ``"double_factorization"``.
)");

  double_factorization.def("aliases", &DoubleFactorization::aliases, R"(
Return all registered names for the implementation.

Returns:
  list[str]: Every lookup name, including the canonical one.
)");

  double_factorization.def("type_name", &DoubleFactorization::type_name, R"(
Return the algorithm type name.

Returns:
  str: ``"hamiltonian_factorization"``.
)");

  double_factorization.def("hash", &DoubleFactorization::hash,
                           py::arg("hamiltonian"));

  double_factorization.def("__repr__", [](const DoubleFactorization &) {
    return "<qdk_chemistry.algorithms.DoubleFactorization>";
  });

  qdk::chemistry::python::bind_create_nested(double_factorization);
}
