// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>

#include <memory>
#include <qdk/chemistry.hpp>

#include "eigen.hpp"
#include "factory_bindings.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::python;

void bind_double_factorization(nb::module_ &m) {
  nb::class_<DoubleFactorization, HamiltonianFactorization>
      double_factorization(m, "DoubleFactorization", R"(
  Double-factorize a restricted Cholesky Hamiltonian into low-rank two-electron fragments.

  The input must be backed by a
  :class:`qdk_chemistry.data.CholeskyHamiltonianContainer` whose MO Cholesky
  vectors ``L_l`` give ``(pq|rs) = sum_l L_l[pq] * L_l[rs]``. Each ``L_l``,
  read as a symmetric ``norb`` by ``norb`` matrix, is then diagonalized as
  ``L_l = U_l diag(w_l) U_l^T``, giving one fragment per vector in a
  :class:`qdk_chemistry.data.DFTHCHamiltonianContainer`.

See Also:
    :class:`qdk_chemistry.data.DFTHCHamiltonianContainer`

References:
    :cite:`vonBurg2021`
)");

  double_factorization.def(nb::init<>(), R"(
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
                           nb::arg("hamiltonian"));

  double_factorization.def("settings", &DoubleFactorization::settings, R"(
Return this factorizer's settings.

Returns:
  qdk_chemistry.data.Settings: Mutable settings, locked after the first call to :meth:`run`.
)",
                           nb::rv_policy::reference_internal);

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
                           nb::arg("hamiltonian"));

  double_factorization.def("__repr__", [](const DoubleFactorization &) {
    return "<qdk_chemistry.algorithms.DoubleFactorization>";
  });

  qdk::chemistry::python::bind_create_nested(double_factorization);
}
