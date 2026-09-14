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
  Double-factorize a restricted Hamiltonian into low-rank two-electron fragments.

  The result is backed by a
  :class:`qdk_chemistry.data.FactorizedHamiltonianContainer` holding a sum of
  low-rank squares. One-body data, core energy, orbitals, inactive Fock data,
  and Hamiltonian type are preserved.

  The factorization has two steps. The first produces Cholesky vectors ``L``
  with ``g = L L^T``; the second diagonalizes each vector into its fragment.
  Only the first step depends on how the input stores its integrals:

  - Dense four-index integrals are reshaped into the supermatrix, given
    chemist permutation symmetry by averaging, and reduced by a pivoted
    Cholesky decomposition in ``O(naux * norb**4)``. Pivoting stops once the
    largest remaining residual diagonal drops to ``truncation_threshold``, or
    to a noise floor scaled to the largest supermatrix diagonal, whichever is
    larger; a threshold below that floor therefore has no further effect.
  - A :class:`qdk_chemistry.data.CholeskyHamiltonianContainer` already stores
    such vectors, so they are consumed directly. ``truncation_threshold`` is
    ignored.

  A Cholesky decomposition exists only for a positive semi-definite
  supermatrix, and the caller guarantees that property. Exact two-electron
  integrals have it by construction. Indefiniteness raises ``ValueError`` once
  it shows up in the residual diagonal; a negative direction whose diagonal
  stays below ``truncation_threshold`` is truncated away undetected.

See Also:
    :class:`qdk_chemistry.data.FactorizedHamiltonianContainer`

References:
    :cite:`vonBurg2021`
)");

  double_factorization.def(py::init<>(), R"(
Create a double factorization algorithm with default settings.
)");

  double_factorization.def("run", &DoubleFactorization::run, R"(
Double-factorize the given Hamiltonian.

Args:
  hamiltonian (qdk_chemistry.data.Hamiltonian): Restricted Hamiltonian containing two-electron integrals.

Returns:
  qdk_chemistry.data.Hamiltonian: New Hamiltonian backed by a factorized container.

Raises:
  ValueError: If the input or its two-electron integrals are invalid, or no fragment survives truncation.
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
