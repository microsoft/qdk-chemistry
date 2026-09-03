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
  py::class_<DoubleFactorizer, py::smart_holder> double_factorizer(
      m, "DoubleFactorizer", R"(
  Double-factorize a restricted Hamiltonian by nested eigendecomposition.

  The result is backed by a
  :class:`qdk_chemistry.data.FactorizedHamiltonianContainer` containing signed
  low-rank fragments. Chemist permutation symmetry is imposed by averaging.
  Modes below the absolute ``truncation_threshold`` are omitted. One-body data,
  core energy, orbitals, inactive Fock data, and Hamiltonian type are preserved.

See Also:
    :class:`qdk_chemistry.data.FactorizedHamiltonianContainer`

References:
    :cite:`vonBurg2021`, :cite:`Patel2025`
)");

  double_factorizer.def(py::init<>(), R"(
Create a double factorizer with default settings.
)");

  double_factorizer.def("run", &DoubleFactorizer::run, R"(
Double-factorize the given Hamiltonian.

Args:
  hamiltonian (qdk_chemistry.data.Hamiltonian): Restricted Hamiltonian containing two-electron integrals.

Returns:
  qdk_chemistry.data.Hamiltonian: New Hamiltonian backed by a factorized container.

Raises:
  ValueError: If the input or its two-electron integrals are invalid, or no fragment survives truncation.
  RuntimeError: If an eigendecomposition fails.

Note:
  Calling this method locks the settings.
)",
                        py::arg("hamiltonian"));

  double_factorizer.def("settings", &DoubleFactorizer::settings, R"(
Return this factorizer's settings.

Returns:
  qdk_chemistry.data.Settings: Mutable settings, locked after the first call to :meth:`run`.
)",
                        py::return_value_policy::reference_internal);

  double_factorizer.def("name", &DoubleFactorizer::name, R"(
Return the implementation name.

Returns:
  str: ``"eigen_decomposition"``.
)");

  double_factorizer.def("aliases", &DoubleFactorizer::aliases, R"(
Return the algorithm's aliases.

Returns:
  list[str]: All registered names for the algorithm, including the primary one.
)");

  double_factorizer.def("type_name", &DoubleFactorizer::type_name, R"(
Return the algorithm type name.

Returns:
  str: ``"double_factorizer"``.
)");

  double_factorizer.def("hash", &DoubleFactorizer::hash,
                        py::arg("hamiltonian"));

  bind_algorithm_factory<DoubleFactorizerFactory, DoubleFactorizer>(
      m, "DoubleFactorizerFactory");

  double_factorizer.def("__repr__", [](const DoubleFactorizer &) {
    return "<qdk_chemistry.algorithms.DoubleFactorizer>";
  });

  qdk::chemistry::python::bind_create_nested(double_factorizer);
}
