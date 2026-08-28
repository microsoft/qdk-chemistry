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
Exact double factorization of a Hamiltonian by nested eigen-decomposition.

Rewrites a Hamiltonian's dense four-index two-electron integrals as a signed
sum of low-rank "perfect square" fragments and returns a new Hamiltonian
backed by a :class:`qdk_chemistry.data.FactorizedHamiltonianContainer`:

.. math::

    g_{pqrs} = \sum_t s_t \Big(\sum_b \epsilon^t_b U^t_{bp} U^t_{bq}\Big)
                            \Big(\sum_{b'} \epsilon^t_{b'} U^t_{b'r} U^t_{b's}\Big)

The one-electron integrals, core energy, orbitals, inactive Fock matrix and
Hamiltonian type are carried over unchanged. Only restricted Hamiltonians are
currently supported.

Settings:
    truncation_threshold (float): Drop fragments whose supermatrix eigenvalue
        magnitude is below this threshold. Must be non-negative; the default
        of 1e-12 discards only the numerically null fragments, keeping the
        factorization exact to well within chemical accuracy. Pass 0.0 to
        retain every fragment.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as algorithms

    factorizer = algorithms.create("double_factorizer")
    factorizer.settings().set("truncation_threshold", 1e-8)
    factorized = factorizer.run(hamiltonian)

See Also:
    :class:`qdk_chemistry.data.FactorizedHamiltonianContainer`

References:
    :cite:`vonBurg2021`, :cite:`Patel2024`
)");

  double_factorizer.def(py::init<>(), R"(
Create a DoubleFactorizer instance.

Initializes a new double factorizer with default settings.
Configuration options can be modified through the ``settings()`` method.
)");

  double_factorizer.def("run", &DoubleFactorizer::run, R"(
Double-factorize the given Hamiltonian.

This method automatically locks settings before execution.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian to factorize.
        Must be restricted and carry two-electron integrals.

Returns:
    qdk_chemistry.data.Hamiltonian: A new Hamiltonian backed by a
    :class:`qdk_chemistry.data.FactorizedHamiltonianContainer`.

Raises:
    ValueError: If the Hamiltonian is unrestricted or has no two-electron
        integrals.
)",
                        py::arg("hamiltonian"));

  double_factorizer.def("settings", &DoubleFactorizer::settings, R"(
Access the double factorizer's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object
)",
                        py::return_value_policy::reference_internal);

  double_factorizer.def("name", &DoubleFactorizer::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm
)");

  double_factorizer.def("type_name", &DoubleFactorizer::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
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
