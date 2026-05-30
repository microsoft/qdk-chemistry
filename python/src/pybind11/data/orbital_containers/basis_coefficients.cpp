// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/orbital_containers/basis_coefficients.hpp>

namespace py = pybind11;

void bind_basis_coefficients(py::module &data) {
  using namespace qdk::chemistry::data;

  py::class_<BasisCoefficients, DataClass, py::smart_holder>(
      data, "BasisCoefficients",
      R"(
Symmetry-blocked single-particle basis coefficients (MO coefficients).

Wraps a rank-2 :class:`SymmetryBlockedTensor` whose two axes are
``[ao_symmetries, mo_symmetries]``. Block ``(ao_label, mo_label)`` holds the
coefficient matrix mapping those symmetry blocks. For restricted (RHF/ROHF)
orbitals the ``(alpha, alpha)`` and ``(beta, beta)`` blocks alias the same
storage (:meth:`is_restricted` is ``True``); for UHF they are distinct.
)")
      .def(py::init<std::shared_ptr<const BasisCoefficients::Sbt>>(),
           py::arg("coefficients"),
           "Construct from a rank-2 symmetry-blocked tensor of coefficients.")
      .def("tensor", &BasisCoefficients::tensor,
           "The underlying rank-2 symmetry-blocked tensor.")
      .def("ao_symmetries", &BasisCoefficients::ao_symmetries,
           "Symmetry vocabulary of the atomic-orbital (row) axis.")
      .def("mo_symmetries", &BasisCoefficients::mo_symmetries,
           "Symmetry vocabulary of the molecular-orbital (column) axis.")
      .def("ao_extents", &BasisCoefficients::ao_extents,
           "Per-label atomic-orbital extents.")
      .def("mo_extents", &BasisCoefficients::mo_extents,
           "Per-label molecular-orbital extents.")
      .def("is_restricted", &BasisCoefficients::is_restricted,
           "True iff the spin blocks alias (restricted coefficients).")
      .def("has_block", &BasisCoefficients::has_block, py::arg("ao_label"),
           py::arg("mo_label"),
           "True iff a coefficient block is stored for the given labels.")
      .def("block", &BasisCoefficients::block, py::arg("ao_label"),
           py::arg("mo_label"), py::return_value_policy::reference_internal,
           "Coefficient matrix for the given (ao_label, mo_label) block.");
}
