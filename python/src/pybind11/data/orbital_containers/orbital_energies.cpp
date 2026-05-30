// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/orbital_containers/orbital_energies.hpp>

namespace py = pybind11;

void bind_orbital_energies(py::module &data) {
  using namespace qdk::chemistry::data;

  py::class_<OrbitalEnergies, DataClass, py::smart_holder>(data,
                                                           "OrbitalEnergies",
                                                           R"(
Symmetry-blocked single-particle (orbital) energies.

Wraps a rank-1 :class:`SymmetryBlockedTensor` over the MO symmetry vocabulary.
Each symmetry block holds the energies of the modes carrying that label.
)")
      .def(py::init<std::shared_ptr<const OrbitalEnergies::Sbt>>(),
           py::arg("energies"),
           "Construct from a rank-1 symmetry-blocked tensor of energies.")
      .def("tensor", &OrbitalEnergies::tensor,
           "The underlying rank-1 symmetry-blocked tensor.")
      .def("symmetries", &OrbitalEnergies::symmetries,
           "Symmetry vocabulary the orbital energies are blocked under.")
      .def("mo_extents", &OrbitalEnergies::mo_extents,
           "Per-label mode extents.")
      .def("has_block", &OrbitalEnergies::has_block, py::arg("label"),
           "True iff an energy block is stored for the given label.")
      .def("block", &OrbitalEnergies::block, py::arg("label"),
           py::return_value_policy::reference_internal,
           "Energy vector for the given symmetry block.");
}
