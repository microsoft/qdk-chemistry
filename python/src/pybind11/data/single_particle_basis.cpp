// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/single_particle_basis.hpp>

namespace py = pybind11;

void bind_single_particle_basis(py::module &data) {
  using namespace qdk::chemistry::data;

  py::class_<SingleParticleBasis, DataClass, py::smart_holder>(
      data, "SingleParticleBasis",
      R"(
Abstract base for objects that expose a symmetry-blocked single-particle layout.

A single-particle basis exposes the symmetry vocabulary its modes are blocked
under, the per-label mode extents, and the total number of modes. Concrete
subclasses (e.g. :class:`Orbitals`) provide the actual storage.
)")
      .def("symmetries", &SingleParticleBasis::symmetries,
           "Symmetry vocabulary the single-particle modes are blocked under.")
      .def("mo_extents", &SingleParticleBasis::mo_extents,
           "Per-label mode extents (number of modes carrying each label).")
      .def("num_modes", &SingleParticleBasis::num_modes,
           "Total number of single-particle modes across all symmetry blocks.");
}
