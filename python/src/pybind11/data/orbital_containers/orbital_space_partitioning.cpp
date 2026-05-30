// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/orbital_containers/orbital_space_partitioning.hpp>

namespace py = pybind11;

void bind_orbital_space_partitioning(py::module &data) {
  using namespace qdk::chemistry::data;

  py::enum_<OrbitalSpace>(data, "OrbitalSpace",
                          "The five disjoint single-particle orbital spaces.")
      .value("Frozen", OrbitalSpace::Frozen)
      .value("Inactive", OrbitalSpace::Inactive)
      .value("Active", OrbitalSpace::Active)
      .value("Virtual", OrbitalSpace::Virtual)
      .value("External", OrbitalSpace::External);

  py::class_<OrbitalSpacePartitioning, DataClass, py::smart_holder>(
      data, "OrbitalSpacePartitioning",
      R"(
Symmetry-blocked assignment of single-particle modes to the five orbital spaces.

The five index sets (Frozen, Inactive, Active, Virtual, External) share the same
symmetry vocabulary and mode extents. Use :meth:`all_active` to build a default
partitioning that places every mode in the Active space.
)")
      .def(py::init<OrbitalSpacePartitioning::IndexSetArray>(),
           py::arg("spaces"),
           "Construct from the five symmetry-blocked index sets (in order: "
           "Frozen, Inactive, Active, Virtual, External).")
      .def("space", &OrbitalSpacePartitioning::space, py::arg("space"),
           "The index set for the given orbital space.")
      .def("frozen", &OrbitalSpacePartitioning::frozen,
           "The frozen-core index set.")
      .def("inactive", &OrbitalSpacePartitioning::inactive,
           "The inactive index set.")
      .def("active", &OrbitalSpacePartitioning::active, "The active index set.")
      .def("virtual_", &OrbitalSpacePartitioning::virtual_orbitals,
           "The virtual index set.")
      .def("external", &OrbitalSpacePartitioning::external,
           "The external index set.")
      .def("symmetries", &OrbitalSpacePartitioning::symmetries,
           "Symmetry vocabulary shared by all five subspaces.")
      .def("mo_extents", &OrbitalSpacePartitioning::mo_extents,
           "Per-label mode extents shared by all five subspaces.")
      .def_static("all_active", &OrbitalSpacePartitioning::all_active,
                  py::arg("symmetries"), py::arg("mo_extents"),
                  "Build a partitioning with all modes in the Active space.");
}
