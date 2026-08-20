// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/orbital_optimization.hpp>

namespace py = pybind11;
using namespace qdk::chemistry::data;

void bind_orbital_optimization_data(py::module& m) {
  py::class_<OrbitalOptimizationResult, py::smart_holder>(
      m, "OrbitalOptimizationResult", R"(
Immutable result of one objective-driven orbital optimization.

An orbital optimizer consumes a correlated :class:`Wavefunction` but returns
rotated :class:`Orbitals`, not a wavefunction. The rotation changes the
integrals, so the state must be re-solved in the proposed basis; the returned
orbitals carry the inactive and active partition labels describing the proposed
subspaces, with the virtual space defined as their complement.
)")
      .def(py::init<std::shared_ptr<Orbitals>, double, double, size_t, bool>(),
           py::arg("orbitals"), py::arg("initial_objective"),
           py::arg("final_objective"), py::arg("iterations"),
           py::arg("converged"))
      .def_property_readonly("orbitals", &OrbitalOptimizationResult::orbitals)
      .def_property_readonly("initial_objective",
                             &OrbitalOptimizationResult::initial_objective)
      .def_property_readonly("final_objective",
                             &OrbitalOptimizationResult::final_objective)
      .def_property_readonly("iterations",
                             &OrbitalOptimizationResult::iterations)
      .def_property_readonly("converged",
                             &OrbitalOptimizationResult::converged);

  py::class_<ActiveSpaceOptimizationResult, py::smart_holder>(
      m, "ActiveSpaceOptimizationResult", R"(
Immutable result of a self-consistent active-space optimization.
)")
      .def(py::init<double, std::shared_ptr<Wavefunction>, bool, size_t,
                    std::vector<double>, std::vector<double>>(),
           py::arg("energy"), py::arg("wavefunction"), py::arg("converged"),
           py::arg("macro_iterations"), py::arg("energy_history"),
           py::arg("objective_history"))
      .def_property_readonly("energy", &ActiveSpaceOptimizationResult::energy)
      .def_property_readonly("wavefunction",
                             &ActiveSpaceOptimizationResult::wavefunction)
      .def_property_readonly("converged",
                             &ActiveSpaceOptimizationResult::converged)
      .def_property_readonly("macro_iterations",
                             &ActiveSpaceOptimizationResult::macro_iterations)
      .def_property_readonly("energy_history",
                             &ActiveSpaceOptimizationResult::energy_history)
      .def_property_readonly("objective_history",
                             &ActiveSpaceOptimizationResult::objective_history);
}
