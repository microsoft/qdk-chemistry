// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class GeometryOptimizerBase : public GeometryOptimizer,
                              public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, GeometryOptimizer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, GeometryOptimizer, aliases);
  }

  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  GeometryOptimizationResult _run_impl(
      std::shared_ptr<Structure> structure, int charge, int spin_multiplicity,
      GeometryOptimizationSeedType seed,
      unsigned int n_inactive_orbitals) const override {
    PYBIND11_OVERRIDE_PURE(GeometryOptimizationResult, GeometryOptimizer,
                           _run_impl, structure, charge, spin_multiplicity,
                           seed, n_inactive_orbitals);
  }
};

void bind_geometry_optimization(py::module& m) {
  py::class_<GeometryOptimizer, GeometryOptimizerBase, py::smart_holder>
      optimizer(m, "GeometryOptimizer",
                R"(
    Base class for geometry optimization algorithms.

    Optimizers derive active-space electron counts for nuclear derivative
    calculations and return the optimized energy, optimized structure,
    optional wavefunction, and optional Hessian.
    )");

  optimizer.def(py::init<>(), R"(Create a geometry optimizer.)");
  optimizer.def(
      "run",
      [](const GeometryOptimizer& self, std::shared_ptr<Structure> structure,
         int charge, int spin_multiplicity, GeometryOptimizationSeedType seed,
         unsigned int n_inactive_orbitals) {
        return self.run(structure, charge, spin_multiplicity, seed,
                        n_inactive_orbitals);
      },
      py::arg("structure"), py::arg("charge"), py::arg("spin_multiplicity"),
      py::arg("seed_or_basis"), py::arg("n_inactive_orbitals") = 0,
      R"(
Optimize a molecular structure.

Args:
    structure: Initial molecular structure to optimize.
    charge: Total molecular charge.
    spin_multiplicity: Spin multiplicity of the molecular system.
    seed_or_basis: Basis name, basis set, orbitals, or wavefunction seed.
    n_inactive_orbitals: Number of doubly occupied orbitals excluded from the active space.

Returns:
    tuple: ``(energy, structure, wavefunction, hessian)``.
)");
  optimizer.def("settings", &GeometryOptimizer::settings,
                py::return_value_policy::reference_internal,
                R"(Return the optimizer settings.)");
  optimizer.def_property(
      "_settings",
      [](GeometryOptimizerBase& self) -> Settings& { return self.settings(); },
      [](GeometryOptimizerBase& self,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        self.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(Internal settings replacement hook for Python subclasses.)");
  optimizer.def("name", &GeometryOptimizer::name,
                R"(Return the implementation name.)");
  optimizer.def("type_name", &GeometryOptimizer::type_name,
                R"(Return the algorithm type name.)");
  optimizer.def("__repr__", [](const GeometryOptimizer& self) {
    return "<qdk_chemistry.algorithms.GeometryOptimizer name='" + self.name() +
           "'>";
  });

  qdk::chemistry::python::bind_algorithm_factory<
      GeometryOptimizerFactory, GeometryOptimizer, GeometryOptimizerBase>(
      m, "GeometryOptimizerFactory");
  qdk::chemistry::python::bind_create_nested(optimizer);
}
