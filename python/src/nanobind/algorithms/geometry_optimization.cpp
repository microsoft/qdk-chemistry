// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class GeometryOptimizerBase : public GeometryOptimizer {
 public:
  NB_TRAMPOLINE(GeometryOptimizer);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  void replace_settings(
      std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  GeometryOptimizationResult _run_impl(
      std::shared_ptr<Structure> structure, int charge, int spin_multiplicity,
      GeometryOptimizationInputType input,
      unsigned int n_inactive_orbitals) const override {
    NB_OVERRIDE_PURE(_run_impl, structure, charge, spin_multiplicity, input,
                     n_inactive_orbitals);
  }
};

void bind_geometry_optimization(nb::module_& m) {
  nb::class_<GeometryOptimizerSettings, Settings>(
      m, "GeometryOptimizerSettings",
      R"(Settings shared by geometry optimization algorithms.)")
      .def(nb::init<>(),
           R"(Create geometry optimizer settings with common defaults.)");

  nb::class_<GeometryOptimizer, GeometryOptimizerBase> optimizer(
      m, "GeometryOptimizer",
      R"(
    Base class for geometry optimization algorithms.

    Optimizers derive active-space electron counts for nuclear derivative
    calculations and return the optimized energy, optimized structure,
    optional Hessian, and optional wavefunction.
    )");

  optimizer.def(nb::init<>(), R"(Create a geometry optimizer.)");
  optimizer.def(
      "run",
      [](const GeometryOptimizer& self, std::shared_ptr<Structure> structure,
         int charge, int spin_multiplicity, GeometryOptimizationInputType input,
         unsigned int n_inactive_orbitals) {
        return self.run(structure, charge, spin_multiplicity, input,
                        n_inactive_orbitals);
      },
      nb::arg("structure"), nb::arg("charge"), nb::arg("spin_multiplicity"),
      nb::arg("input"), nb::arg("n_inactive_orbitals") = 0,
      R"(
Optimize a molecular structure.

Args:
    structure: Initial molecular structure to optimize.
    charge: Total molecular charge.
    spin_multiplicity: Spin multiplicity of the molecular system.
    input: Basis name, basis set, orbitals, or wavefunction input.
    n_inactive_orbitals: Number of doubly occupied orbitals excluded from the active space.

Returns:
  tuple: ``(energy, structure, hessian, wavefunction)``.
)");
  optimizer.def(
      "hash", &GeometryOptimizer::hash, nb::arg("structure"), nb::arg("charge"),
      nb::arg("spin_multiplicity"), nb::arg("input"),
      nb::arg("n_inactive_orbitals") = 0,
      R"(Compute a deterministic content hash for an optimization run.)");
  optimizer.def("settings", &GeometryOptimizer::settings,
                nb::rv_policy::reference_internal,
                R"(Return the optimizer settings.)");
  optimizer.def_prop_rw(
      "_settings",
      [](GeometryOptimizer& self) -> Settings& { return self.settings(); },
      [](GeometryOptimizer& self,
         std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
        dynamic_cast<GeometryOptimizerBase&>(self).replace_settings(
            std::move(new_settings));
      },
      nb::rv_policy::reference_internal,
      R"(Internal settings replacement hook for Python subclasses.)");
  optimizer.def("name", &GeometryOptimizer::name,
                R"(Return the implementation name.)");
  optimizer.def("aliases", &GeometryOptimizer::aliases,
                R"(Return all registered names for the algorithm.)");
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
