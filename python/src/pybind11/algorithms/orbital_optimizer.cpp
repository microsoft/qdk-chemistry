// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class OrbitalOptimizerBase : public OrbitalOptimizer,
                             public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, OrbitalOptimizer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, OrbitalOptimizer, aliases);
  }

  void replace_settings(std::unique_ptr<Settings> new_settings) {
    _settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<OrbitalOptimizationResult> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<OrbitalOptimizationResult>,
                           OrbitalOptimizer, _run_impl, wavefunction);
  }
};

void bind_orbital_optimizer(py::module& m) {
  py::class_<OrbitalOptimizer, OrbitalOptimizerBase, py::smart_holder>
      optimizer(m, "OrbitalOptimizer", R"(
Abstract base class for objective-driven orbital optimizations.

Orbital optimizers may rotate orbitals across inactive, active, and virtual
subspace boundaries. Concrete implementations define the objective and allowed
rotation blocks.

The input is a correlated :class:`Wavefunction` (the objective depends on its
density matrices), but the result carries rotated :class:`Orbitals`, not a
wavefunction. The rotated orbitals are a proposal that must be re-solved in the
new basis.
)");

  optimizer.def(py::init<>());
  optimizer.def("run", &OrbitalOptimizer::run, py::arg("wavefunction"));
  optimizer.def("settings", &OrbitalOptimizer::settings,
                py::return_value_policy::reference_internal);
  optimizer.def_property(
      "_settings",
      [](OrbitalOptimizerBase& self) -> Settings& { return self.settings(); },
      [](OrbitalOptimizerBase& self, std::unique_ptr<Settings> settings) {
        self.replace_settings(std::move(settings));
      },
      py::return_value_policy::reference_internal);
  optimizer.def("name", &OrbitalOptimizer::name);
  optimizer.def("type_name", &OrbitalOptimizer::type_name);
  optimizer.def("hash", &OrbitalOptimizer::hash, py::arg("wavefunction"));
  optimizer.def("__repr__", [](const OrbitalOptimizer&) {
    return "<qdk_chemistry.algorithms.OrbitalOptimizer>";
  });
  qdk::chemistry::python::bind_create_nested(optimizer);

  qdk::chemistry::python::bind_algorithm_factory<
      OrbitalOptimizerFactory, OrbitalOptimizer, OrbitalOptimizerBase>(
      m, "OrbitalOptimizerFactory");
}
