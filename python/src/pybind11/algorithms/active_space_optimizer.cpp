// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class ActiveSpaceOptimizerBase : public ActiveSpaceOptimizer,
                                 public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, ActiveSpaceOptimizer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, ActiveSpaceOptimizer, aliases);
  }

  void replace_settings(std::unique_ptr<Settings> new_settings) {
    _settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<ActiveSpaceOptimizationResult> _run_impl(
      std::shared_ptr<Orbitals> orbitals, unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<ActiveSpaceOptimizationResult>,
                           ActiveSpaceOptimizer, _run_impl, orbitals,
                           n_active_alpha_electrons, n_active_beta_electrons);
  }
};

void bind_active_space_optimizer(py::module& m) {
  py::class_<ActiveSpaceOptimizer, ActiveSpaceOptimizerBase, py::smart_holder>
      optimizer(m, "ActiveSpaceOptimizer", R"(
Abstract base class for self-consistent active-space optimization algorithms.
)");

  optimizer.def(py::init<>());
  optimizer.def("run", &ActiveSpaceOptimizer::run, py::arg("orbitals"),
                py::arg("n_active_alpha_electrons"),
                py::arg("n_active_beta_electrons"));
  optimizer.def("settings", &ActiveSpaceOptimizer::settings,
                py::return_value_policy::reference_internal);
  optimizer.def_property(
      "_settings",
      [](ActiveSpaceOptimizerBase& self) -> Settings& {
        return self.settings();
      },
      [](ActiveSpaceOptimizerBase& self, std::unique_ptr<Settings> settings) {
        self.replace_settings(std::move(settings));
      },
      py::return_value_policy::reference_internal);
  optimizer.def("name", &ActiveSpaceOptimizer::name);
  optimizer.def("type_name", &ActiveSpaceOptimizer::type_name);
  optimizer.def("hash", &ActiveSpaceOptimizer::hash, py::arg("orbitals"),
                py::arg("n_active_alpha_electrons"),
                py::arg("n_active_beta_electrons"));
  optimizer.def("__repr__", [](const ActiveSpaceOptimizer&) {
    return "<qdk_chemistry.algorithms.ActiveSpaceOptimizer>";
  });
  qdk::chemistry::python::bind_create_nested(optimizer);

  qdk::chemistry::python::bind_algorithm_factory<ActiveSpaceOptimizerFactory,
                                                 ActiveSpaceOptimizer,
                                                 ActiveSpaceOptimizerBase>(
      m, "ActiveSpaceOptimizerFactory");
}
