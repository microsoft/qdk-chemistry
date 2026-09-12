// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <qdk/chemistry/algorithms/hamiltonian_factorization.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::python;

class HamiltonianFactorizationBase
    : public HamiltonianFactorization,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, HamiltonianFactorization, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, HamiltonianFactorization,
                      aliases);
  }

  void replace_settings(std::unique_ptr<Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Hamiltonian>,
                           HamiltonianFactorization, _run_impl, hamiltonian);
  }
};

void bind_hamiltonian_factorization(py::module& m) {
  py::class_<HamiltonianFactorization, HamiltonianFactorizationBase,
             py::smart_holder>
      factorization(m, "HamiltonianFactorization", R"(
Abstract base class for Hamiltonian factorization algorithms.

Concrete implementations transform a Hamiltonian into a factorized
representation.
)");

  factorization.def(py::init<>());
  factorization.def("run", &HamiltonianFactorization::run,
                    py::arg("hamiltonian"));
  factorization.def("settings", &HamiltonianFactorization::settings,
                    py::return_value_policy::reference_internal);
  factorization.def_property(
      "_settings",
      [](HamiltonianFactorizationBase& instance) -> Settings& {
        return instance.settings();
      },
      [](HamiltonianFactorizationBase& instance,
         std::unique_ptr<Settings> new_settings) {
        instance.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal);
  factorization.def("name", &HamiltonianFactorization::name);
  factorization.def("aliases", &HamiltonianFactorization::aliases);
  factorization.def("type_name", &HamiltonianFactorization::type_name);
  factorization.def("hash", &HamiltonianFactorization::hash,
                    py::arg("hamiltonian"));
  factorization.def("__repr__", [](const HamiltonianFactorization&) {
    return "<qdk_chemistry.algorithms.HamiltonianFactorization>";
  });

  bind_create_nested(factorization);
  bind_algorithm_factory<HamiltonianFactorizationFactory,
                         HamiltonianFactorization,
                         HamiltonianFactorizationBase>(
      m, "HamiltonianFactorizationFactory");
}
