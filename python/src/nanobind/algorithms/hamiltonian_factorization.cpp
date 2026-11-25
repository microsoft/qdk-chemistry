// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <memory>
#include <qdk/chemistry/algorithms/hamiltonian_factorization.hpp>

#include "factory_bindings.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::python;

class HamiltonianFactorizationBase : public HamiltonianFactorization {
 public:
  NB_TRAMPOLINE(HamiltonianFactorization);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  void replace_settings(std::shared_ptr<Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian) const override {
    NB_OVERRIDE_PURE(_run_impl, hamiltonian);
  }
};

void bind_hamiltonian_factorization(nb::module_& m) {
  nb::class_<HamiltonianFactorization, HamiltonianFactorizationBase>
      factorization(m, "HamiltonianFactorization", R"(
Abstract base class for Hamiltonian factorization algorithms.

Concrete implementations transform a Hamiltonian into a factorized
representation.
)");

  factorization.def(nb::init<>());
  factorization.def("run", &HamiltonianFactorization::run,
                    nb::arg("hamiltonian"));
  factorization.def("settings", &HamiltonianFactorization::settings,
                    nb::rv_policy::reference_internal);
  factorization.def_prop_rw(
      "_settings",
      [](HamiltonianFactorization& instance) -> Settings& {
        return instance.settings();
      },
      [](HamiltonianFactorization& instance,
         std::shared_ptr<Settings> new_settings) {
        dynamic_cast<HamiltonianFactorizationBase&>(instance).replace_settings(
            std::move(new_settings));
      },
      nb::rv_policy::reference_internal);
  factorization.def("name", &HamiltonianFactorization::name);
  factorization.def("aliases", &HamiltonianFactorization::aliases);
  factorization.def("type_name", &HamiltonianFactorization::type_name);
  factorization.def("hash", &HamiltonianFactorization::hash,
                    nb::arg("hamiltonian"));
  factorization.def("__repr__", [](const HamiltonianFactorization&) {
    return "<qdk_chemistry.algorithms.HamiltonianFactorization>";
  });

  bind_create_nested(factorization);
  bind_algorithm_factory<HamiltonianFactorizationFactory,
                         HamiltonianFactorization,
                         HamiltonianFactorizationBase>(
      m, "HamiltonianFactorizationFactory");
}
