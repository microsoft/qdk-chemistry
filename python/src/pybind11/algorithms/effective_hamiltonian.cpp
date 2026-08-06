// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class EffectiveHamiltonianConstructorBase
    : public EffectiveHamiltonianConstructor,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, EffectiveHamiltonianConstructor, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, EffectiveHamiltonianConstructor,
                      aliases);
  }

  void replace_settings(std::unique_ptr<Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Wavefunction> reference,
      std::shared_ptr<Hamiltonian> hamiltonian,
      std::shared_ptr<const SymmetryBlockedIndexSet> p_indices) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Hamiltonian>,
                           EffectiveHamiltonianConstructor, _run_impl,
                           reference, hamiltonian, p_indices);
  }
};

void bind_effective_hamiltonian_constructor(py::module &m) {
  py::class_<EffectiveHamiltonianConstructor,
             EffectiveHamiltonianConstructorBase, py::smart_holder>
      constructor(m, "EffectiveHamiltonianConstructor", R"(
Abstract base class for effective-Hamiltonian construction.

Concrete implementations construct an effective Hamiltonian from a reference
wavefunction and an input Hamiltonian in the explicitly specified P-space.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> constructor = alg.create("effective_hamiltonian_constructor", "algorithm_name")
    >>> effective_hamiltonian = constructor.run(reference, hamiltonian, p_indices)
)");

  constructor.def(py::init<>(), R"(
Create an ``EffectiveHamiltonianConstructor`` instance.

Default constructor for the abstract base class; typically called via
``super().__init__()`` from a derived class.
)");
  constructor.def("run", &EffectiveHamiltonianConstructor::run,
                  py::arg("reference"), py::arg("hamiltonian"),
                  py::arg("p_indices"), R"(
Construct the effective Hamiltonian acting on the target space ``P``.

Args:
    reference: Reference wavefunction providing the reference state.
    hamiltonian: Input Hamiltonian built over the whole window ``W = P union Q``.
    p_indices: The target space ``P`` (indices into the window's active
        space ``W``).

Returns:
    The effective Hamiltonian acting on ``P``.
)");
  constructor.def("settings", &EffectiveHamiltonianConstructor::settings,
                  py::return_value_policy::reference_internal, R"(
Access the constructor's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object.
)");
  constructor.def_property(
      "_settings",
      [](EffectiveHamiltonianConstructorBase &instance) -> Settings & {
        return instance.settings();
      },
      [](EffectiveHamiltonianConstructorBase &instance,
         std::unique_ptr<Settings> new_settings) {
        instance.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal, R"(
Internal settings object property.

Allows derived classes to replace the settings object with a specialized
Settings subclass in their constructor.
)");
  constructor.def("name", &EffectiveHamiltonianConstructor::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm
)");
  constructor.def("type_name", &EffectiveHamiltonianConstructor::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");
  constructor.def("hash", &EffectiveHamiltonianConstructor::hash,
                  py::arg("reference"), py::arg("hamiltonian"),
                  py::arg("p_indices"));
  constructor.def("__repr__", [](const EffectiveHamiltonianConstructor &) {
    return "<qdk_chemistry.algorithms.EffectiveHamiltonianConstructor>";
  });

  qdk::chemistry::python::bind_create_nested(constructor);
  qdk::chemistry::python::bind_algorithm_factory<
      EffectiveHamiltonianConstructorFactory, EffectiveHamiltonianConstructor,
      EffectiveHamiltonianConstructorBase>(
      m, "EffectiveHamiltonianConstructorFactory");
}
