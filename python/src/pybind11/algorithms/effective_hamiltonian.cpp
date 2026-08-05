/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::python;

// Trampoline class for python inheritance
class EffectiveHamiltonianBase : public EffectiveHamiltonian,
                                 public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, EffectiveHamiltonian, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, EffectiveHamiltonian, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian,
      std::shared_ptr<Wavefunction> wavefunction,
      std::shared_ptr<const qdk::chemistry::data::SymmetryBlockedIndexSet>
          p_space_indices) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Hamiltonian>, EffectiveHamiltonian,
                           _run_impl, hamiltonian, wavefunction,
                           p_space_indices);
  }
};

void bind_effective_hamiltonian(py::module &m) {
  // Default instances are registered lazily by the AlgorithmFactory base class
  // the first time the registry is accessed.

  py::class_<EffectiveHamiltonian, EffectiveHamiltonianBase, py::smart_holder>
      eff(m, "EffectiveHamiltonian", R"(
Abstract base class for effective-Hamiltonian (downfolding) algorithms.

An effective-Hamiltonian algorithm transforms a full-space Hamiltonian into an
effective active-space Hamiltonian that folds in dynamical correlation from the
external (non-active) orbitals. The DUCC family realizes this via a unitary
coupled-cluster similarity transformation evaluated through a truncated
Baker-Campbell-Hausdorff (BCH) expansion.

The run signature takes the full-space Hamiltonian, a full-space Wavefunction
supplying the reference coupled-cluster amplitudes (through its amplitude
container), and a SymmetryBlockedIndexSet designating the active (P-space)
orbitals per spin channel. The only setting is the BCH truncation level
(``ducc_level``).

Examples:
  >>> import qdk_chemistry
  >>> builder = qdk_chemistry.algorithms.create("effective_hamiltonian", "ducc")
  >>> builder.settings().set("ducc_level", 2)
  >>> effective = builder.run(hamiltonian, ccsd_wavefunction, p_space_indices)
    )");

  eff.def(py::init<>(),
          R"(
  Create an EffectiveHamiltonian instance.

  Initializes a new effective-Hamiltonian builder with default settings.
  Configuration options can be modified through the ``settings()`` method.
        )");

  eff.def("__repr__", [](const EffectiveHamiltonian &) {
    return "<qdk_chemistry.algorithms.EffectiveHamiltonian>";
  });

  qdk::chemistry::python::bind_create_nested(eff);

  eff.def("run", &EffectiveHamiltonian::run, py::arg("hamiltonian"),
          py::arg("wavefunction"), py::arg("p_space_indices"),
          R"(
  Build the effective active-space Hamiltonian.

  Args:
    hamiltonian (Hamiltonian): The full-space Hamiltonian to transform.
    wavefunction (Wavefunction): A full-space wavefunction whose amplitude container supplies the reference coupled-cluster amplitudes.
    p_space_indices (SymmetryBlockedIndexSet): Active-space (P-space) orbital indices per spin channel.

  Returns:
    Hamiltonian: The effective active-space Hamiltonian.
              )");

  eff.def("name", &EffectiveHamiltonian::name, "Get the algorithm name");

  eff.def("type_name", &EffectiveHamiltonian::type_name,
          R"(
The algorithm's type name.

Returns:
  str: The type name of the algorithm ("effective_hamiltonian")
        )");

  eff.def("hash", &EffectiveHamiltonian::hash, py::arg("hamiltonian"),
          py::arg("wavefunction"), py::arg("p_space_indices"));

  eff.def("settings", &EffectiveHamiltonian::settings,
          R"(
Access the builder's configuration settings.

Returns:
  qdk_chemistry.data.Settings: Reference to the settings object.
        )",
          py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  eff.def_property(
      "_settings",
      [](EffectiveHamiltonianBase &builder) -> Settings & {
        return builder.settings();
      },
      [](EffectiveHamiltonianBase &builder,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        builder.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

Allows derived classes to replace the settings object with a specialized
Settings subclass in their constructors.
        )");

  // Factory bindings
  bind_algorithm_factory<EffectiveHamiltonianFactory, EffectiveHamiltonian,
                         EffectiveHamiltonianBase>(
      m, "EffectiveHamiltonianFactory");
}
