// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/finite_difference_nuclear_derivative.hpp"
#include "qdk/chemistry/algorithms/qdk_nuclear_derivative.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class NuclearDerivativeCalculatorBase : public NuclearDerivativeCalculator {
 public:
  NB_TRAMPOLINE(NuclearDerivativeCalculator);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  void replace_settings(
      std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  NuclearDerivativeResult _run_impl(
      std::shared_ptr<Structure> structure, int charge, int spin_multiplicity,
      NuclearDerivativeSeedType seed,
      unsigned int n_inactive_orbitals) const override {
    NB_OVERRIDE_PURE(_run_impl, structure, charge, spin_multiplicity, seed,
                     n_inactive_orbitals);
  }
};

void bind_nuclear_derivative(nb::module_& m) {
  nb::class_<NuclearDerivativeCalculator, NuclearDerivativeCalculatorBase>
      calculator(m, "NuclearDerivativeCalculator",
                 R"(
    Base class for nuclear derivative algorithms.

    Calculators return the total energy, nuclear gradients, an optional Hessian,
    and an optional wavefunction for a molecular structure.
    )");

  calculator.def(nb::init<>(), R"(Create a nuclear derivative calculator.)");
  calculator.def(
      "run",
      [](const NuclearDerivativeCalculator& self,
         std::shared_ptr<Structure> structure, int charge,
         int spin_multiplicity, NuclearDerivativeSeedType seed,
         unsigned int n_inactive_orbitals) {
        return self.run(structure, charge, spin_multiplicity, seed,
                        n_inactive_orbitals);
      },
      nb::arg("structure"), nb::arg("charge"), nb::arg("spin_multiplicity"),
      nb::arg("seed_or_basis"), nb::arg("n_inactive_orbitals") = 0,
      R"(
Compute nuclear derivatives for a molecular structure.

Args:
    structure: Molecular structure to evaluate.
    charge: Total molecular charge.
    spin_multiplicity: Spin multiplicity of the molecular system.
    seed_or_basis: Basis name, basis set, orbitals, or wavefunction seed.
    n_inactive_orbitals: Number of doubly occupied orbitals excluded from multi-reference active spaces. This value is validated against the electron count even when the selected energy path does not use an active space (e.g., SCF).

Returns:
    tuple: ``(energy, gradients, hessian, wavefunction)``.
)");
  calculator.def(
      "hash", &NuclearDerivativeCalculator::hash, nb::arg("structure"),
      nb::arg("charge"), nb::arg("spin_multiplicity"), nb::arg("seed_or_basis"),
      nb::arg("n_inactive_orbitals") = 0,
      R"(Compute a deterministic content hash for a nuclear derivative run.)");
  calculator.def("settings", &NuclearDerivativeCalculator::settings,
                 nb::rv_policy::reference_internal,
                 R"(Return the calculator settings.)");
  calculator.def_prop_rw(
      "_settings",
      [](NuclearDerivativeCalculator& self) -> Settings& {
        return self.settings();
      },
      [](NuclearDerivativeCalculator& self,
         std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
        dynamic_cast<NuclearDerivativeCalculatorBase&>(self).replace_settings(
            std::move(new_settings));
      },
      nb::rv_policy::reference_internal,
      R"(Internal settings replacement hook for Python subclasses.)");
  calculator.def("name", &NuclearDerivativeCalculator::name,
                 R"(Return the implementation name.)");
  calculator.def("aliases", &NuclearDerivativeCalculator::aliases,
                 R"(Return all registered names for the implementation.)");
  calculator.def("type_name", &NuclearDerivativeCalculator::type_name,
                 R"(Return the algorithm type name.)");
  calculator.def("__repr__", [](const NuclearDerivativeCalculator& self) {
    return "<qdk_chemistry.algorithms.NuclearDerivativeCalculator name='" +
           self.name() + "'>";
  });

  qdk::chemistry::python::bind_algorithm_factory<
      NuclearDerivativeCalculatorFactory, NuclearDerivativeCalculator,
      NuclearDerivativeCalculatorBase>(m, "NuclearDerivativeCalculatorFactory");
  qdk::chemistry::python::bind_create_nested(calculator);

  nb::class_<FiniteDifferenceNuclearDerivativeCalculator,
             NuclearDerivativeCalculator>(
      m, "FiniteDifferenceNuclearDerivativeCalculator",
      R"(Numeric nuclear derivative calculator using central finite differences.)")
      .def(nb::init<>(),
           R"(Create a finite-difference derivative calculator.)");

  nb::class_<QdkNuclearDerivativeCalculator, NuclearDerivativeCalculator>(
      m, "QdkNuclearDerivativeCalculator",
      R"(QDK nuclear derivative calculator using analytic internal SCF gradients.)")
      .def(nb::init<>(),
           R"(Create a QDK analytic nuclear derivative calculator.)");
}
