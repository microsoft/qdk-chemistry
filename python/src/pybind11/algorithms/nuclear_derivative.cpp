// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/finite_difference_nuclear_derivative.hpp"
#include "qdk/chemistry/algorithms/qdk_nuclear_derivative.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class NuclearDerivativeCalculatorBase
    : public NuclearDerivativeCalculator,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, NuclearDerivativeCalculator, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, NuclearDerivativeCalculator,
                      aliases);
  }

  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  NuclearDerivativeResult _run_impl(
      std::shared_ptr<Structure> structure, int charge, int spin_multiplicity,
      NuclearDerivativeSeedType seed, unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const override {
    PYBIND11_OVERRIDE_PURE(NuclearDerivativeResult, NuclearDerivativeCalculator,
                           _run_impl, structure, charge, spin_multiplicity,
                           seed, n_active_alpha_electrons,
                           n_active_beta_electrons);
  }
};

void bind_nuclear_derivative(py::module& m) {
  py::class_<NuclearDerivativeCalculator, NuclearDerivativeCalculatorBase,
             py::smart_holder>
      calculator(m, "NuclearDerivativeCalculator",
                 R"(
    Base class for nuclear derivative algorithms.

    Calculators return the total energy, nuclear gradients, an optional Hessian,
    and an optional wavefunction for a molecular structure.
    )");

  calculator.def(py::init<>(), R"(Create a nuclear derivative calculator.)");
  calculator.def(
      "run",
      [](const NuclearDerivativeCalculator& self,
         std::shared_ptr<Structure> structure, int charge,
         int spin_multiplicity, NuclearDerivativeSeedType seed,
         unsigned int n_active_alpha_electrons,
         unsigned int n_active_beta_electrons) {
        return self.run(structure, charge, spin_multiplicity, seed,
                        n_active_alpha_electrons, n_active_beta_electrons);
      },
      py::arg("structure"), py::arg("charge"), py::arg("spin_multiplicity"),
      py::arg("seed_or_basis"), py::arg("n_active_alpha_electrons"),
      py::arg("n_active_beta_electrons"),
      R"(
Compute nuclear derivatives for a molecular structure.

Args:
    structure: Molecular structure to evaluate.
    charge: Total molecular charge.
    spin_multiplicity: Spin multiplicity of the molecular system.
    seed_or_basis: Basis name, basis set, orbitals, or wavefunction seed.
    n_active_alpha_electrons: Active-space alpha electron count for multi-reference energy paths.
    n_active_beta_electrons: Active-space beta electron count for multi-reference energy paths.

Returns:
    tuple: ``(energy, gradients, hessian, wavefunction)``.
)");
  calculator.def("settings", &NuclearDerivativeCalculator::settings,
                 py::return_value_policy::reference_internal,
                 R"(Return the calculator settings.)");
  calculator.def_property(
      "_settings",
      [](NuclearDerivativeCalculatorBase& self) -> Settings& {
        return self.settings();
      },
      [](NuclearDerivativeCalculatorBase& self,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        self.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(Internal settings replacement hook for Python subclasses.)");
  calculator.def("name", &NuclearDerivativeCalculator::name,
                 R"(Return the implementation name.)");
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

  py::class_<FiniteDifferenceNuclearDerivativeCalculator,
             NuclearDerivativeCalculator, py::smart_holder>(
      m, "FiniteDifferenceNuclearDerivativeCalculator",
      R"(Numeric nuclear derivative calculator using central finite differences.)")
      .def(py::init<>(),
           R"(Create a finite-difference derivative calculator.)");

  py::class_<QdkNuclearDerivativeCalculator, NuclearDerivativeCalculator,
             py::smart_holder>(
      m, "QdkNuclearDerivativeCalculator",
      R"(QDK nuclear derivative calculator using analytic internal SCF gradients.)")
      .def(py::init<>(),
           R"(Create a QDK analytic nuclear derivative calculator.)");
}
