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

using RefDerivedReturnType = std::pair<double, std::shared_ptr<Wavefunction>>;

// Trampoline class for python inheritance
class ReferenceDerivedCalculatorBase
    : public ReferenceDerivedCalculator,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, ReferenceDerivedCalculator, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, ReferenceDerivedCalculator,
                      aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  RefDerivedReturnType _run_impl(
      std::shared_ptr<Ansatz> ansatz) const override {
    PYBIND11_OVERRIDE_PURE(RefDerivedReturnType, ReferenceDerivedCalculator,
                           _run_impl, ansatz);
  }
};

void bind_reference_derived_calculator(py::module &m) {
  // Default implementations are automatically registered by the
  // AlgorithmFactory base class when the registry is first accessed, so no need
  // to call register_default_instances() here

  // ReferenceDerivedCalculator abstract base class
  py::class_<ReferenceDerivedCalculator, ReferenceDerivedCalculatorBase,
             py::smart_holder>
      ref_calc(m, "ReferenceDerivedCalculator", R"(
    Abstract base class for reference-derived quantum chemistry methods.

    This class provides a unified interface for quantum chemistry methods that derive
    corrections from a reference wavefunction, such as Møller-Plesset perturbation
    theory (MP2) and Coupled Cluster (CC) methods.

    The calculator takes an Ansatz (containing both Hamiltonian and reference
    wavefunction) as input and returns both the total energy and an updated
    wavefunction that may contain correlation information.

    Examples
    --------
    >>> import qdk
    >>> # Create ansatz from hamiltonian and wavefunction
    >>> ansatz = qdk.chemistry.data.Ansatz(hamiltonian, wavefunction)
    >>>
    >>> # Create calculator (e.g., MP2) using the registry
    >>> calculator = qdk.chemistry.algorithms.create("reference_derived_calculator", "microsoft_mp2_calculator")
    >>>
    >>> # Run calculation
    >>> total_energy, result_wavefunction = calculator.run(ansatz)
    )");

  ref_calc.def(py::init<>(),
               R"(
        Create a ReferenceDerivedCalculator instance.

        Initializes a new reference-derived calculator with default settings.
        Configuration options can be modified through the ``settings()`` method.

        Examples
        --------
        >>> calc = alg.ReferenceDerivedCalculator()
        >>> calc.settings().set("max_iterations", 100)
        >>> calc.settings().set("convergence_threshold", 1e-8)
        )");

  ref_calc.def("__repr__", [](const ReferenceDerivedCalculator &) {
    return "<qdk.chemistry.algorithms.ReferenceDerivedCalculator>";
  });

  ref_calc.def("run", &ReferenceDerivedCalculator::run, py::arg("ansatz"),
               R"(
    Perform reference-derived calculation.

    Parameters
    ----------
    ansatz : Ansatz
        The Ansatz (Wavefunction and Hamiltonian) describing the quantum system

    Returns
    -------
    tuple[float, Wavefunction]
        A tuple containing the total energy and the resulting wavefunction
              )");

  ref_calc.def("name", &ReferenceDerivedCalculator::name,
               "Get the algorithm name");

  ref_calc.def("type_name", &ReferenceDerivedCalculator::type_name,
               R"(
        The algorithm's type name.

        Returns
        -------
        str
            The type name of the algorithm
        )");

  ref_calc.def("settings", &ReferenceDerivedCalculator::settings,
               R"(
        Access the calculator's configuration settings.

        Returns
        -------
        qdk.chemistry.data.Settings
            Reference to the settings object for configuring the calculator
        )",
               py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  ref_calc.def_property(
      "_settings",
      [](ReferenceDerivedCalculatorBase &calculator) -> Settings & {
        return calculator.settings();
      },
      [](ReferenceDerivedCalculatorBase &calculator,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        calculator.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
        Internal settings object property.

        This property allows derived classes to replace the settings object with
        a specialized Settings subclass in their constructors.

        Examples
        --------
        >>> class MyReferenceDerivedCalculator(alg.ReferenceDerivedCalculator):
        ...     def __init__(self):
        ...         super().__init__()
        ...         from qdk.chemistry.data import ElectronicStructureSettings
        ...         self._settings = ElectronicStructureSettings()
        )");

  // Factory bindings
  bind_algorithm_factory<ReferenceDerivedCalculatorFactory,
                         ReferenceDerivedCalculator,
                         ReferenceDerivedCalculatorBase>(
      m, "ReferenceDerivedCalculatorFactory");
}
