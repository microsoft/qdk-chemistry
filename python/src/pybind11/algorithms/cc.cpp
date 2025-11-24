// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

using CCReturnType =
    std::pair<double, std::shared_ptr<CoupledClusterAmplitudes>>;
// Trampoline class for enabling Python inheritance
class CoupledClusterCalculatorBase
    : public CoupledClusterCalculator,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, CoupledClusterCalculator, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, CoupledClusterCalculator,
                      aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  CCReturnType _run_impl(std::shared_ptr<Ansatz> ansatz) const override {
    PYBIND11_OVERRIDE_PURE(CCReturnType, CoupledClusterCalculator, _run_impl,
                           ansatz);
  }
};

void bind_cc(py::module &m) {
  // CoupledClusterCalculator abstract base class
  py::class_<CoupledClusterCalculator, CoupledClusterCalculatorBase,
             py::smart_holder>
      cc_calculator(m, "CoupledClusterCalculator",
                    R"(
Abstract base class for coupled cluster calculators.

This class defines the interface for coupled cluster-based quantum chemistry calculations.
Concrete implementations should inherit from this class and implement the ``calculate`` method.

Examples:
    To create a custom coupled cluster calculator, inherit from this class:

        >>> import qdk_chemistry.algorithms as alg
        >>> import qdk_chemistry.data as data
        >>> class MyCCCalculator(alg.CoupledClusterCalculator):
        ...     def __init__(self):
        ...         super().__init__()  # Call the base class constructor
        ...     # Implement the _run_impl method (called by calculate())
        ...     def _run_impl(self, ansatz: data.Ansatz) -> tuple[float, data.CoupledClusterAmplitudes]:
            ...         # Custom CC implementation
            ...         return energy, coupled_cluster_amplitudes
)");

  cc_calculator.def(py::init<>(),
                    R"(
Create a ``CoupledClusterCalculator`` instance.

Initializes a new coupled cluster calculator with default settings.
Configuration options such as convergence criteria and maximum iterations can be modified through the ``settings()`` method.

Examples:
    >>> cc = alg.CoupledClusterCalculator()
    >>> cc.settings().set("max_iterations", 50)
    >>> cc.settings().set("energy_threshold", 1e-8)
)");

  cc_calculator.def("run", &CoupledClusterCalculator::run,
                    R"(
Perform coupled cluster calculation on the given ``Ansatz``.

Args:
    ansatz (qdk_chemistry.data.Ansatz): The ``Ansatz`` (``Wavefunction`` and ``Hamiltonian``) describing the quantum system

Returns:
    tuple[float, qdk_chemistry.data.CoupledClusterAmplitudes]: A tuple containing the computed energy and coupled cluster amplitudes

Raises:
    SettingsAreLocked: If attempting to modify settings after run() is called
)",
                    py::arg("ansatz"));

  cc_calculator.def("settings", &CoupledClusterCalculator::settings,
                    R"(
Access the calculator's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the calculator
)",
                    py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  cc_calculator.def_property(
      "_settings",
      [](CoupledClusterCalculatorBase &calc) -> Settings & {
        return calc.settings();
      },
      [](CoupledClusterCalculatorBase &calc,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        calc.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MyCCCalculator(alg.CoupledClusterCalculator):
    ...     def __init__(self):
    ...         super().__init__()
    ...         from qdk_chemistry.data import ElectronicStructureSettings
    ...         self._settings = ElectronicStructureSettings()
)");

  cc_calculator.def("type_name", &CoupledClusterCalculator::type_name,
                    R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");

  // Factory class binding - creates CoupledClusterCalculatorFactory class
  // with static methods
  qdk::chemistry::python::bind_algorithm_factory<
      CoupledClusterCalculatorFactory, CoupledClusterCalculator,
      CoupledClusterCalculatorBase>(m, "CoupledClusterCalculatorFactory");

  cc_calculator.def("__repr__", [](const CoupledClusterCalculator &) {
    return "<qdk_chemistry.algorithms.CoupledClusterCalculator>";
  });
}
