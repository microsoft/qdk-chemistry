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
using namespace qdk::chemistry::python;

using ReturnType = std::pair<bool, std::shared_ptr<StabilityResult>>;
// Trampoline class for enabling Python inheritance
class StabilityCheckerBase : public StabilityChecker,
                             public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, StabilityChecker, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, StabilityChecker, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  ReturnType _run_impl(
      std::shared_ptr<Wavefunction> wavefunction) const override {
    PYBIND11_OVERRIDE_PURE(ReturnType, StabilityChecker, _run_impl,
                           wavefunction);
  }
};

void bind_stability(py::module &m) {
  // StabilityChecker abstract base class
  py::class_<StabilityChecker, StabilityCheckerBase, py::smart_holder>
      stability_checker(m, "StabilityChecker", R"(
Abstract base class for wavefunction stability checkers.

This class defines the interface for checking the stability of wavefunctions in quantum chemistry calculations.
Stability checking examines the second-order response of a wavefunction to determine if it corresponds to a true minimum or if there are directions in which the energy can be lowered.

Examples:
    >>> # To create a custom stability checker, inherit from this class.
    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyStabilityChecker(alg.StabilityChecker):
    ...     def __init__(self):
    ...         super().__init__()  # Call the base class constructor
    ...     # Implement the _run_impl method
    ...     def _run_impl(self, wavefunction: data.Wavefunction) -> tuple[bool, data.StabilityResult]:
    ...         # Custom stability checking implementation
    ...         import numpy as np
    ...         internal_eigenvalues = np.array([1.0, 2.0, 3.0])
    ...         external_eigenvalues = np.array([0.5, 1.5])
    ...         internal_eigenvectors = np.eye(3)
    ...         external_eigenvectors = np.eye(2)
    ...         internal_stable = np.all(internal_eigenvalues > 0)
    ...         external_stable = np.all(external_eigenvalues > 0)
    ...         result = data.StabilityResult(internal_stable, external_stable,
    ...                                       internal_eigenvalues, internal_eigenvectors,
    ...                                       external_eigenvalues, external_eigenvectors)
    ...         return (result.is_stable(), result)
)");

  stability_checker.def(py::init<>(),
                        R"(
Create a StabilityChecker instance.

Initializes a new stability checker with default settings.
Configuration options can be modified through the ``settings()`` method.

Examples:
    >>> checker = alg.StabilityChecker()
    >>> checker.settings().set("nroots", 5)
    >>> checker.settings().set("tolerance", 1e-6)
)");

  stability_checker.def("run", &StabilityChecker::run,
                        R"(
Check the stability of the given wavefunction.

This method automatically locks settings before execution and performs stability analysis on the input wavefunction by examining the eigenvalues of the electronic Hessian matrix.
A stable wavefunction should have all positive eigenvalues for both internal and external stability.

Args:
    wavefunction (qdk_chemistry.data.Wavefunction): The wavefunction to analyze for stability

Returns:
  tuple[bool, qdk_chemistry.data.StabilityResult]: Tuple of the overall stability flag and the detailed stability information (eigenvalues, eigenvectors, and helper accessors).

)",
                        py::arg("wavefunction"));

  stability_checker.def("settings", &StabilityChecker::settings,
                        R"(
Access the stability checker's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the stability checker

)",
                        py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  stability_checker.def_property(
      "_settings",
      [](StabilityCheckerBase &checker) -> Settings & {
        return checker.settings();
      },
      [](StabilityCheckerBase &checker,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        checker.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.
)");

  stability_checker.def("type_name", &StabilityChecker::type_name,
                        R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");

  // Factory class binding - creates StabilityCheckerFactory class with static
  // methods
  bind_algorithm_factory<StabilityCheckerFactory, StabilityChecker,
                         StabilityCheckerBase>(m, "StabilityCheckerFactory");

  stability_checker.def("__repr__", [](const StabilityChecker &) {
    return "<qdk_chemistry.algorithms.StabilityChecker>";
  });
}
