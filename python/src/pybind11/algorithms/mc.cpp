// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/macis_asci.hpp"
#include "qdk/chemistry/algorithms/microsoft/macis_cas.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

using ReturnType = std::pair<double, std::shared_ptr<Wavefunction>>;
// Trampoline class for enabling Python inheritance
class MultiConfigurationCalculatorBase
    : public MultiConfigurationCalculator,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, MultiConfigurationCalculator, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, MultiConfigurationCalculator,
                      aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  ReturnType _run_impl(std::shared_ptr<Hamiltonian> hamiltonian,
                       unsigned int n_active_alpha_electrons,
                       unsigned int n_active_beta_electrons) const override {
    PYBIND11_OVERRIDE_PURE(ReturnType, MultiConfigurationCalculator, _run_impl,
                           hamiltonian, n_active_alpha_electrons,
                           n_active_beta_electrons);
  }
};

void bind_mc(py::module &m) {
  // MultiConfigurationCalculator abstract base class
  py::class_<MultiConfigurationCalculator, MultiConfigurationCalculatorBase,
             py::smart_holder>
      mc_calculator(m, "MultiConfigurationCalculator",
                    R"(
Abstract base class for multi-configuration calculators.

This class defines the interface for multi configuration-based quantum chemistry calculations.
Concrete implementations should inherit from this class and implement the ``calculate`` method.

Examples:
    >>> # To create a custom multi-configuration calculator, inherit from this class.
    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyMultiConfigurationCalculator(alg.MultiConfigurationCalculator):
    ...     def __init__(self):
    ...         super().__init__()  # Call the base class constructor
    ...     # Implement the _run_impl method (called by run())
    ...     def _run_impl(self, hamiltonian: data.Hamiltonian, n_active_alpha_electrons: int, n_active_beta_electrons: int) -> tuple[float, data.Wavefunction]:
    ...         # Custom MC implementation
    ...         return energy, wavefunction

)");

  mc_calculator.def(py::init<>(),
                    R"(
Create a MultiConfigurationCalculator instance.

Default constructor for the abstract base class.
This should typically be called from derived class constructors.

Examples:
    >>> # In a derived class:
    >>> class MyCalculator(alg.MultiConfigurationCalculator):
    ...     def __init__(self):
    ...         super().__init__()  # Calls this constructor

)");

  mc_calculator.def("run", &MultiConfigurationCalculator::run,
                    R"(
Perform multi-configuration calculation on the given Hamiltonian.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian to perform the calculation on
    n_active_alpha_electrons (int): The number of alpha electrons in the active space
    n_active_beta_electrons (int): The number of beta electrons in the active space

Returns:
    tuple[float, qdk_chemistry.data.Wavefunction]: A tuple containing the computed total energy (active + core) and wavefunction

Raises:
    SettingsAreLocked: If attempting to modify settings after run() is called

)",
                    py::arg("hamiltonian"), py::arg("n_active_alpha_electrons"),
                    py::arg("n_active_beta_electrons"));

  mc_calculator.def("settings", &MultiConfigurationCalculator::settings,
                    R"(
Access the calculator's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the calculator

)",
                    py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  mc_calculator.def_property(
      "_settings",
      [](MultiConfigurationCalculatorBase &calc) -> Settings & {
        return calc.settings();
      },
      [](MultiConfigurationCalculatorBase &calc,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        calc.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MyMCCalculator(alg.MultiConfigurationCalculator):
    ...     def __init__(self):
    ...         super().__init__()
    ...         from qdk_chemistry.data import ElectronicStructureSettings
    ...         self._settings = ElectronicStructureSettings()

)");

  mc_calculator.def("type_name", &MultiConfigurationCalculator::type_name,
                    R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  // Factory class binding - creates MultiConfigurationCalculatorFactory class
  // with static methods
  qdk::chemistry::python::bind_algorithm_factory<
      MultiConfigurationCalculatorFactory, MultiConfigurationCalculator,
      MultiConfigurationCalculatorBase>(m,
                                        "MultiConfigurationCalculatorFactory");

  mc_calculator.def("__repr__", [](const MultiConfigurationCalculator &) {
    return "<qdk_chemistry.algorithms.MultiConfigurationCalculator>";
  });

  // Bind concrete microsoft::MacisCas implementation
  py::class_<microsoft::MacisCas, MultiConfigurationCalculator,
             py::smart_holder>(m, "QdkMacisCas", R"(
QDK MACIS-based Complete Active Space (CAS) calculator.

This class provides a concrete implementation of the multi-configuration
calculator using the MACIS library for Complete Active Space Configuration
Interaction (CASCI) calculations.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create a CASCI calculator
    casci = alg.QdkMacisCas()

    # Configure settings if needed
    casci.settings().set("max_iterations", 100)

    # Run calculation
    energy, wavefunction = casci.run(hamiltonian, n_alpha, n_beta)

See Also:
    :class:`MultiConfigurationCalculator`
    :class:`qdk_chemistry.data.Hamiltonian`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a MACIS CAS calculator with default settings.

)");

  // Bind concrete microsoft::MacisAsci implementation
  py::class_<microsoft::MacisAsci, MultiConfigurationCalculator,
             py::smart_holder>(m, "QdkMacisAsci", R"(
QDK MACIS-based Adaptive Sampling Configuration Interaction (ASCI) calculator.

This class provides a concrete implementation of the multi-configuration
calculator using the MACIS library for Adaptive Sampling Configuration
Interaction (ASCI) calculations, which adaptively selects the most important
configurations.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    # Create an ASCI calculator
    asci = alg.QdkMacisAsci()

    # Configure ASCI-specific settings
    asci.settings().set("ntdets_max", 1000)
    asci.settings().set("h_el_tol", 1e-6)

    # Run calculation
    energy, wavefunction = asci.run(hamiltonian, n_alpha, n_beta)

See Also:
    :class:`MultiConfigurationCalculator`
    :class:`qdk_chemistry.data.Hamiltonian`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a MACIS ASCI calculator with default settings.

)");
}
