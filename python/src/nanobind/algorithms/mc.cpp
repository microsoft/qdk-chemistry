// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/macis_asci.hpp"
#include "qdk/chemistry/algorithms/microsoft/macis_cas.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

using ReturnType = std::pair<double, std::shared_ptr<Wavefunction>>;
// Trampoline class for enabling Python inheritance
class MultiConfigurationCalculatorBase : public MultiConfigurationCalculator {
 public:
  NB_TRAMPOLINE(MultiConfigurationCalculator);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  ReturnType _run_impl(std::shared_ptr<Hamiltonian> hamiltonian,
                       unsigned int n_active_alpha_electrons,
                       unsigned int n_active_beta_electrons) const override {
    NB_OVERRIDE_PURE(_run_impl, hamiltonian, n_active_alpha_electrons,
                     n_active_beta_electrons);
  }
};

void bind_mc(nb::module_ &m) {
  // MultiConfigurationCalculator abstract base class
  nb::class_<MultiConfigurationCalculator, MultiConfigurationCalculatorBase>
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

  mc_calculator.def(nb::init<>(),
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
                    nb::arg("hamiltonian"), nb::arg("n_active_alpha_electrons"),
                    nb::arg("n_active_beta_electrons"));

  mc_calculator.def("settings", &MultiConfigurationCalculator::settings,
                    R"(
Access the calculator's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the calculator

)",
                    nb::rv_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  mc_calculator.def_prop_rw(
      "_settings",
      [](MultiConfigurationCalculator &calc) -> Settings & {
        return calc.settings();
      },
      [](MultiConfigurationCalculator &calc,
         std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
        dynamic_cast<MultiConfigurationCalculatorBase &>(calc).replace_settings(
            std::move(new_settings));
      },
      nb::rv_policy::reference_internal,
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

  mc_calculator.def("name", &MultiConfigurationCalculator::name,
                    R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  mc_calculator.def("aliases", &MultiConfigurationCalculator::aliases,
                    R"(
The algorithm's aliases.

Returns:
    list[str]: All registered names for the algorithm

)");

  mc_calculator.def("type_name", &MultiConfigurationCalculator::type_name,
                    R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  mc_calculator.def("hash", &MultiConfigurationCalculator::hash,
                    nb::arg("hamiltonian"), nb::arg("n_active_alpha_electrons"),
                    nb::arg("n_active_beta_electrons"));

  // Factory class binding - creates MultiConfigurationCalculatorFactory class
  // with static methods
  qdk::chemistry::python::bind_algorithm_factory<
      MultiConfigurationCalculatorFactory, MultiConfigurationCalculator,
      MultiConfigurationCalculatorBase>(m,
                                        "MultiConfigurationCalculatorFactory");

  mc_calculator.def("__repr__", [](const MultiConfigurationCalculator &) {
    return "<qdk_chemistry.algorithms.MultiConfigurationCalculator>";
  });

  qdk::chemistry::python::bind_create_nested(mc_calculator);

  // Bind concrete microsoft::MacisCas implementation
  nb::class_<microsoft::MacisCas, MultiConfigurationCalculator>(
      m, "QdkMacisCas", R"(
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
      .def(nb::init<>(), R"(
Default constructor.

Initializes a MACIS CAS calculator with default settings.

)");

  // Bind concrete microsoft::MacisAsci implementation
  nb::class_<microsoft::MacisAsci, MultiConfigurationCalculator>(
      m, "QdkMacisAsci", R"(
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
    asci.settings().set("search_matel_tol", 1e-6)

    # Run calculation
    energy, wavefunction = asci.run(hamiltonian, n_alpha, n_beta)

See Also:
    :class:`MultiConfigurationCalculator`
    :class:`qdk_chemistry.data.Hamiltonian`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes a MACIS ASCI calculator with default settings.

)");
}
