// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/macis_pmc.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

using ReturnType = std::pair<double, std::shared_ptr<Wavefunction>>;

// Trampoline class for enabling Python inheritance
class ProjectedMultiConfigurationCalculatorBase
    : public ProjectedMultiConfigurationCalculator,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, ProjectedMultiConfigurationCalculator,
                           name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>,
                      ProjectedMultiConfigurationCalculator, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  ReturnType _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian,
      const std::vector<Configuration> &configurations) const override {
    PYBIND11_OVERRIDE_PURE(ReturnType, ProjectedMultiConfigurationCalculator,
                           _run_impl, hamiltonian, configurations);
  }
};

void bind_pmc(py::module &m) {
  // ProjectedMultiConfigurationCalculator abstract base class
  py::class_<ProjectedMultiConfigurationCalculator,
             ProjectedMultiConfigurationCalculatorBase, py::smart_holder>
      pmc_calculator(m, "ProjectedMultiConfigurationCalculator",
                     R"(
Abstract base class for projected multi-configurational (PMC) calculations in quantum chemistry.

This class provides the interface for projected multi-configurational-based quantum chemistry calculations.
This contrasts the ``MultiConfigurationCalculator`` in that the space of determinants upon which the Hamiltonian is projected is taken to be a *free parameter* and must be specified.
In this manner, the high-performance solvers which underly other MC algorithms can be interfaced with external methods for selecting important determinants.

The calculator takes a Hamiltonian and a set of configurations as input and returns both the calculated total energy and the corresponding multi-configurational wavefunction.

To create a custom PMC calculator, inherit from this class.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyProjectedMultiConfigurationCalculator(alg.ProjectedMultiConfigurationCalculator):
    ...     def __init__(self):
    ...         super().__init__()  # Call the base class constructor
    ...     def _run_impl(self, hamiltonian: data.Hamiltonian, configurations: list[data.Configuration]) -> tuple[float, data.Wavefunction]:
    ...         # Custom PMC implementation
    ...         return energy, wavefunction

)");

  pmc_calculator.def(py::init<>(),
                     R"(
Create a ProjectedMultiConfigurationCalculator instance.

Default constructor for the abstract base class.
This should typically be called from derived class constructors.

Examples:
    >>> # In a derived class:
    >>> class MyPMC(alg.ProjectedMultiConfigurationCalculator):
    ...     def __init__(self):
    ...         super().__init__()  # Calls this constructor

)");

  pmc_calculator.def("run", &ProjectedMultiConfigurationCalculator::run,
                     R"(
Perform projected multi-configurational calculation.

This method automatically locks settings before execution.

This method takes a Hamiltonian describing the quantum system and a set of configurations/determinants to project the Hamiltonian onto, and returns both the calculated energy and the optimized multi-configurational wavefunction.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian operator describing the quantum system
    configurations (list[qdk_chemistry.data.Configuration]): The set of configurations/determinants to project the Hamiltonian onto

Returns:
    tuple[float, qdk_chemistry.data.Wavefunction]: A tuple containing the calculated total energy (active + core) and the resulting multi-configurational wavefunction

Raises:
    RuntimeError: If the calculation fails
    ValueError: If hamiltonian or configurations are invalid

)",
                     py::arg("hamiltonian"), py::arg("configurations"));

  pmc_calculator.def("settings",
                     &ProjectedMultiConfigurationCalculator::settings,
                     R"(
Access the calculator's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the calculator
)",
                     py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  pmc_calculator.def_property(
      "_settings",
      [](ProjectedMultiConfigurationCalculatorBase &calc) -> Settings & {
        return calc.settings();
      },
      [](ProjectedMultiConfigurationCalculatorBase &calc,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        calc.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MyPMC(alg.ProjectedMultiConfigurationCalculator):
    ...     def __init__(self):
    ...         super().__init__()
    ...         from qdk_chemistry.data import ElectronicStructureSettings
    ...         self._settings = ElectronicStructureSettings()

)");

  pmc_calculator.def("type_name",
                     &ProjectedMultiConfigurationCalculator::type_name,
                     R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  // Factory class binding - creates
  // ProjectedMultiConfigurationCalculatorFactory class with static methods
  qdk::chemistry::python::bind_algorithm_factory<
      ProjectedMultiConfigurationCalculatorFactory,
      ProjectedMultiConfigurationCalculator,
      ProjectedMultiConfigurationCalculatorBase>(
      m, "ProjectedMultiConfigurationCalculatorFactory");

  pmc_calculator.def("__repr__", [](const ProjectedMultiConfigurationCalculator
                                        &) {
    return "<qdk_chemistry.algorithms.ProjectedMultiConfigurationCalculator>";
  });

  // Bind concrete microsoft::MacisPmc implementation
  py::class_<microsoft::MacisPmc, ProjectedMultiConfigurationCalculator,
             py::smart_holder>(m, "QdkMacisPmc", R"(
QDK MACIS-based Projected Multi-Configuration (PMC) calculator.

This class provides a concrete implementation of the projected multi-configuration
calculator using the MACIS library. It performs projections of the Hamiltonian
onto a specified set of determinants to compute energies and wavefunctions for
strongly correlated molecular systems.

The calculator inherits from :class:`ProjectedMultiConfigurationCalculator` and uses
MACIS library routines to perform the actual projected calculations where
the determinant space is provided as input rather than generated adaptively.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg
    import qdk_chemistry.data as data

    # Create a PMC calculator
    pmc = alg.QdkMacisPmc()

    # Configure PMC-specific settings
    pmc.settings().set("davidson_res_tol", 1e-8)
    pmc.settings().set("davidson_max_m", 200)

    # Prepare configurations
    configurations = [...]  # Your list of Configuration objects

    # Run calculation
    energy, wavefunction = pmc.run(hamiltonian, configurations)

See Also:
    :class:`ProjectedMultiConfigurationCalculator`
    :class:`qdk_chemistry.data.Hamiltonian`
    :class:`qdk_chemistry.data.Configuration`
    :class:`qdk_chemistry.data.Wavefunction`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a MACIS PMC calculator with default settings.

)");
}
