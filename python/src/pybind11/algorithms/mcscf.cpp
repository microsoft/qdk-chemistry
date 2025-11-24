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

using MultiConfigurationScfReturnType =
    std::pair<double, std::shared_ptr<Wavefunction>>;

// Trampoline class for enabling Python inheritance
class MultiConfigurationScfBase
    : public MultiConfigurationScf,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, MultiConfigurationScf, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, MultiConfigurationScf, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  MultiConfigurationScfReturnType _run_impl(
      std::shared_ptr<Orbitals> orbitals,
      std::shared_ptr<HamiltonianConstructor> ham_ctor,
      std::shared_ptr<MultiConfigurationCalculator> mc_calculator,
      unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const override {
    PYBIND11_OVERRIDE_PURE(MultiConfigurationScfReturnType,
                           MultiConfigurationScf, _run_impl, orbitals, ham_ctor,
                           mc_calculator, n_active_alpha_electrons,
                           n_active_beta_electrons);
  }
};

void bind_mcscf(py::module& m) {
  // MultiConfigurationScf abstract base class
  py::class_<MultiConfigurationScf, MultiConfigurationScfBase, py::smart_holder>
      multi_configuration_scf(m, "MultiConfigurationScf", R"(
Abstract base class for multi-configurational Self-Consistent Field (MultiConfigurationScf) algorithms.

This class defines the interface for MultiConfigurationScf calculations that simultaneously optimize both molecular orbital coefficients and configuration interaction coefficients.
Concrete implementations should inherit from this class and implement the ``solve`` method.

Examples:
    To create a custom MultiConfigurationScf solver, inherit from this class:

        >>> import qdk_chemistry.algorithms as alg
        >>> import qdk_chemistry.data as data
        >>> class MyMCSCF(alg.MultiConfigurationScf):
        ...     def __init__(self):
        ...         super().__init__()  # Call the base class constructor
        ...     def _run_impl(self,
        ...                   orbitals : data.Orbitals,
        ...                   ham_ctor : alg.HamiltonianConstructor,
        ...                   mc_calculator : alg.MultiConfigurationCalculator,
        ...                   n_active_alpha_electrons : int,
        ...                   n_active_beta_electrons : int) ->tuple[float, data.Wavefunction] :
        ...         # Custom MCSCF implementation
        ...         return -1.0, data.Wavefunction()
)");

  multi_configuration_scf.def(py::init<>(),
                              R"(
Create a MultiConfigurationScf instance.

Default constructor for the abstract base class.
This should typically be called from derived class constructors.

Examples:
    >>> # In a derived class:
    >>> class MyMCSCF(alg.MultiConfigurationScf):
    ...     def __init__(self):
    ...         super().__init__()  # Calls this constructor
)");

  multi_configuration_scf.def(
      "run", &MultiConfigurationScf::run,
      R"(
Perform a MultiConfigurationScf calculation.

This method automatically locks settings before execution.

Args:
    orbitals (qdk_chemistry.data.Orbitals): The initial molecular orbitals for the calculation
    ham_ctor (qdk_chemistry.algorithms.HamiltonianConstructor): The Hamiltonian constructor for building and updating the Hamiltonian
    mc_calculator (qdk_chemistry.algorithms.MultiConfigurationCalculator): The MC calculator to evaluate the active space
    n_active_alpha_electrons (int): The number of alpha electrons in the active space
    n_active_beta_electrons (int): The number of beta electrons in the active space

Returns:
    tuple[float, qdk_chemistry.data.Wavefunction]: A tuple containing the calculated energy and the resulting wavefunction
)",
      py::arg("orbitals"), py::arg("ham_ctor"), py::arg("mc_calculator"),
      py::arg("n_active_alpha_electrons"), py::arg("n_active_beta_electrons"));

  multi_configuration_scf.def("settings", &MultiConfigurationScf::settings,
                              R"(
Access the MultiConfigurationScf calculation settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the MultiConfigurationScf calculation
)",
                              py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  multi_configuration_scf.def_property(
      "_settings",
      [](MultiConfigurationScfBase& multi_configuration_scf_inst) -> Settings& {
        return multi_configuration_scf_inst.settings();
      },
      [](MultiConfigurationScfBase& multi_configuration_scf_inst,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        multi_configuration_scf_inst.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MyMCSCF(alg.MultiConfigurationScf):
    ...     def __init__(self):
    ...         super().__init__()
    ...         from qdk_chemistry.data import ElectronicStructureSettings
    ...         self._settings = ElectronicStructureSettings()
)");

  multi_configuration_scf.def("type_name", &MultiConfigurationScf::type_name,
                              R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");

  multi_configuration_scf.def("__repr__", [](const MultiConfigurationScf&) {
    return "<qdk_chemistry.algorithms.MultiConfigurationScf>";
  });

  // Factory class binding - creates MultiConfigurationScfFactory class with
  // static methods
  qdk::chemistry::python::bind_algorithm_factory<MultiConfigurationScfFactory,
                                                 MultiConfigurationScf,
                                                 MultiConfigurationScfBase>(
      m, "MultiConfigurationScfFactory");
}
