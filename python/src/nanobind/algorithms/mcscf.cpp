// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

using MultiConfigurationScfReturnType =
    std::pair<double, std::shared_ptr<Wavefunction>>;

// Trampoline class for enabling Python inheritance
class MultiConfigurationScfBase : public MultiConfigurationScf {
 public:
  NB_TRAMPOLINE(MultiConfigurationScf);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  MultiConfigurationScfReturnType _run_impl(
      std::shared_ptr<Orbitals> orbitals, unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const override {
    NB_OVERRIDE_PURE(_run_impl, orbitals, n_active_alpha_electrons,
                     n_active_beta_electrons);
  }
};

void bind_mcscf(nb::module_& m) {
  // MultiConfigurationScf abstract base class
  nb::class_<MultiConfigurationScf, MultiConfigurationScfBase>
      multi_configuration_scf(m, "MultiConfigurationScf", R"(
Abstract base class for multi-configurational Self-Consistent Field (MultiConfigurationScf) algorithms.

This class defines the interface for MultiConfigurationScf calculations that simultaneously optimize both molecular orbital coefficients and configuration interaction coefficients.
Concrete implementations should inherit from this class and implement the ``solve`` method.

Examples:
    >>> # To create a custom MultiConfigurationScf algorithm, inherit from this class.
    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyMCSCF(alg.MultiConfigurationScf):
    ...     def __init__(self):
    ...         super().__init__()  # Call the base class constructor
    ...     def _run_impl(self,
    ...                   orbitals : data.Orbitals,
    ...                   n_active_alpha_electrons : int,
    ...                   n_active_beta_electrons : int) ->tuple[float, data.Wavefunction] :
    ...         # Custom MCSCF implementation
    ...         # Use self._create_nested("multi_configuration_calculator")
    ...         # to get the configurable nested algorithm exposed by these settings.
    ...         return -1.0, data.Wavefunction()
)");

  multi_configuration_scf.def(nb::init<>(),
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

  multi_configuration_scf.def("run", &MultiConfigurationScf::run,
                              R"(
Perform a MultiConfigurationScf calculation.

This method automatically locks settings before execution.
The nested HamiltonianConstructor and MultiConfigurationCalculator are
created from AlgorithmRef entries in settings.

Args:
    orbitals (qdk_chemistry.data.Orbitals): The initial molecular orbitals for the calculation
    n_active_alpha_electrons (int): The number of alpha electrons in the active space
    n_active_beta_electrons (int): The number of beta electrons in the active space

Returns:
    tuple[float, qdk_chemistry.data.Wavefunction]: A tuple containing the calculated energy and the resulting wavefunction

)",
                              nb::arg("orbitals"),
                              nb::arg("n_active_alpha_electrons"),
                              nb::arg("n_active_beta_electrons"));

  multi_configuration_scf.def("settings", &MultiConfigurationScf::settings,
                              R"(
Access the MultiConfigurationScf calculation settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the MultiConfigurationScf calculation

)",
                              nb::rv_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  multi_configuration_scf.def_prop_rw(
      "_settings",
      [](MultiConfigurationScf& multi_configuration_scf_inst) -> Settings& {
        return multi_configuration_scf_inst.settings();
      },
      [](MultiConfigurationScf& multi_configuration_scf_inst,
         std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
        dynamic_cast<MultiConfigurationScfBase&>(multi_configuration_scf_inst)
            .replace_settings(std::move(new_settings));
      },
      nb::rv_policy::reference_internal,
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

  multi_configuration_scf.def("name", &MultiConfigurationScf::name,
                              R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  multi_configuration_scf.def("aliases", &MultiConfigurationScf::aliases,
                              R"(
The algorithm's aliases.

Returns:
    list[str]: All registered names for the algorithm

)");

  multi_configuration_scf.def("type_name", &MultiConfigurationScf::type_name,
                              R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  multi_configuration_scf.def(
      "hash", &MultiConfigurationScf::hash, nb::arg("orbitals"),
      nb::arg("n_active_alpha_electrons"), nb::arg("n_active_beta_electrons"));

  multi_configuration_scf.def("__repr__", [](const MultiConfigurationScf&) {
    return "<qdk_chemistry.algorithms.MultiConfigurationScf>";
  });

  // Add _create_nested so C++-backed classes can instantiate nested algorithms
  qdk::chemistry::python::bind_create_nested(multi_configuration_scf);

  // Factory class binding - creates MultiConfigurationScfFactory class with
  // static methods
  qdk::chemistry::python::bind_algorithm_factory<MultiConfigurationScfFactory,
                                                 MultiConfigurationScf,
                                                 MultiConfigurationScfBase>(
      m, "MultiConfigurationScfFactory");
}
