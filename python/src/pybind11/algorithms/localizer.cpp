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

// Trampoline class for enabling Python inheritance
class LocalizerBase : public Localizer,
                      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, Localizer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, Localizer, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Wavefunction> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction,
      const std::vector<size_t> &loc_indices_a,
      const std::vector<size_t> &loc_indices_b) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Wavefunction>, Localizer, _run_impl,
                           wavefunction, loc_indices_a, loc_indices_b);
  }
};

void bind_localizer(py::module &m) {
  // Localizer abstract base class
  py::class_<Localizer, LocalizerBase, py::smart_holder> localizer(
      m, "Localizer", R"(
    Abstract base class for orbital localizers.

    This class defines the interface for localizing molecular orbitals.
    Localization transforms canonical molecular orbitals into localized
    orbitals that are spatially confined to specific regions or bonds.
    Concrete implementations should inherit from this class and implement
    the localize method.

    Examples
    --------
    To create a custom orbital localizer, inherit from this class:

        >>> import qdk_chemistry.algorithms as alg
        >>> import qdk_chemistry.data as data
        >>> class MyLocalizer(alg.Localizer):
        ...     def __init__(self):
        ...         super().__init__()  # Call the base class constructor
        ...     # Implement the _run_impl method
        ...     def _run_impl(self, wavefunction: data.Wavefunction, loc_indices_a: list, loc_indices_b: list) -> data.Wavefunction:
        ...         # Custom localization implementation
        ...         return localized_wavefunction
    )");

  localizer.def(py::init<>(),
                R"(
        Create a Localizer instance.

        Default constructor for the abstract base class.
        This should typically be called from derived class constructors.

        Examples
        --------
        >>> # In a derived class:
        >>> class MyLocalizer(alg.Localizer):
        ...     def __init__(self):
        ...         super().__init__()  # Calls this constructor
        )");

  localizer.def("run", &Localizer::run,
                R"(
        Localize molecular orbitals in the given wavefunction.

        Parameters
        ----------
        wavefunction : qdk_chemistry.data.Wavefunction
            The canonical molecular wavefunction to localize
        loc_indices_a : list of int
            Indices of alpha orbitals to localize (empty for no localization)
        loc_indices_b : list of int
            Indices of beta orbitals to localize (empty for no localization)
            Note: For restricted orbitals, must be identical to loc_indices_a

        Returns
        -------
        qdk_chemistry.data.Wavefunction
            The localized molecular wavefunction

        Raises
        ------
        ValueError
            If orbital indices are invalid or inconsistent
        RuntimeError
            If localization fails due to numerical issues
        )",
                py::arg("wavefunction"), py::arg("loc_indices_a"),
                py::arg("loc_indices_b"));

  localizer.def("settings", &Localizer::settings,
                R"(
        Access the localizer's configuration settings.

        Returns
        -------
        qdk_chemistry.data.Settings
            Reference to the settings object for configuring the localizer
        )",
                py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  localizer.def_property(
      "_settings",
      [](LocalizerBase &loc) -> Settings & { return loc.settings(); },
      [](LocalizerBase &loc,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        loc.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
        Internal settings object property.

        This property allows derived classes to replace the settings object with
        a specialized Settings subclass in their constructors.

        Examples
        --------
        >>> class MyLocalizer(alg.Localizer):
        ...     def __init__(self):
        ...         super().__init__()
        ...         from qdk_chemistry.data import ElectronicStructureSettings
        ...         self._settings = ElectronicStructureSettings()
        )");

  localizer.def("type_name", &Localizer::type_name,
                R"(
        The algorithm's type name.

        Returns
        -------
        str
            The type name of the algorithm
        )");

  localizer.def("__repr__", [](const Localizer &) {
    return "<qdk_chemistry.algorithms.Localizer>";
  });

  // Factory class binding - creates LocalizerFactory class
  // with static methods
  qdk::chemistry::python::bind_algorithm_factory<LocalizerFactory, Localizer,
                                                 LocalizerBase>(
      m, "LocalizerFactory");

  localizer.def("__repr__", [](const Localizer &) {
    return "<qdk_chemistry.algorithms.Localizer>";
  });
}
