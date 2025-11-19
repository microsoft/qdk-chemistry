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
class ActiveSpaceSelectorBase : public ActiveSpaceSelector,
                                public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, ActiveSpaceSelector, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, ActiveSpaceSelector, aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Wavefunction> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Wavefunction>, ActiveSpaceSelector,
                           _run_impl, wavefunction);
  }
};

void bind_active_space(py::module& m) {
  // ActiveSpaceSelector abstract base class
  py::class_<ActiveSpaceSelector, ActiveSpaceSelectorBase, py::smart_holder>
      selector(m, "ActiveSpaceSelector", R"(
    Abstract base class for active space selector algorithms.

    This class defines the interface for selecting active spaces from a set of orbitals.
    Active space selection is a critical step in many quantum chemistry methods,
    particularly for multireference calculations. Concrete implementations should
    inherit from this class and implement the ``select_active_space`` method.

    Return value semantics
    ----------------------
    Implementations return a new ``Wavefunction`` object with active-space data populated.
    Some selectors (e.g., occupation/valence) return a copy with only metadata
    updated. Others (e.g., AVAS) may rotate/canonicalize orbitals and recompute
    occupations, so the returned coefficients/occupations can differ from the input.
    The input ``Wavefunction`` object is never modified.

    Examples
    --------
    To create a custom active space selector, inherit from this class:

    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyActiveSpaceSelector(alg.ActiveSpaceSelector):
    ...     def __init__(self):
    ...         super().__init__()  # Call the base class constructor
    ...     # Implement the _run_impl method (called by run())
    ...     def _run_impl(self, wavefunction: data.Wavefunction) -> data.Wavefunction:
    ...         # Custom active space selection implementation that returns orbitals with active space populated.
    ...         # For selectors that only annotate, return a copied object with metadata set; for others,
    ...         # you may also rotate/transform the coefficients as needed.
    ...         act_orbitals = data.Orbitals(wavefunction.get_orbitals()) # Copy base data by default
    ...         act_orbitals.set_active_space_indices([0, 1, 2, 3])
    ...         act_orbitals.set_num_active_electrons(2)
    ...         ... # Additional logic to modify orbitals if needed
    ...         act_wavefunction = data.Wavefunction(...)
    ...         return act_wavefunction # Return modified wavefunction object
    )");

  selector.def(py::init<>(),
               R"(
        Create an ActiveSpaceSelector instance.

        Default constructor for the abstract base class.
        This should typically be called from derived class constructors.

        Examples
        --------
        >>> # In a derived class:
        >>> class MySelector(alg.ActiveSpaceSelector):
        ...     def __init__(self):
        ...         super().__init__()  # Calls parent constructor
        )");

  selector.def("run", &ActiveSpaceSelector::run, py::arg("wavefunction"),
               R"(
               Select active space orbitals from the given wavefunction.

               This method automatically locks settings before execution to prevent
               modifications during selection.

               Parameters
               ----------
               wavefunction : qdk_chemistry.data.Wavefunction
                   The wavefunction from which to select the active space

               Returns
               -------
               qdk_chemistry.data.Wavefunction
                   Wavefunction with active space data populated

               Raises
               ------
               SettingsAreLocked
                   If attempting to modify settings after run() is called
               )");

  selector.def("settings", &ActiveSpaceSelector::settings,
               py::return_value_policy::reference,
               R"(
        Access the selector's configuration settings.

        Returns
        -------
        qdk_chemistry.data.Settings
            Reference to the settings object for configuring the active space selector
        )");

  // Expose _settings as a writable property for derived classes
  selector.def_property(
      "_settings",
      [](ActiveSpaceSelectorBase& sel) -> Settings& { return sel.settings(); },
      [](ActiveSpaceSelectorBase& sel,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        sel.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
        Internal settings object property.

        This property allows derived classes to replace the settings object with
        a specialized Settings subclass in their constructors.

        Examples
        --------
        >>> class MySelector(alg.ActiveSpaceSelector):
        ...     def __init__(self):
        ...         super().__init__()
        ...         from qdk_chemistry.data import ElectronicStructureSettings
        ...         self._settings = ElectronicStructureSettings()
        )");

  selector.def("type_name", &ActiveSpaceSelector::type_name,
               R"(
        The algorithm's type name.

        Returns
        -------
        str
            The type name of the algorithm
        )");

  // Factory class binding - creates ActiveSpaceSelectorFactory class
  // with static methods
  qdk::chemistry::python::bind_algorithm_factory<
      ActiveSpaceSelectorFactory, ActiveSpaceSelector, ActiveSpaceSelectorBase>(
      m, "ActiveSpaceSelectorFactory");

  selector.def("__repr__", [](const ActiveSpaceSelector&) {
    return "<qdk_chemistry.algorithms.ActiveSpaceSelector>";
  });
}
