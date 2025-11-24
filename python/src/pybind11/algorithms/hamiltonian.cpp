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
class HamiltonianConstructorBase
    : public HamiltonianConstructor,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, HamiltonianConstructor, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, HamiltonianConstructor,
                      aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Orbitals> orbitals) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Hamiltonian>, HamiltonianConstructor,
                           _run_impl, orbitals);
  }
};

void bind_hamiltonian_constructor(py::module &m) {
  // HamiltonianConstructor abstract base class
  py::class_<HamiltonianConstructor, HamiltonianConstructorBase,
             py::smart_holder>
      hamiltonian_constructor(m, "HamiltonianConstructor", R"(
Abstract base class for Hamiltonian constructors.

This class defines the interface for constructing Hamiltonian matrices from orbital data.
Concrete implementations should inherit from this class and implement the construct method.

Examples:
    To create a custom Hamiltonian constructor, inherit from this class:

        >>> import qdk_chemistry.algorithms as alg
        >>> import qdk_chemistry.data as data
        >>> class MyHamiltonianConstructor(alg.HamiltonianConstructor):
        ...     def __init__(self):
        ...         super().__init__()  # Call the base class constructor
        ...     # Implement the _run_impl method
        ...     def _run_impl(self, orbitals: data.Orbitals) -> data.Hamiltonian:
        ...         # Custom Hamiltonian construction implementation
        ...         return hamiltonian
)");

  hamiltonian_constructor.def(py::init<>(),
                              R"(
Create a ``HamiltonianConstructor`` instance.

Default constructor for the abstract base class.
This should typically be called from derived class constructors.

Examples:
    >>> # In a derived class:
    >>> class MyConstructor(alg.HamiltonianConstructor):
    ...     def __init__(self):
    ...         super().__init__()  # Calls this constructor
)");

  hamiltonian_constructor.def("run", &HamiltonianConstructor::run,
                              R"(
Construct a Hamiltonian from the given orbitals.

This method automatically locks settings before execution to prevent
modifications during construction.

Args:
    orbitals (qdk_chemistry.data.Orbitals): The orbital data to construct the Hamiltonian from

Returns:
    qdk_chemistry.data.Hamiltonian: The constructed Hamiltonian matrix

Raises:
    SettingsAreLocked: If attempting to modify settings after run() is called
)",
                              py::arg("orbitals"));

  hamiltonian_constructor.def("settings", &HamiltonianConstructor::settings,
                              R"(
Access the constructor's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the constructor
)",
                              py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  hamiltonian_constructor.def_property(
      "_settings",
      [](HamiltonianConstructorBase &constr) -> Settings & {
        return constr.settings();
      },
      [](HamiltonianConstructorBase &constr,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        constr.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

Examples:
    >>> class MyConstructor(alg.HamiltonianConstructor):
    ...     def __init__(self):
    ...         super().__init__()
    ...         from qdk_chemistry.data import ElectronicStructureSettings
    ...         self._settings = ElectronicStructureSettings()
)");

  hamiltonian_constructor.def("type_name", &HamiltonianConstructor::type_name,
                              R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");

  // Factory class binding - creates HamiltonianConstructorFactory class with
  // static methods
  qdk::chemistry::python::bind_algorithm_factory<HamiltonianConstructorFactory,
                                                 HamiltonianConstructor,
                                                 HamiltonianConstructorBase>(
      m, "HamiltonianConstructorFactory");

  hamiltonian_constructor.def("__repr__", [](const HamiltonianConstructor &) {
    return "<qdk_chemistry.algorithms.HamiltonianConstructor>";
  });
}
