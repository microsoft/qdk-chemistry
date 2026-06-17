// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/ctf12_hamiltonian.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Trampoline class for enabling Python inheritance
class EffectiveHamiltonianConstructorBase
    : public EffectiveHamiltonianConstructor,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, EffectiveHamiltonianConstructor, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, EffectiveHamiltonianConstructor,
                      aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Wavefunction> reference) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Hamiltonian>,
                           EffectiveHamiltonianConstructor, _run_impl,
                           reference);
  }
};

void bind_effective_hamiltonian_constructor(py::module &m) {
  py::class_<EffectiveHamiltonianConstructor,
             EffectiveHamiltonianConstructorBase, py::smart_holder>
      effective_hamiltonian_constructor(m, "EffectiveHamiltonianConstructor",
                                        R"(
Abstract base class for effective Hamiltonian constructors.

This class defines the interface for constructing dressed Hamiltonian operators
from a reference wavefunction, which supplies both the orbital basis and the
reduced density matrices consumed by density-driven similarity transformations.
Concrete implementations should inherit from this class and implement the
``_run_impl`` method.

Examples:
    To create a custom effective Hamiltonian constructor, inherit from this class::

        >>> import qdk_chemistry.algorithms as alg
        >>> import qdk_chemistry.data as data
        >>> class MyEffectiveHamiltonianConstructor(alg.EffectiveHamiltonianConstructor):
        ...     def __init__(self):
        ...         super().__init__()  # Call the base class constructor
        ...     def _run_impl(self, reference: data.Wavefunction) -> data.Hamiltonian:
        ...         # Custom effective Hamiltonian construction implementation
        ...         return hamiltonian

)");

  effective_hamiltonian_constructor.def(py::init<>(),
                                        R"(
Create an ``EffectiveHamiltonianConstructor`` instance.

Default constructor for the abstract base class. This should typically be called
from derived class constructors.

)");

  effective_hamiltonian_constructor.def("run",
                                        &EffectiveHamiltonianConstructor::run,
                                        R"(
Construct an effective Hamiltonian from the given reference wavefunction.

This method automatically locks settings before execution to prevent
modifications during construction.

Args:
    reference (qdk_chemistry.data.Wavefunction): The reference wavefunction supplying orbitals and reduced density matrices

Returns:
    qdk_chemistry.data.Hamiltonian: The constructed effective Hamiltonian

Raises:
    SettingsAreLocked: If attempting to modify settings after run() is called

)",
                                        py::arg("reference"));

  effective_hamiltonian_constructor.def(
      "settings", &EffectiveHamiltonianConstructor::settings,
      R"(
Access the constructor's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the constructor

)",
      py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  effective_hamiltonian_constructor.def_property(
      "_settings",
      [](EffectiveHamiltonianConstructorBase &constr) -> Settings & {
        return constr.settings();
      },
      [](EffectiveHamiltonianConstructorBase &constr,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        constr.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a
specialized Settings subclass in their constructors.

)");

  effective_hamiltonian_constructor.def("name",
                                        &EffectiveHamiltonianConstructor::name,
                                        R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  effective_hamiltonian_constructor.def(
      "type_name", &EffectiveHamiltonianConstructor::type_name,
      R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  effective_hamiltonian_constructor.def(
      "hash", &EffectiveHamiltonianConstructor::hash, py::arg("reference"));

  // Factory class binding
  qdk::chemistry::python::bind_algorithm_factory<
      EffectiveHamiltonianConstructorFactory, EffectiveHamiltonianConstructor,
      EffectiveHamiltonianConstructorBase>(
      m, "EffectiveHamiltonianConstructorFactory");

  effective_hamiltonian_constructor.def(
      "__repr__", [](const EffectiveHamiltonianConstructor &) {
        return "<qdk_chemistry.algorithms.EffectiveHamiltonianConstructor>";
      });

  qdk::chemistry::python::bind_create_nested(effective_hamiltonian_constructor);

  // Bind concrete microsoft::CtF12HamiltonianConstructor implementation
  py::class_<microsoft::CtF12HamiltonianConstructor,
             EffectiveHamiltonianConstructor, py::smart_holder>(
      m, "QdkCtF12HamiltonianConstructor", R"(
QDK implementation of the canonical transcorrelated F12 (CT-F12) effective
Hamiltonian constructor.

This class constructs an a priori, Hermitian, two-body effective Hamiltonian by
an approximate canonical (unitary) similarity transformation of the molecular
Hamiltonian with a fixed-amplitude Slater-type geminal. The reduced density
matrices that close the cumulant reduction are read from the reference
wavefunction, so a single-determinant reference yields the single-reference
flavor while a multi-determinant reference yields the multireference flavor
through the same code path.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    constructor = alg.QdkCtF12HamiltonianConstructor()
    constructor.settings().set("gamma", 1.5)
    h_bar = constructor.run(reference_wavefunction)

See Also:
    :class:`EffectiveHamiltonianConstructor`
    :class:`qdk_chemistry.data.Wavefunction`
    :class:`qdk_chemistry.data.Hamiltonian`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a CT-F12 effective Hamiltonian constructor with default settings.

)");
}
