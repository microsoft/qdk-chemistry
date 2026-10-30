// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/hamiltonian.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/cholesky_hamiltonian.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Trampoline class for enabling Python inheritance
class HamiltonianConstructorBase : public HamiltonianConstructor {
 public:
  NB_TRAMPOLINE(HamiltonianConstructor);
  std::string name() const override { NB_OVERRIDE_PURE(name); }

  std::vector<std::string> aliases() const override { NB_OVERRIDE(aliases); }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Orbitals> orbitals) const override {
    NB_OVERRIDE_PURE(_run_impl, orbitals);
  }
};

void bind_hamiltonian_constructor(nb::module_ &m) {
  // HamiltonianConstructor abstract base class
  nb::class_<HamiltonianConstructor, HamiltonianConstructorBase>
      hamiltonian_constructor(m, "HamiltonianConstructor", R"(
Abstract base class for Hamiltonian constructors.

This class defines the interface for constructing Hamiltonian matrices from orbital data.
Concrete implementations should inherit from this class and implement the construct method.

Examples:
    To create a custom Hamiltonian constructor, inherit from this class::

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

  hamiltonian_constructor.def(nb::init<>(),
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
                              nb::arg("orbitals"));

  hamiltonian_constructor.def("settings", &HamiltonianConstructor::settings,
                              R"(
Access the constructor's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the constructor

)",
                              nb::rv_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  hamiltonian_constructor.def_prop_rw(
      "_settings",
      [](HamiltonianConstructor &constr) -> Settings & {
        return constr.settings();
      },
      [](HamiltonianConstructor &constr,
         std::shared_ptr<qdk::chemistry::data::Settings> new_settings) {
        dynamic_cast<HamiltonianConstructorBase &>(constr).replace_settings(
            std::move(new_settings));
      },
      nb::rv_policy::reference_internal,
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

  hamiltonian_constructor.def("name", &HamiltonianConstructor::name,
                              R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  hamiltonian_constructor.def("aliases", &HamiltonianConstructor::aliases,
                              R"(
The algorithm's aliases.

Returns:
    list[str]: All registered names for the algorithm

)");

  hamiltonian_constructor.def("type_name", &HamiltonianConstructor::type_name,
                              R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  hamiltonian_constructor.def("hash", &HamiltonianConstructor::hash,
                              nb::arg("orbitals"));

  // Factory class binding - creates HamiltonianConstructorFactory class with
  // static methods
  qdk::chemistry::python::bind_algorithm_factory<HamiltonianConstructorFactory,
                                                 HamiltonianConstructor,
                                                 HamiltonianConstructorBase>(
      m, "HamiltonianConstructorFactory");

  hamiltonian_constructor.def("__repr__", [](const HamiltonianConstructor &) {
    return "<qdk_chemistry.algorithms.HamiltonianConstructor>";
  });

  qdk::chemistry::python::bind_create_nested(hamiltonian_constructor);

  // Bind concrete microsoft::HamiltonianConstructor implementation
  nb::class_<microsoft::HamiltonianConstructor, HamiltonianConstructor>(
      m, "QdkHamiltonianConstructor", R"(
QDK implementation of the Hamiltonian constructor.

This class provides a concrete implementation of the Hamiltonian constructor
using the internal backend. It constructs molecular Hamiltonian matrices from
orbital data, computing the necessary one- and two-electron integrals.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg
    import qdk_chemistry.data as data

    # Assuming you have orbitals from an SCF calculation
    constructor = alg.QdkHamiltonianConstructor()

    # Configure settings if needed
    constructor.settings().set("eri_method", "direct")

    # Construct Hamiltonian
    hamiltonian = constructor.run(orbitals)

See Also:
    :class:`HamiltonianConstructor`
    :class:`qdk_chemistry.data.Orbitals`
    :class:`qdk_chemistry.data.Hamiltonian`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes a Hamiltonian constructor with default settings.

)");

  // Bind concrete microsoft::CholeskyHamiltonianConstructor implementation
  nb::class_<microsoft::CholeskyHamiltonianConstructor, HamiltonianConstructor>(
      m, "QdkCholeskyHamiltonianConstructor", R"(
QDK implementation of the Cholesky Hamiltonian constructor.

This class provides a concrete implementation of the Hamiltonian constructor
using Cholesky decomposition to approximate two-electron integrals. It
efficiently constructs molecular Hamiltonian matrices from orbital data by
decomposing the integral tensor.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg
    import qdk_chemistry.data as data

    # Assuming you have orbitals from an SCF calculation
    constructor = alg.QdkCholeskyHamiltonianConstructor()

    # Configure settings if needed
    constructor.settings().set("cholesky_tolerance", 1e-8)

    # Construct Hamiltonian
    hamiltonian = constructor.run(orbitals)

See Also:
    :class:`HamiltonianConstructor`
    :class:`qdk_chemistry.data.Orbitals`
    :class:`qdk_chemistry.data.Hamiltonian`

)")
      .def(nb::init<>(), R"(
Default constructor.

Initializes a Cholesky Hamiltonian constructor with default settings.

)");
}
