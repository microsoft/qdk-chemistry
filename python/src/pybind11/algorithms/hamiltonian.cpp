// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/hamiltonian.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/cholesky_hamiltonian.hpp"
#include "qdk/chemistry/algorithms/microsoft/hamiltonian_basis_transformer.hpp"

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

class HamiltonianBasisTransformerBase
    : public HamiltonianBasisTransformer,
      public py::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, HamiltonianBasisTransformer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, HamiltonianBasisTransformer,
                      aliases);
  }

  void replace_settings(std::unique_ptr<Settings> settings) {
    _settings = std::move(settings);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian,
      std::shared_ptr<Orbitals> target_orbitals) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Hamiltonian>,
                           HamiltonianBasisTransformer, _run_impl, hamiltonian,
                           target_orbitals);
  }
};

void bind_hamiltonian_algorithms(py::module &m) {
  // HamiltonianConstructor abstract base class
  py::class_<HamiltonianConstructor, HamiltonianConstructorBase,
             py::smart_holder>
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

  hamiltonian_constructor.def("name", &HamiltonianConstructor::name,
                              R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  hamiltonian_constructor.def("type_name", &HamiltonianConstructor::type_name,
                              R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  hamiltonian_constructor.def("hash", &HamiltonianConstructor::hash,
                              py::arg("orbitals"));

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
  py::class_<microsoft::HamiltonianConstructor, HamiltonianConstructor,
             py::smart_holder>(m, "QdkHamiltonianConstructor", R"(
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
      .def(py::init<>(), R"(
Default constructor.

Initializes a Hamiltonian constructor with default settings.

)");

  // Bind concrete microsoft::CholeskyHamiltonianConstructor implementation
  py::class_<microsoft::CholeskyHamiltonianConstructor, HamiltonianConstructor,
             py::smart_holder>(m, "QdkCholeskyHamiltonianConstructor", R"(
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
      .def(py::init<>(), R"(
Default constructor.

Initializes a Cholesky Hamiltonian constructor with default settings.

)");

  py::class_<HamiltonianBasisTransformer, HamiltonianBasisTransformerBase,
             py::smart_holder>
      transformer(m, "HamiltonianBasisTransformer", R"(
Abstract algorithm for expressing a Hamiltonian in a target orbital basis.
)");
  // Keep the GIL for the abstract binding because Python subclasses may
  // implement the virtual call in Python.
  transformer.def(py::init<>())
      .def("run", &HamiltonianBasisTransformer::run, py::arg("hamiltonian"),
           py::arg("target_orbitals"))
      .def("settings", &HamiltonianBasisTransformer::settings,
           py::return_value_policy::reference_internal)
      .def_property(
          "_settings",
          [](HamiltonianBasisTransformerBase &self) -> Settings & {
            return self.settings();
          },
          [](HamiltonianBasisTransformerBase &self,
             std::unique_ptr<Settings> settings) {
            self.replace_settings(std::move(settings));
          },
          py::return_value_policy::reference_internal)
      .def("name", &HamiltonianBasisTransformer::name)
      .def("type_name", &HamiltonianBasisTransformer::type_name)
      .def("hash", &HamiltonianBasisTransformer::hash, py::arg("hamiltonian"),
           py::arg("target_orbitals"))
      .def("__repr__", [](const HamiltonianBasisTransformer &) {
        return "<qdk_chemistry.algorithms.HamiltonianBasisTransformer>";
      });

  qdk::chemistry::python::bind_algorithm_factory<
      HamiltonianBasisTransformerFactory, HamiltonianBasisTransformer,
      HamiltonianBasisTransformerBase>(m, "HamiltonianBasisTransformerFactory");
  qdk::chemistry::python::bind_create_nested(transformer);

  py::class_<microsoft::QdkHamiltonianBasisTransformer,
             HamiltonianBasisTransformer, py::smart_holder>(
      m, "QdkHamiltonianBasisTransformer", R"(
QDK basis transformer for restricted Cholesky Hamiltonians.
)")
      .def(py::init<>())
      .def(
          "run",
          [](microsoft::QdkHamiltonianBasisTransformer &self,
             std::shared_ptr<Hamiltonian> hamiltonian,
             std::shared_ptr<Orbitals> target_orbitals) {
            // Freeze settings while the GIL still serializes Python access,
            // then release it for the synchronized native transformation.
            self.settings().lock();
            py::gil_scoped_release release;
            return self.run(std::move(hamiltonian),
                            std::move(target_orbitals));
          },
          py::arg("hamiltonian"), py::arg("target_orbitals"));
}
