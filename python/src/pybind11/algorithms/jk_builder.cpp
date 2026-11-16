// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/jk_builder.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/algorithms/jk_builder.hpp>
#include <string>
#include <vector>

#include "factory_bindings.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

using JKReturnType = std::pair<std::shared_ptr<const SymmetryBlockedTensor<2>>,
                               std::shared_ptr<const SymmetryBlockedTensor<2>>>;
using JKGradientReturnType = std::pair<std::shared_ptr<NuclearGradients>,
                                       std::shared_ptr<NuclearGradients>>;

class JKBuilderBase : public JKBuilder,
                      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, JKBuilder, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, JKBuilder, aliases);
  }

  void replace_settings(std::unique_ptr<Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  JKReturnType _run_impl(std::shared_ptr<BasisSet> basis_set,
                         std::shared_ptr<const SymmetryBlockedTensor<2>>
                             densities) const override {
    PYBIND11_OVERRIDE_PURE(JKReturnType, JKBuilder, _run_impl, basis_set,
                           densities);
  }
};

class JKGradientBuilderBase : public JKGradientBuilder,
                              public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, JKGradientBuilder, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, JKGradientBuilder, aliases);
  }

  void replace_settings(std::unique_ptr<Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  JKGradientReturnType _run_impl(std::shared_ptr<BasisSet> basis_set,
                                 std::shared_ptr<const SymmetryBlockedTensor<2>>
                                     densities) const override {
    PYBIND11_OVERRIDE_PURE(JKGradientReturnType, JKGradientBuilder, _run_impl,
                           basis_set, densities);
  }
};

void bind_jk_builder(py::module& m) {
  py::class_<JKBuilder, JKBuilderBase, py::smart_holder> builder(m, "JKBuilder",
                                                                 R"(
Abstract base class for J/K builders.

A J/K builder builds the Coulomb matrices :math:`J[P_s]` and exchange matrices
:math:`K[P_s]` of spin densities :math:`P_s` with the exchange operator
:math:`v = \alpha / r + \beta \operatorname{erfc}(\omega r) / r` of its
``alpha``, ``beta``, and ``omega`` settings. The ``qdk`` SCF solver builds its
J and K matrices with the builder named by its ``jk_builder`` setting.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> from qdk_chemistry.data.symmetry import make_spin_diagonal_rank2_sbt
    >>> coulomb, exchange = alg.create("jk_builder").run(basis_set, make_spin_diagonal_rank2_sbt(density))
)");

  builder.def(py::init<>(), R"(
Create a ``JKBuilder`` instance.

Default constructor for the abstract base class; typically called via
``super().__init__()`` from a derived class.
)");
  builder.def("run", &JKBuilder::run, py::arg("basis_set"),
              py::arg("densities"), R"(
Build the Coulomb and exchange matrices of spin densities.

Args:
    basis_set (qdk_chemistry.data.BasisSet): Basis set whose atomic orbitals, in their order and with each contracted shell normalized, index the matrices.
    densities (qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2): Symmetric spin densities, the alpha-alpha and beta-beta blocks of a spin-diagonal tensor.

Returns:
    tuple[qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2, qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2]: J and K of each spin density, in the spin structure of ``densities``.
)");
  builder.def("settings", &JKBuilder::settings,
              py::return_value_policy::reference_internal, R"(
Access the builder's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object.
)");
  builder.def_property(
      "_settings",
      [](JKBuilderBase& instance) -> Settings& { return instance.settings(); },
      [](JKBuilderBase& instance, std::unique_ptr<Settings> new_settings) {
        instance.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal, R"(
Internal settings object property.

Allows derived classes to replace the settings object with a specialized
Settings subclass in their constructor.
)");
  builder.def("name", &JKBuilder::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm
)");
  builder.def("aliases", &JKBuilder::aliases, R"(
The algorithm's aliases.

Returns:
    list[str]: All registered names for the algorithm
)");
  builder.def("type_name", &JKBuilder::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");
  builder.def("hash", &JKBuilder::hash, py::arg("basis_set"),
              py::arg("densities"), R"(
Compute a deterministic content hash for a run with these inputs.

Args:
    basis_set (qdk_chemistry.data.BasisSet): Basis set.
    densities (qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2): Spin densities.

Returns:
    str: 16-character hex content hash.
)");
  builder.def("__repr__", [](const JKBuilder&) {
    return "<qdk_chemistry.algorithms.JKBuilder>";
  });

  qdk::chemistry::python::bind_create_nested(builder);

  py::class_<microsoft::JKBuilder, JKBuilder, py::smart_holder>(
      m, "QdkJKBuilder", R"(
QDK implementation of the J/K builder.

Evaluates the integrals with Libint2, directly or in memory as selected by the
``eri_method`` setting, and keeps the integral data of the last basis set for
later runs.
)")
      .def(py::init<>(), R"(
Default constructor.
)");

  qdk::chemistry::python::bind_algorithm_factory<JKBuilderFactory, JKBuilder,
                                                 JKBuilderBase>(
      m, "JKBuilderFactory");

  py::class_<JKGradientBuilder, JKGradientBuilderBase, py::smart_holder>
      gradient_builder(m, "JKGradientBuilder", R"(
Abstract base class for J/K gradient builders.

A J/K gradient builder builds the nuclear gradients, at fixed spin densities
:math:`P_s`, of the Coulomb energy
:math:`E_J = \tfrac{1}{2} \operatorname{Tr}(P J[P])` with
:math:`P = P_\alpha + P_\beta` and of the exchange energy
:math:`E_K = -\tfrac{1}{2} \sum_s \operatorname{Tr}(P_s K[P_s])`, with
:math:`J` and :math:`K` as in :class:`JKBuilder`. The ``qdk`` SCF solver
builds its analytic gradients with the builder named by its
``jk_gradient_builder`` setting.
)");

  gradient_builder.def(py::init<>(), R"(
Create a ``JKGradientBuilder`` instance.

Default constructor for the abstract base class; typically called via
``super().__init__()`` from a derived class.
)");
  gradient_builder.def("run", &JKGradientBuilder::run, py::arg("basis_set"),
                       py::arg("densities"), R"(
Build the nuclear gradients of the Coulomb and exchange energies.

Args:
    basis_set (qdk_chemistry.data.BasisSet): Basis set, as in :meth:`JKBuilder.run`.
    densities (qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2): Spin densities, as in :meth:`JKBuilder.run`.

Returns:
    tuple[qdk_chemistry.data.NuclearGradients, qdk_chemistry.data.NuclearGradients]: The gradients of :math:`E_J` and :math:`E_K` on the basis set's structure.
)");
  gradient_builder.def("settings", &JKGradientBuilder::settings,
                       py::return_value_policy::reference_internal, R"(
Access the builder's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object.
)");
  gradient_builder.def_property(
      "_settings",
      [](JKGradientBuilderBase& instance) -> Settings& {
        return instance.settings();
      },
      [](JKGradientBuilderBase& instance,
         std::unique_ptr<Settings> new_settings) {
        instance.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal, R"(
Internal settings object property.

Allows derived classes to replace the settings object with a specialized
Settings subclass in their constructor.
)");
  gradient_builder.def("name", &JKGradientBuilder::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm
)");
  gradient_builder.def("aliases", &JKGradientBuilder::aliases, R"(
The algorithm's aliases.

Returns:
    list[str]: All registered names for the algorithm
)");
  gradient_builder.def("type_name", &JKGradientBuilder::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");
  gradient_builder.def("hash", &JKGradientBuilder::hash, py::arg("basis_set"),
                       py::arg("densities"), R"(
Compute a deterministic content hash for a run with these inputs.

Args:
    basis_set (qdk_chemistry.data.BasisSet): Basis set.
    densities (qdk_chemistry.data.symmetry.SymmetryBlockedTensorRank2): Spin densities.

Returns:
    str: 16-character hex content hash.
)");
  gradient_builder.def("__repr__", [](const JKGradientBuilder&) {
    return "<qdk_chemistry.algorithms.JKGradientBuilder>";
  });

  qdk::chemistry::python::bind_create_nested(gradient_builder);

  py::class_<microsoft::JKGradientBuilder, JKGradientBuilder, py::smart_holder>(
      m, "QdkJKGradientBuilder", R"(
QDK implementation of the J/K gradient builder.

Evaluates the integral derivatives directly with Libint2 and keeps the integral
screening data of the last basis set for later runs.
)")
      .def(py::init<>(), R"(
Default constructor.
)");

  qdk::chemistry::python::bind_algorithm_factory<
      JKGradientBuilderFactory, JKGradientBuilder, JKGradientBuilderBase>(
      m, "JKGradientBuilderFactory");
}
