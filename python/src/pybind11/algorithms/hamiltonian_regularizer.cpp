// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/flr_bliss/flr_bliss_regularizer.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

using ReturnType = std::shared_ptr<Hamiltonian>;

// Trampoline class for enabling Python inheritance
class HamiltonianRegularizerBase
    : public HamiltonianRegularizer,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, HamiltonianRegularizer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, HamiltonianRegularizer,
                      aliases);
  }

  // Helper method to expose _settings for Python binding
  void replace_settings(
      std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

 protected:
  ReturnType _run_impl(std::shared_ptr<Hamiltonian> hamiltonian,
                       unsigned int n_alpha_electrons,
                       unsigned int n_beta_electrons) const override {
    PYBIND11_OVERRIDE_PURE(ReturnType, HamiltonianRegularizer, _run_impl,
                           hamiltonian, n_alpha_electrons, n_beta_electrons);
  }
};

void bind_hamiltonian_regularizer(py::module &m) {
  py::class_<HamiltonianRegularizer, HamiltonianRegularizerBase,
             py::smart_holder>
      regularizer(m, "HamiltonianRegularizer",
                 R"(
Abstract base class for Hamiltonian regularization/shift algorithms.

A HamiltonianRegularizer maps a Hamiltonian, together with the target
number of alpha/beta electrons, to a new Hamiltonian that is energetically
equivalent within the target electron-number sector but whose LCU/qubitization
coefficients (e.g. the fermionic 1-norm lambda) may be reduced.

Examples:
    >>> # To create a custom regularizer, inherit from this class.
    >>> import qdk_chemistry.algorithms as alg
    >>> import qdk_chemistry.data as data
    >>> class MyHamiltonianRegularizer(alg.HamiltonianRegularizer):
    ...     def __init__(self):
    ...         super().__init__()
    ...     def _run_impl(self, hamiltonian: data.Hamiltonian, n_alpha_electrons: int, n_beta_electrons: int) -> data.Hamiltonian:
    ...         # Custom regularization implementation
    ...         return shifted_hamiltonian

)");

  regularizer.def(py::init<>(), R"(
Create a HamiltonianRegularizer instance.

Default constructor for the abstract base class.
This should typically be called from derived class constructors.

)");

  regularizer.def("run", &HamiltonianRegularizer::run,
                  R"(
Regularize/shift a Hamiltonian for a target electron count.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian to regularize
    n_alpha_electrons (int): The target number of alpha electrons
    n_beta_electrons (int): The target number of beta electrons

Returns:
    qdk_chemistry.data.Hamiltonian: A new, shifted Hamiltonian that agrees
    with the input Hamiltonian's energy in the (n_alpha_electrons,
    n_beta_electrons)-electron sector.

Raises:
    SettingsAreLocked: If attempting to modify settings after run() is called

)",
                  py::arg("hamiltonian"), py::arg("n_alpha_electrons"),
                  py::arg("n_beta_electrons"));

  regularizer.def("settings", &HamiltonianRegularizer::settings,
                  R"(
Access the regularizer's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the regularizer

)",
                  py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  regularizer.def_property(
      "_settings",
      [](HamiltonianRegularizerBase &algo) -> Settings & {
        return algo.settings();
      },
      [](HamiltonianRegularizerBase &algo,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        algo.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

)");

  regularizer.def("name", &HamiltonianRegularizer::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  regularizer.def("type_name", &HamiltonianRegularizer::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  // Factory class binding - creates HamiltonianRegularizerFactory class with
  // static methods
  qdk::chemistry::python::bind_algorithm_factory<
      HamiltonianRegularizerFactory, HamiltonianRegularizer,
      HamiltonianRegularizerBase>(m, "HamiltonianRegularizerFactory");

  regularizer.def("__repr__", [](const HamiltonianRegularizer &) {
    return "<qdk_chemistry.algorithms.HamiltonianRegularizer>";
  });

  qdk::chemistry::python::bind_create_nested(regularizer);

  // Bind concrete microsoft::FlrBlissRegularizer implementation
  py::class_<microsoft::FlrBlissRegularizer, HamiltonianRegularizer,
             py::smart_holder>(m, "QdkFlrBlissRegularizer", R"(
QDK FLR-BLISS Hamiltonian regularizer.

This class provides a concrete implementation of the HamiltonianRegularizer
interface using the block-invariant symmetry shift (BLISS) technique applied
to the fermionic double-factorized representation of a Hamiltonian
(Patel et al., arXiv:2409.18277). It reduces the Hamiltonian's fermionic
1-norm while leaving its energy invariant within a target
(n_alpha, n_beta)-electron sector.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    regularizer = alg.QdkFlrBlissRegularizer()

    # Optionally opt into DF truncation (default is 0.0, i.e. no truncation)
    regularizer.settings().set("df_truncation_threshold", 1e-8)

    shifted_hamiltonian = regularizer.run(hamiltonian, n_alpha, n_beta)

See Also:
    :class:`HamiltonianRegularizer`
    :class:`qdk_chemistry.data.Hamiltonian`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a FLR-BLISS regularizer with default settings
(df_truncation_threshold = 0.0, i.e. no truncation).

)");
}
