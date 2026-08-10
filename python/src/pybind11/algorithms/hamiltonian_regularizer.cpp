// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
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
    : public BlissRegularizer,
      public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE(std::string, BlissRegularizer, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, BlissRegularizer, aliases);
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
    PYBIND11_OVERRIDE(ReturnType, BlissRegularizer, _run_impl, hamiltonian,
                      n_alpha_electrons, n_beta_electrons);
  }
};

void bind_hamiltonian_regularizer(py::module &m) {
  // BlissShift: the (mu1, mu2, xi) shift parameters, decoupled from how they
  // were produced so they can be inspected or supplied from any source.
  py::class_<BlissShift>(m, "BlissShift", R"(
Block-invariant symmetry shift (BLISS) parameters.

Bundles the three quantities (mu1, mu2, xi) that define the BLISS operator
subtracted from a Hamiltonian to reduce its fermionic 1-norm while leaving the
target electron-number sector's energy invariant. A BlissShift carries only the
*result* of a shift computation, so it can come from
:meth:`HamiltonianRegularizer.compute_shift` or from an external source and be
applied via :func:`rebuild_bliss_shifted_hamiltonian`.
)")
      .def(py::init<>())
      .def_readwrite("mu1", &BlissShift::mu1, "One-electron BLISS shift.")
      .def_readwrite("mu2", &BlissShift::mu2, "Two-electron BLISS shift.")
      .def_readwrite("xi", &BlissShift::xi,
                     "Two-electron BLISS shift matrix (norb x norb).")
      .def("__repr__", [](const BlissShift &s) {
        return "<qdk_chemistry.algorithms.BlissShift mu1=" +
               std::to_string(s.mu1) + " mu2=" + std::to_string(s.mu2) +
               " xi=" + std::to_string(s.xi.rows()) + "x" +
               std::to_string(s.xi.cols()) + ">";
      });

  // Module-level rebuild_bliss_shifted_hamiltonian: apply a BlissShift to a
  // Hamiltonian.
  m.def("rebuild_bliss_shifted_hamiltonian", &rebuild_bliss_shifted_hamiltonian,
        py::arg("original"), py::arg("shift"), py::arg("num_electrons"), R"(
Apply a BLISS shift to a Hamiltonian and assemble the shifted Hamiltonian.

Applies the shift parameters (mu1, mu2, xi) in ``shift`` to the dense one- and
two-electron integrals of ``original``. Because the underlying BLISS operator
annihilates every ``num_electrons``-electron state, the energy of that sector
is left invariant. The shift may come from
:meth:`HamiltonianRegularizer.compute_shift` or from any external source.

Args:
    original (qdk_chemistry.data.Hamiltonian): The Hamiltonian to shift. Must be restricted.
    shift (qdk_chemistry.algorithms.BlissShift): The BLISS shift parameters to apply.
    num_electrons (int): Target number of active electrons (Ne). Must be a
        non-negative integer; the invariance guarantee only holds for an
        integer electron count.

Returns:
    qdk_chemistry.data.Hamiltonian: The BLISS-shifted Hamiltonian.

Raises:
    ValueError: If ``original`` is unrestricted, ``shift.xi`` is not norb x norb,
        or ``num_electrons`` is negative or non-integer.
)");

  py::class_<BlissRegularizer, HamiltonianRegularizerBase, py::smart_holder>
      regularizer(m, "HamiltonianRegularizer",
                  R"(
Hamiltonian regularizer implementing block-invariant symmetry shifts (BLISS).

A HamiltonianRegularizer maps a Hamiltonian, together with the target
number of alpha/beta electrons, to a new Hamiltonian that is energetically
equivalent within the target electron-number sector but whose LCU/qubitization
coefficients (e.g. the fermionic 1-norm lambda) may be reduced.

It is a thin composition of two public steps: :meth:`compute_shift` computes
the BLISS parameters (mu1, mu2, xi) via the method selected by the
``shift_method`` setting (default ``"flr_bliss"``), and
:func:`rebuild_bliss_shifted_hamiltonian` applies a shift to a Hamiltonian. Callers can
obtain a :class:`BlissShift` on its own, or supply an externally computed one to
:func:`rebuild_bliss_shifted_hamiltonian` directly.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> regularizer = alg.HamiltonianRegularizer()
    >>> shift = regularizer.compute_shift(hamiltonian, n_alpha, n_beta)
    >>> shifted = alg.rebuild_bliss_shifted_hamiltonian(hamiltonian, shift, n_alpha + n_beta)

)");

  regularizer.def(py::init<>(), R"(
Create a HamiltonianRegularizer instance with default settings
(shift_method = "flr_bliss", df_truncation_threshold = 0.0).

)");

  regularizer.def("run", &BlissRegularizer::run,
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

  regularizer.def("compute_shift", &BlissRegularizer::compute_shift,
                  R"(
Compute the BLISS shift (mu1, mu2, xi) for a target electron count.

Dispatches to the method selected by the ``shift_method`` setting and returns
the resulting parameters *without* rebuilding the Hamiltonian. Use
:func:`rebuild_bliss_shifted_hamiltonian` to apply the returned (or an externally sourced)
:class:`BlissShift`.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian to analyze. Must be restricted.
    n_alpha_electrons (int): The target number of alpha electrons
    n_beta_electrons (int): The target number of beta electrons

Returns:
    qdk_chemistry.algorithms.BlissShift: The computed shift parameters.

Raises:
    ValueError: If the Hamiltonian is unrestricted or the configured
        ``shift_method`` is unknown.

)",
                  py::arg("hamiltonian"), py::arg("n_alpha_electrons"),
                  py::arg("n_beta_electrons"));

  regularizer.def("settings", &BlissRegularizer::settings,
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

  regularizer.def("name", &BlissRegularizer::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  regularizer.def("type_name", &BlissRegularizer::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  // Factory class binding - creates HamiltonianRegularizerFactory class with
  // static methods
  qdk::chemistry::python::bind_algorithm_factory<HamiltonianRegularizerFactory,
                                                 BlissRegularizer,
                                                 HamiltonianRegularizerBase>(
      m, "HamiltonianRegularizerFactory");

  regularizer.def("__repr__", [](const BlissRegularizer &) {
    return "<qdk_chemistry.algorithms.HamiltonianRegularizer>";
  });

  qdk::chemistry::python::bind_create_nested(regularizer);
}
