// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/symmetry_shift/fermionic_low_rank.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

using ReturnType = std::shared_ptr<Hamiltonian>;

// Trampoline class for enabling Python inheritance
class SymmetryShifterBase : public SymmetryShifter,
                            public pybind11::trampoline_self_life_support {
 public:
  std::string name() const override {
    PYBIND11_OVERRIDE_PURE(std::string, SymmetryShifter, name);
  }

  std::vector<std::string> aliases() const override {
    PYBIND11_OVERRIDE(std::vector<std::string>, SymmetryShifter, aliases);
  }

  SymmetryShift compute_shift(const Hamiltonian &hamiltonian,
                              unsigned int n_alpha_electrons,
                              unsigned int n_beta_electrons) const override {
    PYBIND11_OVERRIDE_PURE(SymmetryShift, SymmetryShifter, compute_shift,
                           hamiltonian, n_alpha_electrons, n_beta_electrons);
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
    PYBIND11_OVERRIDE_PURE(ReturnType, SymmetryShifter, _run_impl, hamiltonian,
                           n_alpha_electrons, n_beta_electrons);
  }
};

void bind_symmetry_shift(py::module &m) {
  // SymmetryShift: the (mu1, mu2, xi) shift parameters, decoupled from how
  // they were produced so they can be inspected or supplied from any source.
  py::class_<SymmetryShift>(m, "SymmetryShift", R"(
Number-symmetry shift parameters.

Bundles the three quantities (mu1, mu2, xi) that define the symmetry-shift
operator subtracted from a Hamiltonian to reduce its fermionic 1-norm while
leaving the target electron-number sector's energy invariant. A SymmetryShift
carries only the *result* of a shift computation, so it can come from
:meth:`SymmetryShifter.compute_shift` or from an external source and be
applied via :func:`rebuild_shifted_hamiltonian`.
)")
      .def(py::init<>())
      .def_readwrite("mu1", &SymmetryShift::mu1, "One-electron shift.")
      .def_readwrite("mu2", &SymmetryShift::mu2, "Two-electron shift.")
      .def_readwrite("xi", &SymmetryShift::xi,
                     "Two-electron shift matrix (norb x norb).")
      .def("__repr__", [](const SymmetryShift &s) {
        return "<qdk_chemistry.algorithms.SymmetryShift mu1=" +
               std::to_string(s.mu1) + " mu2=" + std::to_string(s.mu2) +
               " xi=" + std::to_string(s.xi.rows()) + "x" +
               std::to_string(s.xi.cols()) + ">";
      });

  // Module-level rebuild_shifted_hamiltonian: apply a SymmetryShift to a
  // Hamiltonian.
  m.def("rebuild_shifted_hamiltonian", &rebuild_shifted_hamiltonian,
        py::arg("original"), py::arg("shift"), py::arg("num_electrons"), R"(
Apply a symmetry shift to a Hamiltonian and assemble the shifted one.

Applies the shift parameters (mu1, mu2, xi) to the dense integrals of
``original``. Because the corresponding operator K annihilates every
``num_electrons``-electron state, the energy of that sector is unchanged.

How the shift was computed is irrelevant: it may come from
:meth:`SymmetryShifter.compute_shift` or from any external source.

Args:
    original (qdk_chemistry.data.Hamiltonian): The Hamiltonian being shifted.
        Must be restricted.
    shift (qdk_chemistry.algorithms.SymmetryShift): The shift parameters
        (mu1, mu2, xi) to apply.
    num_electrons (int): Target number of active electrons (Ne). Must be a
        non-negative integer; the invariance guarantee only holds for an
        integer electron count.

Returns:
    qdk_chemistry.data.Hamiltonian: The shifted Hamiltonian.

Raises:
    ValueError: If ``original`` is unrestricted or ``shift.xi`` is not
        norb x norb.
    TypeError: If ``num_electrons`` is negative or non-integer.

)");

  // SymmetryShifter abstract base class
  py::class_<SymmetryShifter, SymmetryShifterBase, py::smart_holder> shifter(
      m, "SymmetryShifter",
      R"(
Abstract base class for number-symmetry Hamiltonian shift algorithms.

A SymmetryShifter maps a Hamiltonian, together with the target number of
alpha/beta electrons, to a new Hamiltonian that is energetically equivalent
within the target electron-number sector but whose LCU/qubitization
coefficients (e.g. the fermionic 1-norm lambda) may be reduced.

Every implementation is a thin composition of two public steps:
:meth:`compute_shift` computes the parameters (mu1, mu2, xi) -- this is what
distinguishes one implementation from another -- and
:func:`rebuild_shifted_hamiltonian` applies a shift to a Hamiltonian. Callers
can obtain a :class:`SymmetryShift` on its own, or supply an externally
computed one to :func:`rebuild_shifted_hamiltonian` directly.

Concrete implementations should inherit from this class.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> shifter = alg.FermionicLowRankShifter()
    >>> shift = shifter.compute_shift(hamiltonian, n_alpha, n_beta)
    >>> shifted = alg.rebuild_shifted_hamiltonian(hamiltonian, shift, n_alpha + n_beta)

)");

  shifter.def("run", &SymmetryShifter::run,
              R"(
Shift a Hamiltonian for a target electron count.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian to shift
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

  shifter.def("compute_shift", &SymmetryShifter::compute_shift,
              R"(
Compute the symmetry shift (mu1, mu2, xi) for a target electron count.

Returns the resulting parameters *without* rebuilding the Hamiltonian. Use
:func:`rebuild_shifted_hamiltonian` to apply the returned (or an externally
sourced) :class:`SymmetryShift`.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian to analyze. Must be restricted.
    n_alpha_electrons (int): The target number of alpha electrons
    n_beta_electrons (int): The target number of beta electrons

Returns:
    qdk_chemistry.algorithms.SymmetryShift: The computed shift parameters.

Raises:
    ValueError: If the Hamiltonian is unrestricted.

)",
              py::arg("hamiltonian"), py::arg("n_alpha_electrons"),
              py::arg("n_beta_electrons"));

  shifter.def("settings", &SymmetryShifter::settings,
              R"(
Access the shifter's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object for configuring the shifter

)",
              py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  shifter.def_property(
      "_settings",
      [](SymmetryShifterBase &algo) -> Settings & { return algo.settings(); },
      [](SymmetryShifterBase &algo,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        if (!new_settings) {
          throw py::type_error(
              "_settings must be a Settings instance, not None.");
        }
        algo.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal,
      R"(
Internal settings object property.

This property allows derived classes to replace the settings object with a specialized Settings subclass in their constructors.

)");

  shifter.def("name", &SymmetryShifter::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm

)");

  shifter.def("type_name", &SymmetryShifter::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm

)");

  // Factory class binding - creates SymmetryShifterFactory class with
  // static methods
  qdk::chemistry::python::bind_algorithm_factory<
      SymmetryShifterFactory, SymmetryShifter, SymmetryShifterBase>(
      m, "SymmetryShifterFactory");

  shifter.def("__repr__", [](const SymmetryShifter &) {
    return "<qdk_chemistry.algorithms.SymmetryShifter>";
  });

  qdk::chemistry::python::bind_create_nested(shifter);

  // Bind concrete microsoft::FermionicLowRankShifter implementation
  py::class_<microsoft::FermionicLowRankShifter, SymmetryShifter,
             py::smart_holder>(m, "FermionicLowRankShifter", R"(
Fermionic low-rank BLISS symmetry shifter.

Computes the block-invariant symmetry shift (BLISS) parameters (mu1, mu2, xi)
with the fermionic low-rank method of Patel et al. (arXiv:2409.18277): the
physical two-electron coefficient 1/2 g is double-factorized, each fragment
receives the closed-form median shift, and the one-electron shift is optimized
against the resulting effective one-electron operator.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    shifter = alg.FermionicLowRankShifter()

    # Optionally truncate the double factorization
    shifter.settings().set("df_truncation_threshold", 1e-8)

    shifted = shifter.run(hamiltonian, n_alpha, n_beta)

See Also:
    :class:`SymmetryShifter`
    :func:`rebuild_shifted_hamiltonian`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a fermionic low-rank symmetry shifter with default settings
(df_truncation_threshold = 0.0).

)")
      .def("__repr__", [](const microsoft::FermionicLowRankShifter &) {
        return "<qdk_chemistry.algorithms.FermionicLowRankShifter>";
      });
}
