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
  // SymmetryShiftCoeffs: the (mu1, mu2, xi) shift parameters, decoupled from
  // how they were produced so they can be inspected or supplied from any
  // source.
  py::class_<SymmetryShiftCoeffs>(m, "SymmetryShiftCoeffs", R"(
Number-symmetry shift parameters.

Bundles the three quantities (mu1, mu2, xi) that define the symmetry-shift
operator subtracted from a Hamiltonian to reduce its fermionic 1-norm while
leaving the target electron-number sector's energy invariant. A SymmetryShiftCoeffs
carries only the *result* of a shift computation, so it can come from
:meth:`SymmetryShifter.last_shift` or from an external source. Applying one
is :meth:`SymmetryShifter.run`'s job.
)")
      .def(py::init<>())
      .def_readwrite("mu1", &SymmetryShiftCoeffs::mu1, "One-electron shift.")
      .def_readwrite("mu2", &SymmetryShiftCoeffs::mu2, "Two-electron shift.")
      .def_readwrite("xi", &SymmetryShiftCoeffs::xi,
                     "Two-electron shift matrix (norb x norb).")
      .def("__repr__", [](const SymmetryShiftCoeffs &s) {
        return "<qdk_chemistry.algorithms.SymmetryShiftCoeffs mu1=" +
               std::to_string(s.mu1) + " mu2=" + std::to_string(s.mu2) +
               " xi=" + std::to_string(s.xi.rows()) + "x" +
               std::to_string(s.xi.cols()) + ">";
      });

  // SymmetryShifter abstract base class
  py::class_<SymmetryShifter, SymmetryShifterBase, py::smart_holder> shifter(
      m, "SymmetryShifter",
      R"(
Abstract base class for number-symmetry Hamiltonian shift algorithms.

A SymmetryShifter maps a Hamiltonian, together with the target number of
alpha/beta electrons, to a new Hamiltonian that is energetically equivalent
within the target electron-number sector but whose LCU/qubitization
coefficients (e.g. the fermionic 1-norm lambda) may be reduced.

:meth:`run` computes the shift and applies it in one step; computing one
without applying it is deliberately not exposed, since how a shift folds into
the Hamiltonian depends on the representation the implementation consumes.
The parameters that were applied can be read back afterwards from
:meth:`last_shift`.

Concrete implementations should inherit from this class.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> shifter = alg.FermionicLowRankShifter()
    >>> shifted = shifter.run(hamiltonian, n_alpha, n_beta)
    >>> shift = shifter.last_shift()

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

  shifter.def("last_shift", &SymmetryShifter::last_shift,
              R"(
The symmetry shift (mu1, mu2, xi) applied by the most recent :meth:`run`.

Returns:
    qdk_chemistry.algorithms.SymmetryShiftCoeffs | None: The shift the last
    :meth:`run` on this instance applied, or None if it has not been run or
    the implementation does not report one.

Note:
    Not synchronized. Use one shifter instance per thread if you intend to
    read this back.

)");

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

  shifter.def("aliases", &SymmetryShifter::aliases, R"(
The algorithm's aliases.

Returns:
    list[str]: All registered names for the algorithm

)");

  shifter.def("hash", &SymmetryShifter::hash, py::arg("hamiltonian"),
              py::arg("n_alpha_electrons"), py::arg("n_beta_electrons"), R"(
Deterministic content hash for a run with these inputs.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian to shift
    n_alpha_electrons (int): The target number of alpha electrons
    n_beta_electrons (int): The target number of beta electrons

Returns:
    str: 16-character hex content hash

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
fragments of an already double-factorized Hamiltonian each receive the
closed-form median shift, and the one-electron shift is optimized against the
resulting effective one-electron operator.

The input must be restricted (spin-restricted) and backed by a
``FactorizedHamiltonianContainer`` whose identity weight is zero and whose
rotations are complete orthogonal ones; anything else raises ``ValueError``.
The output is backed by the same container type: the shift is absorbed into
the fragment eigenvalues, so the result can be block-encoded without being
refactorized. Call ``get_two_body_integrals()`` for dense ones.

Only the total electron count ``n_alpha + n_beta`` enters the shift; this
method does not use Sz, so (5, 5) and (6, 4) give the same result.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    factorized = alg.DoubleFactorization().run(hamiltonian)

    shifter = alg.FermionicLowRankShifter()
    shifted = shifter.run(factorized, n_alpha, n_beta)
    shift = shifter.last_shift()

See Also:
    :class:`SymmetryShifter`

)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a fermionic low-rank symmetry shifter. It has no settings.

)")
      .def("__repr__", [](const microsoft::FermionicLowRankShifter &) {
        return "<qdk_chemistry.algorithms.FermionicLowRankShifter>";
      });
}
