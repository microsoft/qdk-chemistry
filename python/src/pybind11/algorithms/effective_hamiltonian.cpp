// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/effective_hamiltonian.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2.hpp"

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
      std::shared_ptr<Wavefunction> reference,
      std::shared_ptr<Hamiltonian> hamiltonian) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Hamiltonian>,
                           EffectiveHamiltonianConstructor, _run_impl,
                           reference, hamiltonian);
  }
};

void bind_effective_hamiltonian_constructor(py::module &m) {
  // Default implementations are registered lazily by the AlgorithmFactory base
  // class on first registry access, so no explicit registration is needed here.

  // EffectiveHamiltonianConstructor abstract base class
  py::class_<EffectiveHamiltonianConstructor,
             EffectiveHamiltonianConstructorBase, py::smart_holder>
      eff_ham(m, "EffectiveHamiltonianConstructor", R"(
Abstract base class for effective-Hamiltonian downfolding.

Given a reference :class:`~qdk_chemistry.data.Wavefunction` (whose active space
defines the model space ``P`` and whose occupations/RDMs define the reference)
and an input :class:`~qdk_chemistry.data.Hamiltonian` built over the whole
downfolding window ``W = P u Q``, a concrete constructor folds the external
space ``Q`` into an effective Hamiltonian acting on ``P``. This is a distinct
algorithm type from :class:`~qdk_chemistry.algorithms.HamiltonianConstructor`,
which builds the bare integral Hamiltonian from
:class:`~qdk_chemistry.data.Orbitals`.

The input Hamiltonian must be built with its active space set to the whole
window ``W`` (every orbital to be folded is "active" to the integral
constructor), otherwise the ``P<->Q`` couplings are already gone.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> downfolder = alg.create("effective_hamiltonian_constructor", "qdk_swpt2")
    >>> h_eff = downfolder.run(reference, window_hamiltonian)
)");

  eff_ham.def(py::init<>(), R"(
Create an ``EffectiveHamiltonianConstructor`` instance.

Default constructor for the abstract base class; typically called from derived
class constructors.
)");

  eff_ham.def("run", &EffectiveHamiltonianConstructor::run,
              py::arg("reference"), py::arg("hamiltonian"), R"(
Fold the window Hamiltonian onto the reference active space.

This method automatically locks settings before execution to prevent
modifications during construction.

Args:
    reference (qdk_chemistry.data.Wavefunction): Reference wavefunction; its
        active space is the model space ``P`` and its occupations define the
        reference.
    hamiltonian (qdk_chemistry.data.Hamiltonian): Input Hamiltonian built over
        the whole window ``W = P u Q``.

Returns:
    qdk_chemistry.data.Hamiltonian: The effective Hamiltonian acting on ``P``.

Raises:
    SettingsAreLocked: If attempting to modify settings after run() is called
)");

  eff_ham.def("settings", &EffectiveHamiltonianConstructor::settings, R"(
Access the constructor's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object.
)",
              py::return_value_policy::reference_internal);

  // Expose _settings as a writable property for derived classes
  eff_ham.def_property(
      "_settings",
      [](EffectiveHamiltonianConstructorBase &constr) -> Settings & {
        return constr.settings();
      },
      [](EffectiveHamiltonianConstructorBase &constr,
         std::unique_ptr<qdk::chemistry::data::Settings> new_settings) {
        constr.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal, R"(
Internal settings object property.

Allows derived classes to replace the settings object with a specialized
Settings subclass in their constructors.
)");

  eff_ham.def("name", &EffectiveHamiltonianConstructor::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm
)");

  eff_ham.def("type_name", &EffectiveHamiltonianConstructor::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");

  eff_ham.def("hash", &EffectiveHamiltonianConstructor::hash,
              py::arg("reference"), py::arg("hamiltonian"));

  eff_ham.def("__repr__", [](const EffectiveHamiltonianConstructor &) {
    return "<qdk_chemistry.algorithms.EffectiveHamiltonianConstructor>";
  });

  qdk::chemistry::python::bind_create_nested(eff_ham);

  // Factory class binding
  qdk::chemistry::python::bind_algorithm_factory<
      EffectiveHamiltonianConstructorFactory, EffectiveHamiltonianConstructor,
      EffectiveHamiltonianConstructorBase>(
      m, "EffectiveHamiltonianConstructorFactory");

  // Bind concrete microsoft::SchriefferWolffPT2Constructor implementation
  py::class_<microsoft::SchriefferWolffPT2Constructor,
             EffectiveHamiltonianConstructor, py::smart_holder>(
      m, "QdkSchriefferWolffPT2Constructor", R"(
Second-order Schrieffer-Wolff (Van Vleck) effective-Hamiltonian downfold.

A single-commutator canonical transformation
``H_eff = H_BD + 1/2 [S, H_OD]``, truncated to ``<= 2``-body, folding the
external space ``Q`` of the window onto the reference active space ``P``. A
separate diagonal generalized-Fock operator defines the generator denominators.
The reference and window must use the same restricted MO basis. RHF, ROHF, and
spin-adapted CAS references are supported; every singly occupied ROHF orbital
must belong to the active space. UHF orbitals are not supported. Registered as
``"qdk_swpt2"`` (aliases ``"sw"``, ``"schrieffer_wolff"``).

The ``regularizer`` setting selects ``"flow"`` (default), ``"shift"``, or
``"bare"``. The corresponding ``denom_flow`` / ``denom_shift`` parameter
controls that scheme; ``denom_floor`` is the bare-denominator cutoff.
``semicanonicalize`` is enabled by default and diagonalizes the generalized
Fock independently within inactive, active, and virtual orbital blocks before
forming denominators; the emitted Hamiltonian is rotated back to the original
reference basis. ROHF uses the spin-traced density and common spin-free
orbital energies, preserving spin symmetry while the active solve selects the
desired spin sector.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    downfolder = alg.create("effective_hamiltonian_constructor", "qdk_swpt2")
    downfolder.settings().set("regularizer", "shift")
    downfolder.settings().set("denom_shift", 0.5)
    h_eff = downfolder.run(reference, window_hamiltonian)

See Also:
    :class:`EffectiveHamiltonianConstructor`
    :class:`qdk_chemistry.data.Wavefunction`
    :class:`qdk_chemistry.data.Hamiltonian`
)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a Schrieffer-Wolff PT2 downfold with flow regularization enabled by
default. Set ``regularizer`` to ``"bare"`` for unregularized second-order PT.
)");
}
