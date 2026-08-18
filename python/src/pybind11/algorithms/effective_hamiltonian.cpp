// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>

#include "factory_bindings.hpp"
#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

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

  void replace_settings(std::unique_ptr<Settings> new_settings) {
    this->_settings = std::move(new_settings);
  }

  void validate_inputs(
      const std::shared_ptr<Wavefunction>& reference,
      const std::shared_ptr<Hamiltonian>& hamiltonian,
      const std::shared_ptr<const SymmetryBlockedIndexSet>& p_indices) const {
    this->_validate_inputs(reference, hamiltonian, p_indices);
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Wavefunction> reference,
      std::shared_ptr<Hamiltonian> hamiltonian,
      std::shared_ptr<const SymmetryBlockedIndexSet> p_indices) const override {
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Hamiltonian>,
                           EffectiveHamiltonianConstructor, _run_impl,
                           reference, hamiltonian, p_indices);
  }
};

void bind_effective_hamiltonian_constructor(py::module& m) {
  py::class_<EffectiveHamiltonianConstructor,
             EffectiveHamiltonianConstructorBase, py::smart_holder>
      constructor(m, "EffectiveHamiltonianConstructor", R"(
Abstract base class for effective-Hamiltonian construction.

Concrete implementations construct an effective Hamiltonian from a reference
wavefunction and an input Hamiltonian in the explicitly specified P-space.

``p_indices`` holds absolute molecular-orbital indices, drawn from the same
index universe as ``Orbitals.active_indices()``.

The returned Hamiltonian is expressed over ``P`` and must satisfy:

- its orbitals have ``active_indices()`` equal to ``p_indices``;
- its orbitals classify fully occupied orbitals of :math:`Q = W \setminus P`
  as inactive and unoccupied orbitals of ``Q`` as virtual, while preserving
  the input Hamiltonian's inactive orbitals;
- its inactive Fock matrix, when present, is consistent with the output
  inactive orbitals and may therefore differ from the input Hamiltonian's
  inactive Fock matrix;
- the scalar shift from folding in ``Q`` is added to the constant (zero-body)
  energy term, and the remaining ``Q`` contribution is folded into the
  integrals.

Input validation is opt-in. ``run()`` does not validate its arguments; concrete
implementations decide whether to call ``_validate_inputs``.

Examples:
    >>> import qdk_chemistry.algorithms as alg
    >>> constructor = alg.create("effective_hamiltonian_constructor", "algorithm_name")
    >>> effective_hamiltonian = constructor.run(reference, hamiltonian, p_indices)
)");

  constructor.def(py::init<>(), R"(
Create an ``EffectiveHamiltonianConstructor`` instance.

Default constructor for the abstract base class; typically called via
``super().__init__()`` from a derived class.
)");
  constructor.def("run", &EffectiveHamiltonianConstructor::run,
                  py::arg("reference"), py::arg("hamiltonian"),
                  py::arg("p_indices"), R"(
Construct the effective Hamiltonian acting on the target space ``P``.

Args:
    reference: Reference wavefunction providing the reference state.
    hamiltonian: Input Hamiltonian built over the whole window :math:`W = P \cup Q`.
    p_indices: Absolute molecular-orbital indices of the target space ``P``, which must lie within the input Hamiltonian's active orbital window.

Returns:
    The effective Hamiltonian acting on ``P``, following the output contract documented on this class.
)");
  constructor.def("settings", &EffectiveHamiltonianConstructor::settings,
                  py::return_value_policy::reference_internal, R"(
Access the constructor's configuration settings.

Returns:
    qdk_chemistry.data.Settings: Reference to the settings object.
)");
  constructor.def(
      "_validate_inputs",
      [](const EffectiveHamiltonianConstructorBase& instance,
         const std::shared_ptr<Wavefunction>& reference,
         const std::shared_ptr<Hamiltonian>& hamiltonian,
         const std::shared_ptr<const SymmetryBlockedIndexSet>& p_indices) {
        instance.validate_inputs(reference, hamiltonian, p_indices);
      },
      py::arg("reference"), py::arg("hamiltonian"), py::arg("p_indices"), R"(
Validate the common input-space contract.

Validation is opt-in: ``run()`` never calls this helper. Concrete
implementations may call it from ``_run_impl`` before performing
method-specific validation or computation.

Args:
    reference: Reference wavefunction whose active orbital space must be a subset of the input Hamiltonian's active orbital window.
    hamiltonian: Input Hamiltonian defining the outer orbital window.
    p_indices: Target P-space as absolute molecular-orbital indices, which must be a subset of the input Hamiltonian's active orbital window.

Raises:
    ValueError: If an input is null, the orbital bases or spin restrictions are incompatible, or the spaces do not satisfy :math:`P \subseteq W_H` and :math:`W_{\mathrm{ref}} \subseteq W_H`.
)");
  constructor.def_property(
      "_settings",
      [](EffectiveHamiltonianConstructorBase& instance) -> Settings& {
        return instance.settings();
      },
      [](EffectiveHamiltonianConstructorBase& instance,
         std::unique_ptr<Settings> new_settings) {
        instance.replace_settings(std::move(new_settings));
      },
      py::return_value_policy::reference_internal, R"(
Internal settings object property.

Allows derived classes to replace the settings object with a specialized
Settings subclass in their constructor.
)");
  constructor.def("name", &EffectiveHamiltonianConstructor::name, R"(
The algorithm's name.

Returns:
    str: The name of the algorithm
)");
  constructor.def("type_name", &EffectiveHamiltonianConstructor::type_name, R"(
The algorithm's type name.

Returns:
    str: The type name of the algorithm
)");
  constructor.def("hash", &EffectiveHamiltonianConstructor::hash,
                  py::arg("reference"), py::arg("hamiltonian"),
                  py::arg("p_indices"), R"(
Compute a deterministic content hash for a run with these inputs.

Args:
    reference: Reference wavefunction providing the reference state.
    hamiltonian: Input Hamiltonian built over the whole window :math:`W = P \cup Q`.
    p_indices: Target ``P`` indices within the input Hamiltonian's active orbital window.

Returns:
    str: 16-character hex content hash.
)");
  constructor.def("__repr__", [](const EffectiveHamiltonianConstructor&) {
    return "<qdk_chemistry.algorithms.EffectiveHamiltonianConstructor>";
  });

  qdk::chemistry::python::bind_create_nested(constructor);
  qdk::chemistry::python::bind_algorithm_factory<
      EffectiveHamiltonianConstructorFactory, EffectiveHamiltonianConstructor,
      EffectiveHamiltonianConstructorBase>(
      m, "EffectiveHamiltonianConstructorFactory");

  py::class_<microsoft::SchriefferWolffPT2Constructor,
             EffectiveHamiltonianConstructor, py::smart_holder>(
      m, "QdkSchriefferWolffPT2Constructor", R"(
Second-order Schrieffer-Wolff (Van Vleck) effective-Hamiltonian downfold.

Computes ``H_eff = H_BD + 1/2 [S, H_OD]``, truncated to ``<= 2``-body, folding
the window's external space ``Q`` onto its kept space ``P``. With bare
denominators, the generator solves ``[F0, S] = H_OD`` for a diagonal
generalized-Fock ``F0``. The flow and imaginary-shift settings instead build a
regularized generator, which solves that equation only approximately.
The reference and window must use the same restricted MO basis. RHF, ROHF, and
spin-adapted CAS references are supported; every singly occupied ROHF orbital
must belong to the active space. UHF orbitals are not supported. Registered as
``"qdk_swpt2"``, with the aliases ``"swpt2"`` and ``"schrieffer_wolff"``.

Denominator regularization is controlled by ``regularizer_sigma2`` (the
:math:`\sigma` of the :math:`\sigma^2` regularizer, in :math:`E_h^{-2}`, default
``1.0``), equivalently the DSRG flow parameter. Larger values regularize less and
``0`` leaves the bare inverse. It borrows the DSRG damping form but is not a full
DSRG calculation.
``semicanonicalize`` is enabled by default and diagonalizes the generalized
Fock independently within inactive, active, and virtual orbital blocks before
forming denominators; the emitted Hamiltonian is rotated back to the original
reference basis. ROHF uses the spin-traced density and common spin-free
orbital energies, preserving spin symmetry while the active solve selects the
desired spin sector.

``fold_above_two_body`` is enabled by default. The transformation generates
three-body terms that a Hamiltonian cannot hold; folding normal-orders them
against the reference density and keeps what falls to two-body, instead of
discarding them outright. Discarding is the larger error whenever the kept
space holds more than two electrons, but folding costs several times the kernel
time, so the setting exists for cases where that matters. It is ignored when the
kept space holds at most two electrons, where a three-body operator has no
matrix elements to contribute.

The active regularization, minimum denominator, maximum raw intruder amplitude,
and semicanonicalization status are logged when construction completes. A
warning is also logged when the raw amplitude exceeds one, where the
perturbation series stops contracting.

The kept space ``P`` is a required ``run()`` argument (``p_indices``): a
:class:`~qdk_chemistry.data.symmetry.SymmetryBlockedIndexSet` of window
(spatial) orbital indices. The reference wavefunction supplies
the density over ``W``; ``P`` selects which orbitals are kept and need not
coincide with the reference active space. A folded external orbital has its
reference occupation rounded to doubly occupied or empty, bounded by
``max_folded_occupation_deviation``. Rounding never changes the total electron
count, because the active space receives whatever the folded orbitals do not
take; the derived active electron count is logged and is the value to pass to
the active-space solver.

Typical usage:

.. code-block:: python

    import qdk_chemistry.algorithms as alg

    downfolder = alg.create("effective_hamiltonian_constructor", "qdk_swpt2")
    downfolder.settings().set("regularizer_sigma2", 0.4)
    h_eff = downfolder.run(reference, window_hamiltonian, p_indices)

See Also:
    :class:`EffectiveHamiltonianConstructor`
    :class:`qdk_chemistry.data.Wavefunction`
    :class:`qdk_chemistry.data.Hamiltonian`
)")
      .def(py::init<>(), R"(
Default constructor.

Initializes a Schrieffer-Wolff PT2 downfold with sigma^2 regularization enabled
by default. Set ``regularizer_sigma2`` to 0 for unregularized second-order PT.
)");
}
