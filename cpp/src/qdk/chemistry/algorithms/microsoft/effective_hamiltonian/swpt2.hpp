// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cmath>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Settings for the second-order Schrieffer-Wolff downfold.
 *
 * `denom_flow` and `denom_imaginary_shift` are mutually exclusive denominator
 * regularizers; a positive value enables one, zero disables it, and with both
 * disabled the floored inverse is used. The current denominator operator is a
 * semicanonical, spin-free generalized Fock.
 */
class SchriefferWolffPT2Settings : public qdk::chemistry::data::Settings {
 public:
  SchriefferWolffPT2Settings() {
    set_default(
        "denom_floor", 1e-8,
        "Hard cutoff used by unregularized denominators and raw-amplitude "
        "diagnostics.",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default(
        "denom_flow", 1.0,
        "Flow-parameter denominator regularizer, 1/D -> (1-exp(-s*D^2))/D, in "
        "units of Eh^-2. Set to 0 to disable. Mutually exclusive with "
        "denom_imaginary_shift; with both disabled the unregularized inverse "
        "is used, floored by denom_floor. This borrows the DSRG damping form; "
        "it does not turn the downfold into a full DSRG calculation.",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default(
        "denom_imaginary_shift", 0.0,
        "CASPT2-like imaginary level shift, 1/D -> D / (D^2 + shift^2), in "
        "units of Eh. Set to 0 to disable. Mutually exclusive with denom_flow; "
        "with both disabled the unregularized inverse is used, floored by "
        "denom_floor.",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default("semicanonicalize", true,
                "Diagonalize the generalized Fock independently within the "
                "inactive, active, and virtual blocks before forming Fock "
                "denominators.");
    set_default(
        "max_folded_occupation_deviation", 0.5,
        "Largest allowed deviation from an integer reference occupation (0 or "
        "2) for an orbital folded into the external space. Folded occupations "
        "are rounded to the nearest of 0 or 2; the total electron count is "
        "preserved because the active space receives whatever the folded "
        "orbitals do not take. Must be below 1, so a singly occupied orbital "
        "is never folded on an arbitrary rounding.",
        data::BoundConstraint<double>{0.0, std::nextafter(1.0, 0.0)});
  }
  ~SchriefferWolffPT2Settings() override = default;
};

/**
 * @brief Second-order Schrieffer-Wolff (Van Vleck) effective-Hamiltonian
 * downfold with semicanonical generalized-Fock orbital-energy denominators.
 *
 * Computes `H_eff = H_BD + 1/2 [S, H_OD]`, truncated to <= 2-body, folding the
 * window's external space Q onto its kept space P. S solves
 * `[F0, S] = H_OD` for a diagonal generalized-Fock F0; a regularizer setting
 * replaces the bare inverse denominators to damp intruder-state channels, at
 * the cost of solving that equation only approximately.
 *
 * The implementation assumes a common restricted MO basis, supporting RHF,
 * ROHF, and spin-adapted CAS references. Every singly occupied ROHF orbital
 * must be active. Noncanonical orbitals are semicanonicalized independently
 * within inactive, active, and virtual blocks. Intruder diagnostics are logged,
 * with an additional warning for large raw amplitudes.
 *
 * The reference wavefunction supplies the density over W; P selects which
 * orbitals are kept and need not coincide with the reference active space.
 * Folded (external) orbitals have their reference occupation rounded to doubly
 * occupied or empty, bounded by `max_folded_occupation_deviation`; the total
 * electron count is preserved because the active space receives whatever the
 * folded orbitals do not take.
 * See `swpt2_kernel.hpp` for the operator and tensor conventions.
 */
class SchriefferWolffPT2Constructor
    : public qdk::chemistry::algorithms::EffectiveHamiltonianConstructor {
 public:
  SchriefferWolffPT2Constructor() {
    _settings = std::make_unique<SchriefferWolffPT2Settings>();
  }
  ~SchriefferWolffPT2Constructor() override = default;

  std::string name() const final { return "qdk_swpt2"; }
  std::vector<std::string> aliases() const override {
    return {"qdk_swpt2", "swpt2", "schrieffer_wolff"};
  }

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices)
      const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
