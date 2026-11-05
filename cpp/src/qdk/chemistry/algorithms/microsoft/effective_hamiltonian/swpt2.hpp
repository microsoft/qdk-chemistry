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
 */
class SchriefferWolffPT2Settings : public qdk::chemistry::data::Settings {
 public:
  SchriefferWolffPT2Settings() {
    set_default(
        "regularizer_flow", 0.5,
        "Flow parameter s of the DSRG regularizer, in Eh^-2: each "
        "second-order energy denominator 1/D becomes (1-exp(-2*s*D^2))/D. "
        "Larger values regularize less; 0 selects the bare inverse.",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default("semicanonicalize", true,
                "Diagonalize each spin's generalized Fock independently within "
                "the inactive, active, and virtual blocks before forming Fock "
                "denominators.");
    set_default(
        "max_folded_occupation_deviation", 0.5,
        "Largest allowed deviation of a folded orbital's reference occupation "
        "from 0 or 2, to which it is rounded; the active space receives the "
        "remaining electrons. Must be below 1, so a singly occupied orbital "
        "is never folded.",
        data::BoundConstraint<double>{0.0, std::nextafter(1.0, 0.0)});
  }
  ~SchriefferWolffPT2Settings() override = default;
};

/**
 * @brief Second-order Schrieffer-Wolff downfold of the window's external
 * orbitals onto the kept space P (`p_indices`).
 *
 * Emits P exp(S) H exp(-S) P through second order, truncated to two-body, for
 * a generator S regularized by `regularizer_flow`, with semicanonical
 * generalized-Fock denominators; three-body terms are folded with the
 * reference 1-RDM. Restricted input gives a spin-free result, unrestricted
 * input an unrestricted one. The emitted two-body blocks are only 4-fold
 * symmetric, so consumers must read the full dense blocks.
 */
class SchriefferWolffPT2Constructor
    : public qdk::chemistry::algorithms::EffectiveHamiltonianConstructor {
 public:
  SchriefferWolffPT2Constructor() {
    _settings = std::make_unique<SchriefferWolffPT2Settings>();
  }
  ~SchriefferWolffPT2Constructor() override = default;

  std::string name() const final { return "qdk_swpt2"; }

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices)
      const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
