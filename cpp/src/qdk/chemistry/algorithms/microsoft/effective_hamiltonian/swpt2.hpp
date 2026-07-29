// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Settings for the second-order Schrieffer-Wolff downfold.
 *
 * Exposes the energy-denominator knobs of the kernel `RegOptions`. The
 * denominator *scheme* (Fock now, approximate-Dyall later) is intended to be a
 * setting too, so it is not part of the method name.
 */
class SchriefferWolffPT2Settings : public qdk::chemistry::data::Settings {
 public:
  SchriefferWolffPT2Settings();
  ~SchriefferWolffPT2Settings() override = default;
};

/**
 * @brief Second-order Schrieffer-Wolff (Van Vleck) effective-Hamiltonian
 * downfold with orbital-energy (Moller-Plesset) denominators.
 *
 * A single-commutator canonical transformation `H_eff = H0 + 1/2 [S, V]`,
 * truncated to <= 2-body, folding the external space Q of the window onto the
 * reference active space P. See `swpt2_kernel.hpp` for the math.
 */
class SchriefferWolffPT2Constructor
    : public qdk::chemistry::algorithms::EffectiveHamiltonianConstructor {
 public:
  SchriefferWolffPT2Constructor() {
    _settings = std::make_unique<SchriefferWolffPT2Settings>();
  }
  ~SchriefferWolffPT2Constructor() override = default;

  std::string name() const final { return "swpt2"; }
  std::vector<std::string> aliases() const override {
    return {"swpt2", "sw", "schrieffer_wolff"};
  }

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
