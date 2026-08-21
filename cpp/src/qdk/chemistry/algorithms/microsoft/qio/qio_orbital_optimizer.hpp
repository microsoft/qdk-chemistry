// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/orbital_optimization.hpp>
#include <string>

#include "jacobi_settings.hpp"

namespace qdk::chemistry::algorithms::microsoft {

class QIOOrbitalOptimizerSettings : public qio::JacobiSettings {};

/**
 * @brief Full-window quantum-information orbital optimizer.
 *
 * Embeds the correlated active-space RDMs into the frozen inactive and empty
 * virtual spaces, then minimizes the total single-orbital entropy over all
 * orbitals.
 */
class QIOOrbitalOptimizer final : public OrbitalOptimizer {
 public:
  QIOOrbitalOptimizer() {
    _settings = std::make_unique<QIOOrbitalOptimizerSettings>();
  }

  std::string name() const final { return "qdk_qio"; }

 protected:
  std::shared_ptr<data::OrbitalOptimizationResult> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const final;
};

}  // namespace qdk::chemistry::algorithms::microsoft
