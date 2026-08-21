// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/orbital_optimization.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @brief Abstract base class for objective-driven orbital optimizations.
 *
 * Unlike an orbital localizer, an orbital optimizer may rotate orbitals across
 * inactive, active, and virtual subspace boundaries.
 *
 * The input and output types are deliberately asymmetric. The optimizer takes a
 * correlated @c Wavefunction because its objective is defined in terms of the
 * current state's reduced density matrices, but it returns an
 * @c OrbitalOptimizationResult carrying only the rotated @c Orbitals. Those
 * orbitals are a *proposal*: rotating the orbital basis changes the underlying
 * integrals, so the correlated state is no longer optimal and must be
 * recomputed by a subsequent solve in the returned basis. See @c
 * data::OrbitalOptimizationResult for the full rationale.
 */
class OrbitalOptimizer
    : public Algorithm<OrbitalOptimizer,
                       std::shared_ptr<data::OrbitalOptimizationResult>,
                       std::shared_ptr<data::Wavefunction>> {
 public:
  OrbitalOptimizer() = default;
  virtual ~OrbitalOptimizer() = default;

  using Algorithm::run;

  virtual std::string name() const = 0;
  std::string type_name() const final { return "orbital_optimizer"; }

 protected:
  virtual std::shared_ptr<data::OrbitalOptimizationResult> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const = 0;
};

struct OrbitalOptimizerFactory
    : public AlgorithmFactory<OrbitalOptimizer, OrbitalOptimizerFactory> {
  static std::string algorithm_type_name() { return "orbital_optimizer"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_qio"; }
};

}  // namespace qdk::chemistry::algorithms
