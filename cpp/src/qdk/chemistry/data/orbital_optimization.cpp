// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/data/orbital_optimization.hpp>
#include <stdexcept>

namespace qdk::chemistry::data {

OrbitalOptimizationResult::OrbitalOptimizationResult(
    std::shared_ptr<Orbitals> orbitals, double initial_objective,
    double final_objective, size_t iterations, bool converged)
    : orbitals_(std::move(orbitals)),
      initial_objective_(initial_objective),
      final_objective_(final_objective),
      iterations_(iterations),
      converged_(converged) {
  if (!orbitals_) {
    throw std::invalid_argument("OrbitalOptimizationResult requires orbitals");
  }
}

ActiveSpaceOptimizationResult::ActiveSpaceOptimizationResult(
    double energy, std::shared_ptr<Wavefunction> wavefunction, bool converged,
    size_t macro_iterations, std::vector<double> energy_history,
    std::vector<double> objective_history)
    : energy_(energy),
      wavefunction_(std::move(wavefunction)),
      converged_(converged),
      macro_iterations_(macro_iterations),
      energy_history_(std::move(energy_history)),
      objective_history_(std::move(objective_history)) {
  if (!wavefunction_) {
    throw std::invalid_argument(
        "ActiveSpaceOptimizationResult requires a wavefunction");
  }
  if (energy_history_.size() != objective_history_.size()) {
    throw std::invalid_argument(
        "Energy and objective histories must have the same length");
  }
  if (macro_iterations_ != energy_history_.size()) {
    throw std::invalid_argument(
        "Macro-iteration count must match the history length");
  }
}

}  // namespace qdk::chemistry::data
