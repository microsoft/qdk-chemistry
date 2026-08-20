// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <memory>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Result of one orbital-optimization operation.
 *
 * An orbital optimizer consumes a correlated @c Wavefunction (it needs the
 * reduced density matrices of the current state to evaluate and differentiate
 * its objective) but returns only a set of rotated @c Orbitals. The output is
 * deliberately an *orbital proposal*, not a wavefunction: the rotation changes
 * the one- and two-electron integrals, so the wavefunction expressed in the new
 * orbital basis is no longer variationally optimal. Recovering a correlated
 * state therefore requires a subsequent solve (for example, an active-space
 * calculation) in the returned basis. Keeping the contract at the orbital level
 * makes this dependency explicit and lets callers choose how to re-solve. The
 * returned @c Orbitals also carry the inactive and active partition labels
 * describing the proposed subspaces; the virtual space is their complement.
 *
 * The remaining semantics are optimizer-agnostic: @c iterations is the number
 * of optimizer iterations performed (whatever an iteration means for the
 * concrete method, e.g. a Jacobi sweep or a Newton step), and
 * @c initial_objective / @c final_objective are the optimizer's own objective
 * before and after optimization.
 */
class OrbitalOptimizationResult {
 public:
  OrbitalOptimizationResult(std::shared_ptr<Orbitals> orbitals,
                            double initial_objective, double final_objective,
                            size_t iterations, bool converged);

  const std::shared_ptr<Orbitals>& orbitals() const { return orbitals_; }
  double initial_objective() const { return initial_objective_; }
  double final_objective() const { return final_objective_; }
  size_t iterations() const { return iterations_; }
  bool converged() const { return converged_; }

 private:
  std::shared_ptr<Orbitals> orbitals_;
  double initial_objective_;
  double final_objective_;
  size_t iterations_;
  bool converged_;
};

/**
 * @brief Result of a self-consistent active-space optimization.
 *
 * History semantics: @c energy_history and @c objective_history each record one
 * entry per completed macro iteration, in chronological order. Entry @c i holds
 * the energy and orbital objective at the end of macro iteration @c i
 * (0-based). Both histories therefore have length @c macro_iterations, and @c
 * energy and the concrete final wavefunction correspond to the last recorded
 * entry (or to the input state when @c macro_iterations is zero). These
 * invariants are enforced by the constructor.
 */
class ActiveSpaceOptimizationResult {
 public:
  ActiveSpaceOptimizationResult(double energy,
                                std::shared_ptr<Wavefunction> wavefunction,
                                bool converged, size_t macro_iterations,
                                std::vector<double> energy_history,
                                std::vector<double> objective_history);

  double energy() const { return energy_; }
  const std::shared_ptr<Wavefunction>& wavefunction() const {
    return wavefunction_;
  }
  bool converged() const { return converged_; }
  size_t macro_iterations() const { return macro_iterations_; }
  const std::vector<double>& energy_history() const { return energy_history_; }
  const std::vector<double>& objective_history() const {
    return objective_history_;
  }

 private:
  double energy_;
  std::shared_ptr<Wavefunction> wavefunction_;
  bool converged_;
  size_t macro_iterations_;
  std::vector<double> energy_history_;
  std::vector<double> objective_history_;
};

}  // namespace qdk::chemistry::data
