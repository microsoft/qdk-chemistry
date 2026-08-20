// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/orbital_optimization.hpp>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @brief Settings shared by self-consistent active-space optimizers.
 */
class ActiveSpaceOptimizerSettings : public data::Settings {
 public:
  ActiveSpaceOptimizerSettings() {
    set_default("hamiltonian_constructor",
                data::AlgorithmRef("hamiltonian_constructor", "qdk"),
                "Hamiltonian constructor used after each orbital update.");
    set_default(
        "multi_configuration_calculator",
        data::AlgorithmRef("multi_configuration_calculator", "macis_cas"),
        "Correlated active-space solver used in each macro iteration.");
    set_default(
        "orbital_optimizer", data::AlgorithmRef("orbital_optimizer", ""),
        "Orbital optimizer used to update the active-space projector. No "
        "default implementation is registered yet; concrete active-space "
        "optimizers must select one before the workflow can run.");
    set_default(
        "max_macro_iterations", static_cast<int64_t>(20),
        "Maximum number of self-consistent macro iterations.",
        data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
    set_default(
        "energy_tolerance", 1e-8, "Absolute energy convergence tolerance.",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default(
        "objective_tolerance", 1e-8,
        "Absolute orbital-objective convergence tolerance.",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
  }
  virtual ~ActiveSpaceOptimizerSettings() = default;
};

/**
 * @brief Abstract base class for self-consistent active-space optimization.
 */
class ActiveSpaceOptimizer
    : public Algorithm<ActiveSpaceOptimizer,
                       std::shared_ptr<data::ActiveSpaceOptimizationResult>,
                       std::shared_ptr<data::Orbitals>, unsigned int,
                       unsigned int> {
 public:
  ActiveSpaceOptimizer() {
    _settings = std::make_unique<ActiveSpaceOptimizerSettings>();
  }
  virtual ~ActiveSpaceOptimizer() = default;

  using Algorithm::run;

  virtual std::string name() const = 0;
  std::string type_name() const final { return "active_space_optimizer"; }

 protected:
  virtual std::shared_ptr<data::ActiveSpaceOptimizationResult> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals,
      unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const = 0;
};

struct ActiveSpaceOptimizerFactory
    : public AlgorithmFactory<ActiveSpaceOptimizer,
                              ActiveSpaceOptimizerFactory> {
  static std::string algorithm_type_name() { return "active_space_optimizer"; }
  static void register_default_instances() {}
  static std::string default_algorithm_name() { return ""; }
};

}  // namespace qdk::chemistry::algorithms
