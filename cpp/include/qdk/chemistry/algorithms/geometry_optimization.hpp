// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <qdk/chemistry/data/nuclear_hessian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <tuple>

namespace qdk::chemistry::algorithms {

/**
 * @brief Energy, optimized structure, optional wavefunction, and optional
 * Hessian from a geometry optimization run.
 */
using GeometryOptimizationResult =
    std::tuple<double, std::shared_ptr<data::Structure>,
               std::optional<std::shared_ptr<data::Wavefunction>>,
               std::optional<std::shared_ptr<data::NuclearHessian>>>;

/**
 * @class GeometryOptimizerSettings
 * @brief Shared settings for geometry optimization algorithms.
 */
class GeometryOptimizerSettings : public data::Settings {
 public:
  /**
   * @brief Construct geometry optimization settings with common defaults.
   */
  GeometryOptimizerSettings() {
    set_default("derivative_calculator",
                data::AlgorithmRef("nuclear_derivative_calculator",
                                   "finite_difference"),
                "Nuclear derivative calculator used to evaluate energies and "
                "gradients during optimization.");
    set_default("transition_state", false,
                "Whether to run transition-state optimization instead of "
                "minimum-energy geometry optimization.");
    set_default("max_iterations", static_cast<int64_t>(300),
                "Maximum number of geometry optimization steps.",
                data::BoundConstraint<int64_t>{1, 1000000});
    set_default("convergence_energy", 1.0e-6,
                "Energy convergence threshold used by optimizers that expose "
                "an energy convergence setting.",
                data::BoundConstraint<double>{0.0, 1.0});
    set_default("convergence_gradient", 3.0e-4,
                "Gradient convergence threshold used by optimizers that expose "
                "a gradient convergence setting.",
                data::BoundConstraint<double>{0.0, 1.0});
    set_default("convergence_displacement", 1.2e-3,
                "Displacement convergence threshold used by optimizers that "
                "expose a displacement convergence setting.",
                data::BoundConstraint<double>{0.0, 1.0});
    set_default("compute_hessian", false,
                "Whether to compute a nuclear Hessian at the optimized "
                "geometry before returning.");
  }
};

/**
 * @class GeometryOptimizer
 * @brief Base class for geometry optimization algorithms.
 *
 * Implementations optimize molecular coordinates using the same input
 * arguments as nuclear derivative calculators. Results include the optimized
 * energy and structure plus optional wavefunction and Hessian values.
 */
class GeometryOptimizer
    : public Algorithm<GeometryOptimizer, GeometryOptimizationResult,
                       std::shared_ptr<data::Structure>, int, int,
                       NuclearDerivativeSeedType> {
 public:
  /**
   * @brief Construct a geometry optimizer with shared settings.
   */
  GeometryOptimizer() {
    _settings = std::make_unique<GeometryOptimizerSettings>();
  }
  virtual ~GeometryOptimizer() = default;

  using Algorithm::run;

  virtual std::string name() const = 0;

  /**
   * @brief Return the factory type name for geometry optimizers.
   */
  std::string type_name() const final { return "geometry_optimizer"; }

 protected:
  /**
   * @brief Implementation hook for derived geometry optimizers.
   */
  virtual GeometryOptimizationResult _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity, NuclearDerivativeSeedType seed) const = 0;
};

/**
 * @brief Factory for geometry optimizer implementations.
 */
struct GeometryOptimizerFactory
    : public AlgorithmFactory<GeometryOptimizer, GeometryOptimizerFactory> {
  /**
   * @brief Return the algorithm type name managed by this factory.
   */
  static std::string algorithm_type_name() { return "geometry_optimizer"; }

  /**
   * @brief Register built-in geometry optimizer implementations.
   */
  static void register_default_instances();

  /**
   * @brief Return the default geometry optimizer implementation name.
   */
  static std::string default_algorithm_name() { return "geometric"; }
};

}  // namespace qdk::chemistry::algorithms
