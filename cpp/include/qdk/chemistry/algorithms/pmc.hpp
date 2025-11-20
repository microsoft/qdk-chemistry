// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <iostream>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @class ProjectedMultiConfigurationSettings
 * @brief Settings class specific to multi-configurational calculations
 *
 * This class extends the base Settings class with parameters specific to
 * projected multi-configurational calculations. It provides default values
 * for commonly used settings in PMC calculations such as reduced density
 * matrix generation and convergence thresholds.
 *
 * @see data::Settings
 */
class ProjectedMultiConfigurationSettings : public MultiConfigurationSettings {
 public:
  /**
   * @brief Default constructor
   *
   * Creates a multi-configurational settings object with default parameter
   * values. It inherits default settings from MultiConfigurationSettings.
   */
  ProjectedMultiConfigurationSettings() : MultiConfigurationSettings() {}

  /**
   * @brief Virtual destructor
   */
  virtual ~ProjectedMultiConfigurationSettings() = default;
};

/**
 * @class ProjectedMultiConfigurationCalculator
 * @brief Abstract base class for projected multi-configurational calculations
 * in quantum chemistry
 *
 * This class provides the interface for projected multi-configurational-based
 * quantum chemistry calculations. This contracts the
 * MultiConfigurationCalculator in that the space of determinants upon which
 * the Hamltonian is projected is taken to be a *free parameter* and must be
 * specified. In this manner, the high-performance solvers which underly other
 * MC algorithms can be interfaced with external methods for selecting
 * important determinants.
 *
 * The calculator takes a Hamiltonian and a set of configurations as input and
 * returns both the calculated energy and the corresponding
 * multi-configurational wavefunction.
 *
 * @see data::Hamiltonian
 * @see data::Wavefunction
 * @see data::Settings
 */
class ProjectedMultiConfigurationCalculator
    : public Algorithm<ProjectedMultiConfigurationCalculator,
                       std::pair<double, std::shared_ptr<data::Wavefunction>>,
                       std::shared_ptr<data::Hamiltonian>,
                       const std::vector<data::Configuration>&> {
 public:
  /**
   * @brief Default constructor
   *
   * Creates a projected multi-configurational calculator with default
   * settings.
   */
  ProjectedMultiConfigurationCalculator() = default;

  /**
   * @brief Virtual destructor
   *
   * Ensures proper cleanup of derived classes.
   */
  virtual ~ProjectedMultiConfigurationCalculator() = default;

  /**
   * @brief Perform projected multi-configurational calculation
   *
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param hamiltonian The Hamiltonian operator describing the quantum system
   * @param configurations The set of configurations/determinants to project
   * the Hamiltonian onto
   * \endcond
   *
   * @return A pair containing the calculated energy (first) and the resulting
   *         multi-configurational wavefunction (second)
   *
   * @throw std::runtime_error if the calculation fails
   * @throw std::invalid_argument if hamiltonian is invalid
   * @throw std::invalid_argument if configurations is invalid
   * @throw SettingsAreLocked if attempting to modify settings after run() is
   * called
   *
   * @note Settings are automatically locked when this method is called and
   * cannot be modified during or after execution.
   *
   * @see data::Hamiltonian
   * @see data::Wavefunction
   * @see data::Configuration
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name
   *
   * @return The algorithm's type name
   */
  std::string type_name() const final {
    return "projected_multi_configuration_calculator";
  }

 protected:
  /**
   * @brief Implementation of projected multi-configurational calculation
   *
   * This method contains the actual calculation logic. It is automatically
   * called by run() after settings have been locked.
   *
   * @param hamiltonian The Hamiltonian operator
   * @param configurations The set of configurations to project onto
   * @return A pair containing the energy and wavefunction
   */
  virtual std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      const std::vector<data::Configuration>& configurations) const = 0;
};

/**
 * @brief Factory class for creating projected multi-configurational
 * calculator instances.
 *
 * The ProjectedMultiConfigurationCalculatorFactory implements the Factory
 * design pattern to dynamically create and manage different implementations
 * of projected multi-configurational calculators. This allows the library to
 * support multiple PMC calculation methods while providing a unified
 * interface for clients.
 *
 * The factory maintains a registry of calculator implementations identified
 * by string keys. New implementations can be registered at runtime using the
 * register_builder method, and instances can be created using the create
 * method.
 *
 * Typical usage:
 * ```
 * // Register a custom implementation
 * ProjectedMultiConfigurationCalculatorFactory::register_instance("my_method",
 * []() { return std::make_unique<MyProjectedMultiConfigurationCalculator>();
 * });
 *
 * // Create an instance
 * auto calculator =
 * ProjectedMultiConfigurationCalculatorFactory::create("my_method");
 *
 * // Get available implementations
 * auto available = ProjectedMultiConfigurationCalculatorFactory::available();
 * ```
 *
 * @see ProjectedMultiConfigurationCalculator for the interface implemented by
 * concrete calculators
 */
struct ProjectedMultiConfigurationCalculatorFactory
    : public AlgorithmFactory<ProjectedMultiConfigurationCalculator,
                              ProjectedMultiConfigurationCalculatorFactory> {
  static std::string algorithm_type_name() {
    return "projected_multi_configuration_calculator";
  }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "macis_pmc"; }
};

}  // namespace qdk::chemistry::algorithms
