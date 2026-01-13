// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/stability.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class StabilityCheckerSettings
 * @brief Settings container for the internal stability checker implementation
 *
 * This class extends the Settings class to provide specific default settings
 * for the internal stability checker. It pre-configures common stability
 * analysis parameters with sensible default values.
 *
 * Default settings include:
 * - internal: true - Check internal stability
 * - external: false - Check external stability
 *
 * Users can override these defaults by modifying the settings object
 * obtained from the StabilityChecker instance.
 *
 * Example:
 * ```cpp
 * auto checker =
 * qdk::chemistry::algorithms::StabilityCheckerFactory::create();
 * auto& settings = checker->settings();
 * ```
 *
 * @see qdk::chemistry::data::Settings
 * @see qdk::chemistry::algorithms::microsoft::StabilityChecker
 */
class StabilityCheckerSettings : public qdk::chemistry::data::Settings {
 public:
  /**
   * @brief Constructor that initializes default stability checker settings
   *
   * Creates a stability checker settings object with the following defaults
   */
  StabilityCheckerSettings() : qdk::chemistry::data::Settings() {
    set_default("internal", true);
    set_default("external", false);
    set_default("max_subspace", 80);
    set_default("davidson_tolerance", 1e-8);
    set_default("stability_tolerance", -1e-4);
    set_default("method", "hf");
  }
};

/**
 * @class StabilityChecker
 * @brief Implementation of the stability checker
 *
 * This class provides a concrete implementation of the stability checker using
 * the backend. It inherits from the base `StabilityChecker` class and
 * implements the stability analysis method to check if a wavefunction
 * corresponds to a true minimum or saddle point and return corresponding
 * eigenvectors.
 *
 * Typical usage:
 * ```cpp
 * // Assuming you have a converged wavefunction from SCF
 * auto checker =
 * qdk::chemistry::algorithms::StabilityCheckerFactory::create();
 *
 * // Configure settings if needed
 * checker->settings().set("external", true);
 *
 * // Perform stability check
 * auto [is_stable, result] = checker->run(wavefunction);
 *
 * if (is_stable) {
 *   std::cout << "Wavefunction is stable" << std::endl;
 * } else {
 *   std::cout << "Wavefunction is unstable" << std::endl;
 *   std::cout << "Smallest eigenvalue: " << result->get_smallest_eigenvalue()
 *             << std::endl;
 * }
 * ```
 *
 * @see qdk::chemistry::algorithms::StabilityChecker
 * @see qdk::chemistry::data::Wavefunction
 * @see qdk::chemistry::data::StabilityResult
 * @see qdk::chemistry::algorithms::StabilityCheckerFactory
 */
class StabilityChecker : public qdk::chemistry::algorithms::StabilityChecker {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes a stability checker with default settings.
   */
  StabilityChecker() {
    _settings = std::make_unique<StabilityCheckerSettings>();
  };

  /**
   * @brief Virtual destructor
   */
  ~StabilityChecker() = default;

  virtual std::string name() const final { return "qdk"; }

 protected:
  /**
   * @brief Perform a stability analysis on the given wavefunction
   *
   * This method performs stability analysis using the internal implementation.
   * It examines the eigenvalues of the electronic Hessian matrix to determine
   * if the wavefunction corresponds to a true minimum. Negative eigenvalues
   * indicate instabilities.
   *
   * @param wavefunction The wavefunction to analyze for stability
   * @return A pair containing:
   *         - bool: Overall stability status (true if stable, false if
   * unstable)
   *         - std::shared_ptr<data::StabilityResult>: Detailed stability
   * information including eigenvalues and eigenvectors
   *
   * @throws std::runtime_error If the stability analysis fails
   *
   * @see qdk::chemistry::data::Wavefunction
   * @see qdk::chemistry::data::StabilityResult
   */
  std::pair<bool, std::shared_ptr<data::StabilityResult>> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
