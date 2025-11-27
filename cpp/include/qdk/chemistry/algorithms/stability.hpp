// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <iostream>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/stability_result.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <tuple>

namespace qdk::chemistry::algorithms {

/**
 * @brief Abstract base class for wavefunction stability checking algorithms
 * with respect to orbital rotation
 *
 * The StabilityChecker class provides a common interface for various stability
 * checking methods used in quantum chemistry. Stability checking examines the
 * second-order response of a wavefunction to determine if it corresponds to a
 * true minimum or if there are directions in which the energy can be lowered.
 *
 * This class uses the Factory design pattern through StabilityCheckerFactory to
 * allow dynamic creation of different stability checking algorithms at runtime.
 *
 * Example usage:
 * @code
 * // Create a concrete stability checker
 * auto checker =
 * qdk::chemistry::algorithms::StabilityCheckerFactory::create("pyscf");
 *
 * // Configure checker settings
 * checker->settings().set("nroots", 5);
 *
 * // Perform stability check
 * auto [is_stable, result] = checker->run(wavefunction);
 *
 * if (is_stable) {
 *   std::cout << "Wavefunction is stable" << std::endl;
 * } else {
 *   std::cout << "Wavefunction is unstable" << std::endl;
 * }
 * auto smallest_eigenvalue = result->get_smallest_eigenvalue();
 * std::cout << "Smallest eigenvalue: " << smallest_eigenvalue << std::endl;
 * @endcode
 */
class StabilityChecker
    : public Algorithm<StabilityChecker,
                       std::pair<bool, std::shared_ptr<data::StabilityResult>>,
                       std::shared_ptr<data::Wavefunction>> {
 public:
  /**
   * @brief Default constructor
   */
  StabilityChecker() = default;

  /**
   * @brief Virtual destructor for proper inheritance
   */
  virtual ~StabilityChecker() = default;

  /**
   * @brief Check the stability of the given wavefunction with respect to
   * orbital rotation
   *
   * This method performs stability analysis on the input wavefunction by
   * examining the eigenvalues of the electronic Hessian matrix. A stable
   * wavefunction should have all non-negative eigenvalues. Near-zero
   * eigenvalues may indicate orbital degeneracy.
   *
   * The specific algorithm (PySCF-based, etc.) is determined by the
   * concrete implementation.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param wavefunction The wavefunction to analyze for stability
   * \endcond
   *
   * @return A pair containing:
   * - bool: Overall stability status (true if stable, false if
   *   unstable)
   * - data::StabilityResult: Detailed stability information including
   *   eigenvalues and eigenvectors of the stability matrix, which can
   *   be used to start a new SCF calculation if an instability is detected.
   *
   * @throws std::runtime_error If stability analysis fails
   * @throws std::invalid_argument If the input wavefunction is invalid for the
   * specified instance of the stability checker
   * @throws qdk::chemistry::data::SettingsAreLocked If attempting to modify
   * settings after run() is called
   *
   * @note Settings are automatically locked when this method is called and
   *       cannot be modified during or after execution.
   * @note The specific requirements for the inputs and settings affecting
   *       the stability check may vary by implementation. See the
   *       documentation for the specific stability checker being used.
   */
  using Algorithm::run;

  std::string type_name() const override { return "stability_checker"; }

  std::string name() const override = 0;

 protected:
  /**
   * @brief Implementation of stability checking
   *
   * This method contains the actual stability checking logic. It is
   * automatically called by run() with identical arguments after settings have
   * been locked.
   *
   * @param wavefunction The wavefunction to analyze for stability
   * @return A pair containing stability status and detailed results
   */
  virtual std::pair<bool, std::shared_ptr<data::StabilityResult>> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const = 0;
};

/**
 * @brief Factory class for creating stability checker instances.
 *
 * This class provides a mechanism to create stability checker instances
 * based on a string key. It allows for easy extension and registration
 * of different stability checking implementations.
 */
struct StabilityCheckerFactory
    : public AlgorithmFactory<StabilityChecker, StabilityCheckerFactory> {
  static std::string algorithm_type_name() { return "stability_checker"; }
  static void register_default_instances() {};
  static std::string default_algorithm_name() { return "pyscf"; }
};

}  // namespace qdk::chemistry::algorithms
