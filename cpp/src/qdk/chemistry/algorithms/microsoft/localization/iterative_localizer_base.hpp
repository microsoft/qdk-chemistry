// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Core>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class IterativeOrbitalLocalizationSettings
 * @brief Base class for all orbital localization algorithms
 *
 * Contains common convergence parameters and general settings used by
 * all localization methods.
 */
class IterativeOrbitalLocalizationSettings : public data::Settings {
 public:
  IterativeOrbitalLocalizationSettings() {
    set_default("tolerance", 1e-6);
    set_default("max_iterations", size_t(10000));
    set_default("small_rotation_tolerance", 1e-12);
  }
};

/**
 * @class OrbitalLocalizationScheme
 * @brief Base class for orbital localization algorithms
 *
 * This abstract base class defines the interface for all iterative orbital
 * localization schemes in the QDK framework. Orbital localization transforms
 * delocalized molecular orbitals into spatially localized orbitals, which can
 * improve interpretability and computational efficiency for certain methods.
 *
 * Derived classes implement specific localization algorithms such as
 * Pipek-Mezey.
 */
class IterativeOrbitalLocalizationScheme {
 protected:
  /** @brief Settings for the localization algorithm */
  IterativeOrbitalLocalizationSettings settings_;

  /** @brief Flag indicating whether the localization algorithm converged */
  bool converged_ = false;

  /** @brief Value of the objective function being optimized */
  double obj_fun_ = 0;

 public:
  /** @brief Type alias for function argument vectors */
  using argument_type = Eigen::VectorXd;

  /** @brief Type alias for function return values */
  using return_type = double;

  /**
   * @brief Constructor for the OrbitalLocalizationScheme
   *
   * @param settings Configuration settings for the localization algorithm
   */
  IterativeOrbitalLocalizationScheme(
      IterativeOrbitalLocalizationSettings settings);

  /**
   * @brief Perform the orbital localization
   *
   * This pure virtual method must be implemented by derived classes to
   * perform the actual localization of the molecular orbitals according
   * to the specific algorithm.
   *
   * @param initial_orbitals Matrix of initial molecular orbital coefficients to
   *                         be localized
   * @return Matrix of localized molecular orbital coefficients
   */
  virtual Eigen::MatrixXd localize(const Eigen::MatrixXd& initial_orbitals) = 0;

  /**
   * @brief Check if the localization algorithm converged
   *
   * @return True if the algorithm converged, false otherwise
   */
  auto converged() const { return converged_; }

  /**
   * @brief Get the final value of the objective function
   *
   * @return Value of the objective function after localization
   */
  auto obj_fun() const { return obj_fun_; }

  /**
   * @brief Virtual destructor
   */
  virtual ~IterativeOrbitalLocalizationScheme() noexcept = default;
};
}  // namespace qdk::chemistry::algorithms::microsoft
