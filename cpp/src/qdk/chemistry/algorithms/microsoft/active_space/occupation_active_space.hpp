// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class OccupationActiveSpaceSettings
 * @brief Settings container for the occupation-based active space selection
 * algorithm
 *
 * This class pre-configures threshold parameters for determining which orbitals
 * should be included in the active space based on their occupation numbers.
 *
 * The occupation-based selection uses orbital occupation numbers to identify
 * orbitals that exhibit partial occupation (not close to 0 or 2 electrons),
 * which typically indicates significant multi-reference character and strong
 * electron correlation.
 *
 * Default settings include:
 * - occupation_threshold: 0.1 - The minimum deviation from integer occupation
 * (0 or 2) for an orbital to be included in the active space
 *
 * @see qdk::chemistry::algorithms::microsoft::OccupationActiveSpaceSelector
 */
class OccupationActiveSpaceSettings : public qdk::chemistry::data::Settings {
 public:
  /**
   * @brief Constructor that initializes default occupation-based active space
   * settings
   *
   * Creates a settings object with the following defaults:
   * - occupation_threshold: 0.1 - Orbitals with occupation deviating from 0 or
   * 2 by at least this amount will be included in the active space
   */
  OccupationActiveSpaceSettings() { set_default("occupation_threshold", 0.1); }
};

/**
 * @class OccupationActiveSpaceSelector
 * @brief Selects active space orbitals based on orbital occupation numbers
 *
 * This class provides a concrete implementation of the ActiveSpaceSelector
 * interface that selects active space orbitals based on their occupation
 * numbers. It identifies orbitals that have partial occupations (not close to 0
 * or 2 electrons).
 *
 * Typical usage:
 * ```cpp
 * // Create an occupation-based active space selector
 * auto selector =
 * qdk::chemistry::algorithms::ActiveSpaceSelectorFactory::create("occupation");
 *
 * // Optionally adjust the occupation threshold
 * selector->settings().set("occupation_threshold", 0.05);
 *
 * // Select the active space orbitals
 * std::vector<unsigned> active_indices =
 * selector->run(orbitals);
 * ```
 *
 * @see qdk::chemistry::algorithms::ActiveSpaceSelector
 * @see qdk::chemistry::data::Orbitals
 * @see OccupationActiveSpaceSettings
 */
class OccupationActiveSpaceSelector : public ActiveSpaceSelector {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes an occupation-based active space selector with default
   * settings.
   */
  OccupationActiveSpaceSelector() {
    _settings = std::make_unique<OccupationActiveSpaceSettings>();
  };

  /**
   * @brief Virtual destructor
   */
  ~OccupationActiveSpaceSelector() override = default;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const final { return "qdk_occupation"; };

 protected:
  /**
   * @brief Select active space orbitals based on occupation numbers
   *
   * This method analyzes the orbital occupation numbers and selects orbitals
   * with occupations that deviate from integer values (0 or 2) by more than
   * the specified threshold.
   *
   * @param wavefunction The input wavefunction from which to select the active
   * space.
   * @return A copy of input orbitals with active space data populated
   *
   * @throws std::runtime_error If the input orbitals already have an active
   * space
   * @throws std::runtime_error If the orbitals are unrestricted
   * @throws std::runtime_error If the system is not closed-shell (alpha !=
   * beta)
   *
   * @see qdk::chemistry::data::Wavefunction
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const override;
};
}  // namespace qdk::chemistry::algorithms::microsoft
