// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/structure.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class ValenceActiveSpaceSettings
 * @brief Settings container for the valence-based active space selection
 * algorithm
 *
 * This class defines settings for specifying the number of active electrons
 * and active orbitals.
 *
 * Default settings:
 * - num_active_electrons: -1 (invalid default, must be set by user)
 * - num_active_orbitals: -1 (invalid default, must be set by user)
 */
class ValenceActiveSpaceSettings : public qdk::chemistry::data::Settings {
 public:
  /**
   * @brief Constructor initializing default valence active space settings
   *
   * Sets initial values for:
   * - num_active_electrons: -1 (invalid, must be set by user)
   * - num_active_orbitals: -1 (invalid, must be set by user)
   */
  ValenceActiveSpaceSettings() {
    set_default("num_active_electrons", -1);
    set_default("num_active_orbitals", -1);
  }
};

/**
 * @class ValenceActiveSpaceSelector
 * @brief Selects active space orbitals based on specified number of electrons
 * and orbitals
 *
 * This class implements the ActiveSpaceSelector interface to create an active
 * space consisting of a specified number of orbitals with a given number of
 * electrons. It selects a contiguous block of orbitals starting after the
 * inactive orbitals.
 *
 * The selector requires:
 * - Number of active electrons
 * - Number of active orbitals
 *
 * Restrictions:
 * - Only supports restricted orbitals
 * - Number of inactive electrons must be even
 * - Number of active electrons must be positive
 * - Number of active orbitals must be positive and cannot exceed total orbitals
 */
class ValenceActiveSpaceSelector : public ActiveSpaceSelector {
 public:
  /**
   * @brief Default constructor
   */
  ValenceActiveSpaceSelector() {
    _settings = std::make_unique<ValenceActiveSpaceSettings>();
  };

  /**
   * @brief Virtual destructor
   */
  ~ValenceActiveSpaceSelector() override = default;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const final { return "qdk_valence"; };

 protected:
  /**
   * @brief Select active space orbitals based on configured parameters
   *
   * Selects a contiguous block of orbitals to form the active space based on
   * the number of active electrons and orbitals specified in the settings.
   * The active space starts immediately after the inactive orbitals.
   * (If no active space is pre-defined in the input orbitals, all orbitals
   * considered active.)
   *
   * @param wavefunction The input wavefunction from which to select the active
   * space.
   * @return A copy of input wavefunction with active space selected
   *
   * @throws std::runtime_error If orbitals/wavefunction are unrestricted
   * @throws std::runtime_error If number of active electrons is not positive
   * @throws std::runtime_error If number of active orbitals is not positive
   * @throws std::runtime_error If number of active electrons exceeds total
   * electrons
   * @throws std::runtime_error If number of inactive electrons is odd
   * @throws std::runtime_error If number of active orbitals exceeds total
   * orbitals
   * @throws std::runtime_error If number of active orbitals exceeds number of
   * pre- existing candidate orbitals
   *
   * @throws std::runtime_error If sum of inactive and active orbitals exceeds
   * total orbitals
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
