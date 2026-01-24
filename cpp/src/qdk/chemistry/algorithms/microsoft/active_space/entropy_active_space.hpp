// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/active_space.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class AutocasEosSettings
 * @brief Settings container for the entropy-based active space selection
 * algorithm
 *
 * This class extends the base Settings class to provide specific default
 * settings for the entropy-based active space selector. It pre-configures
 * threshold parameters for determining which orbitals should be included in the
 * active space based on their single orbital entropies.
 *
 * The entropy-based selection uses single orbitals entropies to identify
 * orbitals that are strongly correlated.
 *
 * Default settings include:
 * - normalize_entropies: true - Whether to normalize entropies by the maximum
 * entropy
 * - diff_threshold: 0.1 - The difference between two entropies to form a
 * plateau
 * - entropy_threshold: 0.14 - The minimum absolute entropy value the plateau
 * has to be above
 *
 * @see qdk::chemistry::data::Settings
 * @see qdk::chemistry::algorithms::microsoft::AutocasEosActiveSpaceSelector
 */
class AutocasEosSettings : public data::Settings {
 public:
  /**
   * @brief Constructor that initializes default occupation-based active space
   * settings
   *
   */
  AutocasEosSettings() {
    set_default<bool>("normalize_entropies", true);
    set_default<double>("diff_threshold", 0.1);
    set_default<double>("entropy_threshold", 0.14);
  }
};

/**
 * @class AutocasEosActiveSpaceSelector
 * @brief Selects active space orbitals based on single orbital entropies
 *
 * This class provides a concrete implementation of the
 * WavefunctionBasedActiveSpaceSelector interface that selects active space
 * orbitals based on single orbital entropies. It identifies gaps in the sorted
 * entropies to determine plateaus of strongly correlated orbitals. Unlike the
 * standard AutoCAS algorithm which uses histogram-based discretization to find
 * plateaus, this variant directly compares consecutive entropy differences
 * against a threshold.
 *
 * Typical usage:
 * ```cpp
 * // Create an entropy-based active space selector
 * auto selector =
 * qdk::chemistry::algorithms::WavefunctionBasedActiveSpaceSelectorFactory::create("entropy");
 *
 * // Optionally adjust the occupation threshold
 * selector->settings().set("diff_threshold", 0.1);
 *
 * // Select the active space orbitals
 * Orbitals new_orbitals =
 * selector->run(wavefunction);
 * ```
 *
 * @see qdk::chemistry::algorithms::WavefunctionBasedActiveSpaceSelector
 * @see qdk::chemistry::data::Orbitals
 * @see AutocasEosSettings
 */
class AutocasEosActiveSpaceSelector : public ActiveSpaceSelector {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes an entropy-based active space selector with default
   * settings.
   */
  AutocasEosActiveSpaceSelector() {
    _settings = std::make_unique<AutocasEosSettings>();
  }
  /**
   * @brief Virtual destructor
   */
  ~AutocasEosActiveSpaceSelector() override = default;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const final { return "qdk_autocas_eos"; };

 protected:
  /**
   * @brief Select active space orbitals based on single orbital entropies.
   *
   * This method analyzes the single orbital entropies and selects orbitals
   * that form plateaus in the sorted entropy values, indicating strong
   * correlation.
   *
   * @param wavefunction The wavefunction containing orbital entropies
   * @return A copy of input orbitals with active space data populated
   *
   * @throws std::runtime_error number of entropies does not match number of
   * active space orbitals
   * @throws std::runtime_error active indices are not the same for alpha and
   * beta orbitals
   * @throws std::runtime_error if maximum entropy is zero
   * @throws std::runtime_error if wavefunction does not have single orbital
   *
   *
   * @see qdk::chemistry::data::Wavefunction
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const override;
};
}  // namespace qdk::chemistry::algorithms::microsoft
