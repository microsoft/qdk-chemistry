// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/active_space.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class AutocasActiveSpaceSettings
 * @brief Settings container for the autocas active space selection
 * algorithm
 *
 * This class extends the base Settings class to provide specific default
 * settings for the autocas active space selector. It pre-configures
 * parameters for determining which orbitals should be included in the
 * active space based on their single orbital entropies.
 *
 * The entropy-based selection uses single orbitals entropies to identify
 * orbitals that are strongly correlated.
 *
 * Default settings include:
 * - num_bins: 100 - The number of bins to discretize the normalized entropies
 * - min_plateau_size: 10 - The minimum size of a plateau to be considered
 * - entropy_threshold: 0.14 - The minimum absolute entropy value the plateau
 * has to be above
 * - normalize_entropies: true - Whether the entropies should be normalized
 *
 * These settings should be adjusted based on the specific system and
 * application.
 *
 * @see qdk::chemistry::data::Settings
 * @see qdk::chemistry::algorithms::microsoft::AutocasEosActiveSpaceSelector
 */
class AutocasActiveSpaceSettings : public data::Settings {
 public:
  /**
   * @brief Constructor that initializes default occupation-based active space
   * settings
   */
  AutocasActiveSpaceSettings() {
    set_default<int64_t>(
        "num_bins", 100, "Number of bins for entropy discretization",
        data::BoundConstraint<int64_t>{2, std::numeric_limits<int64_t>::max()});
    set_default<int64_t>(
        "min_plateau_size", 10,
        "Minimum size of entropy plateau to be considered",
        data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
    set_default<double>(
        "entropy_threshold", 0.14,
        "Minimum absolute entropy value the plateau has to be above");
    set_default<bool>("normalize_entropies", true);
  }
};

/**
 * @class AutocasActiveSpaceSelector
 * @brief Selects active space orbitals based on single orbital entropies
 *
 * This class provides a concrete implementation of the
 * WavefunctionBasedActiveSpaceSelector interface that selects active space
 * orbitals based on single orbital entropies. It identifies gaps in the sorted
 * entropies to determine plateaus of strongly correlated orbitals.
 *
 * Typical usage:
 * ```cpp
 * // Create an autocas active space selector
 * auto selector =
 * qdk::chemistry::algorithms::WavefunctionActiveSpaceSelectorFactory::create("autocas");
 *
 * // Optionally adjust the occupation threshold
 * selector->settings().set("min_plateau_size", 9);
 *
 * // Select the active space orbitals
 * std::vector<unsigned> active_indices =
 * selector->run(wavefunction);
 * ```
 *
 * @see qdk::chemistry::algorithms::WavefunctionBasedActiveSpaceSelector
 * @see qdk::chemistry::data::Orbitals
 * @see AutocasActiveSpaceSettings
 */
class AutocasActiveSpaceSelector : public ActiveSpaceSelector {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes an autocas active space selector with default
   * settings.
   */
  AutocasActiveSpaceSelector() {
    _settings = std::make_unique<AutocasActiveSpaceSettings>();
  }

  /**
   * @brief Virtual destructor
   */
  ~AutocasActiveSpaceSelector() override = default;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const final { return "qdk_autocas"; };

 protected:
  /**
   * @brief Select active space orbitals based on single orbital entropies
   *
   * This method analyzes the single orbital entropies and selects orbitals
   * that form plateaus in the sorted entropy values, indicating strong
   * correlation.
   *
   * @param wavefunction The wavefunction containing orbital entropies
   * @return A copy of the input wavefunction with active space data populated
   *
   * @throws std::runtime_error number of entropies does not match number of
   * active space orbitals
   * @throws std::runtime_error active indices are not the same for alpha and
   * beta orbitals
   * @throws std::runtime_error if maximum entropy is zero
   * @throws std::runtime_error if wavefunction does not have single orbital
   *
   * @see qdk::chemistry::data::Wavefunction
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const override;
};
}  // namespace qdk::chemistry::algorithms::microsoft
