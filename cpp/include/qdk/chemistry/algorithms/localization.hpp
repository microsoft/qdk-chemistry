// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <functional>
#include <iostream>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <vector>

namespace qdk::chemistry::algorithms {

/**
 * @brief Abstract base class for orbital localization algorithms
 *
 * The Localizer class provides a common interface for various orbital
 * localization methods used in quantum chemistry. Orbital localization
 * transforms canonical molecular orbitals (which are typically delocalized
 * across the entire molecule) into localized orbitals that are confined
 * subject to some metric (spatial, bond, etc.).
 *
 * This class uses the Factory design pattern through LocalizerFactory to
 * allow dynamic creation of different localization algorithms at runtime.
 *
 * Example usage:
 * @code
 * // Create a concrete localizer (e.g., BoysLocalizer)
 * auto localizer =
 * qdk::chemistry::algorithms::LocalizerFactory::create_localizer(
 *   "mp2_natural_orbitals");
 * // Create indices for all orbitals
 * auto all_indices = wavefunction->orbitals->get_all_mo_indices();
 * auto localized_orbital_wavefunction = localizer->run(wavefunction,
 *   all_indices, all_indices);
 * @endcode
 */
class Localizer
    : public Algorithm<Localizer, std::shared_ptr<data::Wavefunction>,
                       std::shared_ptr<data::Wavefunction>,
                       const std::vector<size_t>&, const std::vector<size_t>&> {
 public:
  /**
   * @brief Default constructor
   */
  Localizer() = default;

  /**
   * @brief Virtual destructor for proper inheritance
   */
  virtual ~Localizer() = default;

  /**
   * @brief Localize the given molecular orbitals
   *
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param orbitals The orbitals to localize
   * @param loc_indices_a Indices of alpha orbitals to localize (must be sorted,
   * if empty, no alpha orbitals are localized)
   * @param loc_indices_b Indices of beta orbitals to localize (must be sorted,
   * if empty, no beta orbitals are localized)
   * \endcond
   *
   * @return The localized molecular orbitals with updated coefficients.
   *
   * @throws std::runtime_error If localization fails
   * @throws std::invalid_argument If the input orbitals are invalid for the
   *                               specified instance of the localizer
   * @throws SettingsAreLocked if attempting to modify settings after
   *                           run() is called
   * @throws std::invalid_argument If loc_indices_a or loc_indices_b are not
   * sorted
   * @throws SettingsAreLocked if attempting to modify settings after run() is
   * called
   *
   * @note Settings are automatically locked when this method is called and
   * cannot be modified during or after execution.
   * @note The specific requirements for the inputs and settings affecting
   * this method may vary by implementation. See the documentation for
   * the specific localizer being used (docstring).
   * @note The number of electrons is an input for the
   * MP2NaturalOrbitalLocalizer and VVHVLocalizer. Orbital indices <
   * n_alpha_electrons/n_beta_electrons are treated as occupied, indices >=
   * n_alpha_electrons/n_beta_electrons are treated as virtual.
   * @note For restricted orbitals, loc_indices_a and loc_indices_b must be
   * identical
   * @note All localizer implementations require sorted index arrays for
   * performance and consistency reasons
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
  std::string type_name() const final { return "orbital_localizer"; };

 protected:
  /**
   * @brief Implementation of orbital localization
   *
   * This method contains the actual localization logic. It is automatically
   * called by run() after settings have been locked.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param orbitals The orbitals to localize
   * @param loc_indices_a Indices of alpha orbitals to localize (must be sorted)
   * @param loc_indices_b Indices of beta orbitals to localize (must be sorted)
   * \endcond
   * @return The localized orbitals
   */
  virtual std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const = 0;
};

/**
 * @brief Factory class for creating localizer instances.
 *
 * This class provides a mechanism to create localizer instances
 * based on a string key. It allows for easy extension and registration
 * of different localization implementations.
 */
struct LocalizerFactory : public AlgorithmFactory<Localizer, LocalizerFactory> {
  static std::string algorithm_type_name() { return "orbital_localizer"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_pipek_mezey"; }
};

}  // namespace qdk::chemistry::algorithms
