// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <macis/asci/determinant_search.hpp>
#include <macis/mcscf/mcscf.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <set>

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {
std::vector<size_t> get_active_indices(const data::Orbitals& orbitals);
}  // namespace detail

macis::MCSCFSettings get_mcscf_settings_(const data::Settings& settings_);
macis::ASCISettings get_asci_settings_(const data::Settings& settings_);

template <typename Func, typename... Args>
auto dispatch_by_norb(size_t norb, Args&&... args) {
  if (norb < 32) {
    return Func::template impl<64>(std::forward<Args>(args)...);
  } else if (norb < 64) {
    return Func::template impl<128>(std::forward<Args>(args)...);
  } else if (norb < 128) {
    return Func::template impl<256>(std::forward<Args>(args)...);
  } else {
    throw std::runtime_error(
        "Function not implemented for more than 127 orbitals");
    return Func::template impl<64>(std::forward<Args>(args)...);
  }
}

/**
 * @class Macis
 * @brief Many-body Adaptive Configuration Interaction Solver implementation
 *
 * The MACIS class provides a concrete implementation of the
 * MultiConfigurationCalculator interface using the MACIS (Many-body Adaptive
 * Configuration Interaction Solver) library. This solver performs configuration
 * interaction calculations on molecular systems with strong electron
 * correlation.
 *
 * Features:
 * - Complete Active Space Configuration Interaction (CASCI) calculations
 * - Adaptive Sampling Configuration Interaction (ASCI) calculations
 *
 * Typical usage:
 * ```
 * // Create a MACIS calculator
 * auto macis =
 * std::make_unique<qdk::chemistry::algorithms::microsoft::Macis>();
 *
 * // Configure settings if needed
 * macis->settings().set("parameter_name", parameter_value);
 *
 * // Perform the calculation
 * auto [energy, wavefunction] = macis->calculate(hamiltonian);
 * ```
 *
 * The calculation automatically adapts to the size of the active space and
 * selects the appropriate internal representation for the wavefunction.
 *
 * @note Currently supports up to 128 orbitals in the active space.
 *
 * @see qdk::chemistry::algorithms::MultiConfigurationCalculator
 * @see qdk::chemistry::data::Hamiltonian
 * @see qdk::chemistry::data::Wavefunction
 * @see qdk::chemistry::data::Settings
 */
class Macis : public qdk::chemistry::algorithms::MultiConfigurationCalculator {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes a MACIS calculator with default settings.
   */
  Macis() { _settings = std::make_unique<MultiConfigurationSettings>(); };

  /**
   * @brief Virtual destructor
   */
  virtual ~Macis() noexcept override = default;

  virtual std::string name() const = 0;

 protected:
  /**
   * @brief Perform a configuration interaction calculation
   *
   * This method performs a configuration interaction calculation using the
   * MACIS library. It dispatches the calculation to the appropriate
   * implementation based on the number of orbitals in the active space.
   *
   * The method extracts the active space orbital indices and occupations from
   * the Hamiltonian, and performs either a CASCI or ASCI calculation based on
   * the settings provided.
   *
   * @param hamiltonian The Hamiltonian containing the molecular integrals and
   *                    orbital information for the calculation.
   * @param n_active_alpha_electrons The number of alpha electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * @param n_active_beta_electrons The number of beta electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   *
   * @return A pair containing the calculated energy and the resulting
   * wavefunction.
   *
   * @throws std::runtime_error if the number of orbitals exceeds 128
   *
   * @see qdk::chemistry::data::Hamiltonian
   * @see qdk::chemistry::data::Wavefunction
   */
  virtual std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const override = 0;
};

}  // namespace qdk::chemistry::algorithms::microsoft
