// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/mc.hpp>

#include "macis_base.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Helper struct for CASCI calculation dispatch
 */
struct asci_helper {
  /// Maximum total electron count for which the residue arrays algorithm is
  /// feasible.  Beyond this limit the O(n_e^2) per-determinant memory cost
  /// becomes prohibitive and the code falls back to sorted_double_loop.
  static constexpr size_t residual_array_electron_num_limit = 60;
  using return_type = std::pair<double, data::Wavefunction>;

  /**
   * @brief Template implementation of CASCI calculation
   * @tparam N Number of bits for wavefunction representation
   * @param hamiltonian Hamiltonian object containing molecular integrals
   * @param settings_ Settings object storing asci specific settings
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @return std::pair containing energy and wavefunction
   */
  template <size_t N>
  static return_type impl(const data::Hamiltonian& hamiltonian,
                          const data::Settings& settings_, unsigned int nalpha,
                          unsigned int nbeta);
};

/**
 * @class MacisAsciSettings
 * @brief Settings class specific to MACIS ASCI calculations
 *
 * This class extends the base MultiConfigurationSettings class with parameters
 * specific to Adaptive Sampling Configuration Interaction (ASCI) calculations.
 * It provides default values for ASCI-specific settings such as determinant
 * limits, tolerances, and algorithm control parameters.
 *
 * @see MultiConfigurationSettings
 */
class MacisAsciSettings : public MacisSettings {
 public:
  /**
   * @brief Default constructor
   *
   * Creates ASCI settings object with default parameter values taken directly
   * from the MACIS library's ASCISettings struct to ensure consistency.
   */
  MacisAsciSettings();

  /**
   * @brief Virtual destructor
   */
  virtual ~MacisAsciSettings() = default;
};

class MacisAsci : public Macis {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes a MACIS calculator with default settings.
   */
  MacisAsci() { _settings = std::make_unique<MacisAsciSettings>(); };

  ~MacisAsci() noexcept override = default;

  virtual std::string name() const override { return "macis_asci"; }

 protected:
  /**
   * @brief Perform a configuration interaction calculation
   *
   * This method performs a adaptive sampling configuration interaction
   * calculation using the MACIS library. It dispatches the calculation
   * to the appropriate implementation based on the number of orbitals
   * in the active space.
   *
   * The method extracts the active space orbital indices and occupations from
   * the Hamiltonian, and performs a ASCI calculation based on
   * the settings provided.
   *
   * @param hamiltonian The Hamiltonian containing the molecular integrals and
   *                    orbital information for the calculation
   * @param n_active_alpha_electrons The number of alpha electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * @param n_active_beta_electrons The number of beta electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * @return A pair containing the calculated energy and the resulting
   * wavefunction
   *
   * @throws std::runtime_error if the number of orbitals exceeds 128
   *
   * @see qdk::chemistry::data::Hamiltonian
   * @see qdk::chemistry::data::Wavefunction
   */
  std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
