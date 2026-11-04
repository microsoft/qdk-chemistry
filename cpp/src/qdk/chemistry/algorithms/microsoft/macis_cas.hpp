// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <limits>
#include <macis/asci/determinant_search.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>

#include "macis_base.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class MacisCasSettings
 * @brief Settings class for MACIS CAS calculations.
 *
 * Inherits MCSCF defaults from MacisSettings and adds settings for the
 * spin-flip symmetry and the Davidson search space.
 */
class MacisCasSettings : public MacisSettings {
 public:
  MacisCasSettings() {
    set_default<std::string>(
        "spin_flip_symmetry", "none",
        "Restrict the solution to even or odd total spin",
        data::ListConstraint<std::string>{
            {std::vector<std::string>{"none", "even", "odd"}}});
    set_default<int64_t>(
        "max_subspace", 10,
        "Maximum number of vectors in the Davidson search space",
        data::BoundConstraint<int64_t>{2, std::numeric_limits<int64_t>::max()});
  }
  ~MacisCasSettings() override = default;
};

class MacisCas : public Macis {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes a MACIS calculator with default settings.
   */
  MacisCas() { _settings = std::make_unique<MacisCasSettings>(); };

  ~MacisCas() noexcept override = default;

  virtual std::string name() const override { return "macis_cas"; }

 protected:
  /**
   * @brief Perform a configuration interaction calculation.
   *
   * This method performs a configuration interaction calculation using the
   * MACIS library. It dispatches the calculation to the appropriate
   * implementation based on the number of orbitals in the active space.
   *
   * The method extracts the active space orbital indices and occupations from
   * the Hamiltonian, and performs a CASCI calculation based on
   * the settings provided.
   *
   * @param hamiltonian The Hamiltonian containing the molecular integrals and
   *                    orbital information for the calculation.
   * @param n_active_alpha_electrons The number of alpha electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * @param n_active_beta_electrons The number of beta electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * @return A pair containing the calculated energy and the resulting
   * wavefunction.
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
