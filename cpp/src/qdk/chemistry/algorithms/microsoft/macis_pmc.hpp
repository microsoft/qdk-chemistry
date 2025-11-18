// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <macis/asci/determinant_search.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <qdk/chemistry/algorithms/pmc.hpp>

#include "macis_base.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class MacisPmcSettings
 * @brief Settings class specific to MACIS PMC calculations
 *
 * This class extends the base ProjectedMultiConfigurationSettings class with
 * parameters specific to Projected Multi-Configuration (PMC) calculations using
 * the MACIS library. It provides default values for PMC-specific settings such
 * as determinant limits, tolerances, and algorithm control parameters.
 *
 * @see ProjectedMultiConfigurationSettings
 */
class MacisPmcSettings : public ProjectedMultiConfigurationSettings {
 public:
  /**
   * @brief Default constructor
   *
   * Creates PMC settings object with default parameter values taken directly
   * from the MACIS library's ASCISettings struct to ensure consistency.
   *
   * iterative_solver_dimension_cutoff: matrix size cutoff for using iterative
   * eigensolver. If the number of determinants is below this value, dense
   * diagonalization is used instead.
   * H_thresh: Hamiltonian matrix entries threshold for dense diagonalization
   * h_el_tol: electron interaction tolerance, used for Hamiltonian-wavefunction
   * product in iterative solver
   * davidson_res_tol: Residual tolerance for Davidson solver convergence
   * davidson_max_m: Maximum subspace size for Davidson solver
   */
  MacisPmcSettings() {
    // Use MACIS library defaults directly
    macis::ASCISettings macis_defaults;

    // Tolerance parameters
    set_default<size_t>("iterative_solver_dimension_cutoff", 100);
    set_default<double>("H_thresh", 1e-16);
    set_default<double>("h_el_tol", macis_defaults.h_el_tol);
    set_default<double>("davidson_res_tol", 1e-8);
    set_default<size_t>("davidson_max_m", 200);
  }

  /**
   * @brief Virtual destructor
   */
  virtual ~MacisPmcSettings() = default;
};

/**
 * @class MacisPmc
 * @brief MACIS-based Projected Multi-Configuration calculator
 *
 * This class implements projected multi-configuration calculations using the
 * MACIS library. It performs projections of the Hamiltonian onto a specified
 * set of determinants to compute energies and wavefunctions for strongly
 * correlated molecular systems.
 *
 * The calculator inherits from ProjectedMultiConfigurationCalculator and uses
 * MACIS library routines to perform the actual projected calculations where
 * the determinant space is provided as input rather than generated adaptively.
 */
class MacisPmc : public ProjectedMultiConfigurationCalculator {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes a MACIS PMC calculator with default settings.
   */
  MacisPmc() { _settings = std::make_unique<MacisPmcSettings>(); };

  /**
   * @brief Virtual destructor
   */
  ~MacisPmc() noexcept override = default;

  virtual std::string name() const override { return "macis_pmc"; }

 protected:
  /**
   * @brief Implementation of projected multi-configuration calculation
   *
   * This method performs a projected multi-configuration calculation using the
   * MACIS library. It projects the Hamiltonian onto a specified set of
   * determinants and dispatches the calculation to the appropriate
   * implementation based on the number of orbitals in the active space.
   *
   * The method extracts the active space orbital indices and occupations from
   * the Hamiltonian, and performs a projected MC calculation based on
   * the settings provided.
   *
   * @param hamiltonian The Hamiltonian containing the molecular integrals and
   *                    orbital information for the calculation.
   * @param configurations The set of configurations/determinants to project the
   *                       Hamiltonian onto.
   * @return A pair containing the calculated energy and the resulting
   * wavefunction.
   *
   * @throws std::runtime_error if the number of orbitals exceeds 128
   *
   * @see qdk::chemistry::data::Hamiltonian
   * @see qdk::chemistry::data::Wavefunction
   * @see qdk::chemistry::data::Configuration
   */
  std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      const std::vector<data::Configuration>& configurations) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
