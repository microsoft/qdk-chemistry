// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/pmc.hpp>

#include "macis_base.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class MacisPmcSettings
 * @brief Settings class for MACIS PMC calculations.
 *
 * Inherits all solver settings from MacisSettings.
 *
 * @see MacisSettings
 */
class MacisPmcSettings : public MacisSettings {
 public:
  MacisPmcSettings() = default;

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
