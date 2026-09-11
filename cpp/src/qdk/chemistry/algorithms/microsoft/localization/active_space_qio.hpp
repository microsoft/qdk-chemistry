// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/localization.hpp>
#include <string>

#include "../qio/jacobi_settings.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class ActiveSpaceQIOLocalizerSettings
 * @brief Tunable Jacobi-sweep controls for the active-space QIO localizer.
 */
class ActiveSpaceQIOLocalizerSettings : public qio::JacobiSettings {};

/**
 * @class ActiveSpaceQIOLocalizer
 * @brief Quantum-information orbital (QIO) active-space localizer.
 *
 * Rotates restricted active orbitals to minimize the total single-orbital
 * entropy using gradient-free Jacobi sweeps. The input wavefunction must
 * provide spin-dependent active-space 1- and 2-RDMs.
 *
 * This localizer minimizes the QIO objective restricted to rotations within a
 * fixed active space. It does not implement full-space QIO or QICAS, both of
 * which mix orbitals across the active-space boundary.
 *
 * @see Wavefunction::get_single_orbital_entropies
 */
class ActiveSpaceQIOLocalizer : public Localizer {
 public:
  /**
   * @brief Default constructor
   */
  ActiveSpaceQIOLocalizer() {
    _settings = std::make_unique<ActiveSpaceQIOLocalizerSettings>();
  }

  /**
   * @brief Virtual destructor
   */
  ~ActiveSpaceQIOLocalizer() override = default;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const final { return "qdk_active_space_qio"; }

 protected:
  /**
   * @brief Rotate active orbitals to minimize the single-orbital entropy sum.
   *
   * @param wavefunction Input wavefunction carrying restricted active orbitals
   * and spin-dependent active 1- and 2-RDMs.
   * @param loc_indices_a Sorted alpha orbital indices to transform; must match
   * the active-space alpha indices exactly.
   * @param loc_indices_b Sorted beta orbital indices to transform; must match
   * the active-space beta indices exactly.
   * @return Wavefunction with the active orbitals replaced by the
   * quantum-information-optimized orbitals.
   *
   * @throws std::invalid_argument If the selected indices, orbitals, or RDMs do
   * not satisfy the QIO input requirements.
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
