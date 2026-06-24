// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/localization.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Transform the active orbitals to natural orbitals.
 *
 * For restricted inputs, the localizer diagonalizes the spin-traced active
 * 1-RDM. For unrestricted inputs, it builds the spin-summed active AO density
 * from the alpha and beta 1-RDM blocks and projects it into the active alpha
 * orbital subspace before diagonalization. Natural orbitals form a single
 * orbital set, so alpha and beta selection indices must match the active-space
 * indices exactly; the result is always restricted.
 */
class NaturalOrbitalLocalizer : public Localizer {
 public:
  NaturalOrbitalLocalizer() = default;
  ~NaturalOrbitalLocalizer() override = default;
  virtual std::string name() const final { return "qdk_natural_orbitals"; }

 protected:
  /**
   * @brief Build natural orbitals for the active orbital space.
   *
   * @param wavefunction Input wavefunction carrying orbitals and an active
   * 1-RDM.
   * @param loc_indices_a Sorted alpha orbital indices to transform; must match
   * the active-space alpha indices exactly.
   * @param loc_indices_b Sorted beta orbital indices to transform; must match
   * the active-space beta indices exactly.
   * @return Wavefunction with active orbitals replaced by natural orbitals.
   *
   * @throws std::invalid_argument if the selected indices are invalid, the
   * active space or overlap matrix is unavailable, or the required active 1-RDM
   * is unavailable or not real-valued.
   * @throws std::runtime_error if diagonalizing the selected 1-RDM fails.
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
