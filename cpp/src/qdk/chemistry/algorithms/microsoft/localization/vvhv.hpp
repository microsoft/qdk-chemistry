// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/data/orbitals.hpp>

#include "iterative_localizer_base.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class VVHVLocalizerSettings
 * @brief Settings class for VVHV localization
 *
 * Contains configuration parameters specific to Valence Virtual - Hard Virtual
 * (VV-HV) localization, including all parameters from the base iterative
 * localization settings plus any VVHV-specific options.
 */
class VVHVLocalizerSettings : public IterativeOrbitalLocalizationSettings {
 public:
  VVHVLocalizerSettings() {
    set_default("minimal_basis", std::string("sto-3g"));
    set_default("weighted_orthogonalization", true);
  }
};

/**
 * @brief Valence Virtual - Hard Virtual (VV-HV) implementation of the Localizer
 * interface.
 *
 * This class provides a concrete implementation of the Localizer interface
 * using the VV-HV localization algorithm.
 *
 * The VV-HV algorithm partitions virtual orbitals into valence virtuals (VVs)
 * and hard virtuals (HVs) based on projection onto a minimal basis, then
 * localizes each space separately using Pipek-Mezey localization.
 *
 * Implementation based on:
 *   Subotnik et al. JCP 123, 114108 (2005)
 *   Wang et al. JCTC 21, 1163 (2025)
 *
 * @note This localizer requires all orbital indices to be covered in the
 * localize call.
 */
class VVHVLocalizer : public qdk::chemistry::algorithms::Localizer {
 public:
  VVHVLocalizer() { _settings = std::make_unique<VVHVLocalizerSettings>(); };

  /**
   * @brief Localize orbitals using the VV-HV algorithm.
   *
   * This method performs VV-HV localization on the input orbitals.
   * It requires that all orbital indices are covered (loc_indices_a and
   * loc_indices_b must span all orbitals).
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param orbitals The orbitals to localize
   * @param loc_indices_a Indices of alpha orbitals (must cover all orbitals,
   * must be sorted)
   * @param loc_indices_b Indices of beta orbitals (must cover all orbitals,
   * must be sorted)
   * \endcond
   *
   * @return The localized molecular orbitals
   * @throws std::invalid_argument If loc_indices_a or loc_indices_b are not
   * sorted
   * @throws std::invalid_argument If not all orbital indices are covered
   * @throws std::invalid_argument If restricted orbitals have different
   * alpha/beta indices
   * @throws std::runtime_error If basis set has linear dependence issues
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;

  std::string name() const override { return "qdk_vvhv"; }
};

}  // namespace qdk::chemistry::algorithms::microsoft
