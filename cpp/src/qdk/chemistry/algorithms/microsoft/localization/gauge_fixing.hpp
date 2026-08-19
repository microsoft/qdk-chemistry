// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class GaugeFixingLocalizerSettings
 * @brief Settings for the gauge-fixing localizer.
 */
class GaugeFixingLocalizerSettings : public data::Settings {
 public:
  GaugeFixingLocalizerSettings() {
    set_default("degeneracy_tolerance", 1e-6,
                "Maximum occupation-number spread within one degenerate block");
    set_default(
        "angle_samples", 32,
        "Uniform samples over [0, pi) used to bracket each plane "
        "rotation",
        data::BoundConstraint<int64_t>{4, std::numeric_limits<int64_t>::max()});
    set_default(
        "max_sweeps", 3,
        "Maximum number of deterministic passes over all rotation planes",
        data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
    set_default("improvement_tolerance", 1e-10,
                "Minimum coefficient-norm reduction, in Hartree, required to "
                "accept a rotation");
  }
};

/**
 * @brief Fix the orbital gauge inside occupation-degenerate natural-orbital
 * blocks.
 *
 * Natural orbitals with equal occupations span a well-defined subspace but do
 * not have unique orbital vectors. Any orthogonal rotation within the active
 * space leaves the exact CASCI energy unchanged, yet it yields a different
 * qubit Hamiltonian after mapping. Restricting the rotations to
 * occupation-degenerate blocks additionally keeps the spin-traced 1-RDM
 * diagonal with the same occupations, so the returned orbitals remain natural
 * orbitals and the input 1-RDM stays truthful for them. This localizer
 * resolves that freedom deterministically: it anchors every selected
 * degenerate block to the atomic-orbital basis, then runs coordinate-descent
 * sweeps over Givens plane rotations within each block, accepting only
 * rotations that reduce the mapped coefficient norm lambda = sum_l |h_l| of
 * the active-space Hamiltonian.
 *
 * Because the rotations stay inside degenerate blocks, the returned orbitals
 * remain natural orbitals; their occupations are unchanged and are carried on
 * the returned wavefunction. Choosing a gauge is a separate objective from
 * finding the natural orbitals, so the two are composed rather than combined:
 * run @ref NaturalOrbitalLocalizer first.
 */
class GaugeFixingLocalizer : public Localizer {
 public:
  GaugeFixingLocalizer() {
    _settings = std::make_unique<GaugeFixingLocalizerSettings>();
  }
  ~GaugeFixingLocalizer() override = default;
  std::string name() const final { return "qdk_gauge_fixing"; }

 protected:
  /**
   * @brief Choose the coefficient-norm-minimizing gauge for the selected
   * degenerate orbital blocks.
   *
   * @param wavefunction Wavefunction whose orbitals diagonalize its
   * spin-traced active 1-RDM.
   * @param loc_indices_a Sorted, duplicate-free alpha orbital indices to gauge
   * fix; must be a subset of the active-space indices, and selecting a proper
   * subset restricts which blocks are swept without narrowing the objective.
   * @param loc_indices_b Sorted beta orbital indices; must equal
   * @p loc_indices_a.
   * @return Wavefunction carrying the input active space with the selected
   * orbitals in the chosen gauge, and its active 1-RDM in that gauge.
   *
   * @throws std::invalid_argument if the indices are unsorted, duplicated,
   * differ between spin channels, fall outside the active space, if any
   * tolerance setting is not finite, if the orbitals are unrestricted or lack
   * an overlap matrix, or if the active 1-RDM is unavailable, not real-valued,
   * not square, or not diagonal.
   * @throws std::runtime_error if a degenerate block is only partly selected
   * or AO anchoring fails.
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
