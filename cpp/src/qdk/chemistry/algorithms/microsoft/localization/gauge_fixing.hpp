// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <functional>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <utility>

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

/**
 * @brief Orient one degenerate orbital block reproducibly using AO anchors.
 *
 * Selects the atomic orbitals with the largest residual projection onto the
 * block and symmetrically orthogonalizes the block against them, so that any
 * two orientations of the same subspace map onto identical coordinates. The
 * anchors are taken in ascending atomic-orbital index, so the result does not
 * depend on the order in which the selection finds them.
 *
 * @param block_coefficients AO-by-orbital coefficient matrix for one
 * degenerate block.
 * @param ao_overlap Atomic-orbital overlap matrix.
 * @return Coefficients spanning the same subspace in deterministic
 * coordinates.
 *
 * @throws std::runtime_error if independent AO anchors cannot be found.
 */
Eigen::MatrixXd ao_anchor_block(const Eigen::MatrixXd& block_coefficients,
                                const Eigen::MatrixXd& ao_overlap);

/**
 * @brief Refine a bracketed scalar minimum to an absolute argument tolerance.
 *
 * @param objective Scalar function to minimize.
 * @param lower_bound Inclusive lower end of the bracketing interval.
 * @param upper_bound Inclusive upper end of the bracketing interval.
 * @param argument_tolerance Maximum final interval width.
 * @return The best sampled argument and its objective value.
 *
 * @throws std::invalid_argument if the interval or the tolerance is not
 * positive.
 *
 * @note Golden-section contraction is used rather than a Brent-style
 * minimizer because the coefficient norm is a sum of absolute values whose
 * minima can be cusps, where a relative, machine-epsilon-dependent stopping
 * rule leaves platform-dependent residuals.
 */
std::pair<double, double> golden_section_minimum(
    const std::function<double(double)>& objective, double lower_bound,
    double upper_bound, double argument_tolerance = 1e-13);

}  // namespace detail

/**
 * @class GaugeFixingLocalizerSettings
 * @brief Settings for the gauge-fixing localizer.
 */
class GaugeFixingLocalizerSettings : public data::Settings {
 public:
  GaugeFixingLocalizerSettings() {
    set_default(
        "degeneracy_tolerance", 1e-6,
        "Maximum occupation-number spread within one degenerate block",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default(
        "angle_samples", 32,
        "Uniform samples over [0, pi) used to bracket each plane "
        "rotation",
        data::BoundConstraint<int64_t>{4, std::numeric_limits<int64_t>::max()});
    set_default(
        "max_sweeps", 3,
        "Maximum number of deterministic passes over all rotation planes",
        data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
    set_default(
        "improvement_tolerance", 1e-10,
        "Minimum coefficient-norm reduction, in Hartree, required to "
        "accept a rotation",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default(
        "mapper_threshold", 1e-14,
        "Coefficient and integral threshold used by the qubit mapper "
        "during the search",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
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
   * orbitals in the chosen gauge, and its unchanged spin-traced active 1-RDM.
   *
   * @throws std::invalid_argument if the indices are unsorted, duplicated,
   * differ between spin channels, fall outside the active space, if the
   * degeneracy tolerance is not finite and positive, if the orbitals are
   * unrestricted or lack an overlap matrix, or if the active 1-RDM is
   * unavailable, not real-valued, not square, or not diagonal.
   * @throws std::runtime_error if a degenerate block is only partly selected
   * or AO anchoring fails.
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
