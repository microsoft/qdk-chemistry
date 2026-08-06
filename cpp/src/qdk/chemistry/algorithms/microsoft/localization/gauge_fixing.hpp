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
 * two orientations of the same subspace map onto identical coordinates.
 *
 * @param block AO-by-orbital coefficient matrix for one degenerate block.
 * @param overlap Atomic-orbital overlap matrix.
 * @return Coefficients spanning the same subspace in deterministic
 * coordinates.
 *
 * @throws std::runtime_error if independent AO anchors cannot be found.
 */
Eigen::MatrixXd ao_anchor_block(const Eigen::MatrixXd& block,
                                const Eigen::MatrixXd& overlap);

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
    set_default("degeneracy_tolerance", 1e-6,
                "Occupation-number gap below which orbitals share one "
                "degenerate block");
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
    set_default("mapper_threshold", 1e-14,
                "Coefficient and integral threshold used by the qubit mapper "
                "during the search");
  }
};

/**
 * @brief Fix the orbital gauge inside occupation-degenerate natural-orbital
 * blocks.
 *
 * Natural orbitals with equal occupations span a well-defined subspace but do
 * not have unique orbital vectors. Any orthogonal rotation inside such a
 * degenerate block spans the same subspace and leaves the exact CASCI energy
 * unchanged, yet it yields a different qubit Hamiltonian after mapping. This
 * localizer resolves that freedom deterministically: it anchors every selected
 * degenerate block to the atomic-orbital basis, then runs coordinate-descent
 * sweeps over Givens plane rotations within each block, accepting only
 * rotations that reduce the mapped coefficient norm lambda = sum_l |h_l|.
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
   * @param loc_indices_a Sorted alpha orbital indices to gauge fix; must be a
   * subset of the active-space indices.
   * @param loc_indices_b Sorted beta orbital indices; must equal
   * @p loc_indices_a.
   * @return Wavefunction whose active space is the selected orbitals in the
   * chosen gauge, carrying the corresponding block of the spin-traced 1-RDM.
   *
   * @throws std::invalid_argument if the indices are unsorted, differ between
   * spin channels, fall outside the active space, if the orbitals are
   * unrestricted or lack an overlap matrix, or if the active 1-RDM is
   * unavailable, not real-valued, or not diagonal.
   * @throws std::runtime_error if a degenerate block is only partly selected
   * or AO anchoring fails.
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
