// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cstdint>
#include <limits>
#include <numbers>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class ActiveSpaceQIOLocalizerSettings
 * @brief Tunable Jacobi-sweep controls for the active-space QIO localizer.
 */
class ActiveSpaceQIOLocalizerSettings : public data::Settings {
 public:
  ActiveSpaceQIOLocalizerSettings() {
    set_default(
        "max_cycles", int64_t{200},
        "Maximum number of Jacobi sweeps over all active orbital pairs",
        data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
    set_default(
        "convergence_tolerance", 1e-10,
        "Sweep-to-sweep single-orbital entropy-sum change below which the "
        "optimization stops",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default(
        "coarse_angle_step", 0.02,
        "Coarse grid spacing (radians) for the per-pair angle scan over "
        "[0, pi/2); practical range [1e-4, pi/2]",
        data::BoundConstraint<double>{1e-4, std::numbers::pi / 2.0});
    set_default("fine_samples", int64_t{201},
                "Number of samples in the fine-refinement angle scan",
                data::BoundConstraint<int64_t>{
                    4, static_cast<int64_t>(std::numeric_limits<int>::max())});
    set_default(
        "improvement_tolerance", 1e-12,
        "Minimum single-orbital entropy decrease required to accept a "
        "pair rotation",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
  }
};

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
