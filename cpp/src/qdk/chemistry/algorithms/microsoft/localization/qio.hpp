// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cstdint>
#include <limits>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class QIOLocalizerSettings
 * @brief Tunable Jacobi-sweep controls for the QIO localizer.
 *
 * - "max_cycles" (int): maximum number of Jacobi sweeps over all active
 *   orbital pairs (default 200, must be >= 1).
 * - "convergence_tolerance" (double): sweep-to-sweep change in the
 *   single-orbital entropy sum below which the optimization stops
 *   (default 1e-10, must be >= 0).
 * - "coarse_angle_step" (double): coarse grid spacing in radians for the
 *   per-pair angle scan over [0, pi) (default 0.02, must be in [1e-4, pi]).
 *
 * The numeric bounds are enforced at set-time, so out-of-range values are
 * rejected (e.g. a negative max_cycles that would underflow to a huge size_t,
 * or a coarse_angle_step outside [1e-4, pi] that would make the scan run
 * forever or take pathologically many iterations).
 */
class QIOLocalizerSettings : public data::Settings {
 public:
  QIOLocalizerSettings() {
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
        "[0, pi); practical range [1e-4, pi]",
        data::BoundConstraint<double>{1e-4, 3.14159265358979323846});
  }
};

/**
 * @class QIOLocalizer
 * @brief Quantum-information orbital (QIO/QICAS) localizer.
 *
 * This class provides a concrete implementation of the Localizer interface that
 * rotates the active orbitals so as to minimize the total single-orbital
 * entanglement entropy
 *
 *     F_QI = sum_{i in active} S(rho_i),
 *
 * following the quantum-information CAS (QICAS) scheme of Ding, Knecht &
 * Schilling (arXiv:2309.01676). The single-orbital entropy S(rho_i) is built
 * from the four orbital occupation eigenvalues {1 - n_a - n_b + D, n_a - D, n_b
 * - D, D} where D = Gamma_{i ibar i ibar} is the alpha-beta (aabb) two-particle
 * RDM diagonal. This is the same Boguslawski & Tecmer (2015) convention used by
 * Wavefunction::get_single_orbital_entropies.
 *
 * The objective is minimized with the paper's gradient-free Jacobi-sweep
 * scheme: each active orbital pair (i, j) is rotated by the angle that
 * minimizes the only two entropy terms that change (S_i + S_j), located by a
 * coarse-then-fine 1-D scan. The corresponding plane rotation is applied to the
 * cached active 1- and 2-RDMs and accumulated into a single unitary U, and the
 * active orbital coefficients are updated once as C_active <- C_active * U.
 *
 * @note This localizer performs a SINGLE orbital rotation: the Jacobi sweeps
 * are iterated to convergence against the FIXED input RDMs. It does not
 * re-solve the electronic structure problem to refresh the RDMs. Callers
 * wanting the full self-consistent QICAS outer loop (rotate -> recompute RDMs
 * -> repeat) should implement that loop themselves around repeated calls to
 * this localizer.
 *
 * Restrictions:
 * - Requires a single spatial orbital set (RHF or ROHF). Open-shell / high-spin
 *   states are supported: the alpha and beta occupations may differ, and the
 *   spin-resolved 1-RDMs are handled separately. Unrestricted (UHF) orbitals,
 *   where alpha and beta use different spatial orbitals, are not supported --
 *   a single spatial rotation is ill-defined in that case.
 * - Requires loc_indices_a == loc_indices_b, matching the active-space indices
 *   exactly (QIO produces a single spatial orbital set).
 * - Requires spin-dependent active 1- and 2-RDMs in the input wavefunction.
 * - Requires an AO overlap matrix in the orbitals (carried over to the output
 *   orbitals).
 *
 * @see Wavefunction::get_single_orbital_entropies
 */
class QIOLocalizer : public Localizer {
 public:
  /**
   * @brief Default constructor
   */
  QIOLocalizer() { _settings = std::make_unique<QIOLocalizerSettings>(); }

  /**
   * @brief Virtual destructor
   */
  ~QIOLocalizer() override = default;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const final { return "qdk_qio"; };

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
   * @throws std::invalid_argument if the selected indices are invalid or do not
   * match the active space, the orbitals are unrestricted or lack an active
   * space or overlap matrix, or the required spin-dependent active RDMs are
   * unavailable or not real-valued.
   */
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
