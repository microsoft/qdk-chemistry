// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class DuccSolver
 * @brief DUCC effective-Hamiltonian builder evaluated with generated BTAS code.
 *
 * Builds the active-space effective Hamiltonian by a truncated
 * Baker-Campbell-Hausdorff (BCH) expansion of the unitarily
 * similarity-transformed Hamiltonian @f$ \bar H = e^{-\sigma} H e^{\sigma} @f$
 * with anti-Hermitian generator @f$ \sigma = T - T^\dagger @f$ (DUCC).
 *
 * The symbolic transformation, partial Wick contraction, @f$ \le 2 @f$-body
 * truncation and contraction-order optimization are performed OFFLINE by the
 * SeQuant-based code generator in `ducc/export_ducc_btas_spin.cpp`. Its output
 * -- plain BTAS contraction code -- is checked in as `ducc_equations.inc`, so
 * the library carries no symbolic-algebra dependency and the per-run cost is
 * dense tensor contractions only.
 *
 * Design notes:
 * - **Spin-blocked throughout.** The equations are derived over spin-resolved
 *   orbital spaces, so every contraction runs on SPATIAL per-spin blocks fed
 *   straight from the Hamiltonian and amplitude containers. No spin-orbital
 *   tensor is ever formed. Alpha and beta occupancies may differ (open shell).
 * - **Active-only evaluation.** Only the active block of @f$ \bar H @f$ is
 *   computed: the residual (free) legs carry active extents in the generated
 *   contractions, so this is a real work reduction, not compute-then-slice.
 *   The dressed one-/two-body output is likewise active-sized.
 * - **External amplitudes only.** @f$ \sigma @f$ is built from @f$ T_{ext} @f$:
 *   an amplitude all of whose indices are active is zeroed, so with an
 *   all-active space @f$ \sigma = 0 @f$ and every level reduces to level 0.
 *
 * The active space is supplied as a @ref data::SymmetryBlockedIndexSet and the
 * reference coupled-cluster amplitudes by the input @ref data::Wavefunction;
 * the BCH truncation level is the sole setting (`ducc_level`).
 *
 * @note Every level, including 0, runs the generated equations and produces a
 *       spin-blocked (unrestricted-type) Hamiltonian because dressed two-body
 *       integrals do not generally retain the eightfold symmetry expected by a
 *       single restricted block.
 */
class DuccSolver : public qdk::chemistry::algorithms::EffectiveHamiltonian {
 public:
  /**
   * @brief Default constructor. Inherits the shared effective-Hamiltonian
   *        settings (the BCH truncation level `ducc_level`).
   */
  DuccSolver() = default;

  /**
   * @brief Virtual destructor.
   */
  ~DuccSolver() = default;

  /**
   * @brief Access the algorithm's variant name.
   * @return "ducc".
   */
  std::string name() const final { return "ducc"; }

 protected:
  /**
   * @brief Build the DUCC effective active-space Hamiltonian.
   *
   * @param hamiltonian The full-space Hamiltonian to transform.
   * @param wavefunction Full-space wavefunction supplying the reference
   *        coupled-cluster amplitudes.
   * @param p_space_indices Active-space (P-space) orbital indices per spin
   *        channel.
   * @return The effective active-space Hamiltonian.
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<data::Wavefunction> wavefunction,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_space_indices)
      const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
