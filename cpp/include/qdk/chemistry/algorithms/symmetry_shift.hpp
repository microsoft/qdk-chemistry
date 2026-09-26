// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @file
 * @brief Number-symmetry Hamiltonian shifts (BLISS and variants).
 *
 * References:
 * - [1] I. Loaiza and A. F. Izmaylov, "Block-Invariant Symmetry Shift:
 *   Preprocessing technique for second-quantized Hamiltonians to improve
 *   their decompositions to Linear Combination of Unitaries",
 *   arXiv:2304.13772. (Introduces BLISS.)
 * - [2] S. Patel, A. S. Brahmachari, J. T. Cantin, L. Wang and A. F.
 *   Izmaylov, "Global Minimization of Electronic Hamiltonian 1-Norm via
 *   Linear Programming in the Block Invariant Symmetry Shift (BLISS)
 *   Method", arXiv:2409.18277. (Fermionic low-rank BLISS used here.)
 */

/**
 * @struct SymmetryShiftCoeffs
 * @brief Parameters of a number-symmetry shift operator [1].
 *
 * Bundles the parameters of the symmetry-shift operator subtracted from a
 * Hamiltonian H:
 *   K = mu1*(N - Ne) + mu2*(N^2 - Ne^2) + (N - Ne)*sum_ij xi_ij E_ij
 * K annihilates every Ne-electron state, so subtracting it leaves the
 * Ne-sector energy invariant while the fermionic LCU 1-norm may be reduced.
 *
 * This parameterization is shared by every SymmetryShifter implementation
 * (BLISS [1] and its fermionic low-rank variant [2] included); only the way
 * (mu1, mu2, xi) are *computed* differs. A SymmetryShiftCoeffs therefore
 * carries only the *result* of a shift computation, reported by
 * SymmetryShifter::last_shift() after a run.
 *
 * APPLYING ONE BY HAND: Eqs. 6-7 of [2] are written for the paper's
 * non-normal-ordered Hamiltonian (its Eq. 1) and for the physical coefficient
 * V = 1/2 g, so they do NOT transcribe directly onto this repository's
 * integrals. In this repository's normal-ordered convention,
 * H = sum_ij h_ij E_ij + 1/2 sum_ijkl g_ijkl (E_ij E_kl - d_jk E_il), with the
 * raw tensor g, subtracting K gives
 *   h~_ij   = h_ij + (Ne-1)*xi_ij - (mu1 + mu2)*d_ij
 *   g~_ijkl = g_ijkl - 2*mu2*d_ij*d_kl - xi_ij*d_kl - d_ij*xi_kl
 *   E'      = E_core + mu1*Ne + mu2*Ne^2
 * The extra -xi - mu2*I on the one-body tensor is the normal-ordering
 * correction, and the factor 2 on the mu2 term is g = 2V. Applying Eqs. 6-7
 * literally instead lands wrong by exactly those terms.
 */
struct SymmetryShiftCoeffs {
  double mu1 = 0.0;    ///< One-electron shift.
  double mu2 = 0.0;    ///< Two-electron shift.
  Eigen::MatrixXd xi;  ///< Two-electron shift matrix (norb x norb).
};

/**
 * @class SymmetryShifter
 * @brief Abstract interface for number-symmetry Hamiltonian shifts [1,2].
 *
 * Maps a Hamiltonian and a target alpha/beta electron count to a new
 * Hamiltonian that is energetically equivalent within that electron-number
 * sector but whose LCU/qubitization coefficients (e.g. the fermionic 1-norm
 * lambda) may be reduced, shrinking resource estimates for algorithms such as
 * qubitized phase estimation.
 *
 * run() is the whole interface: it computes the shift and applies it, and the
 * shifted Hamiltonian is the result. The (mu1, mu2, xi) that produced it are
 * an OPTIONAL by-product, reported by last_shift() for callers that want to
 * inspect or compare shifts. Applying a shift is deliberately not public: how
 * it folds into the Hamiltonian depends on the representation the
 * implementation consumes, and may need more than (mu1, mu2, xi) carries.
 *
 * What a shifter accepts -- which container types, which spin cases -- is a
 * property of the implementation, not of this interface; see the concrete
 * class for its preconditions.
 *
 * Typical usage:
 * @code
 * auto shifter =
 *   qdk::chemistry::algorithms::SymmetryShifterFactory::create("algorithm_name");
 * shifter->settings().set("parameter_name", value);
 * auto shifted = shifter->run(hamiltonian, n_alpha, n_beta);
 * auto shift = shifter->last_shift();  // optional, for inspection
 * @endcode
 *
 * @see SymmetryShiftCoeffs
 * @see SymmetryShifterFactory for creating instances of symmetry shifters
 * @see data::FactorizedHamiltonianContainer::get_lambda to inspect a
 *      factorized Hamiltonian's fermionic 1-norm without running a shifter.
 */
class SymmetryShifter
    : public Algorithm<SymmetryShifter, std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Hamiltonian>, unsigned int,
                       unsigned int> {
 public:
  /**
   * @brief Default constructor for SymmetryShifter.
   */
  SymmetryShifter() = default;

  /**
   * @brief Virtual destructor.
   */
  virtual ~SymmetryShifter() = default;

  /**
   * @brief Shift a Hamiltonian for a target electron count.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param hamiltonian The Hamiltonian to shift
   * @param n_alpha_electrons The target number of alpha electrons
   * @param n_beta_electrons The target number of beta electrons
   * \endcond
   * @return A new, shifted Hamiltonian that agrees with the input
   *         Hamiltonian's energy in the (n_alpha_electrons,
   *         n_beta_electrons)-electron sector.
   *
   * @note Settings are automatically locked when this method is called.
   */
  using Algorithm::run;

  /**
   * @brief The shift applied by the most recent run(), if any.
   *
   * Implementations record (mu1, mu2, xi) as they apply it, so this costs
   * nothing beyond the run itself and cannot disagree with the Hamiltonian
   * run() returned. Reporting a shift is optional: an implementation whose
   * shift is not expressible as a SymmetryShiftCoeffs leaves this empty.
   *
   * @return The recorded shift, or std::nullopt if run() has not been called
   *         or the implementation does not report one.
   *
   * @note Not synchronized. Concurrent run() calls on ONE shifter instance
   *       race on this slot; give each thread its own instance.
   */
  std::optional<SymmetryShiftCoeffs> last_shift() const { return _last_shift; }

  /**
   * @brief Access the algorithm's name.
   *
   * @return The algorithm's name.
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name.
   *
   * @return The algorithm's type name.
   */
  std::string type_name() const final { return "symmetry_shifter"; };

 protected:
  /**
   * @brief Record the shift that _run_impl() is applying, for last_shift().
   */
  void _record_shift(SymmetryShiftCoeffs shift) const {
    _last_shift = std::move(shift);
  }

  /**
   * @brief Implementation of the symmetry shift.
   *
   * Computes the shift and applies it. Called by run() after settings have
   * been locked.
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_alpha_electrons, unsigned int n_beta_electrons) const = 0;

 private:
  /// Written by _record_shift() from the const _run_impl(); see last_shift().
  mutable std::optional<SymmetryShiftCoeffs> _last_shift;
};

/**
 * @brief Factory class for creating symmetry shifter instances.
 *
 * Typical usage:
 * ```
 * using qdk::chemistry::algorithms::SymmetryShifterFactory;
 * auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
 * auto shifted = shifter->run(factorized_hamiltonian, n_alpha, n_beta);
 * ```
 *
 * @note default_algorithm_name() names the best shifter available, not a
 *       fixed one; it is expected to move as implementations are added. Name
 *       the algorithm explicitly to pin it.
 *
 * @see SymmetryShifter
 */
struct SymmetryShifterFactory
    : public AlgorithmFactory<SymmetryShifter, SymmetryShifterFactory> {
  static std::string algorithm_type_name() { return "symmetry_shifter"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "fermionic_low_rank"; }
};

}  // namespace qdk::chemistry::algorithms
