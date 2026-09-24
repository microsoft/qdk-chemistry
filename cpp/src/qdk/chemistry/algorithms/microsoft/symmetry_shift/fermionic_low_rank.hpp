// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/symmetry_shift.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <vector>

// This header declares the fermionic low-rank BLISS shift method (Patel et
// al., arXiv:2409.18277) -- the FermionicLowRankShifter implementation of
// qdk::chemistry::algorithms::SymmetryShifter -- together with its internal
// building blocks. They run in the order used by
// solve_fermionic_low_rank_shift():
//   1. The caller supplies an ALREADY double-factorized Hamiltonian, i.e. one
//      backed by data::FactorizedHamiltonianContainer. Producing it is the job
//      of the "double_factorization" hamiltonian_factorization algorithm; this
//      shifter never factorizes anything itself.
//   2. accumulate_fragment_shifts() -- per-fragment median shift (Eq. 27),
//      aggregated into a single global two-electron shift (mu2, xi) and the
//      per-fragment medians phi that produced it.
//   3. solve_one_electron_shift() -- optimal one-electron shift mu1
//      against the effective one-electron operator implied by (mu2, xi).
// rebuild_shifted_factorized_hamiltonian() then absorbs the shift into the
// fragment eigenvalues, so input and output are both sum-of-squares
// factorizations and no dense norb^4 tensor is ever built.
//
// The shared detail:: helpers (private to the library, in
// algorithms/symmetry_shift_detail.hpp) keep step 3's Coulomb/exchange-type
// contractions derived from the same closed-form dg_ijkl definition as the
// full tensor, so they cannot silently drift apart.

namespace qdk::chemistry::algorithms::microsoft {

/// Median of a vector's entries (average of the two middle entries for even
/// size), matching the paper's phi^(opt) = median{epsilon_i} rule
/// (Patel et al., arXiv:2409.18277, Eqs. 23 and 27).
inline double median(const Eigen::VectorXd& values) {
  std::vector<double> sorted(values.data(), values.data() + values.size());
  std::sort(sorted.begin(), sorted.end());
  const size_t n = sorted.size();
  if (n == 0) {
    return 0.0;
  }
  if (n % 2 == 1) {
    return sorted[n / 2];
  }
  return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
}

/// Aggregated global BLISS two-electron shift parameters (Patel et al.,
/// arXiv:2409.18277, Eq. 4/24, summed over all fragments), expressed
/// directly in the *original* orbital basis so they can be applied to the
/// dense integrals via Eqs. 6-7. These are the parameters of the operator K
/// that is SUBTRACTED from H; the per-fragment operators are negated
/// during aggregation because the DF+LRPS identity (Eq. C6) adds them.
struct GlobalTwoBodyShift {
  /// `norb` is required so that `xi` is always correctly sized, and
  /// (`num_ranks`, `num_copies`) so that `phi` is, including when there are
  /// no fragments to accumulate.
  GlobalTwoBodyShift(Eigen::Index norb, Eigen::Index num_ranks,
                     Eigen::Index num_copies)
      : xi(Eigen::MatrixXd::Zero(norb, norb)),
        phi(Eigen::MatrixXd::Zero(num_ranks, num_copies)) {}

  double mu2 = 0.0;    ///< Aggregated mu_2 (for H - K).
  Eigen::MatrixXd xi;  ///< Aggregated xi_ij (for H - K), norb x norb.
  /// Per-fragment median shift phi^(opt), [num_ranks, num_copies], on the
  /// BLISS scale eps = W / sqrt(2): shifted eigenvalues are W - sqrt(2)*phi.
  Eigen::MatrixXd phi;
  double lambda_df_baseline = 0.0;  ///< Sum of pre-shift fragment 1-norms.
  double lambda_df_shifted = 0.0;   ///< Sum of post-shift fragment 1-norms.
};

/// Apply the fermionic low-rank BLISS per-fragment median shift (Eq. 27) to
/// every (rank, copy) fragment of `container` and accumulate the resulting
/// global (mu_2, xi) BLISS shift parameters (Eq. 24, summed over fragments and
/// rotated back into the original orbital basis).
///
/// The container stores the fragments of the RAW tensor g, as
///   g_pqrs = Sum_rc M^rc_pq M^rc_rs,  M^rc_pq = Sum_b W^rc_b U^r_bp U^r_bq,
/// whereas BLISS is formulated for the PHYSICAL coefficient V = 1/2 g. The
/// two differ by an overall sqrt(2) on the fragment eigenvalues, so this
/// function uses eps = W / sqrt(2) throughout; that is what puts the
/// aggregated (mu2, xi) on the scale the rebuild expects.
///
/// Note also that `U` is stored flattened [R,B,N] in ROW-major order, so row
/// b of U^r is eigenvector b -- not column b.
///
/// A container with no ranks or no copies yields a well-formed zero shift
/// rather than an unsized xi.
GlobalTwoBodyShift accumulate_fragment_shifts(
    const qdk::chemistry::data::FactorizedHamiltonianContainer& container);

/// Result of solve_one_electron_shift(): the optimal one-electron BLISS
/// shift mu1 and the resulting fermionic 1-norm of the shifted effective
/// one-electron operator.
struct OneElectronShiftResult {
  double mu1 = 0.0;
  double lambda_1e = 0.0;           ///< 1-norm of the shifted, mu1-optimized
                                    ///< effective one-electron operator
                                    ///< (Patel et al., arXiv:2409.18277,
                                    ///< Eq. 15).
  double lambda_1e_baseline = 0.0;  ///< 1-norm of the ORIGINAL (unshifted)
                                    ///< effective one-electron operator, for a
                                    ///< before/after comparison.
};

/// Determine the optimal one-electron BLISS shift mu_1 (Patel et al.,
/// arXiv:2409.18277, Eq. 23) for the fermionic (DF) LCU 1-norm.
///
/// Crucially, mu1 is optimized against the EFFECTIVE one-electron operator of
/// H - K (Eq. 14) -- i.e. the one-body tensor with the two-electron mean-field
/// (Coulomb/exchange) contraction folded in -- NOT the bare modified integral
/// h + Ne*xi. Using the effective operator is what makes minimizing the
/// one-electron 1-norm actually reduce the true DF 1-norm of the shifted
/// Hamiltonian.
///
/// The effective operator is evaluated for the shifted two-electron
/// integrals g~ implied by (mu2, xi), matching the shifted fragments:
///   g~_ijkl = g_ijkl - 2*mu2*d_ij*d_kl - xi_ij*d_kl - d_ij*xi_kl
///   Heff0_ij = h_ij + (Ne-1)*xi_ij - mu2*d_ij
///              + sum_k g~[i,j,k,k] - 1/2 sum_k g~[i,k,k,j]   (mu1 = 0)
/// with mu1 = median{eig(Heff0)} and lambda_1e = sum_i |eig_i - mu1|.
///
/// @param h Bare one-electron integrals (norb x norb).
/// @param coulomb Coulomb contraction of the ORIGINAL g, i.e.
///        coulomb_ij = sum_k g[i,j,k,k]. Taken by value and consumed: the
///        shifted contraction is folded in place.
/// @param exchange Exchange contraction of the ORIGINAL g, i.e.
///        exchange_ij = sum_k g[i,k,k,j]. Also consumed in place.
/// @param mu2 Aggregated two-electron BLISS shift (GlobalTwoBodyShift::mu2).
/// @param xi Aggregated two-electron BLISS shift matrix
///        (GlobalTwoBodyShift::xi).
/// @param num_electrons Target number of active electrons (Ne).
OneElectronShiftResult solve_one_electron_shift(
    const Eigen::MatrixXd& h, Eigen::MatrixXd coulomb, Eigen::MatrixXd exchange,
    double mu2, const Eigen::MatrixXd& xi, double num_electrons);

/// The complete fermionic low-rank BLISS solution: the global (mu1, mu2, xi)
/// and the per-fragment medians that generated them. The two are only
/// mutually consistent when produced together on one particular
/// factorization, so they travel as a single object.
struct FermionicLowRankSolution {
  SymmetryShift shift;
  Eigen::MatrixXd phi;  ///< [num_ranks, num_copies]; zero for a zero shift.
};

/// Compute the full fermionic low-rank BLISS solution for `hamiltonian` in the
/// (n_alpha, n_beta)-electron sector (Patel et al., arXiv:2409.18277): read the
/// already-computed double factorization off the Hamiltonian (READ-ONLY),
/// accumulate the per-fragment median shift into a global (mu2, xi), then solve
/// for the optimal one-electron shift mu1.
///
/// The two 1-norms are minimized sequentially, not jointly, so the total is not
/// guaranteed to decrease; if it would increase, a zero shift (and a zero phi)
/// is returned with a warning, leaving the Hamiltonian unchanged.
///
/// @param hamiltonian The Hamiltonian to analyze. Must be restricted and
///        backed by a data::FactorizedHamiltonianContainer.
/// @param n_alpha_electrons Target number of alpha electrons.
/// @param n_beta_electrons Target number of beta electrons.
/// @return The computed solution, or a zero shift if the computed one would
///         not reduce the fermionic 1-norm.
/// @throws std::invalid_argument if `hamiltonian` is unrestricted, is not
///         backed by a FactorizedHamiltonianContainer, carries a nonzero
///         identity weight wB, or whose rotations are not complete orthogonal
///         rotations (see rebuild_shifted_factorized_hamiltonian).
FermionicLowRankSolution solve_fermionic_low_rank_shift(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons);

/// Thin wrapper over solve_fermionic_low_rank_shift() that discards the
/// per-fragment medians. Use it to INSPECT a shift; applying one is
/// SymmetryShifter::run()'s job.
///
/// @see solve_fermionic_low_rank_shift for the parameters and exceptions.
SymmetryShift compute_fermionic_low_rank_shift(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons);

/// Apply a fermionic low-rank BLISS solution, keeping the sum-of-squares form.
///
/// Because K is built from per-fragment medians, H - K is again a sum of
/// squares over the SAME rotations (Patel et al., Eq. 36): with O_r the
/// fragment's one-body operator and [O_r, N] = 0,
///   O_r^2 = (O_r - phi_r N)^2 + 2 phi_r N O_r - phi_r^2 N^2,
/// whose trailing terms are exactly K. So the shift is absorbed exactly into
/// the fragment eigenvalues and nothing is refactorized:
///   U unchanged,  W~ = W - sqrt(2)*phi_rc,  wB = 0,
///   h~ = h + (Ne-1)*xi - (mu1+mu2)*I,  E' = E_core + mu1*Ne + mu2*Ne^2.
///
/// NOT public: it is only correct when `solution` was produced by
/// solve_fermionic_low_rank_shift() on `container` itself, and phi is not
/// recoverable from a SymmetryShift alone.
///
/// @param original The Hamiltonian being shifted; supplies everything but the
///        two-body integrals.
/// @param container `original`'s factorization. Must satisfy the complete
///        orthogonal rotation precondition, already enforced by
///        solve_fermionic_low_rank_shift().
/// @param solution The shift and the per-fragment medians that generated it.
/// @param num_electrons Target number of active electrons (Ne).
/// @return The shifted Hamiltonian, backed by a
///         data::FactorizedHamiltonianContainer.
std::shared_ptr<qdk::chemistry::data::Hamiltonian>
rebuild_shifted_factorized_hamiltonian(
    const qdk::chemistry::data::Hamiltonian& original,
    const qdk::chemistry::data::FactorizedHamiltonianContainer& container,
    const FermionicLowRankSolution& solution, unsigned int num_electrons);

/**
 * @class FermionicLowRankShifterSettings
 * @brief Settings container for the fermionic low-rank symmetry shifter.
 *
 * The shifter has no tunable settings: it consumes whatever double
 * factorization the caller already computed, so truncation and the choice of
 * decomposition are settings of the "double_factorization"
 * hamiltonian_factorization algorithm instead.
 *
 * @see qdk::chemistry::algorithms::microsoft::FermionicLowRankShifter
 */
class FermionicLowRankShifterSettings : public qdk::chemistry::data::Settings {
 public:
  /**
   * @brief Constructor. There are no settings to initialize.
   */
  FermionicLowRankShifterSettings() = default;
};

/**
 * @class FermionicLowRankShifter
 * @brief Fermionic low-rank BLISS implementation of SymmetryShifter [1,2].
 *
 * Computes the symmetry shift (mu1, mu2, xi) with the fermionic low-rank
 * BLISS method of Patel et al. (arXiv:2409.18277): read the fragments off an
 * already double-factorized Hamiltonian, take the closed-form per-fragment
 * median shift, and solve for the optimal one-electron shift against the
 * effective one-electron operator.
 *
 * The input must be backed by a data::FactorizedHamiltonianContainer whose
 * rotations are complete orthogonal ones, and so is the output: the shift is
 * absorbed into the fragment eigenvalues, so the result can be block-encoded
 * without being refactorized. Call get_two_body_integrals() for dense ones.
 *
 * Typical usage:
 * ```cpp
 * auto factorizer =
 *   qdk::chemistry::algorithms::HamiltonianFactorizationFactory::create(
 *       "double_factorization");
 * auto factorized = factorizer->run(hamiltonian);
 *
 * auto shifter =
 *   qdk::chemistry::algorithms::SymmetryShifterFactory::create(
 *       "fermionic_low_rank");
 * auto shifted = shifter->run(factorized, n_alpha, n_beta);
 * ```
 *
 * @see qdk::chemistry::algorithms::SymmetryShifter
 * @see FermionicLowRankShifterSettings
 */
class FermionicLowRankShifter : public SymmetryShifter {
 public:
  /**
   * @brief Default constructor. Uses default
   *        FermionicLowRankShifterSettings.
   */
  FermionicLowRankShifter() {
    _settings = std::make_unique<FermionicLowRankShifterSettings>();
  }

  /**
   * @brief Virtual destructor.
   */
  ~FermionicLowRankShifter() override = default;

  /**
   * @brief Compute the fermionic low-rank shift (mu1, mu2, xi).
   *
   * @param hamiltonian The Hamiltonian to analyze. Must be restricted.
   * @param n_alpha_electrons The target number of alpha electrons.
   * @param n_beta_electrons The target number of beta electrons.
   * @return The computed symmetry shift parameters.
   *
   * @throws std::invalid_argument if the Hamiltonian is unrestricted.
   */
  SymmetryShift compute_shift(const data::Hamiltonian& hamiltonian,
                              unsigned int n_alpha_electrons,
                              unsigned int n_beta_electrons) const override;

  /**
   * @brief Access the algorithm's name.
   *
   * @return The algorithm's name.
   */
  std::string name() const override { return "fermionic_low_rank"; }

 protected:
  /**
   * @brief Composes solve_fermionic_low_rank_shift() and
   *        rebuild_shifted_factorized_hamiltonian().
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_alpha_electrons,
      unsigned int n_beta_electrons) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
