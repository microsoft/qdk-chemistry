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
#include <utility>
#include <vector>

// Fermionic low-rank BLISS shift (Patel et al., arXiv:2409.18277): the
// FermionicLowRankShifter implementation of SymmetryShifter, plus the
// internal steps solve_fermionic_low_rank_shift() runs in order:
//   1. accumulate_fragment_shifts() -- one pass over the fragments of an
//      ALREADY double-factorized Hamiltonian, producing the mean-field
//      contractions of g and the global two-electron shift (mu2, xi).
//   2. solve_one_electron_shift() -- optimal mu1 against the effective
//      one-electron operator implied by (mu2, xi).
//   3. rebuild_shifted_factorized_hamiltonian() -- absorbs the shift into the
//      fragment eigenvalues, so input and output are both sum-of-squares
//      factorizations and no dense norb^4 tensor is ever built.
//
// CONVENTION: this repository stores the RAW tensor g and normal-orders the
// Hamiltonian as H = sum h_ij E_ij + 1/2 sum g_ijkl (E_ij E_kl - d_jk E_il),
// whereas the paper's Eqs. 1, 6 and 7 are written without normal ordering and
// for the physical coefficient V = 1/2 g. The shift formulas below therefore
// differ from Eqs. 6-7 by an extra -xi - mu2*I on the one-body tensor (the
// normal-ordering correction) and by a factor 2 on the mu2 term (g = 2V).
// Both deviations are deliberate; see the SymmetryShift doc in
// algorithms/symmetry_shift.hpp for the full transcription.

namespace qdk::chemistry::algorithms::microsoft {

/// The minimizers of sum_i |values_i - phi|, as the closed interval [lo, hi]
/// (Patel et al., arXiv:2409.18277, Eqs. 23 and 27). For odd size the
/// endpoints coincide; for even size the objective is flat between them.
/// Returns {0, 0} for an empty vector.
inline std::pair<double, double> median_interval(
    const Eigen::VectorXd& values) {
  std::vector<double> sorted(values.data(), values.data() + values.size());
  std::sort(sorted.begin(), sorted.end());
  const size_t n = sorted.size();
  if (n == 0) {
    return {0.0, 0.0};
  }
  if (n % 2 == 1) {
    return {sorted[n / 2], sorted[n / 2]};
  }
  return {sorted[n / 2 - 1], sorted[n / 2]};
}

/// Everything a single pass over the fragments produces: the mean-field
/// contractions of the ORIGINAL g, and the aggregated global BLISS
/// two-electron shift they imply (Patel et al., arXiv:2409.18277, Eq. 4/24,
/// summed over fragments) in the ORIGINAL orbital basis. (mu2, xi)
/// parametrize the operator K that is SUBTRACTED from H, so they are the
/// negated sum of the per-fragment K^(a) (Eq. C6 adds them).
struct FragmentAccumulation {
  /// The sizes are required so the matrices are well-formed even with no
  /// fragments to accumulate.
  FragmentAccumulation(Eigen::Index norb, Eigen::Index num_ranks,
                       Eigen::Index num_copies)
      : coulomb(Eigen::MatrixXd::Zero(norb, norb)),
        exchange(Eigen::MatrixXd::Zero(norb, norb)),
        xi(Eigen::MatrixXd::Zero(norb, norb)),
        phi(Eigen::MatrixXd::Zero(num_ranks, num_copies)) {}

  /// coulomb_ij = sum_k g[i,j,k,k], taken from the factorization as
  /// sum_rc tr(M^rc) M^rc -- never from a dense norb^4 tensor.
  Eigen::MatrixXd coulomb;
  /// exchange_ij = sum_k g[i,k,k,j] = sum_rc (M^rc M^rc)_ij.
  Eigen::MatrixXd exchange;

  double mu2 = 0.0;    ///< Aggregated mu_2 (for H - K).
  Eigen::MatrixXd xi;  ///< Aggregated xi_ij (for H - K), norb x norb.
  /// Per-fragment median shift phi^(opt), [num_ranks, num_copies], on the
  /// BLISS scale eps = W / sqrt(2): shifted eigenvalues are W - sqrt(2)*phi.
  Eigen::MatrixXd phi;
  double lambda_df_baseline = 0.0;  ///< Sum of pre-shift fragment 1-norms.
  double lambda_df_shifted = 0.0;   ///< Sum of post-shift fragment 1-norms.
};

/// Walk every (rank, copy) fragment of `container` ONCE, accumulating the
/// mean-field contractions of g together with the per-fragment median shift
/// (Eq. 27) aggregated into a global (mu2, xi) BLISS shift (Eq. 24, rotated
/// back into the original orbital basis).
///
/// When B is even the minimizer of Eq. 27 is an interval; phi takes its upper
/// endpoint, which is an actual eps_i and so drops a unitary from the
/// one-electron operator's LCU (text after Eq. 27).
///
/// The container factorizes the RAW tensor g while BLISS is formulated for
/// V = 1/2 g, so this works throughout on eps = W / sqrt(2); that is what puts
/// (mu2, xi) on the scale the rebuild expects. `U` is stored flattened [R,B,N]
/// in ROW-major order, so row b of U^r is eigenvector b -- not column b.
///
/// PRECONDITION: the aggregated (mu2, xi) are only meaningful when every U^r
/// is a complete orthogonal rotation, since the -phi*N_hat step needs
/// Sum_b u_b u_b^T = I. solve_fermionic_low_rank_shift() enforces this; a
/// direct caller must check it too.
FragmentAccumulation accumulate_fragment_shifts(
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
/// H - K -- the one-body tensor with the two-electron mean-field contraction
/// folded in -- NOT the bare h + Ne*xi. Only the effective operator makes
/// minimizing the one-electron 1-norm actually reduce the true DF 1-norm
/// (Eq. 12 is the per-fragment correction, Eq. 14 diagonalizes the sum).
///
/// It is evaluated for the shifted integrals g~ implied by (mu2, xi):
///   g~_ijkl = g_ijkl - 2*mu2*d_ij*d_kl - xi_ij*d_kl - d_ij*xi_kl
///   Heff0_ij = h_ij + (Ne-1)*xi_ij - mu2*d_ij
///              + sum_k g~[i,j,k,k] - 1/2 sum_k g~[i,k,k,j]   (mu1 = 0)
/// with mu1 drawn from median_interval{eig(Heff0)} and
/// lambda_1e = sum_i |eig_i - mu1|. See the CONVENTION note at the top of
/// this header for why the (Ne-1) and -mu2*d_ij terms differ from Eq. 6.
///
/// @param h Bare one-electron integrals (norb x norb).
/// @param coulomb Coulomb contraction of the ORIGINAL g,
///        coulomb_ij = sum_k g[i,j,k,k]. Consumed: the shifted contraction is
///        folded in place.
/// @param exchange Exchange contraction of the ORIGINAL g,
///        exchange_ij = sum_k g[i,k,k,j]. Also consumed in place.
/// @param mu2 Aggregated two-electron BLISS shift
///        (FragmentAccumulation::mu2).
/// @param xi Aggregated two-electron BLISS shift matrix
///        (FragmentAccumulation::xi).
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
/// accumulate the per-fragment median shift into a global (mu2, xi), then
/// solve for the optimal one-electron shift mu1.
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

/// Apply a fermionic low-rank BLISS solution, keeping the sum-of-squares form.
///
/// Because K is built from per-fragment medians, H - K is again a sum of
/// squares over the SAME rotations (Patel et al., Eq. C3, packaged as
/// Eqs. C5-C6): with O_r the fragment's one-body operator and [O_r, N] = 0,
///   O_r^2 = (O_r - phi_r N)^2 + 2 phi_r N O_r - phi_r^2 N^2,
/// whose trailing terms are exactly K. So the shift is absorbed exactly into
/// the fragment eigenvalues and nothing is refactorized:
///   U unchanged,  W~ = W - sqrt(2)*phi_rc,  wB = 0,
///   h~ = h + (Ne-1)*xi - (mu1+mu2)*I,  E' = E_core + mu1*Ne + mu2*Ne^2.
/// See the CONVENTION note at the top of this header for why h~ and the
/// implied dg differ from Eqs. 6-7.
///
/// NOT public: it is only correct when `solution` was produced by
/// solve_fermionic_low_rank_shift() on `container` itself, and phi is not
/// recoverable from a SymmetryShift alone.
///
/// @param original The Hamiltonian being shifted; supplies everything but the
///        two-body integrals. Its inactive Fock matrix is carried over
///        unchanged as provenance rather than recomputed (h and g both move),
///        exactly as FactorizedHamiltonianContainer::clone() does.
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
 * effective one-electron operator. The shift that was applied is reported by
 * SymmetryShifter::last_shift().
 *
 * PRECONDITIONS. The input must be restricted (spin-restricted) and backed by
 * a data::FactorizedHamiltonianContainer whose identity weight wB is zero and
 * whose rotations are complete orthogonal ones; anything else throws
 * std::invalid_argument. The output is backed by the same container type: the
 * shift is absorbed into the fragment eigenvalues, so the result can be
 * block-encoded without being refactorized. Call get_two_body_integrals() for
 * dense ones.
 *
 * Only the TOTAL electron count n_alpha + n_beta enters the shift; this
 * method does not use Sz, so (5,5) and (6,4) produce the same result.
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
   * @brief Access the algorithm's name.
   *
   * @return The algorithm's name.
   */
  std::string name() const override { return "fermionic_low_rank"; }

 protected:
  /**
   * @brief Composes solve_fermionic_low_rank_shift() and
   *        rebuild_shifted_factorized_hamiltonian(), solving once and
   *        recording the applied shift for last_shift().
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_alpha_electrons,
      unsigned int n_beta_electrons) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
