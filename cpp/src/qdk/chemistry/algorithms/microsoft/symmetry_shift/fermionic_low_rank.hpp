// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <qdk/chemistry/algorithms/symmetry_shift.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/utils/double_factorization.hpp>
#include <string>
#include <vector>

// This header declares the fermionic low-rank BLISS shift method (Patel et
// al., arXiv:2409.18277) -- the FermionicLowRankShifter implementation of
// qdk::chemistry::algorithms::SymmetryShifter -- together with its internal
// building blocks. They run in the order used by
// compute_fermionic_low_rank_shift():
//   1. double_factorize() (external, see double_factorization.hpp)
//   2. accumulate_fragment_shifts() -- per-fragment median shift (Eq. 27),
//      aggregated into a single global two-electron shift (mu2, xi).
//   3. solve_one_electron_shift() -- optimal one-electron shift mu1
//      against the effective one-electron operator implied by (mu2, xi).
// The resulting (mu1, mu2, xi) are packaged into a SymmetryShift and applied
// by qdk::chemistry::algorithms::rebuild_shifted_hamiltonian().
//
// The shared detail:: helpers (private to the library, in
// algorithms/symmetry_shift_detail.hpp) keep step 3's
// Coulomb/exchange-type contractions and rebuild_shifted_hamiltonian's full
// O(norb^4) tensor derived from the same closed-form dg_ijkl definition, so
// they cannot silently drift apart.

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
  double mu2 = 0.0;    ///< Aggregated mu_2 (for H - K).
  Eigen::MatrixXd xi;  ///< Aggregated xi_ij (for H - K), norb x norb.
  double lambda_df_baseline = 0.0;  ///< Sum of pre-shift fragment 1-norms.
  double lambda_df_shifted = 0.0;   ///< Sum of post-shift fragment 1-norms.
};

/// Apply the fermionic low-rank BLISS per-fragment median shift (Eq. 27) to
/// every fragment and accumulate the resulting global (mu_2, xi) BLISS shift
/// parameters (Eq. 24, summed over fragments and rotated back into the
/// original orbital basis). Fragments are expected to come from
/// double-factorizing the PHYSICAL two-electron coefficient 1/2 g, so the
/// aggregated (mu2, xi) are directly usable by
/// rebuild_shifted_hamiltonian().
GlobalTwoBodyShift accumulate_fragment_shifts(
    const std::vector<qdk::chemistry::utils::TwoBodyFragment>& fragments);

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
/// integrals g~ implied by (mu2, xi), matching rebuild_shifted_hamiltonian():
///   g~_ijkl = g_ijkl - 2*mu2*d_ij*d_kl - xi_ij*d_kl - d_ij*xi_kl
///   Heff0_ij = h_ij + (Ne-1)*xi_ij - mu2*d_ij
///              + sum_k g~[i,j,k,k] - 1/2 sum_k g~[i,k,k,j]   (mu1 = 0)
/// with mu1 = median{eig(Heff0)} and lambda_1e = sum_i |eig_i - mu1|.
///
/// @param h Bare one-electron integrals (norb x norb).
/// @param two_body_integrals Flattened ORIGINAL g_ijkl tensor (norb^4).
/// @param mu2 Aggregated two-electron BLISS shift (GlobalTwoBodyShift::mu2).
/// @param xi Aggregated two-electron BLISS shift matrix
///        (GlobalTwoBodyShift::xi).
/// @param num_electrons Target number of active electrons (Ne).
OneElectronShiftResult solve_one_electron_shift(
    const Eigen::MatrixXd& h, const Eigen::VectorXd& two_body_integrals,
    double mu2, const Eigen::MatrixXd& xi, double num_electrons);

/// Compute the fermionic low-rank BLISS shift (mu1, mu2, xi) for `hamiltonian`
/// in the (n_alpha, n_beta)-electron sector (Patel et al., arXiv:2409.18277).
///
/// Pipeline: double-factorize the physical two-electron coefficient 1/2 g,
/// accumulate the per-fragment median shift into a global (mu2, xi), then
/// solve for the optimal one-electron shift mu1. The result is returned as a
/// SymmetryShift ready for rebuild_shifted_hamiltonian(). The Hamiltonian
/// must be restricted.
///
/// @param hamiltonian The Hamiltonian to analyze (restricted).
/// @param n_alpha_electrons Target number of alpha electrons.
/// @param n_beta_electrons Target number of beta electrons.
/// @param df_truncation_threshold Fragments whose eigenvalue magnitude falls
///        below this threshold are dropped (0.0 = no truncation).
SymmetryShift compute_fermionic_low_rank_shift(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons,
    double df_truncation_threshold);

/**
 * @class FermionicLowRankShifterSettings
 * @brief Settings container for the fermionic low-rank symmetry shifter.
 *
 * Default settings:
 * - df_truncation_threshold: 0.0 - drop double-factorization fragments whose
 *   eigenvalue magnitude is below this threshold. The default of 0.0 performs
 *   no truncation (exact double factorization).
 *
 * @see qdk::chemistry::algorithms::microsoft::FermionicLowRankShifter
 */
class FermionicLowRankShifterSettings
    : public qdk::chemistry::data::Settings {
 public:
  /**
   * @brief Constructor that initializes the default settings.
   */
  FermionicLowRankShifterSettings() {
    set_default("df_truncation_threshold", 0.0);
  }
};

/**
 * @class FermionicLowRankShifter
 * @brief Fermionic low-rank BLISS implementation of SymmetryShifter [1,2].
 *
 * Computes the symmetry shift (mu1, mu2, xi) with the fermionic low-rank
 * BLISS method of Patel et al. (arXiv:2409.18277): double-factorize the
 * physical two-electron coefficient 1/2 g, take the closed-form per-fragment
 * median shift, and solve for the optimal one-electron shift against the
 * effective one-electron operator.
 *
 * Typical usage:
 * ```cpp
 * auto shifter =
 *   qdk::chemistry::algorithms::SymmetryShifterFactory::create(
 *       "fermionic_low_rank");
 * shifter->settings().set("df_truncation_threshold", 1e-8);
 * auto shifted = shifter->run(hamiltonian, n_alpha, n_beta);
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
   * @brief Composes compute_shift() and rebuild_shifted_hamiltonian().
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_alpha_electrons,
      unsigned int n_beta_electrons) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
