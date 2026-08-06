// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <vector>

#include <qdk/chemistry/algorithms/hamiltonian_regularizer.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/utils/double_factorization.hpp>

// This header collects every internal building block of the FLR-BLISS
// Hamiltonian regularizer (Patel et al., arXiv:2409.18277) in the order they
// are used by FlrBlissRegularizer::_run_impl() (see flr_bliss_regularizer.cpp):
//   1. double_factorize() (external, see double_factorization.hpp)
//   2. accumulate_fragment_shifts() -- per-fragment median shift (Eq. 27),
//      aggregated into a single global two-electron BLISS shift (mu2, xi).
//   3. solve_one_electron_shift() -- optimal one-electron BLISS shift mu1
//      against the effective one-electron operator implied by (mu2, xi).
//   4. rebuild_hamiltonian() -- apply the full (mu1, mu2, xi) shift to the
//      dense one- and two-electron integrals and assemble the shifted
//      Hamiltonian.
// TwoBodyBlissCorrection is a shared helper used by both steps 3 and 4 so
// that the two-electron correction's Coulomb/exchange-type contractions
// (used in solve_one_electron_shift) and the correction's full O(norb^4)
// tensor (used in rebuild_hamiltonian) cannot silently drift apart -- both
// are derived from the same closed-form dg_ijkl definition below.

namespace qdk::chemistry::algorithms::microsoft::flr_bliss {

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

/// The two-body correction that the aggregated BLISS shift (mu2, xi) adds to
/// the two-electron integral tensor g_ijkl:
///
///   dg_ijkl = -2*mu2*delta_ij*delta_kl - xi_ij*delta_kl - delta_ij*xi_kl
///
/// This is the SINGLE SOURCE OF TRUTH for that tensor. rebuild_hamiltonian()
/// adds dg (via add_two_body_correction()) directly onto g to build
/// g~ = g + dg. solve_one_electron_shift() never needs the whole O(norb^4)
/// tensor -- it only needs dg's Coulomb/exchange-type contractions
/// (add_coulomb_contraction/add_exchange_contraction below), which fold into
/// the effective one-electron operator alongside the ORIGINAL g's own
/// Coulomb/exchange contraction. Both call sites derive their numbers from
/// THIS struct, so they cannot silently drift apart;
/// test_hamiltonian_regularizer.cpp checks add_coulomb_contraction/
/// add_exchange_contraction against a direct, brute-force index contraction of
/// add_two_body_correction() for correctness.
struct TwoBodyBlissCorrection {
  double mu2;
  Eigen::MatrixXd xi;

  /// Add Sum_k dg_ijkk (Coulomb-type contraction) onto `coulomb` in place,
  /// obtained by substituting dg_ijkl's definition and summing k over
  /// Sum_k delta_kk = norb, Sum_k xi_kk = tr(xi):
  ///   Sum_k dg_ijkk = -2*mu2*norb*delta_ij - norb*xi_ij - tr(xi)*delta_ij
  ///                 = -(2*mu2*norb + tr(xi))*delta_ij - norb*xi_ij
  /// `norb` is taken from the caller's matrix, and the correction is folded in
  /// place so no separate norb x norb result is allocated: on return `coulomb`
  /// holds coul(g~) = coul(g) + Sum_k dg_ijkk.
  void add_coulomb_contraction(Eigen::MatrixXd& coulomb) const {
    const double norb = static_cast<double>(coulomb.rows());
    coulomb -= norb * xi;
    coulomb.diagonal().array() -= 2.0 * mu2 * norb + xi.trace();
  }

  /// Add Sum_k dg_ikkj (exchange-type contraction) onto `exchange` in place,
  /// using Sum_k delta_ik*delta_kj = delta_ij, Sum_k xi_ik*delta_kj = xi_ij,
  /// Sum_k delta_ik*xi_kj = xi_ij:
  ///   Sum_k dg_ikkj = -2*mu2*delta_ij - 2*xi_ij
  /// The correction is folded in place (no norb x norb allocation): on return
  /// `exchange` holds exch(g~) = exch(g) + Sum_k dg_ikkj.
  void add_exchange_contraction(Eigen::MatrixXd& exchange) const {
    exchange -= 2.0 * xi;
    exchange.diagonal().array() -= 2.0 * mu2;
  }

  /// Add the two-body BLISS correction dg_ijkl onto `g` in place, where `g` is
  /// the flattened two-electron tensor ((i*norb+j)*norb+k)*norb+l with side
  /// length `norb`. On return `g` holds g~ = g + dg without materializing a
  /// separate O(norb^4) dg tensor. Used by rebuild_hamiltonian() to build g~;
  /// tests brute-force-verify the closed-form contractions above against direct
  /// summation of the dg this method adds (starting from a zero `g`).
  void add_two_body_correction(Eigen::VectorXd& g, Eigen::Index norb) const {
    for (Eigen::Index i = 0; i < norb; ++i) {
      for (Eigen::Index j = 0; j < norb; ++j) {
        for (Eigen::Index k = 0; k < norb; ++k) {
          for (Eigen::Index l = 0; l < norb; ++l) {
            double value = 0.0;
            if (i == j && k == l) {
              value -= 2.0 * mu2;
            }
            if (k == l) {
              value -= xi(i, j);
            }
            if (i == j) {
              value -= xi(k, l);
            }
            if (value != 0.0) {
              g[((i * norb + j) * norb + k) * norb + l] += value;
            }
          }
        }
      }
    }
  }
};

/// Aggregated global BLISS two-electron shift parameters (Patel et al.,
/// arXiv:2409.18277, Eq. 4/24, summed over all fragments), expressed
/// directly in the *original* orbital basis so they can be applied to the
/// dense integrals via Eqs. 6-7. These are the parameters of the operator K
/// that is SUBTRACTED from H; the per-fragment operators are negated
/// during aggregation because the DF+LRPS identity (Eq. C6) adds them.
struct GlobalTwoBodyShift {
  double mu2 = 0.0;                 ///< Aggregated mu_2 (for H - K).
  Eigen::MatrixXd xi;                ///< Aggregated xi_ij (for H - K), norb x norb.
  double lambda_df_baseline = 0.0;   ///< Sum of pre-shift fragment 1-norms.
  double lambda_df_shifted = 0.0;    ///< Sum of post-shift fragment 1-norms.
};

/// Apply the FLR-BLISS per-fragment median shift (Eq. 27) to every
/// fragment and accumulate the resulting global (mu_2, xi) BLISS shift
/// parameters (Eq. 24, summed over fragments and rotated back into the
/// original orbital basis). Fragments are expected to come from
/// double-factorizing the PHYSICAL two-electron coefficient 1/2 g, so the
/// aggregated (mu2, xi) are directly usable by rebuild_hamiltonian().
GlobalTwoBodyShift accumulate_fragment_shifts(
    const std::vector<qdk::chemistry::utils::TwoBodyFragment>& fragments);

/// Result of solve_one_electron_shift(): the optimal one-electron BLISS
/// shift mu1 and the resulting fermionic 1-norm of the shifted effective
/// one-electron operator, plus the auxiliary h_prime needed by
/// rebuild_hamiltonian() to recover the physical one-body integrals.
struct OneElectronShiftResult {
  double mu1 = 0.0;
  double lambda_1e = 0.0;            ///< 1-norm of the shifted, mu1-optimized
                                     ///< effective one-electron operator
                                     ///< (Patel et al., arXiv:2409.18277,
                                     ///< Eq. 15).
  double lambda_1e_baseline = 0.0;   ///< 1-norm of the ORIGINAL (unshifted)
                                     ///< effective one-electron operator, for a
                                     ///< before/after comparison.
  Eigen::MatrixXd h_prime;           ///< h_ij + Ne * xi_ij, kept so
                                     ///< rebuild_hamiltonian can recover the
                                     ///< physical one-body via h = h_prime -
                                     ///< Ne*xi.
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
/// The effective operator is evaluated for the BLISS-shifted two-electron
/// integrals g~ implied by (mu2, xi), matching rebuild_hamiltonian():
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

/// Apply the global BLISS shift (mu1, mu2, xi) of Patel et al.
/// (arXiv:2409.18277) directly to the dense one- and two-electron integrals
/// and assemble the resulting Hamiltonian.
///
/// The shift reproduces, in this container's canonical chemist convention
/// g[i,j,k,l] = (ij|kl), the operator
///   K = mu1*(N - Ne) + mu2*(N^2 - Ne^2) + (N - Ne)*sum_ij xi_ij E_ij
/// subtracted from H. Because K annihilates every Ne-electron state, the
/// Ne-sector energy is invariant for any (mu1, mu2, xi). Expanding -K gives
///   h_tilde_ij   = h_ij + (Ne - 1)*xi_ij - (mu1 + mu2)*delta_ij
///   g_tilde_ijkl = g_ijkl - 2*mu2*delta_ij*delta_kl
///                          - xi_ij*delta_kl - delta_ij*xi_kl
///   E_core'      = E_core + mu1*Ne + mu2*Ne^2
/// These coefficients were derived in the container's own convention and
/// verified to machine precision against explicit single-determinant
/// energies. Do NOT substitute the qdk BLISS helper / Appendix C formulas
/// here: those assume a different two-body normalisation and produce spurious
/// factors of 2 (and a non-invariant energy) if copied verbatim.
///
/// NOTE: h_prime carries the FULL h_ij + Ne*xi_ij, which is Appendix C's
/// *effective* one-body operator for the 1-norm / mu1 optimisation only; the
/// physical one-body integrals are recovered internally as h = h_prime - Ne*xi
/// before the shift above is applied.
///
/// @param original The Hamiltonian being shifted. Must be restricted; its
///        orbitals, inactive Fock matrix, and Hamiltonian type are carried
///        through unchanged.
/// @param h_prime h_ij + Ne*xi_ij, i.e. OneElectronShiftResult::h_prime.
/// @param mu1 The one-electron BLISS shift (OneElectronShiftResult::mu1).
/// @param mu2 The aggregated two-electron BLISS shift
///        (GlobalTwoBodyShift::mu2).
/// @param xi The aggregated two-electron BLISS shift matrix
///        (GlobalTwoBodyShift::xi).
/// @param two_body_integrals The *original* (unshifted) flattened g_ijkl
///        tensor.
/// @param num_electrons Target number of active electrons (Ne).
/// @return The BLISS-shifted Hamiltonian.
std::shared_ptr<qdk::chemistry::data::Hamiltonian> rebuild_hamiltonian(
    const qdk::chemistry::data::Hamiltonian& original,
    const Eigen::MatrixXd& h_prime, double mu1, double mu2,
    const Eigen::MatrixXd& xi, const Eigen::VectorXd& two_body_integrals,
    double num_electrons);

}  // namespace qdk::chemistry::algorithms::microsoft::flr_bliss

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class FlrBlissSettings
 * @brief Settings container for the FLR-BLISS Hamiltonian regularizer
 *
 * Default settings include:
 * - df_truncation_threshold: 0.0 - Fragments produced by double-factorizing
 *   the two-electron integrals whose |eigenvalue| falls below this
 *   threshold are dropped. The default of 0.0 performs no truncation
 *   (an exact/lossless double factorization); truncation is only applied
 *   if the user explicitly opts in.
 *
 * @see qdk::chemistry::algorithms::microsoft::FlrBlissRegularizer
 */
class FlrBlissSettings : public qdk::chemistry::data::Settings {
 public:
  FlrBlissSettings() { set_default("df_truncation_threshold", 0.0); }
};

/**
 * @class FlrBlissRegularizer
 * @brief Hamiltonian regularizer implementing the FLR-BLISS symmetry shift
 *
 * Implements the block-invariant symmetry shift (BLISS) technique of
 * Patel et al. (arXiv:2409.18277), applied to the fermionic
 * double-factorized (DF) LCU representation, to reduce a Hamiltonian's
 * fermionic 1-norm while leaving its energy exactly invariant within a
 * target (n_alpha, n_beta)-electron sector.
 *
 * The pipeline (see flr_bliss namespace in this same header, and
 * flr_bliss_regularizer.cpp for the implementation):
 *  1. Double-factorize the (physical) two-electron integral coefficient
 *     via qdk::chemistry::utils::double_factorize().
 *  2. Determine a per-fragment two-electron BLISS shift (closed-form
 *     median optimization) and aggregate it into a single global
 *     (mu2, xi) shift via flr_bliss::accumulate_fragment_shifts().
 *  3. Determine the optimal one-electron BLISS shift mu1 against the
 *     resulting effective one-electron operator via
 *     flr_bliss::solve_one_electron_shift().
 *  4. Rebuild the shifted one- and two-electron integrals and assemble a
 *     new Hamiltonian via flr_bliss::rebuild_hamiltonian().
 *
 * Only restricted (spin-restricted) Hamiltonians are currently supported.
 *
 * @see HamiltonianRegularizer
 * @see FlrBlissSettings
 * @see qdk::chemistry::utils::hamiltonian_one_norm
 */
class FlrBlissRegularizer : public HamiltonianRegularizer {
 public:
  FlrBlissRegularizer() {
    _settings = std::make_unique<FlrBlissSettings>();
  }

  ~FlrBlissRegularizer() override = default;

  std::string name() const final { return "flr_bliss"; }

 protected:
  std::shared_ptr<qdk::chemistry::data::Hamiltonian> _run_impl(
      std::shared_ptr<qdk::chemistry::data::Hamiltonian> hamiltonian,
      unsigned int n_alpha_electrons,
      unsigned int n_beta_electrons) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
