// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "flr_bliss_regularizer.hpp"

#include <cstddef>
#include <cstdint>
#include <lapack.hh>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>

namespace qdk::chemistry::algorithms::microsoft::flr_bliss {

namespace {

inline std::size_t two_body_index(std::size_t i, std::size_t j, std::size_t k,
                                  std::size_t l, std::size_t norb) {
  return ((i * norb + j) * norb + k) * norb + l;
}

}  // namespace

// ---------------------------------------------------------------------------
// Step 2: per-fragment median shift, aggregated into a global (mu2, xi).
// ---------------------------------------------------------------------------

GlobalTwoBodyShift accumulate_fragment_shifts(
    const std::vector<qdk::chemistry::utils::TwoBodyFragment>& fragments) {
  GlobalTwoBodyShift result;

  if (fragments.empty()) {
    return result;
  }

  const Eigen::Index norb = fragments.front().U.rows();
  result.xi = Eigen::MatrixXd::Zero(norb, norb);

  for (const auto& fragment : fragments) {
    result.lambda_df_baseline += fragment.lambda_df;

    // Eq. 27's LP has a closed-form solution: phi^(alpha) = median{eps_i}.
    const double phi = median(fragment.eps);
    const Eigen::VectorXd eps_shifted = fragment.eps.array() - phi;

    const double eps_shifted_abs_sum = eps_shifted.array().abs().sum();
    result.lambda_df_shifted += 0.5 * eps_shifted_abs_sum * eps_shifted_abs_sum;

    // Eq. 24's per-fragment BLISS operator parameters, generalized with an
    // explicit fragment sign (see TwoBodyFragment's docstring).
    const double mu2_alpha = phi * phi;
    const Eigen::VectorXd theta_alpha = -2.0 * phi * fragment.eps;

    // SIGN: the DF+LRPS optimal-fragment identity (Patel et al., Eq. 36) writes
    // the low-1-norm shifted fragment as H^(a) + K^(a) (plus a 1-electron term
    // and a constant), i.e. the per-fragment BLISS operator is *added*. The
    // global operator is *subtracted* from H (H - K, Eq. 5), so the aggregated
    // (mu2, xi) that rebuild_bliss_shifted_hamiltonian applies are the NEGATED sum of the
    // per-fragment K^(a) parameters. (Fragments come from double-factorizing
    // the PHYSICAL coefficient 1/2 g, so mu2/xi are already on the correct
    // scale for rebuild_bliss_shifted_hamiltonian.)
    result.mu2 -= fragment.sign * mu2_alpha;
    result.xi -= fragment.sign * (fragment.U * theta_alpha.asDiagonal() *
                                  fragment.U.transpose());
  }

  return result;
}

// ---------------------------------------------------------------------------
// Step 3: optimal one-electron BLISS shift mu1.
// ---------------------------------------------------------------------------

OneElectronShiftResult solve_one_electron_shift(
    const Eigen::MatrixXd& h, const Eigen::VectorXd& two_body_integrals,
    double mu2, const Eigen::MatrixXd& xi, double num_electrons) {
  OneElectronShiftResult result;

  const std::size_t norb = static_cast<std::size_t>(h.rows());

  // Mean-field contractions of the ORIGINAL two-electron tensor g[i,j,k,l]:
  //   coul_ij = sum_k g[i,j,k,k],  exch_ij = sum_k g[i,k,k,j].
  Eigen::MatrixXd coulomb = Eigen::MatrixXd::Zero(norb, norb);
  Eigen::MatrixXd exchange = Eigen::MatrixXd::Zero(norb, norb);
  for (std::size_t i = 0; i < norb; ++i) {
    for (std::size_t j = 0; j < norb; ++j) {
      double c = 0.0;
      double e = 0.0;
      for (std::size_t k = 0; k < norb; ++k) {
        c += two_body_integrals[two_body_index(i, j, k, k, norb)];
        e += two_body_integrals[two_body_index(i, k, k, j, norb)];
      }
      coulomb(i, j) = c;
      exchange(i, j) = e;
    }
  }

  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(norb, norb);

  // Baseline 1-norm of the ORIGINAL (unshifted) effective operator
  // Heff = h + coul - 1/2 exch, computed BEFORE the BLISS correction is folded
  // into coulomb/exchange in place below. (lapack::syev overwrites its input
  // and reads only the lower triangle; Job::NoVec skips eigenvectors.)
  Eigen::MatrixXd effective_one_body_original = h + coulomb - 0.5 * exchange;
  Eigen::VectorXd eigenvalues_baseline(norb);
  const int64_t baseline_info = lapack::syev(
      lapack::Job::NoVec, lapack::Uplo::Lower, static_cast<int64_t>(norb),
      effective_one_body_original.data(), static_cast<int64_t>(norb),
      eigenvalues_baseline.data());
  if (baseline_info != 0) {
    throw std::runtime_error(
        "solve_one_electron_shift: LAPACK syev failed on the baseline "
        "effective one-body operator (info=" +
        std::to_string(baseline_info) + ").");
  }
  result.lambda_1e_baseline = eigenvalues_baseline.array().abs().sum();

  // In-place: coulomb/exchange now hold the shifted contractions coul(g~)/
  // exch(g~). See TwoBodyBlissCorrection (hamiltonian_regularizer.hpp) for the
  // g~ definition and why this stays consistent with rebuild_bliss_shifted_hamiltonian's
  // full tensor.
  const TwoBodyBlissCorrection correction{mu2, xi};
  correction.add_coulomb_contraction(coulomb);
  correction.add_exchange_contraction(exchange);

  // Effective one-electron operator of H - K with mu1 = 0 (see header).
  const Eigen::MatrixXd h0 = h + (num_electrons - 1.0) * xi - mu2 * identity;
  Eigen::MatrixXd effective_one_body = h0 + coulomb - 0.5 * exchange;

  Eigen::VectorXd eigenvalues(norb);
  const int64_t shifted_info =
      lapack::syev(lapack::Job::NoVec, lapack::Uplo::Lower,
                   static_cast<int64_t>(norb), effective_one_body.data(),
                   static_cast<int64_t>(norb), eigenvalues.data());
  if (shifted_info != 0) {
    throw std::runtime_error(
        "solve_one_electron_shift: LAPACK syev failed on the shifted "
        "effective one-body operator (info=" +
        std::to_string(shifted_info) + ").");
  }
  result.mu1 = median(eigenvalues);
  result.lambda_1e = (eigenvalues.array() - result.mu1).abs().sum();

  return result;
}

// ---------------------------------------------------------------------------
// Top-level FLR-BLISS driver: wires steps 1-3 into a BlissShift.
// ---------------------------------------------------------------------------

BlissShift compute_flr_bliss_shift(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons,
    double df_truncation_threshold) {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian.is_restricted()) {
    throw std::invalid_argument(
        "compute_flr_bliss_shift currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  const double num_electrons = static_cast<double>(n_alpha_electrons) +
                               static_cast<double>(n_beta_electrons);

  auto [h_alpha, h_beta] = hamiltonian.get_one_body_integrals();
  (void)h_beta;
  auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian.get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;

  const size_t norb = static_cast<size_t>(h_alpha.rows());
  QDK_LOGGER().debug(
      "compute_flr_bliss_shift: num_orbitals={}, num_electrons={}, "
      "df_truncation_threshold={}",
      norb, num_electrons, df_truncation_threshold);

  const Eigen::VectorXd two_body_coefficient = 0.5 * g_aaaa;
  auto fragments = qdk::chemistry::utils::double_factorize(
      two_body_coefficient, norb, df_truncation_threshold);

  auto global_shift = accumulate_fragment_shifts(fragments);

  auto one_electron = solve_one_electron_shift(
      h_alpha, g_aaaa, global_shift.mu2, global_shift.xi, num_electrons);

  const double lambda_total_before =
      global_shift.lambda_df_baseline + one_electron.lambda_1e_baseline;
  const double lambda_total_after =
      global_shift.lambda_df_shifted + one_electron.lambda_1e;

  QDK_LOGGER().debug(
      "compute_flr_bliss_shift: lambda_total before={} ({} + {}), after={} ({} "
      "+ {}); lambda_DF baseline={}, shifted={}; lambda_1e baseline={}, "
      "shifted={}; mu1={}, mu2={}",
      lambda_total_before, one_electron.lambda_1e_baseline,
      global_shift.lambda_df_baseline, lambda_total_after,
      one_electron.lambda_1e, global_shift.lambda_df_shifted,
      global_shift.lambda_df_baseline, global_shift.lambda_df_shifted,
      one_electron.lambda_1e_baseline, one_electron.lambda_1e, one_electron.mu1,
      global_shift.mu2);

  BlissShift shift;
  shift.mu1 = one_electron.mu1;
  shift.mu2 = global_shift.mu2;
  shift.xi = global_shift.xi;
  return shift;
}

}  // namespace qdk::chemistry::algorithms::microsoft::flr_bliss
