// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "fermionic_low_rank.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <lapack.hh>
#include <memory>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../../symmetry_shift_detail.hpp"

namespace qdk::chemistry::algorithms::microsoft {

// ---------------------------------------------------------------------------
// Step 2: per-fragment median shift, aggregated into a global (mu2, xi).
// ---------------------------------------------------------------------------

namespace {

/// Read leaf M^{rc} out of the flat [R,B,N] / [R,B,C] arrays.
///   M^{rc}_{pq} = Sum_b W^{rc}_b U^r_{bp} U^r_{bq}
/// `Ur` is [B,N] with eigenvectors in ROWS; the returned matrix is [N,N].
Eigen::MatrixXd leaf_matrix(
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>& Ur,
    const Eigen::VectorXd& w, size_t r, size_t c, size_t B, size_t C,
    size_t norb) {
  Eigen::MatrixXd scaled(B, norb);
  for (size_t b = 0; b < B; ++b) {
    scaled.row(b) = w(r * B * C + b * C + c) * Ur.row(b);
  }
  return Ur.transpose() * scaled;
}

}  // namespace

GlobalTwoBodyShift accumulate_fragment_shifts(
    const qdk::chemistry::data::FactorizedHamiltonianContainer& container) {
  const size_t norb = container.get_num_orbitals();
  const size_t R = container.get_num_ranks();
  const size_t B = container.get_num_bases();
  const size_t C = container.get_num_copies();

  GlobalTwoBodyShift result(static_cast<Eigen::Index>(norb));

  const Eigen::VectorXd& u = container.get_u_matrices();
  const Eigen::VectorXd& w = container.get_w_matrices();

  // SCALE: the container factorizes the raw tensor g, but BLISS is formulated
  // for the physical coefficient V = 1/2 g, so eps = W / sqrt(2).
  const double kInvSqrt2 = 1.0 / std::sqrt(2.0);

  for (size_t r = 0; r < R; ++r) {
    // ORIENTATION: [B,N] row-major, so row b is eigenvector b.
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        Ur(u.data() + r * B * norb, B, norb);

    for (size_t c = 0; c < C; ++c) {
      Eigen::VectorXd eps(B);
      for (size_t b = 0; b < B; ++b) {
        eps(b) = kInvSqrt2 * w(r * B * C + b * C + c);
      }

      const double eps_abs_sum = eps.array().abs().sum();
      result.lambda_df_baseline += 0.5 * eps_abs_sum * eps_abs_sum;

      // Eq. 27's LP has a closed-form solution: phi^(alpha) = median{eps_i}.
      const double phi = median(eps);
      const Eigen::VectorXd eps_shifted = eps.array() - phi;

      const double eps_shifted_abs_sum = eps_shifted.array().abs().sum();
      result.lambda_df_shifted +=
          0.5 * eps_shifted_abs_sum * eps_shifted_abs_sum;

      // Eq. 24's per-fragment BLISS operator parameters.
      const double mu2_alpha = phi * phi;
      const Eigen::VectorXd theta_alpha = -2.0 * phi * eps;

      // SIGN: the DF+LRPS optimal-fragment identity (Patel et al., Eq. 36)
      // writes the low-1-norm shifted fragment as H^(a) + K^(a) (plus a
      // 1-electron term and a constant), i.e. the per-fragment BLISS operator
      // is *added*. The global operator is *subtracted* from H (H - K, Eq. 5),
      // so the aggregated (mu2, xi) that rebuild_shifted_hamiltonian applies
      // are the NEGATED sum of the per-fragment K^(a) parameters. Cholesky
      // fragments are all positive, so there is no per-fragment sign to carry.
      result.mu2 -= mu2_alpha;

      // theta is expressed in the rotated basis; rotate it back. Ur has
      // eigenvectors in rows, so the transform is Ur^T diag(theta) Ur.
      Eigen::MatrixXd scaled(B, norb);
      for (size_t b = 0; b < B; ++b) {
        scaled.row(b) = theta_alpha(b) * Ur.row(b);
      }
      result.xi.noalias() -= Ur.transpose() * scaled;
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Step 3: optimal one-electron BLISS shift mu1.
// ---------------------------------------------------------------------------

OneElectronShiftResult solve_one_electron_shift(
    const Eigen::MatrixXd& h, Eigen::MatrixXd coulomb, Eigen::MatrixXd exchange,
    double mu2, const Eigen::MatrixXd& xi, double num_electrons) {
  OneElectronShiftResult result;

  const std::size_t norb = static_cast<std::size_t>(h.rows());

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
  // exch(g~). See symmetry_shift_detail.hpp for the g~ definition and why this
  // stays consistent with rebuild_shifted_hamiltonian's full tensor.
  detail::add_coulomb_contraction(coulomb, mu2, xi);
  detail::add_exchange_contraction(exchange, mu2, xi);

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
// Top-level fermionic low-rank BLISS driver: wires steps 1-3 into a
// SymmetryShift.
// ---------------------------------------------------------------------------

SymmetryShift compute_fermionic_low_rank_shift(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons) {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian.is_restricted()) {
    throw std::invalid_argument(
        "compute_fermionic_low_rank_shift currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  if (!hamiltonian.has_container_type<
          qdk::chemistry::data::FactorizedHamiltonianContainer>()) {
    throw std::invalid_argument(
        "compute_fermionic_low_rank_shift requires an already double-"
        "factorized Hamiltonian (data::FactorizedHamiltonianContainer). Run "
        "the \"double_factorization\" hamiltonian_factorization algorithm on "
        "it first.");
  }

  const auto& container = hamiltonian.get_container<
      qdk::chemistry::data::FactorizedHamiltonianContainer>();

  // The dense reconstruction below ignores wB, so a nonzero one would be
  // silently dropped. Nothing in tree emits one today; reject rather than
  // guess at its meaning.
  if (!container.get_wb_matrix().isZero(0.0)) {
    throw std::invalid_argument(
        "compute_fermionic_low_rank_shift does not support a factorized "
        "Hamiltonian with a nonzero identity weight wB.");
  }

  const double num_electrons = static_cast<double>(n_alpha_electrons) +
                               static_cast<double>(n_beta_electrons);

  auto [h_alpha, h_beta] = hamiltonian.get_one_body_integrals();
  (void)h_beta;

  const size_t norb = static_cast<size_t>(h_alpha.rows());
  QDK_LOGGER().debug(
      "compute_fermionic_low_rank_shift: num_orbitals={}, num_electrons={}, "
      "num_ranks={}, num_copies={}",
      norb, num_electrons, container.get_num_ranks(),
      container.get_num_copies());

  // The mean-field contractions of the ORIGINAL g, taken straight from the
  // factorization instead of the norb^4 tensor:
  //   coulomb_ij  = Sum_k g[i,j,k,k] = Sum_rc tr(M^rc) M^rc_ij
  //   exchange_ij = Sum_k g[i,k,k,j] = Sum_rc (M^rc M^rc)_ij
  Eigen::MatrixXd coulomb = Eigen::MatrixXd::Zero(norb, norb);
  Eigen::MatrixXd exchange = Eigen::MatrixXd::Zero(norb, norb);
  {
    const Eigen::VectorXd& u = container.get_u_matrices();
    const Eigen::VectorXd& w = container.get_w_matrices();
    const size_t R = container.get_num_ranks();
    const size_t B = container.get_num_bases();
    const size_t C = container.get_num_copies();

    for (size_t r = 0; r < R; ++r) {
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>
          Ur(u.data() + r * B * norb, B, norb);
      for (size_t c = 0; c < C; ++c) {
        const Eigen::MatrixXd M = leaf_matrix(Ur, w, r, c, B, C, norb);
        coulomb += M.trace() * M;
        exchange.noalias() += M * M;
      }
    }
  }

  auto global_shift = accumulate_fragment_shifts(container);

  auto one_electron = solve_one_electron_shift(
      h_alpha, std::move(coulomb), std::move(exchange), global_shift.mu2,
      global_shift.xi, num_electrons);

  const double lambda_total_before =
      global_shift.lambda_df_baseline + one_electron.lambda_1e_baseline;
  const double lambda_total_after =
      global_shift.lambda_df_shifted + one_electron.lambda_1e;

  QDK_LOGGER().debug(
      "compute_fermionic_low_rank_shift: lambda_total before={} ({} + {}), "
      "after={} ({} + {}); lambda_DF baseline={}, shifted={}; lambda_1e "
      "baseline={}, "
      "shifted={}; mu1={}, mu2={}",
      lambda_total_before, one_electron.lambda_1e_baseline,
      global_shift.lambda_df_baseline, lambda_total_after,
      one_electron.lambda_1e, global_shift.lambda_df_shifted,
      global_shift.lambda_df_baseline, global_shift.lambda_df_shifted,
      one_electron.lambda_1e_baseline, one_electron.lambda_1e, one_electron.mu1,
      global_shift.mu2);

  // The two-body and one-body 1-norms are minimized sequentially, not jointly,
  // so the total can come out worse. Fall back to a zero shift, which leaves
  // the Hamiltonian unchanged.
  if (lambda_total_after > lambda_total_before) {
    QDK_LOGGER().warn(
        "compute_fermionic_low_rank_shift: the computed shift would increase "
        "the fermionic 1-norm (before={}, after={}); returning a zero shift, "
        "so the Hamiltonian is left unchanged.",
        lambda_total_before, lambda_total_after);

    SymmetryShift identity_shift;
    identity_shift.xi = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(norb),
                                              static_cast<Eigen::Index>(norb));
    return identity_shift;
  }

  SymmetryShift shift;
  shift.mu1 = one_electron.mu1;
  shift.mu2 = global_shift.mu2;
  shift.xi = global_shift.xi;
  return shift;
}

// ---------------------------------------------------------------------------
// FermionicLowRankShifter: the SymmetryShifter implementation.
// ---------------------------------------------------------------------------

SymmetryShift FermionicLowRankShifter::compute_shift(
    const data::Hamiltonian& hamiltonian, unsigned int n_alpha_electrons,
    unsigned int n_beta_electrons) const {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian.is_restricted()) {
    throw std::invalid_argument(
        "FermionicLowRankShifter currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  return compute_fermionic_low_rank_shift(hamiltonian, n_alpha_electrons,
                                          n_beta_electrons);
}

std::shared_ptr<data::Hamiltonian> FermionicLowRankShifter::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons) const {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian) {
    throw std::invalid_argument("FermionicLowRankShifter: hamiltonian is null");
  }

  const SymmetryShift shift =
      compute_shift(*hamiltonian, n_alpha_electrons, n_beta_electrons);
  const unsigned int num_electrons = n_alpha_electrons + n_beta_electrons;

  return rebuild_shifted_hamiltonian(*hamiltonian, shift, num_electrons);
}

}  // namespace qdk::chemistry::algorithms::microsoft
