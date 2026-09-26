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

namespace qdk::chemistry::algorithms::microsoft {

// ---------------------------------------------------------------------------
// Step 2: one pass over the fragments producing the mean-field contractions
// of g and the global (mu2, xi).
// ---------------------------------------------------------------------------

namespace {

/// Read leaf M^r out of the flat [R,B,N] / [R,B] arrays (num_copies == 1).
///   M^r_{pq} = Sum_b W^r_b U^r_{bp} U^r_{bq}
/// `Ur` is [B,N] with eigenvectors in ROWS; the returned matrix is [N,N].
Eigen::MatrixXd leaf_matrix(
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>>& Ur,
    const Eigen::VectorXd& w, size_t r, size_t B, size_t norb) {
  Eigen::MatrixXd scaled(B, norb);
  for (size_t b = 0; b < B; ++b) {
    scaled.row(b) = w(r * B + b) * Ur.row(b);
  }
  return Ur.transpose() * scaled;
}

/// True when every rank's rotation U^r is a COMPLETE ORTHOGONAL rotation
/// (B == norb and U^r^T U^r == I), the condition under which Sum_b n_b^r ==
/// N_hat and so the shift -phi_r * N_hat can be absorbed into the fragment
/// eigenvalues. The container only checks that each basis row is a unit
/// vector, which is not sufficient.
bool fragments_span_full_rotation(
    const qdk::chemistry::data::FactorizedHamiltonianContainer& container,
    double tol = 1e-10) {
  const size_t norb = container.get_num_orbitals();
  const size_t R = container.get_num_ranks();
  const size_t B = container.get_num_bases();

  if (B != norb) {
    return false;
  }

  const Eigen::VectorXd& u = container.get_u_matrices();
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
      static_cast<Eigen::Index>(norb), static_cast<Eigen::Index>(norb));

  for (size_t r = 0; r < R; ++r) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>
        Ur(u.data() + r * B * norb, static_cast<Eigen::Index>(B),
           static_cast<Eigen::Index>(norb));
    const Eigen::MatrixXd gram = Ur.transpose() * Ur;
    if ((gram - identity).cwiseAbs().maxCoeff() > tol) {
      return false;
    }
  }
  return true;
}

// The two-body correction the aggregated shift (mu2, xi) adds to g:
//   dg_ijkl = -2*mu2*d_ij*d_kl - xi_ij*d_kl - d_ij*xi_kl
// Normal-ordered, raw-g transcription of Patel et al. Eq. 7; the mu2 term
// carries an extra factor 2 because that paper uses V = 1/2 g.

/// coul(g~) = coul(g) + Sum_k dg_ijkk, folded in place; nothing is allocated.
///   Sum_k dg_ijkk = -(2*mu2*norb + tr(xi))*d_ij - norb*xi_ij
void add_coulomb_contraction(Eigen::MatrixXd& coulomb, double mu2,
                             const Eigen::MatrixXd& xi) {
  const double norb = static_cast<double>(coulomb.rows());
  coulomb -= norb * xi;
  coulomb.diagonal().array() -= 2.0 * mu2 * norb + xi.trace();
}

/// exch(g~) = exch(g) + Sum_k dg_ikkj, folded in place.
///   Sum_k dg_ikkj = -2*mu2*d_ij - 2*xi_ij
void add_exchange_contraction(Eigen::MatrixXd& exchange, double mu2,
                              const Eigen::MatrixXd& xi) {
  exchange -= 2.0 * xi;
  exchange.diagonal().array() -= 2.0 * mu2;
}

}  // namespace

FragmentAccumulation accumulate_fragment_shifts(
    const qdk::chemistry::data::FactorizedHamiltonianContainer& container) {
  const size_t norb = container.get_num_orbitals();
  const size_t R = container.get_num_ranks();
  const size_t B = container.get_num_bases();

  FragmentAccumulation result(static_cast<Eigen::Index>(norb),
                              static_cast<Eigen::Index>(R));

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

    // One matrix product serves all three accumulations:
    //   coulomb_ij  = Sum_k g[i,j,k,k] = Sum_r tr(M^r) M^r
    //   exchange_ij = Sum_k g[i,k,k,j] = Sum_r (M^r M^r)_ij
    const Eigen::MatrixXd M = leaf_matrix(Ur, w, r, B, norb);
    result.coulomb += M.trace() * M;
    result.exchange.noalias() += M * M;

    Eigen::VectorXd eps(B);
    for (size_t b = 0; b < B; ++b) {
      eps(b) = kInvSqrt2 * w(r * B + b);
    }

    const double eps_abs_sum = eps.array().abs().sum();
    const double lambda_baseline = 0.5 * eps_abs_sum * eps_abs_sum;

    // Eq. 27's LP is minimized by every point of median_interval(eps). Take
    // the upper endpoint: it is an actual eps_i, which drops a unitary from
    // the one-electron LCU (text after Eq. 27).
    const double phi = median_interval(eps).second;
    result.phi(static_cast<Eigen::Index>(r)) = phi;

    result.lambda_df_baseline += lambda_baseline;
    const double eps_shifted_abs_sum = (eps.array() - phi).abs().sum();
    result.lambda_df_shifted += 0.5 * eps_shifted_abs_sum * eps_shifted_abs_sum;

    // Eq. 24's per-fragment BLISS parameters, negated: Eq. C6 *adds* the
    // per-fragment K^(a), while the global K is *subtracted* from H (Eq. 5).
    // Fragments in M (x) M form always have coefficient +1, so there is no
    // per-fragment sign to carry.
    result.mu2 -= phi * phi;

    // theta = -2*phi*eps rotated back is U^T diag(theta) U = -sqrt(2)*phi*M,
    // so M is all this needs; negating for H - K gives the +=.
    result.xi.noalias() += std::sqrt(2.0) * phi * M;
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

  // Baseline 1-norm of the ORIGINAL effective operator h + coul - 1/2 exch,
  // taken before the BLISS correction is folded in below. (syev overwrites its
  // input and reads only the lower triangle.)
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
  // exch(g~) for the g~ defined above.
  add_coulomb_contraction(coulomb, mu2, xi);
  add_exchange_contraction(exchange, mu2, xi);

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
  // lambda_1e is flat across the median interval, so take an endpoint: mu1 is
  // then an actual eigenvalue, dropping a unitary from the LCU.
  result.mu1 = median_interval(eigenvalues).second;
  result.lambda_1e = (eigenvalues.array() - result.mu1).abs().sum();

  return result;
}

// ---------------------------------------------------------------------------
// Top-level fermionic low-rank BLISS driver: wires steps 1-3 into a
// FermionicLowRankSolution.
// ---------------------------------------------------------------------------

FermionicLowRankSolution solve_fermionic_low_rank_shift(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons) {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian.is_restricted()) {
    throw std::invalid_argument(
        "solve_fermionic_low_rank_shift currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  if (!hamiltonian.has_container_type<
          qdk::chemistry::data::FactorizedHamiltonianContainer>()) {
    throw std::invalid_argument(
        "solve_fermionic_low_rank_shift requires an already double-"
        "factorized Hamiltonian (data::FactorizedHamiltonianContainer). Run "
        "the \"double_factorization\" hamiltonian_factorization algorithm on "
        "it first.");
  }

  const auto& container = hamiltonian.get_container<
      qdk::chemistry::data::FactorizedHamiltonianContainer>();

  // Neither the contractions below nor the rebuild account for wB, so a
  // nonzero one would be silently dropped.
  if (!container.get_wb_matrix().isZero(0.0)) {
    throw std::invalid_argument(
        "solve_fermionic_low_rank_shift does not support a factorized "
        "Hamiltonian with a nonzero identity weight wB.");
  }

  // Everything below reads W with the single-copy stride r*B + b and emits one
  // phi per rank, so a multi-copy factorization would be read wrong.
  if (container.get_num_copies() != 1) {
    throw std::invalid_argument(
        "solve_fermionic_low_rank_shift requires a factorized Hamiltonian "
        "with exactly one copy per rank (num_copies == 1), as produced by the "
        "\"double_factorization\" algorithm.");
  }

  // Without complete rotations the reported 1-norms would not describe the
  // shifted Hamiltonian, so the guard below could not catch the mistake.
  if (!fragments_span_full_rotation(container)) {
    throw std::invalid_argument(
        "solve_fermionic_low_rank_shift requires a factorization whose "
        "rotations are complete orthogonal rotations (num_bases == "
        "num_orbitals and U^T U == I for every rank), as produced by the "
        "\"double_factorization\" algorithm.");
  }

  const double num_electrons = static_cast<double>(n_alpha_electrons) +
                               static_cast<double>(n_beta_electrons);

  auto [h_alpha, h_beta] = hamiltonian.get_one_body_integrals();
  (void)h_beta;

  const size_t norb = static_cast<size_t>(h_alpha.rows());
  QDK_LOGGER().debug(
      "solve_fermionic_low_rank_shift: num_orbitals={}, num_electrons={}, "
      "num_ranks={}",
      norb, num_electrons, container.get_num_ranks());

  // One pass over the fragments produces both the mean-field contractions of
  // the ORIGINAL g and the global (mu2, xi).
  FragmentAccumulation accumulation = accumulate_fragment_shifts(container);

  const OneElectronShiftResult one_electron = solve_one_electron_shift(
      h_alpha, accumulation.coulomb, accumulation.exchange, accumulation.mu2,
      accumulation.xi, num_electrons);

  const double lambda_total_after =
      accumulation.lambda_df_shifted + one_electron.lambda_1e;

  const double lambda_total_before =
      accumulation.lambda_df_baseline + one_electron.lambda_1e_baseline;

  QDK_LOGGER().info(
      "solve_fermionic_low_rank_shift: lambda_total before={} ({} + {}), "
      "after={} ({} + {}); lambda_DF baseline={}, shifted={}; lambda_1e "
      "baseline={}, "
      "shifted={}; mu1={}, mu2={}",
      lambda_total_before, one_electron.lambda_1e_baseline,
      accumulation.lambda_df_baseline, lambda_total_after,
      one_electron.lambda_1e, accumulation.lambda_df_shifted,
      accumulation.lambda_df_baseline, accumulation.lambda_df_shifted,
      one_electron.lambda_1e_baseline, one_electron.lambda_1e, one_electron.mu1,
      accumulation.mu2);

  // The two-body and one-body 1-norms are minimized sequentially, not jointly,
  // so the total can come out worse. Fall back to a zero shift, which leaves
  // the Hamiltonian unchanged.
  if (lambda_total_after > lambda_total_before) {
    QDK_LOGGER().warn(
        "solve_fermionic_low_rank_shift: the computed shift would increase "
        "the fermionic 1-norm (before={}, after={}); returning a zero shift, "
        "so the Hamiltonian is left unchanged.",
        lambda_total_before, lambda_total_after);

    SymmetryShiftCoeffs identity_shift;
    identity_shift.xi = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(norb),
                                              static_cast<Eigen::Index>(norb));
    return {identity_shift, Eigen::VectorXd::Zero(accumulation.phi.size())};
  }

  SymmetryShiftCoeffs shift;
  shift.mu1 = one_electron.mu1;
  shift.mu2 = accumulation.mu2;
  shift.xi = accumulation.xi;
  return {shift, std::move(accumulation.phi)};
}

// ---------------------------------------------------------------------------
// Applying the solution: the shift is absorbed into the fragment eigenvalues,
// so the output stays a sum of squares over the SAME rotations.
// ---------------------------------------------------------------------------

std::shared_ptr<qdk::chemistry::data::Hamiltonian>
rebuild_shifted_factorized_hamiltonian(
    const qdk::chemistry::data::Hamiltonian& original,
    const qdk::chemistry::data::FactorizedHamiltonianContainer& container,
    const FermionicLowRankSolution& solution, unsigned int num_electrons) {
  QDK_LOG_TRACE_ENTERING();

  const SymmetryShiftCoeffs& shift = solution.shift;

  auto [h_alpha, h_beta] = original.get_one_body_integrals();
  (void)h_beta;

  const Eigen::Index norb = h_alpha.rows();
  if (shift.xi.rows() != norb || shift.xi.cols() != norb) {
    throw std::invalid_argument(
        "rebuild_shifted_factorized_hamiltonian: shift.xi must be norb x "
        "norb.");
  }

  const size_t R = container.get_num_ranks();
  const size_t B = container.get_num_bases();

  if (container.get_num_copies() != 1) {
    throw std::invalid_argument(
        "rebuild_shifted_factorized_hamiltonian requires num_copies == 1.");
  }

  if (solution.phi.size() != static_cast<Eigen::Index>(R)) {
    throw std::invalid_argument(
        "rebuild_shifted_factorized_hamiltonian: phi must have one entry per "
        "rank.");
  }

  const double ne = static_cast<double>(num_electrons);

  // One-body part: h~_ij = h_ij + (Ne-1)*xi_ij - (mu1+mu2)*delta_ij.
  Eigen::MatrixXd h_tilde = h_alpha + (ne - 1.0) * shift.xi;
  h_tilde.diagonal().array() -= (shift.mu1 + shift.mu2);

  // Two-body part: subtracting phi_r from each fragment's eigenvalues is the
  // whole shift; sqrt(2) converts back to the stored scale.
  Eigen::VectorXd w_new = container.get_w_matrices();
  for (size_t r = 0; r < R; ++r) {
    const double delta =
        std::sqrt(2.0) * solution.phi(static_cast<Eigen::Index>(r));
    for (size_t b = 0; b < B; ++b) {
      w_new[static_cast<Eigen::Index>(r * B + b)] -= delta;
    }
  }

  // Constant part of -K in the Ne-electron sector: +mu1*Ne + mu2*Ne^2.
  const double core_energy_new =
      original.get_core_energy() + shift.mu1 * ne + shift.mu2 * ne * ne;

  const Eigen::MatrixXd inactive_fock =
      original.has_inactive_fock_matrix()
          ? original.get_inactive_fock_matrix().first
          : Eigen::MatrixXd(0, 0);

  auto shifted =
      std::make_unique<qdk::chemistry::data::FactorizedHamiltonianContainer>(
          h_tilde, container.get_u_matrices(), w_new, container.get_wb_matrix(),
          original.get_orbitals(), core_energy_new, inactive_fock,
          original.get_type());

  return std::make_shared<qdk::chemistry::data::Hamiltonian>(
      std::move(shifted));
}

// ---------------------------------------------------------------------------
// FermionicLowRankShifter: the SymmetryShifter implementation.
// ---------------------------------------------------------------------------

std::shared_ptr<data::Hamiltonian> FermionicLowRankShifter::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons) const {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian) {
    throw std::invalid_argument("FermionicLowRankShifter: hamiltonian is null");
  }

  if (!hamiltonian->is_restricted()) {
    throw std::invalid_argument(
        "FermionicLowRankShifter currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  const FermionicLowRankSolution solution = solve_fermionic_low_rank_shift(
      *hamiltonian, n_alpha_electrons, n_beta_electrons);
  _record_shift(solution.shift);

  // solve_fermionic_low_rank_shift() has already validated the container.
  const auto& container = hamiltonian->get_container<
      qdk::chemistry::data::FactorizedHamiltonianContainer>();

  return rebuild_shifted_factorized_hamiltonian(
      *hamiltonian, container, solution, n_alpha_electrons + n_beta_electrons);
}

}  // namespace qdk::chemistry::algorithms::microsoft
