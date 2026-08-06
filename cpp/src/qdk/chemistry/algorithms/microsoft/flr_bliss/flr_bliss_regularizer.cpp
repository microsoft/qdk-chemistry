// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "flr_bliss_regularizer.hpp"

#include <lapack.hh>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/utils/logger.hpp>

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
    // (mu2, xi) that rebuild_hamiltonian applies are the NEGATED sum of the
    // per-fragment K^(a) parameters. (Fragments come from double-factorizing
    // the PHYSICAL coefficient 1/2 g, so mu2/xi are already on the correct
    // scale for rebuild_hamiltonian.)
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

  // Contractions of the BLISS-shifted tensor
  //   g~ = g - 2*mu2*d*d - xi*d - d*xi   (matching rebuild_hamiltonian):
  //   sum_k g~[i,j,k,k] = coul + coulomb_contraction
  //   sum_k g~[i,k,k,j] = exch + exchange_contraction
  // TwoBodyBlissCorrection is the SAME struct rebuild_hamiltonian() uses to
  // build the full g~ tensor, so these two closed-form contractions are
  // guaranteed consistent with the g~ actually written into the shifted
  // Hamiltonian -- see the top of this file for the index-algebra
  // derivation and test_hamiltonian_regularizer.cpp for a brute-force
  // numerical check.
  const TwoBodyBlissCorrection correction{mu2, xi};
  const Eigen::MatrixXd coulomb_shifted =
      coulomb + correction.coulomb_contraction(static_cast<Eigen::Index>(norb));
  const Eigen::MatrixXd exchange_shifted =
      exchange +
      correction.exchange_contraction(static_cast<Eigen::Index>(norb));

  // Effective one-electron tensor of H - K with mu1 = 0 (Eq. 14):
  //   Heff0 = h + (Ne-1)*xi - mu2*I + coul(g~) - 1/2 exch(g~).
  const Eigen::MatrixXd h0 =
      h + (num_electrons - 1.0) * xi - mu2 * identity;
  Eigen::MatrixXd effective_one_body =
      h0 + coulomb_shifted - 0.5 * exchange_shifted;

  // Baseline: 1-norm of the ORIGINAL effective one-electron operator
  //   Heff = h + coul - 1/2 exch  (no BLISS shift), for before/after reporting.
  // lapack::syev overwrites its input buffer in place and only reads the
  // lower triangle (matching Eigen::SelfAdjointEigenSolver's default
  // convention); Job::NoVec skips eigenvector computation since only the
  // eigenvalues are needed here.
  Eigen::MatrixXd effective_one_body_original =
      h + coulomb - 0.5 * exchange;
  Eigen::VectorXd eigenvalues_baseline(norb);
  lapack::syev(lapack::Job::NoVec, lapack::Uplo::Lower,
               static_cast<int64_t>(norb), effective_one_body_original.data(),
               static_cast<int64_t>(norb), eigenvalues_baseline.data());
  result.lambda_1e_baseline = eigenvalues_baseline.array().abs().sum();

  // Eq. 23's LP has a closed-form solution: mu1 = median{eigenvalues}.
  Eigen::VectorXd eigenvalues(norb);
  lapack::syev(lapack::Job::NoVec, lapack::Uplo::Lower,
               static_cast<int64_t>(norb), effective_one_body.data(),
               static_cast<int64_t>(norb), eigenvalues.data());
  result.mu1 = median(eigenvalues);
  result.lambda_1e = (eigenvalues.array() - result.mu1).abs().sum();

  // h + Ne*xi, kept so rebuild_hamiltonian can recover h = h_prime - Ne*xi.
  result.h_prime = h + num_electrons * xi;

  return result;
}

// ---------------------------------------------------------------------------
// Step 4: apply (mu1, mu2, xi) to the dense integrals and rebuild H.
// ---------------------------------------------------------------------------

std::shared_ptr<qdk::chemistry::data::Hamiltonian> rebuild_hamiltonian(
    const qdk::chemistry::data::Hamiltonian& original,
    const Eigen::MatrixXd& h_prime, double mu1, double mu2,
    const Eigen::MatrixXd& xi, const Eigen::VectorXd& two_body_integrals,
    double num_electrons) {
  using qdk::chemistry::data::CanonicalFourCenterHamiltonianContainer;
  using qdk::chemistry::data::Hamiltonian;

  const size_t norb = static_cast<size_t>(h_prime.rows());

  // The BLISS shift operator (subtracted from H) is
  //   K = mu1*(N - Ne) + mu2*(N^2 - Ne^2) + (N - Ne)*sum_ij xi_ij E_ij,
  // which annihilates every Ne-electron state, so H_tilde = H - K leaves the
  // Ne-sector energy invariant for ANY (mu1, mu2, xi). Expanding -K into the
  // canonical chemist integrals g[i,j,k,l] = (ij|kl) of this container gives
  // the shifts below. These were derived directly in the container's own
  // convention and verified to machine precision against explicit
  // single-determinant energies -- NOT copied from Appendix C / the qdk BLISS
  // helper, whose (different) two-body normalisation would introduce spurious
  // factors of 2 here.
  //
  // CAUTION: h_prime carries h_ij + Ne*xi_ij, which is only Appendix C's
  // *effective* one-body operator (the number operator that multiplies xi is
  // replaced by its eigenvalue Ne to estimate the 1-norm and optimise mu1); it
  // is NOT the physical one-body tensor. Recover the true integrals first.
  const Eigen::MatrixXd h = h_prime - num_electrons * xi;

  // One-body part of -K:
  //   h_tilde_ij = h_ij + (Ne - 1)*xi_ij - (mu1 + mu2)*delta_ij
  Eigen::MatrixXd h_tilde = h + (num_electrons - 1.0) * xi;
  h_tilde.diagonal().array() -= (mu1 + mu2);

  // Two-body part of -K:
  //   g_tilde_ijkl = g_ijkl - 2*mu2*delta_ij*delta_kl
  //                          - xi_ij*delta_kl - delta_ij*xi_kl
  // TwoBodyBlissCorrection::full_tensor() is the single source of truth for
  // this dg_ijkl tensor -- solve_one_electron_shift() derives its Coulomb/
  // exchange-type contractions of the SAME tensor from the same struct, so
  // the two steps cannot silently drift apart.
  const TwoBodyBlissCorrection correction{mu2, xi};
  const Eigen::VectorXd g_tilde =
      two_body_integrals +
      correction.full_tensor(static_cast<Eigen::Index>(norb));

  // Constant part of -K in the Ne-electron sector: +mu1*Ne + mu2*Ne^2.
  const double core_energy_new = original.get_core_energy() +
                                 mu1 * num_electrons +
                                 mu2 * num_electrons * num_electrons;

  const Eigen::MatrixXd inactive_fock =
      original.has_inactive_fock_matrix()
          ? original.get_inactive_fock_matrix().first
          : Eigen::MatrixXd(0, 0);

  auto container = std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      h_tilde, g_tilde, original.get_orbitals(), core_energy_new,
      inactive_fock, original.get_type());

  return std::make_shared<Hamiltonian>(std::move(container));
}

}  // namespace qdk::chemistry::algorithms::microsoft::flr_bliss

namespace qdk::chemistry::algorithms::microsoft {

// ---------------------------------------------------------------------------
// Top-level driver: wires together the four steps above.
// ---------------------------------------------------------------------------

std::shared_ptr<qdk::chemistry::data::Hamiltonian>
FlrBlissRegularizer::_run_impl(
    std::shared_ptr<qdk::chemistry::data::Hamiltonian> hamiltonian,
    unsigned int n_alpha_electrons, unsigned int n_beta_electrons) const {
  QDK_LOG_TRACE_ENTERING();

  if (!hamiltonian) {
    throw std::invalid_argument("FlrBlissRegularizer: hamiltonian is null");
  }
  if (!hamiltonian->is_restricted()) {
    throw std::invalid_argument(
        "FlrBlissRegularizer currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  const double df_truncation_threshold =
      _settings->get<double>("df_truncation_threshold");
  const double num_electrons =
      static_cast<double>(n_alpha_electrons) + static_cast<double>(n_beta_electrons);

  auto [h_alpha, h_beta] = hamiltonian->get_one_body_integrals();
  (void)h_beta;
  auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian->get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;

  const size_t norb = static_cast<size_t>(h_alpha.rows());
  QDK_LOGGER().debug(
      "FlrBlissRegularizer: num_orbitals={}, num_electrons={}, "
      "df_truncation_threshold={}",
      norb, num_electrons, df_truncation_threshold);

  const Eigen::VectorXd two_body_coefficient = 0.5 * g_aaaa;
  auto fragments = qdk::chemistry::utils::double_factorize(
      two_body_coefficient, norb, df_truncation_threshold);

  auto global_shift = flr_bliss::accumulate_fragment_shifts(fragments);

  auto one_electron = flr_bliss::solve_one_electron_shift(
      h_alpha, g_aaaa, global_shift.mu2, global_shift.xi, num_electrons);

  auto shifted_hamiltonian = flr_bliss::rebuild_hamiltonian(
      *hamiltonian, one_electron.h_prime, one_electron.mu1, global_shift.mu2,
      global_shift.xi, g_aaaa, num_electrons);

  const double lambda_total_before =
      global_shift.lambda_df_baseline + one_electron.lambda_1e_baseline;
  const double lambda_total_after =
      global_shift.lambda_df_shifted + one_electron.lambda_1e;

  QDK_LOGGER().debug(
      "FlrBlissRegularizer: lambda_total before={} ({} + {}), after={} ({} + "
      "{}); lambda_DF baseline={}, shifted={}; lambda_1e baseline={}, "
      "shifted={}; mu1={}, mu2={}",
      lambda_total_before, one_electron.lambda_1e_baseline,
      global_shift.lambda_df_baseline, lambda_total_after,
      one_electron.lambda_1e, global_shift.lambda_df_shifted,
      global_shift.lambda_df_baseline, global_shift.lambda_df_shifted,
      one_electron.lambda_1e_baseline, one_electron.lambda_1e,
      one_electron.mu1, global_shift.mu2);

  return shifted_hamiltonian;
}

}  // namespace qdk::chemistry::algorithms::microsoft
