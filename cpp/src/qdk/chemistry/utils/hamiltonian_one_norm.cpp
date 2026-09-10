// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstddef>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/utils/double_factorization.hpp>
#include <qdk/chemistry/utils/hamiltonian_one_norm.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace qdk::chemistry::utils {

HamiltonianOneNorm hamiltonian_one_norm(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    double df_truncation_threshold, DoubleFactorizationMethod method) {
  if (!hamiltonian.is_restricted()) {
    throw std::runtime_error(
        "hamiltonian_one_norm currently only supports restricted "
        "(spin-restricted) Hamiltonians");
  }

  auto [h_alpha, h_beta] = hamiltonian.get_one_body_integrals();
  (void)h_beta;

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  // Double-factorize the PHYSICAL two-electron coefficient V = 1/2 * g into
  // low-rank fragments; each contributes Eq. (17)'s
  // lambda_DF^(alpha) = 1/2 (sum_i |eps_i^(alpha)|)^2. Note that the value
  // depends on `method`, not just its cost -- see the header.
  std::vector<TwoBodyFragment> fragments;
  Eigen::MatrixXd coulomb;
  Eigen::MatrixXd exchange;
  if (hamiltonian.has_container_type<data::CholeskyHamiltonianContainer>()) {
    const auto& [l_alpha, l_beta] =
        hamiltonian.get_container<data::CholeskyHamiltonianContainer>()
            .get_three_center_integrals();
    (void)l_beta;
    const Eigen::MatrixXd half = l_alpha / std::sqrt(2.0);
    fragments = double_factorize_three_center(half, norb,
                                              df_truncation_threshold, method);
    std::tie(coulomb, exchange) =
        mean_field_contractions_three_center(l_alpha, norb);
  } else {
    auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian.get_two_body_integrals();
    (void)g_aabb;
    (void)g_bbbb;
    const Eigen::VectorXd two_body_coefficient = 0.5 * g_aaaa;
    fragments = double_factorize(two_body_coefficient, norb,
                                 df_truncation_threshold, method);
    std::tie(coulomb, exchange) = mean_field_contractions(g_aaaa, norb);
  }

  // Effective one-electron tensor (Eq. 14), in the container's chemist
  // convention g[i,j,k,l] = (ij|kl):
  //   Heff_ij = h_ij + sum_k g[i,j,k,k] - 1/2 sum_k g[i,k,k,j]
  // (the Coulomb and exchange contractions of the two-electron tensor that
  // arise when folding the DF fragment one-electron corrections into H_1e).
  const Eigen::MatrixXd effective_one_body = h_alpha + coulomb - 0.5 * exchange;

  HamiltonianOneNorm result;

  // lambda_1e = sum_i |gamma_i| (Eq. 15), with gamma_i the eigenvalues of the
  // (symmetric) effective one-electron tensor.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(effective_one_body);
  if (solver.info() != Eigen::Success) {
    throw std::runtime_error(
        "hamiltonian_one_norm: failed to diagonalize the effective "
        "one-electron operator.");
  }
  result.one_body = solver.eigenvalues().array().abs().sum();

  double lambda_two_body = 0.0;
  for (const auto& fragment : fragments) {
    lambda_two_body += fragment.lambda_df;
  }
  result.two_body = lambda_two_body;

  result.total = result.one_body + result.two_body;
  return result;
}

}  // namespace qdk::chemistry::utils
