// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Eigenvalues>
#include <cstddef>
#include <qdk/chemistry/utils/double_factorization.hpp>
#include <qdk/chemistry/utils/hamiltonian_one_norm.hpp>
#include <stdexcept>

namespace qdk::chemistry::utils {

namespace {

inline std::size_t two_body_index(std::size_t i, std::size_t j, std::size_t k,
                                  std::size_t l, std::size_t norb) {
  return ((i * norb + j) * norb + k) * norb + l;
}

}  // namespace

HamiltonianOneNorm hamiltonian_one_norm(
    const qdk::chemistry::data::Hamiltonian& hamiltonian,
    double df_truncation_threshold) {
  if (!hamiltonian.is_restricted()) {
    throw std::runtime_error(
        "hamiltonian_one_norm currently only supports restricted "
        "(spin-restricted) Hamiltonians");
  }

  auto [h_alpha, h_beta] = hamiltonian.get_one_body_integrals();
  (void)h_beta;
  auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian.get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  // Effective one-electron tensor (Eq. 14), in the container's chemist
  // convention g[i,j,k,l] = (ij|kl):
  //   Heff_ij = h_ij + sum_k g[i,j,k,k] - 1/2 sum_k g[i,k,k,j]
  // (the Coulomb and exchange contractions of the two-electron tensor that
  // arise when folding the DF fragment one-electron corrections into H_1e).
  Eigen::MatrixXd effective_one_body = h_alpha;
  for (std::size_t i = 0; i < norb; ++i) {
    for (std::size_t j = 0; j < norb; ++j) {
      double coulomb = 0.0;
      double exchange = 0.0;
      for (std::size_t k = 0; k < norb; ++k) {
        coulomb += g_aaaa[two_body_index(i, j, k, k, norb)];
        exchange += g_aaaa[two_body_index(i, k, k, j, norb)];
      }
      effective_one_body(i, j) += coulomb - 0.5 * exchange;
    }
  }

  HamiltonianOneNorm result;

  // lambda_1e = sum_i |gamma_i| (Eq. 15), with gamma_i the eigenvalues of the
  // (symmetric) effective one-electron tensor.
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(effective_one_body);
  result.one_body = solver.eigenvalues().array().abs().sum();

  // lambda_2e: double-factorize the PHYSICAL two-electron coefficient
  // V = 1/2 * g into low-rank fragments; each contributes Eq. (17)'s
  // lambda_DF^(alpha) = 1/2 (sum_i |eps_i^(alpha)|)^2.
  Eigen::VectorXd two_body_coefficient = 0.5 * g_aaaa;
  auto fragments =
      double_factorize(two_body_coefficient, norb, df_truncation_threshold);
  double lambda_two_body = 0.0;
  for (const auto& fragment : fragments) {
    lambda_two_body += fragment.lambda_df;
  }
  result.two_body = lambda_two_body;

  result.total = result.one_body + result.two_body;
  return result;
}

}  // namespace qdk::chemistry::utils
