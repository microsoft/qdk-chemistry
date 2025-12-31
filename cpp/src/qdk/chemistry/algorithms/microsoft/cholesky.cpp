// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "cholesky.hpp"

#include <qdk/chemistry/scf/core/basis_set.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <libint2.hpp>
#include <stdexcept>
#include <vector>

#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {
Eigen::MatrixXd pivoted_cholesky_decomposition(const Eigen::MatrixXd& matrix,
                                               double tolerance) {
  // validate input
  if (tolerance < 0.0 || std::isnan(tolerance) || std::isinf(tolerance)) {
    throw std::invalid_argument(
        "Tolerance must be a non-negative finite number");
  }
  if (matrix.rows() != matrix.cols()) {
    throw std::invalid_argument("Input matrix must be square");
  }

  // Copy to modify
  Eigen::MatrixXd M = matrix;

  // Cholesky vectors
  Eigen::MatrixXd L;
  size_t current_col = 0;

  // Pre-allocate with estimated size (number of aos) to avoid repeated
  // reallocations
  size_t n_aos = static_cast<size_t>(std::sqrt(M.rows()));
  L.resize(M.rows(), n_aos);

  // D_q = M_qq
  Eigen::VectorXd D = M.diagonal();

  // D_max, q_max = max(D)
  Eigen::Index q_max;
  double D_max = D.maxCoeff(&q_max);

  while (D_max >= tolerance) {
    // Q_max = sqrt(1 / D_max)
    double Q_max = std::sqrt(1 / D_max);

    // L = Q_max * M_q_max
    Eigen::VectorXd L_col = Q_max * M.col(q_max);

    // M = M - L_col * L_col^T
    M.noalias() -= L_col * L_col.transpose();

    // Append column (resize if needed)
    if (current_col >= static_cast<size_t>(L.cols())) {
      L.conservativeResize(Eigen::NoChange, L.cols() * 2);
    }
    L.col(current_col) = L_col;
    ++current_col;

    // D_q = M_qq
    D = M.diagonal();
    // D_max, q_max = max(D)
    D_max = D.maxCoeff(&q_max);
  }

  // Resize to actual size
  L.conservativeResize(Eigen::NoChange, current_col);
  return L;
}

Eigen::MatrixXd transform_cholesky_to_mo(
    const Eigen::MatrixXd& ao_cholesky_vectors,
    const Eigen::MatrixXd& mo_coeffs) {
  size_t n_ao = mo_coeffs.rows();
  size_t n_mo = mo_coeffs.cols();
  size_t rank = ao_cholesky_vectors.cols();

  // Validate dimensions
  if (n_ao == 0 || n_mo == 0) {
    throw std::invalid_argument("C matrix has zero dimensions");
  }
  if (ao_cholesky_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_cholesky_vectors dimensions do not match n_ao");
  }

  Eigen::MatrixXd mo_vectors(n_mo * n_mo, rank);

  // iterate over each Cholesky vector
  for (size_t k = 0; k < rank; ++k) {
    // Reshape the flat AO vector to a matrix (n_ao x n_ao)
    Eigen::Map<const Eigen::MatrixXd> V_ao(ao_cholesky_vectors.col(k).data(),
                                           n_ao, n_ao);

    // Transform from AO to MO basis: C^T * V_ao * C
    // Write directly to output column
    Eigen::Map<Eigen::MatrixXd> V_mo_map(mo_vectors.col(k).data(), n_mo, n_mo);
    V_mo_map.noalias() = mo_coeffs.transpose() * V_ao * mo_coeffs;
  }

  return mo_vectors;
}

Eigen::MatrixXd build_J_from_cholesky(
    const Eigen::MatrixXd& ao_cholesky_vectors,
    const Eigen::MatrixXd& density) {
  size_t n_ao = density.rows();
  size_t rank = ao_cholesky_vectors.cols();

  // Validate dimensions
  if (density.cols() != density.rows()) {
    throw std::invalid_argument("Density matrix must be square");
  }
  if (ao_cholesky_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_cholesky_vectors dimensions do not match density matrix");
  }

  // Initialize J
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n_ao, n_ao);
  for (size_t k = 0; k < rank; ++k) {
    // Reshape Cholesky vector to matrix (n_ao x n_ao)
    Eigen::Map<const Eigen::MatrixXd> L_k(ao_cholesky_vectors.col(k).data(),
                                          n_ao, n_ao);

    // V_k = P_\mu\nu L^k_\mu\nu
    double Vk = L_k.cwiseProduct(density).sum();

    // J_\lambda\sigma = L^k_\lambda\sigma * Vk
    J.noalias() += L_k * Vk;
  }
  return J;
}

Eigen::MatrixXd build_K_from_cholesky(
    const Eigen::MatrixXd& ao_cholesky_vectors, const Eigen::MatrixXd& coeffs,
    const std::vector<size_t>& occ_orb_ind) {
  size_t n_ao = coeffs.rows();
  size_t n_mo = coeffs.cols();
  size_t rank = ao_cholesky_vectors.cols();

  // Validate dimensions
  if (ao_cholesky_vectors.rows() != n_ao * n_ao) {
    throw std::invalid_argument(
        "ao_cholesky_vectors dimensions do not match density matrix");
  }

  // initermediates
  Eigen::MatrixXd L_sigma_i(n_ao * n_mo, rank);
  for (size_t k = 0; k < rank; ++k) {
    // Reshape Cholesky vector to matrix (n_ao x n_ao)
    Eigen::Map<const Eigen::MatrixXd> L_k(ao_cholesky_vectors.col(k).data(),
                                          n_ao, n_ao);

    // L^k\sigma,i = L^k_\mu\sigma * C_\mu,i
    L_sigma_i.col(k) = L_k * coeffs;
  }

  // Initialize K
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(n_ao, n_ao);
  for (size_t k = 0; k < rank; ++k) {
    // Temp_{\sigma,i} = L^k_{\sigma,i} for occupied orbitals
    Eigen::Map<const Eigen::MatrixXd> Temp(L_sigma_i.col(k).data(), n_ao, n_mo);

    // K_\lambda\sigma += \sum_{i}^{occ} Temp_\sigma,i * Temp_\lambda,i
    for (auto i : occ_orb_ind) {
      K.noalias() += Temp.col(i) * Temp.col(i).transpose();
    }
  }

  return K;
}

}  // namespace qdk::chemistry::algorithms::microsoft
