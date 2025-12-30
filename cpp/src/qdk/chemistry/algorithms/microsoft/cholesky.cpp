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

}  // namespace qdk::chemistry::algorithms::microsoft
