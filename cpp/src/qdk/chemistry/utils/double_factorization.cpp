// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <lapack.hh>
#include <qdk/chemistry/utils/double_factorization.hpp>
#include <stdexcept>
#include <string>

namespace qdk::chemistry::utils {

namespace {

inline size_t two_body_index(size_t i, size_t j, size_t k, size_t l,
                             size_t norb) {
  return i * norb * norb * norb + j * norb * norb + k * norb + l;
}

}  // namespace

std::vector<TwoBodyFragment> double_factorize(
    const Eigen::VectorXd& two_body_integrals, size_t norb,
    double truncation_threshold) {
  const size_t pair_dim = norb * norb;

  // Reshape g_ijkl into the (ij),(kl) supermatrix.
  Eigen::MatrixXd supermatrix(pair_dim, pair_dim);
  for (size_t i = 0; i < norb; ++i) {
    for (size_t j = 0; j < norb; ++j) {
      const size_t row = i * norb + j;
      for (size_t k = 0; k < norb; ++k) {
        for (size_t l = 0; l < norb; ++l) {
          const size_t col = k * norb + l;
          supermatrix(row, col) =
              two_body_integrals[two_body_index(i, j, k, l, norb)];
        }
      }
    }
  }
  // Defensive symmetrization against numerical noise in the input tensor.
  supermatrix = 0.5 * (supermatrix + supermatrix.transpose());

  Eigen::MatrixXd supermatrix_eigenvectors = supermatrix;
  Eigen::VectorXd supermatrix_eigenvalues(pair_dim);
  // lapack::syev overwrites its input buffer in place with the eigenvectors
  // and only reads the lower triangle (matching Eigen::SelfAdjointEigenSolver's
  // default convention). Job::Vec is needed here since the eigenvectors are
  // used below to build each fragment's orbital matrix.
  const int64_t supermatrix_info = lapack::syev(
      lapack::Job::Vec, lapack::Uplo::Lower, static_cast<int64_t>(pair_dim),
      supermatrix_eigenvectors.data(), static_cast<int64_t>(pair_dim),
      supermatrix_eigenvalues.data());
  if (supermatrix_info != 0) {
    throw std::runtime_error(
        "double_factorize: LAPACK syev failed to diagonalize the two-body "
        "supermatrix (info=" +
        std::to_string(supermatrix_info) + ").");
  }

  // Process fragment candidates by decreasing |eigenvalue| so the largest
  // contributions are retained first if a caller wants to further truncate
  // by fragment count.
  std::vector<size_t> order(pair_dim);
  for (size_t n = 0; n < pair_dim; ++n) {
    order[n] = n;
  }
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return std::abs(supermatrix_eigenvalues[a]) >
           std::abs(supermatrix_eigenvalues[b]);
  });

  std::vector<TwoBodyFragment> fragments;
  fragments.reserve(pair_dim);
  for (size_t n : order) {
    const double eigenvalue = supermatrix_eigenvalues[n];
    if (std::abs(eigenvalue) < truncation_threshold) {
      continue;
    }

    // Reshape the eigenvector into an norb x norb matrix. For a
    // non-degenerate eigenvalue this matrix is automatically symmetric
    // because g_ijkl = g_jikl; symmetrize defensively to guard against
    // degenerate subspaces / numerical noise.
    Eigen::MatrixXd fragment_matrix(norb, norb);
    for (size_t i = 0; i < norb; ++i) {
      for (size_t j = 0; j < norb; ++j) {
        fragment_matrix(i, j) = supermatrix_eigenvectors(i * norb + j, n);
      }
    }
    fragment_matrix = 0.5 * (fragment_matrix + fragment_matrix.transpose());

    Eigen::MatrixXd fragment_eigenvectors = fragment_matrix;
    Eigen::VectorXd fragment_eigenvalues(norb);
    const int64_t fragment_info = lapack::syev(
        lapack::Job::Vec, lapack::Uplo::Lower, static_cast<int64_t>(norb),
        fragment_eigenvectors.data(), static_cast<int64_t>(norb),
        fragment_eigenvalues.data());
    if (fragment_info != 0) {
      throw std::runtime_error(
          "double_factorize: LAPACK syev failed to diagonalize a fragment "
          "matrix (info=" +
          std::to_string(fragment_info) + ").");
    }

    TwoBodyFragment fragment;
    fragment.U = fragment_eigenvectors;
    fragment.sign = (eigenvalue >= 0.0) ? 1.0 : -1.0;
    fragment.eps = std::sqrt(std::abs(eigenvalue)) * fragment_eigenvalues;

    const double eps_abs_sum = fragment.eps.array().abs().sum();
    fragment.lambda_df = 0.5 * eps_abs_sum * eps_abs_sum;

    fragments.push_back(std::move(fragment));
  }

  return fragments;
}

}  // namespace qdk::chemistry::utils
