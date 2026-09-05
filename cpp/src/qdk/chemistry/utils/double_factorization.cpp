// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <lapack.hh>
#include <limits>
#include <qdk/chemistry/utils/double_factorization.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>

namespace qdk::chemistry::utils {

namespace {

inline size_t two_body_index(size_t i, size_t j, size_t k, size_t l,
                             size_t norb) {
  return i * norb * norb * norb + j * norb * norb + k * norb + l;
}

/// Diagonalize a fragment's (symmetric) norb x norb matrix and package it as a
/// TwoBodyFragment. `matrix` is consumed. `eps_scale` multiplies the resulting
/// eigenvalues (used by the eigendecomposition path to fold in
/// sqrt(|supermatrix eigenvalue|)).
TwoBodyFragment make_fragment(Eigen::MatrixXd matrix, double sign,
                              double eps_scale, size_t norb) {
  Eigen::VectorXd fragment_eigenvalues(norb);
  const int64_t fragment_info = lapack::syev(
      lapack::Job::Vec, lapack::Uplo::Lower, static_cast<int64_t>(norb),
      matrix.data(), static_cast<int64_t>(norb), fragment_eigenvalues.data());
  if (fragment_info != 0) {
    throw std::runtime_error(
        "double_factorize: LAPACK syev failed to diagonalize a fragment "
        "matrix (info=" +
        std::to_string(fragment_info) + ").");
  }

  TwoBodyFragment fragment;
  fragment.U = std::move(matrix);
  fragment.sign = sign;
  fragment.eps = eps_scale * fragment_eigenvalues;

  const double eps_abs_sum = fragment.eps.array().abs().sum();
  fragment.lambda_df = 0.5 * eps_abs_sum * eps_abs_sum;
  return fragment;
}

/// Eigendecomposition path: diagonalize the supermatrix and turn every
/// eigenpair above the threshold into a fragment.
std::vector<TwoBodyFragment> factorize_by_eigendecomposition(
    const Eigen::MatrixXd& supermatrix, size_t norb,
    double truncation_threshold) {
  const size_t pair_dim = norb * norb;

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

    fragments.push_back(make_fragment(std::move(fragment_matrix),
                                      (eigenvalue >= 0.0) ? 1.0 : -1.0,
                                      std::sqrt(std::abs(eigenvalue)), norb));
  }

  return fragments;
}

/// Pivoted Cholesky path, yielding supermatrix ~= sum_r L^r (L^r)^T. Because
/// rows (ij) and (ji) of the supermatrix are identical (g_ijkl = g_jikl), each
/// L^r reshapes to a symmetric norb x norb matrix, so the per-fragment
/// diagonalization downstream is unchanged.
///
/// Returns false if the supermatrix is detected to be indefinite, in which
/// case the caller must fall back to the eigendecomposition. The supermatrix
/// is never modified, so that fallback is free. Breakdown must be detected on
/// the MINIMUM residual diagonal, not on the pivot: the pivot is the largest
/// remaining diagonal, which merely decays to zero and terminates the loop
/// normally while negative directions go unnoticed.
bool factorize_by_cholesky(const Eigen::MatrixXd& supermatrix, size_t norb,
                           double truncation_threshold,
                           std::vector<TwoBodyFragment>& fragments) {
  const size_t pair_dim = norb * norb;

  Eigen::VectorXd residual_diagonal = supermatrix.diagonal();

  // A literal 0.0 threshold is never reached in floating point: past the true
  // numerical rank the residual diagonal decays into roundoff. Floor the
  // cutoff so that "lossless" stops at the true numerical rank instead.
  const double epsilon = std::numeric_limits<double>::epsilon();
  const double stop_threshold = std::max(truncation_threshold, epsilon);
  const double indefinite_tolerance = epsilon;

  std::vector<Eigen::VectorXd> cholesky_vectors;
  // The supermatrix is invariant under (ij) -> (ji), so every antisymmetric
  // pair vector is in its null space and the rank cannot exceed the
  // symmetric-pair dimension.
  cholesky_vectors.reserve(norb * (norb + 1) / 2);

  for (size_t step = 0; step < pair_dim; ++step) {
    Eigen::Index pivot = 0;
    Eigen::Index most_negative = 0;
    const double pivot_value = residual_diagonal.maxCoeff(&pivot);
    const double minimum_value = residual_diagonal.minCoeff(&most_negative);

    if (minimum_value < -indefinite_tolerance) {
      QDK_LOGGER().warn(
          "double_factorize: the two-body supermatrix is not positive "
          "semi-definite (residual diagonal {} at pair index {} after {} "
          "Cholesky steps, tolerance {}). Falling back to eigendecomposition, "
          "which supports indefinite input and may return fragments with "
          "sign = -1; note that lambda_df differs between the two methods.",
          minimum_value, static_cast<size_t>(most_negative), step,
          -indefinite_tolerance);
      return false;
    }

    if (pivot_value <= stop_threshold) {
      break;
    }

    // Residual column, formed on the fly so the supermatrix stays intact and
    // the fallback above costs nothing: r = M[:,q] - sum_s L^s L^s_q.
    Eigen::VectorXd column = supermatrix.col(pivot);
    for (const auto& vector : cholesky_vectors) {
      column -= vector * vector[pivot];
    }
    column /= std::sqrt(pivot_value);

    residual_diagonal -= column.array().square().matrix();
    cholesky_vectors.push_back(std::move(column));
  }

  fragments.clear();
  fragments.reserve(cholesky_vectors.size());
  for (const auto& vector : cholesky_vectors) {
    // L^r is symmetric by construction; symmetrize defensively against
    // accumulated roundoff, as the eigendecomposition path does.
    Eigen::MatrixXd fragment_matrix(norb, norb);
    for (size_t i = 0; i < norb; ++i) {
      for (size_t j = 0; j < norb; ++j) {
        fragment_matrix(i, j) = vector[i * norb + j];
      }
    }
    fragment_matrix = 0.5 * (fragment_matrix + fragment_matrix.transpose());

    // Cholesky of a positive semi-definite matrix yields only positive
    // rank-one terms, so every fragment has sign = +1.
    fragments.push_back(
        make_fragment(std::move(fragment_matrix), 1.0, 1.0, norb));
  }

  // Cholesky emits fragments in pivot order, which is not the "largest
  // contribution first" order the API promises. Sort by the fragment's own
  // 1-norm weight so callers can truncate by fragment count meaningfully.
  std::sort(fragments.begin(), fragments.end(),
            [](const TwoBodyFragment& a, const TwoBodyFragment& b) {
              return a.lambda_df > b.lambda_df;
            });

  return true;
}

}  // namespace

std::vector<TwoBodyFragment> double_factorize(
    const Eigen::VectorXd& two_body_integrals, size_t norb,
    double truncation_threshold, DoubleFactorizationMethod method) {
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

  if (method == DoubleFactorizationMethod::Cholesky) {
    std::vector<TwoBodyFragment> fragments;
    if (factorize_by_cholesky(supermatrix, norb, truncation_threshold,
                              fragments)) {
      return fragments;
    }
    // Indefinite supermatrix: fall back (already warned).
  }

  return factorize_by_eigendecomposition(supermatrix, norb,
                                         truncation_threshold);
}

}  // namespace qdk::chemistry::utils
