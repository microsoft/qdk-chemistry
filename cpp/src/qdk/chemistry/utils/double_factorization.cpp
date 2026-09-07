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
#include <utility>
#include <vector>

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

/// Pivoted Cholesky path, yielding supermatrix ~= sum_r L^r (L^r)^T.
///
/// The factorization is carried out in the SYMMETRIC-PAIR basis (i <= j), not
/// in the full (ij) basis, because g_ijkl = g_jikl makes rows (ij) and (ji) of
/// the supermatrix *identical*. Those n(n-1)/2 duplicate directions are exact
/// zero modes, and eliminating a pivot (ij) drives the residual diagonal of
/// its mirror (ji) to a mathematical zero that is computed as the cancellation
/// of two equal O(||M||) numbers. The result lands at +/- eps*||M|| with an
/// arbitrary sign, which is fatal if the mirror is left in the problem: a
/// positive value gets selected as the next pivot and its column is divided by
/// sqrt(eps*||M||), amplifying pure roundoff into a spurious fragment, while a
/// negative value is indistinguishable from genuine indefiniteness.
///
/// Restricting to representative pairs removes the degeneracy exactly rather
/// than by tolerance: with M = S M_red S^T (S the (ij) -> (min,max) selection
/// matrix), any M_red = L_red L_red^T gives M = (S L_red)(S L_red)^T, the rank
/// is structurally bounded by n(n+1)/2, and every expanded vector reshapes to
/// an exactly symmetric norb x norb matrix.
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
  // Representative pairs (i <= j) and the map from a full (ij) index to its
  // representative.
  const size_t reduced_dim = norb * (norb + 1) / 2;
  std::vector<std::pair<size_t, size_t>> pairs;
  pairs.reserve(reduced_dim);
  std::vector<size_t> reduced_index_of(norb * norb);
  for (size_t i = 0; i < norb; ++i) {
    for (size_t j = i; j < norb; ++j) {
      reduced_index_of[i * norb + j] = pairs.size();
      reduced_index_of[j * norb + i] = pairs.size();
      pairs.emplace_back(i, j);
    }
  }

  // M_red is the principal submatrix on the representative pairs. Average the
  // mirror entries instead of picking one, which projects out any (i,j)
  // asymmetry noise in the input tensor the same way the caller's transpose
  // symmetrization does.
  Eigen::MatrixXd reduced(reduced_dim, reduced_dim);
  for (size_t p = 0; p < reduced_dim; ++p) {
    const auto [i, j] = pairs[p];
    for (size_t q = 0; q < reduced_dim; ++q) {
      const auto [k, l] = pairs[q];
      reduced(p, q) = 0.25 * (supermatrix(i * norb + j, k * norb + l) +
                              supermatrix(j * norb + i, k * norb + l) +
                              supermatrix(i * norb + j, l * norb + k) +
                              supermatrix(j * norb + i, l * norb + k));
    }
  }

  Eigen::VectorXd residual_diagonal = reduced.diagonal();

  // Nothing to factorize: an empty orbital space has no fragments, and the
  // scale/pivot reductions below are undefined on an empty vector.
  if (reduced_dim == 0) {
    fragments.clear();
    return true;
  }

  // A literal 0.0 threshold is never reached in floating point: once the
  // numerical rank is exhausted the residual diagonal is a cancellation of
  // O(||M||) terms and decays into roundoff, whose size is *relative* to the
  // matrix scale. Floor the cutoff at that noise level -- an absolute floor
  // such as machine epsilon is dimensionally wrong and would behave
  // differently for the same problem expressed in different units.
  const double diagonal_scale = std::max(residual_diagonal.maxCoeff(), 0.0);
  const double noise_floor = std::numeric_limits<double>::epsilon() *
                             static_cast<double>(reduced_dim) * diagonal_scale;
  const double stop_threshold = std::max(truncation_threshold, noise_floor);
  const double indefinite_tolerance = noise_floor;

  std::vector<Eigen::VectorXd> cholesky_vectors;
  cholesky_vectors.reserve(reduced_dim);

  for (size_t step = 0; step < reduced_dim; ++step) {
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
    Eigen::VectorXd column = reduced.col(pivot);
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
    // Expanding the reduced vector back over all (ij) pairs is exactly the
    // S L product, and it is symmetric by construction: the mirror entries
    // read the same reduced component.
    Eigen::MatrixXd fragment_matrix(norb, norb);
    for (size_t i = 0; i < norb; ++i) {
      for (size_t j = 0; j < norb; ++j) {
        fragment_matrix(i, j) = vector[reduced_index_of[i * norb + j]];
      }
    }

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
