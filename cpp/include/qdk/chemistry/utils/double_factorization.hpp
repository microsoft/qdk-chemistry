// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace qdk::chemistry::utils {

/// A single low-rank ("perfect square") two-electron fragment produced by
/// double factorization of the two-electron integral tensor:
///
///   H^(alpha) = sign * U^T * (sum_p eps_p * n_p)^2 * U
///
/// where n_p is the occupation number operator for the rotated orbital
/// given by column p of U, and sign = +1/-1 accounts for fragments whose
/// underlying eigenvalue of the reshaped two-electron supermatrix is
/// negative (see Patel et al., arXiv:2409.18277, Eq. 16 -- the paper's
/// formula implicitly assumes a positive fragment; the sign here is a
/// generalization derived to also cover negative-eigenvalue fragments).
struct TwoBodyFragment {
  Eigen::MatrixXd U;       ///< norb x norb orbital rotation. Columns are the
                           ///< orthonormal new-orbital vectors expressed in the
                           ///< original orbital basis.
  Eigen::VectorXd eps;     ///< norb eigenvalues (epsilon_p^(alpha)).
  double sign = 1.0;       ///< +1.0 or -1.0.
  double lambda_df = 0.0;  ///< Baseline fermionic 1-norm contribution of
                           ///< this fragment (Eq. 17), before any BLISS
                           ///< shift.
};

/// Selects how the reshaped (ij),(kl) two-electron supermatrix is decomposed
/// into rank-one fragments.
///
/// Both methods reconstruct g_ijkl identically, but writing M = X X^T fixes X
/// only up to X -> X Q for orthogonal Q, and lambda_df is not invariant under
/// that gauge freedom. Choose deliberately when the 1-norm itself matters.
enum class DoubleFactorizationMethod {
  /// Pivoted Cholesky, O(R * norb^4) for rank R. Requires a positive
  /// semi-definite supermatrix, which holds for physical two-electron
  /// integrals; all fragments have sign = +1. Falls back to Eigen with a
  /// warning if the supermatrix is indefinite.
  Cholesky,
  /// Eigendecomposition via LAPACK syev, O(norb^6). Handles indefinite input
  /// (producing sign = -1 fragments) and yields the conventional literature
  /// ordering by decreasing eigenvalue magnitude.
  Eigen,
};

/// Double-factorize the spin-free two-electron integral tensor g_ijkl
/// (flattened in the same chemist-notation layout as
/// CanonicalFourCenterHamiltonianContainer::get_two_body_index(), i.e.
/// index = i*norb^3 + j*norb^2 + k*norb + l) into a set of low-rank
/// fragments, by decomposing the reshaped (ij),(kl) supermatrix (see
/// DoubleFactorizationMethod for the available decompositions).
///
/// This is a standalone diagnostic/analysis utility: it does not require an
/// Algorithm/Settings/Factory instance and can be called directly (e.g. by
/// qdk::chemistry::utils::hamiltonian_one_norm(), or by algorithms such as
/// qdk::chemistry::algorithms::SymmetryShifter implementations that
/// need low-rank two-electron fragments).
///
/// @param two_body_integrals Flattened two-electron tensor, size norb^4. The
///        caller chooses the convention: lambda_df scales linearly with this
///        tensor, and both in-tree callers pass the physical coefficient
///        V = 1/2 * g rather than the raw g.
/// @param norb Number of (spatial) orbitals.
/// @param truncation_threshold Cutoff below which fragment candidates are
///        dropped. The units are method-dependent, so the same numeric value
///        does not give the same rank for both methods: Eigen compares against
///        the supermatrix eigenvalue magnitude, Cholesky against the largest
///        remaining residual diagonal. Defaults to 0.0, meaning "lossless" for
///        both. A literal 0.0 is unreachable in floating point for Cholesky,
///        so that path floors the cutoff at machine epsilon to stop at the
///        true numerical rank rather than emit roundoff fragments.
/// @param method Which decomposition to use. Defaults to Cholesky.
/// @return The list of retained fragments, sorted by decreasing contribution
///         (eigenvalue magnitude for Eigen, sum_p |eps_p| for Cholesky).
std::vector<TwoBodyFragment> double_factorize(
    const Eigen::VectorXd& two_body_integrals, size_t norb,
    double truncation_threshold = 0.0,
    DoubleFactorizationMethod method = DoubleFactorizationMethod::Cholesky);

}  // namespace qdk::chemistry::utils
