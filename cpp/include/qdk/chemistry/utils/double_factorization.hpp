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
/// generalization derived to also cover negative-eigenvalue fragments,
/// worth double-checking against the reference implementation).
struct TwoBodyFragment {
  Eigen::MatrixXd U;    ///< norb x norb orbital rotation. Columns are the
                       ///< orthonormal new-orbital vectors expressed in the
                       ///< original orbital basis.
  Eigen::VectorXd eps;  ///< norb eigenvalues (epsilon_p^(alpha)).
  double sign = 1.0;    ///< +1.0 or -1.0.
  double lambda_df = 0.0;  ///< Baseline fermionic 1-norm contribution of
                          ///< this fragment (Eq. 17), before any BLISS
                          ///< shift.
};

/// Double-factorize the spin-free two-electron integral tensor g_ijkl
/// (flattened in the same chemist-notation layout as
/// CanonicalFourCenterHamiltonianContainer::get_two_body_index(), i.e.
/// index = i*norb^3 + j*norb^2 + k*norb + l) into a set of low-rank
/// fragments via eigendecomposition of the reshaped (ij),(kl) supermatrix.
///
/// This is a standalone diagnostic/analysis utility: it does not require an
/// Algorithm/Settings/Factory instance and can be called directly (e.g. by
/// qdk::chemistry::utils::hamiltonian_one_norm(), or by algorithms such as
/// qdk::chemistry::algorithms::HamiltonianRegularizer implementations that
/// need low-rank two-electron fragments).
///
/// @param two_body_integrals Flattened g_ijkl tensor, size norb^4.
/// @param norb Number of (spatial) orbitals.
/// @param truncation_threshold Fragments whose |eigenvalue| of the reshaped
///        supermatrix falls below this threshold are dropped. Defaults to
///        0.0 (no truncation -- the factorization is exact/lossless unless
///        the caller explicitly opts into compression).
/// @return The list of retained fragments, sorted by decreasing
///         |eigenvalue|.
std::vector<TwoBodyFragment> double_factorize(
    const Eigen::VectorXd& two_body_integrals, size_t norb,
    double truncation_threshold = 0.0);

}  // namespace qdk::chemistry::utils
