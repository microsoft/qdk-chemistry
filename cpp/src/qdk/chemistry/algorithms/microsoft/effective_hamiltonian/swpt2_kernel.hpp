// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Second-order Schrieffer-Wolff downfolding kernel.
//
// Representation: per-spin one-body blocks and chemist aaaa, aabb and bbbb
// two-body blocks (pq|rs), stored flat in C-order
// (index = ((p*norb + q)*norb + r)*norb + s).
//
// References:
//   Schrieffer & Wolff, Phys. Rev. 149, 491 (1966)
//   Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011)
//   Kutzelnigg & Mukherjee, J. Chem. Phys. 107, 432 (1997)  [generalized
//     normal ordering behind the 1-RDM fold of three-body terms]
//   Evangelista, J. Chem. Phys. 141, 054109 (2014)          [DSRG flow form
//     of the sigma^2 regularizer]
//   Shee et al., J. Phys. Chem. Lett. 12, 12084 (2021)      [denominator
//     regularizers]

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::swpt2 {

/// C-order flat index into an (n, n, n, n) tensor.
inline std::size_t idx4(int p, int q, int r, int s, int n) {
  return ((static_cast<std::size_t>(p) * n + q) * n + r) * n + s;
}

/// Denominator cutoff of the guarded bare pseudoinverse: coupled channels
/// below it are omitted rather than divided by a near-zero denominator.
inline constexpr double bare_denominator_floor = 1e-8;

struct RegularizerOptions {
  /// Sigma of the sigma^2 regularizer, 1/D -> (1 - exp(-sigma*D^2))/D, in
  /// Eh^-2. 0 selects the guarded bare pseudoinverse, which maps channels with
  /// |D| < `bare_denominator_floor` to zero.
  double sigma2 = 0.0;
};

/// Regularized inverse energy denominator.
double regularized_inverse(double delta, const RegularizerOptions& reg);

/// Chemist aaaa, aabb and bbbb two-body blocks, which may alias.
using TwoBodyBlocks =
    std::array<std::reference_wrapper<const Eigen::VectorXd>, 3>;

/// Generalized Fock matrices from per-spin 1-RDMs D^s,
/// F^s_pq = h^s_pq + sum_t sum_rs D^t_rs (pq|rs)^st - sum_rs D^s_rs (pr|sq)^ss.
/// Their diagonals supply the denominator energies. `spin_free` input has
/// identical spin blocks, so only the alpha matrix is formed.
std::array<Eigen::MatrixXd, 2> generalized_fock_matrices(
    const std::array<Eigen::MatrixXd, 2>& one_body,
    const TwoBodyBlocks& two_body,
    const std::array<Eigen::MatrixXd, 2>& density, int norb, bool spin_free);

/// Block-diagonal orthogonal rotation that diagonalizes each Fock sub-block.
/// Blocks contain spatial-orbital indices and must be disjoint. Blocks whose
/// off-diagonal max norm does not exceed `tolerance` are left unchanged.
Eigen::MatrixXd semicanonical_rotation(
    const Eigen::MatrixXd& fock, const std::vector<std::vector<int>>& blocks,
    double tolerance);

/// Transform h' = U^T h U for new orbitals C' = C U.
Eigen::MatrixXd rotate_one_body(const Eigen::MatrixXd& one_body,
                                const Eigen::MatrixXd& rotation);

/// Transform chemist integrals (pq|rs)' = U_ap U_bq U_cr U_ds (ab|cd).
Eigen::VectorXd rotate_two_body(const Eigen::VectorXd& two_body,
                                const Eigen::MatrixXd& rotation, int norb);

/// Transform chemist integrals whose (pq) and (rs) pairs rotate separately,
/// as the aabb block of unrestricted orbitals does.
Eigen::VectorXd rotate_two_body(const Eigen::VectorXd& two_body,
                                const Eigen::MatrixXd& first_pair,
                                const Eigen::MatrixXd& second_pair, int norb);

struct DownfoldResult {
  double scalar = 0.0;
  std::array<Eigen::MatrixXd, 2> one_body;
  /// Chemist aabb block; the spin-free integrals for spin-symmetric input.
  Eigen::VectorXd two_body;
  /// Chemist aaaa and bbbb blocks, which are fixed only up to exchange.
  std::array<Eigen::VectorXd, 2> same_spin_two_body;
  /// Smallest denominator and largest raw amplitude among the excitations
  /// that connect P and Q.
  double min_denominator = std::numeric_limits<double>::infinity();
  double max_amplitude = 0.0;
};

/// Spin-blocked downfold. The diagonals of the per-spin generalized Fock
/// matrices `fock` are the orbital energies of the denominators; their
/// off-diagonals are the one-body couplings of the amplitude diagnostic.
/// `spin_free` input has identical spin blocks and is evaluated once.
/// `density` holds the per-spin reference 1-RDMs used to fold terms above
/// two-body; leave it empty to discard them.
DownfoldResult downfold(const std::array<Eigen::MatrixXd, 2>& one_body,
                        const TwoBodyBlocks& two_body,
                        const std::array<Eigen::MatrixXd, 2>& fock,
                        const std::vector<int>& active,
                        const std::vector<int>& inactive,
                        const std::vector<int>& virtuals,
                        const RegularizerOptions& regularizer,
                        double core_energy, bool spin_free,
                        const std::array<Eigen::MatrixXd, 2>& density = {});

}  // namespace qdk::chemistry::algorithms::microsoft::swpt2
