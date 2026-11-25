// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Second-order driven similarity renormalization group (DSRG-PT2) downfolding
// kernel for single-determinant references.
//
// Representation: per-spin one-body blocks and chemist aaaa, aabb and bbbb
// two-body blocks (pq|rs), stored flat in C-order
// (index = ((p*norb + q)*norb + r)*norb + s).
//
// References:
//   Kutzelnigg & Mukherjee, J. Chem. Phys. 107, 432 (1997)  [generalized
//     normal ordering]
//   Evangelista, J. Chem. Phys. 141, 054109 (2014)          [DSRG flow
//     parameter s]
//   Li & Evangelista, J. Chem. Phys. 143, 164104 (2015)     [DSRG-MRPT2]
//   Li & Evangelista, J. Chem. Phys. 146, 124132 (2017)     [second-order
//     DSRG effective Hamiltonian]

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::dsrg_pt2 {

/// C-order flat index into an (n, n, n, n) tensor.
inline std::size_t idx4(int p, int q, int r, int s, int n) {
  return ((static_cast<std::size_t>(p) * n + q) * n + r) * n + s;
}

/// Factors of one channel with gap D and DSRG flow parameter s: the generator
/// amplitude takes g/D and the renormalized coupling takes 2 - g, where
/// g = 1 - exp(-s*D^2). At s = 0 or D = 0, g = 0 so {0, 2}.
struct ChannelFactors {
  double inverse;
  double dressing;
};

ChannelFactors channel_factors(double delta, double flow);

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

/// Block-diagonal orthogonal rotation that diagonalizes each block of the
/// symmetric Fock matrix. Blocks contain spatial-orbital indices and must be
/// disjoint. Blocks whose off-diagonal max norm does not exceed `tolerance`
/// are left unchanged.
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

/// Spin-blocked DSRG-PT2 downfold. The diagonals of the per-spin generalized
/// Fock matrices `fock` are the orbital energies of the denominators; their
/// off-diagonals supply the one-body couplings. `spin_free` input has
/// identical spin blocks and is evaluated once. `density` holds the per-spin
/// reference 1-RDMs used for reference normal ordering; leave it empty for
/// core-normal-ordered two-body truncation. `flow == 0` returns the bare
/// core-folded operator over `active`.
DownfoldResult downfold(const std::array<Eigen::MatrixXd, 2>& one_body,
                        const TwoBodyBlocks& two_body,
                        const std::array<Eigen::MatrixXd, 2>& fock,
                        const std::vector<int>& active,
                        const std::vector<int>& inactive,
                        const std::vector<int>& virtuals, double flow,
                        double core_energy, bool spin_free,
                        const std::array<Eigen::MatrixXd, 2>& density = {});

}  // namespace qdk::chemistry::algorithms::microsoft::dsrg_pt2
