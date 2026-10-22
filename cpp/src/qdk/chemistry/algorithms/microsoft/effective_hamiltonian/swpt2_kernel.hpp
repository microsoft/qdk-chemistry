// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Second-order Schrieffer-Wolff downfolding kernel.
//
// Implements an orbital, diagonal-Fock second-order Schrieffer-Wolff downfold.
// Split H into H_BD, block diagonal in the external occupation, and the
// occupation-changing H_OD. For
// bare denominators, a separate diagonal generalized-Fock operator F0 defines
// the anti-Hermitian generator through [F0, S] = H_OD. The sigma^2 regularizer
// instead builds a regularized generator, which solves that equation only
// approximately. In either case the implemented
// approximation is H_eff = H_BD + 1/2 [S, H_OD], projected onto the reference
// external occupation and truncated to <= 2-body. SeQuant spin-traces the
// projected commutator into contractions over spatial alpha/beta blocks.
//
// Representation: per-spin one-body blocks and chemist aaaa, aabb and bbbb
// two-body blocks (pq|rs), stored flat in C-order
// (index = ((p*norb + q)*norb + r)*norb + s).
//
// References:
//   Schrieffer & Wolff, Phys. Rev. 149, 491 (1966)
//   Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011)
//   Kutzelnigg & Mukherjee, J. Chem. Phys. 107, 432 (1997)  [generalized
//     normal-ordering framework motivating the higher-body fold; this
//     implementation retains 1-RDM pair contractions only]
//   Evangelista, J. Chem. Phys. 141, 054109 (2014)          [sigma^2 / DSRG
//     flow form of the denominator regularizer]
//   Shee et al., J. Phys. Chem. Lett. 12, 12084 (2021)      [survey of the
//     sigma, sigma^2 and kappa regularizers]

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <unordered_set>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::swpt2 {

/// C-order flat index into an (n, n, n, n) tensor.
inline std::size_t idx4(int p, int q, int r, int s, int n) {
  return ((static_cast<std::size_t>(p) * n + q) * n + r) * n + s;
}

// ===========================================================================
// Shared kernel foundations.
// ===========================================================================

/// Fixed cutoff for the guarded bare pseudoinverse. Coupled channels below
/// this magnitude are omitted rather than divided by a singular denominator.
inline constexpr double bare_denominator_floor = 1e-8;

/// SW energy-denominator options.
struct RegularizerOptions {
  /// Sigma of the sigma^2 regularizer, 1/D -> (1 - exp(-sigma*D^2))/D, which
  /// damps near-degenerate (intruder) channels; equivalently the SRG/DSRG flow
  /// parameter (units of inverse energy squared). Larger values regularize
  /// less; 0 selects a guarded bare pseudoinverse that maps coupled channels
  /// with |D| < `bare_denominator_floor` to zero. See Shee et al., J. Phys.
  /// Chem. Lett. 12, 12084 (2021).
  double sigma2 = 0.0;
};

/// Regularized inverse energy denominator.
double regularized_inverse(double delta, const RegularizerOptions& reg);

/// Chemist aaaa, aabb and bbbb two-body blocks, which may alias.
using TwoBodyBlocks =
    std::array<std::reference_wrapper<const Eigen::VectorXd>, 3>;

/// Generalized Fock matrices from per-spin 1-RDMs D^s,
/// F^s_pq = h^s_pq + sum_t sum_rs D^t_rs (pq|rs)^st - sum_rs D^s_rs (pr|sq)^ss.
/// Their diagonals supply the denominator energies. A restricted caller splits
/// the spin-traced density evenly, so eps_alpha == eps_beta and H0 preserves
/// S^2; high-spin states are then realized by the active-space solve.
std::array<Eigen::MatrixXd, 2> generalized_fock_matrices(
    const std::array<Eigen::MatrixXd, 2>& one_body,
    const TwoBodyBlocks& two_body,
    const std::array<Eigen::MatrixXd, 2>& density, int norb);

/// Block-diagonal orthogonal rotation that diagonalizes each Fock sub-block.
/// Blocks contain spatial-orbital indices and must be disjoint. Blocks whose
/// off-diagonal max norm does not exceed `tolerance` are left unchanged.
Eigen::MatrixXd semicanonical_rotation(
    const Eigen::MatrixXd& fock, const std::vector<std::vector<int>>& blocks,
    double tolerance);

/// Transform h' = U^T h U for new orbitals C' = C U.
Eigen::MatrixXd rotate_one_body(const Eigen::MatrixXd& one_body,
                                const Eigen::MatrixXd& rotation);

/// Transform chemist integrals
/// (pq|rs)' = U_ap U_bq U_cr U_ds (ab|cd), retaining the full dense tensor.
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
  double min_denominator = std::numeric_limits<double>::infinity();
  double max_amplitude = 0.0;
};

struct DerivationStatistics {
  std::size_t projected_terms = 0, contractions = 0, maximum_rank = 0;
};

/// Statistics of the contraction plans for the given role extents.
DerivationStatistics downfold_statistics(std::size_t inactive,
                                         std::size_t active,
                                         std::size_t virtuals);

/// Spin-blocked downfold. `density` holds the per-spin reference 1-RDMs used
/// to fold terms above two-body; leave it empty to discard them.
DownfoldResult downfold(const std::array<Eigen::MatrixXd, 2>& one_body,
                        const TwoBodyBlocks& two_body,
                        const std::array<Eigen::VectorXd, 2>& energies,
                        const std::vector<int>& active,
                        const std::vector<int>& inactive,
                        const std::vector<int>& virtuals,
                        const RegularizerOptions& regularizer,
                        double core_energy,
                        const std::array<Eigen::MatrixXd, 2>& density = {});

/// Window roles chosen by the folding policy, plus the rounding diagnostics a
/// caller needs to judge how much the fold perturbed the reference density.
struct WindowPartition {
  /// Spatial positions into the window.
  std::vector<int> active_spatial, inactive_spatial, virtual_spatial;
  int active_electrons = 0;       ///< window electrons the rounded fold leaves
  double worst_deviation = 0.0;   ///< largest folded |occupation - rounded|
  double worst_occupation = 0.0;  ///< occupation attaining `worst_deviation`
  std::size_t worst_orbital = 0;  ///< global index attaining it
  /// Rounded folded electron count minus the reference occupation over the
  /// folded orbitals. Roundings of opposite sign cancel here, so this is the
  /// monopole of the density error and can grow where `worst_deviation` cannot.
  double folded_charge_error = 0.0;
};

/// Apply the folding policy to a window: orbitals in `kept_global` become the
/// active space, and each remaining orbital is folded into the external space
/// as inactive or virtual by rounding its reference occupation to the nearer of
/// 2 or 0. Rounding preserves the total electron count -- the active space
/// receives whatever the folded orbitals do not take -- but it perturbs the
/// mean field the active space feels, so the deviations are reported.
///
/// `occupation` is indexed by window position and `window_global` maps those
/// positions to global orbital indices; `window_electrons` is the (integer)
/// electron count the reference density places in the window. Throws
/// `std::invalid_argument` if a folded orbital deviates from an integer
/// occupation by more than `max_folded_occupation_deviation`, which must lie in
/// [0, 1) so a singly occupied orbital is never folded on an arbitrary
/// rounding.
WindowPartition partition_window(
    const std::vector<double>& occupation,
    const std::vector<std::size_t>& window_global,
    const std::unordered_set<std::size_t>& kept_global, int window_electrons,
    double max_folded_occupation_deviation);

}  // namespace qdk::chemistry::algorithms::microsoft::swpt2
