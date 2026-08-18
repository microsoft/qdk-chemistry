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
// external occupation and truncated to <= 2-body. The projected commutator is
// evaluated by bounded-rank generalized-Wick contractions over external legs.
//
// Representation (spin-orbital, n_so = 2 * norb; interleaved alpha(p)=2p,
// beta(p)=2p+1):
//   H = e0 + sum_{PQ} f[P,Q] a^dag_P a_Q
//         + (1/4) sum_{PQRS} v[P,Q,R,S] a^dag_P a^dag_Q a_R a_S
// with v antisymmetric under P<->Q and under R<->S. Two-body tensors are stored
// flat in C-order (index = ((P*n_so + Q)*n_so + R)*n_so + S).
//
// References:
//   Schrieffer & Wolff, Phys. Rev. 149, 491 (1966)
//   Bravyi, DiVincenzo & Loss, Ann. Phys. 326, 2793 (2011)
//   Kutzelnigg & Mukherjee, J. Chem. Phys. 107, 432 (1997)  [generalized
//     normal ordering, used to fold terms above two-body]
//   Evangelista, J. Chem. Phys. 141, 054109 (2014)          [sigma^2 / DSRG
//     flow form of the denominator regularizer]
//   Shee et al., J. Phys. Chem. Lett. 12, 12084 (2021)      [survey of the
//     sigma, sigma^2 and kappa regularizers]

#pragma once

#include <Eigen/Dense>
#include <cstddef>
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

/// SW energy-denominator options.
struct RegularizerOptions {
  /// Sigma of the sigma^2 regularizer, 1/D -> (1 - exp(-sigma*D^2))/D, which
  /// damps near-degenerate (intruder) channels; equivalently the SRG/DSRG flow
  /// parameter (units of inverse energy squared). Larger values regularize
  /// less; 0 disables, leaving the bare inverse. See Shee et al., J. Phys.
  /// Chem. Lett. 12, 12084 (2021).
  double sigma2 = 0.0;
};

/// Regularized inverse energy denominator.
double regularized_inverse(double delta, const RegularizerOptions& reg);

/// Spin-orbital partition + single-determinant reference occupation.
///
/// P = the active space, kept exactly; Q = the "external" space folded away,
/// split by reference occupation into inactive (occupied) and virtual (empty).
/// Names follow qdk `data::Orbitals` (active / inactive / virtual), so this is
/// a direct view of an `Orbitals` + reference `Wavefunction`. The three roles
/// are mutually exclusive and exhaust the window; the "external" (folded) space
/// is `inactive | virtual`. Inactive and virtual are kept as separate masks so
/// the occupation-change test catches inactive<->virtual excitations (which are
/// net-zero on the combined external set).
struct SpinOrbitalPartition {
  int n_so = 0;
  std::vector<char> is_active, is_inactive, is_virtual;
};

/// Spin-free generalized Fock matrix
/// F_pq = h_pq + sum_rs D_rs [(pq|rs) - 1/2 (pr|sq)]. Its diagonal supplies the
/// denominator energies; the density is spin-traced, so eps_alpha == eps_beta
/// and H0 preserves S^2. High-spin states are realized by the active-space
/// solve, not by polarizing this reference.
Eigen::MatrixXd generalized_fock_matrix(const Eigen::MatrixXd& h1,
                                        const Eigen::VectorXd& two_body,
                                        const Eigen::MatrixXd& density,
                                        int norb);

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

/// Downfolded active-space operator as compact spatial (chemist) integrals,
/// ready to feed a qdk CanonicalFourCenter Hamiltonian.
struct ActiveHamiltonian {
  int norb = 0;              ///< number of active spatial orbitals
  double core_energy = 0.0;  ///< scalar (folded core + downfold shift)
  Eigen::MatrixXd one_body;  ///< (norb, norb) spatial one-body
  Eigen::VectorXd two_body;  ///< (norb^4,) chemist (pq|rs), C-order
};

// ===========================================================================
// Spin-blocked on-the-fly downfold: the path wired into the constructor.
// Stores the two-body as spatial spin blocks and forms every element on
// demand, so no dense n_so^4 tensor is ever materialized.
// ===========================================================================

/// Spin-blocked *spatial* storage of the antisymmetric two-body tensor for a
/// restricted basis: 2*norb^4 versus 16*norb^4 for a dense spin-orbital `v`.
/// The alpha and beta same-spin blocks are identical, so one is stored; every
/// other nonzero spin pattern follows by antisymmetry (`so_v_from_blocked`).
struct SpinBlockedTwoBody {
  int norb = 0;
  Eigen::VectorXd v_aaaa;  ///< (norb^4,) same-spin antisymmetric block
  Eigen::VectorXd v_abab;  ///< (norb^4,) opposite-spin block v[pa,qb,ra,sb]
};

/// Build the independent spin blocks from a restricted chemist tensor.
SpinBlockedTwoBody build_two_body_blocked(const Eigen::VectorXd& g, int norb);

/// Reconstruct a single spin-orbital element `v[P,Q,R,S]` (interleaved spin,
/// alpha=2p, beta=2p+1) from the spin blocks.
double so_v_from_blocked(const SpinBlockedTwoBody& b, int P, int Q, int R,
                         int S);

/// Spin-orbital one-body matrix (n_so, n_so) from spatial alpha/beta blocks --
/// the cheap O(norb^2) one-body input for the spin-blocked downfold path.
Eigen::MatrixXd spin_orbital_one_body(const Eigen::MatrixXd& h1a,
                                      const Eigen::MatrixXd& h1b, int norb);

/// Compact active-only downfold result: the effective operator over the active
/// space only (see `v_abab`), so memory is O(n_active_spatial^4) not the
/// window's O(n_so^4). Produced by `downfold_blocked` for the production path
/// (large windows, small active space).
struct ActiveDownfoldResult {
  /// active spin-orbitals, ascending (size n_active_so)
  std::vector<int> active_so;
  double e = 0.0;
  Eigen::MatrixXd f_active;  ///< (n_active_so, n_active_so) compact one-body
  /// (n_active_spatial^4,) opposite-spin (abab) block over the active *spatial*
  /// orbitals: v_abab[p,q,r,s] = v[alpha p, beta q, alpha r, beta s]. The
  /// downfold is spin-restricted, so this single spatial block is the entire
  /// effective two-body (the same-spin blocks are its antisymmetrization) --
  /// hence O(n_active_spatial^4), not O(n_active_so^4).
  Eigen::VectorXd v_abab;
  double min_denominator = 0.0;  ///< smallest |Delta| over coupled channels
  double max_amplitude = 0.0;    ///< largest raw |V/Delta| (intruder gauge)
};

/// Evaluate the downfold from the spin-blocked store, forming every
/// antisymmetric two-body element on demand: the dense n_so^4 tensors (`v`, the
/// generator `s2`, the block-diagonal / off-diagonal split) are never
/// materialized. Intruder diagnostics still scan the couplings in O(norb^4)
/// time but store nothing.
///
/// The retained Wick contractions are evaluated as BLAS GEMMs over packed
/// active/external panels, reducing operand-internal reference contractions
/// while packing. Zero-contraction products cancel between the two commutator
/// orderings and are skipped. While terms above two-body are discarded, the
/// one-line `S2 * V2` channel instead enumerates the active-index coincidences
/// that survive the truncation, avoiding its rank-three output; folding those
/// terms makes every matching contribute, so that shortcut then does not apply.
///
/// `f` is the spin-orbital one-body from `spin_orbital_one_body`; `e_core` is
/// the scalar core energy.
///
/// Terms above two-body are folded onto `reference_density` (spin-traced, over
/// the window's spatial orbitals) when it is nonempty, which is what makes the
/// emitted operator usable for kept spaces holding more than two electrons;
/// leave it empty to discard them. `occupied_so` selects the equivalent
/// particle-hole fold against a reference determinant; it exists so the tests
/// can cross-check the density path against it.
ActiveDownfoldResult downfold_blocked(
    const Eigen::MatrixXd& f, const SpinBlockedTwoBody& blk,
    const Eigen::VectorXd& eps, const SpinOrbitalPartition& part,
    const RegularizerOptions& reg, double e_core,
    const std::vector<int>& occupied_so = {},
    const Eigen::MatrixXd& reference_density = {});

/// Relabel the compact `ActiveDownfoldResult` active block to compact spatial
/// chemist integrals for a qdk CanonicalFourCenter Hamiltonian.
ActiveHamiltonian to_spatial_chemist(const ActiveDownfoldResult& down,
                                     const SpinOrbitalPartition& part);

/// Assemble the spin-orbital role masks from an explicit spatial partition of
/// a window of `norb` orbitals. The three index lists must be disjoint and
/// together cover [0, norb). Deciding which folded orbital counts as doubly
/// occupied is the caller's policy, not this function's.
SpinOrbitalPartition make_partition(int norb,
                                    const std::vector<int>& active_spatial,
                                    const std::vector<int>& inactive_spatial,
                                    const std::vector<int>& virtual_spatial);

/// Window roles chosen by the folding policy, plus the rounding diagnostics a
/// caller needs to judge how much the fold perturbed the reference density.
struct WindowPartition {
  /// Positions into the window, suitable for `make_partition`.
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
