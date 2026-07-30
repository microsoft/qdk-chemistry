// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Second-order Schrieffer-Wolff downfolding — foundation kernel.
//
// Implements the operator-level second-order Schrieffer-Wolff downfolding as
// closed-form tensor contractions. Given H = H0 + V with a diagonal-Fock H0,
// the anti-Hermitian generator S solves [H0, S] = V and the effective operator
// is H_eff = H0 + 1/2 [S, V], projected onto the reference buffer occupation
// and truncated to <= 2-body. This header covers the *foundation*:
// antisymmetric spin-orbital tensor build from qdk (chemist) integrals,
// diagonal-Fock energies, the generator S = V/Delta, and the block-diagonal
// reference- occupation mean-field fold. The second-order commutator channels
// are added next.
//
// Representation (spin-orbital, n_so = 2 * norb; interleaved alpha(p)=2p,
// beta(p)=2p+1):
//   H = e0 + sum_{PQ} f[P,Q] a^dag_P a_Q
//         + (1/4) sum_{PQRS} v[P,Q,R,S] a^dag_P a^dag_Q a_R a_S
// with v antisymmetric under P<->Q and under R<->S. Two-body tensors are stored
// flat in C-order (index = ((P*n_so + Q)*n_so + R)*n_so + S).

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::swpt2 {

/// C-order flat index into an (n, n, n, n) tensor.
inline std::size_t idx4(int p, int q, int r, int s, int n) {
  return ((static_cast<std::size_t>(p) * n + q) * n + r) * n + s;
}

// ===========================================================================
// Shared foundations (used by BOTH the dense reference and the production
// path).
// ===========================================================================

/// SW energy-denominator regularization (precedence: flow > shift > floor).
struct RegOptions {
  double denom_floor = 1e-8;  ///< hard cutoff: skip couplings with |D| < floor
  double denom_shift = 0.0;   ///< CASPT2-like shift: 1/D -> D / (D^2 + shift^2)
  /// Smooth flow-parameter regularizer 1/D -> (1 - exp(-s*D^2))/D that damps
  /// near-degenerate (intruder) channels; `s` is the SRG/DSRG "flow parameter"
  /// (units of inverse energy squared). < 0 : disabled.
  double denom_flow = -1.0;
};

/// Regularized inverse energy denominator.
double reg_inv(double delta, const RegOptions& reg);

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
struct SoPartition {
  int n_so = 0;
  std::vector<char> is_active, is_inactive, is_virtual;
};

/// Diagonal (generalized) Fock energies per spin-orbital. na,nb: per-spatial
/// reference occupation (alpha/beta); they are spin-averaged (spin-free 1-RDM,
/// n^sigma = (na+nb)/2) so eps_alpha == eps_beta and H0 preserves S^2.
/// High-spin states are realized by the active-space solve, not by polarizing
/// this reference (a spin-polarized H0 would break S^2 of the downfold).
Eigen::VectorXd diagonal_fock_energies(const Eigen::MatrixXd& h1a,
                                       const Eigen::VectorXd& g_aaaa,
                                       const Eigen::VectorXd& na,
                                       const Eigen::VectorXd& nb, int norb);

/// Downfolded active-space operator as compact spatial (chemist) integrals,
/// ready to feed a qdk CanonicalFourCenter Hamiltonian.
struct ActiveHamiltonian {
  int norb = 0;              ///< number of active spatial orbitals
  double core_energy = 0.0;  ///< scalar (folded core + downfold shift)
  Eigen::MatrixXd one_body;  ///< (norb, norb) spatial one-body
  Eigen::VectorXd two_body;  ///< (norb^4,) chemist (pq|rs), C-order
};

/// Build a spin-orbital partition over a window of `norb` spatial orbitals:
/// `active_spatial` become active (the kept space P); the rest split by
/// reference `occupation` (per spatial orbital) into inactive (occ >= 1, i.e.
/// doubly-occupied domo) or virtual (empty). The external space must be
/// closed-shell (occ ~ 0 or 2); put open-shell/magnetic orbitals in the active
/// space.
SoPartition make_partition(int norb, const std::vector<int>& active_spatial,
                           const std::vector<double>& occupation);

// ===========================================================================
// Dense spin-orbital reference (validation ORACLE; NOT used in production).
// A transparent, dense-tensor downfold kept as the independent reference the
// spin-blocked production path is checked against (see the
// `BlockedDownfoldMatchesSpinOrbital` test). Grouped in `namespace reference`
// to mark its test-only status; the names are re-exported into the parent
// namespace below for existing call sites.
// ===========================================================================

namespace reference {

/// Antisymmetric spin-orbital operator tensors.
struct SoTensors {
  int n_so = 0;       ///< number of spin-orbitals (2 * norb)
  double e0 = 0.0;    ///< scalar
  Eigen::MatrixXd f;  ///< (n_so, n_so) one-body
  Eigen::VectorXd v;  ///< (n_so^4,) antisymmetric two-body, C-order
};

/// Build antisymmetric spin-orbital tensors from qdk spin integrals.
/// h1a,h1b: (norb, norb). g_*: flat (norb^4,) chemist (pq|rs), C-order.
/// (aaaa/aabb/bbbb = the qdk get_two_body_integrals channels.)
SoTensors build_tensors(const Eigen::MatrixXd& h1a, const Eigen::MatrixXd& h1b,
                        const Eigen::VectorXd& g_aaaa,
                        const Eigen::VectorXd& g_aabb,
                        const Eigen::VectorXd& g_bbbb, double e_core, int norb);

/// Generator S = (occupation-changing part of H) scaled elementwise by 1/Delta.
struct Generator {
  Eigen::MatrixXd s1;  ///< (n_so, n_so)
  Eigen::VectorXd s2;  ///< (n_so^4,)
  /// Smallest |Delta| over the coupled (nonzero-V) occupation-changing channels
  /// (0 if there are none). Secondary context for `max_amplitude`.
  double min_denominator = 0.0;
  /// Largest RAW amplitude |V / Delta| over those channels (unregularized, so
  /// it flags intruders even when the generator itself is regularized). This is
  /// the perturbation-convergence indicator: values >~ 1 mean second-order PT
  /// is unreliable for that channel. Zero couplings never contribute (V/Delta =
  /// 0).
  double max_amplitude = 0.0;
};
Generator make_generator(const SoTensors& h, const Eigen::VectorXd& eps,
                         const SoPartition& part, const RegOptions& reg);

/// Split (f, v) by external occupation change: block-diagonal (preserving) vs
/// off-diagonal (occupation-changing). e0 stays in the block-diagonal part.
std::pair<SoTensors, SoTensors> split_bd_od(const SoTensors& h,
                                            const SoPartition& part);

/// Reference-occupation mean-field fold of a block-diagonal operator onto the
/// active space (the inactive-Fock + core folding of the external orbitals).
struct MeanFieldResult {
  double e = 0.0;
  Eigen::MatrixXd f_active;  ///< (n_so, n_so)
  Eigen::VectorXd v_active;  ///< (n_so^4,)
};
MeanFieldResult mean_field_fold(const SoTensors& h_bd, const SoPartition& part);

/// The effective active-space operator + the discarded-body diagnostic.
struct DownfoldResult {
  double e = 0.0;
  Eigen::MatrixXd f_active;       ///< (n_so, n_so), active block
  Eigen::VectorXd v_active;       ///< (n_so^4,), active block
  double higher_body_norm = 0.0;  ///< L2 norm of the discarded >= 3-body part
  /// Smallest |Delta| over the coupled occupation-changing channels (context).
  double min_denominator = 0.0;
  /// Largest raw amplitude |V / Delta| (perturbation-convergence indicator; the
  /// quantity to gate an intruder warning on -- prefer this over
  /// `min_denominator`, which flags harmless small gaps that carry no
  /// coupling).
  double max_amplitude = 0.0;
};

/// Full second-order SW downfold: the block-diagonal reference-occupation fold
/// plus 1/2 [S, V] projected onto the active space, truncated to <= 2-body.
/// ½[S, V] is evaluated by generalized Wick contraction over the external legs
/// (buffer summed), emitting operators over the (small) active space only.
/// Result lives on the active spin-orbitals (full n_so indexing; non-active
/// entries are zero).
DownfoldResult downfold(const SoTensors& h, const Eigen::VectorXd& eps,
                        const SoPartition& part, const RegOptions& reg);

/// Restrict + relabel the spin-restricted active block of a DownfoldResult to
/// compact spatial chemist integrals (the inverse of build_tensors). Requires
/// each active spatial orbital to have both spin-orbitals active.
ActiveHamiltonian to_spatial_chemist(const DownfoldResult& down,
                                     const SoPartition& part);

}  // namespace reference

// ===========================================================================
// Spin-blocked on-the-fly downfold (PRODUCTION): the path wired into the
// constructor. Stores the two-body as spatial spin blocks and forms every
// element on demand, so no dense n_so^4 tensor is ever materialized.
// ===========================================================================

/// Spin-blocked *spatial* storage of the antisymmetric two-body tensor: the
/// independent nonzero spin blocks (memory ~ 3*norb^4 vs 16*norb^4 for the
/// dense spin-orbital `v`). Every other nonzero spin pattern is obtained from
/// these by antisymmetry (see `so_v_from_blocked`). This is the memory-lean
/// backing store for the spin-blocked (production) downfold path.
struct SpinBlocked2B {
  int norb = 0;
  Eigen::VectorXd v_aaaa;  ///< (norb^4,) all-alpha antisymmetric block
  Eigen::VectorXd v_bbbb;  ///< (norb^4,) all-beta  antisymmetric block
  Eigen::VectorXd v_abab;  ///< (norb^4,) opposite-spin block v[pa,qb,ra,sb]
};

/// Build the independent spin blocks from qdk chemist integrals (same
/// convention as `reference::build_tensors`).
SpinBlocked2B build_two_body_blocked(const Eigen::VectorXd& g_aaaa,
                                     const Eigen::VectorXd& g_aabb,
                                     const Eigen::VectorXd& g_bbbb, int norb);

/// Reconstruct a single spin-orbital element `v[P,Q,R,S]` (interleaved spin,
/// alpha=2p, beta=2p+1) from the spin blocks -- bridges the blocked store to
/// the spin-orbital convention and validates the block relations.
double so_v_from_blocked(const SpinBlocked2B& b, int P, int Q, int R, int S);

/// Spin-orbital one-body matrix (n_so, n_so) from spatial alpha/beta blocks --
/// the cheap (O(norb^2)) one-body half of `reference::build_tensors`, for the
/// spin-blocked (on-the-fly two-body) downfold path.
Eigen::MatrixXd spin_orbital_one_body(const Eigen::MatrixXd& h1a,
                                      const Eigen::MatrixXd& h1b, int norb);

/// Compact active-only downfold result: the effective operator over the active
/// space only (see `v_abab`), so memory is O(n_act^4) not the window's
/// O(n_so^4). Produced by `downfold_blocked` for the production path (large
/// windows, small active space).
struct ActiveDownfoldResult {
  std::vector<int> active_so;  ///< active spin-orbitals, ascending (size n_ac)
  double e = 0.0;
  Eigen::MatrixXd f_active;  ///< (n_ac, n_ac) compact one-body
  /// (n_act^4,) opposite-spin (abab) block over the n_act active *spatial*
  /// orbitals: v_abab[p,q,r,s] = v[alpha p, beta q, alpha r, beta s]. The
  /// downfold is spin-restricted, so this single spatial block is the entire
  /// effective two-body (the same-spin blocks are its antisymmetrization) --
  /// hence O(n_act^4), not O(n_ac^4) = 16 n_act^4.
  Eigen::VectorXd v_abab;
  double higher_body_norm = 0.0;  ///< L2 norm of the discarded >= 3-body part
  double min_denominator = 0.0;   ///< smallest |Delta| over coupled channels
  double max_amplitude = 0.0;     ///< largest raw |V/Delta| (intruder gauge)
};

/// Memory-lean equivalent of `reference::downfold`: consumes the spin-blocked
/// spatial two-body store (`SpinBlocked2B`) and computes every antisymmetric
/// two-body element on the fly, so the dense n_so^4 tensors (`v`, generator
/// `s2`, the block-diagonal / off-diagonal split) are never materialized. The
/// result is stored as the spatial opposite-spin block over the active
/// *spatial* orbitals (`ActiveDownfoldResult::v_abab`, O(n_act^4) --
/// spin-restriction makes that single block the whole effective two-body).
/// Numerically identical to `reference::downfold` on the active block
/// (validated against it); `f` is the spin-orbital one-body from
/// `spin_orbital_one_body`, `e_core` the scalar core energy.
ActiveDownfoldResult downfold_blocked(const Eigen::MatrixXd& f,
                                      const SpinBlocked2B& blk,
                                      const Eigen::VectorXd& eps,
                                      const SoPartition& part,
                                      const RegOptions& reg, double e_core);

/// Relabel the compact `ActiveDownfoldResult` active block to compact spatial
/// chemist integrals for a qdk CanonicalFourCenter Hamiltonian.
ActiveHamiltonian to_spatial_chemist(const ActiveDownfoldResult& down,
                                     const SoPartition& part);

/// Build a spin-orbital partition over a window of `norb` spatial orbitals:
/// `active_spatial` become active (the kept space P); the rest split by
/// reference `occupation` (per spatial orbital) into inactive (occ >= 1, i.e.
/// doubly-occupied domo) or virtual (empty). The external space must be
/// closed-shell (occ ~ 0 or 2); put open-shell/magnetic orbitals in the active
/// space.
SoPartition make_partition(int norb, const std::vector<int>& active_spatial,
                           const std::vector<double>& occupation);

}  // namespace qdk::chemistry::algorithms::microsoft::swpt2
