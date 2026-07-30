// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Foundation-kernel tests for the second-order Schrieffer-Wolff downfolding.
//
// Reference values are grounded in first principles, not in any external tool:
//   * diagonal (generalized) Fock energies from their closed-form definition,
//   * the SW generator as coupling / energy-gap,
//   * the reference-buffer fold as textbook inactive-Fock mean field
//     (a doubly-occupied buffer orbital contributes 2J - K to the
//     active-space Fock and
//      2 h_dd + (dd|dd) to the core energy),
//   * invariants (hermiticity) and the empty-buffer identity.
// Each expected value is derived in a comment from the chosen input integrals.
// (An independent OpenFermion cross-check of the full method exists as a
//  development-time check; it is deliberately not a dependency of these tests.)

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

namespace {
namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

// Set a chemist integral (pq|rs) in a flat (norb^4) array.
void set_eri(Eigen::VectorXd& g, int p, int q, int r, int s, int norb,
             double val) {
  g(sw::idx4(p, q, r, s, norb)) = val;
}

// ---------------------------------------------------------------------------
// Diagonal-Fock energies: eps_p^sigma = h_pp + sum_r (pp|rr) n_r
//                                              - sum_r (pr|rp) n_r^sigma.
// ---------------------------------------------------------------------------
TEST(Swpt2Kernel, DiagonalFockEnergiesFromDefinition) {
  const int norb = 3;  // active=[0], domo=[1], virtual=[2]
  Eigen::MatrixXd h(3, 3);
  h << -0.50, 0.10, 0.20, 0.10, -1.20, -0.15, 0.20, -0.15, 3.00;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(81);
  set_eri(g, 0, 0, 0, 0, norb, 1.0);
  set_eri(g, 1, 1, 1, 1, norb, 0.8);
  set_eri(g, 2, 2, 2, 2, norb, 0.6);
  set_eri(g, 0, 0, 1, 1, norb, 0.5);
  set_eri(g, 1, 1, 0, 0, norb, 0.5);
  set_eri(g, 0, 0, 2, 2, norb, 0.4);
  set_eri(g, 2, 2, 0, 0, norb, 0.4);
  set_eri(g, 1, 1, 2, 2, norb, 0.3);
  set_eri(g, 2, 2, 1, 1, norb, 0.3);
  set_eri(g, 0, 1, 1, 0, norb, 0.25);
  set_eri(g, 1, 0, 0, 1, norb, 0.25);
  set_eri(g, 0, 2, 2, 0, norb, 0.2);
  set_eri(g, 2, 0, 0, 2, norb, 0.2);

  // reference occupations: active0 singly (alpha), inactive1 doubly, virtual2
  // empty
  Eigen::VectorXd na(3), nb(3);
  na << 1, 1, 0;
  nb << 0, 1, 0;  // total n = (1, 2, 0); spin-averaged n^sigma = (0.5, 1, 0)

  Eigen::VectorXd eps = sw::diagonal_fock_energies(h, g, na, nb, norb);

  // eps[0] = h00 + [(00|00)*1 + (00|11)*2] - [(00|00)*0.5 + (01|10)*1]
  //        = -0.5 + [1.0 + 1.0]           - [0.5 + 0.25]        = 0.75
  // eps[2] (inact) = h11 + [(11|00)*1 + (11|11)*2] - [(11|11)*1 + (10|01)*0.5]
  //              = -1.2 + [0.5 + 1.6]          - [0.8 + 0.125]        = -0.025
  // eps[4] (virt) = h22 + [(22|00)*1 + (22|11)*2] - [(22|22)*0 + (20|02)*0.5]
  //              = 3.0 + [0.4 + 0.6]           - [0.1]               = 3.9
  Eigen::VectorXd ref(6);
  ref << 0.75, 0.75, -0.025, -0.025, 3.9, 3.9;
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(eps(i), ref(i), 1e-9) << "i=" << i;
}

// ---------------------------------------------------------------------------
// Generator: S solves [H0, S] = V, so a single off-diagonal coupling maps to
//   S_PQ = f_PQ / (eps_P - eps_Q)  (coupling over energy gap).
// ---------------------------------------------------------------------------
TEST(Swpt2Kernel, GeneratorIsCouplingOverGap) {
  const int norb = 2;  // active=[0], virtual=[1], hopping tau, gap Delta
  const double tau = 0.2, eps0 = -0.5, eps1 = 3.0;
  Eigen::MatrixXd h(2, 2);
  h << eps0, tau, tau, eps1;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(16);  // no two-body

  auto H = sw::reference::build_tensors(h, h, g, g, g, /*e_core=*/0.0, norb);
  Eigen::VectorXd na(2), nb(2);
  na << 1, 0;
  nb << 0, 0;
  Eigen::VectorXd eps = sw::diagonal_fock_energies(h, g, na, nb, norb);

  sw::SoPartition part;
  part.n_so = 4;
  part.is_active = {1, 1, 0, 0};
  part.is_inactive = {0, 0, 0, 0};
  part.is_virtual = {0, 0, 1, 1};

  auto gen = sw::reference::make_generator(H, eps, part, sw::RegOptions{});
  // active0-alpha (SO 0) -> virtual-alpha (SO 2): S = tau / (eps0 - eps1)
  EXPECT_NEAR(gen.s1(0, 2), tau / (eps0 - eps1), 1e-12);
}

// ---------------------------------------------------------------------------
// Reference-buffer fold = textbook inactive Fock. A doubly-occupied domo d:
//   f_active[i,i] += 2 (ii|dd) - (id|di)          (Coulomb 2J minus exchange K)
//   core        += 2 h_dd + (dd|dd)              (two electrons in d)
// ---------------------------------------------------------------------------
TEST(Swpt2Kernel, InactiveFockFoldMeanField) {
  const int norb = 2;  // active=[0], domo=[1]
  const double h00 = 0.3, hdd = -0.5, J = 1.0, K = 0.2, Udd = 0.5, e0 = 0.7;
  Eigen::MatrixXd h(2, 2);
  h << h00, 0.0, 0.0, hdd;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(16);
  set_eri(g, 0, 0, 1, 1, norb, J);
  set_eri(g, 1, 1, 0, 0, norb, J);
  set_eri(g, 0, 1, 1, 0, norb, K);
  set_eri(g, 1, 0, 0, 1, norb, K);
  set_eri(g, 1, 1, 1, 1, norb, Udd);

  auto H = sw::reference::build_tensors(h, h, g, g, g, e0, norb);
  sw::SoPartition part;
  part.n_so = 4;
  part.is_active = {1, 1, 0, 0};
  part.is_inactive = {0, 0, 1, 1};
  part.is_virtual = {0, 0, 0, 0};

  auto [bd, od] = sw::reference::split_bd_od(H, part);
  auto res = sw::reference::mean_field_fold(bd, part);

  EXPECT_NEAR(res.e, e0 + (2 * hdd + Udd), 1e-12);            // 0.7 - 0.5 = 0.2
  EXPECT_NEAR(res.f_active(0, 0), h00 + (2 * J - K), 1e-12);  // 0.3 + 1.8 = 2.1
  EXPECT_NEAR(res.f_active(1, 1), h00 + (2 * J - K), 1e-12);  // spin symmetry
  // hermiticity of the folded one-body active operator
  const int M = 4;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < M; ++j)
      EXPECT_NEAR(res.f_active(i, j), res.f_active(j, i), 1e-12);
}

// ---------------------------------------------------------------------------
// Empty external space => the fold is the identity: bare active integrals back.
// ---------------------------------------------------------------------------
TEST(Swpt2Kernel, EmptyExternalSpaceIsIdentity) {
  const int norb = 2;
  Eigen::MatrixXd h(2, 2);
  h << 0.3, -0.4, -0.4, 0.9;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(16);
  set_eri(g, 0, 0, 0, 0, norb, 1.0);  // some active interaction
  const double e0 = 0.7;

  auto H = sw::reference::build_tensors(h, h, g, g, g, e0, norb);
  sw::SoPartition part;
  part.n_so = 4;
  part.is_active = {1, 1, 1, 1};
  part.is_inactive = {0, 0, 0, 0};
  part.is_virtual = {0, 0, 0, 0};

  auto [bd, od] = sw::reference::split_bd_od(H, part);
  auto res = sw::reference::mean_field_fold(bd, part);

  EXPECT_NEAR(res.e, e0, 1e-12);
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      EXPECT_NEAR(res.f_active(i, j), H.f(i, j), 1e-12);
}

// ---------------------------------------------------------------------------
// Full downfold, second-order level shift (textbook 2nd-order perturbation
// theory). One active orbital coupled to one external orbital by hopping tau
// with gap Delta gives an effective on-site shift of -/+ tau^2 / Delta.
// ---------------------------------------------------------------------------
namespace {
// Build the 2-orbital (active, external) partition + tensors for a hopping tau.
struct TwoOrbital {
  sw::reference::SoTensors H;
  Eigen::VectorXd eps;
  sw::SoPartition part;
  TwoOrbital(double e_active, double e_ext, double tau,
             bool external_is_virtual) {
    const int norb = 2;
    Eigen::MatrixXd h(2, 2);
    h << e_active, tau, tau, e_ext;
    Eigen::VectorXd g = Eigen::VectorXd::Zero(16);  // no two-body
    H = sw::reference::build_tensors(h, h, g, g, g, /*e_core=*/0.0, norb);

    Eigen::VectorXd na(2), nb(2);
    na << 1,
        (external_is_virtual ? 0
                             : 1);  // active0 singly (alpha); inactive doubly
    nb << 0, (external_is_virtual ? 0 : 1);
    eps = sw::diagonal_fock_energies(h, g, na, nb, norb);

    part.n_so = 4;
    part.is_active = {1, 1, 0, 0};
    part.is_inactive = external_is_virtual ? std::vector<char>{0, 0, 0, 0}
                                           : std::vector<char>{0, 0, 1, 1};
    part.is_virtual = external_is_virtual ? std::vector<char>{0, 0, 1, 1}
                                          : std::vector<char>{0, 0, 0, 0};
  }
};
}  // namespace

TEST(Swpt2Kernel, SecondOrderShiftVirtual) {
  const double e0 = -0.5, e1 = 3.0, tau = 0.2;  // active below virtual
  TwoOrbital sys(e0, e1, tau, /*external_is_virtual=*/true);
  auto res =
      sw::reference::downfold(sys.H, sys.eps, sys.part, sw::RegOptions{});

  const double shift = -tau * tau / (e1 - e0);         // -tau^2 / Delta
  EXPECT_NEAR(res.f_active(0, 0), e0 + shift, 1e-12);  // alpha
  EXPECT_NEAR(res.f_active(1, 1), e0 + shift, 1e-12);  // beta (spin symmetry)
  EXPECT_NEAR(res.higher_body_norm, 0.0, 1e-12);       // no >2-body from 1-body
}

TEST(Swpt2Kernel, SecondOrderShiftInactive) {
  const double e0 = 3.0, e1 = -0.5, tau = 0.2;  // active above inactive
  TwoOrbital sys(e0, e1, tau, /*external_is_virtual=*/false);
  auto res =
      sw::reference::downfold(sys.H, sys.eps, sys.part, sw::RegOptions{});

  const double shift =
      tau * tau / (e0 - e1);  // +tau^2 / Delta (hole excitation)
  EXPECT_NEAR(res.f_active(0, 0), e0 + shift, 1e-12);
  EXPECT_NEAR(res.f_active(1, 1), e0 + shift, 1e-12);
}

// ---------------------------------------------------------------------------
// Convergence to full configuration interaction.
//
// Scaling the active<->external coupling by lambda, each order of the downfold
// (bare mean-field fold -> + 1-body commutator -> + 2-body commutator) recovers
// a successively larger share of the coupling correlation: the full downfold
// removes >90% of it, and the 2-body commutator (which the +/- tau^2/Delta
// level-shift tests never exercise) is a nonzero, energy-lowering piece. This
// checks the full 2-body channel generation (v_active) and the buffer
// projection together.
//
// The reference is a self-contained bitmask full-CI over the *same*
// spin-orbital tensors the kernel uses (no external tool, no convention
// translation). The *absolute* residual is O(lambda^2), NOT O(lambda^3): the
// kernel uses orbital-energy (Moller-Plesset) denominators, which are the exact
// excitation energies only for a self-consistent canonical-HF reference. This
// synthetic (non-HF) reference carries an O(1) denominator error, so the
// coupling correlation is captured only to O(lambda^2). This is the expected
// MP-denominator limitation (why CASPT2/NEVPT2 use Dyall denominators), not a
// kernel defect -- the single-orbital tests above are exact.
//
// Window: spatial {0,1} active, {2} inactive (doubly occupied), {3} virtual;
// 2 active electrons (1 alpha, 1 beta) on top of the doubly-occupied inactive.
// ---------------------------------------------------------------------------
namespace {

// Set a chemist (pq|rs) integral together with its 8-fold permutational
// symmetry partners (real orbitals) so the resulting operator is Hermitian.
void set_sym_eri(Eigen::VectorXd& g, int p, int q, int r, int s, int norb,
                 double val) {
  const int perms[8][4] = {{p, q, r, s}, {q, p, r, s}, {p, q, s, r},
                           {q, p, s, r}, {r, s, p, q}, {s, r, p, q},
                           {r, s, q, p}, {s, r, q, p}};
  for (const auto& e : perms) g(sw::idx4(e[0], e[1], e[2], e[3], norb)) = val;
}

// One ladder operator on a determinant bitmask (Jordan-Wigner ordering:
// spin-orbitals occupied in ascending index order). Returns the sign and new
// mask, or ok=false if the operator annihilates the state (Pauli).
struct Ladder {
  std::uint64_t mask;
  int sign;
  bool ok;
};
Ladder apply_ladder(std::uint64_t mask, int orb, bool creation) {
  const std::uint64_t bit = std::uint64_t{1} << orb;
  const bool occupied = (mask & bit) != 0;
  if (creation == occupied) return {0, 0, false};  // Pauli
  const int below = __builtin_popcountll(mask & (bit - 1));
  return {mask ^ bit, (below & 1) ? -1 : 1, true};
}
// a^dag_P a_Q |mask>
Ladder apply_one(std::uint64_t mask, int P, int Q) {
  const Ladder a = apply_ladder(mask, Q, false);
  if (!a.ok) return a;
  const Ladder b = apply_ladder(a.mask, P, true);
  if (!b.ok) return b;
  return {b.mask, a.sign * b.sign, true};
}
// a^dag_P a^dag_Q a_R a_S |mask>  (apply S, R, then Q^dag, P^dag)
Ladder apply_two(std::uint64_t mask, int P, int Q, int R, int S) {
  Ladder r = apply_ladder(mask, S, false);
  if (!r.ok) return r;
  int sign = r.sign;
  r = apply_ladder(r.mask, R, false);
  if (!r.ok) return r;
  sign *= r.sign;
  r = apply_ladder(r.mask, Q, true);
  if (!r.ok) return r;
  sign *= r.sign;
  r = apply_ladder(r.mask, P, true);
  if (!r.ok) return r;
  return {r.mask, sign * r.sign, true};
}

// Lowest eigenvalue of  e0 + sum_PQ f_PQ a^dag_P a_Q
//                          + (1/4) sum_PQRS v_PQRS a^dag_P a^dag_Q a_R a_S,
// restricted to the spin-orbitals in `orbs` with `na` alpha (even index) and
// `nb` beta (odd index) electrons. Dense diagonalization of the tiny CI matrix.
double fci_ground_energy(double e0, const Eigen::MatrixXd& f,
                         const Eigen::VectorXd& v, int n_so,
                         const std::vector<int>& orbs, int na, int nb) {
  const int m = static_cast<int>(orbs.size());
  std::vector<std::uint64_t> basis;
  for (std::uint64_t sub = 0; sub < (std::uint64_t{1} << m); ++sub) {
    std::uint64_t mask = 0;
    int ca = 0, cb = 0;
    for (int k = 0; k < m; ++k) {
      if (sub & (std::uint64_t{1} << k)) {
        const int so = orbs[k];
        mask |= std::uint64_t{1} << so;
        ((so % 2 == 0) ? ca : cb)++;  // even index = alpha, odd = beta
      }
    }
    if (ca == na && cb == nb) basis.push_back(mask);
  }
  const int D = static_cast<int>(basis.size());
  std::map<std::uint64_t, int> index;
  for (int i = 0; i < D; ++i) index[basis[i]] = i;

  Eigen::MatrixXd Hmat = Eigen::MatrixXd::Zero(D, D);
  for (int col = 0; col < D; ++col) {
    const std::uint64_t ket = basis[col];
    Hmat(col, col) += e0;
    for (int P : orbs)
      for (int Q : orbs) {
        const double fpq = f(P, Q);
        if (fpq == 0.0) continue;
        const Ladder r = apply_one(ket, P, Q);
        if (!r.ok) continue;
        const auto it = index.find(r.mask);
        if (it != index.end()) Hmat(it->second, col) += fpq * r.sign;
      }
    for (int P : orbs)
      for (int Q : orbs)
        for (int R : orbs)
          for (int S : orbs) {
            const double vpqrs = v(sw::idx4(P, Q, R, S, n_so));
            if (vpqrs == 0.0) continue;
            const Ladder r = apply_two(ket, P, Q, R, S);
            if (!r.ok) continue;
            const auto it = index.find(r.mask);
            if (it != index.end())
              Hmat(it->second, col) += 0.25 * vpqrs * r.sign;
          }
  }
  Hmat = 0.5 * (Hmat + Hmat.transpose()).eval();  // H is Hermitian by build
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hmat);
  return es.eigenvalues()(0);
}

// Downfold the lambda-scaled system and measure how each order of the downfold
// approaches the full-CI ground-state energy of the whole window.
struct CiProbe {
  double err_bare;   // |E(bare mean-field fold)        - E_full|
  double err_1body;  // |E(+ 1-body commutator dressing) - E_full|
  double err_sw;     // |E(full second-order downfold)  - E_full|
  double dv2body;    // ||v_active(SW) - v_active(bare)||  (2-body commutator)
};
CiProbe downfold_vs_fci(double lam) {
  const int norb = 4, n_so = 8;
  // Single-determinant active reference |0^2>: active orbital 0 doubly
  // occupied, orbital 1 empty, with NO active-active coupling, so |0^2> is an
  // exact active eigenstate at lambda = 0 and the active space decouples from
  // the environment there. The orbital-energy (Moller-Plesset) denominators are
  // not those of a self-consistent HF reference, so the *absolute* second-order
  // residual is O(lambda^2); the physics under test is that each order of the
  // downfold (mean-field fold -> 1-body commutator -> 2-body commutator)
  // recovers a successively larger share of that O(lambda^2) coupling
  // correlation.
  Eigen::MatrixXd h1 = Eigen::MatrixXd::Zero(norb, norb);
  h1(0, 0) = 0.0;
  h1(1, 1) = 1.0;                    // active orbital energies (clear gap)
  h1(2, 2) = -3.0;                   // inactive (deep)
  h1(3, 3) = 3.0;                    // virtual (high)
  h1(0, 3) = h1(3, 0) = 0.40 * lam;  // active0-virtual  (occupation-changing)
  h1(1, 3) = h1(3, 1) = 0.30 * lam;  // active1-virtual
  h1(1, 2) = h1(2, 1) = 0.35 * lam;  // inactive-active1
  h1(2, 3) = h1(3, 2) = 0.30 * lam;  // inactive-virtual (environment)

  Eigen::VectorXd g = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  set_sym_eri(g, 0, 0, 0, 0, norb,
              0.60);  // active on-site (diagonal, no mixing)
  set_sym_eri(g, 1, 1, 1, 1, norb, 0.55);
  set_sym_eri(g, 2, 2, 2, 2, norb, 0.50);  // external density (unscaled, BD)
  set_sym_eri(g, 3, 3, 3, 3, norb, 0.45);
  set_sym_eri(g, 0, 0, 2, 2, norb, 0.30);
  set_sym_eri(g, 1, 1, 2, 2, norb, 0.28);
  set_sym_eri(g, 0, 0, 3, 3, norb, 0.20);
  set_sym_eri(g, 1, 1, 3, 3, norb, 0.18);
  set_sym_eri(g, 2, 2, 3, 3, norb, 0.15);
  set_sym_eri(g, 0, 3, 0, 0, norb, 0.10 * lam);  // active<->external 2-body
  set_sym_eri(g, 1, 2, 1, 1, norb, 0.10 * lam);  // (occupation-changing:
  set_sym_eri(g, 0, 2, 0, 3, norb, 0.08 * lam);  //  generates v_active)
  set_sym_eri(g, 0, 2, 1, 3, norb, 0.06 * lam);  // active-pair <-> ext-pair

  const auto H =
      sw::reference::build_tensors(h1, h1, g, g, g, /*e_core=*/0.0, norb);
  Eigen::VectorXd na(norb), nb(norb);
  na << 1.0, 0.0, 1.0, 0.0;  // |0^2> + doubly-occupied inactive: n = (2,0,2,0)
  nb << 1.0, 0.0, 1.0, 0.0;
  const Eigen::VectorXd eps = sw::diagonal_fock_energies(h1, g, na, nb, norb);

  sw::SoPartition part;
  part.n_so = n_so;
  part.is_active = {1, 1, 1, 1, 0, 0, 0, 0};
  part.is_inactive = {0, 0, 0, 0, 1, 1, 0, 0};
  part.is_virtual = {0, 0, 0, 0, 0, 0, 1, 1};

  const auto down = sw::reference::downfold(H, eps, part, sw::RegOptions{});
  const auto [bd, od] = sw::reference::split_bd_od(H, part);
  const auto mf = sw::reference::mean_field_fold(bd, part);

  const std::vector<int> active_so = {0, 1, 2, 3};
  const std::vector<int> all_so = {0, 1, 2, 3, 4, 5, 6, 7};
  const double e_full = fci_ground_energy(H.e0, H.f, H.v, n_so, all_so, 2, 2);
  const double e_bare =
      fci_ground_energy(mf.e, mf.f_active, mf.v_active, n_so, active_so, 1, 1);
  // + 1-body commutator dressing (downfolded scalar & one-body, bare two-body)
  const double e_1body = fci_ground_energy(down.e, down.f_active, mf.v_active,
                                           n_so, active_so, 1, 1);
  // full second-order downfold (adds the two-body commutator dressing)
  const double e_sw = fci_ground_energy(down.e, down.f_active, down.v_active,
                                        n_so, active_so, 1, 1);
  return {std::abs(e_bare - e_full), std::abs(e_1body - e_full),
          std::abs(e_sw - e_full), (down.v_active - mf.v_active).norm()};
}
}  // namespace

TEST(Swpt2Kernel, ConvergesToFullCI) {
  // At zero coupling the active space decouples exactly: downfold == full CI,
  // and no second-order (commutator) two-body dressing is produced.
  const CiProbe p0 = downfold_vs_fci(0.0);
  EXPECT_NEAR(p0.err_sw, 0.0, 1e-10);
  EXPECT_NEAR(p0.dv2body, 0.0, 1e-10);

  const CiProbe p = downfold_vs_fci(0.08);
  // The commutator generates a genuine two-body active dressing.
  EXPECT_GT(p.dv2body, 1e-6);
  // Each order of the downfold moves the active-space CI energy closer to full
  // CI: bare mean-field fold -> + 1-body commutator -> + 2-body commutator.
  EXPECT_LT(p.err_1body, 0.5 * p.err_bare);  // 1-body dressing helps
  EXPECT_LT(p.err_sw, 0.5 * p.err_1body);    // 2-body dressing helps further
  EXPECT_LT(p.err_sw, 0.1 * p.err_bare);     // full downfold recovers >90%

  // The residual is genuinely coupling-driven and second-order controlled:
  // halving the coupling shrinks it faster than linearly (here ~4x, O(lambda^2)
  // for this non-self-consistent reference).
  const CiProbe ph = downfold_vs_fci(0.04);
  EXPECT_LT(ph.err_sw, 0.35 * p.err_sw);
}

// ---------------------------------------------------------------------------
// Emission round-trip: spatial chemist -> build_tensors -> downfold with an
// empty external space (the identity fold) -> to_spatial_chemist must recover
// the original one-body and (chemist) two-body integrals. This validates the
// spin-orbital<->spatial-chemist inverse independent of the perturbation.
// ---------------------------------------------------------------------------
TEST(Swpt2Kernel, EmitSpatialChemistRoundTrip) {
  const int norb = 3;
  Eigen::MatrixXd h1(norb, norb);
  h1 << -0.50, 0.10, 0.20, 0.10, -1.20, -0.15, 0.20, -0.15, 3.00;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  set_sym_eri(g, 0, 0, 0, 0, norb, 1.00);
  set_sym_eri(g, 1, 1, 1, 1, norb, 0.80);
  set_sym_eri(g, 2, 2, 2, 2, norb, 0.60);
  set_sym_eri(g, 0, 0, 1, 1, norb, 0.50);
  set_sym_eri(g, 0, 0, 2, 2, norb, 0.40);
  set_sym_eri(g, 1, 1, 2, 2, norb, 0.30);
  set_sym_eri(g, 0, 1, 1, 0, norb, 0.25);  // exchange
  set_sym_eri(g, 0, 2, 2, 0, norb, 0.20);
  set_sym_eri(g, 0, 1, 2, 0, norb, 0.07);  // lower-symmetry, all-distinct pair

  const auto H =
      sw::reference::build_tensors(h1, h1, g, g, g, /*e_core=*/0.7, norb);
  Eigen::VectorXd na(norb), nb(norb);
  na << 1, 1, 0;
  nb << 1, 1, 0;
  const Eigen::VectorXd eps = sw::diagonal_fock_energies(h1, g, na, nb, norb);

  sw::SoPartition part;  // whole window active => downfold is the identity
  part.n_so = 2 * norb;
  part.is_active.assign(2 * norb, 1);
  part.is_inactive.assign(2 * norb, 0);
  part.is_virtual.assign(2 * norb, 0);

  const auto down = sw::reference::downfold(H, eps, part, sw::RegOptions{});
  const auto out = sw::reference::to_spatial_chemist(down, part);

  ASSERT_EQ(out.norb, norb);
  EXPECT_NEAR(out.core_energy, 0.7, 1e-12);
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      EXPECT_NEAR(out.one_body(p, q), h1(p, q), 1e-12) << "h " << p << q;
  for (int i = 0; i < norb * norb * norb * norb; ++i)
    EXPECT_NEAR(out.two_body(i), g(i), 1e-12) << "g flat " << i;
}

// ---------------------------------------------------------------------------
// Emission of a *dressed* operator. With a real external space the commutator
// produces a nonzero 2-body dressing; to_spatial_chemist -> build_tensors must
// preserve the active-space FCI energy. (The identity round-trip above has a
// zero commutator, so it never exercises the dressing's spin-orbital<->chemist
// conversion -- this is the path the data-layer emission uses.)
// active {0,1}, one virtual {2} to fold, 2 active electrons.
// ---------------------------------------------------------------------------
TEST(Swpt2Kernel, EmitPreservesDressedEnergy) {
  const int norb = 3, n_so = 6;
  Eigen::MatrixXd h1 = Eigen::MatrixXd::Zero(norb, norb);
  h1(0, 0) = 0.0;
  h1(1, 1) = 0.5;
  h1(2, 2) = 3.0;  // virtual, well separated
  Eigen::VectorXd g = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  set_sym_eri(g, 0, 0, 0, 0, norb, 0.60);
  set_sym_eri(g, 1, 1, 1, 1, norb, 0.50);
  set_sym_eri(g, 0, 0, 1, 1, norb, 0.30);
  set_sym_eri(g, 2, 2, 2, 2, norb, 0.40);
  set_sym_eri(g, 0, 0, 2, 2, norb, 0.20);
  set_sym_eri(g, 1, 1, 2, 2, norb, 0.20);
  set_sym_eri(g, 0, 2, 1, 2, norb, 0.15);  // active<->virtual: drives dressing
  set_sym_eri(g, 0, 2, 0, 0, norb, 0.10);

  const auto H =
      sw::reference::build_tensors(h1, h1, g, g, g, /*e_core=*/0.0, norb);
  Eigen::VectorXd na(norb), nb(norb);
  na << 1, 0, 0;  // |0^2> active reference, virtual empty
  nb << 1, 0, 0;
  const Eigen::VectorXd eps = sw::diagonal_fock_energies(h1, g, na, nb, norb);

  sw::SoPartition part;
  part.n_so = n_so;
  part.is_active = {1, 1, 1, 1, 0, 0};
  part.is_inactive = {0, 0, 0, 0, 0, 0};
  part.is_virtual = {0, 0, 0, 0, 1, 1};

  const auto down = sw::reference::downfold(H, eps, part, sw::RegOptions{});
  // ground energy directly from the downfold's spin-orbital tensors
  const double e_so = fci_ground_energy(down.e, down.f_active, down.v_active,
                                        n_so, {0, 1, 2, 3}, 1, 1);
  // ... and via the emitted spatial-chemist integrals rebuilt to spin-orbitals
  const auto act = sw::reference::to_spatial_chemist(down, part);
  const auto H2 = sw::reference::build_tensors(
      act.one_body, act.one_body, act.two_body, act.two_body, act.two_body,
      act.core_energy, act.norb);
  const double e_sp =
      fci_ground_energy(H2.e0, H2.f, H2.v, 2 * act.norb, {0, 1, 2, 3}, 1, 1);

  EXPECT_GT((down.v_active - H.v).norm(), 1e-6);  // there IS a dressing
  EXPECT_NEAR(e_so, e_sp, 1e-10);

  // intruder diagnostics are populated: a real fold has coupling (amplitude>0)
  // over a finite gap (denominator>0), and here the amplitude stays small.
  EXPECT_GT(down.max_amplitude, 0.0);
  EXPECT_GT(down.min_denominator, 0.0);
  EXPECT_LT(down.max_amplitude, 1.0);

  // The effective two-body is only 4-fold symmetric (Hermitian: (pq|rs)=(qp|sr)
  // and electron-exchange (pq|rs)=(rs|pq)), NOT the 8-fold of a genuine chemist
  // integral: the bra-swap (pq|rs)=(qp|rs) is *broken*. Consumers that assume
  // 8-fold symmetry (e.g. a chemist-integral Hamiltonian container) must be fed
  // the full dense n^4 block, not the canonical unique elements.
  const int m = act.norb;
  double asym_swap_bra = 0.0, asym_hermitian = 0.0, asym_exchange = 0.0;
  for (int p = 0; p < m; ++p)
    for (int q = 0; q < m; ++q)
      for (int r = 0; r < m; ++r)
        for (int s = 0; s < m; ++s) {
          const double v = act.two_body(sw::idx4(p, q, r, s, m));
          asym_swap_bra = std::max(  // (pq|rs) vs (qp|rs): 8-fold only
              asym_swap_bra,
              std::abs(v - act.two_body(sw::idx4(q, p, r, s, m))));
          asym_hermitian = std::max(  // (pq|rs) vs (qp|sr): Hermiticity (holds)
              asym_hermitian,
              std::abs(v - act.two_body(sw::idx4(q, p, s, r, m))));
          asym_exchange = std::max(  // (pq|rs) vs (rs|pq): exchange (holds)
              asym_exchange,
              std::abs(v - act.two_body(sw::idx4(r, s, p, q, m))));
        }
  EXPECT_LT(asym_hermitian, 1e-12) << "Hermiticity (qp|sr) must hold";
  EXPECT_LT(asym_exchange, 1e-12) << "electron-exchange (rs|pq) must hold";
  EXPECT_GT(asym_swap_bra, 1e-6)  // 4-fold, not 8-fold (documented, expected)
      << "effective 2-body is unexpectedly 8-fold symmetric";
}

// ---------------------------------------------------------------------------
// make_partition: the given orbitals -> active; the rest split by reference
// occupation into inactive (doubly occupied) / virtual (empty).
// ---------------------------------------------------------------------------
TEST(Swpt2Kernel, MakePartitionRolesFromOccupation) {
  // window of 4 spatial orbitals; active = {1,2}; occ = (2,_,_,0):
  // orbital 0 doubly occupied -> inactive, orbital 3 empty -> virtual.
  const int norb = 4;
  const auto part = sw::make_partition(norb, /*active=*/{1, 2},
                                       /*occupation=*/{2.0, 1.0, 0.0, 0.0});
  ASSERT_EQ(part.n_so, 8);
  const std::vector<char> active = {0, 0, 1, 1, 1, 1, 0, 0};
  const std::vector<char> inactive = {1, 1, 0, 0, 0, 0, 0, 0};
  const std::vector<char> virt = {0, 0, 0, 0, 0, 0, 1, 1};
  EXPECT_EQ(part.is_active, active);
  EXPECT_EQ(part.is_inactive, inactive);
  EXPECT_EQ(part.is_virtual, virt);
  // roles are mutually exclusive and exhaust the window
  for (int P = 0; P < 8; ++P)
    EXPECT_EQ(part.is_active[P] + part.is_inactive[P] + part.is_virtual[P], 1);

  EXPECT_THROW(sw::make_partition(2, {0}, {1.0}), std::invalid_argument);
  EXPECT_THROW(sw::make_partition(2, {5}, {1.0, 1.0}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Spin-blocked spatial storage (production path, Increment 1): the independent
// spin blocks reconstruct the dense spin-orbital antisymmetric tensor exactly.
// Arbitrary (distinct) aa/ab/bb integrals exercise every nonzero spin pattern;
// the check is purely algebraic so no physical symmetry of the inputs is
// required.
TEST(Swpt2Kernel, SpinBlockedTwoBodyRoundTrip) {
  const int norb = 3;
  const int M = 2 * norb;
  const int n4 = norb * norb * norb * norb;
  Eigen::VectorXd gaa(n4), gab(n4), gbb(n4);
  for (int i = 0; i < n4; ++i) {
    gaa(i) = std::sin(1.0 * i + 0.1);
    gab(i) = std::cos(0.7 * i + 0.2);
    gbb(i) = std::sin(0.3 * i + 1.3);
  }
  const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(norb, norb);
  const auto so =
      sw::reference::build_tensors(zero, zero, gaa, gab, gbb, 0.0, norb);
  const auto blk = sw::build_two_body_blocked(gaa, gab, gbb, norb);

  double max_dev = 0.0;
  for (int P = 0; P < M; ++P)
    for (int Q = 0; Q < M; ++Q)
      for (int R = 0; R < M; ++R)
        for (int S = 0; S < M; ++S)
          max_dev = std::max(max_dev,
                             std::abs(so.v(sw::idx4(P, Q, R, S, M)) -
                                      sw::so_v_from_blocked(blk, P, Q, R, S)));
  EXPECT_LT(max_dev, 1e-12);
}

// The on-the-fly (spin-blocked) downfold reproduces the dense spin-orbital
// oracle: the emitted effective Hamiltonian (core energy, active one-body,
// active chemist two-body) is identical, as are the intruder diagnostics and
// the discarded higher-body norm. Arbitrary distinct aa/ab/bb integrals
// exercise all spin blocks.
TEST(Swpt2Kernel, BlockedDownfoldMatchesSpinOrbital) {
  const int norb = 4;
  const int n4 = norb * norb * norb * norb;
  Eigen::VectorXd gaa(n4), gab(n4), gbb(n4);
  for (int i = 0; i < n4; ++i) {
    gaa(i) = 0.1 * std::sin(1.0 * i + 0.1);
    gab(i) = 0.1 * std::cos(0.7 * i + 0.2);
    gbb(i) = 0.1 * std::sin(0.3 * i + 1.3);
  }
  Eigen::MatrixXd h1 = Eigen::MatrixXd::Zero(norb, norb);
  for (int p = 0; p < norb; ++p) {
    h1(p, p) = -1.0 - 0.3 * p;
    for (int q = p + 1; q < norb; ++q) {
      h1(p, q) = 0.02 * (p + 1) * (q + 1);
      h1(q, p) = h1(p, q);
    }
  }
  const double e_core = 0.7;
  Eigen::VectorXd na(norb), nb(norb);
  na << 1.0, 0.5, 0.0, 0.0;
  nb = na;
  const auto eps = sw::diagonal_fock_energies(h1, gaa, na, nb, norb);
  const auto part = sw::make_partition(norb, /*active=*/{1, 2},
                                       /*occupation=*/{2.0, 1.0, 0.0, 0.0});
  const sw::RegOptions reg;  // default: plain 1/D with floor

  const auto so =
      sw::reference::build_tensors(h1, h1, gaa, gab, gbb, e_core, norb);
  const auto ref = sw::reference::downfold(so, eps, part, reg);

  const auto blk = sw::build_two_body_blocked(gaa, gab, gbb, norb);
  const auto f = sw::spin_orbital_one_body(h1, h1, norb);
  const auto got = sw::downfold_blocked(f, blk, eps, part, reg, e_core);

  // Compare the emitted effective Hamiltonians (representation-independent:
  // the oracle stores a full spin-orbital tensor, `got` a spatial spin block)
  // plus the diagnostics carried on the result structs.
  const auto ref_ham = sw::reference::to_spatial_chemist(ref, part);
  const auto got_ham = sw::to_spatial_chemist(got, part);
  ASSERT_EQ(got_ham.norb, ref_ham.norb);
  EXPECT_NEAR(got_ham.core_energy, ref_ham.core_energy, 1e-10);
  EXPECT_LT((got_ham.one_body - ref_ham.one_body).cwiseAbs().maxCoeff(), 1e-10);
  EXPECT_LT((got_ham.two_body - ref_ham.two_body).cwiseAbs().maxCoeff(), 1e-10);
  EXPECT_NEAR(got.min_denominator, ref.min_denominator, 1e-10);
  EXPECT_NEAR(got.max_amplitude, ref.max_amplitude, 1e-10);
  EXPECT_NEAR(got.higher_body_norm, ref.higher_body_norm, 1e-10);
}

}  // namespace
