// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Foundation-kernel tests for the second-order Schrieffer-Wolff downfolding.
//
// Reference values are grounded in first principles, not in any external tool:
//   * generalized-Fock denominator energies from their closed-form definition,
//   * the reference-external fold as textbook inactive-Fock mean field
//     (a doubly-occupied external orbital contributes 2J - K to the
//     active-space Fock and
//      2 h_dd + (dd|dd) to the core energy),
//   * invariants (hermiticity) and the empty-external identity.
// Each expected value is derived in a comment from the chosen input integrals.
// (An independent OpenFermion cross-check of the full method exists as a
//  development-time check; it is deliberately not a dependency of these tests.)

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"
#include "testing_utilities_swpt2.hpp"

namespace {
namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

// Set a chemist integral (pq|rs) in a flat (norb^4) array.
void set_eri(Eigen::VectorXd& g, int p, int q, int r, int s, int norb,
             double val) {
  g(sw::idx4(p, q, r, s, norb)) = val;
}

// Spin-orbital denominator energies for a diagonal-occupation reference: the
// diagonal of `generalized_fock_matrix` with density diag(na + nb).
Eigen::VectorXd diagonal_fock_energies(const Eigen::MatrixXd& h1,
                                       const Eigen::VectorXd& g,
                                       const Eigen::VectorXd& na,
                                       const Eigen::VectorXd& nb, int norb) {
  const Eigen::MatrixXd density = Eigen::MatrixXd((na + nb).asDiagonal());
  const Eigen::MatrixXd fock =
      sw::generalized_fock_matrix(h1, g, density, norb);
  Eigen::VectorXd eps(2 * norb);
  for (int p = 0; p < norb; ++p) eps(2 * p) = eps(2 * p + 1) = fock(p, p);
  return eps;
}

// ---------------------------------------------------------------------------
// Generalized-Fock diagonal: eps_p^sigma = h_pp + sum_r (pp|rr) n_r
//                                                 - sum_r (pr|rp) n_r^sigma.
// ---------------------------------------------------------------------------
TEST(Swpt2KernelTest, GeneralizedFockDiagonalFromDefinition) {
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

  Eigen::VectorXd eps = diagonal_fock_energies(h, g, na, nb, norb);

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
// Reference-external fold = textbook inactive Fock. A doubly-occupied domo d:
//   f_active[i,i] += 2 (ii|dd) - (id|di)          (Coulomb 2J minus exchange K)
//   core        += 2 h_dd + (dd|dd)              (two electrons in d)
// ---------------------------------------------------------------------------
TEST(Swpt2KernelTest, InactiveFockFoldMeanField) {
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

  sw::SpinOrbitalPartition part;
  part.n_so = 4;
  part.is_active = {1, 1, 0, 0};
  part.is_inactive = {0, 0, 1, 1};
  part.is_virtual = {0, 0, 0, 0};

  const auto blocked = sw::build_two_body_blocked(g, norb);
  const auto one_body = sw::spin_orbital_one_body(h, h, norb);
  Eigen::VectorXd eps(4);
  eps << h00, h00, hdd, hdd;
  const auto res = sw::downfold_blocked(one_body, blocked, eps, part,
                                        sw::RegularizerOptions{}, e0);

  EXPECT_NEAR(res.e, e0 + (2 * hdd + Udd), 1e-12);            // 0.7 - 0.5 = 0.2
  EXPECT_NEAR(res.f_active(0, 0), h00 + (2 * J - K), 1e-12);  // 0.3 + 1.8 = 2.1
  EXPECT_NEAR(res.f_active(1, 1), h00 + (2 * J - K), 1e-12);  // spin symmetry
  // hermiticity of the folded one-body active operator
  for (int i = 0; i < res.f_active.rows(); ++i)
    for (int j = 0; j < res.f_active.cols(); ++j)
      EXPECT_NEAR(res.f_active(i, j), res.f_active(j, i), 1e-12);
}

// ---------------------------------------------------------------------------
// Empty external space => the fold is the identity: bare active integrals back.
// ---------------------------------------------------------------------------
TEST(Swpt2KernelTest, EmptyExternalSpaceIsIdentity) {
  const int norb = 2;
  Eigen::MatrixXd h(2, 2);
  h << 0.3, -0.4, -0.4, 0.9;
  Eigen::VectorXd g = Eigen::VectorXd::Zero(16);
  set_eri(g, 0, 0, 0, 0, norb, 1.0);  // some active interaction
  const double e0 = 0.7;

  sw::SpinOrbitalPartition part;
  part.n_so = 4;
  part.is_active = {1, 1, 1, 1};
  part.is_inactive = {0, 0, 0, 0};
  part.is_virtual = {0, 0, 0, 0};

  const auto blocked = sw::build_two_body_blocked(g, norb);
  const auto one_body = sw::spin_orbital_one_body(h, h, norb);
  Eigen::VectorXd eps(4);
  eps << h(0, 0), h(0, 0), h(1, 1), h(1, 1);
  const auto res = sw::downfold_blocked(one_body, blocked, eps, part,
                                        sw::RegularizerOptions{}, e0);
  const auto emitted = sw::to_spatial_chemist(res, part);

  EXPECT_NEAR(emitted.core_energy, e0, 1e-12);
  EXPECT_LT((emitted.one_body - h).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT((emitted.two_body - g).cwiseAbs().maxCoeff(), 1e-12);
}

// ---------------------------------------------------------------------------
// Full downfold, second-order level shift (textbook 2nd-order perturbation
// theory). One active orbital coupled to one external orbital by hopping tau
// with gap Delta gives an effective on-site shift of -/+ tau^2 / Delta.
// ---------------------------------------------------------------------------
namespace {
// Build the 2-orbital (active, external) partition + tensors for a hopping tau.
struct TwoOrbital {
  Eigen::MatrixXd one_body;
  sw::SpinBlockedTwoBody two_body;
  Eigen::VectorXd eps;
  sw::SpinOrbitalPartition part;
  TwoOrbital(double e_active, double e_ext, double tau,
             bool external_is_virtual) {
    const int norb = 2;
    Eigen::MatrixXd h(2, 2);
    h << e_active, tau, tau, e_ext;
    Eigen::VectorXd g = Eigen::VectorXd::Zero(16);  // no two-body
    one_body = sw::spin_orbital_one_body(h, h, norb);
    two_body = sw::build_two_body_blocked(g, norb);

    Eigen::VectorXd na(2), nb(2);
    na << 1,
        (external_is_virtual ? 0
                             : 1);  // active0 singly (alpha); inactive doubly
    nb << 0, (external_is_virtual ? 0 : 1);
    eps = diagonal_fock_energies(h, g, na, nb, norb);

    part.n_so = 4;
    part.is_active = {1, 1, 0, 0};
    part.is_inactive = external_is_virtual ? std::vector<char>{0, 0, 0, 0}
                                           : std::vector<char>{0, 0, 1, 1};
    part.is_virtual = external_is_virtual ? std::vector<char>{0, 0, 1, 1}
                                          : std::vector<char>{0, 0, 0, 0};
  }
};
}  // namespace

TEST(Swpt2KernelTest, SecondOrderShiftVirtual) {
  const double e0 = -0.5, e1 = 3.0, tau = 0.2;  // active below virtual
  TwoOrbital sys(e0, e1, tau, /*external_is_virtual=*/true);
  const auto res =
      sw::downfold_blocked(sys.one_body, sys.two_body, sys.eps, sys.part,
                           sw::RegularizerOptions{}, 0.0);

  const double shift = -tau * tau / (e1 - e0);         // -tau^2 / Delta
  EXPECT_NEAR(res.f_active(0, 0), e0 + shift, 1e-12);  // alpha
  EXPECT_NEAR(res.f_active(1, 1), e0 + shift, 1e-12);  // beta (spin symmetry)
}

TEST(Swpt2KernelTest, SecondOrderShiftInactive) {
  const double e0 = 3.0, e1 = -0.5, tau = 0.2;  // active above inactive
  TwoOrbital sys(e0, e1, tau, /*external_is_virtual=*/false);
  const auto res =
      sw::downfold_blocked(sys.one_body, sys.two_body, sys.eps, sys.part,
                           sw::RegularizerOptions{}, 0.0);

  const double shift =
      tau * tau / (e0 - e1);  // +tau^2 / Delta (hole excitation)
  EXPECT_NEAR(res.f_active(0, 0), e0 + shift, 1e-12);
  EXPECT_NEAR(res.f_active(1, 1), e0 + shift, 1e-12);
}

// ---------------------------------------------------------------------------
// Self-contained Fock-space oracle helpers. They assemble the SW parts and
// evaluate full-CI energies directly from the *same* spin-orbital tensors the
// kernel uses (no external tool, no convention translation). Shared by the
// physical-convergence, independent-coefficient, and emission tests below.
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

void add_term_matrix(Eigen::MatrixXd& matrix, int n_so, double coefficient,
                     const std::vector<int>& create,
                     const std::vector<int>& annihilate) {
  if (coefficient == 0.0) return;
  const std::uint64_t dimension = std::uint64_t{1} << n_so;
  for (std::uint64_t ket = 0; ket < dimension; ++ket) {
    Ladder state{ket, 1, true};
    for (auto it = annihilate.rbegin(); it != annihilate.rend(); ++it) {
      const Ladder next = apply_ladder(state.mask, *it, false);
      if (!next.ok) {
        state.ok = false;
        break;
      }
      state.mask = next.mask;
      state.sign *= next.sign;
    }
    if (!state.ok) continue;
    for (auto it = create.rbegin(); it != create.rend(); ++it) {
      const Ladder next = apply_ladder(state.mask, *it, true);
      if (!next.ok) {
        state.ok = false;
        break;
      }
      state.mask = next.mask;
      state.sign *= next.sign;
    }
    if (state.ok)
      matrix(static_cast<int>(state.mask), static_cast<int>(ket)) +=
          coefficient * state.sign;
  }
}

bool changes_external_count(const std::vector<int>& create,
                            const std::vector<int>& annihilate,
                            const std::vector<char>& role) {
  int change = 0;
  for (int orbital : create) change += role[orbital];
  for (int orbital : annihilate) change -= role[orbital];
  return change != 0;
}

struct MatrixSwParts {
  Eigen::MatrixXd block_diagonal;
  Eigen::MatrixXd off_diagonal;
  Eigen::MatrixXd generator;
};

// Build H_BD, H_OD, and S directly in determinant space from restricted
// chemist integrals. This is separate from the kernel's antisymmetric tensors
// and projected-Wick implementation.
MatrixSwParts build_matrix_sw_parts(const Eigen::MatrixXd& h1,
                                    const Eigen::VectorXd& g,
                                    const Eigen::VectorXd& eps,
                                    const sw::SpinOrbitalPartition& part,
                                    double core_energy,
                                    const sw::RegularizerOptions& reg = {}) {
  const int norb = static_cast<int>(h1.rows());
  const int n_so = 2 * norb;
  const int dimension = 1 << n_so;
  MatrixSwParts result{Eigen::MatrixXd::Zero(dimension, dimension),
                       Eigen::MatrixXd::Zero(dimension, dimension),
                       Eigen::MatrixXd::Zero(dimension, dimension)};
  result.block_diagonal.diagonal().array() += core_energy;

  const auto add = [&](double coefficient, const std::vector<int>& create,
                       const std::vector<int>& annihilate) {
    if (coefficient == 0.0) return;
    const bool off_diagonal =
        changes_external_count(create, annihilate, part.is_inactive) ||
        changes_external_count(create, annihilate, part.is_virtual);
    add_term_matrix(off_diagonal ? result.off_diagonal : result.block_diagonal,
                    n_so, coefficient, create, annihilate);
    if (!off_diagonal) return;
    double denominator = 0.0;
    for (int orbital : create) denominator += eps(orbital);
    for (int orbital : annihilate) denominator -= eps(orbital);
    add_term_matrix(result.generator, n_so,
                    coefficient * sw::regularized_inverse(denominator, reg),
                    create, annihilate);
  };

  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int spin = 0; spin < 2; ++spin)
        add(h1(p, q), {2 * p + spin}, {2 * q + spin});

  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s) {
          const double value = g(sw::idx4(p, q, r, s, norb));
          for (int spin = 0; spin < 2; ++spin)
            add(0.5 * value, {2 * p + spin, 2 * r + spin},
                {2 * s + spin, 2 * q + spin});
          add(value, {2 * p, 2 * r + 1}, {2 * s + 1, 2 * q});
        }
  return result;
}

Eigen::MatrixXd spatial_chemist_matrix(
    const sw::ActiveHamiltonian& hamiltonian) {
  const int norb = hamiltonian.norb;
  const int n_so = 2 * norb;
  const int dimension = 1 << n_so;
  Eigen::MatrixXd matrix =
      hamiltonian.core_energy * Eigen::MatrixXd::Identity(dimension, dimension);
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int spin = 0; spin < 2; ++spin)
        add_term_matrix(matrix, n_so, hamiltonian.one_body(p, q),
                        {2 * p + spin}, {2 * q + spin});
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s) {
          const double value = hamiltonian.two_body(sw::idx4(p, q, r, s, norb));
          for (int spin = 0; spin < 2; ++spin)
            add_term_matrix(matrix, n_so, 0.5 * value,
                            {2 * p + spin, 2 * r + spin},
                            {2 * s + spin, 2 * q + spin});
          add_term_matrix(matrix, n_so, value, {2 * p, 2 * r + 1},
                          {2 * s + 1, 2 * q});
        }
  return matrix;
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

}  // namespace

// ---------------------------------------------------------------------------
// Physical convergence (the defining PT property, ported from eff-ham
// tests/downfold_op/test_toy.py::test_virtual_buffer_convergence_with_coupling):
// the second-order downfold error must shrink as the active<->external coupling
// is reduced. A closed-shell reference (spatial orbital 0 doubly occupied) with
// a dimer partner {1} and a well-separated virtual {2}; both the one-body and
// two-body active<->virtual couplings are scaled by lambda. Exact = full-CI
// over the whole {0,1,2} window; effective = full-CI over the emitted active
// {0,1} operator. The residual falls monotonically and faster than linearly as
// lambda -> 0. (The absolute residual is O(lambda^2), not O(lambda^3): the
// synthetic reference is not self-consistent HF, so the orbital-energy
// denominators carry an O(1) error -- the expected MP-denominator limitation,
// not a kernel defect. The single-orbital shift tests above are exact.)
// ---------------------------------------------------------------------------
TEST(Swpt2KernelTest, VirtualExternalSpaceConvergesAsCouplingShrinks) {
  const int norb = 3;  // active {0,1}, virtual {2}
  const auto residual = [&](double lambda) {
    Eigen::MatrixXd h1 = Eigen::MatrixXd::Zero(norb, norb);
    h1(1, 1) = 0.5;
    h1(2, 2) = 3.0;                       // virtual, well separated
    h1(0, 2) = h1(2, 0) = -0.2 * lambda;  // one-body active<->virtual coupling
    Eigen::VectorXd g = Eigen::VectorXd::Zero(norb * norb * norb * norb);
    set_sym_eri(g, 0, 0, 0, 0, norb, 0.60);
    set_sym_eri(g, 1, 1, 1, 1, norb, 0.50);
    set_sym_eri(g, 0, 0, 1, 1, norb, 0.30);
    set_sym_eri(g, 2, 2, 2, 2, norb, 0.40);
    set_sym_eri(g, 0, 0, 2, 2, norb, 0.20);
    set_sym_eri(g, 1, 1, 2, 2, norb, 0.20);
    set_sym_eri(g, 0, 2, 0, 0, norb,
                0.10 * lambda);  // two-body active<->virtual
    set_sym_eri(g, 0, 2, 1, 1, norb, 0.10 * lambda);

    Eigen::VectorXd na(norb), nb(norb);
    na << 1, 0,
        0;  // |0^2> closed-shell reference, dimer partner + virtual empty
    nb << 1, 0, 0;
    const Eigen::VectorXd eps = diagonal_fock_energies(h1, g, na, nb, norb);

    sw::SpinOrbitalPartition part;
    part.n_so = 2 * norb;
    part.is_active = {1, 1, 1, 1, 0, 0};
    part.is_inactive = {0, 0, 0, 0, 0, 0};
    part.is_virtual = {0, 0, 0, 0, 1, 1};

    // exact: full-CI over the whole {0,1,2} window (2 electrons, 1 alpha 1
    // beta)
    const auto full =
        testing::build_spin_orbital_tensors(h1, h1, g, g, g, 0.0, norb);
    const double e_exact =
        fci_ground_energy(full.core_energy, full.one_body, full.two_body,
                          2 * norb, {0, 1, 2, 3, 4, 5}, 1, 1);

    // effective: downfold onto active {0,1}, then full-CI over that operator
    const auto blocked = sw::build_two_body_blocked(g, norb);
    const auto one_body = sw::spin_orbital_one_body(h1, h1, norb);
    const auto down = sw::downfold_blocked(one_body, blocked, eps, part,
                                           sw::RegularizerOptions{}, 0.0);
    const auto act = sw::to_spatial_chemist(down, part);
    const auto rebuilt = testing::build_spin_orbital_tensors(
        act.one_body, act.one_body, act.two_body, act.two_body, act.two_body,
        act.core_energy, act.norb);
    const double e_eff =
        fci_ground_energy(rebuilt.core_energy, rebuilt.one_body,
                          rebuilt.two_body, 2 * act.norb, {0, 1, 2, 3}, 1, 1);
    return std::abs(e_exact - e_eff);
  };

  const double e_04 = residual(0.4);
  const double e_02 = residual(0.2);
  const double e_01 = residual(0.1);
  EXPECT_GT(e_04, e_02);  // error shrinks with the coupling
  EXPECT_GT(e_02, e_01);
  EXPECT_LT(e_02, 0.6 * e_04);  // and faster than linearly (>= second order)
  EXPECT_LT(e_01, 0.6 * e_02);
}

// Independent coefficient-level validation. Build the SW transformation as
// ordinary matrices over the complete window Fock space, project the inactive
// orbitals occupied and the virtual orbitals empty, and compare with the
// production kernel's emitted active operator. Zero-, one-, and two-particle
// active sectors identify every retained scalar, one-body, and two-body
// coefficient while excluding the intentionally discarded >=3-body terms.
TEST(Swpt2KernelTest, ProductionMatchesIndependentFockSpaceMatrix) {
  const int norb = 4;
  const int n_so = 2 * norb;
  Eigen::MatrixXd h1 = Eigen::MatrixXd::Zero(norb, norb);
  h1.diagonal() << -0.8, 0.35, -2.7, 3.4;
  h1(0, 3) = h1(3, 0) = 0.13;
  h1(1, 2) = h1(2, 1) = -0.09;
  h1(2, 3) = h1(3, 2) = 0.07;

  Eigen::VectorXd g = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  set_sym_eri(g, 0, 0, 0, 0, norb, 0.60);
  set_sym_eri(g, 1, 1, 1, 1, norb, 0.52);
  set_sym_eri(g, 2, 2, 2, 2, norb, 0.48);
  set_sym_eri(g, 0, 0, 2, 2, norb, 0.21);
  set_sym_eri(g, 1, 1, 2, 2, norb, 0.17);
  set_sym_eri(g, 0, 0, 3, 3, norb, 0.14);
  set_sym_eri(g, 1, 1, 3, 3, norb, 0.12);
  set_sym_eri(g, 0, 3, 1, 2, norb, 0.08);
  set_sym_eri(g, 0, 2, 1, 3, norb, -0.06);
  set_sym_eri(g, 0, 3, 0, 0, norb, 0.05);
  set_sym_eri(g, 1, 2, 1, 1, norb, -0.04);

  Eigen::VectorXd eps(n_so);
  eps << -0.8, -0.8, 0.35, 0.35, -2.7, -2.7, 3.4, 3.4;
  const double core_energy = 0.23;

  sw::RegularizerOptions bare;
  sw::RegularizerOptions shifted;
  shifted.denom_imaginary_shift = 0.4;
  sw::RegularizerOptions flow;
  flow.denom_flow = 1.2;

  struct PartitionCase {
    std::vector<int> active;
    int inactive;
  };
  const auto validate = [&](const Eigen::MatrixXd& case_h1,
                            const Eigen::VectorXd& case_g) {
    const auto blocked = sw::build_two_body_blocked(case_g, norb);
    const auto one_body = sw::spin_orbital_one_body(case_h1, case_h1, norb);
    for (const auto& partition_case :
         {PartitionCase{{0, 1}, 2}, PartitionCase{{1, 2}, 0}}) {
      std::vector<int> virtual_spatial;
      for (int o = 0; o < norb; ++o) {
        const auto& kept = partition_case.active;
        if (o != partition_case.inactive &&
            std::find(kept.begin(), kept.end(), o) == kept.end())
          virtual_spatial.push_back(o);
      }
      const auto part =
          sw::make_partition(norb, partition_case.active,
                             {partition_case.inactive}, virtual_spatial);
      const std::uint64_t external_reference =
          (std::uint64_t{1} << (2 * partition_case.inactive)) |
          (std::uint64_t{1} << (2 * partition_case.inactive + 1));
      const auto expand_active = [&](std::uint64_t compact) {
        std::uint64_t full = external_reference;
        for (int active = 0; active < 2; ++active)
          for (int spin = 0; spin < 2; ++spin)
            if (compact & (std::uint64_t{1} << (2 * active + spin)))
              full |= std::uint64_t{1}
                      << (2 * partition_case.active[active] + spin);
        return static_cast<int>(full);
      };

      for (const auto& reg : {bare, shifted, flow}) {
        const MatrixSwParts reference =
            build_matrix_sw_parts(case_h1, case_g, eps, part, core_energy, reg);
        const Eigen::MatrixXd reference_effective =
            reference.block_diagonal +
            0.5 * (reference.generator * reference.off_diagonal -
                   reference.off_diagonal * reference.generator);
        const auto downfolded = sw::downfold_blocked(one_body, blocked, eps,
                                                     part, reg, core_energy);
        const auto emitted = sw::to_spatial_chemist(downfolded, part);
        const Eigen::MatrixXd production = spatial_chemist_matrix(emitted);

        double max_error = 0.0;
        double max_correction = 0.0;
        for (std::uint64_t bra = 0; bra < 16; ++bra) {
          if (__builtin_popcountll(bra) > 2) continue;
          for (std::uint64_t ket = 0; ket < 16; ++ket) {
            if (__builtin_popcountll(ket) > 2) continue;
            const int full_bra = expand_active(bra);
            const int full_ket = expand_active(ket);
            max_error = std::max(
                max_error, std::abs(reference_effective(full_bra, full_ket) -
                                    production(static_cast<int>(bra),
                                               static_cast<int>(ket))));
            max_correction = std::max(
                max_correction,
                std::abs(reference_effective(full_bra, full_ket) -
                         reference.block_diagonal(full_bra, full_ket)));
          }
        }
        double max_two_body_dressing = 0.0;
        for (int p = 0; p < emitted.norb; ++p)
          for (int q = 0; q < emitted.norb; ++q)
            for (int r = 0; r < emitted.norb; ++r)
              for (int s = 0; s < emitted.norb; ++s)
                max_two_body_dressing = std::max(
                    max_two_body_dressing,
                    std::abs(
                        emitted.two_body(sw::idx4(p, q, r, s, emitted.norb)) -
                        case_g(sw::idx4(partition_case.active[p],
                                        partition_case.active[q],
                                        partition_case.active[r],
                                        partition_case.active[s], norb))));
        EXPECT_GT(max_correction, 1e-6);
        EXPECT_GT(max_two_body_dressing, 1e-6);
        EXPECT_LT(max_error, 1e-10);
      }
    }
  };

  validate(h1, g);

  std::mt19937 generator(0x5A17u);
  std::uniform_real_distribution<double> coupling(-0.08, 0.08);
  for (int sample = 0; sample < 2; ++sample) {
    Eigen::MatrixXd random_h1 = Eigen::MatrixXd::Zero(norb, norb);
    random_h1.diagonal() << -0.8, 0.35, -2.7, 3.4;
    for (int p = 0; p < norb; ++p)
      for (int q = p + 1; q < norb; ++q)
        random_h1(p, q) = random_h1(q, p) = coupling(generator);

    Eigen::VectorXd random_g = Eigen::VectorXd::Zero(g.size());
    for (int p = 0; p < norb; ++p)
      for (int q = p; q < norb; ++q)
        for (int r = 0; r < norb; ++r)
          for (int s = r; s < norb; ++s)
            set_sym_eri(random_g, p, q, r, s, norb, coupling(generator));
    validate(random_h1, random_g);
  }
}

// ---------------------------------------------------------------------------
// Emission round-trip: spatial chemist -> build_tensors -> downfold with an
// empty external space (the identity fold) -> to_spatial_chemist must recover
// the original one-body and (chemist) two-body integrals. This validates the
// spin-orbital<->spatial-chemist inverse independent of the perturbation.
// ---------------------------------------------------------------------------
TEST(Swpt2KernelTest, EmitSpatialChemistRoundTrip) {
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

  Eigen::VectorXd na(norb), nb(norb);
  na << 1, 1, 0;
  nb << 1, 1, 0;
  const Eigen::VectorXd eps = diagonal_fock_energies(h1, g, na, nb, norb);

  sw::SpinOrbitalPartition
      part;  // whole window active => downfold is the identity
  part.n_so = 2 * norb;
  part.is_active.assign(2 * norb, 1);
  part.is_inactive.assign(2 * norb, 0);
  part.is_virtual.assign(2 * norb, 0);

  const auto blocked = sw::build_two_body_blocked(g, norb);
  const auto one_body = sw::spin_orbital_one_body(h1, h1, norb);
  const auto down = sw::downfold_blocked(one_body, blocked, eps, part,
                                         sw::RegularizerOptions{}, 0.7);
  const auto out = sw::to_spatial_chemist(down, part);

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
TEST(Swpt2KernelTest, EmitPreservesDressedEnergy) {
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

  Eigen::VectorXd na(norb), nb(norb);
  na << 1, 0, 0;  // |0^2> active reference, virtual empty
  nb << 1, 0, 0;
  const Eigen::VectorXd eps = diagonal_fock_energies(h1, g, na, nb, norb);

  sw::SpinOrbitalPartition part;
  part.n_so = n_so;
  part.is_active = {1, 1, 1, 1, 0, 0};
  part.is_inactive = {0, 0, 0, 0, 0, 0};
  part.is_virtual = {0, 0, 0, 0, 1, 1};

  const auto blocked = sw::build_two_body_blocked(g, norb);
  const auto one_body = sw::spin_orbital_one_body(h1, h1, norb);
  const auto down = sw::downfold_blocked(one_body, blocked, eps, part,
                                         sw::RegularizerOptions{}, 0.0);
  const auto act = sw::to_spatial_chemist(down, part);
  const auto rebuilt = testing::build_spin_orbital_tensors(
      act.one_body, act.one_body, act.two_body, act.two_body, act.two_body,
      act.core_energy, act.norb);
  const double e_sp =
      fci_ground_energy(rebuilt.core_energy, rebuilt.one_body, rebuilt.two_body,
                        2 * act.norb, {0, 1, 2, 3}, 1, 1);
  const Eigen::MatrixXd full_matrix = spatial_chemist_matrix(act);
  std::vector<int> sector;
  for (int state = 0; state < full_matrix.rows(); ++state) {
    int n_alpha = 0, n_beta = 0;
    for (int orbital = 0; orbital < 2 * act.norb; ++orbital)
      if (state & (1 << orbital)) ((orbital % 2 == 0) ? n_alpha : n_beta)++;
    if (n_alpha == 1 && n_beta == 1) sector.push_back(state);
  }
  Eigen::MatrixXd sector_matrix(sector.size(), sector.size());
  for (int row = 0; row < static_cast<int>(sector.size()); ++row)
    for (int col = 0; col < static_cast<int>(sector.size()); ++col)
      sector_matrix(row, col) = full_matrix(sector[row], sector[col]);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(sector_matrix);
  const double e_matrix = eigensolver.eigenvalues()(0);

  Eigen::VectorXd bare_active(act.two_body.size());
  for (int p = 0; p < act.norb; ++p)
    for (int q = 0; q < act.norb; ++q)
      for (int r = 0; r < act.norb; ++r)
        for (int s = 0; s < act.norb; ++s)
          bare_active(sw::idx4(p, q, r, s, act.norb)) =
              g(sw::idx4(p, q, r, s, norb));
  EXPECT_GT((act.two_body - bare_active).norm(), 1e-6);
  EXPECT_NEAR(e_sp, e_matrix, 1e-10);

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
// make_partition: explicit active / inactive / virtual index lists must be
// disjoint and cover the window.
// ---------------------------------------------------------------------------
TEST(Swpt2KernelTest, MakePartitionRolesFromExplicitLists) {
  // window of 4 spatial orbitals; active = {1,2}, inactive = {0}, virtual = {3}
  const int norb = 4;
  const auto part = sw::make_partition(norb, /*active=*/{1, 2},
                                       /*inactive=*/{0}, /*virtual=*/{3});
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

  // out of range, assigned twice, and left without a role
  EXPECT_THROW(sw::make_partition(2, {5}, {0}, {1}), std::invalid_argument);
  EXPECT_THROW(sw::make_partition(2, {0}, {0}, {1}), std::invalid_argument);
  EXPECT_THROW(sw::make_partition(2, {0}, {}, {}), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// partition_window: the folding policy. Kept orbitals become active; the rest
// are folded by rounding their reference occupation to the nearer of 2 or 0.
// ---------------------------------------------------------------------------
TEST(Swpt2KernelTest, PartitionWindowRoundsFoldedOccupations) {
  const std::vector<std::size_t> window = {10, 11, 12, 13};
  const std::unordered_set<std::size_t> kept = {11, 12};

  // Integer occupations: nothing to round.
  {
    const auto split = sw::partition_window({2.0, 1.0, 1.0, 0.0}, window, kept,
                                            /*window_electrons=*/4, 0.5);
    EXPECT_EQ(split.active_spatial, (std::vector<int>{1, 2}));
    EXPECT_EQ(split.inactive_spatial, (std::vector<int>{0}));
    EXPECT_EQ(split.virtual_spatial, (std::vector<int>{3}));
    EXPECT_EQ(split.active_electrons, 2);
    EXPECT_DOUBLE_EQ(split.worst_deviation, 0.0);
    EXPECT_DOUBLE_EQ(split.folded_charge_error, 0.0);
  }

  // A correlated pair kept together on the folded side: both orbitals round,
  // but in opposite directions, so the charge error cancels exactly.
  {
    const auto split = sw::partition_window({1.98, 1.0, 1.0, 0.02}, window,
                                            kept, /*window_electrons=*/4, 0.5);
    EXPECT_EQ(split.inactive_spatial, (std::vector<int>{0}));
    EXPECT_EQ(split.virtual_spatial, (std::vector<int>{3}));
    EXPECT_EQ(split.active_electrons, 2);
    EXPECT_NEAR(split.worst_deviation, 0.02, 1e-12);
    EXPECT_EQ(split.worst_orbital, 10u);
    EXPECT_NEAR(split.worst_occupation, 1.98, 1e-12);
    EXPECT_NEAR(split.folded_charge_error, 0.0, 1e-12);
  }
}

TEST(Swpt2KernelTest, PartitionWindowChargeErrorAccumulates) {
  // Three folded orbitals rounding the same way: the charge error is three
  // times the largest single deviation, which worst_deviation alone cannot
  // show.
  const std::vector<std::size_t> window = {0, 1, 2, 3};
  const std::unordered_set<std::size_t> kept = {3};
  const auto split = sw::partition_window({1.9, 1.9, 1.9, 1.3}, window, kept,
                                          /*window_electrons=*/7, 0.5);
  EXPECT_EQ(split.inactive_spatial, (std::vector<int>{0, 1, 2}));
  EXPECT_TRUE(split.virtual_spatial.empty());
  EXPECT_NEAR(split.worst_deviation, 0.1, 1e-12);
  EXPECT_NEAR(split.folded_charge_error, 0.3, 1e-12);
  // The active electron count is what the fold leaves behind, not the
  // reference occupation summed over the kept space (1.3 here).
  EXPECT_EQ(split.active_electrons, 1);
}

TEST(Swpt2KernelTest, PartitionWindowRejectsAmbiguousFolds) {
  const std::vector<std::size_t> window = {0, 1, 2, 3};
  const std::unordered_set<std::size_t> kept = {1, 2};

  // A singly occupied orbital cannot be folded on an arbitrary rounding.
  EXPECT_THROW(sw::partition_window({1.0, 1.0, 1.0, 1.0}, window, kept, 4, 0.5),
               std::invalid_argument);
  // A zero tolerance admits only exactly integer folded occupations.
  EXPECT_NO_THROW(
      sw::partition_window({2.0, 1.0, 1.0, 0.0}, window, kept, 4, 0.0));
  EXPECT_THROW(
      sw::partition_window({1.98, 1.0, 1.0, 0.02}, window, kept, 4, 0.0),
      std::invalid_argument);
  // The tolerance itself must lie in [0, 1).
  EXPECT_THROW(sw::partition_window({2.0, 1.0, 1.0, 0.0}, window, kept, 4, 1.0),
               std::invalid_argument);
  EXPECT_THROW(
      sw::partition_window({2.0, 1.0, 1.0, 0.0}, window, kept, 4, -0.1),
      std::invalid_argument);
  // An empty kept space, and inputs of inconsistent length.
  EXPECT_THROW(sw::partition_window({2.0, 0.0}, {0, 1}, {}, 2, 0.5),
               std::invalid_argument);
  EXPECT_THROW(sw::partition_window({2.0, 0.0}, {0, 1, 2}, {0}, 2, 0.5),
               std::invalid_argument);
  // Rounding that leaves an active electron count the kept space cannot hold.
  EXPECT_THROW(sw::partition_window({2.0, 0.0}, {0, 1}, {1}, 5, 0.5),
               std::invalid_argument);
  EXPECT_THROW(sw::partition_window({2.0, 0.0}, {0, 1}, {1}, 1, 0.5),
               std::invalid_argument);
}

TEST(Swpt2KernelTest, SemicanonicalPrimitivesAreCovariantAndReversible) {
  const int norb = 4;
  Eigen::MatrixXd h(norb, norb);
  h << -1.2, 0.08, -0.03, 0.02, 0.08, -0.4, 0.11, -0.01, -0.03, 0.11, 0.5, 0.07,
      0.02, -0.01, 0.07, 1.4;
  Eigen::VectorXd g(norb * norb * norb * norb);
  for (int i = 0; i < g.size(); ++i) g(i) = 0.02 * std::sin(0.37 * i + 0.2);
  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(norb, norb);
  density.diagonal() << 2.0, 1.4, 0.6, 0.0;
  density(1, 2) = density(2, 1) = 0.15;

  const double angle = 0.37;
  Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(norb, norb);
  basis(1, 1) = basis(2, 2) = std::cos(angle);
  basis(1, 2) = -std::sin(angle);
  basis(2, 1) = std::sin(angle);

  const auto fock = sw::generalized_fock_matrix(h, g, density, norb);
  const auto h_rot = sw::rotate_one_body(h, basis);
  const auto g_rot = sw::rotate_two_body(g, basis, norb);
  const auto d_rot = sw::rotate_one_body(density, basis);
  const auto fock_rot = sw::generalized_fock_matrix(h_rot, g_rot, d_rot, norb);
  EXPECT_LT((fock_rot - basis.transpose() * fock * basis).cwiseAbs().maxCoeff(),
            1e-11);

  const auto semi = sw::semicanonical_rotation(fock_rot, {{0}, {1, 2}, {3}},
                                               /*tolerance=*/1e-14);
  const auto diagonal = sw::rotate_one_body(fock_rot, semi);
  EXPECT_NEAR(diagonal(1, 2), 0.0, 1e-12);
  EXPECT_LT((semi.transpose() * semi - Eigen::MatrixXd::Identity(norb, norb))
                .cwiseAbs()
                .maxCoeff(),
            1e-12);

  const auto h_roundtrip =
      sw::rotate_one_body(sw::rotate_one_body(h, basis), basis.transpose());
  const auto g_roundtrip = sw::rotate_two_body(
      sw::rotate_two_body(g, basis, norb), basis.transpose(), norb);
  EXPECT_LT((h_roundtrip - h).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT((g_roundtrip - g).cwiseAbs().maxCoeff(), 1e-12);

  const auto no_op = sw::semicanonical_rotation(diagonal, {{0}, {1, 2}, {3}},
                                                /*tolerance=*/1e-10);
  EXPECT_EQ(no_op, Eigen::MatrixXd::Identity(norb, norb));
}

TEST(Swpt2KernelTest, SemicanonicalDownfoldIsBlockRotationInvariant) {
  const int norb = 6;
  const std::vector<int> inactive = {0, 1};
  const std::vector<int> active = {2, 3};
  const std::vector<int> virt = {4, 5};
  Eigen::MatrixXd h = Eigen::MatrixXd::Zero(norb, norb);
  h.diagonal() << -2.3, -1.8, -0.6, 0.35, 1.2, 2.1;
  h(0, 1) = h(1, 0) = 0.06;
  h(2, 3) = h(3, 2) = 0.09;
  h(4, 5) = h(5, 4) = -0.07;
  h(2, 4) = h(4, 2) = 0.05;
  h(3, 5) = h(5, 3) = -0.04;

  Eigen::VectorXd g = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  set_sym_eri(g, 0, 0, 0, 0, norb, 0.70);
  set_sym_eri(g, 1, 1, 1, 1, norb, 0.66);
  set_sym_eri(g, 2, 2, 2, 2, norb, 0.62);
  set_sym_eri(g, 3, 3, 3, 3, norb, 0.55);
  set_sym_eri(g, 4, 4, 4, 4, norb, 0.42);
  set_sym_eri(g, 5, 5, 5, 5, norb, 0.38);
  set_sym_eri(g, 0, 0, 1, 1, norb, 0.27);
  set_sym_eri(g, 0, 0, 2, 2, norb, 0.24);
  set_sym_eri(g, 1, 1, 3, 3, norb, 0.21);
  set_sym_eri(g, 2, 2, 4, 4, norb, 0.18);
  set_sym_eri(g, 3, 3, 5, 5, norb, 0.16);
  set_sym_eri(g, 2, 4, 3, 5, norb, 0.08);
  set_sym_eri(g, 2, 5, 2, 3, norb, -0.06);

  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(norb, norb);
  density(0, 0) = 2.0;
  density(1, 1) = 2.0;
  density(2, 2) = 1.35;
  density(3, 3) = 0.65;
  density(2, 3) = density(3, 2) = 0.17;

  struct Result {
    sw::ActiveHamiltonian hamiltonian;
    double min_denominator;
    double max_amplitude;
  };
  const auto run = [&](const Eigen::MatrixXd& h_in, const Eigen::VectorXd& g_in,
                       const Eigen::MatrixXd& density_in) {
    const auto fock = sw::generalized_fock_matrix(h_in, g_in, density_in, norb);
    const auto rotation = sw::semicanonical_rotation(
        fock, {inactive, active, virt}, /*tolerance=*/1e-14);
    const auto h_semi = sw::rotate_one_body(h_in, rotation);
    const auto g_semi = sw::rotate_two_body(g_in, rotation, norb);
    const auto density_semi = sw::rotate_one_body(density_in, rotation);
    const auto fock_semi = sw::rotate_one_body(fock, rotation);
    Eigen::VectorXd eps(2 * norb);
    for (int p = 0; p < norb; ++p)
      eps(2 * p) = eps(2 * p + 1) = fock_semi(p, p);
    const auto part = sw::make_partition(norb, active, inactive, virt);
    const auto blocked = sw::build_two_body_blocked(g_semi, norb);
    const auto one_body = sw::spin_orbital_one_body(h_semi, h_semi, norb);
    sw::RegularizerOptions reg;
    reg.denom_flow = 1.0;
    const auto down =
        sw::downfold_blocked(one_body, blocked, eps, part, reg, 0.3);
    auto effective = sw::to_spatial_chemist(down, part);
    Eigen::MatrixXd active_rotation(active.size(), active.size());
    for (int i = 0; i < static_cast<int>(active.size()); ++i)
      for (int j = 0; j < static_cast<int>(active.size()); ++j)
        active_rotation(i, j) = rotation(active[i], active[j]);
    effective.one_body =
        sw::rotate_one_body(effective.one_body, active_rotation.transpose());
    effective.two_body = sw::rotate_two_body(
        effective.two_body, active_rotation.transpose(), effective.norb);
    return Result{effective, down.min_denominator, down.max_amplitude};
  };

  const Result reference = run(h, g, density);
  Eigen::MatrixXd basis = Eigen::MatrixXd::Identity(norb, norb);
  const double inactive_angle = 0.19, active_angle = 0.31,
               virtual_angle = -0.27;
  basis(0, 0) = basis(1, 1) = std::cos(inactive_angle);
  basis(0, 1) = -std::sin(inactive_angle);
  basis(1, 0) = std::sin(inactive_angle);
  basis(2, 2) = basis(3, 3) = std::cos(active_angle);
  basis(2, 3) = -std::sin(active_angle);
  basis(3, 2) = std::sin(active_angle);
  basis(4, 4) = basis(5, 5) = std::cos(virtual_angle);
  basis(4, 5) = -std::sin(virtual_angle);
  basis(5, 4) = std::sin(virtual_angle);

  Result rotated =
      run(sw::rotate_one_body(h, basis), sw::rotate_two_body(g, basis, norb),
          sw::rotate_one_body(density, basis));
  Eigen::MatrixXd active_basis(2, 2);
  active_basis << basis(2, 2), basis(2, 3), basis(3, 2), basis(3, 3);
  rotated.hamiltonian.one_body = sw::rotate_one_body(
      rotated.hamiltonian.one_body, active_basis.transpose());
  rotated.hamiltonian.two_body = sw::rotate_two_body(
      rotated.hamiltonian.two_body, active_basis.transpose(), 2);

  EXPECT_NEAR(rotated.hamiltonian.core_energy,
              reference.hamiltonian.core_energy, 1e-10);
  EXPECT_LT((rotated.hamiltonian.one_body - reference.hamiltonian.one_body)
                .cwiseAbs()
                .maxCoeff(),
            1e-10);
  EXPECT_LT((rotated.hamiltonian.two_body - reference.hamiltonian.two_body)
                .cwiseAbs()
                .maxCoeff(),
            1e-10);
  EXPECT_NEAR(rotated.min_denominator, reference.min_denominator, 1e-10);
  EXPECT_NEAR(rotated.max_amplitude, reference.max_amplitude, 1e-10);
}

// ---------------------------------------------------------------------------
// Spin-blocked spatial storage (production path): the two independent spin
// blocks reconstruct the dense spin-orbital antisymmetric tensor exactly. The
// sweep visits every nonzero spin pattern; the check is purely algebraic so no
// physical symmetry of the input is required.
TEST(Swpt2KernelTest, SpinBlockedTwoBodyRoundTrip) {
  const int norb = 3;
  const int M = 2 * norb;
  const int n4 = norb * norb * norb * norb;
  Eigen::VectorXd g(n4);
  for (int i = 0; i < n4; ++i) g(i) = std::sin(1.0 * i + 0.1);

  const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(norb, norb);
  const auto so =
      testing::build_spin_orbital_tensors(zero, zero, g, g, g, 0.0, norb);
  const auto blk = sw::build_two_body_blocked(g, norb);

  double max_dev = 0.0;
  for (int P = 0; P < M; ++P)
    for (int Q = 0; Q < M; ++Q)
      for (int R = 0; R < M; ++R)
        for (int S = 0; S < M; ++S)
          max_dev = std::max(max_dev,
                             std::abs(so.two_body(sw::idx4(P, Q, R, S, M)) -
                                      sw::so_v_from_blocked(blk, P, Q, R, S)));
  EXPECT_LT(max_dev, 1e-12);
}

}  // namespace
