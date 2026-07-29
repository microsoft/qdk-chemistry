// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Shared test-only helpers for the Schrieffer-Wolff PT2 tests.
//
// A tiny, self-contained full-CI ground-state solver on the interleaved
// spin-orbital tensors produced by the swpt2 kernel. It is intentionally
// independent of any production solver (MACIS, PySCF, ...): it evaluates the
// second-quantized operator
//
//     e0 + sum_PQ f_PQ a^dag_P a_Q
//        + (1/4) sum_PQRS v_PQRS a^dag_P a^dag_Q a_R a_S
//
// directly from the antisymmetrized spin-orbital tensors via Jordan-Wigner
// bitmask determinants. Used both to derive first-principles reference values
// in the kernel tests and to cross-check the data-layer emission path against
// MACIS on identical integrals.

#ifndef QDK_CHEMISTRY_TESTS_SWPT2_TEST_SUPPORT_HPP
#define QDK_CHEMISTRY_TESTS_SWPT2_TEST_SUPPORT_HPP

#include <cstdint>
#include <map>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

namespace swpt2_test {

namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

// Set a chemist (pq|rs) integral together with its 8-fold permutational
// symmetry partners (real orbitals) so the resulting operator is Hermitian.
inline void set_sym_eri(Eigen::VectorXd& g, int p, int q, int r, int s,
                        int norb, double val) {
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
inline Ladder apply_ladder(std::uint64_t mask, int orb, bool creation) {
  const std::uint64_t bit = std::uint64_t{1} << orb;
  const bool occupied = (mask & bit) != 0;
  if (creation == occupied) return {0, 0, false};  // Pauli
  const int below = __builtin_popcountll(mask & (bit - 1));
  return {mask ^ bit, (below & 1) ? -1 : 1, true};
}
// a^dag_P a_Q |mask>
inline Ladder apply_one(std::uint64_t mask, int P, int Q) {
  const Ladder a = apply_ladder(mask, Q, false);
  if (!a.ok) return a;
  const Ladder b = apply_ladder(a.mask, P, true);
  if (!b.ok) return b;
  return {b.mask, a.sign * b.sign, true};
}
// a^dag_P a^dag_Q a_R a_S |mask>  (apply S, R, then Q^dag, P^dag)
inline Ladder apply_two(std::uint64_t mask, int P, int Q, int R, int S) {
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
inline double fci_ground_energy(double e0, const Eigen::MatrixXd& f,
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

}  // namespace swpt2_test

#endif  // QDK_CHEMISTRY_TESTS_SWPT2_TEST_SUPPORT_HPP
