// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

#include <algorithm>
#include <array>
#include <blas.hh>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::swpt2 {

namespace {
/// C-order flat index into an (n, n, n, n) spatial tensor.
inline std::size_t flat(int p, int q, int r, int s, int n) {
  return ((static_cast<std::size_t>(p) * n + q) * n + r) * n + s;
}
inline int alpha(int p) { return 2 * p; }
inline int beta(int p) { return 2 * p + 1; }
inline Eigen::Index n4(int M) {
  return static_cast<Eigen::Index>(M) * M * M * M;
}
}  // namespace

// ===========================================================================
// Shared foundations (used by BOTH the dense reference and the production
// path): denominator regularization, diagonal-Fock energies, the
// occupation-change masks, and the projected-commutator Wick machinery.
// ===========================================================================

double reg_inv(double delta, const RegOptions& reg) {
  if (reg.denom_flow > 0.0) {
    const double d2 = delta * delta;
    if (reg.denom_flow * d2 < 1e-14) return reg.denom_flow * delta;
    return (1.0 - std::exp(-reg.denom_flow * d2)) / delta;
  }
  if (reg.denom_shift > 0.0) {
    return delta / (delta * delta + reg.denom_shift * reg.denom_shift);
  }
  if (std::abs(delta) < reg.denom_floor) return 0.0;
  return 1.0 / delta;
}

// ===========================================================================
// Dense spin-orbital reference representation (validation only; not called by
// the constructor). It checks the blocked storage and access paths, but shares
// the projected-Wick implementation with production and is not an independent
// mathematical oracle. Some shared helpers
// (`diagonal_fock_energies`, the occupation-change masks, the Wick
// `project_product_t`) live in this region because the production path below
// reuses them.
// ===========================================================================

namespace reference {

SoTensors build_tensors(const Eigen::MatrixXd& h1a, const Eigen::MatrixXd& h1b,
                        const Eigen::VectorXd& g_aaaa,
                        const Eigen::VectorXd& g_aabb,
                        const Eigen::VectorXd& g_bbbb, double e_core,
                        int norb) {
  const int M = 2 * norb;
  SoTensors h;
  h.n_so = M;
  h.e0 = e_core;
  h.f = Eigen::MatrixXd::Zero(M, M);
  for (int p = 0; p < norb; ++p) {
    for (int q = 0; q < norb; ++q) {
      h.f(alpha(p), alpha(q)) = h1a(p, q);
      h.f(beta(p), beta(q)) = h1b(p, q);
    }
  }

  // W[P,Q,R,S] = coefficient of the ordered string a^dag_P a^dag_Q a_R a_S.
  std::vector<double> W(static_cast<std::size_t>(M) * M * M * M, 0.0);
  for (int p = 0; p < norb; ++p) {
    for (int q = 0; q < norb; ++q) {
      for (int r = 0; r < norb; ++r) {
        for (int s = 0; s < norb; ++s) {
          const std::size_t g = flat(p, q, r, s, norb);
          // aa: 0.5 g_aa[p,q,r,s] a^dag_pa a^dag_ra a_sa a_qa -> W[pa,ra,sa,qa]
          W[idx4(alpha(p), alpha(r), alpha(s), alpha(q), M)] += 0.5 * g_aaaa(g);
          W[idx4(beta(p), beta(r), beta(s), beta(q), M)] += 0.5 * g_bbbb(g);
          // ab: g_ab[p,q,r,s] a^dag_pa a^dag_rb a_sb a_qa -> W[pa,rb,sb,qa]
          W[idx4(alpha(p), beta(r), beta(s), alpha(q), M)] += g_aabb(g);
        }
      }
    }
  }

  // Antisymmetrize: v = W - W(swap PQ) - W(swap RS) + W(swap both).
  h.v = Eigen::VectorXd::Zero(n4(M));
  for (int P = 0; P < M; ++P) {
    for (int Q = 0; Q < M; ++Q) {
      for (int R = 0; R < M; ++R) {
        for (int S = 0; S < M; ++S) {
          h.v(idx4(P, Q, R, S, M)) =
              W[idx4(P, Q, R, S, M)] - W[idx4(Q, P, R, S, M)] -
              W[idx4(P, Q, S, R, M)] + W[idx4(Q, P, S, R, M)];
        }
      }
    }
  }
  return h;
}

}  // namespace reference

Eigen::VectorXd diagonal_fock_energies(const Eigen::MatrixXd& h1a,
                                       const Eigen::VectorXd& g_aaaa,
                                       const Eigen::VectorXd& na,
                                       const Eigen::VectorXd& nb, int norb) {
  // Spin-averaged (spin-free 1-RDM) occupations n^sigma = (na+nb)/2, so
  // eps_alpha == eps_beta and the diagonal-Fock H0 is spin-symmetric (preserves
  // S^2). High-spin states are realized in the active-space solve, not by
  // polarizing this reference.
  const Eigen::VectorXd ntot = na + nb;  // total (spin-free) occupation
  Eigen::VectorXd eps = Eigen::VectorXd::Zero(2 * norb);
  for (int p = 0; p < norb; ++p) {
    double coulomb = 0.0, exch = 0.0;
    for (int r = 0; r < norb; ++r) {
      coulomb += g_aaaa(flat(p, p, r, r, norb)) * ntot(r);
      exch += g_aaaa(flat(p, r, r, p, norb)) * 0.5 * ntot(r);  // same-spin n/2
    }
    const double e = h1a(p, p) + coulomb - exch;
    eps(alpha(p)) = e;
    eps(beta(p)) = e;
  }
  return eps;
}

Eigen::MatrixXd generalized_fock_matrix(const Eigen::MatrixXd& h1,
                                        const Eigen::VectorXd& two_body,
                                        const Eigen::MatrixXd& density,
                                        int norb) {
  if (h1.rows() != norb || h1.cols() != norb || density.rows() != norb ||
      density.cols() != norb || two_body.size() != n4(norb))
    throw std::invalid_argument(
        "generalized_fock_matrix: inconsistent tensor dimensions");

  Eigen::MatrixXd fock = h1;
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          fock(p, q) +=
              density(r, s) * (two_body(flat(p, q, r, s, norb)) -
                               0.5 * two_body(flat(p, r, s, q, norb)));
  return 0.5 * (fock + fock.transpose()).eval();
}

Eigen::MatrixXd semicanonical_rotation(
    const Eigen::MatrixXd& fock, const std::vector<std::vector<int>>& blocks,
    double tolerance) {
  if (fock.rows() != fock.cols() || !std::isfinite(tolerance) ||
      tolerance < 0.0)
    throw std::invalid_argument(
        "semicanonical_rotation: invalid Fock matrix or tolerance");

  const int norb = static_cast<int>(fock.rows());
  Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(norb, norb);
  std::vector<char> used(norb, 0);
  for (const auto& block : blocks) {
    for (int p : block) {
      if (p < 0 || p >= norb || used[p])
        throw std::invalid_argument(
            "semicanonical_rotation: invalid or repeated block index");
      used[p] = 1;
    }
    if (block.size() < 2) continue;

    Eigen::MatrixXd sub(block.size(), block.size());
    double max_offdiag = 0.0;
    for (int i = 0; i < static_cast<int>(block.size()); ++i)
      for (int j = 0; j < static_cast<int>(block.size()); ++j) {
        sub(i, j) = fock(block[i], block[j]);
        if (i != j) max_offdiag = std::max(max_offdiag, std::abs(sub(i, j)));
      }
    if (max_offdiag <= tolerance) continue;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        0.5 * (sub + sub.transpose()).eval());
    if (solver.info() != Eigen::Success)
      throw std::runtime_error(
          "semicanonical_rotation: Fock block diagonalization failed");
    for (int i = 0; i < static_cast<int>(block.size()); ++i)
      for (int j = 0; j < static_cast<int>(block.size()); ++j)
        rotation(block[i], block[j]) = solver.eigenvectors()(i, j);
  }
  return rotation;
}

Eigen::MatrixXd rotate_one_body(const Eigen::MatrixXd& one_body,
                                const Eigen::MatrixXd& rotation) {
  if (one_body.rows() != one_body.cols() ||
      rotation.rows() != one_body.rows() || rotation.cols() != one_body.cols())
    throw std::invalid_argument("rotate_one_body: inconsistent dimensions");
  return rotation.transpose() * one_body * rotation;
}

Eigen::VectorXd rotate_two_body(const Eigen::VectorXd& two_body,
                                const Eigen::MatrixXd& rotation, int norb) {
  if (rotation.rows() != norb || rotation.cols() != norb ||
      two_body.size() != n4(norb))
    throw std::invalid_argument("rotate_two_body: inconsistent dimensions");

  Eigen::VectorXd tmp1 = Eigen::VectorXd::Zero(n4(norb));
  Eigen::VectorXd tmp2 = Eigen::VectorXd::Zero(n4(norb));
  Eigen::VectorXd tmp3 = Eigen::VectorXd::Zero(n4(norb));
  Eigen::VectorXd out = Eigen::VectorXd::Zero(n4(norb));
  for (int p = 0; p < norb; ++p)
    for (int b = 0; b < norb; ++b)
      for (int c = 0; c < norb; ++c)
        for (int d = 0; d < norb; ++d)
          for (int a = 0; a < norb; ++a)
            tmp1(flat(p, b, c, d, norb)) +=
                rotation(a, p) * two_body(flat(a, b, c, d, norb));
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int c = 0; c < norb; ++c)
        for (int d = 0; d < norb; ++d)
          for (int b = 0; b < norb; ++b)
            tmp2(flat(p, q, c, d, norb)) +=
                rotation(b, q) * tmp1(flat(p, b, c, d, norb));
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int d = 0; d < norb; ++d)
          for (int c = 0; c < norb; ++c)
            tmp3(flat(p, q, r, d, norb)) +=
                rotation(c, r) * tmp2(flat(p, q, c, d, norb));
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          for (int d = 0; d < norb; ++d)
            out(flat(p, q, r, s, norb)) +=
                rotation(d, s) * tmp3(flat(p, q, r, d, norb));
  return out;
}

namespace {
// net occupation change of a role set for a one-body a^dag_P a_Q.
inline int net1(const std::vector<char>& role, int P, int Q) {
  return role[P] - role[Q];
}
// net occupation change for a two-body a^dag_P a^dag_Q a_R a_S.
inline int net2(const std::vector<char>& role, int P, int Q, int R, int S) {
  return role[P] + role[Q] - role[R] - role[S];
}
// a term is external-off-diagonal if it changes the inactive OR virtual count.
inline bool is_od1(const SoPartition& part, int P, int Q) {
  return net1(part.is_inactive, P, Q) != 0 || net1(part.is_virtual, P, Q) != 0;
}
inline bool is_od2(const SoPartition& part, int P, int Q, int R, int S) {
  return net2(part.is_inactive, P, Q, R, S) != 0 ||
         net2(part.is_virtual, P, Q, R, S) != 0;
}
}  // namespace

namespace reference {

Generator make_generator(const SoTensors& h, const Eigen::VectorXd& eps,
                         const SoPartition& part, const RegOptions& reg) {
  const int M = h.n_so;
  Generator gen;
  gen.s1 = Eigen::MatrixXd::Zero(M, M);
  gen.s2 = Eigen::VectorXd::Zero(n4(M));
  double min_denom = std::numeric_limits<double>::infinity();
  double max_amp = 0.0;
  // raw (unregularized) amplitude, floored only to avoid a literal divide-by-0
  auto track = [&](double coupling, double delta) {
    const double ad = std::abs(delta);
    min_denom = std::min(min_denom, ad);
    max_amp =
        std::max(max_amp, std::abs(coupling) / std::max(ad, reg.denom_floor));
  };
  for (int P = 0; P < M; ++P) {
    for (int Q = 0; Q < M; ++Q) {
      if (is_od1(part, P, Q) && h.f(P, Q) != 0.0) {
        const double d = eps(P) - eps(Q);
        gen.s1(P, Q) = h.f(P, Q) * reg_inv(d, reg);
        track(h.f(P, Q), d);
      }
    }
  }
  for (int P = 0; P < M; ++P) {
    for (int Q = 0; Q < M; ++Q) {
      for (int R = 0; R < M; ++R) {
        for (int S = 0; S < M; ++S) {
          if (!is_od2(part, P, Q, R, S)) continue;
          const double vv = h.v(idx4(P, Q, R, S, M));
          if (vv == 0.0) continue;
          const double d = eps(P) + eps(Q) - eps(R) - eps(S);
          gen.s2(idx4(P, Q, R, S, M)) = vv * reg_inv(d, reg);
          track(vv, d);
        }
      }
    }
  }
  gen.min_denominator = std::isinf(min_denom) ? 0.0 : min_denom;
  gen.max_amplitude = max_amp;
  return gen;
}

std::pair<SoTensors, SoTensors> split_bd_od(const SoTensors& h,
                                            const SoPartition& part) {
  const int M = h.n_so;
  SoTensors bd, od;
  bd.n_so = od.n_so = M;
  bd.e0 = h.e0;
  od.e0 = 0.0;
  bd.f = Eigen::MatrixXd::Zero(M, M);
  od.f = Eigen::MatrixXd::Zero(M, M);
  bd.v = Eigen::VectorXd::Zero(n4(M));
  od.v = Eigen::VectorXd::Zero(n4(M));
  for (int P = 0; P < M; ++P) {
    for (int Q = 0; Q < M; ++Q) {
      (is_od1(part, P, Q) ? od.f : bd.f)(P, Q) = h.f(P, Q);
    }
  }
  for (int P = 0; P < M; ++P)
    for (int Q = 0; Q < M; ++Q)
      for (int R = 0; R < M; ++R)
        for (int S = 0; S < M; ++S) {
          const std::size_t i = idx4(P, Q, R, S, M);
          (is_od2(part, P, Q, R, S) ? od.v : bd.v)(i) = h.v(i);
        }
  return {bd, od};
}

MeanFieldResult mean_field_fold(const SoTensors& h_bd,
                                const SoPartition& part) {
  const int M = h_bd.n_so;
  // external reference occupation: inactive filled (1), virtual empty (0)
  Eigen::VectorXd nb = Eigen::VectorXd::Zero(M);
  for (int P = 0; P < M; ++P)
    if (part.is_inactive[P]) nb(P) = 1.0;

  MeanFieldResult res;
  res.e = h_bd.e0;
  for (int P = 0; P < M; ++P) res.e += h_bd.f(P, P) * nb(P);
  for (int P = 0; P < M; ++P)
    for (int Q = 0; Q < M; ++Q)
      res.e -= 0.5 * h_bd.v(idx4(P, Q, P, Q, M)) * nb(P) * nb(Q);

  res.f_active = Eigen::MatrixXd::Zero(M, M);
  for (int i = 0; i < M; ++i) {
    if (!part.is_active[i]) continue;
    for (int j = 0; j < M; ++j) {
      if (!part.is_active[j]) continue;
      double fold = 0.0;
      for (int b = 0; b < M; ++b) fold += h_bd.v(idx4(i, b, b, j, M)) * nb(b);
      res.f_active(i, j) = h_bd.f(i, j) + fold;
    }
  }

  res.v_active = Eigen::VectorXd::Zero(n4(M));
  for (int P = 0; P < M; ++P) {
    if (!part.is_active[P]) continue;
    for (int Q = 0; Q < M; ++Q) {
      if (!part.is_active[Q]) continue;
      for (int R = 0; R < M; ++R) {
        if (!part.is_active[R]) continue;
        for (int S = 0; S < M; ++S) {
          if (!part.is_active[S]) continue;
          const std::size_t i = idx4(P, Q, R, S, M);
          res.v_active(i) = h_bd.v(i);
        }
      }
    }
  }
  return res;
}

}  // namespace reference

// ---------------------------------------------------------------------------
// Projected commutator 1/2 [S, V] via generalized Wick over the external legs.
// ---------------------------------------------------------------------------
namespace {

// A number-conserving operator over the (small) active space: a normal-ordered
// term (creations ascending, then annihilations ascending) -> coefficient.
using Term = std::vector<std::pair<int, int>>;  // (spin-orbital, 1=cre / 0=ann)
using FermionOp = std::map<Term, double>;

struct RetainedOperator {
  using Operator = std::pair<int, int>;

  explicit RetainedOperator(const std::vector<int>& active_so)
      : so2c(active_so.empty() ? 0 : active_so.back() + 1, -1),
        one_body(Eigen::MatrixXd::Zero(active_so.size(), active_so.size())),
        two_body(Eigen::VectorXd::Zero(n4(active_so.size() / 2))) {
    for (int compact = 0; compact < static_cast<int>(active_so.size());
         ++compact)
      so2c[active_so[compact]] = compact;
  }

  void add(const Term& term, double coeff) {
    add(term.data(), static_cast<int>(term.size()), coeff);
  }

  void add(const Operator* term, int size, double coeff) {
    if (size == 0) {
      scalar += coeff;
    } else if (size == 2) {
      one_body(compact(term[0].first), compact(term[1].first)) += coeff;
    } else if (size == 4) {
      const int c0 = compact(term[0].first), c1 = compact(term[1].first);
      const int a0 = compact(term[2].first), a1 = compact(term[3].first);
      add_abab(c0, c1, a0, a1, +coeff);
      add_abab(c1, c0, a0, a1, -coeff);
      add_abab(c0, c1, a1, a0, -coeff);
      add_abab(c1, c0, a1, a0, +coeff);
    }
  }

  double scalar = 0.0;
  std::vector<int> so2c;
  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;

 private:
  int compact(int orbital) const { return so2c[orbital]; }

  void add_abab(int i, int j, int k, int l, double coeff) {
    if ((i & 1) == 0 && (j & 1) == 1 && (k & 1) == 0 && (l & 1) == 1) {
      const int nactive = one_body.rows() / 2;
      two_body(idx4(i >> 1, j >> 1, k >> 1, l >> 1, nactive)) += coeff;
    }
  }
};

void add_term(FermionOp& out, const Term& term, double coeff) {
  out[term] += coeff;
}

void add_term(RetainedOperator& out, const Term& term, double coeff) {
  out.add(term, coeff);
}

// Normal-order a raw operator string (anticommutation), accumulating into
// `out`.
template <class Output>
void normalize(Term ops, double coeff, Output& out) {
  const int n = static_cast<int>(ops.size());
  for (int i = 0; i + 1 < n; ++i) {
    const int ai = ops[i].first, aa = ops[i].second;
    const int bi = ops[i + 1].first, ba = ops[i + 1].second;
    const bool wrong = (aa == 0 && ba == 1) || (aa == ba && ai > bi);
    if (wrong) {
      Term swapped = ops;
      std::swap(swapped[i], swapped[i + 1]);
      normalize(swapped, -coeff, out);  // a b = -b a + {a, b}
      if (aa != ba && ai == bi) {       // contraction {a, b} = 1
        Term c;
        c.reserve(n - 2);
        for (int k = 0; k < n; ++k)
          if (k != i && k != i + 1) c.push_back(ops[k]);
        normalize(c, coeff, out);
      }
      return;
    }
    if (aa == ba && ai == bi) return;  // repeated operator -> 0
  }
  if (n <= 4) add_term(out, ops, coeff);
}

void normalize_retained(std::array<RetainedOperator::Operator, 8> ops, int n,
                        double coeff, RetainedOperator& out) {
  for (int i = 0; i + 1 < n; ++i) {
    const int ai = ops[i].first, aa = ops[i].second;
    const int bi = ops[i + 1].first, ba = ops[i + 1].second;
    const bool wrong = (aa == 0 && ba == 1) || (aa == ba && ai > bi);
    if (wrong) {
      auto swapped = ops;
      std::swap(swapped[i], swapped[i + 1]);
      normalize_retained(swapped, n, -coeff, out);
      if (aa != ba && ai == bi) {
        auto contracted = ops;
        std::move(contracted.begin() + i + 2, contracted.begin() + n,
                  contracted.begin() + i);
        normalize_retained(contracted, n - 2, coeff, out);
      }
      return;
    }
    if (aa == ba && ai == bi) return;
  }
  if (n <= 4) out.add(ops.data(), n, coeff);
}

void normalize(const Term& ops, double coeff, RetainedOperator& out) {
  std::array<RetainedOperator::Operator, 8> fixed_ops{};
  std::copy(ops.begin(), ops.end(), fixed_ops.begin());
  normalize_retained(fixed_ops, static_cast<int>(ops.size()), coeff, out);
}

template <std::size_t N, class Output>
void normalize_slots(const std::array<int, N>& values,
                     const std::array<int, N>& is_cre,
                     const std::vector<int>& slots, double coeff, Output& out) {
  Term ops;
  ops.reserve(slots.size());
  for (int slot : slots) ops.push_back({values[slot], is_cre[slot]});
  normalize(std::move(ops), coeff, out);
}

template <std::size_t N>
void normalize_slots(const std::array<int, N>& values,
                     const std::array<int, N>& is_cre,
                     const std::vector<int>& slots, double coeff,
                     RetainedOperator& out) {
  std::array<RetainedOperator::Operator, 8> ops{};
  int size = 0;
  for (int slot : slots) ops[size++] = {values[slot], is_cre[slot]};
  normalize_retained(ops, size, coeff, out);
}

// All disjoint-pair matchings (i < j) of `slots`, including the empty matching.
void enum_matchings(const std::vector<int>& slots,
                    std::vector<std::pair<int, int>> cur,
                    std::vector<std::vector<std::pair<int, int>>>& res) {
  if (slots.empty()) {
    res.push_back(cur);
    return;
  }
  std::vector<int> rest(slots.begin() + 1, slots.end());
  enum_matchings(rest, cur, res);  // first slot unpaired
  for (std::size_t k = 0; k < rest.size(); ++k) {
    std::vector<int> rem;
    for (std::size_t t = 0; t < rest.size(); ++t)
      if (t != k) rem.push_back(rest[t]);
    auto c2 = cur;
    c2.push_back({slots[0], rest[k]});
    enum_matchings(rem, c2, res);
  }
}

int perm_sign(const std::vector<int>& seq) {
  int s = 1;
  for (std::size_t i = 0; i < seq.size(); ++i)
    for (std::size_t j = i + 1; j < seq.size(); ++j)
      if (seq[i] > seq[j]) s = -s;
  return s;
}

bool is_two_body_between_matching(
    int rA, int rB, const std::vector<std::pair<int, int>>& match) {
  if (rA != 2 || rB != 2 || match.size() != 2) return false;
  return std::all_of(match.begin(), match.end(), [](const auto& pair) {
    return (pair.first < 4) != (pair.second < 4);
  });
}

bool is_one_line_between_matching(
    int rA, int rB, const std::vector<std::pair<int, int>>& match) {
  if (match.size() != 1) return false;
  const int boundary = 2 * rA;
  return (match[0].first < boundary) != (match[0].second < boundary);
}

Eigen::Index integer_power(Eigen::Index base, int exponent) {
  Eigen::Index result = 1;
  for (int i = 0; i < exponent; ++i) result *= base;
  return result;
}

void assign_active_slots(std::array<int, 8>& slots,
                         const std::vector<int>& external_slots,
                         Eigen::Index row,
                         const std::vector<int>& active_list) {
  const Eigen::Index nactive = active_list.size();
  for (auto slot = external_slots.rbegin(); slot != external_slots.rend();
       ++slot) {
    slots[*slot] = active_list[row % nactive];
    row /= nactive;
  }
}

// Every one-cross-line product is a matrix multiplication after flattening
// the active legs remaining on each operator into rows and the buffer line
// into the shared column. This covers retained and discarded-body channels.
template <class A2Get, class B2Get, class Output>
void project_one_line_between_blas(const Eigen::MatrixXd& A1, A2Get A2get,
                                   const Eigen::MatrixXd& B1, B2Get B2get,
                                   const SoPartition& part, double scale,
                                   Output& out) {
  std::vector<int> active_list, inactive_list, virtual_list;
  for (int orbital = 0; orbital < part.n_so; ++orbital) {
    if (part.is_active[orbital]) active_list.push_back(orbital);
    if (part.is_inactive[orbital]) inactive_list.push_back(orbital);
    if (part.is_virtual[orbital]) virtual_list.push_back(orbital);
  }
  if (active_list.empty()) return;

  for (const auto [rA, rB] :
       {std::pair{1, 1}, std::pair{1, 2}, std::pair{2, 1}, std::pair{2, 2}}) {
    const int boundary = 2 * rA;
    const int nslots = boundary + 2 * rB;
    std::array<int, 8> is_cre{};
    for (int slot = 0; slot < rA; ++slot) is_cre[slot] = 1;
    for (int slot = 0; slot < rB; ++slot) is_cre[boundary + slot] = 1;

    std::vector<int> slots(nslots);
    std::iota(slots.begin(), slots.end(), 0);
    std::vector<std::vector<std::pair<int, int>>> matches;
    enum_matchings(slots, {}, matches);
    for (const auto& match : matches) {
      if (!is_one_line_between_matching(rA, rB, match)) continue;
      const auto [left, right] = match[0];
      if (is_cre[left] == is_cre[right]) continue;

      std::vector<int> ext, ext_a, ext_b;
      for (int slot = 0; slot < nslots; ++slot) {
        if (slot == left || slot == right) continue;
        ext.push_back(slot);
        (slot < boundary ? ext_a : ext_b).push_back(slot);
      }
      std::vector<int> order{left, right};
      order.insert(order.end(), ext.begin(), ext.end());
      const double normA = rA == 2 ? 0.25 : 1.0;
      const double normB = rB == 2 ? 0.25 : 1.0;
      const double prefactor = scale * perm_sign(order) * normA * normB;
      const auto& buffer_list = is_cre[left] ? inactive_list : virtual_list;
      if (buffer_list.empty()) continue;

      const Eigen::Index nactive = active_list.size();
      const Eigen::Index nrow_a = integer_power(nactive, ext_a.size());
      const Eigen::Index nrow_b = integer_power(nactive, ext_b.size());
      const Eigen::Index nbuffer = buffer_list.size();
      Eigen::MatrixXd a(nrow_a, nbuffer);
      Eigen::MatrixXd b(nrow_b, nbuffer);
      for (Eigen::Index row = 0; row < nrow_a; ++row) {
        for (Eigen::Index q = 0; q < nbuffer; ++q) {
          std::array<int, 8> values{};
          assign_active_slots(values, ext_a, row, active_list);
          values[left] = values[right] = buffer_list[q];
          a(row, q) = rA == 1
                          ? A1(values[0], values[1])
                          : A2get(values[0], values[1], values[2], values[3]);
        }
      }
      for (Eigen::Index row = 0; row < nrow_b; ++row) {
        for (Eigen::Index q = 0; q < nbuffer; ++q) {
          std::array<int, 8> values{};
          assign_active_slots(values, ext_b, row, active_list);
          values[left] = values[right] = buffer_list[q];
          b(row, q) = rB == 1
                          ? B1(values[boundary], values[boundary + 1])
                          : B2get(values[boundary], values[boundary + 1],
                                  values[boundary + 2], values[boundary + 3]);
        }
      }

      if (rA == 2 && rB == 2) {
        // The raw one-line S2 * V2 product has three active operators from
        // each operand. Its rank-three part is discarded. A retained term can
        // arise only when an annihilator from A equals a creator from B, so
        // enumerate that union instead of all nactive^6 row pairs.
        std::vector<int> a_annihilators, b_creators;
        for (int slot : ext_a)
          if (!is_cre[slot]) a_annihilators.push_back(slot);
        for (int slot : ext_b)
          if (is_cre[slot]) b_creators.push_back(slot);

        std::vector<int> active_position(part.n_so, -1);
        for (int index = 0; index < static_cast<int>(active_list.size());
             ++index)
          active_position[active_list[index]] = index;
        std::vector<char> seen(nrow_b, 0);
        std::vector<Eigen::Index> candidates;
        candidates.reserve(a_annihilators.size() * b_creators.size() *
                           integer_power(nactive, ext_b.size() - 1));

        for (Eigen::Index row_a = 0; row_a < nrow_a; ++row_a) {
          std::array<int, 8> values_a{};
          assign_active_slots(values_a, ext_a, row_a, active_list);
          candidates.clear();
          for (int a_slot : a_annihilators) {
            for (int b_slot : b_creators) {
              const Eigen::Index free_count =
                  integer_power(nactive, ext_b.size() - 1);
              for (Eigen::Index free = 0; free < free_count; ++free) {
                std::array<int, 8> values_b{};
                values_b[b_slot] = values_a[a_slot];
                Eigen::Index remaining = free;
                for (auto slot = ext_b.rbegin(); slot != ext_b.rend(); ++slot) {
                  if (*slot == b_slot) continue;
                  values_b[*slot] = active_list[remaining % nactive];
                  remaining /= nactive;
                }
                Eigen::Index row_b = 0;
                for (int slot : ext_b)
                  row_b = row_b * nactive + active_position[values_b[slot]];
                if (!seen[row_b]) {
                  seen[row_b] = 1;
                  candidates.push_back(row_b);
                }
              }
            }
          }

          for (Eigen::Index row_b : candidates) {
            double product = 0.0;
            for (Eigen::Index q = 0; q < nbuffer; ++q)
              product += a(row_a, q) * b(row_b, q);
            const double coeff = prefactor * product;
            if (coeff == 0.0) continue;
            std::array<int, 8> values = values_a;
            assign_active_slots(values, ext_b, row_b, active_list);
            normalize_slots(values, is_cre, ext, coeff, out);
          }
          for (Eigen::Index row_b : candidates) seen[row_b] = 0;
        }
      } else {
        Eigen::MatrixXd product = Eigen::MatrixXd::Zero(nrow_a, nrow_b);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                   nrow_a, nrow_b, nbuffer, 1.0, a.data(), nrow_a, b.data(),
                   nrow_b, 0.0, product.data(), nrow_a);
        for (Eigen::Index row_a = 0; row_a < nrow_a; ++row_a) {
          for (Eigen::Index row_b = 0; row_b < nrow_b; ++row_b) {
            const double coeff = prefactor * product(row_a, row_b);
            if (coeff == 0.0) continue;
            std::array<int, 8> values{};
            assign_active_slots(values, ext_a, row_a, active_list);
            assign_active_slots(values, ext_b, row_b, active_list);
            normalize_slots(values, is_cre, ext, coeff, out);
          }
        }
      }
    }
  }
}

// Two cross-operator contractions in A2 * B2 leave two active legs on each
// tensor. Flattening each active pair into a row and the two buffer indices
// into a column turns every Wick matching into one matrix multiplication.
template <class A2Get, class B2Get, class Output>
void project_two_body_between_blas(A2Get A2get, B2Get B2get,
                                   const SoPartition& part, double scale,
                                   const Eigen::VectorXd& hvec,
                                   const Eigen::VectorXd& pvec, Output& out) {
  std::vector<int> active_list, inactive_list, virtual_list;
  for (int orbital = 0; orbital < part.n_so; ++orbital) {
    if (part.is_active[orbital]) active_list.push_back(orbital);
    if (part.is_inactive[orbital]) inactive_list.push_back(orbital);
    if (part.is_virtual[orbital]) virtual_list.push_back(orbital);
  }
  if (active_list.empty()) return;

  constexpr int nslots = 8;
  const std::array<int, nslots> is_cre{1, 1, 0, 0, 1, 1, 0, 0};
  std::vector<int> slots(nslots);
  std::iota(slots.begin(), slots.end(), 0);
  std::vector<std::vector<std::pair<int, int>>> matches;
  enum_matchings(slots, {}, matches);

  for (const auto& match : matches) {
    if (!is_two_body_between_matching(2, 2, match)) continue;
    if (std::any_of(match.begin(), match.end(), [&](const auto& pair) {
          return is_cre[pair.first] == is_cre[pair.second];
        }))
      continue;

    std::array<char, nslots> contracted{};
    for (const auto& pair : match) {
      contracted[pair.first] = 1;
      contracted[pair.second] = 1;
    }
    std::vector<int> ext, ext_a, ext_b;
    for (int slot = 0; slot < nslots; ++slot) {
      if (contracted[slot]) continue;
      ext.push_back(slot);
      (slot < 4 ? ext_a : ext_b).push_back(slot);
    }

    std::vector<int> order;
    for (const auto& pair : match) {
      order.push_back(pair.first);
      order.push_back(pair.second);
    }
    order.insert(order.end(), ext.begin(), ext.end());
    const double prefactor = scale * perm_sign(order) / 16.0;

    std::array<const std::vector<int>*, 2> buffer_lists{};
    bool empty_buffer = false;
    for (int pair = 0; pair < 2; ++pair) {
      buffer_lists[pair] =
          is_cre[match[pair].first] ? &inactive_list : &virtual_list;
      empty_buffer = empty_buffer || buffer_lists[pair]->empty();
    }
    if (empty_buffer) continue;

    const Eigen::Index nactive = active_list.size();
    const Eigen::Index nrow = nactive * nactive;
    const Eigen::Index nbuffer0 = buffer_lists[0]->size();
    const Eigen::Index nbuffer1 = buffer_lists[1]->size();
    const Eigen::Index ncol = nbuffer0 * nbuffer1;
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(nrow, ncol);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(nrow, ncol);

    for (Eigen::Index i = 0; i < nactive; ++i) {
      for (Eigen::Index j = 0; j < nactive; ++j) {
        const Eigen::Index row = i * nactive + j;
        for (Eigen::Index q0 = 0; q0 < nbuffer0; ++q0) {
          for (Eigen::Index q1 = 0; q1 < nbuffer1; ++q1) {
            const Eigen::Index col = q0 * nbuffer1 + q1;
            std::array<int, nslots> sv{};
            sv[ext_a[0]] = active_list[i];
            sv[ext_a[1]] = active_list[j];
            sv[match[0].first] = sv[match[0].second] = (*buffer_lists[0])[q0];
            sv[match[1].first] = sv[match[1].second] = (*buffer_lists[1])[q1];
            double prop = is_cre[match[0].first] ? pvec(sv[match[0].first])
                                                 : hvec(sv[match[0].first]);
            prop *= is_cre[match[1].first] ? pvec(sv[match[1].first])
                                           : hvec(sv[match[1].first]);
            a(row, col) = A2get(sv[0], sv[1], sv[2], sv[3]) * prop;

            sv[ext_b[0]] = active_list[i];
            sv[ext_b[1]] = active_list[j];
            b(row, col) = B2get(sv[4], sv[5], sv[6], sv[7]);
          }
        }
      }
    }

    Eigen::MatrixXd product = Eigen::MatrixXd::Zero(nrow, nrow);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, nrow,
               nrow, ncol, 1.0, a.data(), nrow, b.data(), nrow, 0.0,
               product.data(), nrow);

    for (Eigen::Index row_a = 0; row_a < nrow; ++row_a) {
      for (Eigen::Index row_b = 0; row_b < nrow; ++row_b) {
        const double coeff = prefactor * product(row_a, row_b);
        if (coeff == 0.0) continue;
        std::array<int, nslots> sv{};
        sv[ext_a[0]] = active_list[row_a / nactive];
        sv[ext_a[1]] = active_list[row_a % nactive];
        sv[ext_b[0]] = active_list[row_b / nactive];
        sv[ext_b[1]] = active_list[row_b % nactive];
        Term ops;
        ops.reserve(ext.size());
        for (int slot : ext) ops.push_back({sv[slot], is_cre[slot]});
        normalize(ops, coeff, out);
      }
    }
  }
}

// All remaining nonempty matchings factor into two operand-local panels.
// External active legs form the rows, cross-operator buffer lines form the
// shared column, and contractions internal to one operand are reduced while
// packing that operand's panel. Matchings owned by the specialized one- and
// two-line kernels above are excluded.
template <class A2Get, class B2Get, class Output>
void project_remaining_blas(const Eigen::MatrixXd& A1, A2Get A2get,
                            const Eigen::MatrixXd& B1, B2Get B2get,
                            const SoPartition& part, double scale,
                            Output& out) {
  std::vector<int> active_list, inactive_list, virtual_list;
  for (int orbital = 0; orbital < part.n_so; ++orbital) {
    if (part.is_active[orbital]) active_list.push_back(orbital);
    if (part.is_inactive[orbital]) inactive_list.push_back(orbital);
    if (part.is_virtual[orbital]) virtual_list.push_back(orbital);
  }
  if (active_list.empty()) return;

  for (const auto [rA, rB] :
       {std::pair{1, 1}, std::pair{1, 2}, std::pair{2, 1}, std::pair{2, 2}}) {
    const int boundary = 2 * rA;
    const int nslots = boundary + 2 * rB;
    std::array<int, 8> is_cre{};
    for (int slot = 0; slot < rA; ++slot) is_cre[slot] = 1;
    for (int slot = 0; slot < rB; ++slot) is_cre[boundary + slot] = 1;

    std::vector<int> slots(nslots);
    std::iota(slots.begin(), slots.end(), 0);
    std::vector<std::vector<std::pair<int, int>>> matches;
    enum_matchings(slots, {}, matches);
    for (const auto& match : matches) {
      if (match.empty() || is_one_line_between_matching(rA, rB, match) ||
          is_two_body_between_matching(rA, rB, match))
        continue;
      if (std::any_of(match.begin(), match.end(), [&](const auto& pair) {
            return is_cre[pair.first] == is_cre[pair.second];
          }))
        continue;

      std::vector<int> cross_pairs, internal_a_pairs, internal_b_pairs;
      std::array<char, 8> contracted{};
      for (int pair = 0; pair < static_cast<int>(match.size()); ++pair) {
        const auto [left, right] = match[pair];
        contracted[left] = contracted[right] = 1;
        if ((left < boundary) != (right < boundary))
          cross_pairs.push_back(pair);
        else if (left < boundary)
          internal_a_pairs.push_back(pair);
        else
          internal_b_pairs.push_back(pair);
      }

      std::vector<int> ext, ext_a, ext_b;
      for (int slot = 0; slot < nslots; ++slot) {
        if (contracted[slot]) continue;
        ext.push_back(slot);
        (slot < boundary ? ext_a : ext_b).push_back(slot);
      }
      std::vector<int> order;
      for (const auto& pair : match) {
        order.push_back(pair.first);
        order.push_back(pair.second);
      }
      order.insert(order.end(), ext.begin(), ext.end());
      const double normA = rA == 2 ? 0.25 : 1.0;
      const double normB = rB == 2 ? 0.25 : 1.0;
      const double prefactor = scale * perm_sign(order) * normA * normB;

      std::vector<const std::vector<int>*> pair_lists(match.size());
      bool empty_buffer = false;
      for (int pair = 0; pair < static_cast<int>(match.size()); ++pair) {
        pair_lists[pair] =
            is_cre[match[pair].first] ? &inactive_list : &virtual_list;
        empty_buffer = empty_buffer || pair_lists[pair]->empty();
      }
      if (empty_buffer) continue;

      const auto combination_count = [&](const std::vector<int>& pairs) {
        Eigen::Index count = 1;
        for (int pair : pairs) count *= pair_lists[pair]->size();
        return count;
      };
      const Eigen::Index nactive = active_list.size();
      const Eigen::Index nrow_a = integer_power(nactive, ext_a.size());
      const Eigen::Index nrow_b = integer_power(nactive, ext_b.size());
      const Eigen::Index ncol = combination_count(cross_pairs);
      Eigen::MatrixXd a(nrow_a, ncol);
      Eigen::MatrixXd b(nrow_b, ncol);

      const auto assign_pairs = [&](std::array<int, 8>& values,
                                    const std::vector<int>& pairs,
                                    Eigen::Index combination) {
        for (auto it = pairs.rbegin(); it != pairs.rend(); ++it) {
          const int pair = *it;
          const auto& candidates = *pair_lists[pair];
          const int value = candidates[combination % candidates.size()];
          combination /= candidates.size();
          values[match[pair].first] = values[match[pair].second] = value;
        }
      };
      const auto panel_value = [&](bool left_operand,
                                   std::array<int, 8> values) {
        const auto& internal_pairs =
            left_operand ? internal_a_pairs : internal_b_pairs;
        const Eigen::Index count = combination_count(internal_pairs);
        double sum = 0.0;
        for (Eigen::Index combination = 0; combination < count; ++combination) {
          assign_pairs(values, internal_pairs, combination);
          if (left_operand) {
            const double value =
                rA == 1 ? A1(values[0], values[1])
                        : A2get(values[0], values[1], values[2], values[3]);
            sum += value;
          } else {
            const double value =
                rB == 1 ? B1(values[boundary], values[boundary + 1])
                        : B2get(values[boundary], values[boundary + 1],
                                values[boundary + 2], values[boundary + 3]);
            sum += value;
          }
        }
        return sum;
      };

      for (Eigen::Index col = 0; col < ncol; ++col) {
        std::array<int, 8> cross_values{};
        assign_pairs(cross_values, cross_pairs, col);
        for (Eigen::Index row = 0; row < nrow_a; ++row) {
          auto values = cross_values;
          assign_active_slots(values, ext_a, row, active_list);
          a(row, col) = panel_value(true, values);
        }
        for (Eigen::Index row = 0; row < nrow_b; ++row) {
          auto values = cross_values;
          assign_active_slots(values, ext_b, row, active_list);
          b(row, col) = panel_value(false, values);
        }
      }

      Eigen::MatrixXd product = Eigen::MatrixXd::Zero(nrow_a, nrow_b);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                 nrow_a, nrow_b, ncol, 1.0, a.data(), nrow_a, b.data(), nrow_b,
                 0.0, product.data(), nrow_a);
      for (Eigen::Index row_a = 0; row_a < nrow_a; ++row_a) {
        for (Eigen::Index row_b = 0; row_b < nrow_b; ++row_b) {
          const double coeff = prefactor * product(row_a, row_b);
          if (coeff == 0.0) continue;
          std::array<int, 8> values{};
          assign_active_slots(values, ext_a, row_a, active_list);
          assign_active_slots(values, ext_b, row_b, active_list);
          Term ops;
          ops.reserve(ext.size());
          for (int slot : ext) ops.push_back({values[slot], is_cre[slot]});
          normalize(ops, coeff, out);
        }
      }
    }
  }
}

// Scalar active-space reference projection of A * B (A left, B right), buffer
// summed. This is used only by the dense regression implementation.
// A/B given as (1-body dense, 2-body element accessor) operators; the 2-body
// accessors `A2get`/`B2get` are `double(int P,int Q,int R,int S)` so the caller
// can back them by a dense tensor or compute elements on the fly (spin-blocked
// path). Contributions are emitted (scaled) into `out` over active
// spin-orbitals; hvec/pvec are the hole/particle propagators.
template <class A2Get, class B2Get>
void project_product_t(const Eigen::MatrixXd& A1, A2Get A2get,
                       const Eigen::MatrixXd& B1, B2Get B2get,
                       const SoPartition& part, double scale,
                       const Eigen::VectorXd& hvec, const Eigen::VectorXd& pvec,
                       FermionOp& out) {
  const int M = part.n_so;
  // Candidate orbital sets: contracted buffer legs sum only over their support
  // (inactive = particle line, virtual = hole line); output legs are active.
  std::vector<int> active_list, inactive_list, virtual_list;
  for (int o = 0; o < M; ++o) {
    if (part.is_active[o]) active_list.push_back(o);
    if (part.is_inactive[o]) inactive_list.push_back(o);
    if (part.is_virtual[o]) virtual_list.push_back(o);
  }
  for (int rA = 1; rA <= 2; ++rA) {
    const double normA = (rA == 1) ? 1.0 : 0.25;
    for (int rB = 1; rB <= 2; ++rB) {
      const double normB = (rB == 1) ? 1.0 : 0.25;
      const int nslots = 2 * rA + 2 * rB;
      std::vector<int> is_cre(nslots, 0);
      for (int k = 0; k < rA; ++k) is_cre[k] = 1;
      for (int k = 0; k < rB; ++k) is_cre[2 * rA + k] = 1;

      std::vector<int> slots(nslots);
      std::iota(slots.begin(), slots.end(), 0);
      std::vector<std::vector<std::pair<int, int>>> matches;
      enum_matchings(slots, {}, matches);

      for (const auto& match : matches) {
        bool same_type = false;
        for (const auto& pr : match)
          if (is_cre[pr.first] == is_cre[pr.second]) same_type = true;
        if (same_type) continue;  // only opposite-type pairs contract

        std::vector<char> contracted(nslots, 0);
        for (const auto& pr : match) {
          contracted[pr.first] = 1;
          contracted[pr.second] = 1;
        }
        std::vector<int> ext;
        for (int s = 0; s < nslots; ++s)
          if (!contracted[s]) ext.push_back(s);

        std::vector<int> order;
        for (const auto& pr : match) {
          order.push_back(pr.first);
          order.push_back(pr.second);
        }
        for (int s : ext) order.push_back(s);
        const double sign = perm_sign(order);

        const int npairs = static_cast<int>(match.size());
        const int next = static_cast<int>(ext.size());
        const int ngroups = npairs + next;
        std::vector<int> slot_group(nslots, -1);
        for (int g = 0; g < npairs; ++g) {
          slot_group[match[g].first] = g;
          slot_group[match[g].second] = g;
        }
        for (int g = 0; g < next; ++g) slot_group[ext[g]] = npairs + g;

        // Restrict each group's iteration to its support: output legs are
        // active; a contracted pair sums only over the buffer orbitals where
        // its propagator is nonzero (inactive for a particle line, virtual for
        // a hole line). This replaces the O(M^ngroups) sweep with
        // O(|active|^next * |buffer|^npairs) -- the source of the speedup.
        std::array<const std::vector<int>*, 8> gcand{};
        bool empty_group = false;
        for (int g = 0; g < npairs; ++g) {
          gcand[g] = is_cre[match[g].first] ? &inactive_list : &virtual_list;
          empty_group = empty_group || gcand[g]->empty();
        }
        for (int g = 0; g < next; ++g) {
          gcand[npairs + g] = &active_list;
          empty_group = empty_group || active_list.empty();
        }
        if (empty_group) continue;

        std::vector<int> idxv(ngroups, 0);
        while (true) {
          std::array<int, 8> sv{};
          for (int s = 0; s < nslots; ++s)
            sv[s] = (*gcand[slot_group[s]])[idxv[slot_group[s]]];

          const double valA =
              (rA == 1) ? A1(sv[0], sv[1]) : A2get(sv[0], sv[1], sv[2], sv[3]);
          const double valB = (rB == 1) ? B1(sv[2 * rA], sv[2 * rA + 1])
                                        : B2get(sv[2 * rA], sv[2 * rA + 1],
                                                sv[2 * rA + 2], sv[2 * rA + 3]);
          if (valA != 0.0 && valB != 0.0) {
            double prop = 1.0;
            for (int g = 0; g < npairs; ++g) {
              const int v = (*gcand[g])[idxv[g]];
              prop *= is_cre[match[g].first] ? pvec(v) : hvec(v);
            }
            if (prop != 0.0) {
              const double coeff =
                  scale * sign * normA * normB * valA * valB * prop;
              Term ops;
              ops.reserve(next);
              for (int s : ext) ops.push_back({sv[s], is_cre[s]});
              normalize(ops, coeff, out);
            }
          }

          int d = ngroups - 1;
          for (; d >= 0; --d) {
            if (++idxv[d] < static_cast<int>(gcand[d]->size())) break;
            idxv[d] = 0;
          }
          if (d < 0) break;
        }
      }
    }
  }
}

// Dense-tensor projection (reference path): the 2-body operators are full
// spin-orbital tensors indexed directly.
void project_product(const Eigen::MatrixXd& A1, const Eigen::VectorXd& A2,
                     const Eigen::MatrixXd& B1, const Eigen::VectorXd& B2,
                     const SoPartition& part, double scale,
                     const Eigen::VectorXd& hvec, const Eigen::VectorXd& pvec,
                     FermionOp& out) {
  const int M = part.n_so;
  project_product_t(
      A1, [&](int P, int Q, int R, int S) { return A2(idx4(P, Q, R, S, M)); },
      B1, [&](int P, int Q, int R, int S) { return B2(idx4(P, Q, R, S, M)); },
      part, scale, hvec, pvec, out);
}

}  // namespace

namespace reference {

DownfoldResult downfold(const SoTensors& h, const Eigen::VectorXd& eps,
                        const SoPartition& part, const RegOptions& reg) {
  const int M = h.n_so;
  const Generator gen = make_generator(h, eps, part, reg);
  const auto [bd, od] = split_bd_od(h, part);
  const MeanFieldResult ps = mean_field_fold(bd, part);

  // reference-occupation propagators: hole line on virtual (empty),
  // particle line on inactive (filled)
  Eigen::VectorXd hvec = Eigen::VectorXd::Zero(M);
  Eigen::VectorXd pvec = Eigen::VectorXd::Zero(M);
  for (int P = 0; P < M; ++P) {
    if (part.is_virtual[P]) hvec(P) = 1.0;
    if (part.is_inactive[P]) pvec(P) = 1.0;
  }

  // projected 1/2 [S, V] = 1/2 (project(S*V) - project(V*S))
  FermionOp comm;
  project_product(gen.s1, gen.s2, od.f, od.v, part, +0.5, hvec, pvec, comm);
  project_product(od.f, od.v, gen.s1, gen.s2, part, -0.5, hvec, pvec, comm);

  DownfoldResult res;
  res.e = ps.e;
  res.f_active = ps.f_active;
  res.v_active = ps.v_active;
  res.min_denominator = gen.min_denominator;
  res.max_amplitude = gen.max_amplitude;
  for (const auto& [term, coeff] : comm) {
    const int len = static_cast<int>(term.size());
    if (len == 0) {
      res.e += coeff;
    } else if (len == 2) {
      res.f_active(term[0].first, term[1].first) += coeff;
    } else if (len == 4) {
      const int c0 = term[0].first, c1 = term[1].first;
      const int a0 = term[2].first, a1 = term[3].first;
      res.v_active(idx4(c0, c1, a0, a1, M)) += coeff;
      res.v_active(idx4(c1, c0, a0, a1, M)) -= coeff;
      res.v_active(idx4(c0, c1, a1, a0, M)) -= coeff;
      res.v_active(idx4(c1, c0, a1, a0, M)) += coeff;
    }
  }
  return res;
}

ActiveHamiltonian to_spatial_chemist(const DownfoldResult& down,
                                     const SoPartition& part) {
  const int M = part.n_so;
  // active spatial orbitals: both spin-orbitals must be active
  // (spin-restricted)
  std::vector<int> spatial;
  for (int o = 0; 2 * o + 1 < M; ++o) {
    const bool a = part.is_active[2 * o], b = part.is_active[2 * o + 1];
    if (a != b)
      throw std::invalid_argument(
          "to_spatial_chemist: active space is not spin-restricted");
    if (a) spatial.push_back(o);
  }
  const int norb = static_cast<int>(spatial.size());

  ActiveHamiltonian out;
  out.norb = norb;
  out.core_energy = down.e;
  out.one_body = Eigen::MatrixXd::Zero(norb, norb);
  out.two_body = Eigen::VectorXd::Zero(n4(norb));
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      out.one_body(p, q) = down.f_active(2 * spatial[p], 2 * spatial[q]);

  // chemist (pq|rs) = -v[alpha(p), beta(r), alpha(q), beta(s)]: inverse of
  // build_tensors' antisymmetrization, read off the opposite-spin block.
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          out.two_body(idx4(p, q, r, s, norb)) =
              -down.v_active(idx4(2 * spatial[p], 2 * spatial[r] + 1,
                                  2 * spatial[q], 2 * spatial[s] + 1, M));
  return out;
}

}  // namespace reference

// ===========================================================================
// Spin-blocked on-the-fly downfold (PRODUCTION): the path wired into the
// constructor. Stores the two-body as spatial spin blocks and forms every
// element on demand, so no dense n_so^4 tensor is ever materialized.
// ===========================================================================

SpinBlocked2B build_two_body_blocked(const Eigen::VectorXd& g_aaaa,
                                     const Eigen::VectorXd& g_aabb,
                                     const Eigen::VectorXd& g_bbbb, int norb) {
  const Eigen::Index n4 = static_cast<Eigen::Index>(norb) * norb * norb * norb;
  SpinBlocked2B b;
  b.norb = norb;
  b.v_aaaa = Eigen::VectorXd::Zero(n4);
  b.v_bbbb = Eigen::VectorXd::Zero(n4);
  b.v_abab = Eigen::VectorXd::Zero(n4);

  // Same-spin block: W[A,B,C,D] += 0.5 g[flat(A,D,B,C)], then antisymmetrize
  // over (P<->Q) and (R<->S). Purely spatial (mirrors build_tensors restricted
  // to a single all-same-spin channel).
  const auto build_same = [&](const Eigen::VectorXd& g, Eigen::VectorXd& v) {
    std::vector<double> W(static_cast<std::size_t>(n4), 0.0);
    for (int p = 0; p < norb; ++p)
      for (int q = 0; q < norb; ++q)
        for (int r = 0; r < norb; ++r)
          for (int s = 0; s < norb; ++s)
            W[idx4(p, r, s, q, norb)] += 0.5 * g(idx4(p, q, r, s, norb));
    for (int P = 0; P < norb; ++P)
      for (int Q = 0; Q < norb; ++Q)
        for (int R = 0; R < norb; ++R)
          for (int S = 0; S < norb; ++S)
            v(idx4(P, Q, R, S, norb)) =
                W[idx4(P, Q, R, S, norb)] - W[idx4(Q, P, R, S, norb)] -
                W[idx4(P, Q, S, R, norb)] + W[idx4(Q, P, S, R, norb)];
  };
  build_same(g_aaaa, b.v_aaaa);
  build_same(g_bbbb, b.v_bbbb);

  // Opposite-spin block v[alpha(p), beta(q), alpha(r), beta(s)]:
  // the cross-spin W terms vanish under (P<->Q)/(R<->S) except one, giving
  // v_abab[p,q,r,s] = -g_aabb[flat(p,r,q,s)].
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          b.v_abab(idx4(p, q, r, s, norb)) = -g_aabb(idx4(p, r, q, s, norb));
  return b;
}

SpinBlocked2B build_two_body_blocked_restricted(const Eigen::VectorXd& g,
                                                int norb) {
  const Eigen::Index n4 = static_cast<Eigen::Index>(norb) * norb * norb * norb;
  SpinBlocked2B b;
  b.norb = norb;
  b.v_aaaa = Eigen::VectorXd::Zero(n4);
  b.v_abab = Eigen::VectorXd::Zero(n4);

  std::vector<double> W(static_cast<std::size_t>(n4), 0.0);
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          W[idx4(p, r, s, q, norb)] += 0.5 * g(idx4(p, q, r, s, norb));
  for (int P = 0; P < norb; ++P)
    for (int Q = 0; Q < norb; ++Q)
      for (int R = 0; R < norb; ++R)
        for (int S = 0; S < norb; ++S)
          b.v_aaaa(idx4(P, Q, R, S, norb)) =
              W[idx4(P, Q, R, S, norb)] - W[idx4(Q, P, R, S, norb)] -
              W[idx4(P, Q, S, R, norb)] + W[idx4(Q, P, S, R, norb)];

  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          b.v_abab(idx4(p, q, r, s, norb)) = -g(idx4(p, r, q, s, norb));
  return b;
}

double so_v_from_blocked(const SpinBlocked2B& b, int P, int Q, int R, int S) {
  const int n = b.norb;
  const int sP = P & 1, sQ = Q & 1, sR = R & 1, sS = S & 1;
  const int p = P >> 1, q = Q >> 1, r = R >> 1, s = S >> 1;
  if (sP == sQ && sQ == sR && sR == sS) {
    const auto& same_spin =
        sP == 0 || b.v_bbbb.size() == 0 ? b.v_aaaa : b.v_bbbb;
    return same_spin(idx4(p, q, r, s, n));
  }
  // Sz conservation: mixed blocks are nonzero only with one alpha and one beta
  // in each creation/annihilation pair.
  if (sP + sQ != 1 || sR + sS != 1) return 0.0;
  if (sP == 0 && sR == 0) return b.v_abab(idx4(p, q, r, s, n));   // abab
  if (sP == 0 && sR == 1) return -b.v_abab(idx4(p, q, s, r, n));  // abba
  if (sP == 1 && sR == 0) return -b.v_abab(idx4(q, p, r, s, n));  // baab
  return b.v_abab(idx4(q, p, s, r, n));                           // baba
}

Eigen::MatrixXd spin_orbital_one_body(const Eigen::MatrixXd& h1a,
                                      const Eigen::MatrixXd& h1b, int norb) {
  const int M = 2 * norb;
  Eigen::MatrixXd f = Eigen::MatrixXd::Zero(M, M);
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q) {
      f(alpha(p), alpha(q)) = h1a(p, q);
      f(beta(p), beta(q)) = h1b(p, q);
    }
  return f;
}

ActiveDownfoldResult downfold_blocked(const Eigen::MatrixXd& f,
                                      const SpinBlocked2B& blk,
                                      const Eigen::VectorXd& eps,
                                      const SoPartition& part,
                                      const RegOptions& reg, double e_core) {
  const int M = part.n_so;

  // Active spin-orbitals (ascending) + full -> compact index map, so the
  // result is stored over the active space only.
  std::vector<int> active_so;
  for (int o = 0; o < M; ++o)
    if (part.is_active[o]) active_so.push_back(o);

  // On-the-fly two-body element accessors (no dense n_so^4 storage).
  const auto v_at = [&](int P, int Q, int R, int S) {
    return so_v_from_blocked(blk, P, Q, R, S);
  };
  const auto od_v_at = [&](int P, int Q, int R, int S) {
    return is_od2(part, P, Q, R, S) ? v_at(P, Q, R, S) : 0.0;
  };
  const auto bd_v_at = [&](int P, int Q, int R, int S) {
    return is_od2(part, P, Q, R, S) ? 0.0 : v_at(P, Q, R, S);
  };
  const auto s2_at = [&](int P, int Q, int R, int S) -> double {
    if (!is_od2(part, P, Q, R, S)) return 0.0;
    const double vv = v_at(P, Q, R, S);
    if (vv == 0.0) return 0.0;
    const double d = eps(P) + eps(Q) - eps(R) - eps(S);
    return vv * reg_inv(d, reg);
  };

  // Generator: s1 dense (cheap), s2 on the fly; plus intruder diagnostics.
  Eigen::MatrixXd s1 = Eigen::MatrixXd::Zero(M, M);
  double min_denom = std::numeric_limits<double>::infinity();
  double max_amp = 0.0;
  const auto track = [&](double coupling, double delta) {
    const double ad = std::abs(delta);
    min_denom = std::min(min_denom, ad);
    max_amp =
        std::max(max_amp, std::abs(coupling) / std::max(ad, reg.denom_floor));
  };
  for (int P = 0; P < M; ++P)
    for (int Q = 0; Q < M; ++Q)
      if (is_od1(part, P, Q) && f(P, Q) != 0.0) {
        const double d = eps(P) - eps(Q);
        s1(P, Q) = f(P, Q) * reg_inv(d, reg);
        track(f(P, Q), d);
      }
  for (int P = 0; P < M; ++P)
    for (int Q = 0; Q < M; ++Q)
      for (int R = 0; R < M; ++R)
        for (int S = 0; S < M; ++S) {
          if (!is_od2(part, P, Q, R, S)) continue;
          const double vv = v_at(P, Q, R, S);
          if (vv == 0.0) continue;
          track(vv, eps(P) + eps(Q) - eps(R) - eps(S));
        }

  // Block-diagonal / off-diagonal one-body split (dense, cheap).
  Eigen::MatrixXd od_f = Eigen::MatrixXd::Zero(M, M);
  Eigen::MatrixXd bd_f = Eigen::MatrixXd::Zero(M, M);
  for (int P = 0; P < M; ++P)
    for (int Q = 0; Q < M; ++Q)
      (is_od1(part, P, Q) ? od_f : bd_f)(P, Q) = f(P, Q);

  // Reference-occupation mean-field fold of the block-diagonal part.
  Eigen::VectorXd nb = Eigen::VectorXd::Zero(M);
  for (int P = 0; P < M; ++P)
    if (part.is_inactive[P]) nb(P) = 1.0;

  ActiveDownfoldResult res;
  res.active_so = active_so;
  const int n_ac = static_cast<int>(active_so.size());
  // The emitted effective operator is spin-restricted, so the active space must
  // be closed under spin (paired spin-orbitals 2s, 2s+1); only the
  // opposite-spin block is stored and emitted.
  const int n_act = n_ac / 2;
  for (int k = 0; k < n_act; ++k)
    if (active_so[2 * k] % 2 != 0 ||
        active_so[2 * k + 1] != active_so[2 * k] + 1)
      throw std::invalid_argument(
          "downfold_blocked: active space is not spin-restricted");
  res.e = e_core;
  for (int P = 0; P < M; ++P) res.e += bd_f(P, P) * nb(P);
  for (int P = 0; P < M; ++P)
    for (int Q = 0; Q < M; ++Q)
      res.e -= 0.5 * bd_v_at(P, Q, P, Q) * nb(P) * nb(Q);

  res.f_active = Eigen::MatrixXd::Zero(n_ac, n_ac);
  for (int ci = 0; ci < n_ac; ++ci) {
    const int i = active_so[ci];
    for (int cj = 0; cj < n_ac; ++cj) {
      const int j = active_so[cj];
      double fold = 0.0;
      for (int b = 0; b < M; ++b) fold += bd_v_at(i, b, b, j) * nb(b);
      res.f_active(ci, cj) = bd_f(i, j) + fold;
    }
  }

  res.v_abab = Eigen::VectorXd::Zero(n4(n_act));
  for (int p = 0; p < n_act; ++p)
    for (int q = 0; q < n_act; ++q)
      for (int r = 0; r < n_act; ++r)
        for (int s = 0; s < n_act; ++s)
          res.v_abab(idx4(p, q, r, s, n_act)) =
              bd_v_at(active_so[2 * p], active_so[2 * q + 1], active_so[2 * r],
                      active_so[2 * s + 1]);
  // reference-occupation propagators: hole line on virtual, particle on
  // inactive
  Eigen::VectorXd hvec = Eigen::VectorXd::Zero(M);
  Eigen::VectorXd pvec = Eigen::VectorXd::Zero(M);
  for (int P = 0; P < M; ++P) {
    if (part.is_virtual[P]) hvec(P) = 1.0;
    if (part.is_inactive[P]) pvec(P) = 1.0;
  }

  // projected 1/2 [S, V] = 1/2 (project(S*V) - project(V*S)), on the fly.
  RetainedOperator comm(active_so);
  project_two_body_between_blas(s2_at, od_v_at, part, +0.5, hvec, pvec, comm);
  project_two_body_between_blas(od_v_at, s2_at, part, -0.5, hvec, pvec, comm);
  project_one_line_between_blas(s1, s2_at, od_f, od_v_at, part, +0.5, comm);
  project_one_line_between_blas(od_f, od_v_at, s1, s2_at, part, -0.5, comm);
  project_remaining_blas(s1, s2_at, od_f, od_v_at, part, +0.5, comm);
  project_remaining_blas(od_f, od_v_at, s1, s2_at, part, -0.5, comm);

  res.min_denominator = std::isinf(min_denom) ? 0.0 : min_denom;
  res.max_amplitude = max_amp;
  res.e += comm.scalar;
  res.f_active += comm.one_body;
  res.v_abab += comm.two_body;
  return res;
}

ActiveHamiltonian to_spatial_chemist(const ActiveDownfoldResult& down,
                                     const SoPartition& part) {
  const int M = part.n_so;
  std::vector<int> spatial;
  for (int o = 0; 2 * o + 1 < M; ++o) {
    const bool a = part.is_active[2 * o], b = part.is_active[2 * o + 1];
    if (a != b)
      throw std::invalid_argument(
          "to_spatial_chemist: active space is not spin-restricted");
    if (a) spatial.push_back(o);
  }
  const int norb = static_cast<int>(spatial.size());
  const int n_ac = static_cast<int>(down.active_so.size());
  std::vector<int> so2c(M, -1);
  for (int c = 0; c < n_ac; ++c) so2c[down.active_so[c]] = c;

  ActiveHamiltonian out;
  out.norb = norb;
  out.core_energy = down.e;
  out.one_body = Eigen::MatrixXd::Zero(norb, norb);
  out.two_body = Eigen::VectorXd::Zero(n4(norb));
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      out.one_body(p, q) =
          down.f_active(so2c[2 * spatial[p]], so2c[2 * spatial[q]]);

  // chemist (pq|rs) = -v_abab[p, r, q, s]: the stored opposite-spin block is
  // indexed by active spatial position (ascending active-orbital order).
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          out.two_body(idx4(p, q, r, s, norb)) =
              -down.v_abab(idx4(p, r, q, s, norb));
  return out;
}

SoPartition make_partition(int norb, const std::vector<int>& active_spatial,
                           const std::vector<double>& occupation) {
  if (static_cast<int>(occupation.size()) != norb)
    throw std::invalid_argument("make_partition: occupation size != norb");
  const int M = 2 * norb;
  SoPartition part;
  part.n_so = M;
  part.is_active.assign(M, 0);
  part.is_inactive.assign(M, 0);
  part.is_virtual.assign(M, 0);
  std::vector<char> is_kept(norb, 0);
  for (int o : active_spatial) {
    if (o < 0 || o >= norb)
      throw std::invalid_argument("make_partition: active index out of range");
    is_kept[o] = 1;
  }
  for (int o = 0; o < norb; ++o) {
    std::vector<char>& role =
        is_kept[o]
            ? part.is_active
            : (occupation[o] >= 1.0 ? part.is_inactive : part.is_virtual);
    role[2 * o] = 1;
    role[2 * o + 1] = 1;
  }
  return part;
}

}  // namespace qdk::chemistry::algorithms::microsoft::swpt2
