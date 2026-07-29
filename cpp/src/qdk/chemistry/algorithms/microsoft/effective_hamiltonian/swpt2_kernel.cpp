// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

#include <algorithm>
#include <array>
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

// ---------------------------------------------------------------------------
// Projected commutator 1/2 [S, V] via generalized Wick over the external legs.
// ---------------------------------------------------------------------------
namespace {

// A number-conserving operator over the (small) active space: a normal-ordered
// term (creations ascending, then annihilations ascending) -> coefficient.
using Term = std::vector<std::pair<int, int>>;  // (spin-orbital, 1=cre / 0=ann)
using FermionOp = std::map<Term, double>;

// Normal-order a raw operator string (anticommutation), accumulating into
// `out`.
void normalize(Term ops, double coeff, FermionOp& out) {
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
  out[ops] += coeff;
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

// Active-space reference projection of A * B (A left, B right), buffer summed.
// A/B given as (1-body, 2-body) tensors; contributions emitted (scaled) into
// `out` over active spin-orbitals. hvec/pvec are the hole/particle propagators.
void project_product(const Eigen::MatrixXd& A1, const Eigen::VectorXd& A2,
                     const Eigen::MatrixXd& B1, const Eigen::VectorXd& B2,
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

          const double valA = (rA == 1)
                                  ? A1(sv[0], sv[1])
                                  : A2(idx4(sv[0], sv[1], sv[2], sv[3], M));
          const double valB = (rB == 1)
                                  ? B1(sv[2 * rA], sv[2 * rA + 1])
                                  : B2(idx4(sv[2 * rA], sv[2 * rA + 1],
                                            sv[2 * rA + 2], sv[2 * rA + 3], M));
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

}  // namespace

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
  double higher2 = 0.0;
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
    } else {
      higher2 += coeff * coeff;  // discarded >= 3-body
    }
  }
  res.higher_body_norm = std::sqrt(higher2);
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
