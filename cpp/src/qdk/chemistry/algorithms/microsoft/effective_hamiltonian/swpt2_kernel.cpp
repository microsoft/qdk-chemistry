// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

#include <algorithm>
#include <array>
#include <blas.hh>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::swpt2 {

namespace {
inline int alpha(int p) { return 2 * p; }
inline int beta(int p) { return 2 * p + 1; }
inline Eigen::Index n4(int n) {
  return static_cast<Eigen::Index>(n) * n * n * n;
}
}  // namespace

// ===========================================================================
// Shared kernel foundations: denominator regularization, generalized Fock,
// occupation-change masks, and projected-commutator Wick machinery.
// ===========================================================================

double regularized_inverse(double delta, const RegularizerOptions& reg) {
  if (reg.denom_flow > 0.0) {
    const double d2 = delta * delta;
    if (reg.denom_flow * d2 < 1e-14) return reg.denom_flow * delta;
    return (1.0 - std::exp(-reg.denom_flow * d2)) / delta;
  }
  if (reg.denom_imaginary_shift > 0.0) {
    return delta / (delta * delta +
                    reg.denom_imaginary_shift * reg.denom_imaginary_shift);
  }
  if (std::abs(delta) < reg.denom_floor) return 0.0;
  return 1.0 / delta;
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
              density(r, s) * (two_body(idx4(p, q, r, s, norb)) -
                               0.5 * two_body(idx4(p, r, s, q, norb)));
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

  using RowMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Index n = norb;
  const Eigen::Index n3 = n * n * n;
  // Each pass transforms the leading index and cycles it to the back, so four
  // passes restore (p, q, r, s) order.
  Eigen::VectorXd current = two_body;
  Eigen::VectorXd next(n4(norb));
  for (int pass = 0; pass < 4; ++pass) {
    const Eigen::Map<const RowMajorMatrix> in(current.data(), n, n3);
    const Eigen::MatrixXd transformed = rotation.transpose() * in;
    Eigen::Map<RowMajorMatrix>(next.data(), n3, n) = transformed.transpose();
    current.swap(next);
  }
  return current;
}

namespace {
// Net occupation change of a role set under a^dag_P a_Q / a^dag_P a^dag_Q a_R
// a_S.
inline int net_change(const std::vector<char>& role, int P, int Q) {
  return role[P] - role[Q];
}
inline int net_change(const std::vector<char>& role, int P, int Q, int R,
                      int S) {
  return role[P] + role[Q] - role[R] - role[S];
}
// A term belongs to H_OD when it changes the inactive OR the virtual count;
// tracking them separately catches inactive<->virtual excitations, which are
// net-zero on the combined external set.
inline bool changes_external_occupation(const SpinOrbitalPartition& part, int P,
                                        int Q) {
  return net_change(part.is_inactive, P, Q) != 0 ||
         net_change(part.is_virtual, P, Q) != 0;
}
inline bool changes_external_occupation(const SpinOrbitalPartition& part, int P,
                                        int Q, int R, int S) {
  return net_change(part.is_inactive, P, Q, R, S) != 0 ||
         net_change(part.is_virtual, P, Q, R, S) != 0;
}
}  // namespace

// ---------------------------------------------------------------------------
// Projected commutator 1/2 [S, V] via generalized Wick over the external legs.
// ---------------------------------------------------------------------------
namespace {

/// One ladder operator: (spin-orbital, 1 = creation / 0 = annihilation).
using Operator = std::pair<int, int>;
/// A product of at most two two-body operators, so at most eight ladders.
using OperatorString = std::array<Operator, 8>;

struct RetainedOperator {
  RetainedOperator(const std::vector<int>& active_so,
                   const std::vector<int>& occupied_so = {},
                   const Eigen::MatrixXd& reference_density = {})
      : so2c(active_so.empty() ? 0 : active_so.back() + 1, -1),
        occupied(so2c.size(), 0),
        density(reference_density),
        one_body(Eigen::MatrixXd::Zero(active_so.size(), active_so.size())),
        two_body(Eigen::VectorXd::Zero(n4(active_so.size() / 2))) {
    for (int compact = 0; compact < static_cast<int>(active_so.size());
         ++compact)
      so2c[active_so[compact]] = compact;
    for (int orbital : occupied_so) occupied[orbital] = 1;
    has_density = density.size() > 0;
    folds_above_two_body = !occupied_so.empty() || has_density;
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

  bool is_occupied(int orbital) const { return occupied[orbital] != 0; }

  double scalar = 0.0;
  bool folds_above_two_body = false;
  bool has_density = false;
  std::vector<int> so2c;
  std::vector<char> occupied;
  Eigen::MatrixXd density;  // spin-traced, over window spatial orbitals
  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;

  /// Reference propagator <a^dag_P a_Q> for spin-orbitals P, Q. The reference
  /// is spin-restricted, so the spin-traced density splits evenly and opposite
  /// spins do not contract.
  double gamma(int P, int Q) const {
    if (((P ^ Q) & 1) != 0) return 0.0;
    return 0.5 * density(P >> 1, Q >> 1);
  }

 private:
  int compact(int orbital) const { return so2c[orbital]; }

  void add_abab(int i, int j, int k, int l, double coeff) {
    if ((i & 1) == 0 && (j & 1) == 1 && (k & 1) == 0 && (l & 1) == 1) {
      const int nactive = one_body.rows() / 2;
      two_body(idx4(i >> 1, j >> 1, k >> 1, l >> 1, nactive)) += coeff;
    }
  }
};

// Walk every nonempty set of disjoint (creation, annihilation) contractions,
// carrying the accumulated propagator product and sign and visiting the
// uncontracted remainder. Pairs whose propagator vanishes are pruned during the
// walk rather than after it, which is what keeps a diagonal reference density
// cheap. Terms arrive canonically ordered -- creations first -- so every
// annihilation sits to the right of every creation and removing a pair
// preserves the order.
template <class Visit>
void walk_contractions(const OperatorString& ops, int n,
                       const RetainedOperator& out, int position,
                       std::array<char, 8>& removed, int depth, double weight,
                       const Visit& visit) {
  if (position >= n) {
    if (depth == 0) return;
    OperatorString remainder{};
    int size = 0;
    for (int i = 0; i < n; ++i)
      if (!removed[i]) remainder[size++] = ops[i];
    visit(remainder, size, weight);
    return;
  }
  if (ops[position].second != 1 || removed[position]) {
    walk_contractions(ops, n, out, position + 1, removed, depth, weight, visit);
    return;
  }
  walk_contractions(ops, n, out, position + 1, removed, depth, weight, visit);
  for (int j = position + 1; j < n; ++j) {
    if (ops[j].second != 0 || removed[j]) continue;
    const double propagator = out.gamma(ops[position].first, ops[j].first);
    if (propagator == 0.0) continue;
    int crossings = 0;
    for (int between = position + 1; between < j; ++between)
      if (!removed[between]) ++crossings;
    removed[position] = removed[j] = 1;
    walk_contractions(ops, n, out, position + 1, removed, depth + 1,
                      (crossings % 2 != 0 ? -weight : weight) * propagator,
                      visit);
    removed[position] = removed[j] = 0;
  }
}

// Reference-normal-ordered {A}, expressed back in vacuum-ordered strings. This
// is the Wick identity A = sum_S eps(S) prod(gamma) {A_S} solved for {A}, so
// the recursion is well founded: every term drops two operators.
template <class Sink>
void reference_normal_ordered(const OperatorString& ops, int n, double coeff,
                              const RetainedOperator& out, const Sink& sink) {
  sink(ops, n, coeff);
  std::array<char, 8> removed{};
  walk_contractions(
      ops, n, out, 0, removed, 0, 1.0,
      [&](const OperatorString& remainder, int size, double weight) {
        reference_normal_ordered(remainder, size, -coeff * weight, out, sink);
      });
}

// A term above two-body cannot be emitted, and discarding it throws away its
// reference contractions, which dominate. Keep everything except the
// reference-normal-ordered residual: A - {A} = sum over nonempty contraction
// sets. Unlike the particle-hole construction this needs no idempotency, so it
// covers correlated and fractionally occupied references.
void fold_onto_density(const OperatorString& ops, int n, double coeff,
                       RetainedOperator& out) {
  std::array<char, 8> removed{};
  walk_contractions(
      ops, n, out, 0, removed, 0, 1.0,
      [&](const OperatorString& remainder, int size, double weight) {
        reference_normal_ordered(
            remainder, size, coeff * weight, out,
            [&out](const OperatorString& term, int emitted_size, double value) {
              // Anything still above two-body is the residual.
              if (emitted_size <= 4) out.add(term.data(), emitted_size, value);
            });
      });
}

// Vacuum normal ordering. `sink` receives every fully ordered term, including
// those above two-body, and decides what to do with them.
template <class Sink>
void normal_order(OperatorString ops, int n, double coeff, const Sink& sink) {
  for (int i = 0; i + 1 < n; ++i) {
    const int ai = ops[i].first, aa = ops[i].second;
    const int bi = ops[i + 1].first, ba = ops[i + 1].second;
    const bool wrong = (aa == 0 && ba == 1) || (aa == ba && ai > bi);
    if (wrong) {
      auto swapped = ops;
      std::swap(swapped[i], swapped[i + 1]);
      normal_order(swapped, n, -coeff, sink);
      if (aa != ba && ai == bi) {
        auto contracted = ops;
        std::move(contracted.begin() + i + 2, contracted.begin() + n,
                  contracted.begin() + i);
        normal_order(contracted, n - 2, coeff, sink);
      }
      return;
    }
    if (aa == ba && ai == bi) return;
  }
  sink(ops, n, coeff);
}

// The same fold against a reference determinant: exchange creation and
// annihilation roles on the occupied spin-orbitals, normal-order against that
// vacuum, and keep whatever falls to two-body. Equivalent to
// `fold_onto_density` for an idempotent density, which is what the tests
// cross-check; production always takes the density path.
void fold_onto_determinant(const OperatorString& ops, int n, double coeff,
                           RetainedOperator& out) {
  const auto exchange = [&out](OperatorString term, int size) {
    for (int i = 0; i < size; ++i)
      if (out.is_occupied(term[i].first)) term[i].second = 1 - term[i].second;
    return term;
  };
  const auto emit = [&out](const OperatorString& term, int size,
                           double weight) {
    if (size <= 4) out.add(term.data(), size, weight);
  };
  normal_order(exchange(ops, n), n, coeff,
               [&](const OperatorString& term, int size, double weight) {
                 if (size > 4) return;  // the residual this approximation drops
                 normal_order(exchange(term, size), size, weight, emit);
               });
}

void accumulate_normal_ordered(OperatorString ops, int n, double coeff,
                               RetainedOperator& out) {
  normal_order(ops, n, coeff,
               [&out](const OperatorString& term, int size, double weight) {
                 if (size <= 4)
                   out.add(term.data(), size, weight);
                 else if (out.has_density)
                   fold_onto_density(term, size, weight, out);
                 else if (out.folds_above_two_body)
                   fold_onto_determinant(term, size, weight, out);
               });
}

template <std::size_t N>
void accumulate_slots(const std::array<int, N>& values,
                      const std::array<int, N>& is_cre,
                      const std::vector<int>& slots, double coeff,
                      RetainedOperator& out) {
  OperatorString ops{};
  int size = 0;
  for (int slot : slots) ops[size++] = {values[slot], is_cre[slot]};
  accumulate_normal_ordered(ops, size, coeff, out);
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

/// Spin-orbitals of each role, ascending, plus the inverse map for the active
/// ones (-1 elsewhere).
struct RoleLists {
  std::vector<int> active, inactive, virtuals, active_position;
};

RoleLists role_lists(const SpinOrbitalPartition& part) {
  RoleLists lists;
  lists.active_position.assign(part.n_so, -1);
  for (int orbital = 0; orbital < part.n_so; ++orbital) {
    if (part.is_active[orbital]) {
      lists.active_position[orbital] = static_cast<int>(lists.active.size());
      lists.active.push_back(orbital);
    }
    if (part.is_inactive[orbital]) lists.inactive.push_back(orbital);
    if (part.is_virtual[orbital]) lists.virtuals.push_back(orbital);
  }
  return lists;
}

// All matchings factor into two operand-local panels.
// External active legs form the rows, cross-operator external lines form the
// shared column, and contractions internal to one operand are reduced while
// packing that operand's panel.
//
// A matching leaving more than four active legs produces a rank-three or
// higher operator, which survives the two-body truncation only if an
// annihilator of A coincides with a creator of B. Those channels therefore
// enumerate the coinciding row pairs instead of forming the full product.
template <class A2Get, class B2Get, class Output>
void project_matchings(const Eigen::MatrixXd& A1, A2Get A2get,
                       const Eigen::MatrixXd& B1, B2Get B2get,
                       const SpinOrbitalPartition& part, double scale,
                       Output& out) {
  const RoleLists lists = role_lists(part);
  const std::vector<int>& active_list = lists.active;
  const std::vector<int>& active_position = lists.active_position;
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
      // With nothing contracted both commutator orderings normal-order to the
      // same string -- every operand has an even number of legs -- so they
      // cancel.
      if (match.empty()) continue;
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
      // Above two-body rank only an A-annihilator / B-creator coincidence
      // survives normal ordering, so without one the whole matching is dropped.
      // That holds only while terms above two-body are discarded; when they are
      // folded onto the reference instead, every matching contributes.
      std::vector<int> a_annihilators, b_creators;
      for (int slot : ext_a)
        if (!is_cre[slot]) a_annihilators.push_back(slot);
      for (int slot : ext_b)
        if (is_cre[slot]) b_creators.push_back(slot);
      const bool needs_coincidence =
          ext.size() > 4 && !out.folds_above_two_body;
      if (needs_coincidence && (a_annihilators.empty() || b_creators.empty()))
        continue;
      // Folding relaxes that gate but not entirely: reaching two-body still
      // costs one creation and one annihilation per contraction, so a matching
      // without enough of either cannot contribute.
      if (out.folds_above_two_body && ext.size() > 4) {
        int creations = 0;
        for (int slot : ext)
          if (is_cre[slot]) ++creations;
        const int annihilations = static_cast<int>(ext.size()) - creations;
        const int required = (static_cast<int>(ext.size()) - 4) / 2;
        if (creations < required || annihilations < required) continue;
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
      bool empty_external = false;
      for (int pair = 0; pair < static_cast<int>(match.size()); ++pair) {
        pair_lists[pair] =
            is_cre[match[pair].first] ? &lists.inactive : &lists.virtuals;
        empty_external = empty_external || pair_lists[pair]->empty();
      }
      if (empty_external) continue;

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

      const auto emit = [&](Eigen::Index row_a, Eigen::Index row_b,
                            double dot) {
        const double coeff = prefactor * dot;
        if (coeff == 0.0) return;
        std::array<int, 8> values{};
        assign_active_slots(values, ext_a, row_a, active_list);
        assign_active_slots(values, ext_b, row_b, active_list);
        accumulate_slots(values, is_cre, ext, coeff, out);
      };

      if (!needs_coincidence) {
        Eigen::MatrixXd product = Eigen::MatrixXd::Zero(nrow_a, nrow_b);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                   nrow_a, nrow_b, ncol, 1.0, a.data(), nrow_a, b.data(),
                   nrow_b, 0.0, product.data(), nrow_a);
        for (Eigen::Index row_a = 0; row_a < nrow_a; ++row_a)
          for (Eigen::Index row_b = 0; row_b < nrow_b; ++row_b)
            emit(row_a, row_b, product(row_a, row_b));
        continue;
      }

      // `assign_active_slots` reads ext_b back to front, so the k-th ext_b slot
      // is the digit of weight nactive^(|ext_b| - 1 - k) in row_b. Fixing that
      // digit to an A-annihilator's orbital enumerates the surviving rows.
      std::vector<char> seen(nrow_b, 0);
      std::vector<Eigen::Index> candidates;
      for (Eigen::Index row_a = 0; row_a < nrow_a; ++row_a) {
        std::array<int, 8> values_a{};
        assign_active_slots(values_a, ext_a, row_a, active_list);
        candidates.clear();
        for (std::size_t k = 0; k < ext_b.size(); ++k) {
          if (!is_cre[ext_b[k]]) continue;
          const Eigen::Index stride =
              integer_power(nactive, static_cast<int>(ext_b.size() - 1 - k));
          const Eigen::Index high_count =
              integer_power(nactive, static_cast<int>(k));
          for (int a_slot : a_annihilators) {
            const Eigen::Index digit = active_position[values_a[a_slot]];
            for (Eigen::Index high = 0; high < high_count; ++high)
              for (Eigen::Index low = 0; low < stride; ++low) {
                const Eigen::Index row_b =
                    (high * nactive + digit) * stride + low;
                if (seen[row_b]) continue;
                seen[row_b] = 1;
                candidates.push_back(row_b);
              }
          }
        }
        for (Eigen::Index row_b : candidates) {
          double dot = 0.0;
          for (Eigen::Index col = 0; col < ncol; ++col)
            dot += a(row_a, col) * b(row_b, col);
          emit(row_a, row_b, dot);
        }
        for (Eigen::Index row_b : candidates) seen[row_b] = 0;
      }
    }
  }
}

}  // namespace

// ===========================================================================
// Spin-blocked on-the-fly downfold.
// ===========================================================================

SpinBlockedTwoBody build_two_body_blocked(const Eigen::VectorXd& g, int norb) {
  const Eigen::Index size = n4(norb);
  SpinBlockedTwoBody b;
  b.norb = norb;
  b.v_aaaa = Eigen::VectorXd::Zero(size);
  b.v_abab = Eigen::VectorXd::Zero(size);

  // Same-spin block: ordered[A,B,C,D] += 0.5 g[idx4(A,D,B,C)], then
  // antisymmetrize over (P<->Q) and (R<->S).
  std::vector<double> ordered(static_cast<std::size_t>(size), 0.0);
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          ordered[idx4(p, r, s, q, norb)] += 0.5 * g(idx4(p, q, r, s, norb));
  for (int P = 0; P < norb; ++P)
    for (int Q = 0; Q < norb; ++Q)
      for (int R = 0; R < norb; ++R)
        for (int S = 0; S < norb; ++S)
          b.v_aaaa(idx4(P, Q, R, S, norb)) = ordered[idx4(P, Q, R, S, norb)] -
                                             ordered[idx4(Q, P, R, S, norb)] -
                                             ordered[idx4(P, Q, S, R, norb)] +
                                             ordered[idx4(Q, P, S, R, norb)];

  // Opposite-spin block v[alpha(p), beta(q), alpha(r), beta(s)]: the cross-spin
  // terms vanish under (P<->Q)/(R<->S) except one.
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q)
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          b.v_abab(idx4(p, q, r, s, norb)) = -g(idx4(p, r, q, s, norb));
  return b;
}

double so_v_from_blocked(const SpinBlockedTwoBody& blocks, int P, int Q, int R,
                         int S) {
  const int n = blocks.norb;
  const int sP = P & 1, sQ = Q & 1, sR = R & 1, sS = S & 1;
  const int p = P >> 1, q = Q >> 1, r = R >> 1, s = S >> 1;
  if (sP == sQ && sQ == sR && sR == sS)
    return blocks.v_aaaa(idx4(p, q, r, s, n));
  // Sz conservation: mixed blocks are nonzero only with one alpha and one beta
  // in each creation/annihilation pair.
  if (sP + sQ != 1 || sR + sS != 1) return 0.0;
  if (sP == 0 && sR == 0) return blocks.v_abab(idx4(p, q, r, s, n));   // abab
  if (sP == 0 && sR == 1) return -blocks.v_abab(idx4(p, q, s, r, n));  // abba
  if (sP == 1 && sR == 0) return -blocks.v_abab(idx4(q, p, r, s, n));  // baab
  return blocks.v_abab(idx4(q, p, s, r, n));                           // baba
}

Eigen::MatrixXd spin_orbital_one_body(const Eigen::MatrixXd& h1a,
                                      const Eigen::MatrixXd& h1b, int norb) {
  const int n_so = 2 * norb;
  Eigen::MatrixXd f = Eigen::MatrixXd::Zero(n_so, n_so);
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q) {
      f(alpha(p), alpha(q)) = h1a(p, q);
      f(beta(p), beta(q)) = h1b(p, q);
    }
  return f;
}

ActiveDownfoldResult downfold_blocked(
    const Eigen::MatrixXd& f, const SpinBlockedTwoBody& blk,
    const Eigen::VectorXd& eps, const SpinOrbitalPartition& part,
    const RegularizerOptions& reg, double e_core,
    const std::vector<int>& occupied_so,
    const Eigen::MatrixXd& reference_density) {
  const int n_so = part.n_so;

  std::vector<int> active_so;
  for (int o = 0; o < n_so; ++o)
    if (part.is_active[o]) active_so.push_back(o);

  // On-the-fly two-body element accessors (no dense n_so^4 storage).
  const auto v_at = [&](int P, int Q, int R, int S) {
    return so_v_from_blocked(blk, P, Q, R, S);
  };
  const auto od_v_at = [&](int P, int Q, int R, int S) {
    return changes_external_occupation(part, P, Q, R, S) ? v_at(P, Q, R, S)
                                                         : 0.0;
  };
  const auto bd_v_at = [&](int P, int Q, int R, int S) {
    return changes_external_occupation(part, P, Q, R, S) ? 0.0
                                                         : v_at(P, Q, R, S);
  };
  const auto s2_at = [&](int P, int Q, int R, int S) -> double {
    if (!changes_external_occupation(part, P, Q, R, S)) return 0.0;
    const double vv = v_at(P, Q, R, S);
    if (vv == 0.0) return 0.0;
    const double d = eps(P) + eps(Q) - eps(R) - eps(S);
    return vv * regularized_inverse(d, reg);
  };

  // Generator: s1 dense (cheap), s2 on the fly; plus intruder diagnostics.
  Eigen::MatrixXd s1 = Eigen::MatrixXd::Zero(n_so, n_so);
  double min_denom = std::numeric_limits<double>::infinity();
  double max_amp = 0.0;
  const auto track = [&](double coupling, double delta) {
    const double ad = std::abs(delta);
    min_denom = std::min(min_denom, ad);
    max_amp =
        std::max(max_amp, std::abs(coupling) / std::max(ad, reg.denom_floor));
  };
  for (int P = 0; P < n_so; ++P)
    for (int Q = 0; Q < n_so; ++Q)
      if (changes_external_occupation(part, P, Q) && f(P, Q) != 0.0) {
        const double d = eps(P) - eps(Q);
        s1(P, Q) = f(P, Q) * regularized_inverse(d, reg);
        track(f(P, Q), d);
      }
  // Diagnostics only, materializing nothing. Restricted to the six nonzero
  // spin patterns rather than the full (2*norb)^4 grid.
  static constexpr int spin_patterns[6][4] = {{0, 0, 0, 0}, {1, 1, 1, 1},
                                              {0, 1, 0, 1}, {0, 1, 1, 0},
                                              {1, 0, 0, 1}, {1, 0, 1, 0}};
  const int n_spatial = n_so / 2;
  for (int p = 0; p < n_spatial; ++p)
    for (int q = 0; q < n_spatial; ++q)
      for (int r = 0; r < n_spatial; ++r)
        for (int s = 0; s < n_spatial; ++s)
          for (const auto& sp : spin_patterns) {
            const int P = 2 * p + sp[0], Q = 2 * q + sp[1];
            const int R = 2 * r + sp[2], S = 2 * s + sp[3];
            if (!changes_external_occupation(part, P, Q, R, S)) continue;
            const double vv = v_at(P, Q, R, S);
            if (vv == 0.0) continue;
            track(vv, eps(P) + eps(Q) - eps(R) - eps(S));
          }

  // Block-diagonal / off-diagonal one-body split (dense, cheap).
  Eigen::MatrixXd od_f = Eigen::MatrixXd::Zero(n_so, n_so);
  Eigen::MatrixXd bd_f = Eigen::MatrixXd::Zero(n_so, n_so);
  for (int P = 0; P < n_so; ++P)
    for (int Q = 0; Q < n_so; ++Q)
      (changes_external_occupation(part, P, Q) ? od_f : bd_f)(P, Q) = f(P, Q);

  // Reference-occupation mean-field fold of the block-diagonal part. Folded
  // occupations are integers by construction, one per occupied spin-orbital.
  Eigen::VectorXd reference_occupation = Eigen::VectorXd::Zero(n_so);
  for (int P = 0; P < n_so; ++P)
    if (part.is_inactive[P]) reference_occupation(P) = 1.0;

  ActiveDownfoldResult res;
  res.active_so = active_so;
  const int n_active_so = static_cast<int>(active_so.size());
  // The emitted effective operator is spin-restricted, so the active space must
  // be closed under spin (paired spin-orbitals 2s, 2s+1); only the
  // opposite-spin block is stored and emitted.
  const int n_active_spatial = n_active_so / 2;
  bool spin_restricted = n_active_so % 2 == 0;
  for (int k = 0; k < n_active_spatial && spin_restricted; ++k)
    spin_restricted = active_so[2 * k] % 2 == 0 &&
                      active_so[2 * k + 1] == active_so[2 * k] + 1;
  if (!spin_restricted)
    throw std::invalid_argument(
        "downfold_blocked: active space is not spin-restricted");
  res.e = e_core;
  for (int P = 0; P < n_so; ++P) res.e += bd_f(P, P) * reference_occupation(P);
  for (int P = 0; P < n_so; ++P)
    for (int Q = 0; Q < n_so; ++Q)
      res.e -= 0.5 * bd_v_at(P, Q, P, Q) * reference_occupation(P) *
               reference_occupation(Q);

  res.f_active = Eigen::MatrixXd::Zero(n_active_so, n_active_so);
  for (int ci = 0; ci < n_active_so; ++ci) {
    const int i = active_so[ci];
    for (int cj = 0; cj < n_active_so; ++cj) {
      const int j = active_so[cj];
      double fold = 0.0;
      for (int b = 0; b < n_so; ++b)
        fold += bd_v_at(i, b, b, j) * reference_occupation(b);
      res.f_active(ci, cj) = bd_f(i, j) + fold;
    }
  }

  res.v_abab = Eigen::VectorXd::Zero(n4(n_active_spatial));
  for (int p = 0; p < n_active_spatial; ++p)
    for (int q = 0; q < n_active_spatial; ++q)
      for (int r = 0; r < n_active_spatial; ++r)
        for (int s = 0; s < n_active_spatial; ++s)
          res.v_abab(idx4(p, q, r, s, n_active_spatial)) =
              bd_v_at(active_so[2 * p], active_so[2 * q + 1], active_so[2 * r],
                      active_so[2 * s + 1]);

  // projected 1/2 [S, V] = 1/2 (project(S*V) - project(V*S)), on the fly.
  RetainedOperator comm(active_so, occupied_so, reference_density);
  project_matchings(s1, s2_at, od_f, od_v_at, part, +0.5, comm);
  project_matchings(od_f, od_v_at, s1, s2_at, part, -0.5, comm);

  res.min_denominator = std::isinf(min_denom) ? 0.0 : min_denom;
  res.max_amplitude = max_amp;
  res.e += comm.scalar;
  res.f_active += comm.one_body;
  res.v_abab += comm.two_body;
  return res;
}

ActiveHamiltonian to_spatial_chemist(const ActiveDownfoldResult& down,
                                     const SpinOrbitalPartition& part) {
  const int n_so = part.n_so;
  std::vector<int> spatial;
  for (int o = 0; 2 * o + 1 < n_so; ++o) {
    const bool a = part.is_active[2 * o], b = part.is_active[2 * o + 1];
    if (a != b)
      throw std::invalid_argument(
          "to_spatial_chemist: active space is not spin-restricted");
    if (a) spatial.push_back(o);
  }
  const int norb = static_cast<int>(spatial.size());
  const int n_active_so = static_cast<int>(down.active_so.size());
  std::vector<int> so2c(n_so, -1);
  for (int c = 0; c < n_active_so; ++c) so2c[down.active_so[c]] = c;

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

SpinOrbitalPartition make_partition(int norb,
                                    const std::vector<int>& active_spatial,
                                    const std::vector<int>& inactive_spatial,
                                    const std::vector<int>& virtual_spatial) {
  if (norb < 0)
    throw std::invalid_argument("make_partition: negative orbital count");
  const int n_so = 2 * norb;
  SpinOrbitalPartition part;
  part.n_so = n_so;
  part.is_active.assign(n_so, 0);
  part.is_inactive.assign(n_so, 0);
  part.is_virtual.assign(n_so, 0);

  std::vector<char> assigned(norb, 0);
  const auto assign = [&](const std::vector<int>& orbitals,
                          std::vector<char>& role) {
    for (int o : orbitals) {
      if (o < 0 || o >= norb)
        throw std::invalid_argument("make_partition: index out of range");
      if (assigned[o])
        throw std::invalid_argument(
            "make_partition: orbital assigned to more than one role");
      assigned[o] = 1;
      role[2 * o] = 1;
      role[2 * o + 1] = 1;
    }
  };
  assign(active_spatial, part.is_active);
  assign(inactive_spatial, part.is_inactive);
  assign(virtual_spatial, part.is_virtual);
  for (int o = 0; o < norb; ++o)
    if (!assigned[o])
      throw std::invalid_argument("make_partition: orbital has no role");
  return part;
}

WindowPartition partition_window(
    const std::vector<double>& occupation,
    const std::vector<std::size_t>& window_global,
    const std::unordered_set<std::size_t>& kept_global, int window_electrons,
    double max_folded_occupation_deviation) {
  if (occupation.size() != window_global.size())
    throw std::invalid_argument(
        "SchriefferWolffPT2: occupation and window index lists differ in "
        "length.");
  if (!std::isfinite(max_folded_occupation_deviation) ||
      max_folded_occupation_deviation < 0.0 ||
      max_folded_occupation_deviation >= 1.0)
    throw std::invalid_argument(
        "SchriefferWolffPT2: max_folded_occupation_deviation must lie in "
        "[0, 1); a half-occupied orbital cannot be folded unambiguously.");

  WindowPartition out;
  const int norb = static_cast<int>(occupation.size());
  for (int i = 0; i < norb; ++i) {
    if (kept_global.count(window_global[i])) {
      out.active_spatial.push_back(i);
      continue;
    }
    const double to_occupied = std::abs(occupation[i] - 2.0);
    const double to_empty = std::abs(occupation[i]);
    const double deviation = std::min(to_occupied, to_empty);
    if (deviation > max_folded_occupation_deviation)
      throw std::invalid_argument(
          "SchriefferWolffPT2: folded external orbital " +
          std::to_string(window_global[i]) + " has reference occupation " +
          std::to_string(occupation[i]) +
          ", which deviates from an integer occupation by more than "
          "max_folded_occupation_deviation. Keep strongly correlated or "
          "open-shell orbitals in the active space P (p_indices), or raise "
          "the setting to accept the rounding.");
    if (deviation > out.worst_deviation) {
      out.worst_deviation = deviation;
      out.worst_occupation = occupation[i];
      out.worst_orbital = window_global[i];
    }
    (to_occupied <= to_empty ? out.inactive_spatial : out.virtual_spatial)
        .push_back(i);
  }
  if (out.active_spatial.empty())
    throw std::invalid_argument(
        "SchriefferWolffPT2: the kept active space P is empty.");

  // Rounding preserves the total, so the active electron count is fixed by
  // what the folded orbitals take. This is the count to hand the active-space
  // solver; it need not equal the reference occupation summed over P.
  const int folded_electrons =
      2 * static_cast<int>(out.inactive_spatial.size());
  out.active_electrons = window_electrons - folded_electrons;
  if (out.active_electrons < 0 ||
      out.active_electrons > 2 * static_cast<int>(out.active_spatial.size()))
    throw std::invalid_argument(
        "SchriefferWolffPT2: the rounded external occupation leaves an "
        "impossible active electron count; the active/external partition is "
        "inconsistent with the reference density.");

  double folded_occupation = 0.0;
  for (int i : out.inactive_spatial) folded_occupation += occupation[i];
  for (int i : out.virtual_spatial) folded_occupation += occupation[i];
  out.folded_charge_error = folded_electrons - folded_occupation;
  return out;
}

}  // namespace qdk::chemistry::algorithms::microsoft::swpt2
