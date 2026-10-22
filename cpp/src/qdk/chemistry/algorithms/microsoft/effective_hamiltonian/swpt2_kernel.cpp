// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

#include <btas/btas.h>

#include <SeQuant/core/eval/backends/btas/eval_expr.hpp>
#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <SeQuant/core/eval/eval.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/wick.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/spin.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::swpt2 {

namespace {
inline Eigen::Index n4(int n) {
  return static_cast<Eigen::Index>(n) * n * n * n;
}

// Smaller loops run serially; starting threads would dominate their cost.
constexpr Eigen::Index min_parallel_work = Eigen::Index{1} << 16;

// Chemist (pq|rs) with p, q of spin `left` and r, s of spin `right`.
double chemist(const TwoBodyBlocks& two_body, int left, int right, int p, int q,
               int r, int s, int norb) {
  if (left == right) return two_body[2 * left].get()(idx4(p, q, r, s, norb));
  return left == 0 ? two_body[1].get()(idx4(p, q, r, s, norb))
                   : two_body[1].get()(idx4(r, s, p, q, norb));
}

// Identical spin blocks make the operator spin free.
bool spin_symmetric(const std::array<Eigen::MatrixXd, 2>& one_body,
                    const TwoBodyBlocks& two_body,
                    const std::array<Eigen::MatrixXd, 2>& density) {
  const auto same = [](const Eigen::VectorXd& left,
                       const Eigen::VectorXd& right) {
    return &left == &right || left == right;
  };
  return one_body[0] == one_body[1] && density[0] == density[1] &&
         same(two_body[0], two_body[1]) && same(two_body[1], two_body[2]);
}
}  // namespace

// ===========================================================================
// Shared kernel foundations: denominator regularization, generalized Fock,
// and orbital rotations.
// ===========================================================================

double regularized_inverse(double delta, const RegularizerOptions& reg) {
  if (reg.sigma2 > 0.0) {
    if (delta == 0.0) return 0.0;  // the damped inverse vanishes in the limit
    // expm1 rather than 1-exp: the difference cancels catastrophically for
    // small sigma*delta^2, which is exactly where the regularizer works.
    return -std::expm1(-reg.sigma2 * delta * delta) / delta;
  }
  if (std::abs(delta) < bare_denominator_floor) return 0.0;
  return 1.0 / delta;
}

std::array<Eigen::MatrixXd, 2> generalized_fock_matrices(
    const std::array<Eigen::MatrixXd, 2>& one_body,
    const TwoBodyBlocks& two_body,
    const std::array<Eigen::MatrixXd, 2>& density, int norb) {
  bool consistent = true;
  for (int spin = 0; spin < 2; ++spin)
    consistent = consistent && one_body[spin].rows() == norb &&
                 one_body[spin].cols() == norb &&
                 density[spin].rows() == norb && density[spin].cols() == norb;
  for (const auto& block : two_body)
    consistent = consistent && block.get().size() == n4(norb);
  if (!consistent)
    throw std::invalid_argument(
        "generalized_fock_matrices: inconsistent tensor dimensions");

  using RowMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Index pairs = static_cast<Eigen::Index>(norb) * norb;
  // C-order (pq|rs) storage is a row-major matrix over the pairs (pq), (rs).
  const auto pair_matrix = [pairs](const Eigen::VectorXd& block) {
    return Eigen::Map<const RowMajorMatrix>(block.data(), pairs, pairs);
  };
  std::array<Eigen::VectorXd, 2> pair_density;
  for (int spin = 0; spin < 2; ++spin) {
    const RowMajorMatrix row_major = density[spin];
    pair_density[spin] =
        Eigen::Map<const Eigen::VectorXd>(row_major.data(), pairs);
  }
  const auto mixed = pair_matrix(two_body[1]);

  std::array<Eigen::MatrixXd, 2> fock = one_body;
  const bool spin_free = spin_symmetric(one_body, two_body, density);
  for (int spin = 0; spin < (spin_free ? 1 : 2); ++spin) {
    const auto same = pair_matrix(two_body[2 * spin]);
    // The beta-alpha Coulomb block is the transposed aabb block.
    const Eigen::VectorXd coulomb =
        same * pair_density[spin] +
        (spin == 0 ? Eigen::VectorXd(mixed * pair_density[1])
                   : Eigen::VectorXd(mixed.transpose() * pair_density[0]));
    RowMajorMatrix exchange = RowMajorMatrix::Zero(norb, norb);
#pragma omp parallel for schedule(static) if (n4(norb) >= min_parallel_work)
    for (int p = 0; p < norb; ++p)
      for (int r = 0; r < norb; ++r) {
        const Eigen::Index pair = static_cast<Eigen::Index>(p) * norb + r;
        for (int s = 0; s < norb; ++s) {
          const double weight = density[spin](r, s);
          // (pr|sq) is contiguous in q.
          const double* row = same.data() + pair * pairs + s * norb;
          for (int q = 0; q < norb; ++q) exchange(p, q) += weight * row[q];
        }
      }
    auto& block = fock[spin];
    block +=
        Eigen::Map<const RowMajorMatrix>(coulomb.data(), norb, norb) - exchange;
    block = 0.5 * (block + block.transpose()).eval();
  }
  if (spin_free) fock[1] = fock[0];
  return fock;
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
  return rotate_two_body(two_body, rotation, rotation, norb);
}

Eigen::VectorXd rotate_two_body(const Eigen::VectorXd& two_body,
                                const Eigen::MatrixXd& first_pair,
                                const Eigen::MatrixXd& second_pair, int norb) {
  for (const auto* rotation : {&first_pair, &second_pair})
    if (rotation->rows() != norb || rotation->cols() != norb)
      throw std::invalid_argument("rotate_two_body: inconsistent dimensions");
  if (two_body.size() != n4(norb))
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
    const auto& rotation = pass < 2 ? first_pair : second_pair;
    const Eigen::Map<const RowMajorMatrix> in(current.data(), n, n3);
    const Eigen::MatrixXd transformed = rotation.transpose() * in;
    Eigen::Map<RowMajorMatrix>(next.data(), n3, n) = transformed.transpose();
    current.swap(next);
  }
  return current;
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

namespace {
namespace sq = sequant;
using Tensor = btas::Tensor<double>;

struct Term {
  sq::ResultExpr result;
  int rank;
  int spin_case;
  bool folded;
};

struct Equations {
  std::shared_ptr<sq::IndexSpaceRegistry> spaces;
  std::vector<Term> terms;
  std::size_t projected_terms = 0;
};

struct Plan {
  sq::EvalNodeBTAS tree;
  int rank;
  int spin_case;
  bool folded;
};

struct Plans {
  std::vector<Plan> plans;
  std::size_t maximum_rank = 0;
};

// Group by output body rank, spin case and number of density contractions. The
// spin case is alpha or beta for one-body outputs and aaaa, aabb or bbbb for
// two-body outputs.
using EquationSums =
    std::map<std::tuple<int, int, int>, sq::HashingAccumulator>;

// Spin of output slot `axis`; aabb outputs order alpha before beta per pair.
bool output_is_beta(int rank, int spin_case, int axis) {
  if (rank == 1) return spin_case == 1;
  return spin_case == 1 ? axis % 2 == 1 : spin_case == 2;
}

sq::Index output_index(int rank, int spin_case, int axis) {
  // Ordinals 21..24 stay clear of the operand indices 1..8.
  const sq::Index index(L"u_" + std::to_wstring(21 + axis));
  return output_is_beta(rank, spin_case, axis)
             ? sq::mbpt::make_spinbeta(index)
             : sq::mbpt::make_spinalpha(index);
}

std::span<const sq::ExprPtr> summands(const sq::ExprPtr& expression) {
  if (expression->is<sq::Sum>()) {
    const auto& terms = expression->as<sq::Sum>().summands();
    return {terms.data(), terms.size()};
  }
  return {&expression, 1};
}

struct DensityPattern {
  double coefficient;
  std::vector<std::pair<int, int>> pairs;
  std::vector<int> remaining;
};

std::vector<DensityPattern> density_patterns() {
  auto context = sq::set_scoped_default_context(
      {.index_space_registry_shared_ptr =
           sq::mbpt::make_min_mr_spaces(sq::mbpt::SpinConvention::Legacy),
       .vacuum = sq::Vacuum::Physical});
  // Particle-hole reversal turns the three-body Gaussian fold into ordinary
  // Wick pairings of a a a a+ a+ a+.
  std::vector<sq::Index> slots;
  sq::ExprPtr operand;
  for (int axis = 0; axis < 6; ++axis) {
    slots.emplace_back(L"u_" + std::to_wstring(axis + 1));
    const auto ladder =
        axis < 3 ? sq::fannx(slots.back()) : sq::fcrex(slots.back());
    operand = operand ? operand * ladder : ladder;
  }
  const auto expansion =
      sq::FWickTheorem{operand}.full_contractions(false).compute();
  std::vector<DensityPattern> patterns;
  for (const auto& term : summands(expansion)) {
    if (!term->is<sq::Product>()) continue;
    const auto& product = term->as<sq::Product>();
    DensityPattern pattern{
        sq::Constant(product.scalar()).value<double>(), {}, {}};
    std::array<bool, 6> removed{};
    for (const auto& factor : product.factors()) {
      if (factor->is<sq::FNOperator>()) {
        const auto& residual = factor->as<sq::FNOperator>();
        // FNOperator stores annihilators in reverse literal ladder order.
        std::vector<int> order;
        for (const auto& op : residual.annihilators())
          order.push_back(std::find(slots.begin(), slots.end(), op.index()) -
                          slots.begin());
        std::reverse(order.begin(), order.end());
        for (const auto& op : residual.creators())
          order.push_back(std::find(slots.begin(), slots.end(), op.index()) -
                          slots.begin());
        for (std::size_t left = 0; left < order.size(); ++left)
          for (std::size_t right = left + 1; right < order.size(); ++right)
            if (order[left] > order[right])
              pattern.coefficient = -pattern.coefficient;
      }
      if (!factor->is<sq::Tensor>()) continue;
      const auto& delta = factor->as<sq::Tensor>();
      if (delta.label() != sq::reserved::overlap_label()) continue;
      const int left =
          std::find(slots.begin(), slots.end(), delta.bra()[0]) - slots.begin();
      const int right =
          std::find(slots.begin(), slots.end(), delta.ket()[0]) - slots.begin();
      pattern.pairs.emplace_back(std::min(left, right), std::max(left, right));
      removed[left] = removed[right] = true;
    }
    if (pattern.pairs.empty()) continue;
    std::sort(pattern.pairs.begin(), pattern.pairs.end());
    for (int axis = 0; axis < 6; ++axis)
      if (!removed[axis]) pattern.remaining.push_back(axis);
    patterns.push_back(std::move(pattern));
  }
  return patterns;
}

bool changes_occupation(const std::string& roles) {
  // Creator roles precede annihilator roles: i=inactive, u=active, a=virtual.
  const auto middle = roles.begin() + roles.size() / 2;
  for (const char role : {'i', 'a'})
    if (std::count(roles.begin(), middle, role) !=
        std::count(middle, roles.end(), role))
      return true;
  return false;
}

std::vector<std::string> off_diagonal_blocks(int rank) {
  std::vector<std::string> result{std::string{}};
  for (int axis = 0; axis < 2 * rank; ++axis) {
    std::vector<std::string> next;
    for (const auto& roles : result)
      for (const char role : {'i', 'u', 'a'}) next.push_back(roles + role);
    result = std::move(next);
  }
  std::erase_if(result, [rank](const auto& roles) {
    return !changes_occupation(roles) ||
           (rank == 2 && (roles[0] > roles[1] || roles[2] > roles[3]));
  });
  return result;
}

double block_normalization(const std::string& roles) {
  // Equal-role pairs retain the 1/2 factors; distinct-role permutations are
  // represented once by off_diagonal_blocks().
  return roles.size() == 2 ? 1.0
                           : (roles[0] == roles[1] ? 0.5 : 1.0) *
                                 (roles[2] == roles[3] ? 0.5 : 1.0);
}

sq::ExprPtr normal_ordered_operand(const std::string& roles, int offset,
                                   const wchar_t* label) {
  std::vector<sq::Index> bra, ket;
  for (std::size_t axis = 0; axis < roles.size(); ++axis) {
    auto index = sq::Index(std::wstring(1, roles[axis]) + L"_" +
                           std::to_wstring(offset + axis + 1));
    (axis < roles.size() / 2 ? bra : ket).push_back(index);
  }
  return sq::ex<sq::Tensor>(label, sq::bra(bra), sq::ket(ket)) *
         sq::ex<sq::FNOperator>(sq::cre(bra), sq::ann(ket));
}

void accumulate_projected_term(const sq::Product& product,
                               const std::vector<DensityPattern>& folding,
                               EquationSums& sums,
                               std::size_t& projected_terms) {
  std::vector<sq::Index> output;
  std::vector<sq::Tensor> sources;
  for (const auto& factor : product.factors()) {
    if (factor->is<sq::FNOperator>()) {
      const auto& residual = factor->as<sq::FNOperator>();
      for (const auto& op : residual.creators()) output.push_back(op.index());
      std::vector<sq::Index> annihilators;
      for (const auto& op : residual.annihilators())
        annihilators.push_back(op.index());
      output.insert(output.end(), annihilators.rbegin(), annihilators.rend());
    } else {
      sources.push_back(factor->as<sq::Tensor>());
    }
  }

  // Projection leaves only active residual operators; rank three is folded
  // here, symbolically.
  const int residual_rank = output.size() / 2;
  if (residual_rank > 3 ||
      std::any_of(output.begin(), output.end(), [](const auto& index) {
        return index.space().base_key()[0] != L'u';
      }))
    return;
  ++projected_terms;
  std::vector<DensityPattern> reductions;
  if (residual_rank == 3) {
    reductions = folding;
  } else {
    DensityPattern identity{1.0, {}, {}};
    for (int axis = 0; axis < 2 * residual_rank; ++axis)
      identity.remaining.push_back(axis);
    reductions.push_back(std::move(identity));
  }

  auto source_product = sq::ex<sq::Constant>(product.scalar());
  for (const auto& source : sources) {
    if (source.label() != L"S" && source.label() != L"V")
      throw std::runtime_error("SW-PT2: unreduced source contraction");
    source_product *=
        source.bra_rank() == 1
            ? sq::ex<sq::Tensor>(source)
            : sq::ex<sq::Tensor>(
                  source.label(), sq::bra(source.bra()), sq::ket(source.ket()),
                  sq::TensorSymmetries{.perm = sq::Symmetry::Antisymm});
  }

  // SeQuant enumerates the spin cases that conserve spin in every operand and
  // keeps the labels, so each traced term is one spin block.
  sq::container::svector<sq::container::svector<sq::Index>> output_groups;
  for (const auto& index : output) output_groups.push_back({index});
  auto traced = sq::mbpt::spintrace(source_product, output_groups,
                                    /*spinfree_index_spaces=*/false);
  sq::expand(traced);
  sq::rapid_simplify(traced);
  for (const auto& term : summands(traced)) {
    if (!term->is<sq::Product>()) continue;
    // Spin tracing may rename a dummy to the number of an output with the
    // other spin, so only once-occurring indices identify the outputs.
    std::map<sq::Index, int> occurrences;
    for (const auto& factor : term->as<sq::Product>().factors())
      for (const auto& index : factor->as<sq::Tensor>().const_braket_indices())
        ++occurrences[index];
    std::map<sq::Index, sq::Index> labelled;
    for (const auto& [index, count] : occurrences)
      if (count == 1) labelled.emplace(sq::mbpt::make_spinfree(index), index);
    const auto spin = [&](int axis) {
      return sq::mbpt::to_spin(labelled.at(output[axis]).space().qns()) ==
             sq::mbpt::Spin::beta;
    };

    for (const auto& reduction : reductions) {
      if (std::any_of(reduction.pairs.begin(), reduction.pairs.end(),
                      [&](const auto& pair) {
                        return spin(pair.first) != spin(pair.second);
                      }))
        continue;
      std::vector<int> target(reduction.remaining);
      const int rank = target.size() / 2;
      if (rank == 1 && spin(target[0]) != spin(target[1])) continue;
      double phase = reduction.coefficient;
      int spin_case = rank == 1 ? spin(target[0]) : 0;
      if (rank == 2) {
        if (spin(target[0]) == spin(target[1])) {
          spin_case = spin(target[0]) ? 2 : 0;
        } else {
          // aabb outputs put alpha first in both pairs; each swap flips sign.
          spin_case = 1;
          for (const int axis : {0, 2})
            if (spin(target[axis])) {
              std::swap(target[axis], target[axis + 1]);
              phase = -phase;
            }
        }
      }
      auto expression = sq::ex<sq::Constant>(phase) * term->clone();
      for (const auto& [creator, annihilator] : reduction.pairs)
        expression *= sq::ex<sq::Tensor>(
            L"D", sq::bra{labelled.at(output[annihilator])},
            sq::ket{labelled.at(output[creator])},
            sq::TensorSymmetries{.hermiticity = sq::Hermiticity::Hermitian});

      // Give equivalent free outputs the same names before combining terms.
      std::map<sq::Index, sq::Index> output_names;
      for (std::size_t axis = 0; axis < target.size(); ++axis)
        output_names.emplace(labelled.at(output[target[axis]]),
                             output_index(rank, spin_case, axis));
      expression->visit(
          [&](sq::ExprPtr& factor) {
            if (!factor->is<sq::Tensor>()) return;
            factor->as<sq::Tensor>().transform_indices(output_names);
            factor->as<sq::Tensor>().reset_tags();
          },
          true);
      sq::expand(expression);
      sq::rapid_simplify(expression);
      auto& sum = sums[{rank, spin_case, reduction.pairs.size()}];
      for (auto summand : summands(expression)) {
        sq::canonicalize(summand);
        sq::rapid_simplify(summand);
        sum.append(std::move(summand));
      }
    }
  }
}

Equations derive() {
  const auto folding = density_patterns();
  Equations equations;
  equations.spaces =
      sq::mbpt::make_min_mr_spaces(sq::mbpt::SpinConvention::Default);
  auto context = sq::set_scoped_default_context(
      {.index_space_registry_shared_ptr = equations.spaces,
       .vacuum = sq::Vacuum::SingleProduct});
  EquationSums sums;
  const std::array roles{off_diagonal_blocks(1), off_diagonal_blocks(2)};

  // Form 1/2 (S H_OD - H_OD S). Only pairs with no net inactive/virtual
  // occupation change can survive projection onto P.
  for (const int rank_s : {1, 2})
    for (const int rank_v : {1, 2})
      for (const auto& roles_s : roles[rank_s - 1])
        for (const auto& roles_v : roles[rank_v - 1]) {
          if (changes_occupation(
                  roles_s.substr(0, rank_s) + roles_v.substr(0, rank_v) +
                  roles_s.substr(rank_s) + roles_v.substr(rank_v)))
            continue;
          const auto generator = normal_ordered_operand(roles_s, 0, L"S");
          const auto coupling =
              normal_ordered_operand(roles_v, 2 * rank_s, L"V");
          for (const bool generator_first : {true, false}) {
            const double scale = (generator_first ? 0.5 : -0.5) *
                                 block_normalization(roles_s) *
                                 block_normalization(roles_v);
            auto wick =
                sq::FWickTheorem{sq::ex<sq::Constant>(scale) *
                                 (generator_first ? generator * coupling
                                                  : coupling * generator)};
            wick.full_contractions(false).use_topology(false);
            wick.set_nop_connections(std::vector<std::pair<int, int>>{{0, 1}});
            wick.set_external_indices(std::vector<sq::Index>{});
            const auto expansion = wick.compute(false, true);
            for (const auto& term : summands(expansion)) {
              if (!term->is<sq::Product>()) continue;
              accumulate_projected_term(term->as<sq::Product>(), folding, sums,
                                        equations.projected_terms);
            }
          }
        }

  for (auto& [key, sum] : sums) {
    const auto [rank, spin_case, density_order] = key;
    const auto expression = sum.make_expr(true);
    std::vector<sq::Index> bra, ket;
    for (int axis = 0; axis < rank; ++axis) {
      bra.push_back(output_index(rank, spin_case, axis));
      ket.push_back(output_index(rank, spin_case, rank + axis));
    }
    for (const auto& term : summands(expression)) {
      if (term->is_zero()) continue;
      auto result =
          rank == 0 ? sq::ResultExpr(sq::Variable(L"E"), term)
                    : sq::ResultExpr(
                          sq::Tensor(L"F", sq::bra(bra), sq::ket(ket)), term);
      equations.terms.push_back(
          {std::move(result), rank, spin_case, density_order != 0});
    }
  }
  return equations;
}

const Equations& reusable_equations() {
  static const Equations equations = derive();
  return equations;
}

Plans build_plans(const Equations& equations,
                  const std::array<std::size_t, 3>& extents, bool spin_free) {
  auto context = sq::set_scoped_default_context(
      {.index_space_registry_shared_ptr = equations.spaces,
       .vacuum = sq::Vacuum::SingleProduct});
  const auto extent = [&extents](const sq::Index& index) -> std::size_t {
    const wchar_t role = index.space().base_key()[0];
    return extents[role == L'i' ? 0 : role == L'u' ? 1 : 2];
  };
  Plans plans;
  for (const auto& term : equations.terms) {
    // Spin-free input needs only the alpha one-body and aabb outputs.
    if (spin_free && term.spin_case != (term.rank == 2 ? 1 : 0)) continue;
    auto result = term.result;
    result.expression() = result.expression()->clone();
    // Only density-folded products have more than two tensor factors.
    if (term.folded)
      sq::optimize(result,
                   {.objective_function = sq::ObjectiveFunction::DenseSpaceTime,
                    .idx_to_extent = extent});
    auto tree = sq::binarize<sq::EvalExprBTAS>(result);
    tree.visit([&](const auto& node) {
      plans.maximum_rank =
          std::max(plans.maximum_rank, node->canon_indices().size());
    });
    plans.plans.push_back(
        {std::move(tree), term.rank, term.spin_case, term.folded});
  }
  return plans;
}

std::shared_ptr<const Plans> reusable_plans(std::size_t inactive,
                                            std::size_t active,
                                            std::size_t virtuals,
                                            bool spin_free) {
  // Plans touching an empty role are skipped, so empty roles share extent one.
  const std::array<std::size_t, 3> extents{std::max<std::size_t>(inactive, 1),
                                           std::max<std::size_t>(active, 1),
                                           std::max<std::size_t>(virtuals, 1)};
  // Plan sets are large, so only the latest one is kept.
  static std::mutex mutex;
  static std::array<std::size_t, 3> cached_extents{};
  static bool cached_spin_free = false;
  static std::shared_ptr<const Plans> cached;
  const std::lock_guard lock(mutex);
  if (!cached || cached_extents != extents || cached_spin_free != spin_free) {
    cached.reset();
    cached = std::make_shared<const Plans>(
        build_plans(reusable_equations(), extents, spin_free));
    cached_extents = extents;
    cached_spin_free = spin_free;
  }
  return cached;
}
}  // namespace

DerivationStatistics downfold_statistics(std::size_t inactive,
                                         std::size_t active,
                                         std::size_t virtuals) {
  const auto& equations = reusable_equations();
  return {equations.projected_terms, equations.terms.size(),
          reusable_plans(inactive, active, virtuals, false)->maximum_rank};
}

DownfoldResult downfold(const std::array<Eigen::MatrixXd, 2>& one_body,
                        const TwoBodyBlocks& two_body,
                        const std::array<Eigen::VectorXd, 2>& energies,
                        const std::vector<int>& active,
                        const std::vector<int>& inactive,
                        const std::vector<int>& virtuals,
                        const RegularizerOptions& regularizer,
                        double core_energy,
                        const std::array<Eigen::MatrixXd, 2>& density) {
  const int norb = energies[0].size();
  const int active_count = active.size();
  const bool fold = density[0].size() != 0;
  bool consistent = norb > 0 && active_count > 0;
  for (int spin = 0; spin < 2; ++spin)
    consistent =
        consistent && energies[spin].size() == norb &&
        one_body[spin].rows() == norb && one_body[spin].cols() == norb &&
        (fold ? density[spin].rows() == norb && density[spin].cols() == norb
              : density[spin].size() == 0);
  for (const auto& block : two_body)
    consistent = consistent && block.get().size() == n4(norb);
  if (!consistent)
    throw std::invalid_argument(
        "SW-PT2: inconsistent spatial tensor dimensions");
  const std::map<wchar_t, const std::vector<int>*> roles{
      {L'i', &inactive}, {L'u', &active}, {L'a', &virtuals}};
  std::vector<char> orbital_roles(norb, 0);
  for (const auto& [role, orbitals] : roles)
    for (const int orbital : *orbitals) {
      if (orbital < 0 || orbital >= norb || orbital_roles[orbital] != 0)
        throw std::invalid_argument(
            "SW-PT2: invalid or repeated spatial role index");
      orbital_roles[orbital] = static_cast<char>(role);
    }
  if (std::find(orbital_roles.begin(), orbital_roles.end(), 0) !=
      orbital_roles.end())
    throw std::invalid_argument(
        "SW-PT2: spatial roles do not cover the window");
  // A spin-free operator needs only its alpha one-body and aabb blocks.
  const bool spin_free =
      spin_symmetric(one_body, two_body, density) && energies[0] == energies[1];
  const auto integral = [&](int left, int right, int p, int q, int r, int s) {
    return chemist(two_body, left, right, p, q, r, s, norb);
  };
  const auto inverse_two_body = [&](double first, double second, double third,
                                    double fourth) {
    const double creators = first + second;
    // Keep creator-first arithmetic at the bare cutoff; average annihilator
    // orders to preserve their exchange symmetry despite roundoff.
    return 0.5 * (regularized_inverse(creators - third - fourth, regularizer) +
                  regularized_inverse(creators - fourth - third, regularizer));
  };
  // Normal order against the filled inactive space, not the density used for
  // the generalized Fock. Each generator trace must be regularized separately.
  std::array<Eigen::MatrixXd, 2> normal_ordered_one_body = one_body;
  std::array<Eigen::MatrixXd, 2> generator_one_body = one_body;
  for (int spin = 0; spin < 2; ++spin) {
    const auto& eps = energies[spin];
    for (int row = 0; row < norb; ++row)
      for (int col = 0; col < norb; ++col) {
        generator_one_body[spin](row, col) *=
            regularized_inverse(eps(row) - eps(col), regularizer);
        for (const int occupied : inactive) {
          // Coulomb with both spins of the occupied orbital, exchange with one.
          double trace = 0.0, generator_trace = 0.0;
          for (int other = 0; other < 2; ++other) {
            const double coulomb =
                integral(spin, other, row, col, occupied, occupied);
            trace += coulomb;
            generator_trace +=
                coulomb * inverse_two_body(eps(row), energies[other](occupied),
                                           energies[other](occupied), eps(col));
          }
          const double exchange =
              integral(spin, spin, row, occupied, occupied, col);
          trace -= exchange;
          generator_trace -=
              exchange * inverse_two_body(eps(row), eps(occupied),
                                          eps(occupied), eps(col));
          normal_ordered_one_body[spin](row, col) += trace;
          generator_one_body[spin](row, col) += generator_trace;
        }
      }
  }

  // Start with P H_BD P; the contraction plans add 1/2 P [S, H_OD] P below.
  DownfoldResult result;
  result.scalar = core_energy;
  for (const int occupied : inactive)
    for (int spin = 0; spin < 2; ++spin) {
      result.scalar += one_body[spin](occupied, occupied);
      for (const int other : inactive) {
        for (int partner = 0; partner < 2; ++partner)
          result.scalar +=
              0.5 * integral(spin, partner, occupied, occupied, other, other);
        result.scalar -=
            0.5 * integral(spin, spin, occupied, other, other, occupied);
      }
    }
  for (int spin = 0; spin < 2; ++spin) {
    result.one_body[spin].resize(active_count, active_count);
    for (int row = 0; row < active_count; ++row)
      for (int col = 0; col < active_count; ++col)
        result.one_body[spin](row, col) =
            normal_ordered_one_body[spin](active[row], active[col]);
  }
  const auto active_block = [&](int left, int right) {
    Eigen::VectorXd block(n4(active_count));
    for (int first = 0; first < active_count; ++first)
      for (int second = 0; second < active_count; ++second)
        for (int third = 0; third < active_count; ++third)
          for (int fourth = 0; fourth < active_count; ++fourth)
            block(idx4(first, second, third, fourth, active_count)) =
                integral(left, right, active[first], active[second],
                         active[third], active[fourth]);
    return block;
  };
  result.two_body = active_block(0, 1);
  result.same_spin_two_body =
      spin_free ? std::array{result.two_body, result.two_body}
                : std::array{active_block(0, 0), active_block(1, 1)};
  if (inactive.empty() && virtuals.empty()) return result;

  // Diagnostics use raw couplings, before normal ordering or damping.
  double min_denominator = result.min_denominator;
  double max_amplitude = result.max_amplitude;
#pragma omp parallel for collapse(2) schedule(static) \
    reduction(min : min_denominator)                  \
    reduction(max : max_amplitude) if (n4(norb) >= min_parallel_work)
  for (int first = 0; first < norb; ++first)
    for (int second = 0; second < norb; ++second) {
      const auto track = [&](double coupling, double delta) {
        if (coupling == 0.0) return;
        min_denominator = std::min(min_denominator, std::abs(delta));
        max_amplitude =
            std::max(max_amplitude,
                     std::abs(coupling) /
                         std::max(std::abs(delta), bare_denominator_floor));
      };
      if (orbital_roles[first] != orbital_roles[second])
        for (int spin = 0; spin < (spin_free ? 1 : 2); ++spin)
          track(one_body[spin](first, second),
                energies[spin](first) - energies[spin](second));
      for (int third = 0; third < norb; ++third)
        for (int fourth = 0; fourth < norb; ++fourth) {
          if (!changes_occupation({orbital_roles[first], orbital_roles[second],
                                   orbital_roles[third],
                                   orbital_roles[fourth]}))
            continue;
          // Particles (first, third) and (second, fourth); beta-alpha repeats
          // alpha-beta with the particles swapped.
          const auto track_pair = [&](int spin, int partner) {
            const double direct =
                integral(spin, partner, first, third, second, fourth);
            const double delta =
                energies[spin](first) + energies[partner](second) -
                energies[spin](third) - energies[partner](fourth);
            track(spin == partner ? direct - integral(spin, spin, first, fourth,
                                                      second, third)
                                  : direct,
                  delta);
          };
          track_pair(0, 0);
          track_pair(0, 1);
          // Beta-beta repeats alpha-alpha for spin-free input.
          if (!spin_free) track_pair(1, 1);
        }
    }
  result.min_denominator = min_denominator;
  result.max_amplitude = max_amplitude;
  using OperandKey =
      std::tuple<std::wstring, std::vector<wchar_t>, std::vector<int>>;
  std::map<OperandKey, sq::ResultPtr> operand_cache;
  const auto evaluate_leaf =
      [&](const sq::EvalNodeBTAS& node) -> sq::ResultPtr {
    if (!node->is_tensor())
      return sq::eval_result<sq::ResultScalar<double>>(
          node->as_constant().value<double>());
    const auto& source = node->as_tensor();
    std::vector<wchar_t> block_roles;
    std::vector<int> spins;
    std::vector<const std::vector<int>*> orbitals;
    std::vector<std::size_t> extents;
    for (const auto& index : source.const_braket_indices()) {
      block_roles.push_back(index.space().base_key()[0]);
      spins.push_back(!spin_free && sq::mbpt::to_spin(index.space().qns()) ==
                                        sq::mbpt::Spin::beta);
      orbitals.push_back(roles.at(block_roles.back()));
      extents.push_back(orbitals.back()->size());
    }
    const OperandKey key{std::wstring(source.label()), block_roles, spins};
    if (operand_cache.contains(key)) return operand_cache.at(key);
    const bool is_generator = source.label() == L"S";
    const bool is_density = source.label() == L"D";
    const bool is_one_body = source.bra_rank() == 1;
    // Two-body slots (0, 2) and (1, 3) belong to the same particle.
    const int spin = spins[0], partner = is_one_body ? spin : spins[1];
    const auto& one_body_block =
        (is_generator ? generator_one_body : normal_ordered_one_body)[spin];
    const auto& eps = energies[spin];
    const auto& partner_eps = energies[partner];
    Tensor values{btas::Range{extents}};
    const auto size = static_cast<std::ptrdiff_t>(values.size());
#pragma omp parallel for schedule(static) if (size >= min_parallel_work)
    for (std::ptrdiff_t ordinal = 0; ordinal < size; ++ordinal) {
      // BTAS tensors are row-major.
      std::array<int, 4> orbital{};
      auto rest = static_cast<std::size_t>(ordinal);
      for (std::size_t axis = orbitals.size(); axis-- > 0;) {
        orbital[axis] = (*orbitals[axis])[rest % extents[axis]];
        rest /= extents[axis];
      }
      double value;
      if (is_density) {
        value = density[spin](orbital[1], orbital[0]);
      } else if (is_one_body) {
        value = one_body_block(orbital[0], orbital[1]);
      } else {
        // Symbolic (bra, ket) slots map to chemist (pq|rs) order.
        value = integral(spin, partner, orbital[0], orbital[2], orbital[1],
                         orbital[3]);
        if (is_generator)
          value *= inverse_two_body(eps(orbital[0]), partner_eps(orbital[1]),
                                    eps(orbital[2]), partner_eps(orbital[3]));
      }
      values.data()[ordinal] = value;
    }
    auto value =
        sq::eval_result<sq::ResultTensorBTAS<Tensor>>(std::move(values));
    operand_cache.emplace(key, value);
    return value;
  };
  // SeQuant's BTAS evaluation updates a process-wide label map, so plans run
  // serially.
  const auto plans = reusable_plans(inactive.size(), active.size(),
                                    virtuals.size(), spin_free);
  for (const auto& plan : plans->plans) {
    if (plan.folded && !fold) continue;
    bool empty = false;
    plan.tree.visit([&](const auto& node) {
      if (!node.leaf() || !node->is_tensor()) return;
      for (const auto& index : node->as_tensor().const_braket_indices())
        empty = empty || roles.at(index.space().base_key()[0])->empty();
    });
    if (empty) continue;
    const auto value = sq::evaluate(plan.tree, evaluate_leaf);
    if (plan.rank == 0) {
      result.scalar += value->get<double>();
      continue;
    }
    const auto& values = value->get<Tensor>();
    // BTAS stores canonical axes, which need not match the requested output.
    std::vector<std::size_t> output_axes;
    for (const auto& index : plan.tree->as_tensor().const_braket_indices())
      output_axes.push_back(std::find(plan.tree->canon_indices().begin(),
                                      plan.tree->canon_indices().end(), index) -
                            plan.tree->canon_indices().begin());
    for (const auto& index : values.range()) {
      const auto at = [&](int axis) { return index[output_axes[axis]]; };
      if (plan.rank == 1) {
        result.one_body[plan.spin_case](at(0), at(1)) += values(index);
        continue;
      }
      // Chemist storage reverses the annihilator pair, giving the minus sign.
      if (plan.spin_case == 1) {
        result.two_body(idx4(at(0), at(2), at(1), at(3), active_count)) -=
            values(index);
        continue;
      }
      // A same-spin block also takes the copy with the particles exchanged.
      auto& block = result.same_spin_two_body[plan.spin_case / 2];
      block(idx4(at(0), at(2), at(1), at(3), active_count)) -= values(index);
      block(idx4(at(1), at(3), at(0), at(2), active_count)) -= values(index);
    }
  }
  if (spin_free) {
    result.one_body[1] = result.one_body[0];
    result.same_spin_two_body = {result.two_body, result.two_body};
  }
  return result;
}

}  // namespace qdk::chemistry::algorithms::microsoft::swpt2
