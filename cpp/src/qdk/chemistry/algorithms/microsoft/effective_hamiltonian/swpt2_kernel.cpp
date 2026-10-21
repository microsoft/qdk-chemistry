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

Eigen::MatrixXd generalized_fock_matrix(const Eigen::MatrixXd& h1,
                                        const Eigen::VectorXd& two_body,
                                        const Eigen::MatrixXd& density,
                                        int norb) {
  if (h1.rows() != norb || h1.cols() != norb || density.rows() != norb ||
      density.cols() != norb || two_body.size() != n4(norb))
    throw std::invalid_argument(
        "generalized_fock_matrix: inconsistent tensor dimensions");

  Eigen::MatrixXd fock = h1;
#pragma omp parallel for collapse(2) \
    schedule(static) if (n4(norb) >= min_parallel_work)
  for (int p = 0; p < norb; ++p)
    for (int q = 0; q < norb; ++q) {
      double value = fock(p, q);
      for (int r = 0; r < norb; ++r)
        for (int s = 0; s < norb; ++s)
          value += density(r, s) * (two_body(idx4(p, q, r, s, norb)) -
                                    0.5 * two_body(idx4(p, r, s, q, norb)));
      fock(p, q) = value;
    }
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
  int spin;
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
  int spin;
  bool folded;
};

struct Plans {
  std::vector<Plan> plans;
  std::size_t maximum_rank = 0;
};

// Group by output body rank, one-body spin, and number of density contractions.
using EquationSums =
    std::map<std::tuple<int, int, int>, sq::HashingAccumulator>;

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

  std::vector<sq::Index> indices;
  for (const auto& source : sources) {
    if (source.label() != L"S" && source.label() != L"V")
      throw std::runtime_error("SW-PT2: unreduced source contraction");
    for (const auto& index : source.const_braket_indices())
      if (std::find(indices.begin(), indices.end(), index) == indices.end())
        indices.push_back(index);
  }

  // Resolve alpha/beta assignments symbolically; all numerical leaves remain
  // spatial tensors. Each operand must conserve its spin population.
  for (unsigned assignment = 0; assignment < (1u << indices.size());
       ++assignment) {
    const auto spin = [&](const sq::Index& index) {
      return static_cast<int>(
          (assignment >> (std::find(indices.begin(), indices.end(), index) -
                          indices.begin())) &
          1u);
    };
    if (std::any_of(sources.begin(), sources.end(), [&](const auto& source) {
          int change = 0;
          for (const auto& index : source.bra()) change += spin(index);
          for (const auto& index : source.ket()) change -= spin(index);
          return change != 0;
        }))
      continue;
    std::map<sq::Index, sq::Index> spin_indices;
    for (const auto& index : indices)
      spin_indices.emplace(index, spin(index) == 0
                                      ? sq::mbpt::make_spinalpha(index)
                                      : sq::mbpt::make_spinbeta(index));
    auto expression = sq::ex<sq::Constant>(product.scalar());
    for (auto source : sources) {
      source.transform_indices(spin_indices);
      source.reset_tags();
      if (source.bra_rank() == 1) {
        source.set_label(std::wstring(source.label()) +
                         (sq::mbpt::to_spin(source.bra()[0].space().qns()) ==
                                  sq::mbpt::Spin::beta
                              ? L"1"
                              : L"0"));
        expression *= sq::ex<sq::Tensor>(source);
      } else {
        expression *= sq::mbpt::expand_antisymm(sq::Tensor(
            source.label(), sq::bra(source.bra()), sq::ket(source.ket()),
            sq::TensorSymmetries{.perm = sq::Symmetry::Antisymm}));
      }
    }
    sq::expand(expression);
    sq::rapid_simplify(expression);
    const auto spatial_coefficient = sq::mbpt::remove_spin(expression);

    for (const auto& reduction : reductions) {
      if (std::any_of(reduction.pairs.begin(), reduction.pairs.end(),
                      [&](const auto& pair) {
                        return spin(output[pair.first]) !=
                               spin(output[pair.second]);
                      }))
        continue;
      std::vector<sq::Index> target;
      for (const int axis : reduction.remaining) target.push_back(output[axis]);
      const int rank = target.size() / 2;
      double phase = reduction.coefficient;
      if (rank == 1 && spin(target[0]) != spin(target[1])) continue;
      if (rank == 2) {
        // Store only the alpha-beta two-body channel; antisymmetry fixes the
        // signs when a creator or annihilator pair needs to be swapped.
        if (spin(target[0]) == spin(target[1]) ||
            spin(target[2]) == spin(target[3]))
          continue;
        for (const int axis : {0, 2})
          if (spin(target[axis]) == 1) {
            std::swap(target[axis], target[axis + 1]);
            phase = -phase;
          }
      }
      expression = sq::ex<sq::Constant>(phase) * spatial_coefficient->clone();
      for (const auto& [creator, annihilator] : reduction.pairs)
        expression *= sq::ex<sq::Tensor>(
            L"D", sq::bra{output[annihilator]}, sq::ket{output[creator]},
            sq::TensorSymmetries{.hermiticity = sq::Hermiticity::Hermitian});

      // Give equivalent free outputs the same names before combining terms.
      // Ordinals 21..24 stay clear of the operand indices 1..8.
      std::map<sq::Index, sq::Index> output_names;
      for (std::size_t axis = 0; axis < target.size(); ++axis)
        output_names.emplace(target[axis],
                             sq::Index(L"u_" + std::to_wstring(21 + axis)));
      expression->visit(
          [&](sq::ExprPtr& factor) {
            if (!factor->is<sq::Tensor>()) return;
            factor->as<sq::Tensor>().transform_indices(output_names);
            factor->as<sq::Tensor>().reset_tags();
          },
          true);
      sq::expand(expression);
      sq::rapid_simplify(expression);
      auto& sum =
          sums[{rank, rank == 1 ? spin(target[0]) : 0, reduction.pairs.size()}];
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
    const auto [rank, spin, density_order] = key;
    const auto expression = sum.make_expr(true);
    std::vector<sq::Index> bra, ket;
    for (int axis = 0; axis < rank; ++axis) {
      bra.emplace_back(L"u_" + std::to_wstring(21 + axis));
      ket.emplace_back(L"u_" + std::to_wstring(21 + rank + axis));
    }
    for (const auto& term : summands(expression)) {
      if (term->is_zero()) continue;
      auto result =
          rank == 0 ? sq::ResultExpr(sq::Variable(L"E"), term)
                    : sq::ResultExpr(
                          sq::Tensor(L"F", sq::bra(bra), sq::ket(ket)), term);
      equations.terms.push_back(
          {std::move(result), rank, spin, density_order != 0});
    }
  }
  return equations;
}

const Equations& reusable_equations() {
  static const Equations equations = derive();
  return equations;
}

Plans build_plans(const Equations& equations,
                  const std::array<std::size_t, 3>& extents) {
  auto context = sq::set_scoped_default_context(
      {.index_space_registry_shared_ptr = equations.spaces,
       .vacuum = sq::Vacuum::SingleProduct});
  const auto extent = [&extents](const sq::Index& index) -> std::size_t {
    const wchar_t role = index.space().base_key()[0];
    return extents[role == L'i' ? 0 : role == L'u' ? 1 : 2];
  };
  Plans plans;
  for (const auto& term : equations.terms) {
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
    plans.plans.push_back({std::move(tree), term.rank, term.spin, term.folded});
  }
  return plans;
}

std::shared_ptr<const Plans> reusable_plans(std::size_t inactive,
                                            std::size_t active,
                                            std::size_t virtuals) {
  // Plans touching an empty role are skipped, so empty roles share extent one.
  const std::array<std::size_t, 3> extents{std::max<std::size_t>(inactive, 1),
                                           std::max<std::size_t>(active, 1),
                                           std::max<std::size_t>(virtuals, 1)};
  // Plan sets are large, so only the latest extents are kept.
  static std::mutex mutex;
  static std::array<std::size_t, 3> cached_extents{};
  static std::shared_ptr<const Plans> cached;
  const std::lock_guard lock(mutex);
  if (!cached || cached_extents != extents) {
    cached.reset();
    cached = std::make_shared<const Plans>(
        build_plans(reusable_equations(), extents));
    cached_extents = extents;
  }
  return cached;
}
}  // namespace

DerivationStatistics downfold_statistics(std::size_t inactive,
                                         std::size_t active,
                                         std::size_t virtuals) {
  const auto& equations = reusable_equations();
  return {equations.projected_terms, equations.terms.size(),
          reusable_plans(inactive, active, virtuals)->maximum_rank};
}

DownfoldResult downfold(
    const Eigen::MatrixXd& one_alpha, const Eigen::MatrixXd& one_beta,
    const Eigen::VectorXd& integrals, const Eigen::VectorXd& energies,
    const std::vector<int>& active, const std::vector<int>& inactive,
    const std::vector<int>& virtuals, const RegularizerOptions& regularizer,
    double core_energy, const Eigen::MatrixXd& reference_density) {
  const int norb = energies.size();
  const int active_count = active.size();
  const Eigen::Index tensor_size =
      static_cast<Eigen::Index>(norb) * norb * norb * norb;
  if (norb <= 0 || active_count == 0 || one_alpha.rows() != norb ||
      one_alpha.cols() != norb || one_beta.rows() != norb ||
      one_beta.cols() != norb || integrals.size() != tensor_size ||
      (reference_density.size() != 0 &&
       (reference_density.rows() != norb || reference_density.cols() != norb)))
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
  const auto integral = [&](int first, int second, int third, int fourth) {
    return integrals(idx4(first, second, third, fourth, norb));
  };
  const auto inverse_two_body = [&](int first, int second, int third,
                                    int fourth) {
    const double creators = energies(first) + energies(second);
    // Keep creator-first arithmetic at the bare cutoff; average annihilator
    // orders to preserve their exchange symmetry despite roundoff.
    return 0.5 *
           (regularized_inverse(creators - energies(third) - energies(fourth),
                                regularizer) +
            regularized_inverse(creators - energies(fourth) - energies(third),
                                regularizer));
  };
  const std::array<Eigen::Ref<const Eigen::MatrixXd>, 2> one_body{one_alpha,
                                                                  one_beta};
  // Normal order against the filled inactive space, not the density used for
  // the generalized Fock. Each generator trace must be regularized separately.
  std::array<Eigen::MatrixXd, 2> normal_ordered_one_body{one_alpha, one_beta};
  std::array<Eigen::MatrixXd, 2> generator_one_body{one_alpha, one_beta};
  for (int row = 0; row < norb; ++row)
    for (int col = 0; col < norb; ++col) {
      for (auto& block : generator_one_body)
        block(row, col) *=
            regularized_inverse(energies(row) - energies(col), regularizer);
      for (const int occupied : inactive) {
        const double trace = 2 * integral(row, col, occupied, occupied) -
                             integral(row, occupied, occupied, col);
        for (int spin = 0; spin < 2; ++spin) {
          normal_ordered_one_body[spin](row, col) += trace;
          generator_one_body[spin](row, col) +=
              trace * inverse_two_body(row, occupied, occupied, col);
        }
      }
    }

  // Start with P H_BD P; the contraction plans add 1/2 P [S, H_OD] P below.
  DownfoldResult result;
  result.scalar = core_energy;
  for (const int occupied : inactive) {
    result.scalar +=
        one_body[0](occupied, occupied) + one_body[1](occupied, occupied);
    for (const int other : inactive)
      result.scalar += 2 * integral(occupied, occupied, other, other) -
                       integral(occupied, other, other, occupied);
  }
  for (int spin = 0; spin < 2; ++spin) {
    result.one_body[spin].resize(active_count, active_count);
    for (int row = 0; row < active_count; ++row)
      for (int col = 0; col < active_count; ++col)
        result.one_body[spin](row, col) =
            normal_ordered_one_body[spin](active[row], active[col]);
  }
  result.two_body.resize(n4(active_count));
  for (int first = 0; first < active_count; ++first)
    for (int second = 0; second < active_count; ++second)
      for (int third = 0; third < active_count; ++third)
        for (int fourth = 0; fourth < active_count; ++fourth)
          result.two_body(idx4(first, second, third, fourth, active_count)) =
              integral(active[first], active[second], active[third],
                       active[fourth]);
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
        for (const auto& block : one_body)
          track(block(first, second), energies(first) - energies(second));
      for (int third = 0; third < norb; ++third)
        for (int fourth = 0; fourth < norb; ++fourth) {
          if (!changes_occupation({orbital_roles[first], orbital_roles[second],
                                   orbital_roles[third],
                                   orbital_roles[fourth]}))
            continue;
          const double direct = integral(first, third, second, fourth);
          const double delta = energies(first) + energies(second) -
                               energies(third) - energies(fourth);
          track(direct, delta);
          track(direct - integral(first, fourth, second, third), delta);
        }
    }
  result.min_denominator = min_denominator;
  result.max_amplitude = max_amplitude;
  using OperandKey = std::tuple<std::wstring, std::vector<wchar_t>>;
  std::map<OperandKey, sq::ResultPtr> operand_cache;
  const auto evaluate_leaf =
      [&](const sq::EvalNodeBTAS& node) -> sq::ResultPtr {
    if (!node->is_tensor())
      return sq::eval_result<sq::ResultScalar<double>>(
          node->as_constant().value<double>());
    const auto& source = node->as_tensor();
    std::vector<wchar_t> block_roles;
    std::vector<const std::vector<int>*> orbitals;
    std::vector<std::size_t> extents;
    for (const auto& index : source.const_braket_indices()) {
      block_roles.push_back(index.space().base_key()[0]);
      orbitals.push_back(roles.at(block_roles.back()));
      extents.push_back(orbitals.back()->size());
    }
    const OperandKey key{std::wstring(source.label()), block_roles};
    if (operand_cache.contains(key)) return operand_cache.at(key);
    const bool is_generator = source.label()[0] == L'S';
    const bool is_density = source.label() == L"D";
    const bool is_one_body = source.bra_rank() == 1;
    const auto& one_body_block =
        (is_generator ? generator_one_body
                      : normal_ordered_one_body)[source.label().back() == L'1'];
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
        // D is spin traced; a spin-diagonal Gaussian contraction uses D/2.
        value = 0.5 * reference_density(orbital[1], orbital[0]);
      } else if (is_one_body) {
        value = one_body_block(orbital[0], orbital[1]);
      } else {
        // Symbolic (bra, ket) slots map to spatial chemist (pq|rs) order.
        value = integral(orbital[0], orbital[2], orbital[1], orbital[3]);
        if (is_generator)
          value *=
              inverse_two_body(orbital[0], orbital[1], orbital[2], orbital[3]);
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
  const auto plans =
      reusable_plans(inactive.size(), active.size(), virtuals.size());
  for (const auto& plan : plans->plans) {
    if (plan.folded && reference_density.size() == 0) continue;
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
      if (plan.rank == 1) {
        result.one_body[plan.spin](index[output_axes[0]],
                                   index[output_axes[1]]) += values(index);
      } else {
        // Chemist storage reverses the annihilator pair, giving the minus sign.
        result.two_body(idx4(index[output_axes[0]], index[output_axes[2]],
                             index[output_axes[1]], index[output_axes[3]],
                             active_count)) -= values(index);
      }
    }
  }
  return result;
}

}  // namespace qdk::chemistry::algorithms::microsoft::swpt2
