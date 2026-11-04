// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "swpt2_kernel.hpp"

#include <btas/btas.h>
#include <qdk/chemistry/scf/util/blas_threads.h>

#include <SeQuant/core/eval/backends/btas/eval_expr.hpp>
#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <SeQuant/core/eval/eval.hpp>
#include <SeQuant/core/index_basis_registry.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/wick.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/spin.hpp>
#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
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

constexpr Eigen::Index min_parallel_work = Eigen::Index{1} << 16;

// Chemist (pq|rs) with p, q of spin `left` and r, s of spin `right`.
double chemist(const TwoBodyBlocks& two_body, int left, int right, int p, int q,
               int r, int s, int norb) {
  if (left == right) return two_body[2 * left].get()(idx4(p, q, r, s, norb));
  return left == 0 ? two_body[1].get()(idx4(p, q, r, s, norb))
                   : two_body[1].get()(idx4(r, s, p, q, norb));
}
}  // namespace

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
    const std::array<Eigen::MatrixXd, 2>& density, int norb, bool spin_free) {
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
  for (int spin = 0; spin < (spin_free ? 1 : 2); ++spin) {
    const auto same = pair_matrix(two_body[2 * spin]);
    // Spin-free input has equal same-spin and opposite-spin Coulomb terms, so
    // one is formed and doubled.
    Eigen::VectorXd coulomb(pairs);
#pragma omp parallel for schedule(static) if (n4(norb) >= min_parallel_work)
    for (Eigen::Index pair = 0; pair < pairs; ++pair) {
      const double own = same.row(pair).dot(pair_density[spin]);
      coulomb(pair) = spin_free   ? 2.0 * own
                      : spin == 0 ? own + mixed.row(pair).dot(pair_density[1])
                                  : own;
    }
    if (spin == 1) {
      // The beta-alpha Coulomb block is the transposed aabb block, accumulated
      // over column ranges so that each task writes its own elements.
      constexpr Eigen::Index width = 256;
      const Eigen::Index ranges = (pairs + width - 1) / width;
#pragma omp parallel for schedule(static) if (n4(norb) >= min_parallel_work)
      for (Eigen::Index range = 0; range < ranges; ++range) {
        const Eigen::Index begin = range * width;
        const Eigen::Index count = std::min(width, pairs - begin);
        for (Eigen::Index row = 0; row < pairs; ++row)
          coulomb.segment(begin, count) +=
              pair_density[0](row) *
              mixed.row(row).segment(begin, count).transpose();
      }
    }
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
  Eigen::MatrixXd rotation =
      Eigen::MatrixXd::Identity(fock.rows(), fock.cols());
  for (const auto& block : blocks) {
    if (block.size() < 2) continue;
    const Eigen::MatrixXd sub = fock(block, block);
    Eigen::MatrixXd off_diagonal = sub;
    off_diagonal.diagonal().setZero();
    if (off_diagonal.cwiseAbs().maxCoeff() <= tolerance) continue;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        0.5 * (sub + sub.transpose()).eval());
    if (solver.info() != Eigen::Success)
      throw std::runtime_error(
          "semicanonical_rotation: Fock block diagonalization failed");
    rotation(block, block) = solver.eigenvectors();
  }
  return rotation;
}

Eigen::MatrixXd rotate_one_body(const Eigen::MatrixXd& one_body,
                                const Eigen::MatrixXd& rotation) {
  return rotation.transpose() * one_body * rotation;
}

Eigen::VectorXd rotate_two_body(const Eigen::VectorXd& two_body,
                                const Eigen::MatrixXd& rotation, int norb) {
  return rotate_two_body(two_body, rotation, rotation, norb);
}

Eigen::VectorXd rotate_two_body(const Eigen::VectorXd& two_body,
                                const Eigen::MatrixXd& first_pair,
                                const Eigen::MatrixXd& second_pair, int norb) {
  using RowMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Index n = norb;
  const Eigen::Index n3 = n * n * n;
  // Each pass transforms the leading index and moves it to the back, so four
  // passes restore (p, q, r, s) order. The column-major (n, n^3) product
  // already has the moved index last, so it needs no transpose.
  constexpr Eigen::Index panel = 1024;
  const Eigen::Index panels = (n3 + panel - 1) / panel;
  Eigen::VectorXd current(n4(norb));
  Eigen::VectorXd next(n4(norb));
  for (int pass = 0; pass < 4; ++pass) {
    const Eigen::MatrixXd transposed =
        (pass < 2 ? first_pair : second_pair).transpose();
    const Eigen::Map<const RowMajorMatrix> in(
        pass == 0 ? two_body.data() : current.data(), n, n3);
    Eigen::Map<Eigen::MatrixXd> out(next.data(), n, n3);
#pragma omp parallel for schedule(static) if (n4(norb) >= min_parallel_work)
    for (Eigen::Index index = 0; index < panels; ++index) {
      const Eigen::Index begin = index * panel;
      const Eigen::Index width = std::min(panel, n3 - begin);
      out.middleCols(begin, width).noalias() =
          transposed * in.middleCols(begin, width);
    }
    current.swap(next);
  }
  return current;
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
  std::shared_ptr<sq::IndexBasisRegistry> spaces;
  std::vector<Term> terms;
};

struct Plan {
  sq::EvalNodeBTAS tree;
  int rank;
  int spin_case;
  bool folded;
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
      {.index_basis_registry_shared_ptr =
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
                               EquationSums& sums) {
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
      {.index_basis_registry_shared_ptr = equations.spaces,
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
            const auto expansion = wick.compute(false, true);
            for (const auto& term : summands(expansion)) {
              if (!term->is<sq::Product>()) continue;
              accumulate_projected_term(term->as<sq::Product>(), folding, sums);
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

std::vector<Plan> build_plans(const Equations& equations,
                              const std::array<std::size_t, 3>& extents,
                              bool spin_free) {
  auto context = sq::set_scoped_default_context(
      {.index_basis_registry_shared_ptr = equations.spaces,
       .vacuum = sq::Vacuum::SingleProduct});
  const auto extent = [&extents](const sq::Index& index) -> std::size_t {
    const wchar_t role = index.space().base_key()[0];
    return extents[role == L'i' ? 0 : role == L'u' ? 1 : 2];
  };
  std::vector<Plan> plans;
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
    plans.push_back({std::move(tree), term.rank, term.spin_case, term.folded});
  }
  return plans;
}

// Couplings at or below this are numerical zeros of symmetry-forbidden
// integrals and stay out of the diagnostics.
constexpr double negligible_coupling = 1e-12;

// Raw (undamped) amplitude, with the denominator floored at the bare cutoff.
double raw_amplitude(double coupling, double delta) {
  return std::abs(coupling) / std::max(std::abs(delta), bare_denominator_floor);
}

// Inputs shared by the steps of one downfold call.
struct Inputs {
  const TwoBodyBlocks& two_body;
  const std::vector<int>& active;
  const std::vector<int>& inactive;
  const std::vector<int>& virtuals;
  const RegularizerOptions& regularizer;
  const std::array<Eigen::MatrixXd, 2>& density;
  bool spin_free;
  int norb;
  std::array<Eigen::VectorXd, 2> energies;

  const std::vector<int>& orbitals(wchar_t role) const {
    return role == L'i' ? inactive : role == L'u' ? active : virtuals;
  }
  double integral(int left, int right, int p, int q, int r, int s) const {
    return chemist(two_body, left, right, p, q, r, s, norb);
  }
  // Summing creators and annihilators separately makes the denominator
  // exactly antisymmetric under the adjoint and symmetric under exchange,
  // so S stays anti-Hermitian even at the bare cutoff.
  static double denominator(double first, double second, double third,
                            double fourth) {
    return (first + second) - (third + fourth);
  }
  double inverse_two_body(double first, double second, double third,
                          double fourth) const {
    return regularized_inverse(denominator(first, second, third, fourth),
                               regularizer);
  }
};

struct OneBodyOperators {
  std::array<Eigen::MatrixXd, 2> normal_ordered, generator;
};

// Normal order against the filled inactive space, not the density used for
// the generalized Fock. Each generator trace must be regularized separately.
OneBodyOperators one_body_operators(
    const Inputs& in, const std::array<Eigen::MatrixXd, 2>& one_body) {
  OneBodyOperators out{one_body, one_body};
  const int norb = in.norb;
  for (int spin = 0; spin < 2; ++spin) {
    const auto& eps = in.energies[spin];
#pragma omp parallel for schedule(static) if (n4(norb) >= min_parallel_work)
    for (int row = 0; row < norb; ++row)
      for (int col = 0; col < norb; ++col) {
        out.generator[spin](row, col) *=
            regularized_inverse(eps(row) - eps(col), in.regularizer);
        for (const int occupied : in.inactive) {
          // Coulomb with both spins of the occupied orbital, exchange with one.
          double trace = 0.0, generator_trace = 0.0;
          for (int other = 0; other < 2; ++other) {
            const double coulomb =
                in.integral(spin, other, row, col, occupied, occupied);
            trace += coulomb;
            generator_trace +=
                coulomb *
                in.inverse_two_body(eps(row), in.energies[other](occupied),
                                    in.energies[other](occupied), eps(col));
          }
          const double exchange =
              in.integral(spin, spin, row, occupied, occupied, col);
          trace -= exchange;
          generator_trace -=
              exchange * in.inverse_two_body(eps(row), eps(occupied),
                                             eps(occupied), eps(col));
          out.normal_ordered[spin](row, col) += trace;
          out.generator[spin](row, col) += generator_trace;
        }
      }
  }
  return out;
}

// P H_BD P: the energy of the filled inactive space and the active blocks of
// the normal-ordered operator.
DownfoldResult block_diagonal_part(
    const Inputs& in, const std::array<Eigen::MatrixXd, 2>& one_body,
    const std::array<Eigen::MatrixXd, 2>& normal_ordered, double core_energy) {
  DownfoldResult result;
  result.scalar = core_energy;
  for (const int occupied : in.inactive)
    for (int spin = 0; spin < 2; ++spin) {
      result.scalar += one_body[spin](occupied, occupied);
      for (const int other : in.inactive) {
        for (int partner = 0; partner < 2; ++partner)
          result.scalar += 0.5 * in.integral(spin, partner, occupied, occupied,
                                             other, other);
        result.scalar -=
            0.5 * in.integral(spin, spin, occupied, other, other, occupied);
      }
    }
  for (int spin = 0; spin < 2; ++spin)
    result.one_body[spin] = normal_ordered[spin](in.active, in.active);
  const int count = static_cast<int>(in.active.size());
  const auto active_block = [&](int left, int right) {
    Eigen::VectorXd block(n4(count));
    for (int first = 0; first < count; ++first)
      for (int second = 0; second < count; ++second)
        for (int third = 0; third < count; ++third)
          for (int fourth = 0; fourth < count; ++fourth)
            block(idx4(first, second, third, fourth, count)) =
                in.integral(left, right, in.active[first], in.active[second],
                            in.active[third], in.active[fourth]);
    return block;
  };
  result.two_body = active_block(0, 1);
  // Spin-free same-spin blocks are copies of the final aabb block.
  if (!in.spin_free)
    result.same_spin_two_body = {active_block(0, 0), active_block(1, 1)};
  return result;
}

// One-body diagnostics: the single excitations out of the reference external
// occupation (active or virtual created, inactive or active annihilated),
// with the generalized-Fock coupling, which includes inactive spectators.
// Double excitations are recorded while their generator operands are built.
void track_single_couplings(
    const Inputs& in, const std::array<Eigen::MatrixXd, 2>& normal_ordered,
    const std::array<Eigen::MatrixXd, 2>& fock, DownfoldResult& result) {
  std::vector<char> active(in.norb, 0);
  for (const int orbital : in.active) active[orbital] = 1;
  std::vector<int> creatable(in.active), annihilable(in.inactive);
  creatable.insert(creatable.end(), in.virtuals.begin(), in.virtuals.end());
  annihilable.insert(annihilable.end(), in.active.begin(), in.active.end());
  for (const int creator : creatable)
    for (const int annihilator : annihilable) {
      // Active-active excitations leave the external occupation unchanged.
      if (active[creator] && active[annihilator]) continue;
      for (int spin = 0; spin < (in.spin_free ? 1 : 2); ++spin) {
        const double delta =
            in.energies[spin](creator) - in.energies[spin](annihilator);
        // The generator's one-body coefficient is the normal-ordered one.
        if (std::abs(normal_ordered[spin](creator, annihilator)) >
            negligible_coupling)
          result.min_denominator =
              std::min(result.min_denominator, std::abs(delta));
        result.max_amplitude =
            std::max(result.max_amplitude,
                     raw_amplitude(fock[spin](creator, annihilator), delta));
      }
    }
}

// Plans whose operands touch an empty role, and folded plans without a
// density, contribute nothing.
std::vector<const Plan*> select_plans(const std::vector<Plan>& plans,
                                      const Inputs& in) {
  const bool fold = in.density[0].size() != 0;
  std::vector<const Plan*> selected;
  for (const auto& plan : plans) {
    if (plan.folded && !fold) continue;
    bool empty = false;
    plan.tree.visit([&](const auto& node) {
      if (!node.leaf() || !node->is_tensor()) return;
      for (const auto& index : node->as_tensor().const_braket_indices())
        empty = empty || in.orbitals(index.space().base_key()[0]).empty();
    });
    if (!empty) selected.push_back(&plan);
  }
  return selected;
}

// Label, orbital roles and spins of a plan leaf.
using OperandKey =
    std::tuple<std::wstring, std::vector<wchar_t>, std::vector<int>>;

OperandKey operand_key(const sq::Tensor& tensor, bool spin_free) {
  OperandKey key{std::wstring(tensor.label()), {}, {}};
  for (const auto& index : tensor.const_braket_indices()) {
    std::get<1>(key).push_back(index.space().base_key()[0]);
    std::get<2>(key).push_back(!spin_free &&
                               sq::mbpt::to_spin(index.space().qns()) ==
                                   sq::mbpt::Spin::beta);
  }
  return key;
}

// The numerical operand of one leaf: an integral, generator or density block.
// Generator two-body operands also record the diagnostics of their raw
// couplings; a spin-free operand stands for both the same-spin channel, which
// subtracts exchange, and the opposite-spin one.
sq::ResultPtr build_operand(const Inputs& in, const OneBodyOperators& operators,
                            const OperandKey& key, DownfoldResult& result) {
  const auto& [label, block_roles, spins] = key;
  std::vector<const std::vector<int>*> orbitals;
  std::vector<std::size_t> extents;
  for (const wchar_t role : block_roles) {
    orbitals.push_back(&in.orbitals(role));
    extents.push_back(orbitals.back()->size());
  }
  const bool is_generator = label == L"S";
  const bool is_density = label == L"D";
  const bool is_one_body = block_roles.size() == 2;
  // Two-body slots (0, 2) and (1, 3) belong to the same particle.
  const int spin = spins[0], partner = is_one_body ? spin : spins[1];
  const auto& one_body_block =
      (is_generator ? operators.generator : operators.normal_ordered)[spin];
  const auto& eps = in.energies[spin];
  const auto& partner_eps = in.energies[partner];
  const bool same_spin = in.spin_free || spin == partner;
  const bool opposite_spin = in.spin_free || spin != partner;
  double min_denominator = result.min_denominator;
  double max_amplitude = result.max_amplitude;
  Tensor values{btas::Range{extents}};
  const auto size = static_cast<std::ptrdiff_t>(values.size());
#pragma omp parallel for schedule(static) reduction(min : min_denominator) \
    reduction(max : max_amplitude) if (size >= min_parallel_work)
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
      value = in.density[spin](orbital[1], orbital[0]);
    } else if (is_one_body) {
      value = one_body_block(orbital[0], orbital[1]);
    } else {
      // Symbolic (bra, ket) slots map to chemist (pq|rs) order.
      const double direct = in.integral(spin, partner, orbital[0], orbital[2],
                                        orbital[1], orbital[3]);
      value = direct;
      if (is_generator) {
        const double delta =
            Inputs::denominator(eps(orbital[0]), partner_eps(orbital[1]),
                                eps(orbital[2]), partner_eps(orbital[3]));
        value *= regularized_inverse(delta, in.regularizer);
        const auto record = [&](double coupling) {
          if (std::abs(coupling) <= negligible_coupling) return;
          min_denominator = std::min(min_denominator, std::abs(delta));
          max_amplitude =
              std::max(max_amplitude, raw_amplitude(coupling, delta));
        };
        if (opposite_spin) record(direct);
        if (same_spin)
          record(direct - in.integral(spin, spin, orbital[0], orbital[3],
                                      orbital[1], orbital[2]));
      }
    }
    values.data()[ordinal] = value;
  }
  result.min_denominator = min_denominator;
  result.max_amplitude = max_amplitude;
  return sq::eval_result<sq::ResultTensorBTAS<Tensor>>(std::move(values));
}

using Operands = std::map<OperandKey, sq::ResultPtr>;

Operands build_operands(const std::vector<const Plan*>& selected,
                        const Inputs& in, const OneBodyOperators& operators,
                        DownfoldResult& result) {
  Operands operands;
  for (const Plan* plan : selected)
    plan->tree.visit([&](const auto& node) {
      if (!node.leaf() || !node->is_tensor()) return;
      auto key = operand_key(node->as_tensor(), in.spin_free);
      if (!operands.contains(key))
        operands.emplace(key, build_operand(in, operators, key, result));
    });
  return operands;
}

// Plans run concurrently with BLAS pinned to one thread, reading the
// prebuilt operands.
std::vector<sq::ResultPtr> evaluate_plans(
    const std::vector<const Plan*>& selected, const Operands& operands,
    const Inputs& in) {
  const auto leaf = [&](const sq::EvalNodeBTAS& node) -> sq::ResultPtr {
    if (!node->is_tensor())
      return sq::eval_result<sq::ResultScalar<double>>(
          node->as_constant().value<double>());
    return operands.at(operand_key(node->as_tensor(), in.spin_free));
  };
  std::vector<sq::ResultPtr> values(selected.size());
  std::exception_ptr failure;
  const auto plan_count = static_cast<std::ptrdiff_t>(selected.size());
  {
    const qdk::chemistry::scf::util::ScopedBlasThreads blas_thread_guard;
#pragma omp parallel for schedule(dynamic, \
                                      1) if (n4(in.norb) >= min_parallel_work)
    for (std::ptrdiff_t ordinal = 0; ordinal < plan_count; ++ordinal) {
      try {
        values[ordinal] = sq::evaluate(selected[ordinal]->tree, leaf);
      } catch (...) {
#pragma omp critical(swpt2_plan_failure)
        if (!failure) failure = std::current_exception();
      }
    }
  }
  if (failure) std::rethrow_exception(failure);
  return values;
}

// Adds one plan's value to the active-space operator.
void merge(const Plan& plan, const sq::ResultPtr& value, int active_count,
           DownfoldResult& result) {
  if (plan.rank == 0) {
    result.scalar += value->get<double>();
    return;
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
}  // namespace

DownfoldResult downfold(const std::array<Eigen::MatrixXd, 2>& one_body,
                        const TwoBodyBlocks& two_body,
                        const std::array<Eigen::MatrixXd, 2>& fock,
                        const std::vector<int>& active,
                        const std::vector<int>& inactive,
                        const std::vector<int>& virtuals,
                        const RegularizerOptions& regularizer,
                        double core_energy, bool spin_free,
                        const std::array<Eigen::MatrixXd, 2>& density) {
  const Inputs in{two_body,
                  active,
                  inactive,
                  virtuals,
                  regularizer,
                  density,
                  spin_free,
                  static_cast<int>(fock[0].rows()),
                  {fock[0].diagonal(), fock[1].diagonal()}};
  const auto operators = one_body_operators(in, one_body);
  // Start with P H_BD P; the contraction plans add 1/2 P [S, H_OD] P.
  auto result =
      block_diagonal_part(in, one_body, operators.normal_ordered, core_energy);
  if (!inactive.empty() || !virtuals.empty()) {
    track_single_couplings(in, operators.normal_ordered, fock, result);
    // Plans touching an empty role are skipped, so empty roles share extent
    // one.
    const auto extent = [](const std::vector<int>& orbitals) {
      return std::max<std::size_t>(orbitals.size(), 1);
    };
    const auto plans = build_plans(
        reusable_equations(),
        {extent(inactive), extent(active), extent(virtuals)}, spin_free);
    const auto selected = select_plans(plans, in);
    const auto operands = build_operands(selected, in, operators, result);
    const auto values = evaluate_plans(selected, operands, in);
    for (std::size_t ordinal = 0; ordinal < selected.size(); ++ordinal)
      merge(*selected[ordinal], values[ordinal],
            static_cast<int>(active.size()), result);
  }
  if (spin_free) {
    result.one_body[1] = result.one_body[0];
    result.same_spin_two_body = {result.two_body, result.two_body};
  }
  return result;
}

}  // namespace qdk::chemistry::algorithms::microsoft::swpt2
