// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "dsrg_pt2_kernel.hpp"

#include <btas/btas.h>
#include <qdk/chemistry/scf/util/blas_threads.h>

#include <SeQuant/core/eval/backends/btas/eval_expr.hpp>
#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <SeQuant/core/eval/eval.hpp>
#include <SeQuant/core/index_basis_registry.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/wick.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/spin.hpp>
#include <algorithm>
#include <cmath>
#include <exception>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::dsrg_pt2 {

namespace {
inline Eigen::Index n4(int n) {
  return static_cast<Eigen::Index>(n) * n * n * n;
}

constexpr Eigen::Index min_parallel_work = Eigen::Index{1} << 16;
constexpr Eigen::Index min_parallel_operand_size = Eigen::Index{1} << 12;

// Chemist (pq|rs) with p, q of spin `left` and r, s of spin `right`.
double chemist(const TwoBodyBlocks& two_body, int left, int right, int p, int q,
               int r, int s, int norb) {
  if (left == right) return two_body[2 * left].get()(idx4(p, q, r, s, norb));
  return left == 0 ? two_body[1].get()(idx4(p, q, r, s, norb))
                   : two_body[1].get()(idx4(r, s, p, q, norb));
}
}  // namespace

// The second-order DSRG expansion [S, H_OD] + [S, [S, F0]]/2 equals [S, V]/2
// with each doubles coupling and singles Fock coupling scaled by 2 - g, so a
// doubles channel's energy carries g (2 - g) = 1 - exp(-2s*D^2). The singles'
// active-space correction is scaled by 1 - g instead.
ChannelFactors channel_factors(double delta, double flow) {
  if (delta == 0.0) return {0.0, 2.0};
  // expm1 rather than 1-exp avoids cancellation for small s*delta^2.
  const double kept = -std::expm1(-flow * delta * delta);
  return {kept / delta, 2.0 - kept};
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

SemicanonicalRotation semicanonical_rotation(
    const Eigen::MatrixXd& fock, const std::vector<std::vector<int>>& blocks,
    double tolerance) {
  SemicanonicalRotation out{Eigen::MatrixXd::Identity(fock.rows(), fock.cols()),
                            false};
  for (const auto& block : blocks) {
    if (block.size() < 2) continue;
    const Eigen::MatrixXd sub = fock(block, block);
    const double scale = sub.cwiseAbs().maxCoeff();
    Eigen::MatrixXd off_diagonal = sub;
    off_diagonal.diagonal().setZero();
    if (off_diagonal.cwiseAbs().maxCoeff() <= tolerance * scale) continue;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sub);
    if (solver.info() != Eigen::Success)
      throw std::runtime_error(
          "semicanonical_rotation: Fock block diagonalization failed");
    out.matrix(block, block) = solver.eigenvectors();
    out.rotated = true;
  }
  return out;
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
  std::vector<int> all(norb);
  std::iota(all.begin(), all.end(), 0);
  return rotate_two_body(two_body, first_pair, second_pair, {{{}, all, {}}},
                         norb);
}

Eigen::VectorXd rotate_two_body(const Eigen::VectorXd& two_body,
                                const Eigen::MatrixXd& first_pair,
                                const Eigen::MatrixXd& second_pair,
                                const std::array<std::vector<int>, 3>& roles,
                                int norb) {
  using RowMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  constexpr Eigen::Index panel = 1024;
  const Eigen::Index n = norb;
  const bool parallel = n4(norb) >= min_parallel_work;
  Eigen::VectorXd out(n4(norb));
  for (int role_p = 0; role_p < 3; ++role_p)
    for (int role_q = 0; role_q < 3; ++role_q)
      for (int role_r = 0; role_r < 3; ++role_r)
        for (int role_s = 0; role_s < 3; ++role_s) {
          if ((role_p == 2) + (role_q == 2) + (role_r == 2) + (role_s == 2) > 2)
            continue;
          const auto& ip = roles[role_p];
          const auto& iq = roles[role_q];
          const auto& ir = roles[role_r];
          const auto& is = roles[role_s];
          const Eigen::Index np = ip.size(), nq = iq.size(), nr = ir.size(),
                             ns = is.size();
          const Eigen::Index size = np * nq * nr * ns;
          if (size == 0) continue;
          // The rotation keeps each role, so every role block transforms on
          // its own.
          Eigen::VectorXd block(size);
          const Eigen::Index rows = np * nq * nr;
#pragma omp parallel for schedule(static) if (parallel)
          for (Eigen::Index row = 0; row < rows; ++row) {
            const double* src =
                two_body.data() +
                ((ip[row / (nq * nr)] * n + iq[row / nr % nq]) * n +
                 ir[row % nr]) *
                    n;
            for (Eigen::Index s = 0; s < ns; ++s)
              block[row * ns + s] = src[is[s]];
          }
          const Eigen::MatrixXd ur = second_pair(ir, ir).transpose();
          const Eigen::MatrixXd us = second_pair(is, is);
#pragma omp parallel for schedule(static) if (parallel)
          for (Eigen::Index pq = 0; pq < np * nq; ++pq) {
            Eigen::Map<RowMajorMatrix> slice(block.data() + pq * nr * ns, nr,
                                             ns);
            const Eigen::MatrixXd tmp = ur * slice;
            slice.noalias() = tmp * us;
          }
          // Multiplies each of `count` consecutive row-major (rows x cols)
          // slices of the block from the left by `left`.
          const auto left_multiply = [&](Eigen::Index count, Eigen::Index cols,
                                         const Eigen::MatrixXd& left) {
            const Eigen::Index panels = (cols + panel - 1) / panel;
#pragma omp parallel for schedule(static) if (parallel)
            for (Eigen::Index task = 0; task < count * panels; ++task) {
              const Eigen::Index begin = task % panels * panel;
              const Eigen::Index width = std::min(panel, cols - begin);
              Eigen::Map<RowMajorMatrix> slice(
                  block.data() + task / panels * left.cols() * cols,
                  left.cols(), cols);
              const Eigen::MatrixXd tmp = left * slice.middleCols(begin, width);
              slice.middleCols(begin, width) = tmp;
            }
          };
          left_multiply(np, nr * ns, first_pair(iq, iq).transpose());
          left_multiply(1, nq * nr * ns, first_pair(ip, ip).transpose());
#pragma omp parallel for schedule(static) if (parallel)
          for (Eigen::Index row = 0; row < rows; ++row) {
            double* dst = out.data() +
                          ((ip[row / (nq * nr)] * n + iq[row / nr % nq]) * n +
                           ir[row % nr]) *
                              n;
            for (Eigen::Index s = 0; s < ns; ++s)
              dst[is[s]] = block[row * ns + s];
          }
        }
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
  std::shared_ptr<sq::IndexBasisRegistry> spaces;
  std::vector<Term> terms;
};

struct Plan {
  sq::EvalNodeBTAS tree;
  int rank;
  int spin_case;
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
          order.push_back(static_cast<int>(
              std::find(slots.begin(), slots.end(), op.index()) -
              slots.begin()));
        std::reverse(order.begin(), order.end());
        for (const auto& op : residual.creators())
          order.push_back(static_cast<int>(
              std::find(slots.begin(), slots.end(), op.index()) -
              slots.begin()));
        for (std::size_t left = 0; left < order.size(); ++left)
          for (std::size_t right = left + 1; right < order.size(); ++right)
            if (order[left] > order[right])
              pattern.coefficient = -pattern.coefficient;
      }
      if (!factor->is<sq::Tensor>()) continue;
      const auto& delta = factor->as<sq::Tensor>();
      if (delta.label() != sq::reserved::overlap_label()) continue;
      const int left = static_cast<int>(
          std::find(slots.begin(), slots.end(), delta.bra()[0]) -
          slots.begin());
      const int right = static_cast<int>(
          std::find(slots.begin(), slots.end(), delta.ket()[0]) -
          slots.begin());
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
  const int residual_rank = static_cast<int>(output.size() / 2);
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
      throw std::runtime_error("DSRG-PT2: unreduced source contraction");
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
      const int rank = static_cast<int>(target.size() / 2);
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
        output_names.emplace(
            labelled.at(output[target[axis]]),
            output_index(rank, spin_case, static_cast<int>(axis)));
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
          sums[{rank, spin_case, static_cast<int>(reduction.pairs.size())}];
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

  // Form 1/2 (S V - V S). Only pairs with no net inactive/virtual
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
    bool empty = false;
    term.result.expression()->visit(
        [&](const sq::ExprPtr& node) {
          if (!node->is<sq::Tensor>()) return;
          for (const auto& index :
               node->as<sq::Tensor>().const_braket_indices())
            if (extent(index) == 0) empty = true;
        },
        true);
    if (empty) continue;
    auto result = term.result;
    result.expression() = result.expression()->clone();
    // Only density-folded products have more than two tensor factors.
    if (term.folded)
      sq::optimize(result,
                   {.objective_function = sq::ObjectiveFunction::DenseSpaceTime,
                    .idx_to_extent = extent});
    auto tree = sq::binarize<sq::EvalExprBTAS>(result);
    plans.push_back({std::move(tree), term.rank, term.spin_case});
  }
  return plans;
}

// Relative cutoff below which couplings are treated as numerical zeros of
// symmetry-forbidden integrals and kept out of diagnostics.
constexpr double negligible_coupling = 1e-12;

// Inputs shared by the steps of one downfold call.
struct Inputs {
  const TwoBodyBlocks& two_body;
  const std::vector<int>& active;
  const std::vector<int>& inactive;
  const std::vector<int>& virtuals;
  double flow;
  const std::array<Eigen::MatrixXd, 2>& density;
  bool spin_free;
  int norb;
  std::array<Eigen::VectorXd, 2> energies;
  double coupling_tolerance;

  const std::vector<int>& orbitals(wchar_t role) const {
    return role == L'i' ? inactive : role == L'u' ? active : virtuals;
  }
  double integral(int left, int right, int p, int q, int r, int s) const {
    return chemist(two_body, left, right, p, q, r, s, norb);
  }
  // Summing creators and annihilators separately makes the denominator
  // exactly antisymmetric under the adjoint and symmetric under exchange.
  static double denominator(double first, double second, double third,
                            double fourth) {
    return (first + second) - (third + fourth);
  }
  ChannelFactors two_body_factors(double first, double second, double third,
                                  double fourth) const {
    return channel_factors(denominator(first, second, third, fourth), flow);
  }
};

struct OneBodyOperators {
  std::array<Eigen::MatrixXd, 2> normal_ordered, generator, coupling;
};

// Normal order against the filled inactive space, build the DSRG-PT2 singles
// (including the active-space correction b^a_i) converted to core normal
// order, and record single-excitation diagnostics.
OneBodyOperators one_body_operators(
    const Inputs& in, const std::array<Eigen::MatrixXd, 2>& one_body,
    DownfoldResult& result) {
  const int norb = in.norb;
  OneBodyOperators out{
      one_body,
      {Eigen::MatrixXd::Zero(norb, norb), Eigen::MatrixXd::Zero(norb, norb)},
      {Eigen::MatrixXd::Zero(norb, norb), Eigen::MatrixXd::Zero(norb, norb)}};
  for (int spin = 0; spin < (in.spin_free ? 1 : 2); ++spin) {
#pragma omp parallel for schedule(static) if (n4(norb) >= min_parallel_work)
    for (int row = 0; row < norb; ++row)
      for (int col = 0; col < norb; ++col) {
        double trace = 0.0;
        for (const int occupied : in.inactive) {
          for (int other = 0; other < 2; ++other)
            trace += in.integral(spin, other, row, col, occupied, occupied);
          trace -= in.integral(spin, spin, row, occupied, occupied, col);
        }
        out.normal_ordered[spin](row, col) += trace;
      }
  }
  if (in.flow > 0.0) {
    std::vector<char> is_active(norb, 0);
    for (const int orbital : in.active) is_active[orbital] = 1;
    std::vector<int> particles(in.active);
    particles.insert(particles.end(), in.virtuals.begin(), in.virtuals.end());
    std::vector<int> holes(in.inactive);
    holes.insert(holes.end(), in.active.begin(), in.active.end());
    for (int spin = 0; spin < (in.spin_free ? 1 : 2); ++spin) {
      const auto& eps = in.energies[spin];
      for (const int a : particles)
        for (const int i : holes) {
          if (is_active[a] && is_active[i]) continue;
          double active_fock = 0.0, b = 0.0, t_trace = 0.0, v_trace = 0.0;
          for (int other = 0; other < 2; ++other) {
            const auto& other_eps = in.energies[other];
            for (const int x : in.active)
              for (const int u : in.active) {
                const double gamma = in.density[other](x, u);
                if (gamma == 0.0) continue;
                double v = in.integral(spin, other, a, i, x, u);
                if (other == spin) v -= in.integral(spin, spin, a, u, x, i);
                const auto f2 = in.two_body_factors(eps(a), other_eps(x),
                                                    eps(i), other_eps(u));
                const double t = -v * f2.inverse;
                active_fock += gamma * v;
                b += (other_eps(x) - other_eps(u)) * gamma * t;
                t_trace += gamma * t;
                v_trace += gamma * v * f2.dressing;
              }
          }
          const double delta = eps(a) - eps(i);
          const double f_ai = out.normal_ordered[spin](a, i) + active_fock;
          const double coupling = f_ai + b;
          const auto f1 = channel_factors(delta, in.flow);
          if (std::abs(coupling) > in.coupling_tolerance) {
            result.min_denominator =
                std::min(result.min_denominator, std::abs(delta));
            result.max_amplitude =
                std::max(result.max_amplitude, std::abs(coupling * f1.inverse));
          }
          const double s_ai = coupling * f1.inverse + t_trace;
          const double v_ai =
              f_ai * f1.dressing + b * (f1.dressing - 1.0) - v_trace;
          out.generator[spin](a, i) = s_ai;
          out.generator[spin](i, a) = -s_ai;
          out.coupling[spin](a, i) = v_ai;
          out.coupling[spin](i, a) = v_ai;
        }
    }
  }
  if (in.spin_free) {
    out.normal_ordered[1] = out.normal_ordered[0];
    out.generator[1] = out.generator[0];
    out.coupling[1] = out.coupling[0];
  }
  return out;
}

// P H_BD P: the energy of the filled inactive space and the active blocks of
// the normal-ordered operator.
void populate_block_diagonal_part(
    const Inputs& in, const std::array<Eigen::MatrixXd, 2>& one_body,
    const std::array<Eigen::MatrixXd, 2>& normal_ordered, double core_energy,
    DownfoldResult& result) {
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

sq::ResultPtr build_one_body_operand(const Inputs& in,
                                     const OneBodyOperators& operators,
                                     const OperandKey& key) {
  const auto& [label, block_roles, spins] = key;
  const auto& rows = in.orbitals(block_roles[0]);
  const auto& cols = in.orbitals(block_roles[1]);
  const int spin = spins[0];
  const bool is_density = label == L"D";
  const auto& matrix = is_density      ? in.density[spin]
                       : label == L"S" ? operators.generator[spin]
                                       : operators.coupling[spin];
  Tensor values{btas::Range{rows.size(), cols.size()}};
  double* data = values.data();
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const int o0 = rows[r];
    for (std::size_t c = 0; c < cols.size(); ++c) {
      const int o1 = cols[c];
      *data++ = is_density ? matrix(o1, o0) : matrix(o0, o1);
    }
  }
  return sq::eval_result<sq::ResultTensorBTAS<Tensor>>(std::move(values));
}

using Operands = std::map<OperandKey, sq::ResultPtr>;

// Builds matching generator (S) and dressed-coupling (V) two-body operands in
// one pass and records double-excitation diagnostics.
void build_two_body_operands(const Inputs& in,
                             const std::vector<wchar_t>& block_roles,
                             const std::vector<int>& spins, bool need_s,
                             bool need_v, Operands& operands,
                             DownfoldResult& result) {
  const auto& orb0 = in.orbitals(block_roles[0]);
  const auto& orb1 = in.orbitals(block_roles[1]);
  const auto& orb2 = in.orbitals(block_roles[2]);
  const auto& orb3 = in.orbitals(block_roles[3]);
  const std::size_t e0 = orb0.size(), e1 = orb1.size(), e2 = orb2.size(),
                    e3 = orb3.size();
  const int spin = spins[0], partner = spins[1];
  const auto& eps = in.energies[spin];
  const auto& partner_eps = in.energies[partner];
  const bool same_spin = in.spin_free || spin == partner;
  const bool opposite_spin = in.spin_free || spin != partner;
  double min_denominator = result.min_denominator;
  double max_amplitude = result.max_amplitude;

  Tensor s_values, v_values;
  if (need_s) s_values = Tensor{btas::Range{e0, e1, e2, e3}};
  if (need_v) v_values = Tensor{btas::Range{e0, e1, e2, e3}};
  double* s_data = need_s ? s_values.data() : nullptr;
  double* v_data = need_v ? v_values.data() : nullptr;

  const auto total = static_cast<Eigen::Index>(e0 * e1 * e2 * e3);
  const auto outer = static_cast<std::ptrdiff_t>(e0 * e1 * e2);
#pragma omp parallel for schedule(static) reduction(min : min_denominator) \
    reduction(max : max_amplitude) if (total >= min_parallel_operand_size)
  for (std::ptrdiff_t idx = 0; idx < outer; ++idx) {
    const std::size_t i2 = static_cast<std::size_t>(idx) % e2;
    const std::size_t rem = static_cast<std::size_t>(idx) / e2;
    const std::size_t i1 = rem % e1;
    const std::size_t i0 = rem / e1;
    const int o0 = orb0[i0], o1 = orb1[i1], o2 = orb2[i2];
    const std::size_t base = static_cast<std::size_t>(idx) * e3;
    for (std::size_t i3 = 0; i3 < e3; ++i3) {
      const int o3 = orb3[i3];
      // Symbolic (bra, ket) slots map to chemist (pq|rs) order.
      const double direct = in.integral(spin, partner, o0, o2, o1, o3);
      const double delta = Inputs::denominator(eps(o0), partner_eps(o1),
                                               eps(o2), partner_eps(o3));
      const auto factors = channel_factors(delta, in.flow);
      if (need_s) {
        s_data[base + i3] = direct * factors.inverse;
        const auto record = [&](double coupling) {
          if (std::abs(coupling) <= in.coupling_tolerance) return;
          min_denominator = std::min(min_denominator, std::abs(delta));
          max_amplitude =
              std::max(max_amplitude, std::abs(coupling * factors.inverse));
        };
        if (opposite_spin) record(direct);
        if (same_spin) record(direct - in.integral(spin, spin, o0, o3, o1, o2));
      }
      if (need_v) v_data[base + i3] = direct * factors.dressing;
    }
  }
  result.min_denominator = min_denominator;
  result.max_amplitude = max_amplitude;
  if (need_s)
    operands.emplace(
        OperandKey{L"S", block_roles, spins},
        sq::eval_result<sq::ResultTensorBTAS<Tensor>>(std::move(s_values)));
  if (need_v)
    operands.emplace(
        OperandKey{L"V", block_roles, spins},
        sq::eval_result<sq::ResultTensorBTAS<Tensor>>(std::move(v_values)));
}

Operands build_operands(const std::vector<Plan>& plans, const Inputs& in,
                        const OneBodyOperators& operators,
                        DownfoldResult& result) {
  Operands operands;
  using TwoBodyKey = std::pair<std::vector<wchar_t>, std::vector<int>>;
  std::map<TwoBodyKey, std::pair<bool, bool>> two_body_requests;
  for (const Plan& plan : plans)
    plan.tree.visit([&](const auto& node) {
      if (!node.leaf() || !node->is_tensor()) return;
      auto key = operand_key(node->as_tensor(), in.spin_free);
      if (std::get<1>(key).size() == 2) {
        if (!operands.contains(key))
          operands.emplace(key, build_one_body_operand(in, operators, key));
        return;
      }
      auto& flags = two_body_requests[{std::move(std::get<1>(key)),
                                       std::move(std::get<2>(key))}];
      if (std::get<0>(key) == L"S")
        flags.first = true;
      else
        flags.second = true;
    });
  for (const auto& [key, flags] : two_body_requests)
    build_two_body_operands(in, key.first, key.second, flags.first,
                            flags.second, operands, result);
  return operands;
}

// Adds one plan's value to the active-space operator.
void merge(const Plan& plan, const sq::ResultPtr& value, int active_count,
           DownfoldResult& result) {
  if (plan.rank == 0) {
    result.scalar += value->get<double>();
    return;
  }
  const auto& values = value->get<Tensor>();
  const double* data = values.data();
  const auto n = static_cast<std::size_t>(active_count);
  // BTAS stores canonical axes, which need not match the requested output.
  std::array<std::size_t, 4> output_axes{};
  std::size_t axis_index = 0;
  for (const auto& index : plan.tree->as_tensor().const_braket_indices())
    output_axes[axis_index++] = static_cast<std::size_t>(
        std::find(plan.tree->canon_indices().begin(),
                  plan.tree->canon_indices().end(), index) -
        plan.tree->canon_indices().begin());
  if (plan.rank == 1) {
    const std::array<std::size_t, 2> canon_stride{n, 1};
    const std::size_t s0 = canon_stride[output_axes[0]];
    const std::size_t s1 = canon_stride[output_axes[1]];
    auto& target = result.one_body[plan.spin_case];
    for (int col = 0; col < active_count; ++col)
      for (int row = 0; row < active_count; ++row)
        target(row, col) += data[row * s0 + col * s1];
    return;
  }
  const std::array<std::size_t, 4> canon_stride{n * n * n, n * n, n, 1};
  const std::size_t s0 = canon_stride[output_axes[0]];
  const std::size_t s1 = canon_stride[output_axes[1]];
  const std::size_t s2 = canon_stride[output_axes[2]];
  const std::size_t s3 = canon_stride[output_axes[3]];
  // Chemist storage reverses the annihilator pair, giving the minus sign.
  if (plan.spin_case == 1) {
    double* dst = result.two_body.data();
    for (int p = 0; p < active_count; ++p)
      for (int r = 0; r < active_count; ++r)
        for (int q = 0; q < active_count; ++q) {
          const std::size_t base = p * s0 + q * s1 + r * s2;
          for (int s = 0; s < active_count; ++s) *dst++ -= data[base + s * s3];
        }
    return;
  }
  // A same-spin block also takes the copy with the particles exchanged.
  double* dst = result.same_spin_two_body[plan.spin_case / 2].data();
  std::size_t linear = 0;
  for (int p = 0; p < active_count; ++p)
    for (int r = 0; r < active_count; ++r)
      for (int q = 0; q < active_count; ++q) {
        const std::size_t base = p * s0 + q * s1 + r * s2;
        for (int s = 0; s < active_count; ++s, ++linear) {
          const double v = data[base + s * s3];
          dst[linear] -= v;
          dst[idx4(q, s, p, r, active_count)] -= v;
        }
      }
}

void evaluate_and_merge(const std::vector<Plan>& plans,
                        const Operands& operands, const Inputs& in,
                        DownfoldResult& result) {
  const auto leaf = [&](const sq::EvalNodeBTAS& node) -> sq::ResultPtr {
    if (!node->is_tensor())
      return sq::eval_result<sq::ResultScalar<double>>(
          node->as_constant().value<double>());
    return operands.at(operand_key(node->as_tensor(), in.spin_free));
  };
  const int active_count = static_cast<int>(in.active.size());
  // Results merge in plan order, so the sum does not depend on the thread
  // count.
  std::vector<sq::ResultPtr> values(plans.size());
  std::size_t merged = 0;
  std::exception_ptr failure;
  const auto plan_count = static_cast<std::ptrdiff_t>(plans.size());
  const qdk::chemistry::scf::util::ScopedBlasThreads blas_thread_guard;
#pragma omp parallel for schedule(dynamic, \
                                      1) if (n4(in.norb) >= min_parallel_work)
  for (std::ptrdiff_t ordinal = 0; ordinal < plan_count; ++ordinal) {
    sq::ResultPtr value;
    try {
      value = sq::evaluate(plans[ordinal].tree, leaf);
    } catch (...) {
#pragma omp critical(dsrg_pt2_merge)
      if (!failure) failure = std::current_exception();
      continue;
    }
#pragma omp critical(dsrg_pt2_merge)
    {
      values[ordinal] = std::move(value);
      for (; merged < values.size() && values[merged]; ++merged) {
        merge(plans[merged], values[merged], active_count, result);
        values[merged].reset();
      }
    }
  }
  if (failure) std::rethrow_exception(failure);
}
}  // namespace

DownfoldResult downfold(const std::array<Eigen::MatrixXd, 2>& one_body,
                        const TwoBodyBlocks& two_body,
                        const std::array<Eigen::VectorXd, 2>& orbital_energies,
                        const std::vector<int>& active,
                        const std::vector<int>& inactive,
                        const std::vector<int>& virtuals, double flow,
                        double core_energy, bool spin_free,
                        const std::array<Eigen::MatrixXd, 2>& density) {
  double scale = 0.0;
  for (int spin = 0; spin < (spin_free ? 1 : 2); ++spin)
    scale = std::max({scale, one_body[spin].cwiseAbs().maxCoeff(),
                      orbital_energies[spin].cwiseAbs().maxCoeff()});
  const Inputs in{two_body,
                  active,
                  inactive,
                  virtuals,
                  flow,
                  density,
                  spin_free,
                  static_cast<int>(orbital_energies[0].size()),
                  orbital_energies,
                  negligible_coupling * scale};
  DownfoldResult result;
  const auto operators = one_body_operators(in, one_body, result);
  // Start with P H_BD P; at positive flow the contraction plans add
  // 1/2 P [S, V] P with the dressed coupling V.
  populate_block_diagonal_part(in, one_body, operators.normal_ordered,
                               core_energy, result);
  if (flow > 0.0 && (!inactive.empty() || !virtuals.empty())) {
    const auto plans = build_plans(
        reusable_equations(), {inactive.size(), active.size(), virtuals.size()},
        spin_free);
    const auto operands = build_operands(plans, in, operators, result);
    evaluate_and_merge(plans, operands, in, result);
  }
  if (spin_free) {
    result.one_body[1] = result.one_body[0];
    result.same_spin_two_body = {result.two_body, result.two_body};
  }
  return result;
}

}  // namespace qdk::chemistry::algorithms::microsoft::dsrg_pt2
