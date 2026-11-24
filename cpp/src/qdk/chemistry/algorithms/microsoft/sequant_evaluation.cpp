// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "sequant_evaluation.hpp"

#include <btas/btas.h>
#include <qdk/chemistry/scf/util/blas_threads.h>

#include <SeQuant/core/binary_node.hpp>
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/eval/backends/btas/eval_expr.hpp>
#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <SeQuant/core/eval/eval.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/expressions/variable.hpp>
#include <SeQuant/core/op.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/utility/indices.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/spin.hpp>
#include <algorithm>
#include <compare>
#include <cstdint>
#include <exception>
#include <map>
#include <range/v3/range/conversion.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qdk::chemistry::algorithms::microsoft::sequant_evaluation {

std::size_t tensor_size(std::span<const std::size_t> extents) {
  std::size_t size = 1;
  for (const auto extent : extents) {
    if (extent && size > std::vector<double>().max_size() / extent)
      throw std::length_error("Tensor extents exceed addressable storage.");
    size *= extent;
  }
  return size;
}

namespace {

using namespace sequant;
using BTensor = btas::Tensor<double>;

constexpr std::size_t terms_per_worker = 16;
constexpr std::size_t result_buffer_bytes = std::size_t{64} << 20;

const Context& evaluation_context() {
  static const Context context(
      {.index_basis_registry_shared_ptr =
           mbpt::make_min_sr_spaces(mbpt::SpinConvention::Default),
       .vacuum = Vacuum::SingleProduct});
  return context;
}

enum class Space : std::uint8_t { occupied, virtuals, general };

/// Orbital range of one tensor slot; residual slots are limited to the target.
struct Slot {
  Space space = Space::general;
  int spin = 0;
  bool residual = false;
  auto operator<=>(const Slot&) const = default;
};

Slot slot(const Index& resolved, bool residual) {
  const auto spin = mbpt::to_spin(resolved.space().qns());
  if (spin != mbpt::Spin::alpha && spin != mbpt::Spin::beta)
    throw std::logic_error(
        "SeQuant evaluation requires spin-resolved indices.");
  const auto space = [&] {
    switch (resolved.space().base_key().front()) {
      case L'i':
        return Space::occupied;
      case L'a':
        return Space::virtuals;
      case L'p':
        return Space::general;
      default:
        throw std::logic_error(
            "SeQuant evaluation encountered an unsupported index space.");
    }
  }();
  return {space, spin == mbpt::Spin::beta ? 1 : 0, residual};
}

/// Spin orbitals of each slot, and the output positions of residual slots.
class IndexRanges {
 public:
  explicit IndexRanges(const SpinOrbitalSpace& space) {
    std::unordered_map<std::size_t, std::size_t> position;
    for (std::size_t i = 0; i < space.target.size(); ++i)
      position.emplace(space.target[i], i);
    for (int spin = 0; spin < 2; ++spin) {
      _all[0][spin] = space.occupied[spin];
      _all[1][spin] = space.virtuals[spin];
      _all[2][spin] = space.occupied[spin];
      _all[2][spin].insert(_all[2][spin].end(), space.virtuals[spin].begin(),
                           space.virtuals[spin].end());
      for (std::size_t kind = 0; kind < 3; ++kind)
        for (const auto orbital : _all[kind][spin])
          if (const auto found = position.find(orbital);
              found != position.end()) {
            _target[kind][spin].push_back(orbital);
            _positions[kind][spin].push_back(found->second);
          }
    }
  }

  const std::vector<std::size_t>& orbitals(const Slot& slot) const {
    const auto kind = static_cast<std::size_t>(slot.space);
    return slot.residual ? _target[kind][slot.spin] : _all[kind][slot.spin];
  }

  const std::vector<std::size_t>& positions(const Slot& slot) const {
    return _positions[static_cast<std::size_t>(slot.space)][slot.spin];
  }

 private:
  std::array<std::array<std::vector<std::size_t>, 2>, 3> _all, _target,
      _positions;
};

struct LeafKey {
  std::wstring label;
  std::vector<Slot> slots;
  auto operator<=>(const LeafKey&) const = default;
};

using LeafStore = std::map<LeafKey, ResultPtr>;

/// One spin assignment of a term: its plan in PreparedContraction::plans, its
/// spin-summation multiplicity and the spin-resolved version of each index.
struct SpinAssignment {
  std::size_t plan_index, multiplicity;
  container::map<Index, Index> indices;
};

/// A term ready for evaluation: residual indices, result layout, one plan per
/// set of index extents, and the spin assignments to sum.
struct PreparedContraction {
  container::svector<Index> free;
  container::svector<long> target;
  std::vector<EvalNodeBTAS> plans;
  std::vector<SpinAssignment> assignments;
};

bool is_free(const Index& index, const container::svector<Index>& free) {
  return std::find(free.begin(), free.end(), index) != free.end();
}

LeafKey leaf_key(const EvalExprBTAS& leaf, const SpinAssignment& assignment,
                 const container::svector<Index>& free) {
  const auto& tensor = leaf.expr()->as<Tensor>();
  LeafKey key{std::wstring(tensor.label()), {}};
  for (const auto& index : tensor.const_braket_indices())
    key.slots.push_back(
        slot(assignment.indices.at(index), is_free(index, free)));
  return key;
}

/// Leaf blocks of one spin assignment, for sequant::evaluate.
class AssignmentLeaves {
 public:
  AssignmentLeaves(const LeafStore& store, const SpinAssignment& assignment,
                   const container::svector<Index>& free)
      : _store(store), _assignment(assignment), _free(free) {}

  ResultPtr operator()(sequant::meta::can_evaluate auto const& node) const {
    if (node->result_type() != ResultType::Tensor)
      return eval_result<ResultScalar<double>>(
          node->as_constant().template value<double>());
    const auto& block = _store.at(leaf_key(*node, _assignment, _free));
    // Canonicalization can permute antisymmetric leaf slots.
    return node->canon_phase() == -1 ? block->mult_by_phase(-1) : block;
  }

 private:
  const LeafStore& _store;
  const SpinAssignment& _assignment;
  const container::svector<Index>& _free;
};

ResultPtr build_leaf(const LeafKey& key, const IndexRanges& index_ranges,
                     const LeafElement& element) {
  std::vector<const std::vector<std::size_t>*> axes;
  std::vector<std::size_t> extents;
  for (const auto& slot : key.slots) {
    axes.push_back(&index_ranges.orbitals(slot));
    extents.push_back(axes.back()->size());
  }
  tensor_size(extents);
  BTensor block{btas::Range{extents}};
  std::vector<std::size_t> position(axes.size(), 0), orbitals(axes.size());
  for (std::size_t flat = 0; flat < block.size(); ++flat) {
    for (std::size_t axis = 0; axis < axes.size(); ++axis)
      orbitals[axis] = (*axes[axis])[position[axis]];
    block.data()[flat] = element(key.label, orbitals);
    for (std::size_t axis = axes.size(); axis-- > 0;) {
      if (++position[axis] < extents[axis]) break;
      position[axis] = 0;
    }
  }
  return eval_result<ResultTensorBTAS<BTensor>>(std::move(block));
}

/// The tensor factors of a term and the indices of its normal operator,
/// creators first.
std::pair<ExprPtr, container::svector<Index>> split_term(
    const ExprPtr& source) {
  const auto term = source->clone();
  ExprPtr coefficient = ex<Product>(1, ExprPtrList{});
  container::svector<Index> free;
  auto visit = [&](const ExprPtr& factor) {
    if (factor.is<FNOperator>()) {
      const auto& op = factor.as<FNOperator>();
      if (!free.empty() || op.ncreators() != op.nannihilators() ||
          op.ncreators() > 2)
        throw std::logic_error(
            "SeQuant evaluation supports one number-conserving normal operator "
            "of rank at most two per term.");
      for (const auto& creator : op.creators()) free.push_back(creator.index());
      for (const auto& annihilator : op.annihilators())
        free.push_back(annihilator.index());
    } else
      coefficient.as<Product>().append(1, factor, Product::Flatten::No);
  };
  if (term.is<Product>()) {
    coefficient = ex<Product>(term.as<Product>().scalar(), ExprPtrList{});
    for (const auto& factor : term.as<Product>().factors()) visit(factor);
  } else
    visit(term);
  return {coefficient, free};
}

/// Indices of a coefficient with, per tensor, the bra-minus-ket count of each
/// index, and the groups of interchangeable indices.
struct IndexGraph {
  std::vector<Index> indices;
  container::map<Index, std::size_t> positions;
  std::vector<std::vector<int>> conservation;
  std::vector<std::vector<std::size_t>> equivalent;
  std::vector<std::size_t> previous;
};

IndexGraph index_graph(const ExprPtr& coefficient,
                       const container::svector<Index>& free) {
  IndexGraph graph;
  const auto used = get_used_indices<container::set<Index>>(coefficient);
  graph.indices.assign(used.begin(), used.end());
  for (std::size_t i = 0; i < graph.indices.size(); ++i)
    graph.positions.emplace(graph.indices[i], i);
  using Endpoint = std::pair<std::size_t, int>;
  std::vector<std::vector<Endpoint>> endpoints(graph.indices.size());
  std::vector<Symmetry> symmetries;
  for (const auto& factor : coefficient.as<Product>().factors()) {
    if (!factor.is<Tensor>()) continue;
    const auto& tensor = factor.as<Tensor>();
    const auto ordinal = symmetries.size();
    symmetries.push_back(tensor.symmetry());
    std::vector<int> row(graph.indices.size(), 0);
    for (const auto& index : tensor.bra()) {
      ++row[graph.positions.at(index)];
      endpoints[graph.positions.at(index)].emplace_back(ordinal, 0);
    }
    for (const auto& index : tensor.ket()) {
      --row[graph.positions.at(index)];
      endpoints[graph.positions.at(index)].emplace_back(ordinal, 1);
    }
    for (const auto& index : tensor.aux())
      endpoints[graph.positions.at(index)].emplace_back(ordinal, 2);
    graph.conservation.push_back(std::move(row));
  }
  graph.previous.assign(graph.indices.size(), graph.indices.size());
  for (std::size_t i = 0; i < graph.indices.size(); ++i) {
    const auto& edge = endpoints[i];
    if (graph.indices[i].has_proto_indices() || edge.size() != 2 ||
        edge[0].first == edge[1].first || edge[0].second == 2 ||
        edge[1].second == 2 || is_free(graph.indices[i], free))
      continue;
    const auto symmetry = symmetries[edge[0].first];
    if ((symmetry != Symmetry::Antisymm && symmetry != Symmetry::Symm) ||
        symmetry != symmetries[edge[1].first])
      continue;
    auto group =
        std::find_if(graph.equivalent.begin(), graph.equivalent.end(),
                     [&](const auto& candidate) {
                       return graph.indices[candidate.front()].space() ==
                                  graph.indices[i].space() &&
                              endpoints[candidate.front()] == edge;
                     });
    if (group == graph.equivalent.end())
      graph.equivalent.push_back({i});
    else {
      graph.previous[i] = group->back();
      group->push_back(i);
    }
  }
  return graph;
}

/// Calls visit(spins) with each spin-conserving assignment (0 alpha, 1 beta),
/// listing interchangeable indices in nondecreasing spin order only.
template <typename Visit>
void enumerate_spins(const IndexGraph& graph, Visit&& visit) {
  std::vector<int> spins(graph.indices.size(), 0);
  auto next_index = [&](auto&& self, std::size_t next) -> void {
    for (const auto& row : graph.conservation) {
      int balance = 0, lower = 0, upper = 0;
      for (std::size_t i = 0; i < row.size(); ++i) {
        if (i < next)
          balance += row[i] * spins[i];
        else if (row[i] < 0)
          lower += row[i];
        else
          upper += row[i];
      }
      if (balance + lower > 0 || balance + upper < 0) return;
    }
    if (next == graph.indices.size()) {
      visit(spins);
      return;
    }
    for (int spin : {0, 1}) {
      if (graph.previous[next] != graph.indices.size() &&
          spin < spins[graph.previous[next]])
        continue;
      spins[next] = spin;
      self(self, next + 1);
    }
  };
  next_index(next_index, 0);
}

std::size_t multiplicity(const IndexGraph& graph,
                         const std::vector<int>& spins) {
  // Parallel equivalent edges contribute two cancelling permutation signs.
  std::size_t result = 1;
  for (const auto& group : graph.equivalent) {
    std::size_t beta = 0, choices = 1;
    for (const auto i : group) beta += spins[i];
    for (std::size_t k = 1; k <= beta; ++k)
      choices = choices * (group.size() - beta + k) / k;
    result *= choices;
  }
  return result;
}

EvalNodeBTAS plan(const ExprPtr& coefficient,
                  const container::svector<Index>& free,
                  const IndexGraph& graph,
                  const std::vector<std::size_t>& extents) {
  OptimizeOptions options;
  options.idx_to_extent = [&](const Index& index) {
    return extents[graph.positions.at(index)];
  };
  const auto optimized = sequant::optimize(coefficient, options);
  if (free.empty())
    return binarize<EvalExprBTAS>(ResultExpr(Variable(L"E"), optimized));
  const auto rank = free.size() / 2;
  const Tensor result(
      L"R", bra(container::svector<Index>(free.begin(), free.begin() + rank)),
      ket(container::svector<Index>(free.begin() + rank, free.end())),
      Symmetry::Nonsymm);
  return binarize<EvalExprBTAS>(ResultExpr(result, optimized));
}

PreparedContraction prepare(const ExprPtr& term,
                            const IndexRanges& index_ranges) {
  PreparedContraction result;
  ExprPtr coefficient;
  std::tie(coefficient, result.free) = split_term(term);
  result.target = EvalExprBTAS::index_hash(result.free) |
                  ranges::to<container::svector<long>>;
  const auto graph = index_graph(coefficient, result.free);
  std::vector<std::array<Index, 2>> spin_indices;
  for (const auto& index : graph.indices)
    spin_indices.push_back(
        {mbpt::make_spinalpha(index), mbpt::make_spinbeta(index)});
  std::map<std::vector<std::size_t>, std::size_t> plans;
  enumerate_spins(graph, [&](const std::vector<int>& spins) {
    container::map<Index, Index> resolved;
    std::vector<std::size_t> extents;
    for (std::size_t i = 0; i < graph.indices.size(); ++i) {
      const auto& index = spin_indices[i][spins[i]];
      resolved.emplace(graph.indices[i], index);
      extents.push_back(
          index_ranges
              .orbitals(slot(index, is_free(graph.indices[i], result.free)))
              .size());
      if (extents.back() == 0) return;
    }
    auto found = plans.find(extents);
    if (found == plans.end()) {
      found = plans.emplace(extents, result.plans.size()).first;
      result.plans.push_back(plan(coefficient, result.free, graph, extents));
    }
    result.assignments.push_back(
        {found->second, multiplicity(graph, spins), std::move(resolved)});
  });
  return result;
}

std::vector<Slot> residual_slots(const PreparedContraction& term,
                                 const SpinAssignment& assignment) {
  std::vector<Slot> slots;
  for (const auto& index : term.free)
    slots.push_back(slot(assignment.indices.at(index), true));
  return slots;
}

/// Number of values in a term's result: one per residual element of each
/// distinct residual spin pattern, or one scalar.
std::size_t result_size(const PreparedContraction& term,
                        const IndexRanges& index_ranges) {
  if (term.free.empty()) return 1;
  std::set<std::vector<Slot>> patterns;
  std::size_t size = 0;
  for (const auto& assignment : term.assignments) {
    auto slots = residual_slots(term, assignment);
    std::size_t block = 1;
    for (const auto& slot : slots) block *= index_ranges.orbitals(slot).size();
    if (patterns.insert(std::move(slots)).second) size += block;
  }
  return size;
}

/// Weighted sum of a term's spin assignments: a scalar, or one block per
/// residual spin pattern.
struct TermResult {
  double scalar = 0.0;
  std::vector<std::pair<std::vector<Slot>, BTensor>> blocks;
};

TermResult evaluate_term(const PreparedContraction& term,
                         const LeafStore& store) {
  TermResult result;
  for (const auto& assignment : term.assignments) {
    const AssignmentLeaves leaves(store, assignment, term.free);
    const auto& node = term.plans[assignment.plan_index];
    const auto weight = static_cast<double>(assignment.multiplicity);
    if (term.free.empty()) {
      result.scalar += weight * sequant::evaluate(node, leaves)->get<double>();
      continue;
    }
    auto slots = residual_slots(term, assignment);
    const auto value = sequant::evaluate(node, term.target, leaves);
    const auto& block = value->get<BTensor>();
    auto found =
        std::find_if(result.blocks.begin(), result.blocks.end(),
                     [&](const auto& entry) { return entry.first == slots; });
    if (found == result.blocks.end()) {
      result.blocks.emplace_back(std::move(slots), block);
      for (std::size_t i = 0; i < block.size(); ++i)
        result.blocks.back().second.data()[i] *= weight;
      continue;
    }
    for (std::size_t i = 0; i < block.size(); ++i)
      found->second.data()[i] += weight * block.data()[i];
  }
  return result;
}

void accumulate(const TermResult& result, const IndexRanges& index_ranges,
                std::size_t n, NormalOrderedOperator& total) {
  total.scalar += result.scalar;
  for (const auto& [slots, block] : result.blocks) {
    auto& target = slots.size() == 2 ? total.one_body : total.two_body;
    std::vector<const std::vector<std::size_t>*> axes;
    for (const auto& slot : slots)
      axes.push_back(&index_ranges.positions(slot));
    std::vector<std::size_t> position(axes.size(), 0);
    for (std::size_t flat = 0; flat < block.size(); ++flat) {
      std::size_t address = 0;
      for (std::size_t axis = 0; axis < axes.size(); ++axis)
        address = address * n + (*axes[axis])[position[axis]];
      target[address] += block.data()[flat];
      for (std::size_t axis = axes.size(); axis-- > 0;) {
        if (++position[axis] < axes[axis]->size()) break;
        position[axis] = 0;
      }
    }
  }
}

/// Calls body(i) for i in [0, count) on up to `workers` threads and rethrows
/// the first exception.
template <typename Body>
void parallel_for(std::size_t count, std::size_t workers, Body&& body) {
  workers = std::max<std::size_t>(1, std::min(workers, count));
  std::exception_ptr failure;
#ifdef _OPENMP
#pragma omp parallel if (workers > 1) num_threads(workers)
#endif
  {
    const auto scope = set_scoped_default_context(evaluation_context());
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
    for (std::size_t i = 0; i < count; ++i) {
      try {
        body(i);
      } catch (...) {
#ifdef _OPENMP
#pragma omp critical(sequant_evaluation_failure)
#endif
        {
          if (!failure) failure = std::current_exception();
        }
      }
    }
  }
  if (failure) std::rethrow_exception(failure);
}

/// Adds the leaf blocks that the prepared terms use and the store lacks.
void add_leaves(const std::vector<PreparedContraction>& prepared,
                const IndexRanges& index_ranges, const LeafElement& element,
                std::size_t workers, LeafStore& store) {
  std::vector<LeafStore::value_type*> missing;
  for (const auto& term : prepared)
    for (const auto& assignment : term.assignments)
      term.plans[assignment.plan_index].visit_leaf(
          [&](const EvalNodeBTAS& node) {
            if (node->result_type() != ResultType::Tensor) return;
            const auto [entry, inserted] =
                store.try_emplace(leaf_key(*node, assignment, term.free));
            if (inserted) missing.push_back(&*entry);
          });
  parallel_for(missing.size(), workers, [&](std::size_t i) {
    missing[i]->second = build_leaf(missing[i]->first, index_ranges, element);
  });
}

}  // namespace

NormalOrderedOperator evaluate(const sequant::ExprPtr& expression,
                               const SpinOrbitalSpace& space,
                               const LeafElement& leaf) {
  const auto n = space.target.size();
  NormalOrderedOperator total{
      0.0, std::vector<double>(tensor_size({n, n}), 0.0),
      std::vector<double>(tensor_size({n, n, n, n}), 0.0)};
  std::vector<sequant::ExprPtr> terms;
  if (expression.is<sequant::Sum>()) {
    for (const auto& term : expression.as<sequant::Sum>().summands())
      terms.push_back(term);
  } else
    terms.push_back(expression);
  const IndexRanges index_ranges(space);
  std::size_t workers = 1;
#ifdef _OPENMP
  workers = static_cast<std::size_t>(omp_get_max_threads());
#endif
  const auto scope = sequant::set_scoped_default_context(evaluation_context());
  const scf::util::ScopedBlasThreads blas_threads;
  const auto window = terms_per_worker * workers;
  LeafStore store;
  for (std::size_t begin = 0; begin < terms.size(); begin += window) {
    const auto count = std::min(window, terms.size() - begin);
    std::vector<PreparedContraction> prepared(count);
    parallel_for(count, workers, [&](std::size_t i) {
      prepared[i] = prepare(terms[begin + i], index_ranges);
    });
    add_leaves(prepared, index_ranges, leaf, workers, store);
    std::vector<std::size_t> sizes(count);
    for (std::size_t i = 0; i < count; ++i)
      sizes[i] = sizeof(double) * result_size(prepared[i], index_ranges);
    for (std::size_t first = 0, last = 0; first < count; first = last) {
      std::size_t bytes = sizes[first];
      for (last = first + 1;
           last < count && bytes + sizes[last] <= result_buffer_bytes; ++last)
        bytes += sizes[last];
      std::vector<TermResult> results(last - first);
      parallel_for(last - first, workers, [&](std::size_t i) {
        results[i] = evaluate_term(prepared[first + i], store);
      });
      // Term-order reduction is identical for serial and parallel evaluation.
      for (const auto& result : results)
        accumulate(result, index_ranges, n, total);
    }
  }
  return total;
}

}  // namespace qdk::chemistry::algorithms::microsoft::sequant_evaluation
