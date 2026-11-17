// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ducc.hpp"

#include <btas/btas.h>
#include <qdk/chemistry/scf/util/blas_threads.h>

#include <Eigen/Dense>
#include <SeQuant/core/binary_node.hpp>
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/eval/backends/btas/eval_expr.hpp>
#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <SeQuant/core/eval/eval.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/expressions/variable.hpp>
#include <SeQuant/core/op.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/utility/indices.hpp>
#include <SeQuant/core/wick.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/op.hpp>
#include <SeQuant/domain/mbpt/spin.hpp>
#include <SeQuant/domain/mbpt/vac_av.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <range/v3/range/conversion.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qdk::chemistry::algorithms::microsoft {
namespace {

using namespace sequant;
using BTensor = btas::Tensor<double>;

std::size_t tensor_size(std::initializer_list<std::size_t> dimensions) {
  std::size_t size = 1;
  for (const auto dimension : dimensions) {
    if (dimension && size > std::vector<double>().max_size() / dimension)
      throw std::length_error(
          "DUCC tensor dimensions exceed addressable storage.");
    size *= dimension;
  }
  return size;
}

std::size_t index4(std::size_t p, std::size_t q, std::size_t r, std::size_t s,
                   std::size_t n) {
  return ((p * n + q) * n + r) * n + s;
}

struct Orbital {
  std::size_t spatial, amplitude;
  int spin;
};

struct Input {
  std::size_t nmo = 0, nocc = 0;
  std::array<std::size_t, 2> occupied{}, virtuals{};
  std::array<Eigen::MatrixXd, 2> fock;
  std::array<const Eigen::VectorXd*, 3> eri{}, doubles{};
  std::array<const Eigen::VectorXd*, 2> singles{};
  std::vector<Orbital> orbitals;
  std::vector<bool> active;
  std::array<std::vector<std::size_t>, 2> kept;
  std::vector<std::size_t> output;
  bool restricted = false;
  double scalar = 0.0;

  double chemist(std::size_t p, std::size_t q, std::size_t r,
                 std::size_t s) const {
    const auto &a = orbitals[p], &b = orbitals[q], &c = orbitals[r],
               &d = orbitals[s];
    if (a.spin != b.spin || c.spin != d.spin) return 0.0;
    if (a.spin == 1 && c.spin == 0)
      return (*eri[1])[index4(c.spatial, d.spatial, a.spatial, b.spatial, nmo)];
    return (*eri[a.spin + c.spin])[index4(a.spatial, b.spatial, c.spatial,
                                          d.spatial, nmo)];
  }

  double two(std::size_t p, std::size_t q, std::size_t r, std::size_t s) const {
    return chemist(p, r, q, s) - chemist(p, s, q, r);
  }

  double one(std::size_t p, std::size_t q) const {
    const auto &a = orbitals[p], &b = orbitals[q];
    return a.spin == b.spin ? fock[a.spin](a.spatial, b.spatial) : 0.0;
  }

  double t1(std::size_t a, std::size_t i) const {
    if (active[a] && active[i]) return 0.0;
    const auto &v = orbitals[a], &o = orbitals[i];
    if (v.spin != o.spin) return 0.0;
    return (*singles[o.spin])[o.amplitude * virtuals[o.spin] + v.amplitude];
  }

  double t2(std::size_t a, std::size_t b, std::size_t i, std::size_t j) const {
    if (active[a] && active[b] && active[i] && active[j]) return 0.0;
    const auto &v = orbitals[a], &w = orbitals[b], &o = orbitals[i],
               &p = orbitals[j];
    if (v.spin + w.spin != o.spin + p.spin) return 0.0;
    const auto ab = [&](std::size_t ia, std::size_t ib, std::size_t va,
                        std::size_t vb) {
      return (*doubles[1])[((ia * occupied[1] + ib) * virtuals[0] + va) *
                               virtuals[1] +
                           vb];
    };
    if (o.spin == p.spin) {
      if (restricted)
        return ab(o.amplitude, p.amplitude, v.amplitude, w.amplitude) -
               ab(o.amplitude, p.amplitude, w.amplitude, v.amplitude);
      return (*doubles[2 * o.spin])
          [((o.amplitude * occupied[o.spin] + p.amplitude) * virtuals[o.spin] +
            v.amplitude) *
               virtuals[o.spin] +
           w.amplitude];
    }
    if (o.spin == v.spin)
      return o.spin == 0
                 ? ab(o.amplitude, p.amplitude, v.amplitude, w.amplitude)
                 : ab(p.amplitude, o.amplitude, w.amplitude, v.amplitude);
    return o.spin == 0
               ? -ab(o.amplitude, p.amplitude, w.amplitude, v.amplitude)
               : -ab(p.amplitude, o.amplitude, v.amplitude, w.amplitude);
  }
};

const Context& evaluation_context() {
  static const Context context(
      {.index_basis_registry_shared_ptr =
           mbpt::make_min_sr_spaces(mbpt::SpinConvention::Default),
       .vacuum = Vacuum::SingleProduct});
  return context;
}

ExprPtr normal_ordered_bch(int level) {
  static std::mutex mutex;
  static std::array<ExprPtr, 3> cache;
  const std::lock_guard lock(mutex);
  if (!cache[level]) {
    const auto scope = set_scoped_default_context(
        {.index_basis_registry_shared_ptr =
             mbpt::make_min_sr_spaces(mbpt::SpinConvention::None),
         .vacuum = Vacuum::SingleProduct});
    const auto mbpt_scope = mbpt::set_scoped_default_mbpt_context(
        {.op_registry_ptr = mbpt::make_minimal_registry()});
    const auto cluster = mbpt::T(2);
    const auto sigma = cluster - adjoint(cluster);
    auto nested = mbpt::H();
    ExprPtr correction = ex<Constant>(0);
    std::int64_t factorial = 1;
    for (int order = 1; order <= level; ++order) {
      nested = nested * sigma - sigma * nested;
      simplify(nested);
      factorial *= order;
      correction += ex<Constant>(rational{1, factorial}) * nested;
    }
    nested = mbpt::F();
    for (int order = 1; order <= level + 1; ++order) {
      nested = nested * sigma - sigma * nested;
      simplify(nested);
    }
    correction += ex<Constant>(rational{1, factorial * (level + 1)}) * nested;
    simplify(correction);
    mbpt::op::lower_to_tensor_form(correction);
    simplify(correction);
    FWickTheorem wick{correction};
    // Cap emitted operators, not intermediate contraction paths.
    wick.full_contractions(false).use_topology(true).max_ops(4);
    auto result = wick.compute(false, true);
    simplify(result);
    cache[level] = std::move(result);
  }
  return cache[level]->clone();
}

char space_letter(const Index& index) {
  const bool beta = mbpt::to_spin(index.space().qns()) == mbpt::Spin::beta;
  switch (index.space().base_key().front()) {
    case L'i':
      return beta ? 'O' : 'o';
    case L'a':
      return beta ? 'V' : 'v';
    case L'p':
      return beta ? 'P' : 'p';
    default:
      throw std::logic_error("DUCC encountered an unsupported index space.");
  }
}

class DataYielder {
 public:
  explicit DataYielder(const Input& input) : _input(input) {}

  void set_free(const container::svector<Index>& indices) const {
    _free.clear();
    for (const auto& index : indices)
      _free.insert(std::wstring(index.full_label()));
  }

  void set_spins(const container::map<Index, Index>& indices) const {
    _spins = indices;
  }

  std::vector<std::size_t> spin_range(const Index& index) const {
    const auto spin = mbpt::to_spin(_spins.at(index).space().qns());
    if (spin != mbpt::Spin::alpha && spin != mbpt::Spin::beta)
      throw std::logic_error("DUCC numerical indices must have resolved spin.");
    const auto space = index.space().base_key().front();
    const bool free = _free.contains(std::wstring(index.full_label()));
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < _input.orbitals.size(); ++i) {
      if (_input.orbitals[i].spin != (spin == mbpt::Spin::beta ? 1 : 0))
        continue;
      if ((space == L'i' && i >= _input.nocc) ||
          (space == L'a' && i < _input.nocc))
        continue;
      if (!free || _input.active[i]) result.push_back(i);
    }
    return result;
  }

  ResultPtr operator()(sequant::meta::can_evaluate auto const& node) const {
    if (node->result_type() != ResultType::Tensor)
      return eval_result<ResultScalar<double>>(
          node->as_constant().template value<double>());
    const auto& tensor = node->expr()->template as<Tensor>();
    std::vector<Index> indices(tensor.const_braket_indices().begin(),
                               tensor.const_braket_indices().end());
    const auto label = tensor.label();
    std::string key(label.begin(), label.end());
    for (const auto& index : indices) key += space_letter(_spins.at(index));
    key += '#';
    for (const auto& index : indices)
      key += _free.contains(std::wstring(index.full_label())) ? 'A' : '-';
    auto found = _cache.find(key);
    if (found == _cache.end()) {
      std::vector<std::vector<std::size_t>> ranges;
      std::vector<std::size_t> extents;
      for (const auto& index : indices) {
        ranges.push_back(spin_range(index));
        extents.push_back(ranges.back().size());
      }
      if (indices.size() != 2 && indices.size() != 4)
        throw std::logic_error(
            "DUCC supports only rank-one and rank-two leaves.");
      if (indices.size() == 2)
        tensor_size({extents[0], extents[1]});
      else
        tensor_size({extents[0], extents[1], extents[2], extents[3]});
      BTensor block{btas::Range{extents}};
      const bool cluster = label == L"t";
      const bool adjoint_cluster =
          cluster && indices.front().space().base_key().front() == L'i';
      std::vector<std::size_t> position(indices.size(), 0);
      for (std::size_t flat = 0; flat < block.size(); ++flat) {
        std::array<std::size_t, 4> slots{};
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
          const auto source = adjoint_cluster
                                  ? (axis + indices.size() / 2) % indices.size()
                                  : axis;
          slots[axis] = ranges[source][position[source]];
        }
        if (label == L"f")
          block.data()[flat] = _input.one(slots[0], slots[1]);
        else if (label == L"g")
          block.data()[flat] =
              _input.two(slots[0], slots[1], slots[2], slots[3]);
        else if (cluster && indices.size() == 2)
          block.data()[flat] = _input.t1(slots[0], slots[1]);
        else if (cluster)
          block.data()[flat] =
              _input.t2(slots[0], slots[1], slots[2], slots[3]);
        else
          throw std::logic_error(
              "DUCC encountered an unsupported tensor label.");
        for (std::size_t axis = indices.size(); axis-- > 0;) {
          if (++position[axis] < extents[axis]) break;
          position[axis] = 0;
        }
      }
      found =
          _cache
              .emplace(key,
                       eval_result<ResultTensorBTAS<BTensor>>(std::move(block)))
              .first;
    }
    // Canonicalization can permute antisymmetric leaf slots.
    return node->canon_phase() == -1 ? found->second->mult_by_phase(-1)
                                     : found->second;
  }

 private:
  const Input& _input;
  mutable std::map<std::string, ResultPtr> _cache;
  mutable container::map<Index, Index> _spins;
  mutable std::set<std::wstring> _free;
};

struct SpinAssignment {
  std::size_t plan, multiplicity;
  container::map<Index, Index> indices;
};

struct PreparedContraction {
  container::svector<Index> free;
  container::svector<long> target;
  std::vector<EvalNodeBTAS> plans;
  std::vector<SpinAssignment> assignments;
};

PreparedContraction prepare_contraction(const ExprPtr& source,
                                        const DataYielder& yield) {
  const auto term = source->clone();
  PreparedContraction result;
  ExprPtr coefficient = ex<Product>(1, ExprPtrList{});
  auto visit = [&](const ExprPtr& factor) {
    if (factor.is<FNOperator>()) {
      if (!result.free.empty() || factor.as<FNOperator>().ncreators() > 2)
        throw std::logic_error("DUCC encountered an invalid Wick residual.");
      for (const auto& op : factor.as<FNOperator>().creators())
        result.free.push_back(op.index());
      for (const auto& op : factor.as<FNOperator>().annihilators())
        result.free.push_back(op.index());
    } else
      coefficient.as<Product>().append(1, factor, Product::Flatten::No);
  };
  if (term.is<Product>()) {
    coefficient = ex<Product>(term.as<Product>().scalar(), ExprPtrList{});
    for (const auto& factor : term.as<Product>().factors()) visit(factor);
  } else
    visit(term);
  result.target = EvalExprBTAS::index_hash(result.free) |
                  ranges::to<container::svector<long>>;
  const auto used = get_used_indices<container::set<Index>>(coefficient);
  const std::vector<Index> indices(used.begin(), used.end());
  container::map<Index, std::size_t> positions;
  for (std::size_t i = 0; i < indices.size(); ++i)
    positions.emplace(indices[i], i);
  using Endpoint = std::pair<std::size_t, int>;
  std::vector<std::vector<Endpoint>> endpoints(indices.size());
  std::vector<Symmetry> symmetries;
  std::vector<std::vector<int>> constraints;
  for (const auto& factor : coefficient.as<Product>().factors()) {
    if (!factor.is<Tensor>()) continue;
    const auto& tensor = factor.as<Tensor>();
    const auto ordinal = symmetries.size();
    symmetries.push_back(tensor.symmetry());
    std::vector<int> row(indices.size(), 0);
    for (const auto& index : tensor.bra()) {
      ++row[positions.at(index)];
      endpoints[positions.at(index)].emplace_back(ordinal, 0);
    }
    for (const auto& index : tensor.ket()) {
      --row[positions.at(index)];
      endpoints[positions.at(index)].emplace_back(ordinal, 1);
    }
    for (const auto& index : tensor.aux())
      endpoints[positions.at(index)].emplace_back(ordinal, 2);
    constraints.push_back(std::move(row));
  }
  std::vector<std::vector<std::size_t>> groups;
  std::vector<std::size_t> previous(indices.size(), indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const auto& edge = endpoints[i];
    if (indices[i].has_proto_indices() || edge.size() != 2 ||
        edge[0].first == edge[1].first || edge[0].second == 2 ||
        edge[1].second == 2 ||
        std::find(result.free.begin(), result.free.end(), indices[i]) !=
            result.free.end())
      continue;
    const auto symmetry = symmetries[edge[0].first];
    if ((symmetry != Symmetry::Antisymm && symmetry != Symmetry::Symm) ||
        symmetry != symmetries[edge[1].first])
      continue;
    auto group =
        std::find_if(groups.begin(), groups.end(), [&](const auto& candidate) {
          return indices[candidate.front()].space() == indices[i].space() &&
                 endpoints[candidate.front()] == edge;
        });
    if (group == groups.end())
      groups.push_back({i});
    else {
      previous[i] = group->back();
      group->push_back(i);
    }
  }
  yield.set_free(result.free);
  std::vector<std::array<Index, 2>> spin_indices;
  container::map<Index, Index> replacements;
  for (const auto& index : indices) {
    spin_indices.push_back(
        {mbpt::make_spinalpha(index), mbpt::make_spinbeta(index)});
    replacements.emplace(index, spin_indices.back()[0]);
  }
  std::map<std::vector<std::size_t>, std::size_t> plans;
  std::vector<int> spins(indices.size(), 0);
  auto enumerate = [&](auto&& self, std::size_t next) -> void {
    for (const auto& row : constraints) {
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
    if (next != indices.size()) {
      for (int spin : {0, 1}) {
        if (previous[next] != indices.size() && spin < spins[previous[next]])
          continue;
        spins[next] = spin;
        self(self, next + 1);
      }
      return;
    }
    for (std::size_t i = 0; i < indices.size(); ++i)
      replacements.at(indices[i]) = spin_indices[i][spins[i]];
    yield.set_spins(replacements);
    std::vector<std::size_t> dimensions;
    for (const auto& index : indices) {
      dimensions.push_back(yield.spin_range(index).size());
      if (dimensions.back() == 0) return;
    }
    auto plan = plans.find(dimensions);
    if (plan == plans.end()) {
      OptimizeOptions options;
      options.idx_to_extent = [&](const Index& index) {
        return dimensions[positions.at(index)];
      };
      const auto optimized = sequant::optimize(coefficient, options);
      const auto node = [&]() {
        if (result.free.empty())
          return binarize<EvalExprBTAS>(ResultExpr(Variable(L"E"), optimized));
        const auto rank = result.free.size() / 2;
        const Tensor target(
            L"R",
            bra(container::svector<Index>(result.free.begin(),
                                          result.free.begin() + rank)),
            ket(container::svector<Index>(result.free.begin() + rank,
                                          result.free.end())),
            Symmetry::Nonsymm);
        return binarize<EvalExprBTAS>(ResultExpr(target, optimized));
      }();
      plan = plans.emplace(dimensions, result.plans.size()).first;
      result.plans.push_back(node);
    }
    // Parallel equivalent edges contribute two cancelling permutation signs.
    std::size_t multiplicity = 1;
    for (const auto& group : groups) {
      std::size_t beta = 0, choices = 1;
      for (const auto i : group) beta += spins[i];
      for (std::size_t k = 1; k <= beta; ++k)
        choices = choices * (group.size() - beta + k) / k;
      multiplicity *= choices;
    }
    result.assignments.push_back({plan->second, multiplicity, replacements});
  };
  enumerate(enumerate, 0);
  for (const auto& index : result.free) (void)index.full_label();
  return result;
}

struct ContractionOutput {
  double scalar = 0.0;
  std::vector<double> one, two;
  explicit ContractionOutput(std::size_t n)
      : one(tensor_size({n, n}), 0.0), two(tensor_size({n, n, n, n}), 0.0) {}
};

void contract(const PreparedContraction& term, const DataYielder& yield,
              const std::vector<std::size_t>& local, std::size_t n,
              ContractionOutput& output) {
  yield.set_free(term.free);
  for (const auto& assignment : term.assignments) {
    yield.set_spins(assignment.indices);
    const auto& node = term.plans[assignment.plan];
    const auto weight = static_cast<double>(assignment.multiplicity);
    if (term.free.empty()) {
      output.scalar += weight * evaluate(node, yield)->get<double>();
      continue;
    }
    const auto value = evaluate(node, term.target, yield);
    const auto& block = value->get<BTensor>();
    std::vector<std::vector<std::size_t>> ranges;
    for (const auto& index : term.free) {
      auto range = yield.spin_range(index);
      for (auto& i : range) i = local.at(i);
      ranges.push_back(std::move(range));
    }
    auto& target = term.free.size() == 2 ? output.one : output.two;
    std::vector<std::size_t> position(term.free.size(), 0);
    for (std::size_t flat = 0; flat < block.size(); ++flat) {
      std::size_t address = 0;
      for (std::size_t axis = 0; axis < term.free.size(); ++axis)
        address = address * n + ranges[axis][position[axis]];
      target.at(address) += weight * block.data()[flat];
      for (std::size_t axis = term.free.size(); axis-- > 0;) {
        if (++position[axis] < ranges[axis].size()) break;
        position[axis] = 0;
      }
    }
  }
}

ContractionOutput dress(const Input& input, int level) {
  const auto n = input.output.size();
  ContractionOutput total(n);
  if (level > 0) {
    const auto expression = normal_ordered_bch(level);
    const auto scope = set_scoped_default_context(evaluation_context());
    std::vector<ExprPtr> source;
    if (expression.is<Sum>()) {
      for (const auto& term : expression.as<Sum>().summands())
        source.push_back(term);
    } else
      source.push_back(expression);
    std::vector<std::size_t> local(input.orbitals.size(), n);
    for (std::size_t i = 0; i < n; ++i) local[input.output[i]] = i;
    const auto entries = 1 + total.one.size() + total.two.size();
    const auto batch_size = std::max<std::size_t>(
        1, std::min<std::size_t>(
               256, (64 * 1024 * 1024) / sizeof(double) / entries));
    std::size_t requested = 1;
#ifdef _OPENMP
    requested = static_cast<std::size_t>(omp_get_max_threads());
#endif
    const auto workers = std::max<std::size_t>(
        1, std::min({requested, batch_size, source.size()}));
    DataYielder planning_yield(input);
    std::vector<DataYielder> yields;
    yields.reserve(workers);
    for (std::size_t i = 0; i < workers; ++i) yields.emplace_back(input);
    const scf::util::ScopedBlasThreads blas_threads;
    for (std::size_t begin = 0; begin < source.size(); begin += batch_size) {
      const auto count = std::min(batch_size, source.size() - begin);
      std::vector<PreparedContraction> jobs;
      std::vector<ContractionOutput> results;
      jobs.reserve(count);
      results.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        jobs.push_back(prepare_contraction(source[begin + i], planning_yield));
        results.emplace_back(n);
      }
      std::exception_ptr failure;
#ifdef _OPENMP
#pragma omp parallel if (workers > 1) num_threads(workers)
#endif
      {
        std::size_t worker = 0;
#ifdef _OPENMP
        worker = static_cast<std::size_t>(omp_get_thread_num());
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::size_t i = 0; i < count; ++i) {
          try {
            const auto worker_scope =
                set_scoped_default_context(evaluation_context());
            contract(jobs[i], yields[worker], local, n, results[i]);
          } catch (...) {
#ifdef _OPENMP
#pragma omp critical(ducc_contraction_failure)
#endif
            {
              if (!failure) failure = std::current_exception();
            }
          }
        }
      }
      if (failure) std::rethrow_exception(failure);
      // Term-order reduction is identical for serial and parallel evaluation.
      for (const auto& result : results) {
        total.scalar += result.scalar;
        for (std::size_t i = 0; i < total.one.size(); ++i)
          total.one[i] += result.one[i];
        for (std::size_t i = 0; i < total.two.size(); ++i)
          total.two[i] += result.two[i];
      }
    }
  }
  total.scalar += input.scalar;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      total.one[p * n + q] += input.one(input.output[p], input.output[q]);
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          total.two[index4(p, q, r, s, n)] +=
              0.25 * input.two(input.output[p], input.output[q],
                               input.output[r], input.output[s]);
    }
  const auto finite = [](double x) { return std::isfinite(x); };
  if (!finite(total.scalar) ||
      !std::all_of(total.one.begin(), total.one.end(), finite) ||
      !std::all_of(total.two.begin(), total.two.end(), finite))
    throw std::overflow_error("DUCC evaluation produced non-finite integrals.");
  return total;
}

}  // namespace

std::shared_ptr<data::Hamiltonian> DuccConstructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices) const {
  QDK_LOG_TRACE_ENTERING();
  const auto level = _settings->get<int64_t>("level");
  if (level < 0 || level > 2)
    throw std::invalid_argument("DUCC level must be between 0 and 2.");
  _validate_inputs(reference, hamiltonian, p_indices);
  if (!reference->has_container_type<data::AmplitudeContainer>())
    throw std::invalid_argument(
        "DUCC requires an AmplitudeContainer reference.");
  if (!hamiltonian->is_hermitian())
    throw std::invalid_argument("DUCC requires a Hermitian Hamiltonian.");
  const auto& amplitudes = reference->get_container<data::AmplitudeContainer>();
  const auto determinant_reference = amplitudes.get_wavefunction();
  _validate_inputs(determinant_reference, hamiltonian, p_indices);
  if (!amplitudes.has_t1_amplitudes() || !amplitudes.has_t2_amplitudes())
    throw std::invalid_argument(
        "DUCC requires stored T1 and T2 amplitudes over the Hamiltonian "
        "window.");
  const auto& determinants = determinant_reference->get_active_determinants();
  if (determinants.size() != 1)
    throw std::invalid_argument(
        "DUCC amplitudes must be relative to a single determinant.");
  const auto& determinant = determinants.front();
  const auto orbitals = hamiltonian->get_orbitals();
  Input input;
  input.restricted = hamiltonian->is_restricted();
  std::array<std::vector<std::size_t>, 2> window, kept, occupied, virtuals,
      inactive;
  const std::array spins{data::axes::alpha(), data::axes::beta()};
  for (int spin = 0; spin < 2; ++spin) {
    window[spin] =
        data::spin_channel_indices(orbitals->active_indices(), spins[spin]);
    kept[spin] = data::spin_channel_indices(p_indices, spins[spin]);
    inactive[spin] =
        data::spin_channel_indices(orbitals->inactive_indices(), spins[spin]);
    for (const auto& other :
         {reference->get_orbitals(), determinant_reference->get_orbitals()}) {
      if (data::spin_channel_indices(other->active_indices(), spins[spin]) !=
              window[spin] ||
          data::spin_channel_indices(other->inactive_indices(), spins[spin]) !=
              inactive[spin])
        throw std::invalid_argument(
            "DUCC Hamiltonian, amplitudes, and determinant must have identical "
            "active windows and inactive cores.");
    }
    for (std::size_t i = 0; i < window[spin].size(); ++i) {
      const bool filled = spin == 0 ? determinant.has_alpha_electron(i)
                                    : determinant.has_beta_electron(i);
      (filled ? occupied[spin] : virtuals[spin]).push_back(i);
      if (filled && !std::binary_search(kept[spin].begin(), kept[spin].end(),
                                        window[spin][i]))
        inactive[spin].push_back(window[spin][i]);
    }
    std::sort(inactive[spin].begin(), inactive[spin].end());
    input.occupied[spin] = occupied[spin].size();
    input.virtuals[spin] = virtuals[spin].size();
  }
  if (window[0].size() != window[1].size() || kept[0].size() != kept[1].size())
    throw std::invalid_argument(
        "DUCC requires equal alpha/beta orbital counts in W and P.");
  if (kept[0].empty())
    throw std::invalid_argument("DUCC requires a nonempty target space P.");
  if (input.restricted && (occupied[0] != occupied[1] || kept[0] != kept[1]))
    throw std::invalid_argument(
        "Restricted DUCC requires a closed-shell determinant and a common "
        "P-space; use unrestricted orbitals for an open-shell reference.");
  input.nmo = window[0].size();
  input.nocc = occupied[0].size() + occupied[1].size();
  tensor_size({2, input.nmo});
  const auto two_size =
      tensor_size({input.nmo, input.nmo, input.nmo, input.nmo});
  for (const auto& partition : {occupied, virtuals})
    for (int spin = 0; spin < 2; ++spin)
      for (std::size_t i = 0; i < partition[spin].size(); ++i)
        input.orbitals.push_back({partition[spin][i], i, spin});
  input.active.resize(input.orbitals.size(), false);
  for (int spin = 0; spin < 2; ++spin)
    for (const auto global : kept[spin]) {
      const auto spatial = static_cast<std::size_t>(
          std::lower_bound(window[spin].begin(), window[spin].end(), global) -
          window[spin].begin());
      for (std::size_t i = 0; i < input.orbitals.size(); ++i)
        if (input.orbitals[i].spin == spin &&
            input.orbitals[i].spatial == spatial) {
          input.kept[spin].push_back(i);
          input.active[i] = true;
        }
    }
  for (std::size_t i = 0; i < input.orbitals.size(); ++i)
    if (input.active[i]) input.output.push_back(i);
  const auto [h1a, h1b] = hamiltonian->get_one_body_integrals();
  const auto [gaa, gab, gbb] = hamiltonian->get_two_body_integrals();
  input.eri = {&gaa, &gab, &gbb};
  for (const auto* block : input.eri)
    if (static_cast<std::size_t>(block->size()) != two_size ||
        !block->allFinite())
      throw std::invalid_argument(
          "DUCC requires finite two-body integrals matching W.");
  for (const auto* block : {&h1a, &h1b})
    if (block->rows() != static_cast<Eigen::Index>(input.nmo) ||
        block->cols() != block->rows() || !block->allFinite())
      throw std::invalid_argument(
          "DUCC requires finite one-body integrals matching W.");
  const auto [t1a, t1b] = amplitudes.get_t1_amplitudes();
  const auto [t2ab, t2aa, t2bb] = amplitudes.get_t2_amplitudes();
  const auto real = [](const data::ContainerTypes::VectorVariant& value,
                       std::size_t size) {
    const auto* vector = std::get_if<Eigen::VectorXd>(&value);
    if (!vector || static_cast<std::size_t>(vector->size()) != size ||
        !vector->allFinite())
      throw std::invalid_argument(
          "DUCC requires finite real amplitudes with dimensions matching the "
          "reference window.");
    return vector;
  };
  input.singles = {
      real(t1a, tensor_size({input.occupied[0], input.virtuals[0]})),
      real(t1b, tensor_size({input.occupied[1], input.virtuals[1]}))};
  input.doubles = {
      real(t2aa, tensor_size({input.occupied[0], input.occupied[0],
                              input.virtuals[0], input.virtuals[0]})),
      real(t2ab, tensor_size({input.occupied[0], input.occupied[1],
                              input.virtuals[0], input.virtuals[1]})),
      real(t2bb, tensor_size({input.occupied[1], input.occupied[1],
                              input.virtuals[1], input.virtuals[1]}))};
  if (input.restricted) {
    if (!input.singles[0]->isApprox(*input.singles[1], 1e-12))
      throw std::invalid_argument(
          "Restricted DUCC requires equal alpha/beta singles.");
    const auto no = input.occupied[0], nv = input.virtuals[0];
    std::array<bool, 2> spatial{true, true}, antisymmetric{true, true};
    const auto ab = [&](std::size_t i, std::size_t j, std::size_t a,
                        std::size_t b) {
      return (*input.doubles[1])[((i * no + j) * nv + a) * nv + b];
    };
    for (std::size_t i = 0; i < no; ++i)
      for (std::size_t j = 0; j < no; ++j)
        for (std::size_t a = 0; a < nv; ++a)
          for (std::size_t b = 0; b < nv; ++b) {
            const auto value = ab(i, j, a, b);
            if (std::abs(value - ab(j, i, b, a)) > 1e-12)
              throw std::invalid_argument(
                  "Restricted DUCC requires pair-symmetric spatial doubles.");
            for (int spin = 0; spin < 2; ++spin) {
              const auto same =
                  (*input.doubles[2 * spin])[((i * no + j) * nv + a) * nv + b];
              spatial[spin] = spatial[spin] && std::abs(same - value) <= 1e-12;
              antisymmetric[spin] =
                  antisymmetric[spin] &&
                  std::abs(same - (value - ab(i, j, b, a))) <= 1e-12;
            }
          }
    for (int spin = 0; spin < 2; ++spin)
      if (!spatial[spin] && !antisymmetric[spin])
        throw std::invalid_argument(
            "Restricted DUCC same-spin doubles must match the spatial "
            "amplitudes or their antisymmetrization.");
  }
  input.fock = {h1a, h1b};
  for (std::size_t p = 0; p < input.orbitals.size(); ++p)
    for (std::size_t q = 0; q < input.orbitals.size(); ++q) {
      const auto &a = input.orbitals[p], &b = input.orbitals[q];
      if (a.spin != b.spin) continue;
      for (std::size_t m = 0; m < input.nocc; ++m)
        input.fock[a.spin](a.spatial, b.spatial) += input.two(p, m, q, m);
    }
  input.scalar = hamiltonian->get_core_energy();
  for (std::size_t m = 0; m < input.nocc; ++m) {
    const auto& orbital = input.orbitals[m];
    const auto& h1 = orbital.spin == 0 ? h1a : h1b;
    input.scalar +=
        0.5 * (h1(orbital.spatial, orbital.spatial) + input.one(m, m));
  }
  const auto dressed = dress(input, static_cast<int>(level));
  const auto n = input.output.size();
  std::vector<std::size_t> local(input.orbitals.size(), n), active_occupied;
  for (std::size_t i = 0; i < n; ++i) {
    local[input.output[i]] = i;
    if (input.output[i] < input.nocc) active_occupied.push_back(i);
  }
  const auto two = [&](std::size_t p, std::size_t q, std::size_t r,
                       std::size_t s) {
    return dressed.two[index4(p, q, r, s, n)] -
           dressed.two[index4(q, p, r, s, n)] -
           dressed.two[index4(p, q, s, r, n)] +
           dressed.two[index4(q, p, s, r, n)];
  };
  const auto one = [&](std::size_t p, std::size_t q) {
    double value = dressed.one[p * n + q];
    for (const auto m : active_occupied) value -= two(p, m, q, m);
    return value;
  };
  double core = dressed.scalar;
  for (const auto m : active_occupied) {
    core -= one(m, m);
    for (const auto k : active_occupied) core -= 0.5 * two(m, k, m, k);
  }
  for (auto& channel : input.kept)
    for (auto& i : channel) i = local[i];
  const auto np = kept[0].size();
  std::array<Eigen::MatrixXd, 2> one_body;
  std::array<Eigen::VectorXd, 3> two_body;
  for (int spin = 0; spin < (input.restricted ? 1 : 2); ++spin) {
    one_body[spin].resize(np, np);
    for (std::size_t p = 0; p < np; ++p)
      for (std::size_t q = 0; q < np; ++q)
        one_body[spin](p, q) = one(input.kept[spin][p], input.kept[spin][q]);
  }
  // Only the antisymmetric part of same-spin interactions is defined.
  for (int channel = 0; channel < 3; ++channel) {
    if (input.restricted && channel != 1) continue;
    const auto &left = input.kept[channel == 2 ? 1 : 0],
               &right = input.kept[channel == 0 ? 0 : 1];
    two_body[channel].resize(tensor_size({np, np, np, np}));
    for (std::size_t p = 0; p < np; ++p)
      for (std::size_t q = 0; q < np; ++q)
        for (std::size_t r = 0; r < np; ++r)
          for (std::size_t s = 0; s < np; ++s)
            two_body[channel][index4(p, q, r, s, np)] =
                (channel == 1 ? 1.0 : 0.5) *
                two(left[p], right[r], left[q], right[s]);
  }
  if (!std::isfinite(core) || !one_body[0].allFinite() ||
      !one_body[1].allFinite() || !two_body[0].allFinite() ||
      !two_body[1].allFinite() || !two_body[2].allFinite())
    throw std::overflow_error(
        "DUCC output assembly produced non-finite integrals.");
  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>>
      inactive_map;
  if (p_indices->symmetries()->has_axis(data::AxisName::Spin)) {
    for (int spin = 0; spin < 2; ++spin)
      inactive_map[spins[spin]] = std::vector<std::uint32_t>(
          inactive[spin].begin(), inactive[spin].end());
  } else {
    inactive_map[data::SymmetryLabel{}] =
        std::vector<std::uint32_t>(inactive[0].begin(), inactive[0].end());
  }
  const auto inactive_indices =
      std::make_shared<const data::SymmetryBlockedIndexSet>(
          p_indices->symmetries(), p_indices->extents(),
          std::move(inactive_map));
  std::shared_ptr<data::Orbitals> output_orbitals;
  if (std::dynamic_pointer_cast<data::ModelOrbitals>(orbitals)) {
    output_orbitals =
        std::make_shared<data::ModelOrbitals>(p_indices, inactive_indices);
  } else {
    std::optional<Eigen::MatrixXd> overlap;
    if (orbitals->has_overlap_matrix())
      overlap = orbitals->get_overlap_matrix();
    output_orbitals = std::make_shared<data::Orbitals>(
        orbitals->coefficients(),
        orbitals->has_energies() ? orbitals->energies() : nullptr, overlap,
        orbitals->has_basis_set() ? orbitals->get_basis_set() : nullptr,
        p_indices, inactive_indices);
  }
  const Eigen::MatrixXd empty_fock;
  if (input.restricted)
    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
            one_body[0], two_body[1], output_orbitals, core, empty_fock));
  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          one_body[0], one_body[1], two_body[0], two_body[1], two_body[2],
          output_orbitals, core, empty_fock, empty_fock));
}

}  // namespace qdk::chemistry::algorithms::microsoft
