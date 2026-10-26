/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <catch2/catch_template_test_macros.hpp>
#include <chrono>
#include <future>
#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <new>
#include <thread>

#include "ut_common.hpp"

namespace macis::test_support {

/**
 * @brief CPU reference executor for result parity and executor-lifetime tests.
 * @tparam N Determinant bit width.
 * @note Borrows the Hamiltonian and determinant iterators until finish or
 * drain. A pending asynchronous task and injected phase failures exercise
 * unwinding; the numerical search and diagonalization use the CPU routines.
 */
template <size_t N>
class CpuExecution final : public macis::ASCIExecutionContext<N> {
 public:
  using Hamiltonian = typename macis::ASCIExecutionContext<N>::Hamiltonian;
  using Iterator = typename macis::ASCIExecutionContext<N>::Iterator;
  enum class Failure { none, search, start, finish };

  explicit CpuExecution(Hamiltonian& ham_gen) : _ham_gen(ham_gen) {}
  ~CpuExecution() noexcept override { drain(); }

  Failure failure = Failure::none;
  size_t searches = 0;
  size_t updates = 0;
  size_t drains = 0;

  bool pending() const { return _build.valid(); }

  std::vector<macis::wfn_t<N>> search(macis::ASCISettings settings,
                                      size_t target, Iterator begin,
                                      Iterator end, double E0,
                                      const std::vector<double>& C, size_t norb,
                                      Hamiltonian& ham_gen) override {
    ++searches;
    if (failure == Failure::search) throw std::runtime_error("search failure");
    return macis::asci_search(
        settings, target, begin, end, E0, C, norb, ham_gen.T(), ham_gen.G_red(),
        ham_gen.V_red(), ham_gen.V(), ham_gen MACIS_MPI_CODE(, MPI_COMM_WORLD));
  }

  void start_diagonalization(Iterator begin, Iterator end,
                             const Hamiltonian& ham_gen,
                             const macis::MCSCFSettings& settings) override {
    REQUIRE(&ham_gen == &_ham_gen);
    REQUIRE_FALSE(pending());
    _begin = begin;
    _end = end;
    _settings = settings;
    _build = std::async(std::launch::async, [] {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
    if (failure == Failure::start) throw std::runtime_error("start failure");
  }

  double finish_diagonalization(std::vector<double>& C) override {
    if (failure == Failure::finish) throw std::runtime_error("finish failure");
    _build.get();
    return macis::selected_ci_diag<int64_t>(
        _begin, _end, _ham_gen, _settings.ci_matel_tol,
        _settings.ci_max_subspace, _settings.ci_res_tol,
        C MACIS_MPI_CODE(, MPI_COMM_WORLD));
  }

  void update_integrals(const Hamiltonian&) override {
    REQUIRE_FALSE(pending());
    ++updates;
  }

  void drain() noexcept override {
    ++drains;
    _build = {};
  }

 private:
  Hamiltonian& _ham_gen;
  Iterator _begin, _end;
  macis::MCSCFSettings _settings;
  std::future<void> _build;
};

/**
 * @brief Synthetic executor that forces ASCI refinement into union
 * stabilization.
 * @note Alternates overlapping two-determinant spaces and nonconverging
 * energies, then accepts the three-determinant union. Records requested
 * targets, build sizes and the union warm start; it is not a numerical solver.
 */
class AlternatingExecution final : public macis::ASCIExecutionContext<64> {
 public:
  std::vector<macis::wfn_t<64>> space;
  std::vector<size_t> targets;
  std::vector<size_t> builds;
  std::vector<double> union_guess;

  std::vector<macis::wfn_t<64>> search(macis::ASCISettings, size_t target,
                                       Iterator, Iterator, double,
                                       const std::vector<double>&, size_t,
                                       Hamiltonian&) override {
    targets.push_back(target);
    if (target == space.size()) return space;
    return {space[0], space[targets.size() % 2 ? 1 : 2]};
  }

  void start_diagonalization(Iterator begin, Iterator end, const Hamiltonian&,
                             const macis::MCSCFSettings&) override {
    builds.push_back(std::distance(begin, end));
  }

  double finish_diagonalization(std::vector<double>& C) override {
    if (builds.size() == 4) union_guess = C;
    C.assign(builds.back(), 1.0 / std::sqrt(double(builds.back())));
    if (builds.size() >= 4) return -1.2;
    return builds.size() % 2 ? -1.1 : -1.0;
  }

  void update_integrals(const Hamiltonian&) override {}
  void drain() noexcept override {}
};

}  // namespace macis::test_support

using macis::test_support::AlternatingExecution;
using macis::test_support::CpuExecution;

TEMPLATE_TEST_CASE("ASCI execution boundary", "[asci_execution]",
                   macis::wfn_t<64>, macis::wfn_t<512>) {
#ifdef MACIS_ENABLE_MPI
  if (macis::comm_size(MPI_COMM_WORLD) > 1) SKIP("Single-rank executor");
#endif
  constexpr size_t N = macis::wavefunction_traits<TestType>::size();
  constexpr size_t norb = 6;
  std::vector<double> T(norb * norb), V(norb * norb * norb * norb, 0.0);
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q < norb; ++q)
      T[p + q * norb] = p == q ? -2.0 + 0.2 * p : 0.02;
  macis::SortedDoubleLoopHamiltonianGenerator<TestType> ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
  const auto hf =
      macis::wavefunction_traits<TestType>::canonical_hf_determinant(3, 2);
  const double E0 = ham_gen.matrix_element(hf, hf);
  macis::ASCISettings settings;
  settings.constraint_level = 0;
  settings.ntdets_max = 16;
  settings.ntdets_min = 4;
  settings.max_refine_iter = 1;
  settings.refine_energy_tol = 1.0;
  macis::MCSCFSettings ci;
  CpuExecution<N> execution(ham_gen);

  SECTION(
      "CPU remains the default and a supplied executor preserves the result") {
    const auto cpu = macis::asci_iter<N, int64_t>(settings, ci, 16, E0, {hf},
                                                  {1.0}, ham_gen, norb);
    const auto delegated = macis::asci_iter<N, int64_t>(
        settings, ci, 16, E0, {hf}, {1.0}, ham_gen, norb,
        nullptr MACIS_MPI_CODE(, MPI_COMM_WORLD), &execution);
    REQUIRE(std::get<0>(delegated) == std::get<0>(cpu));
    REQUIRE(std::get<1>(delegated) == std::get<1>(cpu));
    REQUIRE(std::get<2>(delegated) == std::get<2>(cpu));
    REQUIRE(execution.searches == 1);
    REQUIRE_FALSE(execution.pending());
  }

  SECTION("Failures propagate and pending work is drained before returning") {
    for (auto failure :
         {CpuExecution<N>::Failure::search, CpuExecution<N>::Failure::start,
          CpuExecution<N>::Failure::finish}) {
      execution.failure = failure;
      REQUIRE_THROWS_AS(
          (macis::asci_iter<N, int64_t>(
              settings, ci, 16, E0, {hf}, {1.0}, ham_gen, norb,
              nullptr MACIS_MPI_CODE(, MPI_COMM_WORLD), &execution)),
          std::runtime_error);
      REQUIRE_FALSE(execution.pending());
    }
    REQUIRE(execution.drains == 2);
  }

  SECTION("Grow rotations refresh the executor and refine reuses it") {
    settings.grow_with_rot = true;
    settings.rot_size_start = 1;
    auto [E, dets, C] = macis::asci_grow<N, int64_t>(
        settings, ci, E0, {hf}, {1.0}, ham_gen,
        norb MACIS_MPI_CODE(, MPI_COMM_WORLD), &execution);
    REQUIRE(execution.updates > 0);
    const size_t grow_searches = execution.searches;
    std::tie(E, dets, C) = macis::asci_refine<N, int64_t>(
        settings, ci, E, std::move(dets), std::move(C), ham_gen,
        norb MACIS_MPI_CODE(, MPI_COMM_WORLD), &execution);
    REQUIRE(execution.searches > grow_searches);
    REQUIRE_FALSE(execution.pending());
  }
}

TEST_CASE("ASCI execution preserves union stabilization", "[asci_execution]") {
#ifdef MACIS_ENABLE_MPI
  SKIP("Union stabilization is not supported in MPI builds");
#endif
  constexpr size_t norb = 6;
  std::vector<double> T(norb * norb, 0.0), V(norb * norb * norb * norb, 0.0);
  macis::SortedDoubleLoopHamiltonianGenerator<macis::wfn_t<64>> ham_gen(
      macis::matrix_span<double>(T.data(), norb, norb),
      macis::rank4_span<double>(V.data(), norb, norb, norb, norb));
  AlternatingExecution execution;
  const auto hf =
      macis::wavefunction_traits<macis::wfn_t<64>>::canonical_hf_determinant(3,
                                                                             2);
  auto first = hf, second = hf;
  first.flip(0).flip(3);
  second.flip(1).flip(4);
  execution.space = {hf, first, second};
  std::sort(execution.space.begin(), execution.space.end(),
            macis::wavefunction_traits<macis::wfn_t<64>>::spin_comparator{});
  macis::ASCISettings settings;
  settings.max_refine_iter = 4;
  settings.ntdets_min = 1;
  const auto result = macis::asci_refine<64, int64_t>(
      settings, macis::MCSCFSettings{}, -1.0,
      {execution.space[0], execution.space[2]},
      {std::sqrt(0.5), std::sqrt(0.5)}, ham_gen,
      norb MACIS_MPI_CODE(, MPI_COMM_WORLD), &execution);
  REQUIRE(std::get<0>(result) == -1.2);
  REQUIRE(std::get<1>(result) == execution.space);
  REQUIRE(execution.targets == std::vector<size_t>{2, 2, 2, 3});
  REQUIRE(execution.builds == std::vector<size_t>{2, 2, 2, 3, 3});
  REQUIRE(execution.union_guess.size() == 3);
  REQUIRE(std::count(execution.union_guess.begin(), execution.union_guess.end(),
                     0.0) == 1);
}

TEST_CASE("ASCI constraint work providers preserve the partition",
          "[asci_execution]") {
  using Wfn = macis::wfn_t<64>;
  using Traits = macis::wavefunction_traits<Wfn>;
  using Constraint = macis::alpha_constraint<Traits>;
  const size_t norb = 8, ns_beta = 15, nd_beta = 30;
  std::vector<Wfn> dets = {Traits::canonical_hf_determinant(4, 3)};
  auto other = dets.front();
  other.flip(0).flip(6);
  dets.push_back(other);
  std::sort(dets.begin(), dets.end(), Traits::spin_comparator{});
  const auto unique_alpha = macis::get_unique_alpha(dets.begin(), dets.end());
  auto work = [&](const auto& constraints) {
    std::vector<size_t> result(constraints.size(), 0);
    for (size_t i = 0; i < constraints.size(); ++i)
      for (const auto& [alpha, count] : unique_alpha)
        result[i] += count * macis::constraint_histogram(
                                 alpha, ns_beta, nd_beta, constraints[i]);
    return result;
  };
  auto triplet_work = [&] {
    std::vector<Constraint> triplets;
    for (size_t i = 0; i < norb; ++i)
      for (size_t j = 0; j < i; ++j)
        for (size_t k = 0; k < j; ++k)
          triplets.push_back(Constraint::make_triplet(i, j, k));
    return work(triplets);
  };
  for (size_t minimum_level : {size_t{0}, size_t{1}}) {
    const auto reference = macis::gen_constraints_general<Wfn>(
        2, norb, ns_beta, nd_beta, unique_alpha, 4, minimum_level);
    const auto delegated = macis::gen_constraints_general<Wfn>(
        2, norb, ns_beta, nd_beta, unique_alpha, 4, minimum_level, -1, work,
        triplet_work);
    REQUIRE(delegated.size() == reference.size());
    for (size_t i = 0; i < reference.size(); ++i) {
      REQUIRE(delegated[i].first.C() == reference[i].first.C());
      REQUIRE(delegated[i].first.B() == reference[i].first.B());
      REQUIRE(delegated[i].first.C_min() == reference[i].first.C_min());
      REQUIRE(delegated[i].second == reference[i].second);
    }
  }
  auto invalid_work = [](const auto&) { return std::vector<size_t>{}; };
  REQUIRE_THROWS_AS(
      macis::gen_constraints_general<Wfn>(2, norb, ns_beta, nd_beta,
                                          unique_alpha, 4, 0, -1, invalid_work),
      std::invalid_argument);
  auto invalid_triplets = [] { return std::vector<size_t>{}; };
  REQUIRE_THROWS_AS(macis::gen_constraints_general<Wfn>(
                        2, norb, ns_beta, nd_beta, unique_alpha, 4, 0, -1, work,
                        invalid_triplets),
                    std::invalid_argument);
}

TEMPLATE_TEST_CASE("ASCI core selection preserves order and coefficients",
                   "[asci_execution][asci_core_selection]", macis::wfn_t<64>,
                   macis::wfn_t<512>) {
  const std::vector<TestType> dets = {8, 2, 1, 4, 16};
  const std::vector<double> C = {0.2, -0.5, 0.5, 0.0, -0.8};
  std::vector<TestType> selected_dets;
  std::vector<double> selected_C;

  macis::select_ci_on_coeff(dets, C, 0.5, selected_dets, selected_C);
  REQUIRE(selected_dets == std::vector<TestType>{2, 1, 16});
  REQUIRE(selected_C == std::vector<double>{-0.5, 0.5, -0.8});

  macis::select_ci_on_coeff(dets, C, 1.0, selected_dets, selected_C);
  REQUIRE(selected_dets.empty());
  REQUIRE(selected_C.empty());

  macis::select_ci_on_coeff(dets, C, 0.0, selected_dets, selected_C);
  REQUIRE(selected_dets == dets);
  REQUIRE(selected_C == C);

  macis::select_ci_on_coeff(std::vector<TestType>{}, {}, 0.5, selected_dets,
                            selected_C);
  REQUIRE(selected_dets.empty());
  REQUIRE(selected_C.empty());
}

TEST_CASE("ASCI core selection propagates resize failures",
          "[asci_execution][asci_core_selection]") {
  struct ThrowingDeterminant {
    explicit ThrowingDeterminant(bool fail = true) {
      if (fail) throw std::bad_alloc{};
    }
  };
  const std::vector<ThrowingDeterminant> dets = {ThrowingDeterminant{false}};
  std::vector<ThrowingDeterminant> selected_dets;
  std::vector<double> selected_C;

  REQUIRE_THROWS_AS(
      macis::select_ci_on_coeff(dets, {1.0}, 0.5, selected_dets, selected_C),
      std::bad_alloc);
}
