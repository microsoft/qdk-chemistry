// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <SeQuant/core/context.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/expressions/variable.hpp>
#include <SeQuant/core/op.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <qdk/chemistry/utils/tensor_index.hpp>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "../src/qdk/chemistry/algorithms/microsoft/sequant_evaluation.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using namespace sequant;
namespace evaluation =
    qdk::chemistry::algorithms::microsoft::sequant_evaluation;
using qdk::chemistry::utils::index4;

auto scoped_context() {
  return set_scoped_default_context(
      {.index_basis_registry_shared_ptr =
           mbpt::make_min_sr_spaces(mbpt::SpinConvention::None),
       .vacuum = Vacuum::SingleProduct});
}

// Alpha 0, 1 | 3, 4 and beta 2 | 5, 6, 7; the target is a permuted subset.
evaluation::SpinOrbitalSpace test_space() {
  evaluation::SpinOrbitalSpace space;
  space.occupied = {std::vector<std::size_t>{0, 1},
                    std::vector<std::size_t>{2}};
  space.virtuals = {std::vector<std::size_t>{3, 4},
                    std::vector<std::size_t>{5, 6, 7}};
  space.target = {5, 0, 3, 2, 7};
  return space;
}

int spin(std::size_t orbital) { return orbital == 2 || orbital > 4 ? 1 : 0; }

double f_element(std::size_t p, std::size_t q) {
  return spin(p) == spin(q) ? 1.0 + 10.0 * p + q : 0.0;
}

// Antisymmetric within bra and within ket, and zero unless spin is conserved.
template <typename Pair>
double antisymmetric(Pair pair, std::size_t p, std::size_t q, std::size_t r,
                     std::size_t s) {
  const auto direct = [&](std::size_t a, std::size_t b, std::size_t c,
                          std::size_t d) {
    return spin(a) == spin(c) && spin(b) == spin(d) ? pair(a, c) * pair(b, d)
                                                    : 0.0;
  };
  return direct(p, q, r, s) - direct(q, p, r, s);
}

double g_element(std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  const auto pair = [](std::size_t a, std::size_t b) {
    return 0.3 + 0.07 * a - 0.02 * b;
  };
  return antisymmetric(pair, p, q, r, s);
}

double t_element(std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  const auto pair = [](std::size_t a, std::size_t b) {
    return -0.1 + 0.05 * a * b;
  };
  return antisymmetric(pair, p, q, r, s);
}

double test_leaf(std::wstring_view label, std::span<const std::size_t> slots) {
  if (label == L"f") return f_element(slots[0], slots[1]);
  if (label == L"g") return g_element(slots[0], slots[1], slots[2], slots[3]);
  return t_element(slots[0], slots[1], slots[2], slots[3]);
}

ExprPtr fock_operator() {
  const Index p1{L"p_1"}, p2{L"p_2"};
  return ex<Tensor>(L"f", bra{p1}, ket{p2}) * ex<FNOperator>(cre{p1}, ann{p2});
}

TEST(SeQuantEvaluationTest, OneBodyTermFollowsTargetOrderAndSpinBlocks) {
  const auto context = scoped_context();
  const auto space = test_space();
  const auto result = evaluation::evaluate(fock_operator(), space, test_leaf);
  const auto n = space.target.size();
  EXPECT_EQ(result.scalar, 0.0);
  for (std::size_t x = 0; x < n; ++x)
    for (std::size_t y = 0; y < n; ++y)
      EXPECT_EQ(result.one_body[x * n + y],
                f_element(space.target[x], space.target[y]));
  EXPECT_TRUE(std::all_of(result.two_body.begin(), result.two_body.end(),
                          [](double value) { return value == 0.0; }));
}

TEST(SeQuantEvaluationTest, TwoBodyTermIsAccumulatedPerSpinPattern) {
  const auto context = scoped_context();
  const Index p1{L"p_1"}, p2{L"p_2"}, p3{L"p_3"}, p4{L"p_4"};
  const auto expression = ex<Tensor>(L"g", bra{p1, p2}, ket{p3, p4}) *
                          ex<FNOperator>(cre{p1, p2}, ann{p3, p4});
  const auto element = [](std::size_t p, std::size_t q, std::size_t r,
                          std::size_t s) {
    return 1.0 + 1000.0 * p + 100.0 * q + 10.0 * r + s;
  };
  const auto space = test_space();
  const auto result = evaluation::evaluate(
      expression, space,
      [&](std::wstring_view, std::span<const std::size_t> slots) {
        return element(slots[0], slots[1], slots[2], slots[3]);
      });
  const auto n = space.target.size();
  const auto& orbitals = space.target;
  for (std::size_t x = 0; x < n; ++x)
    for (std::size_t y = 0; y < n; ++y)
      for (std::size_t z = 0; z < n; ++z)
        for (std::size_t w = 0; w < n; ++w) {
          const auto p = orbitals[x], q = orbitals[y], r = orbitals[z],
                     s = orbitals[w];
          const bool conserved = spin(p) + spin(q) == spin(r) + spin(s);
          EXPECT_EQ(result.two_body[index4(x, y, z, w, n)],
                    conserved ? element(p, q, r, s) : 0.0);
        }
  EXPECT_TRUE(std::all_of(result.one_body.begin(), result.one_body.end(),
                          [](double value) { return value == 0.0; }));
}

TEST(SeQuantEvaluationTest, ContractionsMatchSpinOrbitalSums) {
  const auto context = scoped_context();
  const auto space = test_space();
  const std::vector<std::size_t> occupied{0, 1, 2}, virtuals{3, 4, 5, 6, 7};
  const Index i1{L"i_1"}, i2{L"i_2"}, a1{L"a_1"}, a2{L"a_2"};
  const auto energy =
      ex<Constant>(rational{1, 4}) *
      ex<Tensor>(L"g", bra{i1, i2}, ket{a1, a2}, Symmetry::Antisymm) *
      ex<Tensor>(L"t", bra{a1, a2}, ket{i1, i2}, Symmetry::Antisymm);
  double expected = 0.0;
  for (const auto i : occupied)
    for (const auto j : occupied)
      for (const auto a : virtuals)
        for (const auto b : virtuals)
          expected += 0.25 * g_element(i, j, a, b) * t_element(a, b, i, j);
  ASSERT_NE(expected, 0.0);
  EXPECT_NEAR(evaluation::evaluate(energy, space, test_leaf).scalar, expected,
              1e-12 * std::abs(expected));

  const Index p1{L"p_1"}, p2{L"p_2"};
  const auto product = ex<Tensor>(L"f", bra{p1}, ket{i1}) *
                       ex<Tensor>(L"f", bra{i1}, ket{p2}) *
                       ex<FNOperator>(cre{p1}, ann{p2});
  const auto result = evaluation::evaluate(product, space, test_leaf);
  const auto n = space.target.size();
  for (std::size_t x = 0; x < n; ++x)
    for (std::size_t y = 0; y < n; ++y) {
      double sum = 0.0;
      for (const auto i : occupied)
        sum += f_element(space.target[x], i) * f_element(i, space.target[y]);
      EXPECT_NEAR(result.one_body[x * n + y], sum, 1e-9);
    }
}

TEST(SeQuantEvaluationTest, RejectsInconsistentSpaces) {
  const auto context = scoped_context();
  std::vector<evaluation::SpinOrbitalSpace> spaces(5, test_space());
  spaces[0].occupied[1].push_back(1);
  spaces[1].virtuals[0].push_back(0);
  spaces[2].occupied[0].push_back(0);
  spaces[3].target.push_back(8);
  spaces[4].target.push_back(0);
  for (const auto& space : spaces)
    EXPECT_THROW(evaluation::evaluate(fock_operator(), space, test_leaf),
                 std::invalid_argument);
}

TEST(SeQuantEvaluationTest, RejectsUnsupportedExpressions) {
  const auto context = scoped_context();
  const Index p1{L"p_1"}, p2{L"p_2"}, i1{L"i_1"}, i2{L"i_2"}, a1{L"a_1"};
  const auto f = [](const Index& bra_index, const Index& ket_index) {
    return ex<Tensor>(L"f", bra{bra_index}, ket{ket_index});
  };
  const auto excitation = [&] { return ex<FNOperator>(cre{p1}, ann{p2}); };
  const std::vector<ExprPtr> expressions{
      f(i1, i1),
      ex<Tensor>(L"g", bra{p1, i1}, ket{p2, i1}) * excitation(),
      f(i1, a1) * ex<Tensor>(L"g", bra{a1, i2}, ket{i1, i2}),
      f(i1, i1) * f(i2, i2),
      ex<Variable>(L"x") * fock_operator(),
      ex<Constant>(2) * excitation(),
      f(p1, p2) * ex<FNOperator>(cre{p1}, ann{p2, i1})};
  for (const auto& expression : expressions)
    EXPECT_THROW(evaluation::evaluate(expression, test_space(), test_leaf),
                 std::invalid_argument);
}

TEST(SeQuantEvaluationTest, ConstantTermAddsToTheScalar) {
  const auto context = scoped_context();
  const auto result = evaluation::evaluate(
      fock_operator() + ex<Constant>(rational{3, 2}), test_space(), test_leaf);
  EXPECT_EQ(result.scalar, 1.5);
}

TEST(SeQuantEvaluationTest, LeafErrorsPropagateFromEveryWorker) {
  const auto context = scoped_context();
  const Index p1{L"p_1"}, p2{L"p_2"}, p3{L"p_3"}, p4{L"p_4"}, i1{L"i_1"};
  const auto contraction = ex<Tensor>(L"f", bra{p1}, ket{i1}) *
                           ex<Tensor>(L"f", bra{i1}, ket{p2}) *
                           ex<FNOperator>(cre{p1}, ann{p2});
  const auto two_body = ex<Tensor>(L"g", bra{p1, p2}, ket{p3, p4}) *
                        ex<FNOperator>(cre{p1, p2}, ann{p3, p4});
  const auto expression = fock_operator() + contraction + two_body;
  const auto failing = [](std::wstring_view label,
                          std::span<const std::size_t> slots) {
    if (std::find(slots.begin(), slots.end(), 7) != slots.end())
      throw std::runtime_error("leaf failure");
    return test_leaf(label, slots);
  };
#ifdef _OPENMP
  struct RestoreThreads {
    int threads = omp_get_max_threads();
    ~RestoreThreads() { omp_set_num_threads(threads); }
  } restore;
  for (int workers : {1, 4}) {
    SCOPED_TRACE(workers);
    omp_set_num_threads(workers);
    EXPECT_THROW(evaluation::evaluate(expression, test_space(), failing),
                 std::runtime_error);
  }
#else
  EXPECT_THROW(evaluation::evaluate(expression, test_space(), failing),
               std::runtime_error);
#endif
}

}  // namespace
