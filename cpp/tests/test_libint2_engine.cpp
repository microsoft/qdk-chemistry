// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <libint2/shell.h>

#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include "util/libint2_engine.h"

namespace {

namespace integrals = qdk::chemistry::scf::libint2_util;

TEST(Libint2EngineTest, CopiesOwnTheirBuffersAndMovesRetainResults) {
  const libint2::Shell centered({1.0}, {{0, true, {1.0}}}, {{0.0, 0.0, 0.0}});
  const libint2::Shell shifted({1.0}, {{0, true, {1.0}}}, {{1.0, 0.0, 0.0}});
  integrals::Engine original(integrals::Operator::overlap, 1, 0);
  const auto& original_results = original.results();
  original.compute1(centered, centered);
  ASSERT_EQ(original_results.size(), 1);
  EXPECT_NEAR(original_results[0][0], 1.0, 1e-14);

  auto copy = original;
  const auto& copy_results = copy.results();
  copy.compute1(centered, shifted);
  EXPECT_NEAR(copy_results[0][0], std::exp(-0.5), 1e-14);
  EXPECT_NEAR(original_results[0][0], 1.0, 1e-14);

  integrals::Engine assigned;
  assigned = copy;
  auto moved = std::move(assigned);
  moved.compute1(shifted, shifted);
  EXPECT_NEAR(moved.results()[0][0], 1.0, 1e-14);
  EXPECT_NEAR(copy_results[0][0], std::exp(-0.5), 1e-14);
}

TEST(Libint2EngineTest, CoulombAndRangeSeparatedKernelsMatchAnalyticIntegrals) {
  const libint2::Shell shell({1.0}, {{0, true, {1.0}}}, {{0.0, 0.0, 0.0}});
  integrals::Engine coulomb(integrals::Operator::coulomb, 1, 0);
  const auto& coulomb_results =
      coulomb.compute2<integrals::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
          shell, shell, shell, shell);
  ASSERT_NE(coulomb_results[0], nullptr);
  EXPECT_NEAR(coulomb_results[0][0], 2.0 / std::sqrt(std::numbers::pi), 1e-13);

  integrals::Engine screened(integrals::Operator::erf_coulomb, 1, 0);
  const auto& screened_results = screened.results();
  for (double omega : {0.5, 1.0, 2.0}) {
    screened.set_params(omega);
    screened
        .compute2<integrals::Operator::erf_coulomb, libint2::BraKet::xx_xx, 0>(
            shell, shell, shell, shell);
    ASSERT_NE(screened_results[0], nullptr);
    EXPECT_NEAR(screened_results[0][0],
                2.0 / std::sqrt(std::numbers::pi) * omega /
                    std::sqrt(omega * omega + 1.0),
                1e-13);
  }
}

TEST(Libint2EngineTest, NuclearParametersUpdateRetainedResultView) {
  const libint2::Shell shell({1.0}, {{0, true, {1.0}}}, {{0.0, 0.0, 0.0}});
  integrals::Engine engine(integrals::Operator::nuclear, 1, 0);
  const auto& results = engine.results();
  for (double charge : {1.0, 2.0}) {
    engine.set_params(std::vector<std::pair<double, std::array<double, 3>>>{
        {charge, {0.0, 0.0, 0.0}}});
    engine.compute1(shell, shell);
    ASSERT_EQ(results.size(), 1);
    EXPECT_NEAR(results[0][0], -charge * std::sqrt(8.0 / std::numbers::pi),
                1e-13);
  }
}

TEST(Libint2EngineTest, BasisCopyOutlivesOriginalStorageOwner) {
  auto basis = [] {
    integrals::Basis original(std::vector<libint2::Shell>{
        libint2::Shell({1.0}, {{0, true, {1.0}}}, {{0.0, 0.0, 0.0}})});
    return integrals::Basis(original);
  }();
  EXPECT_EQ(basis.size(), 1);
  EXPECT_EQ(basis.nbf(), 1);
  EXPECT_EQ(basis.max_nprim(), 1);
  EXPECT_EQ(basis.max_l(), 0);
  EXPECT_EQ(basis.shell2bf(), std::vector<size_t>{0});

  integrals::Engine engine(integrals::Operator::overlap, 1, 0);
  engine.compute1(basis[0], basis[0]);
  EXPECT_NEAR(engine.results()[0][0], 1.0, 1e-14);
}

}  // namespace
