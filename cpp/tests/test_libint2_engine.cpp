// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <libint2/shell.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
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

TEST(Libint2EngineTest, CopyAssignmentReconfiguresAndSupportsMovedFromTargets) {
  const libint2::Shell shell({1.0}, {{0, true, {1.0}}}, {{0.0, 0.0, 0.0}});
  integrals::Engine overlap(integrals::Operator::overlap, 1, 0);
  integrals::Engine coulomb(integrals::Operator::coulomb, 3, 2);
  const auto& results = overlap.results();

  overlap = coulomb;
  overlap.compute2<integrals::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
      shell, shell, shell, shell);
  EXPECT_NEAR(results[0][0], 2.0 / std::sqrt(std::numbers::pi), 1e-13);

  auto moved = std::move(overlap);
  overlap = moved;
  overlap.compute2<integrals::Operator::coulomb, libint2::BraKet::xx_xx, 0>(
      shell, shell, shell, shell);
  EXPECT_NEAR(results[0][0], 2.0 / std::sqrt(std::numbers::pi), 1e-13);
  EXPECT_NE(results[0], moved.results()[0]);

  const auto* buffer = results[0];
  const auto& same_engine = overlap;
  overlap = same_engine;
  EXPECT_EQ(results[0], buffer);
}

TEST(Libint2EngineTest, UnusableCopySourceLeavesDestinationUnchanged) {
  const libint2::Shell shell({1.0}, {{0, true, {1.0}}}, {{0.0, 0.0, 0.0}});
  integrals::Engine target(integrals::Operator::overlap, 1, 0);
  target.compute1(shell, shell);
  const auto* buffer = target.results()[0];
  integrals::Engine uninitialized;
  EXPECT_THROW(target = uninitialized, std::logic_error);
  EXPECT_THROW({ integrals::Engine copy(uninitialized); }, std::logic_error);
  EXPECT_EQ(target.results()[0], buffer);
  EXPECT_NEAR(buffer[0], 1.0, 1e-14);

  auto moved = std::move(target);
  EXPECT_THROW(moved = target, std::logic_error);
  EXPECT_EQ(moved.results()[0], buffer);
  target = moved;
  target.compute1(shell, shell);
  EXPECT_NEAR(target.results()[0][0], 1.0, 1e-14);
}

TEST(Libint2EngineTest, PoolMovesPrototypeAndKeepsWorkerBuffersIndependent) {
  const libint2::Shell centered({1.0}, {{0, true, {1.0}}}, {{0.0, 0.0, 0.0}});
  integrals::Engine prototype(integrals::Operator::overlap, 1, 0);
  const auto* original_buffer = prototype.compute1(centered, centered)[0];
  auto engines = integrals::Engine::make_pool(4, std::move(prototype));
  ASSERT_EQ(engines.size(), 4);
  EXPECT_EQ(engines.front().results()[0], original_buffer);

  for (size_t i = 0; i < engines.size(); ++i) {
    auto shifted = centered;
    shifted.O[0] = static_cast<double>(i);
    engines[i].compute1(centered, shifted);
  }
  for (size_t i = 0; i < engines.size(); ++i) {
    EXPECT_NEAR(engines[i].results()[0][0],
                std::exp(-0.5 * static_cast<double>(i * i)), 1e-14);
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(engines[i].results()[0], engines[j].results()[0]);
    }
  }
}

TEST(Libint2EngineTest, EmptyPoolDoesNotConsumePrototype) {
  const libint2::Shell shell({1.0}, {{0, true, {1.0}}}, {{0.0, 0.0, 0.0}});
  integrals::Engine prototype(integrals::Operator::overlap, 1, 0);
  EXPECT_TRUE(integrals::Engine::make_pool(0, std::move(prototype)).empty());
  prototype.compute1(shell, shell);
  EXPECT_NEAR(prototype.results()[0][0], 1.0, 1e-14);

  integrals::Engine uninitialized;
  EXPECT_THROW(integrals::Engine::make_pool(1, std::move(uninitialized)),
               std::logic_error);
}

TEST(Libint2EngineTest, ShellConversionPreservesUnnormalizedCoefficients) {
  qdk::chemistry::scf::Shell shell{};
  shell.O = {0.0, 1.0, 2.0};
  shell.angular_momentum = 2;
  shell.contraction = 2;
  shell.exponents[0] = 0.7;
  shell.exponents[1] = 0.2;
  shell.coefficients[0] = 0.3;
  shell.coefficients[1] = 0.5;
  const auto converted = integrals::convert_to_libint_shell(shell, true);
  ASSERT_EQ(converted.alpha.size(), 2);
  ASSERT_EQ(converted.contr.size(), 1);
  ASSERT_EQ(converted.contr[0].coeff.size(), 2);
  EXPECT_EQ(converted.O, shell.O);
  EXPECT_TRUE(converted.contr[0].pure);
  EXPECT_EQ(converted.contr[0].l, 2);
  EXPECT_DOUBLE_EQ(converted.alpha[0], 0.7);
  EXPECT_DOUBLE_EQ(converted.alpha[1], 0.2);
  EXPECT_DOUBLE_EQ(converted.contr[0].coeff[0], 0.3);
  EXPECT_DOUBLE_EQ(converted.contr[0].coeff[1], 0.5);
}

}  // namespace
