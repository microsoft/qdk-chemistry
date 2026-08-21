// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <qdk/chemistry/algorithms/active_space_optimization.hpp>
#include <qdk/chemistry/algorithms/orbital_optimization.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/orbital_optimization.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

std::shared_ptr<Orbitals> make_orbitals() {
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(1, 1);
  Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(1, 1);
  auto basis_set = testing::create_random_basis_set(1, "test");
  return std::make_shared<Orbitals>(coeffs, std::nullopt,
                                    std::make_optional(overlap), basis_set);
}

std::shared_ptr<Wavefunction> make_wavefunction(
    const std::shared_ptr<Orbitals>& orbitals) {
  Eigen::VectorXcd coefficients(1);
  coefficients(0) = 1.0;
  Wavefunction::DeterminantVector determinants{
      Configuration::from_spin_half_string("2")};
  return std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
      coefficients, determinants, orbitals));
}

std::shared_ptr<Wavefunction> make_qio_wavefunction() {
  Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(2, 2);
  auto orbitals = std::make_shared<Orbitals>(
      coefficients, std::nullopt, std::make_optional(overlap),
      testing::create_random_basis_set(2, "test"));
  orbitals = testing::with_active_space(orbitals, {0, 1}, {});

  Eigen::MatrixXd rdm_alpha(2, 2);
  rdm_alpha << 0.5, 0.5, 0.5, 0.5;
  const Eigen::MatrixXd rdm_beta = Eigen::MatrixXd::Zero(2, 2);
  const Eigen::VectorXd zero_two_rdm = Eigen::VectorXd::Zero(16);
  Eigen::VectorXd ci_coefficients(1);
  ci_coefficients << 1.0;
  const Wavefunction::DeterminantVector determinants{
      Configuration::from_spin_half_string("20")};
  return std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
      ContainerTypes::VectorVariant(ci_coefficients), determinants, orbitals,
      std::nullopt,
      std::make_optional<ContainerTypes::MatrixVariant>(rdm_alpha),
      std::make_optional<ContainerTypes::MatrixVariant>(rdm_beta), std::nullopt,
      std::make_optional<ContainerTypes::VectorVariant>(zero_two_rdm),
      std::make_optional<ContainerTypes::VectorVariant>(zero_two_rdm),
      std::make_optional<ContainerTypes::VectorVariant>(zero_two_rdm)));
}

class TestOrbitalOptimizer : public OrbitalOptimizer {
 public:
  std::string name() const override { return "test_orbital_optimizer"; }

 protected:
  std::shared_ptr<OrbitalOptimizationResult> _run_impl(
      std::shared_ptr<Wavefunction> wavefunction) const override {
    return std::make_shared<OrbitalOptimizationResult>(
        wavefunction->get_orbitals(), 2.0, 1.0, 3, true);
  }
};

class TestActiveSpaceOptimizer : public ActiveSpaceOptimizer {
 public:
  std::string name() const override { return "test_active_space_optimizer"; }

 protected:
  std::shared_ptr<ActiveSpaceOptimizationResult> _run_impl(
      std::shared_ptr<Orbitals> orbitals, unsigned int,
      unsigned int) const override {
    return std::make_shared<ActiveSpaceOptimizationResult>(
        -1.0, make_wavefunction(orbitals), true, 2,
        std::vector<double>{-0.9, -1.0}, std::vector<double>{2.0, 1.0});
  }
};

}  // namespace

TEST(OrbitalOptimizationTest, EmptyFactoriesSupportRegistration) {
  EXPECT_EQ(OrbitalOptimizerFactory::available(),
            std::vector<std::string>{"qdk_qio"});
  EXPECT_TRUE(ActiveSpaceOptimizerFactory::available().empty());

  OrbitalOptimizerFactory::register_instance(
      [] { return std::make_unique<TestOrbitalOptimizer>(); });
  ActiveSpaceOptimizerFactory::register_instance(
      [] { return std::make_unique<TestActiveSpaceOptimizer>(); });

  EXPECT_NE(OrbitalOptimizerFactory::create("test_orbital_optimizer"), nullptr);
  EXPECT_NE(ActiveSpaceOptimizerFactory::create("test_active_space_optimizer"),
            nullptr);

  EXPECT_TRUE(
      OrbitalOptimizerFactory::unregister_instance("test_orbital_optimizer"));
  EXPECT_TRUE(ActiveSpaceOptimizerFactory::unregister_instance(
      "test_active_space_optimizer"));
}

TEST(OrbitalOptimizationTest, QIOOptimizerReturnsOrbitalProposal) {
  auto optimizer = OrbitalOptimizerFactory::create();
  EXPECT_EQ(optimizer->name(), "qdk_qio");

  auto result = optimizer->run(make_qio_wavefunction());

  ASSERT_NE(result, nullptr);
  ASSERT_NE(result->orbitals(), nullptr);
  EXPECT_LT(result->final_objective(), result->initial_objective());
  EXPECT_NEAR(result->final_objective(), 0.0, 1e-8);
  EXPECT_GT(result->iterations(), 0);
  EXPECT_TRUE(result->converged());
  EXPECT_FALSE(result->orbitals()->has_energies());
  EXPECT_TRUE(result->orbitals()->has_active_space());
}

TEST(OrbitalOptimizationTest, OrbitalOptimizerRunReturnsResult) {
  TestOrbitalOptimizer optimizer;
  auto orbitals = make_orbitals();
  auto result = optimizer.run(make_wavefunction(orbitals));

  ASSERT_NE(result, nullptr);
  EXPECT_NE(result->orbitals(), nullptr);
  EXPECT_DOUBLE_EQ(result->initial_objective(), 2.0);
  EXPECT_DOUBLE_EQ(result->final_objective(), 1.0);
  EXPECT_EQ(result->iterations(), 3u);
  EXPECT_TRUE(result->converged());
}

TEST(OrbitalOptimizationTest, ActiveSpaceOptimizerRunReturnsResult) {
  TestActiveSpaceOptimizer optimizer;
  EXPECT_EQ(optimizer.type_name(), "active_space_optimizer");

  auto orbitals = make_orbitals();
  auto result = optimizer.run(orbitals, 1u, 1u);

  ASSERT_NE(result, nullptr);
  EXPECT_NE(result->wavefunction(), nullptr);
  EXPECT_DOUBLE_EQ(result->energy(), -1.0);
  EXPECT_TRUE(result->converged());
  EXPECT_EQ(result->macro_iterations(), 2u);
  ASSERT_EQ(result->energy_history().size(), 2u);
  ASSERT_EQ(result->objective_history().size(), 2u);
  EXPECT_DOUBLE_EQ(result->energy_history().back(), -1.0);
  EXPECT_DOUBLE_EQ(result->objective_history().front(), 2.0);
}

TEST(OrbitalOptimizationTest, ActiveSpaceOptimizerDefaultSettings) {
  TestActiveSpaceOptimizer optimizer;
  const auto& settings = optimizer.settings();
  EXPECT_EQ(settings.get<int64_t>("max_macro_iterations"), 20);
  EXPECT_DOUBLE_EQ(settings.get<double>("energy_tolerance"), 1e-8);
  EXPECT_DOUBLE_EQ(settings.get<double>("objective_tolerance"), 1e-8);

  const auto ref = settings.get<AlgorithmRef>("orbital_optimizer");
  EXPECT_EQ(ref.get_algorithm_type(), "orbital_optimizer");
  EXPECT_EQ(ref.get_algorithm_name(), "qdk_qio");
}

TEST(OrbitalOptimizationTest, ResultValidation) {
  EXPECT_THROW(OrbitalOptimizationResult(nullptr, 1.0, 0.5, 1, true),
               std::invalid_argument);
  EXPECT_THROW(
      ActiveSpaceOptimizationResult(0.0, nullptr, false, 1, {0.0}, {1.0}),
      std::invalid_argument);
}

TEST(OrbitalOptimizationTest, ActiveSpaceHistoryInvariants) {
  auto orbitals = make_orbitals();
  auto wavefunction = make_wavefunction(orbitals);

  // Energy and objective histories must have the same length.
  EXPECT_THROW(ActiveSpaceOptimizationResult(-1.0, wavefunction, true, 2,
                                             std::vector<double>{-0.9, -1.0},
                                             std::vector<double>{1.0}),
               std::invalid_argument);

  // The macro-iteration count must equal the history length.
  EXPECT_THROW(ActiveSpaceOptimizationResult(-1.0, wavefunction, true, 3,
                                             std::vector<double>{-0.9, -1.0},
                                             std::vector<double>{2.0, 1.0}),
               std::invalid_argument);

  // A consistent zero-iteration result (empty histories) is valid.
  EXPECT_NO_THROW(ActiveSpaceOptimizationResult(0.0, wavefunction, false, 0,
                                                std::vector<double>{},
                                                std::vector<double>{}));
}
