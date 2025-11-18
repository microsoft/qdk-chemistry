// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/mcscf.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class MultiConfigurationScfTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class TestMultiConfigurationScfSolver : public MultiConfigurationScf {
  std::string name() const override { return "test_mcscf"; }

 protected:
  std::pair<double, std::shared_ptr<Wavefunction>> _run_impl(
      std::shared_ptr<Orbitals> orbitals,
      std::shared_ptr<HamiltonianConstructor> hamiltonian,
      std::shared_ptr<MultiConfigurationCalculator> /*mc_calculator*/,
      unsigned int /*nalpha*/, unsigned int /*nbeta*/) const override {
    // Dummy implementation for testing
    Eigen::VectorXcd coeffs(1);
    coeffs(0) = std::complex<double>(1.0, 0.0);
    Wavefunction::DeterminantVector dets{Configuration("2")};
    auto container =
        std::make_unique<CasWavefunctionContainer>(coeffs, dets, orbitals);
    Wavefunction wfn(std::move(container));
    return {-100.0, std::make_shared<Wavefunction>(std::move(wfn))};
  }
};

class MockMCCalculator : public MultiConfigurationCalculator {
 public:
  std::string name() const override { return "mc_2"; }

 protected:
  std::pair<double, std::shared_ptr<Wavefunction>> _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian, unsigned int /*nalpha*/,
      unsigned int /*nbeta*/) const override {
    // Dummy implementation for testing - create a simple wavefunction
    Eigen::VectorXcd coeffs(1);
    coeffs(0) = std::complex<double>(1.0, 0.0);
    Wavefunction::DeterminantVector dets{Configuration("2")};
    auto orbitals = hamiltonian->get_orbitals();
    auto container =
        std::make_unique<CasWavefunctionContainer>(coeffs, dets, orbitals);
    auto wfn = std::make_shared<Wavefunction>(std::move(container));
    return {-100.0, wfn};
  }
};

class TestMultiConfigurationScfSolverAlternative
    : public MultiConfigurationScf {
 public:
  std::string name() const override { return "mcscf_2"; }

 protected:
  std::pair<double, std::shared_ptr<Wavefunction>> _run_impl(
      std::shared_ptr<Orbitals> orbitals,
      std::shared_ptr<HamiltonianConstructor> hamiltonian,
      std::shared_ptr<MultiConfigurationCalculator> /*mc_calculator*/,
      unsigned int /*nalpha*/, unsigned int /*nbeta*/) const override {
    // Alternative dummy implementation for testing - create a simple
    // wavefunction
    Eigen::VectorXcd coeffs(1);
    coeffs(0) = std::complex<double>(1.0, 0.0);

    Wavefunction::DeterminantVector dets;
    dets.push_back(Configuration("2"));  // Simple single determinant

    auto hamil = hamiltonian->run(orbitals);
    auto container =
        std::make_unique<CasWavefunctionContainer>(coeffs, dets, orbitals);
    auto wfn = std::make_shared<Wavefunction>(std::move(container));
    return {-200.0, wfn};
  }
};

TEST_F(MultiConfigurationScfTest, FactoryEmptyRegistry) {
  // Test behavior when no implementations are registered
  auto available_solvers = MultiConfigurationScfFactory::available();
  EXPECT_EQ(available_solvers.size(), 0);

  // Should throw when trying to create with empty registry
  EXPECT_THROW(MultiConfigurationScfFactory::create(), std::runtime_error);
  EXPECT_THROW(MultiConfigurationScfFactory::create(""), std::runtime_error);
  EXPECT_THROW(MultiConfigurationScfFactory::create("nonexistent"),
               std::runtime_error);
}

TEST_F(MultiConfigurationScfTest, FactoryRegisterInstance) {
  // Test registering a new instance
  EXPECT_NO_THROW(MultiConfigurationScfFactory::register_instance(
      []() -> MultiConfigurationScfFactory::return_type {
        return std::make_unique<TestMultiConfigurationScfSolver>();
      }));

  // Verify it's available
  auto available_solvers = MultiConfigurationScfFactory::available();
  EXPECT_EQ(available_solvers.size(), 1);
  EXPECT_EQ(available_solvers[0], "test_mcscf");

  // Test duplicate registration should throw
  EXPECT_THROW(MultiConfigurationScfFactory::register_instance(
                   []() -> MultiConfigurationScfFactory::return_type {
                     return std::make_unique<TestMultiConfigurationScfSolver>();
                   }),
               std::runtime_error);
}

TEST_F(MultiConfigurationScfTest, FactoryCreateInstance) {
  // Register a test instance
  auto available = MultiConfigurationScfFactory::available();
  if (std::find(available.begin(), available.end(), "test_mcscf") ==
      available.end()) {
    MultiConfigurationScfFactory::register_instance(
        []() -> MultiConfigurationScfFactory::return_type {
          return std::make_unique<TestMultiConfigurationScfSolver>();
        });
  }

  // Test creating with explicit key
  auto mcscf = MultiConfigurationScfFactory::create("test_mcscf");
  EXPECT_NE(mcscf, nullptr);

  // Test that the created instance works
  EXPECT_NO_THROW({ auto& settings = mcscf->settings(); });
}

TEST_F(MultiConfigurationScfTest, FactoryUnregisterInstance) {
  // Register a test instance
  auto available = MultiConfigurationScfFactory::available();
  if (std::find(available.begin(), available.end(), "test_mcscf") ==
      available.end()) {
    MultiConfigurationScfFactory::register_instance(
        []() -> MultiConfigurationScfFactory::return_type {
          return std::make_unique<TestMultiConfigurationScfSolver>();
        });
  }

  // Verify it's registered
  auto available_before = MultiConfigurationScfFactory::available();
  EXPECT_TRUE(std::find(available_before.begin(), available_before.end(),
                        "test_mcscf") != available_before.end());

  // Test successful unregistration
  EXPECT_TRUE(MultiConfigurationScfFactory::unregister_instance("test_mcscf"));

  // Verify it's no longer available
  auto available_after = MultiConfigurationScfFactory::available();
  EXPECT_TRUE(std::find(available_after.begin(), available_after.end(),
                        "test_mcscf") == available_after.end());

  // Test unregistering nonexistent key
  EXPECT_FALSE(
      MultiConfigurationScfFactory::unregister_instance("nonexistent_key"));
}

TEST_F(MultiConfigurationScfTest, FactoryMultipleInstances) {
  // Register multiple instances
  MultiConfigurationScfFactory::register_instance(
      []() -> MultiConfigurationScfFactory::return_type {
        return std::make_unique<TestMultiConfigurationScfSolver>();
      });

  MultiConfigurationScfFactory::register_instance(
      []() -> MultiConfigurationScfFactory::return_type {
        return std::make_unique<TestMultiConfigurationScfSolverAlternative>();
      });

  // Verify both are available (may have others from previous tests)
  auto available_solvers = MultiConfigurationScfFactory::available();
  EXPECT_GE(available_solvers.size(), 2);
  EXPECT_TRUE(std::find(available_solvers.begin(), available_solvers.end(),
                        "test_mcscf") != available_solvers.end());
  EXPECT_TRUE(std::find(available_solvers.begin(), available_solvers.end(),
                        "mcscf_2") != available_solvers.end());

  // Test creating different instances
  auto mcscf1 = MultiConfigurationScfFactory::create("test_mcscf");
  auto mcscf2 = MultiConfigurationScfFactory::create("mcscf_2");
}

TEST_F(MultiConfigurationScfTest, SolverInterface) {
  // Test the MultiConfigurationScf interface through a concrete implementation
  TestMultiConfigurationScfSolver solver;

  // Create a dummy Hamiltonian for testing using ut_common helpers
  auto test_orbitals = testing::create_test_orbitals(3, 2, true);

  // Create minimal one-body and two-body integrals for 2 orbitals
  Eigen::MatrixXd one_body(2, 2);
  one_body << -1.0, 0.1, 0.1, -0.5;

  // Create minimal two-body integrals for 2 orbitals (16 elements total)
  Eigen::VectorXd two_body(16);
  two_body.setZero();
  two_body(0) = 0.5;  // <00|00> element

  double core_energy = 1.0;
  auto inactive_fock = Eigen::MatrixXd::Zero(0, 0);

  auto hamil_ctor = std::shared_ptr<HamiltonianConstructor>(
      std::move(HamiltonianConstructorFactory::create()));

  auto dummy_ham = std::make_shared<Hamiltonian>(
      one_body, two_body, test_orbitals, core_energy, inactive_fock);

  // Create a mock MC calculator
  auto mc_calculator = std::make_shared<MockMCCalculator>();

  // Test solve method
  auto [energy, wavefunction] =
      solver.run(test_orbitals, hamil_ctor, mc_calculator, 1, 1);
  EXPECT_EQ(energy, -100.0);

  // Test settings access
  EXPECT_NO_THROW({ auto& settings = solver.settings(); });
}

// Clean up registered instances after tests
class MultiConfigurationScfTestCleanup : public ::testing::Test {
 protected:
  void TearDown() override {
    // Clean up any registered test instances
    MultiConfigurationScfFactory::unregister_instance("test_mcscf");
    MultiConfigurationScfFactory::unregister_instance("test_create");
    MultiConfigurationScfFactory::unregister_instance("test_docstring");
    MultiConfigurationScfFactory::unregister_instance("test_unregister");
    MultiConfigurationScfFactory::unregister_instance("mcscf_1");
    MultiConfigurationScfFactory::unregister_instance("mcscf_2");
    MultiConfigurationScfFactory::unregister_instance("first_solver");
    MultiConfigurationScfFactory::unregister_instance("second_solver");
  }
};

TEST_F(MultiConfigurationScfTestCleanup, CleanupAfterTests) {
  // This test ensures cleanup happens
  EXPECT_TRUE(true);
}
