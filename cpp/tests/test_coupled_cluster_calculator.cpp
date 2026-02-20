// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cc.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Mock implementation of a DynamicalCorrelationCalculator that produces CC
// results
class MockCoupledClusterCalculator : public DynamicalCorrelationCalculator {
 public:
  MockCoupledClusterCalculator() {}
  ~MockCoupledClusterCalculator() override = default;
  std::string name() const override { return "mock_cc"; }
  std::string type_name() const override {
    return "dynamical_correlation_calculator";
  }

 protected:
  DynamicalCorrelationResult _run_impl(
      std::shared_ptr<Ansatz> ansatz) const override {
    // Create a wavefunction with CC container
    auto original_wfn = ansatz->get_wavefunction();
    auto orbs = original_wfn->get_orbitals();

    // Create dummy T1 and T2 amplitudes
    Eigen::VectorXd t1(1);
    t1(0) = 0.01;

    Eigen::VectorXd t2(1);
    t2(0) = 0.005;

    std::optional<CoupledClusterContainer::VectorVariant> t1_opt = t1;
    std::optional<CoupledClusterContainer::VectorVariant> t2_opt = t2;

    auto cc_container = std::make_unique<CoupledClusterContainer>(
        orbs, original_wfn, t1_opt, t2_opt);

    // Create wavefunction with CC container
    auto result_wfn = std::make_shared<Wavefunction>(std::move(cc_container));

    double total_energy = -10.0;  // Dummy value
    return {total_energy, result_wfn, std::nullopt};
  }
};

TEST(CoupledClusterCalculatorTest, Factory) {
  // Register a mock implementation with DynamicalCorrelationCalculatorFactory
  const std::string key = "mock_cc";

  DynamicalCorrelationCalculatorFactory::register_instance(
      []() { return std::make_unique<MockCoupledClusterCalculator>(); });

  // Check if the mock implementation is available
  auto available = DynamicalCorrelationCalculatorFactory::available();
  ASSERT_TRUE(std::find(available.begin(), available.end(), key) !=
              available.end());

  // Create a calculator using the factory
  auto calculator = DynamicalCorrelationCalculatorFactory::create(key);
  ASSERT_NE(calculator, nullptr);

  // Unregister the implementation
  EXPECT_TRUE(DynamicalCorrelationCalculatorFactory::unregister_instance(key));

  // Verify it was removed
  available = DynamicalCorrelationCalculatorFactory::available();
  EXPECT_TRUE(std::find(available.begin(), available.end(), key) ==
              available.end());
}

TEST(CoupledClusterCalculatorTest, Calculate) {
  // Create a mock calculator
  MockCoupledClusterCalculator calculator;

  // Create a dummy Orbitals object for testing
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(2, 2);
  auto basis = testing::create_random_basis_set(2);
  auto dummy_orbitals = std::make_shared<Orbitals>(
      coeffs, std::nullopt, std::nullopt, basis, std::nullopt);

  // Create a dummy Hamiltonian for testing
  Eigen::MatrixXd empty_one_body = Eigen::MatrixXd::Zero(2, 2);
  Eigen::VectorXd empty_two_body = Eigen::VectorXd::Zero(16);
  Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);
  Hamiltonian hamiltonian(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          empty_one_body, empty_two_body, dummy_orbitals, 0.0, empty_fock));

  // Perform calculation with electron counts
  Wavefunction wfn(std::make_unique<SlaterDeterminantContainer>(
      Configuration("20"), dummy_orbitals));
  auto ansatz_ptr =
      std::make_shared<Ansatz>(std::move(hamiltonian), std::move(wfn));
  auto [energy, result_wavefunction, bra_wavefunction] =
      calculator.run(ansatz_ptr);

  // Verify the results
  EXPECT_DOUBLE_EQ(energy, -10.0);
  EXPECT_NE(result_wavefunction, nullptr);
  EXPECT_FALSE(bra_wavefunction.has_value());

  // Test that we can get the amplitudes from the wavefunction container
  auto& cc_container =
      result_wavefunction->get_container<CoupledClusterContainer>();
  EXPECT_TRUE(cc_container.has_t1_amplitudes());
  EXPECT_TRUE(cc_container.has_t2_amplitudes());
}
