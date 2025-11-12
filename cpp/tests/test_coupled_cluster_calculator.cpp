// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <qdk/chemistry/algorithms/coupled_cluster.hpp>
#include <qdk/chemistry/data/coupled_cluster.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Mock implementation of the CoupledClusterCalculator
class MockCoupledClusterCalculator : public CoupledClusterCalculator {
 public:
  MockCoupledClusterCalculator() {}
  ~MockCoupledClusterCalculator() noexcept override = default;
  std::string name() const override { return "mock_cc"; }

 protected:
  std::pair<double, std::shared_ptr<CoupledClusterAmplitudes>> _run_impl(
      std::shared_ptr<Ansatz> /*ansatz*/) const override {
    // Create a minimal valid CoupledClusterAmplitudes object
    double total_energy = -10.0;  // Dummy value

    // Populate with canonical data (occupations must be 0, 1, or 2 for
    // restricted)
    Eigen::MatrixXd coeffs(2, 2);
    coeffs << 1.0, 0.0, 0.0, 1.0;
    Eigen::VectorXd eps(2);
    eps << -1.0, -0.5;

    auto basis = testing::create_random_basis_set(2);
    auto orbs = std::make_shared<Orbitals>(coeffs, eps, std::nullopt, basis,
                                           std::nullopt);

    // Create dummy amplitudes
    CoupledClusterAmplitudes::amplitude_type t1(1);
    t1(0) = 0.01;

    CoupledClusterAmplitudes::amplitude_type t2(1);
    t2(0) = 0.005;

    CoupledClusterAmplitudes cc(orbs, t1, t2, 1, 1);
    return {total_energy,
            std::make_shared<CoupledClusterAmplitudes>(std::move(cc))};
  }
};

TEST(CoupledClusterCalculatorTest, Factory) {
  // Register a mock implementation
  const std::string key = "mock_cc";

  CoupledClusterCalculatorFactory::register_instance(
      []() { return std::make_unique<MockCoupledClusterCalculator>(); });

  // Check if the mock implementation is available
  auto available = CoupledClusterCalculatorFactory::available();
  ASSERT_TRUE(std::find(available.begin(), available.end(), key) !=
              available.end());

  // Create a calculator using the factory
  auto calculator = CoupledClusterCalculatorFactory::create(key);
  ASSERT_NE(calculator, nullptr);

  // Unregister the implementation
  EXPECT_TRUE(CoupledClusterCalculatorFactory::unregister_instance(key));

  // Verify it was removed
  available = CoupledClusterCalculatorFactory::available();
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
  Hamiltonian hamiltonian(empty_one_body, empty_two_body, dummy_orbitals, 0.0,
                          empty_fock);

  // Perform calculation with electron counts
  Wavefunction wfn(std::make_unique<SlaterDeterminantContainer>(
      Configuration("20"), dummy_orbitals));
  auto ansatz_ptr =
      std::make_shared<Ansatz>(std::move(hamiltonian), std::move(wfn));
  auto [energy, cc_result] = calculator.run(ansatz_ptr);

  // Verify the results
  EXPECT_DOUBLE_EQ(energy, -10.0);
  EXPECT_TRUE(cc_result->has_t1_amplitudes());
  EXPECT_TRUE(cc_result->has_t2_amplitudes());
}
