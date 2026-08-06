// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <string>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Minimal EffectiveHamiltonianConstructor that records the forwarded reference
// and P-space index set and echoes the input Hamiltonian.
class MockEffectiveHamiltonianConstructor
    : public EffectiveHamiltonianConstructor {
 public:
  std::string name() const override { return "mock_effective_hamiltonian"; }

  mutable std::shared_ptr<Wavefunction> received_reference;
  mutable std::shared_ptr<const SymmetryBlockedIndexSet> received_p_indices;

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Wavefunction> reference,
      std::shared_ptr<Hamiltonian> hamiltonian,
      std::shared_ptr<const SymmetryBlockedIndexSet> p_indices) const override {
    received_reference = reference;
    received_p_indices = p_indices;
    return hamiltonian;
  }
};

// The abstract interface participates in its factory: a derived implementation
// can be registered, created by name, and unregistered.
TEST(EffectiveHamiltonianConstructorTest, Factory) {
  const std::string key = "mock_effective_hamiltonian";

  EffectiveHamiltonianConstructorFactory::register_instance(
      []() { return std::make_unique<MockEffectiveHamiltonianConstructor>(); });

  auto available = EffectiveHamiltonianConstructorFactory::available();
  ASSERT_TRUE(std::find(available.begin(), available.end(), key) !=
              available.end());

  auto constructor = EffectiveHamiltonianConstructorFactory::create(key);
  ASSERT_NE(constructor, nullptr);
  EXPECT_EQ(constructor->name(), key);
  EXPECT_EQ(constructor->type_name(), "effective_hamiltonian_constructor");

  EXPECT_TRUE(EffectiveHamiltonianConstructorFactory::unregister_instance(key));

  available = EffectiveHamiltonianConstructorFactory::available();
  EXPECT_TRUE(std::find(available.begin(), available.end(), key) ==
              available.end());
}

// run() forwards the reference and the (restricted or unrestricted) P-space
// index set unchanged to _run_impl, and returns its result.
TEST(EffectiveHamiltonianConstructorTest, RunForwardsArguments) {
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(2, 2);
  auto basis = testing::create_random_basis_set(2);
  auto orbitals =
      std::make_shared<Orbitals>(coeffs, std::nullopt, std::nullopt, basis);

  auto reference =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          Configuration::from_spin_half_string("20"), orbitals));

  Eigen::MatrixXd one_body = Eigen::MatrixXd::Zero(2, 2);
  Eigen::VectorXd two_body = Eigen::VectorXd::Zero(16);
  Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);
  auto hamiltonian = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body, two_body, orbitals, 0.0, empty_fock));

  MockEffectiveHamiltonianConstructor constructor;

  for (const auto& p_indices :
       {testing::restricted_index_set(4, {1, 2}),
        testing::unrestricted_index_set(4, {0, 2}, {1, 3})}) {
    auto result = constructor.run(reference, hamiltonian, p_indices);
    EXPECT_EQ(result, hamiltonian);
    EXPECT_EQ(constructor.received_reference, reference);
    EXPECT_EQ(constructor.received_p_indices, p_indices);
  }
}
