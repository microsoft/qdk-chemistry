// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <string>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class TestEffectiveHamiltonianConstructor
    : public EffectiveHamiltonianConstructor {
 public:
  std::string name() const override {
    return "_test_effective_hamiltonian_constructor";
  }

  using EffectiveHamiltonianConstructor::_validate_inputs;

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Wavefunction>, std::shared_ptr<Hamiltonian> hamiltonian,
      std::shared_ptr<const SymmetryBlockedIndexSet>) const override {
    return hamiltonian;
  }
};

namespace {

std::shared_ptr<Hamiltonian> make_hamiltonian(
    std::shared_ptr<Orbitals> orbitals) {
  const auto size =
      spin_channel_indices(orbitals->active_indices(), axes::alpha()).size();
  return std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          Eigen::MatrixXd::Identity(size, size),
          Eigen::VectorXd::Zero(size * size * size * size), std::move(orbitals),
          0.0, Eigen::MatrixXd::Zero(0, 0)));
}

std::shared_ptr<Wavefunction> make_wavefunction(
    std::shared_ptr<Orbitals> orbitals) {
  const auto size =
      spin_channel_indices(orbitals->active_indices(), axes::alpha()).size();
  std::string configuration(size, '0');
  if (!configuration.empty()) configuration[0] = '2';
  return std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string(configuration),
      std::move(orbitals)));
}

}  // namespace

TEST(EffectiveHamiltonianConstructorTest, MetaData) {
  TestEffectiveHamiltonianConstructor constructor;
  EXPECT_NO_THROW({ auto settings = constructor.settings(); });
  EXPECT_EQ(constructor.type_name(), "effective_hamiltonian_constructor");
}

TEST(EffectiveHamiltonianConstructorTest, Factory) {
  EXPECT_TRUE(EffectiveHamiltonianConstructorFactory::available().empty());
  EXPECT_THROW(
      EffectiveHamiltonianConstructorFactory::create("nonexistent_constructor"),
      std::runtime_error);
  EXPECT_NO_THROW(EffectiveHamiltonianConstructorFactory::register_instance(
      []() -> EffectiveHamiltonianConstructorFactory::return_type {
        return std::make_unique<TestEffectiveHamiltonianConstructor>();
      }));
  EXPECT_THROW(
      EffectiveHamiltonianConstructorFactory::register_instance(
          []() -> EffectiveHamiltonianConstructorFactory::return_type {
            return std::make_unique<TestEffectiveHamiltonianConstructor>();
          }),
      std::runtime_error);

  auto constructor = EffectiveHamiltonianConstructorFactory::create(
      "_test_effective_hamiltonian_constructor");
  EXPECT_NE(constructor, nullptr);

  EXPECT_FALSE(EffectiveHamiltonianConstructorFactory::unregister_instance(
      "nonexistent_constructor"));
  EXPECT_TRUE(EffectiveHamiltonianConstructorFactory::unregister_instance(
      "_test_effective_hamiltonian_constructor"));
  EXPECT_FALSE(EffectiveHamiltonianConstructorFactory::unregister_instance(
      "_test_effective_hamiltonian_constructor"));
}

TEST(EffectiveHamiltonianConstructorTest, AcceptsNestedOrbitalSpaces) {
  auto base_orbitals = testing::create_test_orbitals(4, 4);
  auto hamiltonian_orbitals =
      testing::with_active_space(base_orbitals, {0, 1, 2, 3}, {});
  auto wavefunction_orbitals =
      testing::with_active_space(base_orbitals, {1, 2, 3}, {0});
  auto hamiltonian = make_hamiltonian(hamiltonian_orbitals);
  auto reference = make_wavefunction(wavefunction_orbitals);
  auto p_indices = testing::restricted_index_set(4, {1, 3});

  TestEffectiveHamiltonianConstructor constructor;
  EXPECT_NO_THROW(
      constructor._validate_inputs(reference, hamiltonian, p_indices));
}

TEST(EffectiveHamiltonianConstructorTest, RejectsIncompatibleOrbitalBases) {
  auto hamiltonian_orbitals = testing::create_test_orbitals(4, 4);
  auto hamiltonian = make_hamiltonian(hamiltonian_orbitals);
  TestEffectiveHamiltonianConstructor constructor;

  auto different_size = testing::create_test_orbitals(4, 3);
  EXPECT_THROW(constructor._validate_inputs(
                   make_wavefunction(different_size), hamiltonian,
                   testing::restricted_index_set(3, {0})),
               std::invalid_argument);

  const auto& coefficients = hamiltonian_orbitals->coefficients()->block(
      {axes::alpha(), axes::alpha()});
  auto unrestricted = std::make_shared<Orbitals>(
      coefficients, coefficients, std::nullopt, std::nullopt, std::nullopt,
      hamiltonian_orbitals->get_basis_set(),
      testing::unrestricted_index_set(4, {0, 1, 2, 3}, {0, 1, 2, 3}));
  EXPECT_THROW(constructor._validate_inputs(
                   make_wavefunction(unrestricted), hamiltonian,
                   testing::unrestricted_index_set(4, {0}, {0})),
               std::invalid_argument);
}

TEST(EffectiveHamiltonianConstructorTest, RejectsNonNestedOrbitalSpaces) {
  auto base_orbitals = testing::create_test_orbitals(4, 4);
  auto hamiltonian_orbitals =
      testing::with_active_space(base_orbitals, {0, 1, 2}, {3});
  auto wavefunction_orbitals =
      testing::with_active_space(base_orbitals, {1, 2, 3}, {0});
  auto hamiltonian = make_hamiltonian(hamiltonian_orbitals);
  auto reference = make_wavefunction(wavefunction_orbitals);
  TestEffectiveHamiltonianConstructor constructor;

  EXPECT_THROW(
      constructor._validate_inputs(reference, hamiltonian,
                                   testing::restricted_index_set(4, {1, 2})),
      std::invalid_argument);

  auto valid_wavefunction_orbitals =
      testing::with_active_space(base_orbitals, {1, 2}, {0});
  auto valid_reference = make_wavefunction(valid_wavefunction_orbitals);
  EXPECT_THROW(
      constructor._validate_inputs(valid_reference, hamiltonian,
                                   testing::restricted_index_set(4, {0})),
      std::invalid_argument);
  EXPECT_THROW(
      constructor._validate_inputs(valid_reference, hamiltonian,
                                   testing::restricted_index_set(5, {1})),
      std::invalid_argument);
}

TEST(EffectiveHamiltonianConstructorTest, RejectsNullInputs) {
  auto orbitals = testing::create_test_orbitals(2, 2);
  auto hamiltonian = make_hamiltonian(orbitals);
  auto reference = make_wavefunction(orbitals);
  auto p_indices = testing::restricted_index_set(2, {0});
  TestEffectiveHamiltonianConstructor constructor;

  EXPECT_THROW(constructor._validate_inputs(nullptr, hamiltonian, p_indices),
               std::invalid_argument);
  EXPECT_THROW(constructor._validate_inputs(reference, nullptr, p_indices),
               std::invalid_argument);
  EXPECT_THROW(constructor._validate_inputs(reference, hamiltonian, nullptr),
               std::invalid_argument);
}
