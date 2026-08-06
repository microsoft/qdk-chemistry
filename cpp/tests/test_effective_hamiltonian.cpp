// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <string>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

class TestEffectiveHamiltonianConstructor
    : public EffectiveHamiltonianConstructor {
 public:
  std::string name() const override {
    return "_test_effective_hamiltonian_constructor";
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Wavefunction>, std::shared_ptr<Hamiltonian> hamiltonian,
      std::shared_ptr<const SymmetryBlockedIndexSet>) const override {
    return hamiltonian;
  }
};

TEST(EffectiveHamiltonianConstructorTest, MetaData) {
  TestEffectiveHamiltonianConstructor constructor;
  EXPECT_NO_THROW({ auto settings = constructor.settings(); });
  EXPECT_EQ(constructor.type_name(), "effective_hamiltonian_constructor");
}

TEST(EffectiveHamiltonianConstructorTest, Factory) {
  EXPECT_TRUE(EffectiveHamiltonianConstructorFactory::has("ducc"));
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
