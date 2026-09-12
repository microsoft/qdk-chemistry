// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <qdk/chemistry/algorithms/hamiltonian_factorization.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <stdexcept>
#include <string>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

class TestHamiltonianFactorizationSettings : public Settings {
 public:
  TestHamiltonianFactorizationSettings() {
    set_default<std::int64_t>("test_setting", 1);
  }
};

class TestHamiltonianFactorization : public HamiltonianFactorization {
 public:
  TestHamiltonianFactorization() {
    _settings = std::make_unique<TestHamiltonianFactorizationSettings>();
  }

  std::string name() const override {
    return "_test_hamiltonian_factorization";
  }

 protected:
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian) const override {
    return hamiltonian;
  }
};

}  // namespace

TEST(HamiltonianFactorizationTest, MetadataAndRunContract) {
  TestHamiltonianFactorization factorization;

  EXPECT_EQ(factorization.type_name(), "hamiltonian_factorization");
  EXPECT_EQ(factorization.name(), "_test_hamiltonian_factorization");

  std::shared_ptr<Hamiltonian> hamiltonian;
  EXPECT_EQ(factorization.run(hamiltonian), hamiltonian);
  EXPECT_THROW(
      factorization.settings().set("test_setting", std::int64_t{2}),
               SettingsAreLocked);
}

TEST(HamiltonianFactorizationTest, Factory) {
  EXPECT_EQ(HamiltonianFactorizationFactory::algorithm_type_name(),
            "hamiltonian_factorization");
  EXPECT_EQ(HamiltonianFactorizationFactory::default_algorithm_name(),
            "double_factorization");

  const auto available = HamiltonianFactorizationFactory::available();
  EXPECT_NE(std::find(available.begin(), available.end(),
                      "double_factorization"),
            available.end());
  EXPECT_EQ(HamiltonianFactorizationFactory::create()->name(),
            "double_factorization");
  EXPECT_THROW(
      HamiltonianFactorizationFactory::create("nonexistent_factorization"),
      std::runtime_error);

  EXPECT_NO_THROW(HamiltonianFactorizationFactory::register_instance(
      []() -> HamiltonianFactorizationFactory::return_type {
        return std::make_unique<TestHamiltonianFactorization>();
      }));
  EXPECT_THROW(
      HamiltonianFactorizationFactory::register_instance(
          []() -> HamiltonianFactorizationFactory::return_type {
            return std::make_unique<TestHamiltonianFactorization>();
          }),
      std::runtime_error);

  auto test_factorization = HamiltonianFactorizationFactory::create(
      "_test_hamiltonian_factorization");
  EXPECT_NE(test_factorization, nullptr);
  EXPECT_EQ(test_factorization->type_name(), "hamiltonian_factorization");

  EXPECT_FALSE(HamiltonianFactorizationFactory::unregister_instance(
      "nonexistent_factorization"));
  EXPECT_TRUE(HamiltonianFactorizationFactory::unregister_instance(
      "_test_hamiltonian_factorization"));
  EXPECT_FALSE(HamiltonianFactorizationFactory::unregister_instance(
      "_test_hamiltonian_factorization"));
}
