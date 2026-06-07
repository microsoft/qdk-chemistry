// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "qdk/chemistry/algorithms/algorithm.hpp"
#include "qdk/chemistry/data/settings.hpp"

namespace {

class AlgorithmHashSettings : public qdk::chemistry::data::Settings {
 public:
  AlgorithmHashSettings() {
    set_default("threshold", 0.0);
    set_default("nested",
                qdk::chemistry::data::AlgorithmRef(
                    "child_type", "child_a",
                    std::make_shared<qdk::chemistry::data::Settings>()));
  }
};

class TestAlgorithm
    : public qdk::chemistry::algorithms::Algorithm<
          TestAlgorithm, int, int, std::string, const std::vector<size_t>&> {
 public:
  explicit TestAlgorithm(std::string algorithm_name = "test_algorithm")
      : algorithm_name_(std::move(algorithm_name)) {
    _settings = std::make_unique<AlgorithmHashSettings>();
  }

  std::string name() const override { return algorithm_name_; }

  std::string type_name() const override { return "test_algorithm_type"; }

 protected:
  int _run_impl(int value, std::string label,
                const std::vector<size_t>& indices) const override {
    return value + static_cast<int>(label.size() + indices.size());
  }

 private:
  std::string algorithm_name_;
};

}  // namespace

TEST(AlgorithmHashTest, SameRunInputsProduceSameHash) {
  TestAlgorithm first;
  TestAlgorithm second;
  const std::vector<size_t> indices{1, 2, 3};

  EXPECT_EQ(first.hash(7, "basis", indices), second.hash(7, "basis", indices));
}

TEST(AlgorithmHashTest, RunArgumentsParticipateInHash) {
  TestAlgorithm algorithm;
  const std::vector<size_t> indices{1, 2, 3};
  const std::vector<size_t> different_indices{1, 2, 4};

  EXPECT_NE(algorithm.hash(7, "basis", indices),
            algorithm.hash(8, "basis", indices));
  EXPECT_NE(algorithm.hash(7, "basis", indices),
            algorithm.hash(7, "other", indices));
  EXPECT_NE(algorithm.hash(7, "basis", indices),
            algorithm.hash(7, "basis", different_indices));
}

TEST(AlgorithmHashTest, AlgorithmIdentityAndSettingsParticipateInHash) {
  TestAlgorithm first("first");
  TestAlgorithm second("second");
  TestAlgorithm with_settings_change("first");
  const std::vector<size_t> indices{1, 2, 3};

  with_settings_change.settings().set("threshold", 1.0);

  EXPECT_NE(first.hash(7, "basis", indices), second.hash(7, "basis", indices));
  EXPECT_NE(first.hash(7, "basis", indices),
            with_settings_change.hash(7, "basis", indices));
}

TEST(AlgorithmHashTest, NestedAlgorithmRefSettingsParticipateInHash) {
  TestAlgorithm first;
  TestAlgorithm second;
  const std::vector<size_t> indices{1, 2, 3};

  second.settings().set(
      "nested", qdk::chemistry::data::AlgorithmRef(
                    "child_type", "child_b",
                    std::make_shared<qdk::chemistry::data::Settings>()));

  EXPECT_NE(first.hash(7, "basis", indices), second.hash(7, "basis", indices));
}
