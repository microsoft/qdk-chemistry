// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/algorithms/population_analysis.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <vector>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

std::shared_ptr<Structure> create_h2_structure() {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};
  return std::make_shared<Structure>(coords, symbols);
}

}  // namespace

TEST(PopulationAnalyzerTest, FactoryRegistersQdkAnalyzer) {
  auto analyzer = PopulationAnalyzerFactory::create();

  ASSERT_NE(analyzer, nullptr);
  EXPECT_EQ(analyzer->name(), "qdk");
  EXPECT_EQ(analyzer->type_name(), "population_analyzer");
}

TEST(PopulationAnalyzerTest, QdkAnalyzerRunsOnStructureInput) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto charges = analyzer->run(create_h2_structure(), 1, 1, 0);

  ASSERT_EQ(charges.size(), 2);
  EXPECT_DOUBLE_EQ(charges[0], 0.5);
  EXPECT_DOUBLE_EQ(charges[1], 0.5);
}
