// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/algorithms/population_analysis.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <vector>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

namespace {

std::shared_ptr<Structure> create_h2_structure() {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  return std::make_shared<Structure>(coords, symbols);
}

std::shared_ptr<Wavefunction> create_model_wavefunction() {
  auto orbitals = std::make_shared<ModelOrbitals>(3);
  auto determinant = Configuration::from_bitstring("110");
  auto container =
      std::make_unique<StateVectorContainer>(determinant, orbitals);
  return std::make_shared<Wavefunction>(std::move(container));
}

std::shared_ptr<Wavefunction> create_molecular_wavefunction() {
  std::vector<Shell> shells = {
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}),
      Shell(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0})};
  auto basis =
      std::make_shared<BasisSet>("minimal", shells, create_h2_structure());
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(2, 2);
  auto orbitals = std::make_shared<Orbitals>(identity, std::nullopt, identity,
                                             std::move(basis));
  auto determinant = Configuration::from_spin_half_string("20");
  auto container =
      std::make_unique<StateVectorContainer>(determinant, orbitals);
  return std::make_shared<Wavefunction>(std::move(container));
}

}  // namespace

TEST(PopulationAnalyzerTest, FactoryRegistersQdkAnalyzer) {
  auto analyzer = PopulationAnalyzerFactory::create();

  ASSERT_NE(analyzer, nullptr);
  EXPECT_EQ(analyzer->name(), "qdk");
  EXPECT_EQ(analyzer->type_name(), "population_analyzer");
}

TEST(PopulationAnalyzerTest, QdkAnalyzerRequiresWavefunctionInput) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  EXPECT_THROW(analyzer->run(create_h2_structure(), 1, 1, 0),
               std::invalid_argument);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerReturnsModelSitePopulations) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations = analyzer->run(create_model_wavefunction(), 0, 1, 0);

  ASSERT_EQ(populations.size(), 3);
  EXPECT_DOUBLE_EQ(populations[0], 1.0);
  EXPECT_DOUBLE_EQ(populations[1], 1.0);
  EXPECT_DOUBLE_EQ(populations[2], 0.0);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerReturnsMolecularPopulations) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations = analyzer->run(create_molecular_wavefunction(), 0, 1, 0);

  ASSERT_EQ(populations.size(), 2);
  EXPECT_DOUBLE_EQ(populations[0], 2.0);
  EXPECT_DOUBLE_EQ(populations[1], 0.0);
}
