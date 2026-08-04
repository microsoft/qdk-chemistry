// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <qdk/chemistry/algorithms/population_analysis.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <vector>

#include "ut_common.hpp"

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

std::shared_ptr<Wavefunction> create_correlated_model_wavefunction() {
  auto orbitals = std::make_shared<ModelOrbitals>(2);
  std::vector<Configuration> determinants = {
      Configuration::from_bitstring("10"), Configuration::from_bitstring("01")};
  Eigen::VectorXd coefficients(2);
  coefficients << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
  Eigen::MatrixXd one_rdm(2, 2);
  one_rdm << 0.5, 0.5, 0.5, 0.5;
  auto container = std::make_unique<StateVectorContainer>(
      coefficients, determinants, orbitals, one_rdm, std::nullopt);
  return std::make_shared<Wavefunction>(std::move(container));
}

std::shared_ptr<Wavefunction> create_molecular_wavefunction() {
  std::vector<Shell> shells = {
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}),
      Shell(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0})};
  auto basis =
      std::make_shared<BasisSet>("minimal", shells, create_h2_structure());
  Eigen::MatrixXd overlap(2, 2);
  overlap << 1.0, 0.5, 0.5, 1.0;
  const double bonding_norm = 1.0 / std::sqrt(3.0);
  Eigen::MatrixXd coefficients(2, 2);
  coefficients << bonding_norm, 1.0, bonding_norm, -1.0;
  auto orbitals = std::make_shared<Orbitals>(coefficients, std::nullopt,
                                             overlap, std::move(basis));
  auto determinant = Configuration::from_spin_half_string("20");
  auto container =
      std::make_unique<StateVectorContainer>(determinant, orbitals);
  return std::make_shared<Wavefunction>(std::move(container));
}

std::shared_ptr<Wavefunction>
create_correlated_molecular_wavefunction_with_inactive_core() {
  std::vector<Eigen::Vector3d> coordinates = {
      {0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}, {0.0, 0.0, 2.8}};
  std::vector<std::string> symbols = {"H", "H", "H"};
  auto structure = std::make_shared<Structure>(coordinates, symbols);
  std::vector<Shell> shells = {
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}),
      Shell(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0}),
      Shell(2, OrbitalType::S, std::vector{1.0}, std::vector{1.0})};
  auto basis = std::make_shared<BasisSet>("minimal", shells, structure);
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(3, 3);
  auto orbitals =
      std::make_shared<Orbitals>(identity, std::nullopt, identity, basis,
                                 testing::restricted_index_set(3, {1, 2}),
                                 testing::restricted_index_set(3, {0}));

  std::vector<Configuration> determinants = {
      Configuration::from_spin_half_string("u0"),
      Configuration::from_spin_half_string("0u")};
  Eigen::VectorXd coefficients(2);
  coefficients << 1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0);
  Eigen::MatrixXd one_rdm(2, 2);
  one_rdm << 0.5, 0.5, 0.5, 0.5;
  auto container = std::make_unique<StateVectorContainer>(
      coefficients, determinants, orbitals, one_rdm, std::nullopt);
  return std::make_shared<Wavefunction>(std::move(container));
}

}  // namespace

TEST(PopulationAnalyzerTest, FactoryRegistersQdkAnalyzer) {
  auto analyzer = PopulationAnalyzerFactory::create();

  ASSERT_NE(analyzer, nullptr);
  EXPECT_EQ(analyzer->name(), "qdk");
  EXPECT_EQ(analyzer->type_name(), "population_analyzer");
}

TEST(PopulationAnalyzerTest, QdkAnalyzerDefaultsToMullikenMethod) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  ASSERT_TRUE(analyzer->settings().has("method"));
  EXPECT_EQ(analyzer->settings().get<std::string>("method"), "mulliken");
  EXPECT_THROW(analyzer->settings().set("method", "unsupported"),
               std::invalid_argument);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerDoesNotUseMethodAliases) {
  EXPECT_THROW(PopulationAnalyzerFactory::create("internal"),
               std::runtime_error);
  EXPECT_THROW(PopulationAnalyzerFactory::create("mulliken"),
               std::runtime_error);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerReturnsModelSitePopulations) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations = analyzer->run(create_model_wavefunction(), 0, 1, 0);

  ASSERT_EQ(populations.size(), 3);
  EXPECT_DOUBLE_EQ(populations[0], 1.0);
  EXPECT_DOUBLE_EQ(populations[1], 1.0);
  EXPECT_DOUBLE_EQ(populations[2], 0.0);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerUsesModelOneRdmInSiteBasis) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations =
      analyzer->run(create_correlated_model_wavefunction(), 0, 2, 0);

  ASSERT_EQ(populations.size(), 2);
  EXPECT_DOUBLE_EQ(populations[0], 0.5);
  EXPECT_DOUBLE_EQ(populations[1], 0.5);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerReturnsMolecularPopulations) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations = analyzer->run(create_molecular_wavefunction(), 0, 1, 0);

  ASSERT_EQ(populations.size(), 2);
  EXPECT_NEAR(populations[0], 1.0, 1e-12);
  EXPECT_NEAR(populations[1], 1.0, 1e-12);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerUsesMolecularOneRdmAndInactiveCore) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations = analyzer->run(
      create_correlated_molecular_wavefunction_with_inactive_core(), 0, 2, 1);

  ASSERT_EQ(populations.size(), 3);
  EXPECT_DOUBLE_EQ(populations[0], 2.0);
  EXPECT_DOUBLE_EQ(populations[1], 0.5);
  EXPECT_DOUBLE_EQ(populations[2], 0.5);
}
