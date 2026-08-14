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

std::shared_ptr<Structure> create_lih_structure() {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 3.0}};
  std::vector<std::string> symbols = {"Li", "H"};
  return std::make_shared<Structure>(coords, symbols);
}

std::shared_ptr<BasisSet> create_lih_basis() {
  std::vector<Shell> shells = {
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}),
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}),
      Shell(0, OrbitalType::P, std::vector{1.0}, std::vector{1.0}),
      Shell(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0})};
  return std::make_shared<BasisSet>("minimal", shells, create_lih_structure());
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
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(6, 6);
  auto orbitals = std::make_shared<Orbitals>(identity, std::nullopt, identity,
                                             create_lih_basis());
  auto determinant = Configuration::from_spin_half_string("00002u");
  auto container =
      std::make_unique<StateVectorContainer>(determinant, orbitals);
  return std::make_shared<Wavefunction>(std::move(container));
}

std::shared_ptr<Wavefunction> create_unrestricted_molecular_wavefunction() {
  Eigen::MatrixXd coefficients_alpha = Eigen::MatrixXd::Identity(6, 6);
  Eigen::MatrixXd coefficients_beta = Eigen::MatrixXd::Identity(6, 6);
  coefficients_beta.col(0).swap(coefficients_beta.col(5));
  Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(6, 6);
  auto orbitals = std::make_shared<Orbitals>(
      coefficients_alpha, coefficients_beta, std::nullopt, std::nullopt,
      overlap, create_lih_basis());
  auto determinant = Configuration::from_spin_half_string("0000u2");
  auto container =
      std::make_unique<StateVectorContainer>(determinant, orbitals);
  return std::make_shared<Wavefunction>(std::move(container));
}

std::shared_ptr<Wavefunction>
create_correlated_molecular_wavefunction_with_ao_overlap() {
  std::vector<Eigen::Vector3d> coordinates = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  auto structure = std::make_shared<Structure>(coordinates, symbols);
  std::vector<Shell> shells = {
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}),
      Shell(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0})};
  auto basis = std::make_shared<BasisSet>("minimal", shells, structure);

  Eigen::MatrixXd overlap(2, 2);
  overlap << 1.0, 0.3, 0.3, 1.0;
  Eigen::MatrixXd coefficients =
      overlap.llt().matrixU().solve(Eigen::MatrixXd::Identity(2, 2));
  auto orbitals =
      std::make_shared<Orbitals>(coefficients, std::nullopt, overlap, basis);

  std::vector<Configuration> determinants = {
      Configuration::from_spin_half_string("20")};
  Eigen::VectorXd state_coefficients = Eigen::VectorXd::Ones(1);
  Eigen::MatrixXd one_rdm(2, 2);
  one_rdm << 1.5, 0.5, 0.5, 0.5;
  auto container = std::make_unique<StateVectorContainer>(
      state_coefficients, determinants, orbitals, one_rdm, std::nullopt);
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

  auto populations = analyzer->run(create_model_wavefunction());

  ASSERT_EQ(populations.size(), 3);
  EXPECT_DOUBLE_EQ(populations[0], 1.0);
  EXPECT_DOUBLE_EQ(populations[1], 1.0);
  EXPECT_DOUBLE_EQ(populations[2], 0.0);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerUsesModelOneRdmInSiteBasis) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations = analyzer->run(create_correlated_model_wavefunction());

  ASSERT_EQ(populations.size(), 2);
  EXPECT_DOUBLE_EQ(populations[0], 0.5);
  EXPECT_DOUBLE_EQ(populations[1], 0.5);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerAssignsHeteronuclearAoBlocksToAtoms) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations = analyzer->run(create_molecular_wavefunction());

  ASSERT_EQ(populations.size(), 2);
  EXPECT_NEAR(populations[0], 2.0, 1e-12);
  EXPECT_NEAR(populations[1], 1.0, 1e-12);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerUsesUnrestrictedMolecularDensity) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations =
      analyzer->run(create_unrestricted_molecular_wavefunction());

  ASSERT_EQ(populations.size(), 2);
  EXPECT_NEAR(populations[0], 2.0, 1e-12);
  EXPECT_NEAR(populations[1], 1.0, 1e-12);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerContractsDensityWithAoOverlap) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations =
      analyzer->run(create_correlated_molecular_wavefunction_with_ao_overlap());

  ASSERT_EQ(populations.size(), 2);
  EXPECT_NEAR(populations[0], 1.342757274492, 1e-12);
  EXPECT_NEAR(populations[1], 0.657242725508, 1e-12);
}

TEST(PopulationAnalyzerTest, QdkAnalyzerUsesMolecularOneRdmAndInactiveCore) {
  auto analyzer = PopulationAnalyzerFactory::create("qdk");

  auto populations = analyzer->run(
      create_correlated_molecular_wavefunction_with_inactive_core());

  ASSERT_EQ(populations.size(), 3);
  EXPECT_DOUBLE_EQ(populations[0], 2.0);
  EXPECT_DOUBLE_EQ(populations[1], 0.5);
  EXPECT_DOUBLE_EQ(populations[2], 0.5);
}
