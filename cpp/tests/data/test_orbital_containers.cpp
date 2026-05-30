// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <memory>
#include <qdk/chemistry/data/errors.hpp>
#include <qdk/chemistry/data/orbital_containers/basis_coefficients.hpp>
#include <qdk/chemistry/data/orbital_containers/orbital_energies.hpp>
#include <qdk/chemistry/data/orbital_containers/orbital_space_partitioning.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>

using namespace qdk::chemistry::data;

namespace {

SymmetryLabel alpha() { return SymmetryLabel({axes::alpha()}); }
SymmetryLabel beta() { return SymmetryLabel({axes::beta()}); }

std::shared_ptr<const Symmetries> restricted_spin() {
  return std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, /*equivalent=*/true)}));
}

std::unordered_map<SymmetryLabel, std::size_t> extents(std::size_t n) {
  std::unordered_map<SymmetryLabel, std::size_t> e;
  e.emplace(alpha(), n);
  e.emplace(beta(), n);
  return e;
}

}  // namespace

TEST(OrbitalEnergiesTest, WrapsTensorAndExposesAccessors) {
  using Sbt = OrbitalEnergies::Sbt;
  auto sym = restricted_spin();
  auto block = std::make_shared<const Eigen::VectorXd>(Eigen::VectorXd(3));
  Sbt::BlockMap blocks;
  blocks.emplace(Sbt::Labels{alpha()}, block);
  auto sbt = std::make_shared<const Sbt>(Sbt::SymmetriesArray{sym},
                                         Sbt::ExtentsArray{extents(3)}, blocks);

  OrbitalEnergies energies(sbt);
  EXPECT_EQ(energies.get_data_type_name(), "orbital_energies");
  EXPECT_TRUE(energies.has_block(alpha()));
  EXPECT_TRUE(energies.has_block(beta()));
  EXPECT_EQ(energies.block(alpha()).size(), 3);
  EXPECT_EQ(energies.mo_extents().at(alpha()), 3u);
  EXPECT_EQ(energies.symmetries().get(), sym.get());
}

TEST(OrbitalEnergiesTest, NullTensorRejected) {
  EXPECT_THROW(OrbitalEnergies(nullptr), std::invalid_argument);
}

TEST(BasisCoefficientsTest, RestrictedAliasesSpinBlocks) {
  using Sbt = BasisCoefficients::Sbt;
  auto sym = restricted_spin();
  auto block =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(4, 4));
  Sbt::BlockMap blocks;
  blocks.emplace(Sbt::Labels{alpha(), alpha()}, block);
  auto sbt = std::make_shared<const Sbt>(
      Sbt::SymmetriesArray{sym, sym}, Sbt::ExtentsArray{extents(4), extents(4)},
      blocks);

  BasisCoefficients coeffs(sbt);
  EXPECT_EQ(coeffs.get_data_type_name(), "basis_coefficients");
  EXPECT_TRUE(coeffs.is_restricted());
  EXPECT_TRUE(coeffs.has_block(alpha(), alpha()));
  EXPECT_TRUE(coeffs.has_block(beta(), beta()));
  EXPECT_EQ(coeffs.block(alpha(), alpha()).rows(), 4);
  EXPECT_EQ(coeffs.ao_extents().at(alpha()), 4u);
  EXPECT_EQ(coeffs.mo_extents().at(beta()), 4u);
}

TEST(BasisCoefficientsTest, JsonRoundTrip) {
  using Sbt = BasisCoefficients::Sbt;
  auto sym = restricted_spin();
  auto block =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(2, 2));
  Sbt::BlockMap blocks;
  blocks.emplace(Sbt::Labels{alpha(), alpha()}, block);
  auto sbt = std::make_shared<const Sbt>(
      Sbt::SymmetriesArray{sym, sym}, Sbt::ExtentsArray{extents(2), extents(2)},
      blocks);

  BasisCoefficients coeffs(sbt);
  auto restored = BasisCoefficients::from_json(coeffs.to_json());
  EXPECT_TRUE(restored->is_restricted());
  EXPECT_EQ(restored->block(alpha(), alpha()).rows(), 2);
}

TEST(OrbitalSpacePartitioningTest, AllActivePutsEverythingInActive) {
  auto sym = restricted_spin();
  auto partitioning = OrbitalSpacePartitioning::all_active(sym, extents(5));

  EXPECT_EQ(partitioning->get_data_type_name(), "orbital_space_partitioning");
  EXPECT_EQ(partitioning->active()->indices(alpha()).size(), 5u);
  EXPECT_EQ(partitioning->active()->indices(beta()).size(), 5u);
  EXPECT_EQ(partitioning->frozen()->indices(alpha()).size(), 0u);
  EXPECT_EQ(partitioning->inactive()->indices(alpha()).size(), 0u);
  EXPECT_EQ(partitioning->virtual_orbitals()->indices(alpha()).size(), 0u);
  EXPECT_EQ(partitioning->external()->indices(alpha()).size(), 0u);
}

TEST(OrbitalSpacePartitioningTest, JsonRoundTrip) {
  auto sym = restricted_spin();
  auto partitioning = OrbitalSpacePartitioning::all_active(sym, extents(3));
  auto restored = OrbitalSpacePartitioning::from_json(partitioning->to_json());
  EXPECT_EQ(restored->active()->indices(alpha()).size(), 3u);
  EXPECT_EQ(restored->frozen()->indices(alpha()).size(), 0u);
}
