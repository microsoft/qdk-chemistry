// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace qdk::chemistry::data;

static std::vector<Shell> make_shells() {
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::P, std::vector{1.0}, std::vector{1.0}));
  return shells;
}

static std::shared_ptr<Structure> make_structure() {
  Eigen::MatrixXd coords(1, 3);
  coords << 0.0, 0.0, 0.0;
  return std::make_shared<Structure>(coords, std::vector<std::string>{"H"});
}

TEST(BasisSetAoSymmetries, DefaultIsRestrictedSpin) {
  BasisSet basis("custom", make_shells());

  auto sym = basis.ao_symmetries();
  ASSERT_NE(sym, nullptr);
  EXPECT_TRUE(sym->has_axis(AxisName::Spin));
  EXPECT_TRUE(sym->axis(AxisName::Spin).equivalent());

  const auto& extents = basis.ao_extents();
  const std::size_t num_ao = basis.get_num_atomic_orbitals();
  EXPECT_EQ(extents.size(), 2u);
  EXPECT_EQ(extents.at(SymmetryLabel{axes::alpha()}), num_ao);
  EXPECT_EQ(extents.at(SymmetryLabel{axes::beta()}), num_ao);
}

TEST(BasisSetAoSymmetries, ExplicitDefaultedExtents) {
  auto sym = std::make_shared<const SymmetryProduct>(
      std::vector<SymmetryAxis>{axes::spin(1, true)});
  BasisSet basis("custom", make_shells(), make_structure(), sym);

  const std::size_t num_ao = basis.get_num_atomic_orbitals();
  EXPECT_EQ(basis.ao_extents().at(SymmetryLabel{axes::alpha()}), num_ao);
  EXPECT_EQ(basis.ao_extents().at(SymmetryLabel{axes::beta()}), num_ao);
}

TEST(BasisSetAoSymmetries, RestrictedExtentMismatchThrows) {
  auto sym = std::make_shared<const SymmetryProduct>(
      std::vector<SymmetryAxis>{axes::spin(1, true)});
  std::unordered_map<SymmetryLabel, std::size_t> extents{
      {SymmetryLabel{axes::alpha()}, 4},
      {SymmetryLabel{axes::beta()}, 5},
  };
  EXPECT_THROW(BasisSet("custom", make_shells(), make_structure(), sym,
                        std::move(extents)),
               std::invalid_argument);
}

TEST(BasisSetAoSymmetries, UnrestrictedAllowsDistinctExtents) {
  auto sym = std::make_shared<const SymmetryProduct>(
      std::vector<SymmetryAxis>{axes::spin(1, false)});
  std::unordered_map<SymmetryLabel, std::size_t> extents{
      {SymmetryLabel{axes::alpha()}, 4},
      {SymmetryLabel{axes::beta()}, 5},
  };
  BasisSet basis("custom", make_shells(), make_structure(), sym,
                 std::move(extents));
  EXPECT_EQ(basis.ao_extents().at(SymmetryLabel{axes::alpha()}), 4u);
  EXPECT_EQ(basis.ao_extents().at(SymmetryLabel{axes::beta()}), 5u);
}

TEST(BasisSetAoSymmetries, CopyPreservesAoSymmetries) {
  auto sym = std::make_shared<const SymmetryProduct>(
      std::vector<SymmetryAxis>{axes::spin(1, false)});
  std::unordered_map<SymmetryLabel, std::size_t> extents{
      {SymmetryLabel{axes::alpha()}, 4},
      {SymmetryLabel{axes::beta()}, 5},
  };
  BasisSet basis("custom", make_shells(), make_structure(), sym,
                 std::move(extents));
  BasisSet copy(basis);
  EXPECT_EQ(copy.ao_extents().at(SymmetryLabel{axes::beta()}), 5u);
  EXPECT_FALSE(copy.ao_symmetries()->axis(AxisName::Spin).equivalent());
}
