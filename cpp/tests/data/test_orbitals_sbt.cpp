// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/single_particle_basis.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>

#include "../ut_common.hpp"

using namespace qdk::chemistry::data;

namespace {

SymmetryLabel alpha() { return SymmetryLabel({axes::alpha()}); }
SymmetryLabel beta() { return SymmetryLabel({axes::beta()}); }

}  // namespace

TEST(OrbitalsSbtTest, IsSingleParticleBasis) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(3, 3);
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals =
      std::make_shared<Orbitals>(c, std::nullopt, std::nullopt, basis);
  std::shared_ptr<const SingleParticleBasis> spb = orbitals;
  EXPECT_EQ(spb->num_modes(), 3u);
  EXPECT_EQ(spb->mo_extents().at(alpha()), 3u);
  EXPECT_EQ(spb->mo_extents().at(beta()), 3u);
  EXPECT_NE(spb->symmetries(), nullptr);
}

TEST(OrbitalsSbtTest, RestrictedBasisCoefficientsAlias) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(4, 4);
  Eigen::VectorXd e = Eigen::VectorXd::Random(4);
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals = std::make_shared<Orbitals>(c, e, std::nullopt, basis);

  auto bc = orbitals->basis_coefficients();
  EXPECT_TRUE(bc->is_restricted());
  EXPECT_TRUE(bc->has_block(alpha(), alpha()));
  EXPECT_TRUE(bc->has_block(beta(), beta()));
  EXPECT_TRUE(bc->block(alpha(), alpha()).isApprox(c));

  auto oe = orbitals->orbital_energies();
  EXPECT_TRUE(oe->block(alpha()).isApprox(e));
  EXPECT_TRUE(oe->block(beta()).isApprox(e));
}

TEST(OrbitalsSbtTest, UnrestrictedBasisCoefficientsDistinct) {
  Eigen::MatrixXd ca = Eigen::MatrixXd::Random(4, 4);
  Eigen::MatrixXd cb = Eigen::MatrixXd::Random(4, 4);
  Eigen::VectorXd ea = Eigen::VectorXd::Random(4);
  Eigen::VectorXd eb = Eigen::VectorXd::Random(4);
  auto basis = testing::create_random_basis_set(ca.rows());
  auto orbitals =
      std::make_shared<Orbitals>(ca, cb, ea, eb, std::nullopt, basis);

  auto bc = orbitals->basis_coefficients();
  EXPECT_FALSE(bc->is_restricted());
  EXPECT_TRUE(bc->block(alpha(), alpha()).isApprox(ca));
  EXPECT_TRUE(bc->block(beta(), beta()).isApprox(cb));
}

TEST(OrbitalsSbtTest, PartitioningDefaultsAllActive) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(5, 5);
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals =
      std::make_shared<Orbitals>(c, std::nullopt, std::nullopt, basis);
  auto part = orbitals->orbital_space_partitioning();
  EXPECT_EQ(part->active()->indices(alpha()).size(), 5u);
  EXPECT_EQ(part->inactive()->indices(alpha()).size(), 0u);
}

TEST(OrbitalsSbtTest, PartitioningReflectsActiveInactive) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(5, 5);
  // active = {1, 2}, inactive = {0}
  Orbitals::RestrictedCASIndices indices({1, 2}, {0});
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals = std::make_shared<Orbitals>(
      c, std::nullopt, std::nullopt, basis, std::make_optional(indices));
  auto part = orbitals->orbital_space_partitioning();
  EXPECT_EQ(part->active()->indices(alpha()).size(), 2u);
  EXPECT_EQ(part->inactive()->indices(alpha()).size(), 1u);
  // virtual = {3, 4}
  EXPECT_EQ(part->virtual_orbitals()->indices(alpha()).size(), 2u);
  EXPECT_EQ(part->frozen()->indices(alpha()).size(), 0u);
}

TEST(OrbitalsSbtTest, SbtNativeConstructorRoundTrips) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(4, 4);
  Eigen::VectorXd e = Eigen::VectorXd::Random(4);
  auto basis = testing::create_random_basis_set(c.rows());
  auto src = std::make_shared<Orbitals>(c, e, std::nullopt, basis);

  // Reconstruct from the SBT-native containers.
  auto rebuilt = std::make_shared<Orbitals>(
      src->basis_coefficients(), src->orbital_energies(),
      src->orbital_space_partitioning(), std::nullopt, basis);

  EXPECT_TRUE(rebuilt->is_restricted());
  EXPECT_EQ(rebuilt->get_num_molecular_orbitals(), 4u);
  EXPECT_TRUE(
      rebuilt->basis_coefficients()->block(alpha(), alpha()).isApprox(c));
  EXPECT_TRUE(rebuilt->orbital_energies()->block(alpha()).isApprox(e));
}

TEST(OrbitalsSbtTest, AoSymmetriesHelperReturnsBasisSymmetriesForOrbitals) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(4, 4);
  Eigen::VectorXd e = Eigen::VectorXd::Random(4);
  auto basis = testing::create_random_basis_set(c.rows());
  std::shared_ptr<const SingleParticleBasis> orbitals =
      std::make_shared<Orbitals>(c, e, std::nullopt, basis);

  auto syms = ao_symmetries(orbitals);
  ASSERT_NE(syms, nullptr);
  EXPECT_EQ(syms.get(), basis->ao_symmetries().get());
}

TEST(OrbitalsSbtTest, AoSymmetriesHelperReturnsNullForModelOrbitals) {
  std::shared_ptr<const SingleParticleBasis> model =
      std::make_shared<ModelOrbitals>(4, true);
  EXPECT_EQ(ao_symmetries(model), nullptr);
}

TEST(OrbitalsSbtTest, AoSymmetriesHelperReturnsNullForNullptr) {
  EXPECT_EQ(ao_symmetries(std::shared_ptr<const SingleParticleBasis>()),
            nullptr);
}
