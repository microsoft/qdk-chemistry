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

TEST(OrbitalsSbtTest, RestrictedCoefficientsAlias) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(4, 4);
  Eigen::VectorXd e = Eigen::VectorXd::Random(4);
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals = std::make_shared<Orbitals>(c, e, std::nullopt, basis);

  auto coefficients = orbitals->coefficients();
  EXPECT_TRUE(coefficients->has_block({alpha(), alpha()}));
  EXPECT_TRUE(coefficients->has_block({beta(), beta()}));
  // Restricted: alpha and beta blocks share storage.
  EXPECT_EQ(coefficients->block_ptr({alpha(), alpha()}).get(),
            coefficients->block_ptr({beta(), beta()}).get());
  EXPECT_TRUE(coefficients->block({alpha(), alpha()}).isApprox(c));

  auto energies = orbitals->energies();
  EXPECT_TRUE(energies->block({alpha()}).isApprox(e));
  EXPECT_TRUE(energies->block({beta()}).isApprox(e));
}

TEST(OrbitalsSbtTest, UnrestrictedCoefficientsDistinct) {
  Eigen::MatrixXd ca = Eigen::MatrixXd::Random(4, 4);
  Eigen::MatrixXd cb = Eigen::MatrixXd::Random(4, 4);
  Eigen::VectorXd ea = Eigen::VectorXd::Random(4);
  Eigen::VectorXd eb = Eigen::VectorXd::Random(4);
  auto basis = testing::create_random_basis_set(ca.rows());
  auto orbitals =
      std::make_shared<Orbitals>(ca, cb, ea, eb, std::nullopt, basis);

  auto coefficients = orbitals->coefficients();
  // Unrestricted: alpha and beta blocks should be distinct.
  EXPECT_NE(coefficients->block_ptr({alpha(), alpha()}).get(),
            coefficients->block_ptr({beta(), beta()}).get());
  EXPECT_TRUE(coefficients->block({alpha(), alpha()}).isApprox(ca));
  EXPECT_TRUE(coefficients->block({beta(), beta()}).isApprox(cb));
}

TEST(OrbitalsSbtTest, ActiveInactiveVectorsReflectIndices) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(5, 5);
  // active = {1, 2}, inactive = {0}
  Orbitals::RestrictedCASIndices indices({1, 2}, {0});
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals = std::make_shared<Orbitals>(
      c, std::nullopt, std::nullopt, basis, std::make_optional(indices));
  auto [active_alpha, active_beta] = orbitals->get_active_space_indices();
  auto [inactive_alpha, inactive_beta] = orbitals->get_inactive_space_indices();
  EXPECT_EQ(active_alpha, std::vector<size_t>({1, 2}));
  EXPECT_EQ(active_beta, std::vector<size_t>({1, 2}));
  EXPECT_EQ(inactive_alpha, std::vector<size_t>({0}));
  EXPECT_EQ(inactive_beta, std::vector<size_t>({0}));
  // virtual = {3, 4}
  auto [virtual_alpha, virtual_beta] = orbitals->get_virtual_space_indices();
  EXPECT_EQ(virtual_alpha, std::vector<size_t>({3, 4}));
  EXPECT_EQ(virtual_beta, std::vector<size_t>({3, 4}));
}

TEST(OrbitalsSbtTest, SbtNativeConstructorRoundTrips) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(4, 4);
  Eigen::VectorXd e = Eigen::VectorXd::Random(4);
  auto basis = testing::create_random_basis_set(c.rows());
  auto src = std::make_shared<Orbitals>(c, e, std::nullopt, basis);

  // Reconstruct from the SBT-native containers.
  auto rebuilt = std::make_shared<Orbitals>(
      src->coefficients(), src->energies(), std::nullopt, basis);

  EXPECT_TRUE(rebuilt->is_restricted());
  EXPECT_EQ(rebuilt->get_num_molecular_orbitals(), 4u);
  EXPECT_TRUE(rebuilt->coefficients()->block({alpha(), alpha()}).isApprox(c));
  EXPECT_TRUE(rebuilt->energies()->block({alpha()}).isApprox(e));
}
