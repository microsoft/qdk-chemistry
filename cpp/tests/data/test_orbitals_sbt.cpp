// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <memory>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>

#include "../ut_common.hpp"

using namespace qdk::chemistry::data;

TEST(OrbitalsSbtTest, ExposesModeSymmetryLayout) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(3, 3);
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals =
      std::make_shared<Orbitals>(c, std::nullopt, std::nullopt, basis);
  EXPECT_EQ(orbitals->num_modes(), 3u);
  EXPECT_EQ(orbitals->mo_extents().at(axes::alpha()), 3u);
  EXPECT_EQ(orbitals->mo_extents().at(axes::beta()), 3u);
  EXPECT_NE(orbitals->symmetries(), nullptr);
}

TEST(OrbitalsSbtTest, RestrictedCoefficientsAlias) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Random(4, 4);
  Eigen::VectorXd e = Eigen::VectorXd::Random(4);
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals = std::make_shared<Orbitals>(c, e, std::nullopt, basis);

  auto coefficients = orbitals->coefficients();
  EXPECT_TRUE(coefficients->has_block({axes::alpha(), axes::alpha()}));
  EXPECT_TRUE(coefficients->has_block({axes::beta(), axes::beta()}));
  // Restricted: alpha and beta blocks share storage.
  EXPECT_EQ(coefficients->block_ptr({axes::alpha(), axes::alpha()}).get(),
            coefficients->block_ptr({axes::beta(), axes::beta()}).get());
  EXPECT_TRUE(coefficients->block({axes::alpha(), axes::alpha()}).isApprox(c));

  auto energies = orbitals->energies();
  EXPECT_TRUE(energies->block({axes::alpha()}).isApprox(e));
  EXPECT_TRUE(energies->block({axes::beta()}).isApprox(e));
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
  EXPECT_NE(coefficients->block_ptr({axes::alpha(), axes::alpha()}).get(),
            coefficients->block_ptr({axes::beta(), axes::beta()}).get());
  EXPECT_TRUE(coefficients->block({axes::alpha(), axes::alpha()}).isApprox(ca));
  EXPECT_TRUE(coefficients->block({axes::beta(), axes::beta()}).isApprox(cb));
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

TEST(OrbitalsSbtTest, ActiveInactiveIndexSetsAreBuiltLazilyFromDenseVectors) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(5, 5);
  Orbitals::UnrestrictedCASIndices indices({1, 3}, {0, 4}, {0}, {2});
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals =
      std::make_shared<Orbitals>(c, c, std::nullopt, std::nullopt, std::nullopt,
                                 basis, std::make_optional(indices));

  auto active = orbitals->active_indices();
  auto inactive = orbitals->inactive_indices();
  ASSERT_NE(active, nullptr);
  ASSERT_NE(inactive, nullptr);
  EXPECT_EQ(active.get(), orbitals->active_indices().get());
  EXPECT_EQ(inactive.get(), orbitals->inactive_indices().get());

  auto active_alpha = active->indices(axes::alpha());
  auto active_beta = active->indices(axes::beta());
  ASSERT_EQ(active_alpha.size(), 2u);
  ASSERT_EQ(active_beta.size(), 2u);
  EXPECT_EQ(active_alpha[0], 1u);
  EXPECT_EQ(active_alpha[1], 3u);
  EXPECT_EQ(active_beta[0], 0u);
  EXPECT_EQ(active_beta[1], 4u);
  EXPECT_EQ(active->extents().at(axes::alpha()), 5u);
  EXPECT_EQ(active->extents().at(axes::beta()), 5u);

  auto inactive_alpha = inactive->indices(axes::alpha());
  auto inactive_beta = inactive->indices(axes::beta());
  ASSERT_EQ(inactive_alpha.size(), 1u);
  ASSERT_EQ(inactive_beta.size(), 1u);
  EXPECT_EQ(inactive_alpha[0], 0u);
  EXPECT_EQ(inactive_beta[0], 2u);
}

TEST(OrbitalsSbtTest, IndexSetsRebuildFromSerializedDenseIndices) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(4, 4);
  Orbitals::RestrictedCASIndices indices({1, 2}, {0});
  auto basis = testing::create_random_basis_set(c.rows());
  auto orbitals = std::make_shared<Orbitals>(
      c, std::nullopt, std::nullopt, basis, std::make_optional(indices));

  auto restored_from_json = Orbitals::from_json(orbitals->to_json());
  auto active_from_json = restored_from_json->active_indices();
  auto inactive_from_json = restored_from_json->inactive_indices();
  ASSERT_NE(active_from_json, nullptr);
  ASSERT_NE(inactive_from_json, nullptr);
  EXPECT_EQ(active_from_json->indices(axes::alpha()).size(), 2u);
  EXPECT_EQ(active_from_json->indices(axes::beta()).size(), 2u);
  EXPECT_EQ(inactive_from_json->indices(axes::alpha()).size(), 1u);
  EXPECT_EQ(inactive_from_json->indices(axes::beta()).size(), 1u);

  const auto filename = "orbitals_sbt_index_sets.orbitals.h5";
  orbitals->to_hdf5_file(filename);
  auto cleanup = [&filename]() { std::filesystem::remove(filename); };

  auto restored_from_hdf5 = Orbitals::from_hdf5_file(filename);
  cleanup();

  auto active_from_hdf5 = restored_from_hdf5->active_indices();
  auto inactive_from_hdf5 = restored_from_hdf5->inactive_indices();
  ASSERT_NE(active_from_hdf5, nullptr);
  ASSERT_NE(inactive_from_hdf5, nullptr);
  EXPECT_EQ(active_from_hdf5->indices(axes::alpha())[0], 1u);
  EXPECT_EQ(active_from_hdf5->indices(axes::alpha())[1], 2u);
  EXPECT_EQ(active_from_hdf5->indices(axes::beta())[0], 1u);
  EXPECT_EQ(active_from_hdf5->indices(axes::beta())[1], 2u);
  EXPECT_EQ(inactive_from_hdf5->indices(axes::alpha())[0], 0u);
  EXPECT_EQ(inactive_from_hdf5->indices(axes::beta())[0], 0u);
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
  EXPECT_TRUE(rebuilt->coefficients()
                  ->block({axes::alpha(), axes::alpha()})
                  .isApprox(c));
  EXPECT_TRUE(rebuilt->energies()->block({axes::alpha()}).isApprox(e));
}
