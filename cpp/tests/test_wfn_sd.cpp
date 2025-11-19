// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class SlaterdeterminantTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(SlaterdeterminantTest, BasicProperties) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  Configuration det("2200");
  SlaterDeterminantContainer sd(det, orbitals);

  EXPECT_EQ(sd.size(), 1);
  EXPECT_DOUBLE_EQ(std::get<double>(sd.get_coefficient(det)), 1.0);
  EXPECT_DOUBLE_EQ(std::get<double>(sd.get_coefficient(Configuration("2000"))),
                   0.0);
  EXPECT_TRUE(sd.contains_determinant(det));
  EXPECT_FALSE(sd.contains_determinant(Configuration("2000")));
  EXPECT_EQ(sd.get_active_determinants().size(), 1);
  EXPECT_EQ(sd.get_active_determinants()[0].to_string(), "2200");
  EXPECT_DOUBLE_EQ(sd.norm(), 1.0);
  EXPECT_DOUBLE_EQ(sd.get_total_num_electrons().first, 2);
  EXPECT_DOUBLE_EQ(sd.get_total_num_electrons().second, 2);

  // Test new electron counting functions (no inactive orbitals in this test)
  auto [total_alpha_elec, total_beta_elec] = sd.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sd.get_active_num_electrons();
  EXPECT_EQ(total_alpha_elec, 2);
  EXPECT_EQ(total_beta_elec, 2);
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);
  // With no inactive orbitals, total should equal active
  EXPECT_EQ(total_alpha_elec, active_alpha_elec);
  EXPECT_EQ(total_beta_elec, active_beta_elec);

  auto [alpha_occ, beta_occ] = sd.get_total_orbital_occupations();
  EXPECT_EQ(alpha_occ.size(), 4);
  EXPECT_EQ(beta_occ.size(), 4);

  Eigen::VectorXd expected_alpha(4);
  expected_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_beta(4);
  expected_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(alpha_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_occ.isApprox(expected_beta, testing::wf_tolerance));

  // Test total vs active orbital occupations (should be equal with no inactive
  // orbitals)
  auto [alpha_total_occ, beta_total_occ] = sd.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sd.get_active_orbital_occupations();
  EXPECT_EQ(alpha_total_occ.size(), 4);
  EXPECT_EQ(beta_total_occ.size(), 4);
  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);
  EXPECT_TRUE(alpha_total_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(expected_beta, testing::wf_tolerance));
  EXPECT_TRUE(alpha_active_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_active_occ.isApprox(expected_beta, testing::wf_tolerance));
  // With no inactive orbitals, total should equal active
  EXPECT_TRUE(
      alpha_total_occ.isApprox(alpha_active_occ, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(beta_active_occ, testing::wf_tolerance));
}

// Test SD with inactive orbitals to verify total vs active distinction
TEST_F(SlaterdeterminantTest, WithInactiveOrbitals) {
  // Create orbitals with inactive space: 6 orbitals total, 2 inactive, 4 active
  auto base_orbitals = testing::create_test_orbitals(6, 6, true);
  std::vector<size_t> active_indices = {2, 3, 4, 5};  // orbitals 2-5 are active
  std::vector<size_t> inactive_indices = {
      0, 1};  // orbitals 0-1 are inactive (doubly occupied)
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  // Create determinant using only the active space (4 orbitals)
  Configuration det("2200");  // 2 alpha electrons in first 2 active orbitals
  SlaterDeterminantContainer sd(det, orbitals);

  // Test electron counting with inactive orbitals
  auto [total_alpha_elec, total_beta_elec] = sd.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sd.get_active_num_electrons();

  // Active space has 2 alpha + 2 beta = 4 electrons from determinant
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  // Total includes 2 doubly occupied inactive orbitals = 2 alpha + 2 beta
  // additional
  EXPECT_EQ(total_alpha_elec, 4);  // 2 active + 2 inactive
  EXPECT_EQ(total_beta_elec, 4);   // 2 active + 2 inactive

  // Test orbital occupations with inactive orbitals
  auto [alpha_total_occ, beta_total_occ] = sd.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sd.get_active_orbital_occupations();

  // Total occupations should have 6 orbitals (inactive + active)
  EXPECT_EQ(alpha_total_occ.size(), 6);
  EXPECT_EQ(beta_total_occ.size(), 6);

  // Active occupations should have 4 orbitals (only active space)
  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);

  // Active space occupations should match the determinant pattern
  Eigen::VectorXd expected_active_alpha(4);
  expected_active_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_active_beta(4);
  expected_active_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_active_occ.isApprox(expected_active_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_active_occ.isApprox(expected_active_beta, testing::wf_tolerance));

  // Total occupations: inactive (1.0) + active (from determinant) + virtual
  // (0.0)
  Eigen::VectorXd expected_total_alpha(6);
  expected_total_alpha << 1.0, 1.0, 1.0, 1.0, 0.0,
      0.0;  // 2 inactive + 2 active + 2 virtual
  Eigen::VectorXd expected_total_beta(6);
  expected_total_beta << 1.0, 1.0, 1.0, 1.0, 0.0,
      0.0;  // 2 inactive + 2 active + 2 virtual

  EXPECT_TRUE(
      alpha_total_occ.isApprox(expected_total_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_total_occ.isApprox(expected_total_beta, testing::wf_tolerance));
}

// Test SD with non-continuous active space indices
TEST_F(SlaterdeterminantTest, WithNonContinuousActiveSpace) {
  // Create orbitals with non-continuous active space: 8 orbitals total
  // Inactive: {0, 1} (orbitals 0-1)
  // Active: {2, 4, 6, 7} (non-continuous indices!)
  // Virtual: {3, 5} (orbitals 3, 5 are virtual between active orbitals)
  auto base_orbitals = testing::create_test_orbitals(8, 8, true);
  std::vector<size_t> active_indices = {2, 4, 6, 7};  // Non-continuous!
  std::vector<size_t> inactive_indices = {0, 1};      // Doubly occupied
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  // Create determinant using only the active space (4 orbitals)
  Configuration det("2200");  // 2 alpha electrons in first 2 active orbitals
  SlaterDeterminantContainer sd(det, orbitals);

  // Test electron counting with non-continuous active space
  auto [total_alpha_elec, total_beta_elec] = sd.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sd.get_active_num_electrons();

  // Active space should still have 2 alpha + 2 beta from determinant
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  // Total includes 2 doubly occupied inactive orbitals
  EXPECT_EQ(total_alpha_elec, 4);  // 2 active + 2 inactive
  EXPECT_EQ(total_beta_elec, 4);   // 2 active + 2 inactive

  // Test orbital occupations with non-continuous active space
  auto [alpha_total_occ, beta_total_occ] = sd.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sd.get_active_orbital_occupations();

  // Total occupations should have 8 orbitals (all orbitals)
  EXPECT_EQ(alpha_total_occ.size(), 8);
  EXPECT_EQ(beta_total_occ.size(), 8);

  // Active occupations should have 4 orbitals (only active space)
  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);

  // Active space occupations should match the determinant pattern
  Eigen::VectorXd expected_active_alpha(4);
  expected_active_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_active_beta(4);
  expected_active_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_active_occ.isApprox(expected_active_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_active_occ.isApprox(expected_active_beta, testing::wf_tolerance));

  // Total occupations: mapping should be correct for non-continuous indices
  // Orbital indices: 0,   1,   2,   3,   4,   5,   6,   7
  // Types:          ina, ina, act, vir, act, vir, act, act
  // Expected:       1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0
  Eigen::VectorXd expected_total_alpha(8);
  expected_total_alpha << 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  Eigen::VectorXd expected_total_beta(8);
  expected_total_beta << 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_total_occ.isApprox(expected_total_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_total_occ.isApprox(expected_total_beta, testing::wf_tolerance));

  // Verify specific orbital types are correctly handled
  EXPECT_DOUBLE_EQ(alpha_total_occ(3), 0.0);  // Virtual orbital between actives
  EXPECT_DOUBLE_EQ(alpha_total_occ(5), 0.0);  // Virtual orbital between actives
  EXPECT_DOUBLE_EQ(alpha_total_occ(0),
                   1.0);  // Inactive orbital (doubly occupied)
  EXPECT_DOUBLE_EQ(alpha_total_occ(1),
                   1.0);  // Inactive orbital (doubly occupied)
}

// Test JSON serialization/deserialization
TEST_F(SlaterdeterminantTest, JsonSerialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  Configuration det("2200");

  SlaterDeterminantContainer original(det, orbitals);

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Deserialize from JSON
  auto restored = SlaterDeterminantContainer::from_json(j);

  // Verify key properties match
  EXPECT_EQ(original.size(), restored->size());
  EXPECT_EQ(original.get_active_determinants().size(),
            restored->get_active_determinants().size());
  EXPECT_EQ(original.get_active_determinants()[0],
            restored->get_active_determinants()[0]);
}

// Test HDF5 serialization/deserialization
TEST_F(SlaterdeterminantTest, Hdf5Serialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  Configuration det("2200");

  SlaterDeterminantContainer original(det, orbitals);

  std::string filename = "test_sd_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    auto restored = SlaterDeterminantContainer::from_hdf5(root);

    // Verify key properties match
    EXPECT_EQ(original.size(), restored->size());
    EXPECT_EQ(original.get_active_determinants().size(),
              restored->get_active_determinants().size());
    EXPECT_EQ(original.get_active_determinants()[0],
              restored->get_active_determinants()[0]);

    file.close();
  }

  std::remove(filename.c_str());
}
