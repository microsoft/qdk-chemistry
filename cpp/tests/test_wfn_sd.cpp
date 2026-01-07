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

  // Deserialize from JSON using container-specific method
  auto restored = SlaterDeterminantContainer::from_json(j);

  // Also test base Wavefunction::from_json() by wrapping container in
  // Wavefunction
  auto original_wf = std::make_shared<Wavefunction>(
      std::make_unique<SlaterDeterminantContainer>(det, orbitals));
  nlohmann::json wf_j = original_wf->to_json();
  auto wf_restored = Wavefunction::from_json(wf_j);
  EXPECT_EQ(wf_restored->get_container_type(), "sd");
  auto& wf_restored_container =
      wf_restored->get_container<SlaterDeterminantContainer>();

  // Verify key properties match
  EXPECT_EQ(original.size(), restored->size());
  EXPECT_EQ(original.get_active_determinants().size(),
            restored->get_active_determinants().size());
  EXPECT_EQ(original.get_active_determinants()[0],
            restored->get_active_determinants()[0]);

  // Verify that base Wavefunction::from_json gives the same result
  EXPECT_EQ(restored->size(), wf_restored_container.size());
  EXPECT_EQ(restored->get_active_determinants().size(),
            wf_restored_container.get_active_determinants().size());
  EXPECT_EQ(restored->get_active_determinants()[0],
            wf_restored_container.get_active_determinants()[0]);
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

    // Deserialize from HDF5 using container-specific method
    auto restored = SlaterDeterminantContainer::from_hdf5(root);

    // Verify key properties match
    EXPECT_EQ(original.size(), restored->size());
    EXPECT_EQ(original.get_active_determinants().size(),
              restored->get_active_determinants().size());
    EXPECT_EQ(original.get_active_determinants()[0],
              restored->get_active_determinants()[0]);

    file.close();
  }

  // Also test base Wavefunction::from_hdf5() by creating a separate file with
  // Wavefunction wrapper
  std::string wf_filename = "test_sd_wavefunction_serialization.h5";
  {
    // Create and serialize a Wavefunction wrapping the container
    auto original_wf = std::make_shared<Wavefunction>(
        std::make_unique<SlaterDeterminantContainer>(det, orbitals));
    H5::H5File file(wf_filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original_wf->to_hdf5(root);
    file.close();
  }
  {
    // Deserialize using Wavefunction::from_hdf5
    H5::H5File file(wf_filename, H5F_ACC_RDONLY);
    H5::Group root = file.openGroup("/");
    auto wf_restored = Wavefunction::from_hdf5(root);
    EXPECT_EQ(wf_restored->get_container_type(), "sd");
    auto& wf_restored_container =
        wf_restored->get_container<SlaterDeterminantContainer>();

    // Get the restored container from container-specific method for comparison
    H5::H5File file2(filename, H5F_ACC_RDONLY);
    H5::Group root2 = file2.openGroup("/");
    auto restored = SlaterDeterminantContainer::from_hdf5(root2);

    EXPECT_EQ(restored->size(), wf_restored_container.size());
    EXPECT_EQ(restored->get_active_determinants().size(),
              wf_restored_container.get_active_determinants().size());
    EXPECT_EQ(restored->get_active_determinants()[0],
              wf_restored_container.get_active_determinants()[0]);

    file.close();
    file2.close();
  }

  std::remove(filename.c_str());
  std::remove(wf_filename.c_str());
}

// Test 1- and 2-RDM for closed-shell system
TEST_F(SlaterdeterminantTest, ClosedShellReducedDensityMatrices) {
  // get slater determinant
  size_t norb = 4;
  auto orbitals = testing::create_test_orbitals(norb, norb, true);
  Configuration det("2200");
  SlaterDeterminantContainer sd(det, orbitals);

  // get RDMs
  auto one_rdm = std::get<Eigen::MatrixXd>(sd.get_active_one_rdm_spin_traced());
  auto two_rdm = std::get<Eigen::VectorXd>(sd.get_active_two_rdm_spin_traced());
  auto [one_rdm_aa, one_rdm_bb] = sd.get_active_one_rdm_spin_dependent();
  auto [two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb] =
      sd.get_active_two_rdm_spin_dependent();

  // get bbaa from transposed aabb
  auto two_rdm_bbaa =
      detail::transpose_ijkl_klij_vector_variant(two_rdm_aabb, norb);

  // Expected 1-RDM
  Eigen::MatrixXd expected_one_rdm = Eigen::MatrixXd::Zero(norb, norb);
  expected_one_rdm(0, 0) = 2.0;
  expected_one_rdm(1, 1) = 2.0;
  EXPECT_TRUE(
      one_rdm.isApprox(expected_one_rdm, testing::numerical_zero_tolerance));

  // spin traced 1-RDM is the sum of spin dependent 1-RDMs
  Eigen::MatrixXd one_rdm_sum = std::get<Eigen::MatrixXd>(one_rdm_aa) +
                                std::get<Eigen::MatrixXd>(one_rdm_bb);
  EXPECT_TRUE(one_rdm_sum.isApprox(expected_one_rdm,
                                   testing::numerical_zero_tolerance));

  // check spin dependent 2-RDM
  auto sum_two_rdm = std::get<Eigen::VectorXd>(two_rdm_aabb) +
                     std::get<Eigen::VectorXd>(*two_rdm_bbaa) +
                     std::get<Eigen::VectorXd>(two_rdm_aaaa) +
                     std::get<Eigen::VectorXd>(two_rdm_bbbb);

  for (size_t i = 0; i < two_rdm.size(); ++i) {
    EXPECT_NEAR(two_rdm(i), sum_two_rdm(i), testing::numerical_zero_tolerance);
  }
}

// Test 1- and 2-RDM for open-shell system
TEST_F(SlaterdeterminantTest, OpenShellReducedDensityMatrices) {
  // get slater determinant
  size_t norb = 4;
  auto orbitals = testing::create_test_orbitals(norb, norb, true);
  Configuration det("2uu0");
  SlaterDeterminantContainer sd(det, orbitals);

  // get RDMs
  auto one_rdm = std::get<Eigen::MatrixXd>(sd.get_active_one_rdm_spin_traced());
  auto two_rdm = std::get<Eigen::VectorXd>(sd.get_active_two_rdm_spin_traced());
  auto [one_rdm_aa, one_rdm_bb] = sd.get_active_one_rdm_spin_dependent();
  auto [two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb] =
      sd.get_active_two_rdm_spin_dependent();

  // get bbaa from transposed aabb
  auto two_rdm_bbaa =
      detail::transpose_ijkl_klij_vector_variant(two_rdm_aabb, norb);

  // Expected 1-RDM
  Eigen::MatrixXd expected_one_rdm = Eigen::MatrixXd::Zero(norb, norb);
  expected_one_rdm(0, 0) = 2.0;
  expected_one_rdm(1, 1) = 1.0;
  expected_one_rdm(2, 2) = 1.0;
  EXPECT_TRUE(
      one_rdm.isApprox(expected_one_rdm, testing::numerical_zero_tolerance));

  // spin traced 1-RDM is the sum of spin dependent 1-RDMs
  Eigen::MatrixXd one_rdm_sum = std::get<Eigen::MatrixXd>(one_rdm_aa) +
                                std::get<Eigen::MatrixXd>(one_rdm_bb);
  EXPECT_TRUE(one_rdm_sum.isApprox(expected_one_rdm,
                                   testing::numerical_zero_tolerance));

  // check spin dependent 2-RDM
  auto sum_two_rdm = std::get<Eigen::VectorXd>(two_rdm_aabb) +
                     std::get<Eigen::VectorXd>(*two_rdm_bbaa) +
                     std::get<Eigen::VectorXd>(two_rdm_aaaa) +
                     std::get<Eigen::VectorXd>(two_rdm_bbbb);

  for (size_t i = 0; i < two_rdm.size(); ++i) {
    EXPECT_NEAR(two_rdm(i), sum_two_rdm(i), testing::numerical_zero_tolerance);
  }
}

// Test 1- and 2-RDM for non Aufbau determinants
TEST_F(SlaterdeterminantTest, NonContinuousDeterminantReducedDensityMatrices) {
  // get slater determinant
  size_t norb = 12;
  auto orbitals = testing::create_test_orbitals(norb, norb, true);
  Configuration det("2ud0200u0u2d");
  SlaterDeterminantContainer sd(det, orbitals);

  // get RDMs
  auto one_rdm = std::get<Eigen::MatrixXd>(sd.get_active_one_rdm_spin_traced());
  auto two_rdm = std::get<Eigen::VectorXd>(sd.get_active_two_rdm_spin_traced());
  auto [one_rdm_aa, one_rdm_bb] = sd.get_active_one_rdm_spin_dependent();
  auto [two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb] =
      sd.get_active_two_rdm_spin_dependent();

  // get bbaa from transposed aabb
  auto two_rdm_bbaa =
      detail::transpose_ijkl_klij_vector_variant(two_rdm_aabb, norb);

  // Expected 1-RDM
  Eigen::MatrixXd expected_one_rdm_aa = Eigen::MatrixXd::Zero(norb, norb);
  Eigen::MatrixXd expected_one_rdm_bb = Eigen::MatrixXd::Zero(norb, norb);
  expected_one_rdm_aa(0, 0) = 1.0;
  expected_one_rdm_aa(1, 1) = 1.0;
  expected_one_rdm_aa(4, 4) = 1.0;
  expected_one_rdm_aa(7, 7) = 1.0;
  expected_one_rdm_aa(9, 9) = 1.0;
  expected_one_rdm_aa(10, 10) = 1.0;
  expected_one_rdm_bb(0, 0) = 1.0;
  expected_one_rdm_bb(2, 2) = 1.0;
  expected_one_rdm_bb(4, 4) = 1.0;
  expected_one_rdm_bb(10, 10) = 1.0;
  expected_one_rdm_bb(11, 11) = 1.0;

  // spin traced 1-RDM is the sum of spin dependent 1-RDMs
  EXPECT_TRUE(
      std::get<Eigen::MatrixXd>(one_rdm_aa)
          .isApprox(expected_one_rdm_aa, testing::numerical_zero_tolerance));
  EXPECT_TRUE(
      std::get<Eigen::MatrixXd>(one_rdm_bb)
          .isApprox(expected_one_rdm_bb, testing::numerical_zero_tolerance));
  Eigen::MatrixXd one_rdm_sum = std::get<Eigen::MatrixXd>(one_rdm_aa) +
                                std::get<Eigen::MatrixXd>(one_rdm_bb);
  EXPECT_TRUE(one_rdm_sum.isApprox(one_rdm, testing::numerical_zero_tolerance));

  // check spin dependent 2-RDM
  auto sum_two_rdm = std::get<Eigen::VectorXd>(two_rdm_aabb) +
                     std::get<Eigen::VectorXd>(*two_rdm_bbaa) +
                     std::get<Eigen::VectorXd>(two_rdm_aaaa) +
                     std::get<Eigen::VectorXd>(two_rdm_bbbb);

  for (size_t i = 0; i < two_rdm.size(); ++i) {
    EXPECT_NEAR(two_rdm(i), sum_two_rdm(i), testing::numerical_zero_tolerance);
  }
}

// Test entropies
TEST_F(SlaterdeterminantTest, EntropiesTest) {
  // get slater determinant
  size_t norb = 12;
  auto orbitals = testing::create_test_orbitals(norb, norb, true);
  Configuration det("2ud0200u0u2d");
  SlaterDeterminantContainer sd(det, orbitals);
  sd.get_active_one_rdm_spin_dependent();
  auto [aabb, aaaa, bbbb] = sd.get_active_two_rdm_spin_dependent();

  // get entropies
  auto s1 = sd.get_single_orbital_entropies();

  // Expected entropies are all zero for single determinant
  Eigen::VectorXd expected_s1 = Eigen::VectorXd::Zero(norb);
  EXPECT_TRUE(s1.isApprox(expected_s1, testing::numerical_zero_tolerance));
}
