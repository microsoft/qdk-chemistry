// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class CasWavefunctionTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(CasWavefunctionTest, BasicProperties) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);  // Normalized coefficients

  // dummy rdm for occupation checks
  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  CasWavefunctionContainer cas(coeffs, dets, orbitals, one_rdm, std::nullopt);

  EXPECT_EQ(cas.size(), 3);
  EXPECT_DOUBLE_EQ(std::get<double>(cas.get_coefficient(Configuration("2200"))),
                   0.5);
  EXPECT_DOUBLE_EQ(std::get<double>(cas.get_coefficient(Configuration("2020"))),
                   0.5);
  EXPECT_DOUBLE_EQ(std::get<double>(cas.get_coefficient(Configuration("2002"))),
                   1.0 / sqrt(2));
  EXPECT_THROW(std::get<double>(cas.get_coefficient(Configuration("2000"))),
               std::runtime_error);
  EXPECT_EQ(cas.get_active_determinants().size(), 3);
  EXPECT_EQ(cas.get_active_determinants()[0].to_string(), "2200");
  EXPECT_EQ(cas.get_active_determinants()[1].to_string(), "2020");
  EXPECT_EQ(cas.get_active_determinants()[2].to_string(), "2002");
  EXPECT_EQ(cas.get_total_num_electrons().first, 2);
  EXPECT_EQ(cas.get_total_num_electrons().second, 2);

  // Test new electron counting functions (no inactive orbitals in this test)
  auto [total_alpha_elec, total_beta_elec] = cas.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = cas.get_active_num_electrons();
  EXPECT_EQ(total_alpha_elec, 2);
  EXPECT_EQ(total_beta_elec, 2);
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);
  // With no inactive orbitals, total should equal active
  EXPECT_EQ(total_alpha_elec, active_alpha_elec);
  EXPECT_EQ(total_beta_elec, active_beta_elec);

  auto [alpha_occ, beta_occ] = cas.get_total_orbital_occupations();
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
  auto [alpha_total_occ, beta_total_occ] = cas.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      cas.get_active_orbital_occupations();
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

// Test bounds checking for empty determinant vectors
TEST_F(CasWavefunctionTest, EmptyDeterminantsThrows) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> empty_dets;
  Eigen::VectorXd empty_coeffs(0);

  CasWavefunctionContainer cas(empty_coeffs, empty_dets, orbitals);

  // All these methods should throw when determinants are empty
  EXPECT_THROW(cas.get_coefficient(Configuration("2200")), std::runtime_error);
  EXPECT_THROW(cas.get_total_num_electrons(), std::runtime_error);
  EXPECT_THROW(cas.get_active_num_electrons(), std::runtime_error);
  EXPECT_THROW(cas.get_total_orbital_occupations(), std::runtime_error);
  EXPECT_THROW(cas.get_active_orbital_occupations(), std::runtime_error);

  // size() should return 0, not throw
  EXPECT_EQ(cas.size(), 0);
}

// Test specific error messages for bounds checking
TEST_F(CasWavefunctionTest, ErrorMessagesAreDescriptive) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> empty_dets;
  Eigen::VectorXd empty_coeffs(0);

  CasWavefunctionContainer cas(empty_coeffs, empty_dets, orbitals);

  // Check that error messages are descriptive
  try {
    cas.get_total_orbital_occupations();
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "No determinants available");
  }
}

// Test entropy calculation with missing RDMs
TEST_F(CasWavefunctionTest, EntropyWithMissingRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {Configuration("2200")};
  Eigen::VectorXd coeffs(1);
  coeffs << 1.0;

  CasWavefunctionContainer cas(coeffs, dets, orbitals);

  // Should throw when trying to calculate entropies without RDMs
  EXPECT_THROW(cas.get_single_orbital_entropies(), std::runtime_error);
}

// Test CAS with inactive orbitals to verify total vs active distinction
TEST_F(CasWavefunctionTest, WithInactiveOrbitals) {
  // Create orbitals with inactive space: 6 orbitals total, 2 inactive, 4 active
  auto base_orbitals = testing::create_test_orbitals(6, 6, true);
  std::vector<size_t> active_indices = {2, 3, 4, 5};  // orbitals 2-5 are active
  std::vector<size_t> inactive_indices = {
      0, 1};  // orbitals 0-1 are inactive (doubly occupied)
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  // Create determinants using only the active space (4 orbitals)
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);  // Normalized coefficients

  // dummy rdm for occupation checks
  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  CasWavefunctionContainer cas(coeffs, dets, orbitals, one_rdm, std::nullopt);

  // Test electron counting with inactive orbitals
  auto [total_alpha_elec, total_beta_elec] = cas.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = cas.get_active_num_electrons();

  // Active space has 2 alpha + 2 beta = 4 electrons from determinants
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  // Total includes 2 doubly occupied inactive orbitals = 2 alpha + 2 beta
  // additional
  EXPECT_EQ(total_alpha_elec, 4);  // 2 active + 2 inactive
  EXPECT_EQ(total_beta_elec, 4);   // 2 active + 2 inactive

  // Test orbital occupations with inactive orbitals
  auto [alpha_total_occ, beta_total_occ] = cas.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      cas.get_active_orbital_occupations();

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

// Test CAS with non-continuous active space indices
TEST_F(CasWavefunctionTest, WithNonContinuousActiveSpace) {
  // Create orbitals with non-continuous active space: 8 orbitals total
  // Inactive: {0, 1} (orbitals 0-1)
  // Active: {2, 4, 6, 7} (non-continuous indices!)
  // Virtual: {3, 5} (orbitals 3, 5 are virtual between active orbitals)
  auto base_orbitals = testing::create_test_orbitals(8, 8, true);
  std::vector<size_t> active_indices = {2, 4, 6, 7};  // Non-continuous!
  std::vector<size_t> inactive_indices = {0, 1};      // Doubly occupied
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  // Create determinants using only the active space (4 orbitals)
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);

  // dummy rdm for occupation checks
  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  CasWavefunctionContainer cas(coeffs, dets, orbitals, one_rdm, std::nullopt);

  // Test electron counting with non-continuous active space
  auto [total_alpha_elec, total_beta_elec] = cas.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = cas.get_active_num_electrons();

  // Active space should still have 2 alpha + 2 beta from determinants
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  // Total includes 2 doubly occupied inactive orbitals
  EXPECT_EQ(total_alpha_elec, 4);  // 2 active + 2 inactive
  EXPECT_EQ(total_beta_elec, 4);   // 2 active + 2 inactive

  // Test orbital occupations with non-continuous active space
  auto [alpha_total_occ, beta_total_occ] = cas.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      cas.get_active_orbital_occupations();

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

  // Verify that virtual orbitals (indices 3, 5) have zero occupation
  EXPECT_DOUBLE_EQ(alpha_total_occ(3), 0.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(5), 0.0);
  EXPECT_DOUBLE_EQ(beta_total_occ(3), 0.0);
  EXPECT_DOUBLE_EQ(beta_total_occ(5), 0.0);

  // Verify that inactive orbitals (indices 0, 1) have occupation 1.0
  EXPECT_DOUBLE_EQ(alpha_total_occ(0), 1.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(1), 1.0);
  EXPECT_DOUBLE_EQ(beta_total_occ(0), 1.0);
  EXPECT_DOUBLE_EQ(beta_total_occ(1), 1.0);

  // Verify that active orbitals (indices 2, 4, 6, 7) match active space pattern
  EXPECT_DOUBLE_EQ(alpha_total_occ(2), 1.0);  // First active orbital (occupied)
  EXPECT_DOUBLE_EQ(alpha_total_occ(4),
                   1.0);  // Second active orbital (occupied)
  EXPECT_DOUBLE_EQ(alpha_total_occ(6),
                   0.0);  // Third active orbital (unoccupied)
  EXPECT_DOUBLE_EQ(alpha_total_occ(7),
                   0.0);  // Fourth active orbital (unoccupied)
}

// Test JSON serialization/deserialization
TEST_F(CasWavefunctionTest, JsonSerialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;

  CasWavefunctionContainer original(coeffs, dets, orbitals, one_rdm,
                                    std::nullopt);

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Deserialize from JSON
  auto restored = CasWavefunctionContainer::from_json(j);

  // Verify key properties match
  EXPECT_EQ(original.size(), restored->size());
  EXPECT_EQ(original.get_active_determinants().size(),
            restored->get_active_determinants().size());

  // Verify coefficients match
  const auto& orig_coeffs =
      std::get<Eigen::VectorXd>(original.get_coefficients());
  const auto& rest_coeffs =
      std::get<Eigen::VectorXd>(restored->get_coefficients());
  EXPECT_TRUE(orig_coeffs.isApprox(rest_coeffs, testing::wf_tolerance));

  // Verify determinants match
  for (size_t i = 0; i < original.get_active_determinants().size(); ++i) {
    EXPECT_EQ(original.get_active_determinants()[i],
              restored->get_active_determinants()[i]);
  }
}

// Test HDF5 serialization/deserialization
TEST_F(CasWavefunctionTest, Hdf5Serialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;

  CasWavefunctionContainer original(coeffs, dets, orbitals, one_rdm,
                                    std::nullopt);

  std::string filename = "test_cas_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    auto restored = CasWavefunctionContainer::from_hdf5(root);

    // Verify key properties match
    EXPECT_EQ(original.size(), restored->size());
    EXPECT_EQ(original.get_active_determinants().size(),
              restored->get_active_determinants().size());

    // Verify coefficients match
    const auto& orig_coeffs =
        std::get<Eigen::VectorXd>(original.get_coefficients());
    const auto& rest_coeffs =
        std::get<Eigen::VectorXd>(restored->get_coefficients());
    EXPECT_TRUE(orig_coeffs.isApprox(rest_coeffs, testing::wf_tolerance));

    // Verify determinants match
    for (size_t i = 0; i < original.get_active_determinants().size(); ++i) {
      EXPECT_EQ(original.get_active_determinants()[i],
                restored->get_active_determinants()[i]);
    }

    file.close();
  }

  std::remove(filename.c_str());
}

// Test serialization with complex coefficients
TEST_F(CasWavefunctionTest, SerializationComplex) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {Configuration("2200"),
                                     Configuration("2020")};
  Eigen::VectorXcd coeffs(2);
  coeffs << std::complex<double>(0.5, 0.3), std::complex<double>(0.6, -0.2);

  CasWavefunctionContainer original(coeffs, dets, orbitals);

  // Test JSON
  nlohmann::json j = original.to_json();
  auto restored_json = CasWavefunctionContainer::from_json(j);

  const auto& orig_coeffs =
      std::get<Eigen::VectorXcd>(original.get_coefficients());
  const auto& rest_coeffs =
      std::get<Eigen::VectorXcd>(restored_json->get_coefficients());
  EXPECT_TRUE(orig_coeffs.isApprox(rest_coeffs, testing::wf_tolerance));

  // Test HDF5
  std::string filename = "test_cas_complex_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original.to_hdf5(root);
    auto restored_hdf5 = CasWavefunctionContainer::from_hdf5(root);

    const auto& rest_coeffs_h5 =
        std::get<Eigen::VectorXcd>(restored_hdf5->get_coefficients());
    EXPECT_TRUE(orig_coeffs.isApprox(rest_coeffs_h5, testing::wf_tolerance));

    file.close();
  }
  std::remove(filename.c_str());
}
