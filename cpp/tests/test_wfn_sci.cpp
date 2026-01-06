// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class SciWavefunctionTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(SciWavefunctionTest, BasicProperties) {
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

  SciWavefunctionContainer sci(coeffs, dets, orbitals, one_rdm, std::nullopt);

  EXPECT_EQ(sci.size(), 3);
  EXPECT_DOUBLE_EQ(std::get<double>(sci.get_coefficient(Configuration("2200"))),
                   0.5);
  EXPECT_DOUBLE_EQ(std::get<double>(sci.get_coefficient(Configuration("2020"))),
                   0.5);
  EXPECT_DOUBLE_EQ(std::get<double>(sci.get_coefficient(Configuration("2002"))),
                   1.0 / sqrt(2));
  EXPECT_THROW(std::get<double>(sci.get_coefficient(Configuration("2000"))),
               std::runtime_error);
  EXPECT_EQ(sci.get_active_determinants().size(), 3);
  EXPECT_EQ(sci.get_active_determinants()[0].to_string(), "2200");
  EXPECT_EQ(sci.get_active_determinants()[1].to_string(), "2020");
  EXPECT_EQ(sci.get_active_determinants()[2].to_string(), "2002");
  EXPECT_EQ(sci.get_total_num_electrons().first, 2);
  EXPECT_EQ(sci.get_total_num_electrons().second, 2);

  // Test new electron counting functions (no inactive orbitals in this test)
  auto [total_alpha_elec, total_beta_elec] = sci.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sci.get_active_num_electrons();
  EXPECT_EQ(total_alpha_elec, 2);
  EXPECT_EQ(total_beta_elec, 2);
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);
  // With no inactive orbitals, total should equal active
  EXPECT_EQ(total_alpha_elec, active_alpha_elec);
  EXPECT_EQ(total_beta_elec, active_beta_elec);

  auto [alpha_occ, beta_occ] = sci.get_total_orbital_occupations();
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
  auto [alpha_total_occ, beta_total_occ] = sci.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sci.get_active_orbital_occupations();
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
TEST_F(SciWavefunctionTest, EmptyDeterminantsThrows) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> empty_dets;
  Eigen::VectorXd empty_coeffs(0);

  SciWavefunctionContainer sci(empty_coeffs, empty_dets, orbitals);

  // All these methods should throw when determinants are empty
  EXPECT_THROW(sci.get_coefficient(Configuration("2200")), std::runtime_error);
  EXPECT_THROW(sci.get_total_num_electrons(), std::runtime_error);
  EXPECT_THROW(sci.get_active_num_electrons(), std::runtime_error);
  EXPECT_THROW(sci.get_total_orbital_occupations(), std::runtime_error);
  EXPECT_THROW(sci.get_active_orbital_occupations(), std::runtime_error);

  // size() should return 0, not throw
  EXPECT_EQ(sci.size(), 0);
}

// Test specific error messages for bounds checking
TEST_F(SciWavefunctionTest, ErrorMessagesAreDescriptive) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> empty_dets;
  Eigen::VectorXd empty_coeffs(0);

  SciWavefunctionContainer sci(empty_coeffs, empty_dets, orbitals);

  // Check that error messages are descriptive
  try {
    sci.get_total_num_electrons();
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "No determinants available");
  }
}

// Test entropy calculation with missing RDMs
TEST_F(SciWavefunctionTest, EntropyWithMissingRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {Configuration("2200")};
  Eigen::VectorXd coeffs(1);
  coeffs << 1.0;

  SciWavefunctionContainer sci(coeffs, dets, orbitals);

  // Should throw when trying to calculate entropies without RDMs
  EXPECT_THROW(sci.get_single_orbital_entropies(), std::runtime_error);
}

// Test SCI with inactive orbitals to verify total vs active distinction
TEST_F(SciWavefunctionTest, WithInactiveOrbitals) {
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

  SciWavefunctionContainer sci(coeffs, dets, orbitals, one_rdm, std::nullopt);

  // Test electron counting with inactive orbitals
  auto [total_alpha_elec, total_beta_elec] = sci.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sci.get_active_num_electrons();

  // Active space has 2 alpha + 2 beta = 4 electrons from determinants
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  // Total includes 2 doubly occupied inactive orbitals = 2 alpha + 2 beta
  // additional
  EXPECT_EQ(total_alpha_elec, 4);  // 2 active + 2 inactive
  EXPECT_EQ(total_beta_elec, 4);   // 2 active + 2 inactive

  // Test orbital occupations with inactive orbitals
  auto [alpha_total_occ, beta_total_occ] = sci.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sci.get_active_orbital_occupations();

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

// Test SCI with non-continuous active space indices
TEST_F(SciWavefunctionTest, WithNonContinuousActiveSpace) {
  // Create orbitals with non-continuous active space: 8 orbitals total
  // Inactive: {0, 1} (orbitals 0-1)
  // Active: {2, 4, 6, 7} (non-continuous indices!)
  // Virtual: {3, 5} (orbitals 3, 5 are virtual between active orbitals)
  auto base_orbitals = testing::create_test_orbitals(8, 8, true);
  std::vector<size_t> active_indices = {2, 4, 6, 7};  // Non-continuous!
  std::vector<size_t> inactive_indices = {0, 1};      // Doubly occupied
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);
  // Make RDM for occupation checks
  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  // Create determinants using only the active space (4 orbitals)
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);
  SciWavefunctionContainer sci(coeffs, dets, orbitals, one_rdm, std::nullopt);

  // Test electron counting with non-continuous active space
  auto [total_alpha_elec, total_beta_elec] = sci.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sci.get_active_num_electrons();

  // Active space should still have 2 alpha + 2 beta from determinants
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  // Total includes 2 doubly occupied inactive orbitals
  EXPECT_EQ(total_alpha_elec, 4);  // 2 active + 2 inactive
  EXPECT_EQ(total_beta_elec, 4);   // 2 active + 2 inactive

  // Test orbital occupations with non-continuous active space
  auto [alpha_total_occ, beta_total_occ] = sci.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sci.get_active_orbital_occupations();

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

  // Verify specific orbital types
  EXPECT_DOUBLE_EQ(alpha_total_occ(3), 0.0);  // Virtual orbital
  EXPECT_DOUBLE_EQ(alpha_total_occ(5), 0.0);  // Virtual orbital
  EXPECT_DOUBLE_EQ(alpha_total_occ(0), 1.0);  // Inactive orbital
  EXPECT_DOUBLE_EQ(alpha_total_occ(1), 1.0);  // Inactive orbital
}

// Test JSON serialization/deserialization
TEST_F(SciWavefunctionTest, JsonSerialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  SciWavefunctionContainer original(coeffs, dets, orbitals);

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Deserialize from JSON
  auto restored = std::unique_ptr<SciWavefunctionContainer>(
      dynamic_cast<SciWavefunctionContainer*>(
          WavefunctionContainer::from_json(j).release()));

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
TEST_F(SciWavefunctionTest, Hdf5Serialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  SciWavefunctionContainer original(coeffs, dets, orbitals);

  std::string filename = "test_sci_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    auto restored = SciWavefunctionContainer::from_hdf5(root);

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

// Test serialization with RDMs
TEST_F(SciWavefunctionTest, Hdf5SerializationRDMs) {
  // create H4 linear chain structure
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H", "H", "H", "H"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  // scf
  auto scf_solver = ScfSolverFactory::create();
  const char* basis_set = "sto-3g";

  auto [E_default, wfn_default] = scf_solver->run(structure, 0, 1, basis_set);

  // build hamiltonian
  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(wfn_default->get_orbitals());

  // run SCI with RDM calculation on (4,4) active space
  auto mc = MultiConfigurationCalculatorFactory::create("macis_asci");
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  mc->settings().set("ntdets_max", 1);
  mc->settings().set("max_refine_iter", 0);
  mc->settings().set("grow_factor", 2);

  auto [E_sci, wfn_sci] = mc->run(H, 4, 4);

  const auto& original = wfn_sci->get_container<SciWavefunctionContainer>();

  EXPECT_TRUE(original.has_one_rdm_spin_dependent());
  EXPECT_TRUE(original.has_one_rdm_spin_traced());
  EXPECT_TRUE(original.has_two_rdm_spin_dependent());
  EXPECT_TRUE(original.has_two_rdm_spin_traced());

  // save to hdf5
  std::string filename = "test_sci_rdm_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    auto restored = SciWavefunctionContainer::from_hdf5(root);

    // Verify rdms are still there
    EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_one_rdm_spin_traced());
    EXPECT_TRUE(restored->has_two_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_two_rdm_spin_traced());

    // Verify that they match
    auto [restored_aa_rdm, restored_bb_rdm] =
        restored->get_active_one_rdm_spin_dependent();
    auto [original_aa_rdm, original_bb_rdm] =
        original.get_active_one_rdm_spin_dependent();
    // extract data from variants
    const auto& restored_aa_rdm_r = std::get<Eigen::MatrixXd>(restored_aa_rdm);
    const auto& restored_bb_rdm_r = std::get<Eigen::MatrixXd>(restored_bb_rdm);
    const auto& original_aa_rdm_r = std::get<Eigen::MatrixXd>(original_aa_rdm);
    const auto& original_bb_rdm_r = std::get<Eigen::MatrixXd>(original_bb_rdm);

    EXPECT_TRUE(
        restored_aa_rdm_r.isApprox(original_aa_rdm_r, testing::rdm_tolerance));
    EXPECT_TRUE(
        restored_bb_rdm_r.isApprox(original_bb_rdm_r, testing::rdm_tolerance));

    auto restored_one_rdm = restored->get_active_one_rdm_spin_traced();
    auto original_one_rdm = original.get_active_one_rdm_spin_traced();
    // extract data from variants
    const auto& restored_one_rdm_r =
        std::get<Eigen::MatrixXd>(restored_one_rdm);
    const auto& original_one_rdm_r =
        std::get<Eigen::MatrixXd>(original_one_rdm);

    EXPECT_TRUE(restored_one_rdm_r.isApprox(original_one_rdm_r,
                                            testing::rdm_tolerance));

    auto [restored_aabb_rdm, restored_aaaa_rdm, restored_bbbb_rdm] =
        restored->get_active_two_rdm_spin_dependent();
    auto [original_aabb_rdm, original_aaaa_rdm, original_bbbb_rdm] =
        original.get_active_two_rdm_spin_dependent();
    // extract data from variants
    const auto& restored_aabb_rdm_r =
        std::get<Eigen::VectorXd>(restored_aabb_rdm);
    const auto& restored_aaaa_rdm_r =
        std::get<Eigen::VectorXd>(restored_aaaa_rdm);
    const auto& restored_bbbb_rdm_r =
        std::get<Eigen::VectorXd>(restored_bbbb_rdm);
    const auto& original_aabb_rdm_r =
        std::get<Eigen::VectorXd>(original_aabb_rdm);
    const auto& original_aaaa_rdm_r =
        std::get<Eigen::VectorXd>(original_aaaa_rdm);
    const auto& original_bbbb_rdm_r =
        std::get<Eigen::VectorXd>(original_bbbb_rdm);

    EXPECT_TRUE(restored_aabb_rdm_r.isApprox(original_aabb_rdm_r,
                                             testing::rdm_tolerance));
    EXPECT_TRUE(restored_aaaa_rdm_r.isApprox(original_aaaa_rdm_r,
                                             testing::rdm_tolerance));
    EXPECT_TRUE(restored_bbbb_rdm_r.isApprox(original_bbbb_rdm_r,
                                             testing::rdm_tolerance));

    auto restored_two_rdm = restored->get_active_two_rdm_spin_traced();
    auto original_two_rdm = original.get_active_two_rdm_spin_traced();
    // extract data from variants
    const auto& restored_two_rdm_r =
        std::get<Eigen::VectorXd>(restored_two_rdm);
    const auto& original_two_rdm_r =
        std::get<Eigen::VectorXd>(original_two_rdm);
    EXPECT_TRUE(restored_two_rdm_r.isApprox(original_two_rdm_r,
                                            testing::rdm_tolerance));

    file.close();
  }

  std::remove(filename.c_str());
}

// Test JSON serialization with RDMs
TEST_F(SciWavefunctionTest, JsonSerializationRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  // Create one-RDM
  Eigen::MatrixXd one_rdm_aa(4, 4);
  one_rdm_aa.setIdentity();
  one_rdm_aa *= 2.0;
  one_rdm_aa(2, 2) = 0.0;
  one_rdm_aa(3, 3) = 0.0;

  // Create two-RDM
  size_t two_rdm_size = 4 * 4 * 4 * 4;
  Eigen::VectorXd two_rdm_aabb(two_rdm_size);
  two_rdm_aabb.setOnes();
  two_rdm_aabb *= 0.5;

  Eigen::VectorXd two_rdm_aaaa(two_rdm_size);
  two_rdm_aaaa.setOnes();
  two_rdm_aaaa *= 0.25;

  SciWavefunctionContainer original(coeffs, dets, orbitals, std::nullopt,
                                    one_rdm_aa, one_rdm_aa, std::nullopt,
                                    two_rdm_aabb, two_rdm_aaaa, two_rdm_aaaa);

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Verify RDMs are in JSON
  EXPECT_TRUE(j.contains("rdms"));
  EXPECT_TRUE(j["rdms"].contains("one_rdm_aa"));
  EXPECT_TRUE(j["rdms"].contains("two_rdm_aabb"));
  EXPECT_TRUE(j["rdms"].contains("two_rdm_aaaa"));

  // Deserialize from JSON
  auto restored = std::unique_ptr<SciWavefunctionContainer>(
      dynamic_cast<SciWavefunctionContainer*>(
          WavefunctionContainer::from_json(j).release()));

  // Verify RDMs are available after deserialization
  EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
  EXPECT_TRUE(restored->has_two_rdm_spin_dependent());

  // Verify RDM values match
  auto [orig_one_aa, orig_one_bb] =
      original.get_active_one_rdm_spin_dependent();
  auto [rest_one_aa, rest_one_bb] =
      restored->get_active_one_rdm_spin_dependent();

  EXPECT_TRUE(std::get<Eigen::MatrixXd>(orig_one_aa)
                  .isApprox(std::get<Eigen::MatrixXd>(rest_one_aa),
                            testing::wf_tolerance));

  auto [orig_two_aabb, orig_two_aaaa, orig_two_bbbb] =
      original.get_active_two_rdm_spin_dependent();
  auto [rest_two_aabb, rest_two_aaaa, rest_two_bbbb] =
      restored->get_active_two_rdm_spin_dependent();

  EXPECT_TRUE(std::get<Eigen::VectorXd>(orig_two_aabb)
                  .isApprox(std::get<Eigen::VectorXd>(rest_two_aabb),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(orig_two_aaaa)
                  .isApprox(std::get<Eigen::VectorXd>(rest_two_aaaa),
                            testing::wf_tolerance));
}

// Test JSON serialization with RDMs for unrestricted system
TEST_F(SciWavefunctionTest, JsonSerializationRDMsUnrestricted) {
  // create Li atom structure
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}};
  std::vector<std::string> symbols = {"Li"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  // scf with multiplicity = 2 (doublet state for Li)
  auto scf_solver = ScfSolverFactory::create();
  const char* basis_set = "sto-3g";

  auto [E_default, wfn_default] = scf_solver->run(structure, 0, 2, basis_set);

  // build hamiltonian
  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(wfn_default->get_orbitals());

  // run SCI with RDM calculation
  auto mc = MultiConfigurationCalculatorFactory::create("macis_asci");
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  mc->settings().set("ntdets_max", 1);
  mc->settings().set("max_refine_iter", 0);
  mc->settings().set("grow_factor", 2);
  auto [E_sci, wfn_sci] = mc->run(H, 3, 5);  // 3 electrons in 5 orbitals

  const auto& original = wfn_sci->get_container<SciWavefunctionContainer>();

  EXPECT_TRUE(original.has_one_rdm_spin_dependent());
  EXPECT_TRUE(original.has_one_rdm_spin_traced());
  EXPECT_TRUE(original.has_two_rdm_spin_dependent());
  EXPECT_TRUE(original.has_two_rdm_spin_traced());

  // Verify it's unrestricted
  EXPECT_FALSE(original.get_orbitals()->is_restricted());

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Verify RDMs are in JSON
  EXPECT_TRUE(j.contains("rdms"));
  EXPECT_TRUE(j["rdms"].contains("one_rdm_aa"));
  EXPECT_TRUE(j["rdms"].contains("one_rdm_bb"));
  EXPECT_TRUE(j["rdms"].contains("two_rdm_aabb"));
  EXPECT_TRUE(j["rdms"].contains("two_rdm_aaaa"));
  EXPECT_TRUE(j["rdms"].contains("two_rdm_bbbb"));

  // Deserialize from JSON
  auto restored = std::unique_ptr<SciWavefunctionContainer>(
      dynamic_cast<SciWavefunctionContainer*>(
          WavefunctionContainer::from_json(j).release()));

  // Verify rdms are still there
  EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
  EXPECT_TRUE(restored->has_one_rdm_spin_traced());
  EXPECT_TRUE(restored->has_two_rdm_spin_dependent());
  EXPECT_TRUE(restored->has_two_rdm_spin_traced());

  // Verify it's still unrestricted
  EXPECT_FALSE(restored->get_orbitals()->is_restricted());

  // Verify that alpha and beta RDMs match
  auto [restored_aa_rdm, restored_bb_rdm] =
      restored->get_active_one_rdm_spin_dependent();
  auto [original_aa_rdm, original_bb_rdm] =
      original.get_active_one_rdm_spin_dependent();

  // extract data from variants
  const auto& restored_aa_rdm_r = std::get<Eigen::MatrixXd>(restored_aa_rdm);
  const auto& restored_bb_rdm_r = std::get<Eigen::MatrixXd>(restored_bb_rdm);
  const auto& original_aa_rdm_r = std::get<Eigen::MatrixXd>(original_aa_rdm);
  const auto& original_bb_rdm_r = std::get<Eigen::MatrixXd>(original_bb_rdm);

  EXPECT_TRUE(
      restored_aa_rdm_r.isApprox(original_aa_rdm_r, testing::rdm_tolerance));
  EXPECT_TRUE(
      restored_bb_rdm_r.isApprox(original_bb_rdm_r, testing::rdm_tolerance));

  // alpha and beta RDMs should be different for Li (unpaired electron)
  EXPECT_FALSE(
      restored_aa_rdm_r.isApprox(restored_bb_rdm_r, testing::rdm_tolerance));

  auto restored_one_rdm = restored->get_active_one_rdm_spin_traced();
  auto original_one_rdm = original.get_active_one_rdm_spin_traced();

  // extract data from variants
  const auto& restored_one_rdm_r = std::get<Eigen::MatrixXd>(restored_one_rdm);
  const auto& original_one_rdm_r = std::get<Eigen::MatrixXd>(original_one_rdm);

  EXPECT_TRUE(
      restored_one_rdm_r.isApprox(original_one_rdm_r, testing::rdm_tolerance));

  auto [restored_aabb_rdm, restored_aaaa_rdm, restored_bbbb_rdm] =
      restored->get_active_two_rdm_spin_dependent();
  auto [original_aabb_rdm, original_aaaa_rdm, original_bbbb_rdm] =
      original.get_active_two_rdm_spin_dependent();

  // extract data from variants
  const auto& restored_aabb_rdm_r =
      std::get<Eigen::VectorXd>(restored_aabb_rdm);
  const auto& restored_aaaa_rdm_r =
      std::get<Eigen::VectorXd>(restored_aaaa_rdm);
  const auto& restored_bbbb_rdm_r =
      std::get<Eigen::VectorXd>(restored_bbbb_rdm);
  const auto& original_aabb_rdm_r =
      std::get<Eigen::VectorXd>(original_aabb_rdm);
  const auto& original_aaaa_rdm_r =
      std::get<Eigen::VectorXd>(original_aaaa_rdm);
  const auto& original_bbbb_rdm_r =
      std::get<Eigen::VectorXd>(original_bbbb_rdm);

  EXPECT_TRUE(restored_aabb_rdm_r.isApprox(original_aabb_rdm_r,
                                           testing::rdm_tolerance));
  EXPECT_TRUE(restored_aaaa_rdm_r.isApprox(original_aaaa_rdm_r,
                                           testing::rdm_tolerance));
  EXPECT_TRUE(restored_bbbb_rdm_r.isApprox(original_bbbb_rdm_r,
                                           testing::rdm_tolerance));

  // aaaa and bbbb 2-RDMs should be different
  EXPECT_FALSE(restored_aaaa_rdm_r.isApprox(restored_bbbb_rdm_r,
                                            testing::rdm_tolerance));

  auto restored_two_rdm = restored->get_active_two_rdm_spin_traced();
  auto original_two_rdm = original.get_active_two_rdm_spin_traced();

  // extract data from variants
  const auto& restored_two_rdm_r = std::get<Eigen::VectorXd>(restored_two_rdm);
  const auto& original_two_rdm_r = std::get<Eigen::VectorXd>(original_two_rdm);
  EXPECT_TRUE(
      restored_two_rdm_r.isApprox(original_two_rdm_r, testing::rdm_tolerance));
}

// Test serialization with RDMs for unrestricted system
TEST_F(SciWavefunctionTest, Hdf5SerializationRDMsUnrestricted) {
  // create Li atom structure
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}};
  std::vector<std::string> symbols = {"Li"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  // scf with multiplicity = 2 (doublet state for Li)
  auto scf_solver = ScfSolverFactory::create();
  const char* basis_set = "sto-3g";

  auto [E_default, wfn_default] = scf_solver->run(structure, 0, 2, basis_set);

  // build hamiltonian
  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(wfn_default->get_orbitals());

  // run SCI with RDM calculation
  auto mc = MultiConfigurationCalculatorFactory::create("macis_asci");
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  mc->settings().set("ntdets_max", 1);
  mc->settings().set("max_refine_iter", 0);
  mc->settings().set("grow_factor", 2);
  auto [E_sci, wfn_sci] = mc->run(H, 3, 5);  // 3 electrons in 5 orbitals

  const auto& original = wfn_sci->get_container<SciWavefunctionContainer>();

  EXPECT_TRUE(original.has_one_rdm_spin_dependent());
  EXPECT_TRUE(original.has_one_rdm_spin_traced());
  EXPECT_TRUE(original.has_two_rdm_spin_dependent());
  EXPECT_TRUE(original.has_two_rdm_spin_traced());

  // Verify it's unrestricted
  EXPECT_FALSE(original.get_orbitals()->is_restricted());

  // save to hdf5
  std::string filename = "test_sci_rdm_unrestricted_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    auto restored = SciWavefunctionContainer::from_hdf5(root);

    // Verify rdms are still there
    EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_one_rdm_spin_traced());
    EXPECT_TRUE(restored->has_two_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_two_rdm_spin_traced());

    // Verify it's still unrestricted
    EXPECT_FALSE(restored->get_orbitals()->is_restricted());

    // Verify that alpha and beta RDMs match
    auto [restored_aa_rdm, restored_bb_rdm] =
        restored->get_active_one_rdm_spin_dependent();
    auto [original_aa_rdm, original_bb_rdm] =
        original.get_active_one_rdm_spin_dependent();

    // extract data from variants
    const auto& restored_aa_rdm_r = std::get<Eigen::MatrixXd>(restored_aa_rdm);
    const auto& restored_bb_rdm_r = std::get<Eigen::MatrixXd>(restored_bb_rdm);
    const auto& original_aa_rdm_r = std::get<Eigen::MatrixXd>(original_aa_rdm);
    const auto& original_bb_rdm_r = std::get<Eigen::MatrixXd>(original_bb_rdm);

    EXPECT_TRUE(
        restored_aa_rdm_r.isApprox(original_aa_rdm_r, testing::rdm_tolerance));
    EXPECT_TRUE(
        restored_bb_rdm_r.isApprox(original_bb_rdm_r, testing::rdm_tolerance));

    // alpha and beta RDMs should be different for Li (unpaired electron)
    EXPECT_FALSE(
        restored_aa_rdm_r.isApprox(restored_bb_rdm_r, testing::rdm_tolerance));

    auto restored_one_rdm = restored->get_active_one_rdm_spin_traced();
    auto original_one_rdm = original.get_active_one_rdm_spin_traced();

    // extract data from variants
    const auto& restored_one_rdm_r =
        std::get<Eigen::MatrixXd>(restored_one_rdm);
    const auto& original_one_rdm_r =
        std::get<Eigen::MatrixXd>(original_one_rdm);

    EXPECT_TRUE(restored_one_rdm_r.isApprox(original_one_rdm_r,
                                            testing::rdm_tolerance));

    auto [restored_aabb_rdm, restored_aaaa_rdm, restored_bbbb_rdm] =
        restored->get_active_two_rdm_spin_dependent();
    auto [original_aabb_rdm, original_aaaa_rdm, original_bbbb_rdm] =
        original.get_active_two_rdm_spin_dependent();

    // extract data from variants
    const auto& restored_aabb_rdm_r =
        std::get<Eigen::VectorXd>(restored_aabb_rdm);
    const auto& restored_aaaa_rdm_r =
        std::get<Eigen::VectorXd>(restored_aaaa_rdm);
    const auto& restored_bbbb_rdm_r =
        std::get<Eigen::VectorXd>(restored_bbbb_rdm);
    const auto& original_aabb_rdm_r =
        std::get<Eigen::VectorXd>(original_aabb_rdm);
    const auto& original_aaaa_rdm_r =
        std::get<Eigen::VectorXd>(original_aaaa_rdm);
    const auto& original_bbbb_rdm_r =
        std::get<Eigen::VectorXd>(original_bbbb_rdm);

    EXPECT_TRUE(restored_aabb_rdm_r.isApprox(original_aabb_rdm_r,
                                             testing::rdm_tolerance));
    EXPECT_TRUE(restored_aaaa_rdm_r.isApprox(original_aaaa_rdm_r,
                                             testing::rdm_tolerance));
    EXPECT_TRUE(restored_bbbb_rdm_r.isApprox(original_bbbb_rdm_r,
                                             testing::rdm_tolerance));

    // aaaa and bbbb 2-RDMs should be different
    EXPECT_FALSE(restored_aaaa_rdm_r.isApprox(restored_bbbb_rdm_r,
                                              testing::rdm_tolerance));

    auto restored_two_rdm = restored->get_active_two_rdm_spin_traced();
    auto original_two_rdm = original.get_active_two_rdm_spin_traced();

    // extract data from variants
    const auto& restored_two_rdm_r =
        std::get<Eigen::VectorXd>(restored_two_rdm);
    const auto& original_two_rdm_r =
        std::get<Eigen::VectorXd>(original_two_rdm);
    EXPECT_TRUE(restored_two_rdm_r.isApprox(original_two_rdm_r,
                                            testing::rdm_tolerance));

    file.close();
  }

  std::remove(filename.c_str());
}
