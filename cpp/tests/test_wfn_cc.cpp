/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cc.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class CoupledClusterContainerTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}

  // Helper function to create a simple Wavefunction from a Configuration
  std::shared_ptr<Wavefunction> create_test_wavefunction(
      const Configuration& ref, std::shared_ptr<Orbitals> orbitals) {
    auto sd_container =
        std::make_unique<SlaterDeterminantContainer>(ref, orbitals);
    return std::make_shared<Wavefunction>(std::move(sd_container));
  }
};

TEST_F(CoupledClusterContainerTest, BasicProperties) {
  size_t nocc = 2;
  size_t nvirt = 2;
  size_t nmo = nocc + nvirt;

  auto orbitals = testing::create_test_orbitals(nocc + nvirt, 4, true);
  Configuration ref("2200");
  auto wavefunction = create_test_wavefunction(ref, orbitals);

  Eigen::VectorXd t1_amplitudes = Eigen::VectorXd::Random(nocc * nvirt);
  Eigen::VectorXd t2_amplitudes =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);

  CoupledClusterContainer cc(orbitals, wavefunction, t1_amplitudes,
                             t2_amplitudes);

  // check amplitudes
  auto [t1_alpha, t1_beta] = cc.get_t1_amplitudes();
  auto [t2_abab, t2_aaaa, t2_bbbb] = cc.get_t2_amplitudes();

  EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_alpha).isApprox(
      t1_amplitudes, testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_beta).isApprox(
      t1_amplitudes, testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_abab).isApprox(
      t2_amplitudes, testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_aaaa).isApprox(
      t2_amplitudes, testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_bbbb).isApprox(
      t2_amplitudes, testing::wf_tolerance));
}

// Test it throws when amplitude sizes are wrong
TEST_F(CoupledClusterContainerTest, InvalidAmplitudeSizesThrow) {
  size_t nocc = 2;
  size_t nvirt = 2;
  size_t nmo = nocc + nvirt;

  auto orbitals = testing::create_test_orbitals(nocc + nvirt, 4, true);
  Configuration ref("2200");
  auto wavefunction = create_test_wavefunction(ref, orbitals);

  // Correct sizes: T1 = nocc * nvirt = 4, T2 = nocc * nocc * nvirt * nvirt = 16

  // Test: T1 amplitude with wrong size (too small)
  Eigen::VectorXd t1_wrong_size = Eigen::VectorXd::Random(2);  // Should be 4
  Eigen::VectorXd t2_correct = Eigen::VectorXd::Random(16);
  EXPECT_THROW(CoupledClusterContainer(orbitals, wavefunction, t1_wrong_size,
                                       t2_correct),
               std::invalid_argument);

  // Test: T1 amplitude with wrong size (too large)
  Eigen::VectorXd t1_too_large = Eigen::VectorXd::Random(10);  // Should be 4
  EXPECT_THROW(
      CoupledClusterContainer(orbitals, wavefunction, t1_too_large, t2_correct),
      std::invalid_argument);

  // Test: T2 amplitude with wrong size (too small)
  Eigen::VectorXd t1_correct = Eigen::VectorXd::Random(4);
  Eigen::VectorXd t2_wrong_size = Eigen::VectorXd::Random(10);  // Should be 16
  EXPECT_THROW(CoupledClusterContainer(orbitals, wavefunction, t1_correct,
                                       t2_wrong_size),
               std::invalid_argument);

  // Test: T2 amplitude with wrong size (too large)
  Eigen::VectorXd t2_too_large = Eigen::VectorXd::Random(20);  // Should be 16
  EXPECT_THROW(
      CoupledClusterContainer(orbitals, wavefunction, t1_correct, t2_too_large),
      std::invalid_argument);

  // Test: Both amplitudes with wrong sizes
  EXPECT_THROW(CoupledClusterContainer(orbitals, wavefunction, t1_wrong_size,
                                       t2_wrong_size),
               std::invalid_argument);

  // Test: Correct sizes should not throw
  EXPECT_NO_THROW(
      CoupledClusterContainer(orbitals, wavefunction, t1_correct, t2_correct));
}

// Test JSON serialization/deserialization
TEST_F(CoupledClusterContainerTest, JsonSerializationSpatial) {
  size_t nocc = 2;
  size_t nvirt = 2;
  size_t nmo = nocc + nvirt;

  auto orbitals = testing::create_test_orbitals(nocc + nvirt, 4, true);
  Configuration ref("2200");
  auto wavefunction = create_test_wavefunction(ref, orbitals);

  Eigen::VectorXd t1_amplitudes = Eigen::VectorXd::Random(nocc * nvirt);
  Eigen::VectorXd t2_amplitudes =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);

  CoupledClusterContainer original(orbitals, wavefunction, t1_amplitudes,
                                   t2_amplitudes);

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Deserialize from JSON using container-specific method
  auto restored = CoupledClusterContainer::from_json(j);

  // Also test base Wavefunction::from_json() by wrapping container in
  // Wavefunction
  auto original_wf =
      std::make_shared<Wavefunction>(std::make_unique<CoupledClusterContainer>(
          orbitals, wavefunction, t1_amplitudes, t2_amplitudes));
  nlohmann::json wf_j = original_wf->to_json();
  auto wf_restored = Wavefunction::from_json(wf_j);
  EXPECT_EQ(wf_restored->get_container_type(), "coupled_cluster");
  auto& wf_restored_container =
      wf_restored->get_container<CoupledClusterContainer>();

  // check amplitudes
  auto [t1_alpha_orig, t1_beta_orig] = original.get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original.get_t2_amplitudes();
  auto [t1_alpha_rest, t1_beta_rest] = restored->get_t1_amplitudes();
  auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
      restored->get_t2_amplitudes();
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_alpha_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t1_alpha_rest),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_beta_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t1_beta_rest),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_abab_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t2_abab_rest),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_aaaa_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t2_aaaa_rest),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_bbbb_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t2_bbbb_rest),
                            testing::wf_tolerance));

  // Verify that base Wavefunction::from_json gives the same result
  auto [t1_alpha_wf, t1_beta_wf] = wf_restored_container.get_t1_amplitudes();
  auto [t2_abab_wf, t2_aaaa_wf, t2_bbbb_wf] =
      wf_restored_container.get_t2_amplitudes();
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_alpha_rest)
                  .isApprox(std::get<Eigen::VectorXd>(t1_alpha_wf),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_beta_rest)
                  .isApprox(std::get<Eigen::VectorXd>(t1_beta_wf),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_abab_rest)
                  .isApprox(std::get<Eigen::VectorXd>(t2_abab_wf),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_aaaa_rest)
                  .isApprox(std::get<Eigen::VectorXd>(t2_aaaa_wf),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_bbbb_rest)
                  .isApprox(std::get<Eigen::VectorXd>(t2_bbbb_wf),
                            testing::wf_tolerance));
}

TEST_F(CoupledClusterContainerTest, JsonSerializationSpin) {
  size_t nocc = 2;
  size_t nvirt = 2;
  size_t nmo = nocc + nvirt;

  auto orbitals = testing::create_test_orbitals(nocc + nvirt, 4, true);
  Configuration ref("2200");
  auto wavefunction = create_test_wavefunction(ref, orbitals);

  Eigen::VectorXd t1_aa = Eigen::VectorXd::Random(nocc * nvirt);
  Eigen::VectorXd t1_bb = Eigen::VectorXd::Random(nocc * nvirt);
  Eigen::VectorXd t2_abab =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);
  Eigen::VectorXd t2_aaaa =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);
  Eigen::VectorXd t2_bbbb =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);

  CoupledClusterContainer original(orbitals, wavefunction, t1_aa, t1_bb,
                                   t2_abab, t2_aaaa, t2_bbbb);

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Deserialize from JSON
  auto restored = CoupledClusterContainer::from_json(j);

  // check amplitudes
  auto [t1_alpha_orig, t1_beta_orig] = original.get_t1_amplitudes();
  auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
      original.get_t2_amplitudes();
  auto [t1_alpha_rest, t1_beta_rest] = restored->get_t1_amplitudes();
  auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
      restored->get_t2_amplitudes();
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_alpha_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t1_alpha_rest),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_beta_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t1_beta_rest),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_abab_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t2_abab_rest),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_aaaa_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t2_aaaa_rest),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_bbbb_orig)
                  .isApprox(std::get<Eigen::VectorXd>(t2_bbbb_rest),
                            testing::wf_tolerance));
}

// Test HDF5 serialization/deserialization
TEST_F(CoupledClusterContainerTest, Hdf5SerializationSpatial) {
  size_t nocc = 2;
  size_t nvirt = 2;
  size_t nmo = nocc + nvirt;

  auto orbitals = testing::create_test_orbitals(nocc + nvirt, 4, true);
  Configuration ref("2200");
  auto wavefunction = create_test_wavefunction(ref, orbitals);

  Eigen::VectorXd t1_amplitudes = Eigen::VectorXd::Random(nocc * nvirt);
  Eigen::VectorXd t2_amplitudes =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);

  CoupledClusterContainer original(orbitals, wavefunction, t1_amplitudes,
                                   t2_amplitudes);

  std::string filename = "test_cc_spatial_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5 using container-specific method
    auto restored = CoupledClusterContainer::from_hdf5(root);

    // check amplitudes
    auto [t1_alpha_orig, t1_beta_orig] = original.get_t1_amplitudes();
    auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
        original.get_t2_amplitudes();
    auto [t1_alpha_rest, t1_beta_rest] = restored->get_t1_amplitudes();
    auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
        restored->get_t2_amplitudes();
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_alpha_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t1_alpha_rest),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_beta_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t1_beta_rest),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_abab_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t2_abab_rest),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_aaaa_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t2_aaaa_rest),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_bbbb_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t2_bbbb_rest),
                              testing::wf_tolerance));

    file.close();
  }

  // Also test base Wavefunction::from_hdf5() by creating a separate file with
  // Wavefunction wrapper
  std::string wf_filename = "test_cc_wavefunction_serialization.h5";
  {
    // Create and serialize a Wavefunction wrapping the container
    auto original_wf = std::make_shared<Wavefunction>(
        std::make_unique<CoupledClusterContainer>(
            orbitals, wavefunction, t1_amplitudes, t2_amplitudes));
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
    EXPECT_EQ(wf_restored->get_container_type(), "coupled_cluster");
    auto& wf_restored_container =
        wf_restored->get_container<CoupledClusterContainer>();

    // Get the restored amplitudes from container-specific method for comparison
    H5::H5File file2(filename, H5F_ACC_RDONLY);
    H5::Group root2 = file2.openGroup("/");
    auto restored = CoupledClusterContainer::from_hdf5(root2);

    auto [t1_alpha_rest, t1_beta_rest] = restored->get_t1_amplitudes();
    auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
        restored->get_t2_amplitudes();
    auto [t1_alpha_wf, t1_beta_wf] = wf_restored_container.get_t1_amplitudes();
    auto [t2_abab_wf, t2_aaaa_wf, t2_bbbb_wf] =
        wf_restored_container.get_t2_amplitudes();

    EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_alpha_rest)
                    .isApprox(std::get<Eigen::VectorXd>(t1_alpha_wf),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_beta_rest)
                    .isApprox(std::get<Eigen::VectorXd>(t1_beta_wf),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_abab_rest)
                    .isApprox(std::get<Eigen::VectorXd>(t2_abab_wf),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_aaaa_rest)
                    .isApprox(std::get<Eigen::VectorXd>(t2_aaaa_wf),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_bbbb_rest)
                    .isApprox(std::get<Eigen::VectorXd>(t2_bbbb_wf),
                              testing::wf_tolerance));

    file.close();
    file2.close();
  }

  std::remove(filename.c_str());
  std::remove(wf_filename.c_str());
}

// Test HDF5 serialization/deserialization
TEST_F(CoupledClusterContainerTest, Hdf5SerializationSpin) {
  size_t nocc = 2;
  size_t nvirt = 2;
  size_t nmo = nocc + nvirt;

  auto orbitals = testing::create_test_orbitals(nocc + nvirt, 4, true);
  Configuration ref("2200");
  auto wavefunction = create_test_wavefunction(ref, orbitals);

  Eigen::VectorXd t1_aa = Eigen::VectorXd::Random(nocc * nvirt);
  Eigen::VectorXd t1_bb = Eigen::VectorXd::Random(nocc * nvirt);
  Eigen::VectorXd t2_abab =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);
  Eigen::VectorXd t2_aaaa =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);
  Eigen::VectorXd t2_bbbb =
      Eigen::VectorXd::Random(nocc * nocc * nvirt * nvirt);

  CoupledClusterContainer original(orbitals, wavefunction, t1_aa, t1_bb,
                                   t2_abab, t2_aaaa, t2_bbbb);

  std::string filename = "test_cc_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    auto restored = CoupledClusterContainer::from_hdf5(root);

    // check amplitudes
    auto [t1_alpha_orig, t1_beta_orig] = original.get_t1_amplitudes();
    auto [t2_abab_orig, t2_aaaa_orig, t2_bbbb_orig] =
        original.get_t2_amplitudes();
    auto [t1_alpha_rest, t1_beta_rest] = restored->get_t1_amplitudes();
    auto [t2_abab_rest, t2_aaaa_rest, t2_bbbb_rest] =
        restored->get_t2_amplitudes();
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_alpha_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t1_alpha_rest),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t1_beta_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t1_beta_rest),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_abab_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t2_abab_rest),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_aaaa_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t2_aaaa_rest),
                              testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(t2_bbbb_orig)
                    .isApprox(std::get<Eigen::VectorXd>(t2_bbbb_rest),
                              testing::wf_tolerance));

    file.close();
  }

  std::remove(filename.c_str());
}
