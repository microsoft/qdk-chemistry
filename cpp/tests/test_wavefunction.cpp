// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <complex>
#include <cstdio>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <stdexcept>
#include <tuple>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

// ===== Tests for Core Wavefunction Functionality =====

class WavefunctionCoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up a wavefunction with some basic properties
    size = 3;

    // Create orbitals with enough MOs for configurations (need 6 orbitals)
    orbitals = testing::create_test_orbitals(6, 6, true);

    // Create vectors for coefficients and determinants
    Eigen::VectorXcd coeffs(size);
    Wavefunction::DeterminantVector dets(size);

    // Initialize with some values
    coeffs(0) = std::complex<double>(1.0, 0.0);
    coeffs(1) = std::complex<double>(0.0, 1.0);
    coeffs(2) = std::complex<double>(0.5, 0.5);

    // Create some test configurations using string constructor (6 orbitals
    // each)
    Configuration config1("ud2000");
    Configuration config2("u2d000");
    Configuration config3("220000");

    dets[0] = config1;
    dets[1] = config2;
    dets[2] = config3;

    // Create a test wavefunction
    wf = std::make_unique<Wavefunction>(
        std::make_unique<CasWavefunctionContainer>(
            coeffs, dets, orbitals, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt));
  }

  std::unique_ptr<Wavefunction> wf;
  std::shared_ptr<Orbitals> orbitals;
  int size;
};

// Test the size method
TEST_F(WavefunctionCoreTest, Size) {
  EXPECT_EQ(wf->size(), size);

  // Test empty wavefunction created with empty vectors
  Eigen::VectorXcd empty_coeffs(0);
  Wavefunction::DeterminantVector empty_dets(0);
  Wavefunction empty_wf(std::make_unique<CasWavefunctionContainer>(
      empty_coeffs, empty_dets, testing::create_test_orbitals(), std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt));
  EXPECT_EQ(empty_wf.size(), 0);

  // Test wavefunction with explicit size
  Eigen::VectorXcd sized_coeffs(5);
  sized_coeffs.setZero();
  Wavefunction::DeterminantVector sized_dets(5);
  for (int i = 0; i < 5; ++i) {
    sized_dets[i] = Configuration("u0");
  }
  Wavefunction sized_wf(std::make_unique<CasWavefunctionContainer>(
      sized_coeffs, sized_dets, testing::create_test_orbitals(), std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt));
  EXPECT_EQ(sized_wf.size(), 5);
}

// Test get coefficients (immutable interface)
TEST_F(WavefunctionCoreTest, CoefficientsGet) {
  // Get initial coefficients
  auto original_coeffs =
      wf->get_container<CasWavefunctionContainer>().get_coefficients();
  EXPECT_EQ(std::get<Eigen::VectorXcd>(original_coeffs).size(), size);
  EXPECT_EQ(std::get<Eigen::VectorXcd>(original_coeffs)(0),
            std::complex<double>(1.0, 0.0));
  EXPECT_EQ(std::get<Eigen::VectorXcd>(original_coeffs)(1),
            std::complex<double>(0.0, 1.0));
  EXPECT_EQ(std::get<Eigen::VectorXcd>(original_coeffs)(2),
            std::complex<double>(0.5, 0.5));

  // Create new wavefunction with different coefficients to verify immutability
  Eigen::VectorXcd new_coeffs(size);
  new_coeffs(0) = std::complex<double>(0.2, 0.3);
  new_coeffs(1) = std::complex<double>(0.4, 0.5);
  new_coeffs(2) = std::complex<double>(0.6, 0.7);

  auto original_dets = wf->get_active_determinants();
  Wavefunction new_wf(std::make_unique<CasWavefunctionContainer>(
      new_coeffs, original_dets, orbitals, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt));

  // Verify new wavefunction has different coefficients
  auto updated_coeffs =
      new_wf.get_container<CasWavefunctionContainer>().get_coefficients();
  EXPECT_EQ(std::get<Eigen::VectorXcd>(updated_coeffs).size(), size);
  EXPECT_EQ(std::get<Eigen::VectorXcd>(updated_coeffs)(0),
            std::complex<double>(0.2, 0.3));
  EXPECT_EQ(std::get<Eigen::VectorXcd>(updated_coeffs)(1),
            std::complex<double>(0.4, 0.5));
  EXPECT_EQ(std::get<Eigen::VectorXcd>(updated_coeffs)(2),
            std::complex<double>(0.6, 0.7));

  // Original wavefunction should be unchanged
  EXPECT_EQ(std::get<Eigen::VectorXcd>(original_coeffs)(0),
            std::complex<double>(1.0, 0.0));
  EXPECT_EQ(std::get<Eigen::VectorXcd>(original_coeffs)(1),
            std::complex<double>(0.0, 1.0));
  EXPECT_EQ(std::get<Eigen::VectorXcd>(original_coeffs)(2),
            std::complex<double>(0.5, 0.5));

  auto coeffs_from_method = new_wf.get_coefficients();
  EXPECT_EQ(std::get<Eigen::VectorXcd>(coeffs_from_method).size(), size);
  EXPECT_EQ(std::get<Eigen::VectorXcd>(coeffs_from_method)(0),
            std::complex<double>(0.2, 0.3));
  EXPECT_EQ(std::get<Eigen::VectorXcd>(coeffs_from_method)(1),
            std::complex<double>(0.4, 0.5));
  EXPECT_EQ(std::get<Eigen::VectorXcd>(coeffs_from_method)(2),
            std::complex<double>(0.6, 0.7));
}

// Test get determinants (immutable interface)
TEST_F(WavefunctionCoreTest, DeterminantsGet) {
  // Get initial determinants
  auto original_dets = wf->get_active_determinants();
  EXPECT_EQ(
      original_dets.size(),
      size);  // Create new wavefunction with different determinants to verify
  // immutability
  Wavefunction::DeterminantVector new_dets(size);

  Configuration config1("ud0000");
  Configuration config2("du0000");
  Configuration config3("020000");

  new_dets[0] = config1;
  new_dets[1] = config2;
  new_dets[2] = config3;

  auto original_coeffs =
      wf->get_container<CasWavefunctionContainer>().get_coefficients();
  Wavefunction new_wf(std::make_unique<CasWavefunctionContainer>(
      original_coeffs, new_dets, orbitals, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt));

  // Verify new wavefunction has different determinants
  auto updated_dets = new_wf.get_active_determinants();
  EXPECT_EQ(updated_dets.size(), size);

  // Check configurations match by using operator== instead of checking
  // individual bits
  EXPECT_TRUE(updated_dets[0] == config1);
  EXPECT_TRUE(updated_dets[1] == config2);
  EXPECT_TRUE(updated_dets[2] == config3);
}

// Test wavefunction overlap calculation
TEST_F(WavefunctionCoreTest, OverlapCalculation) {
  // Create coefficients for overlap test
  Eigen::VectorXcd coeffs1(size);
  coeffs1(0) = std::complex<double>(1.0, 0.0);
  coeffs1(1) = std::complex<double>(0.0, 1.0);
  coeffs1(2) = std::complex<double>(0.0, 0.0);

  Eigen::VectorXcd coeffs2(size);
  coeffs2(0) = std::complex<double>(0.0, 1.0);
  coeffs2(1) = std::complex<double>(1.0, 0.0);
  coeffs2(2) = std::complex<double>(0.0, 0.0);

  // Create first wavefunction
  auto original_dets = wf->get_active_determinants();
  Wavefunction wf1(std::make_unique<CasWavefunctionContainer>(
      coeffs1, original_dets, orbitals));

  // Create second wavefunction using same determinants
  Wavefunction wf2(std::make_unique<CasWavefunctionContainer>(
      coeffs2, original_dets, orbitals));

  // Calculate overlap: conj(coeffs1) dot coeffs2
  // conj([1+0i, 0+i, 0]) dot [0+i, 1+0i, 0] =
  // [1-0i, 0-i, 0] dot [0+i, 1+0i, 0] = 0 + 0 + 0 = 0
  auto overlap = std::get<std::complex<double>>(wf1.overlap(wf2));
  EXPECT_NEAR(overlap.real(), 0.0, testing::wf_tolerance);

  // Test self-overlap (should be norm squared)
  auto self_overlap = std::get<std::complex<double>>(wf1.overlap(wf1));
  double expected_norm_squared = wf1.norm() * wf1.norm();
  EXPECT_NEAR(self_overlap.real(), expected_norm_squared,
              testing::wf_tolerance);
}

// Test new explicit electron counting and orbital occupation functions
TEST_F(WavefunctionCoreTest, ExplicitElectronAndOccupationFunctions) {
  // Create a CAS wavefunction for testing
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {Configuration("2200")};
  Eigen::VectorXcd coeffs(3);
  coeffs << std::complex<double>(0.5, 0.0), std::complex<double>(0.5, 0.0),
      std::complex<double>(1.0 / sqrt(2), 0.0);

  // one rdm for occupation
  Eigen::MatrixXd one_rdm = Eigen::MatrixXd::Zero(4, 4);
  one_rdm(0, 0) = 2.0;
  one_rdm(1, 1) = 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;
  Wavefunction wf(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, one_rdm, std::nullopt));

  // Test electron counting functions (no inactive orbitals)
  auto [total_alpha_elec, total_beta_elec] = wf.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = wf.get_active_num_electrons();
  EXPECT_EQ(total_alpha_elec, 2);
  EXPECT_EQ(total_beta_elec, 2);
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);
  // With no inactive orbitals, total should equal active
  EXPECT_EQ(total_alpha_elec, active_alpha_elec);
  EXPECT_EQ(total_beta_elec, active_beta_elec);

  // Test orbital occupation functions
  auto [alpha_total_occ, beta_total_occ] = wf.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      wf.get_active_orbital_occupations();
  EXPECT_EQ(alpha_total_occ.size(), 4);
  EXPECT_EQ(beta_total_occ.size(), 4);
  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);

  // Expected occupations from the determinant pattern
  Eigen::VectorXd expected_alpha(4);
  expected_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_beta(4);
  expected_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(alpha_total_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(expected_beta, testing::wf_tolerance));
  EXPECT_TRUE(alpha_active_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_active_occ.isApprox(expected_beta, testing::wf_tolerance));
  // With no inactive orbitals, total should equal active
  EXPECT_TRUE(
      alpha_total_occ.isApprox(alpha_active_occ, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(beta_active_occ, testing::wf_tolerance));
}

// Test new functions with inactive orbitals
TEST_F(WavefunctionCoreTest, ExplicitFunctionsWithInactiveOrbitals) {
  // Create orbitals with inactive space: 6 orbitals total, 2 inactive, 4
  // active
  auto base_orbitals = testing::create_test_orbitals(6, 6, true);
  std::vector<size_t> active_indices = {2, 3, 4, 5};  // orbitals 2-5 are active
  std::vector<size_t> inactive_indices = {
      0, 1};  // orbitals 0-1 are inactive (doubly occupied)
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  // Create determinants using only the active space (4 orbitals)
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXcd coeffs(3);
  coeffs << std::complex<double>(0.5, 0.0), std::complex<double>(0.5, 0.0),
      std::complex<double>(1.0 / sqrt(2), 0.0);

  // Dummy one rdm for occupation
  Eigen::MatrixXd one_rdm = Eigen::MatrixXd::Zero(4, 4);
  one_rdm(0, 0) = 2.0;
  one_rdm(1, 1) = 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  Wavefunction wf(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, one_rdm, std::nullopt));

  // Test electron counting with inactive orbitals
  auto [total_alpha_elec, total_beta_elec] = wf.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = wf.get_active_num_electrons();

  // Active space has 2 alpha + 2 beta = 4 electrons from determinants
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  // Total includes 2 doubly occupied inactive orbitals = 2 alpha + 2 beta
  // additional
  EXPECT_EQ(total_alpha_elec, 4);  // 2 active + 2 inactive
  EXPECT_EQ(total_beta_elec, 4);   // 2 active + 2 inactive

  // Test orbital occupations with inactive orbitals
  auto [alpha_total_occ, beta_total_occ] = wf.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      wf.get_active_orbital_occupations();

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

// Test wavefunction serialization
class WavefunctionSerializationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    orbitals = testing::create_test_orbitals();

    // Create test coefficients and determinants
    Eigen::VectorXd coeffs_real(3);
    coeffs_real << 0.8, 0.5, 0.3;

    Eigen::VectorXcd coeffs_complex(3);
    coeffs_complex << std::complex<double>(0.8, 0.1),
        std::complex<double>(0.5, -0.2), std::complex<double>(0.3, 0.4);

    // Configurations must match active space size (2 orbitals by default)
    Wavefunction::DeterminantVector dets = {
        Configuration("20"), Configuration("ud"), Configuration("02")};

    // Create test wavefunctions
    cas_real = std::make_shared<Wavefunction>(
        std::make_unique<CasWavefunctionContainer>(coeffs_real, dets,
                                                   orbitals));

    cas_complex = std::make_shared<Wavefunction>(
        std::make_unique<CasWavefunctionContainer>(coeffs_complex, dets,
                                                   orbitals));

    sd_wavefunction = std::make_shared<Wavefunction>(
        std::make_unique<SlaterDeterminantContainer>(Configuration("20"),
                                                     orbitals));

    // Create SCI wavefunctions with 4 configurations (2 orbitals each)
    // All must have same electron count (2 electrons)
    Wavefunction::DeterminantVector sci_dets = {
        Configuration("20"), Configuration("ud"), Configuration("02"),
        Configuration("du")};

    // Create SCI coefficients with proper size
    Eigen::VectorXd sci_coeffs_real(4);
    sci_coeffs_real << 0.7, 0.4, 0.3, 0.2;

    Eigen::VectorXcd sci_coeffs_complex(4);
    sci_coeffs_complex << std::complex<double>(0.7, 0.1),
        std::complex<double>(0.4, -0.2), std::complex<double>(0.3, 0.3),
        std::complex<double>(0.2, 0.1);

    sci_real = std::make_shared<Wavefunction>(
        std::make_unique<SciWavefunctionContainer>(sci_coeffs_real, sci_dets,
                                                   orbitals));

    sci_complex = std::make_shared<Wavefunction>(
        std::make_unique<SciWavefunctionContainer>(sci_coeffs_complex, sci_dets,
                                                   orbitals));
  }

  std::shared_ptr<Orbitals> orbitals;
  std::shared_ptr<Wavefunction> cas_real;
  std::shared_ptr<Wavefunction> cas_complex;
  std::shared_ptr<Wavefunction> sd_wavefunction;
  std::shared_ptr<Wavefunction> sci_real;
  std::shared_ptr<Wavefunction> sci_complex;
};

TEST_F(WavefunctionSerializationTest, JSONSerializationCASReal) {
  // Test JSON serialization for real CAS wavefunction
  nlohmann::json j = cas_real->to_json();

  // Verify essential fields are present
  EXPECT_TRUE(j.contains("container_type"));
  EXPECT_TRUE(j.contains("container"));
  EXPECT_EQ(j["container_type"], "cas");

  // Test round-trip serialization
  auto wf_reconstructed = Wavefunction::from_json(j);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), cas_real->size());
  EXPECT_NEAR(wf_reconstructed->norm(), cas_real->norm(),
              testing::wf_tolerance);
  EXPECT_EQ(wf_reconstructed->get_type(), cas_real->get_type());

  // Compare determinants
  auto orig_dets = cas_real->get_active_determinants();
  auto recon_dets = wf_reconstructed->get_active_determinants();
  EXPECT_EQ(recon_dets.size(), orig_dets.size());

  for (size_t i = 0; i < orig_dets.size(); ++i) {
    EXPECT_EQ(recon_dets[i].to_string(), orig_dets[i].to_string());
  }
}

TEST_F(WavefunctionSerializationTest, JSONSerializationCASComplex) {
  // Test JSON serialization for complex CAS wavefunction
  nlohmann::json j = cas_complex->to_json();

  // Verify essential fields are present
  EXPECT_TRUE(j.contains("container_type"));
  EXPECT_TRUE(j.contains("container"));
  EXPECT_EQ(j["container_type"], "cas");

  // Test round-trip serialization
  auto wf_reconstructed = Wavefunction::from_json(j);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), cas_complex->size());
  EXPECT_NEAR(wf_reconstructed->norm(), cas_complex->norm(),
              testing::wf_tolerance);
  EXPECT_TRUE(wf_reconstructed->is_complex());
}

TEST_F(WavefunctionSerializationTest, JSONSerializationSlaterDeterminant) {
  // Test JSON serialization for Slater determinant wavefunction
  nlohmann::json j = sd_wavefunction->to_json();

  // Verify essential fields are present
  EXPECT_TRUE(j.contains("container_type"));
  EXPECT_TRUE(j.contains("container"));
  EXPECT_EQ(j["container_type"], "sd");

  // Test round-trip serialization
  auto wf_reconstructed = Wavefunction::from_json(j);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), 1);
  EXPECT_NEAR(wf_reconstructed->norm(), 1.0, testing::wf_tolerance);
  EXPECT_FALSE(wf_reconstructed->is_complex());
}

TEST_F(WavefunctionSerializationTest, JSONSerializationSCIReal) {
  // Test JSON serialization for real SCI wavefunction
  nlohmann::json j = sci_real->to_json();

  // Verify essential fields are present
  EXPECT_TRUE(j.contains("container_type"));
  EXPECT_TRUE(j.contains("container"));
  EXPECT_EQ(j["container_type"], "sci");

  // Test round-trip serialization
  auto wf_reconstructed = Wavefunction::from_json(j);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), sci_real->size());
  EXPECT_EQ(wf_reconstructed->get_type(), sci_real->get_type());
  EXPECT_FALSE(wf_reconstructed->is_complex());

  // Compare determinants
  auto orig_dets = sci_real->get_active_determinants();
  auto recon_dets = wf_reconstructed->get_active_determinants();
  EXPECT_EQ(recon_dets.size(), orig_dets.size());

  for (size_t i = 0; i < orig_dets.size(); ++i) {
    EXPECT_EQ(recon_dets[i].to_string(), orig_dets[i].to_string());
  }
}

TEST_F(WavefunctionSerializationTest, JSONSerializationSCIComplex) {
  // Test JSON serialization for complex SCI wavefunction
  nlohmann::json j = sci_complex->to_json();

  // Verify essential fields are present
  EXPECT_TRUE(j.contains("container_type"));
  EXPECT_TRUE(j.contains("container"));
  EXPECT_EQ(j["container_type"], "sci");

  // Test round-trip serialization
  auto wf_reconstructed = Wavefunction::from_json(j);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), sci_complex->size());
  EXPECT_TRUE(wf_reconstructed->is_complex());

  // Compare determinants
  auto orig_dets = sci_complex->get_active_determinants();
  auto recon_dets = wf_reconstructed->get_active_determinants();
  EXPECT_EQ(recon_dets.size(), orig_dets.size());

  for (size_t i = 0; i < orig_dets.size(); ++i) {
    EXPECT_EQ(recon_dets[i].to_string(), orig_dets[i].to_string());
  }
}

TEST_F(WavefunctionSerializationTest, HDF5SerializationCASReal) {
  // Test HDF5 serialization for real CAS wavefunction
  std::string filename = "test_wavefunction_cas_real.wavefunction.h5";

  // Save to HDF5 file
  cas_real->to_hdf5_file(filename);

  // Load from HDF5 file
  auto wf_reconstructed = Wavefunction::from_hdf5_file(filename);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), cas_real->size());
  EXPECT_NEAR(wf_reconstructed->norm(), cas_real->norm(),
              testing::wf_tolerance);
  EXPECT_EQ(wf_reconstructed->get_type(), cas_real->get_type());

  // Compare determinants
  auto orig_dets = cas_real->get_active_determinants();
  auto recon_dets = wf_reconstructed->get_active_determinants();
  EXPECT_EQ(recon_dets.size(), orig_dets.size());

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(WavefunctionSerializationTest, HDF5SerializationCASComplex) {
  // Test HDF5 serialization for complex CAS wavefunction
  std::string filename = "test_wavefunction_cas_complex.wavefunction.h5";

  // Save to HDF5 file
  cas_complex->to_hdf5_file(filename);

  // Load from HDF5 file
  auto wf_reconstructed = Wavefunction::from_hdf5_file(filename);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), cas_complex->size());
  EXPECT_NEAR(wf_reconstructed->norm(), cas_complex->norm(),
              testing::wf_tolerance);
  EXPECT_TRUE(wf_reconstructed->is_complex());

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(WavefunctionSerializationTest, HDF5SerializationSlaterDeterminant) {
  // Test HDF5 serialization for Slater determinant wavefunction
  std::string filename = "test_wavefunction_sd.wavefunction.h5";

  // Save to HDF5 file
  sd_wavefunction->to_hdf5_file(filename);

  // Load from HDF5 file
  auto wf_reconstructed = Wavefunction::from_hdf5_file(filename);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), 1);
  EXPECT_NEAR(wf_reconstructed->norm(), 1.0, testing::wf_tolerance);
  EXPECT_FALSE(wf_reconstructed->is_complex());

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(WavefunctionSerializationTest, HDF5SerializationSCIReal) {
  // Test HDF5 serialization for real SCI wavefunction
  std::string filename = "test_wavefunction_sci_real.wavefunction.h5";

  // Save to HDF5 file
  sci_real->to_hdf5_file(filename);

  // Load from HDF5 file
  auto wf_reconstructed = Wavefunction::from_hdf5_file(filename);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), sci_real->size());
  EXPECT_EQ(wf_reconstructed->get_type(), sci_real->get_type());
  EXPECT_FALSE(wf_reconstructed->is_complex());

  // Compare determinants
  auto orig_dets = sci_real->get_active_determinants();
  auto recon_dets = wf_reconstructed->get_active_determinants();
  EXPECT_EQ(recon_dets.size(), orig_dets.size());

  for (size_t i = 0; i < orig_dets.size(); ++i) {
    EXPECT_EQ(recon_dets[i].to_string(), orig_dets[i].to_string());
  }

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(WavefunctionSerializationTest, HDF5SerializationSCIComplex) {
  // Test HDF5 serialization for complex SCI wavefunction
  std::string filename = "test_wavefunction_sci_complex.wavefunction.h5";

  // Save to HDF5 file
  sci_complex->to_hdf5_file(filename);

  // Load from HDF5 file
  auto wf_reconstructed = Wavefunction::from_hdf5_file(filename);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), sci_complex->size());
  EXPECT_TRUE(wf_reconstructed->is_complex());

  // Compare determinants
  auto orig_dets = sci_complex->get_active_determinants();
  auto recon_dets = wf_reconstructed->get_active_determinants();
  EXPECT_EQ(recon_dets.size(), orig_dets.size());

  for (size_t i = 0; i < orig_dets.size(); ++i) {
    EXPECT_EQ(recon_dets[i].to_string(), orig_dets[i].to_string());
  }

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(WavefunctionSerializationTest, JSONFileIO) {
  // Test JSON file I/O
  std::string filename = "test_wavefunction.wavefunction.json";

  // Save to JSON file
  cas_real->to_json_file(filename);

  // Load from JSON file
  auto wf_reconstructed = Wavefunction::from_json_file(filename);
  EXPECT_NE(wf_reconstructed, nullptr);

  // Verify properties are preserved
  EXPECT_EQ(wf_reconstructed->size(), cas_real->size());
  EXPECT_NEAR(wf_reconstructed->norm(), cas_real->norm(),
              testing::wf_tolerance);

  // Clean up
  std::remove(filename.c_str());
}

TEST_F(WavefunctionSerializationTest, GenericFileIO) {
  // Test generic file I/O with different formats
  std::string json_filename = "test_wavefunction_generic.wavefunction.json";
  std::string hdf5_filename = "test_wavefunction_generic.wavefunction.h5";

  // Test JSON format
  cas_real->to_file(json_filename, "json");
  auto wf_json = Wavefunction::from_file(json_filename, "json");
  EXPECT_NE(wf_json, nullptr);
  EXPECT_NEAR(wf_json->norm(), cas_real->norm(), testing::wf_tolerance);

  // Test HDF5 format
  cas_real->to_file(hdf5_filename, "hdf5");
  auto wf_hdf5 = Wavefunction::from_file(hdf5_filename, "hdf5");
  EXPECT_NE(wf_hdf5, nullptr);
  EXPECT_NEAR(wf_hdf5->norm(), cas_real->norm(), testing::wf_tolerance);

  // Test invalid format
  EXPECT_THROW(cas_real->to_file("test.wavefunction.xyz", "xyz"),
               std::invalid_argument);
  EXPECT_THROW(Wavefunction::from_file("test.wavefunction.xyz", "xyz"),
               std::invalid_argument);

  // Clean up
  std::remove(json_filename.c_str());
  std::remove(hdf5_filename.c_str());
}

// Test fixture for active/total determinant conversion
class WavefunctionActiveSpaceConversionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create orbitals with active space defined
    // Total: 10 orbitals (3 inactive, 4 active, 3 virtual)
    size_t num_molecular_orbitals = 10;

    // Define active space: orbitals 3, 4, 5, 6 (0-indexed)
    std::vector<size_t> active_indices = {3, 4, 5, 6};

    // Define inactive space: orbitals 0, 1, 2
    std::vector<size_t> inactive_indices = {0, 1, 2};

    // Create base orbitals and then add active space
    auto base_orbitals = testing::create_test_orbitals(
        num_molecular_orbitals, num_molecular_orbitals, true);
    orbitals = testing::with_active_space(base_orbitals, active_indices,
                                          inactive_indices);

    // Create active space determinants (4 orbitals in active space)
    // Format: 4 active orbitals
    active_det1 = Configuration("2ud0");  // doubly, up, down, empty
    active_det2 = Configuration("u2d0");  // up, doubly, down, empty
    active_det3 = Configuration("ud20");  // up, down, doubly, empty

    // Expected total determinants (10 orbitals: 3 inactive + 4 active + 3
    // virtual) Format: 222 (inactive) + active + 000 (virtual)
    expected_total_det1 = Configuration("2222ud0000");
    expected_total_det2 = Configuration("222u2d0000");
    expected_total_det3 = Configuration("222ud20000");
  }

  std::shared_ptr<Orbitals> orbitals;
  Configuration active_det1, active_det2, active_det3;
  Configuration expected_total_det1, expected_total_det2, expected_total_det3;
};

TEST_F(WavefunctionActiveSpaceConversionTest, GetTotalDeterminantSingle) {
  // Create a wavefunction with a single active determinant
  Eigen::VectorXd coeffs(1);
  coeffs << 1.0;

  Wavefunction::DeterminantVector dets = {active_det1};

  auto wf = std::make_unique<Wavefunction>(
      std::make_unique<SciWavefunctionContainer>(coeffs, dets, orbitals));

  // Convert active determinant to total determinant
  Configuration total_det = wf->get_total_determinant(active_det1);

  // Verify the conversion
  EXPECT_EQ(total_det.to_string(), expected_total_det1.to_string());

  // Verify electron counts
  auto [n_alpha_active, n_beta_active] = active_det1.get_n_electrons();
  auto [n_alpha_total, n_beta_total] = total_det.get_n_electrons();

  // Total should have 3 extra doubly occupied orbitals (6 extra electrons)
  EXPECT_EQ(n_alpha_total, n_alpha_active + 3);
  EXPECT_EQ(n_beta_total, n_beta_active + 3);
}

TEST_F(WavefunctionActiveSpaceConversionTest, GetActiveDeterminantSingle) {
  // Create a wavefunction
  Eigen::VectorXd coeffs(1);
  coeffs << 1.0;

  Wavefunction::DeterminantVector dets = {active_det1};

  auto wf = std::make_unique<Wavefunction>(
      std::make_unique<SciWavefunctionContainer>(coeffs, dets, orbitals));

  // Convert total determinant to active determinant
  Configuration active_det = wf->get_active_determinant(expected_total_det1);

  // Verify the conversion
  EXPECT_EQ(active_det.to_string(), active_det1.to_string());

  // Verify orbital capacity
  EXPECT_EQ(active_det.get_orbital_capacity(), 4);  // 4 active orbitals
}

TEST_F(WavefunctionActiveSpaceConversionTest, RoundTripConversion) {
  // Create a wavefunction
  Eigen::VectorXd coeffs(1);
  coeffs << 1.0;

  Wavefunction::DeterminantVector dets = {active_det1};

  auto wf = std::make_unique<Wavefunction>(
      std::make_unique<SciWavefunctionContainer>(coeffs, dets, orbitals));

  // Round trip: active -> total -> active
  Configuration total_det = wf->get_total_determinant(active_det1);
  Configuration recovered_active = wf->get_active_determinant(total_det);

  EXPECT_EQ(recovered_active.to_string(), active_det1.to_string());

  // Round trip: total -> active -> total
  Configuration active_det = wf->get_active_determinant(expected_total_det1);
  Configuration recovered_total = wf->get_total_determinant(active_det);

  EXPECT_EQ(recovered_total.to_string(), expected_total_det1.to_string());
}

TEST_F(WavefunctionActiveSpaceConversionTest, GetTotalDeterminantsMultiple) {
  // Create a wavefunction with multiple active determinants
  Eigen::VectorXd coeffs(3);
  coeffs << 0.7, 0.2, 0.1;

  Wavefunction::DeterminantVector active_dets = {active_det1, active_det2,
                                                 active_det3};

  auto wf =
      std::make_unique<Wavefunction>(std::make_unique<SciWavefunctionContainer>(
          coeffs, active_dets, orbitals));

  // Get all total determinants
  auto total_dets = wf->get_total_determinants();

  // Verify count
  EXPECT_EQ(total_dets.size(), 3);

  // Verify each conversion
  EXPECT_EQ(total_dets[0].to_string(), expected_total_det1.to_string());
  EXPECT_EQ(total_dets[1].to_string(), expected_total_det2.to_string());
  EXPECT_EQ(total_dets[2].to_string(), expected_total_det3.to_string());

  // Verify all have correct orbital capacity
  for (const auto& det : total_dets) {
    EXPECT_EQ(det.get_orbital_capacity(), 12);  // 10 total orbitals
  }
}

TEST_F(WavefunctionActiveSpaceConversionTest,
       ActiveDetsPreservedInWavefunction) {
  // Verify that get_active_determinants returns the original active space
  // dets
  Eigen::VectorXd coeffs(3);
  coeffs << 0.7, 0.2, 0.1;

  Wavefunction::DeterminantVector active_dets = {active_det1, active_det2,
                                                 active_det3};

  auto wf =
      std::make_unique<Wavefunction>(std::make_unique<SciWavefunctionContainer>(
          coeffs, active_dets, orbitals));

  // Get active determinants (should be unchanged)
  const auto& retrieved_active = wf->get_active_determinants();

  EXPECT_EQ(retrieved_active.size(), 3);
  EXPECT_EQ(retrieved_active[0].to_string(), active_det1.to_string());
  EXPECT_EQ(retrieved_active[1].to_string(), active_det2.to_string());
  EXPECT_EQ(retrieved_active[2].to_string(), active_det3.to_string());

  // Verify orbital capacity is for active space only
  for (const auto& det : retrieved_active) {
    EXPECT_EQ(det.get_orbital_capacity(), 4);  // 4 active orbitals
  }
}

TEST_F(WavefunctionActiveSpaceConversionTest, NoActiveSpacePassthrough) {
  // Test behavior when no active space is defined
  size_t num_atomic_orbitals = 6;
  size_t num_molecular_orbitals = 6;
  Eigen::MatrixXd coeffs_no_active =
      Eigen::MatrixXd::Identity(num_atomic_orbitals, num_molecular_orbitals);

  auto orbitals_no_active = std::make_shared<Orbitals>(
      coeffs_no_active, std::nullopt, std::nullopt,
      testing::create_random_basis_set(num_molecular_orbitals), std::nullopt);

  Configuration det = Configuration("2ud0ud");

  Eigen::VectorXd wf_coeffs(1);
  wf_coeffs << 1.0;

  auto wf =
      std::make_unique<Wavefunction>(std::make_unique<SciWavefunctionContainer>(
          wf_coeffs, Wavefunction::DeterminantVector{det}, orbitals_no_active));

  // When no active space is defined, functions should pass through
  Configuration total = wf->get_total_determinant(det);
  Configuration active = wf->get_active_determinant(det);

  EXPECT_EQ(total.to_string(), det.to_string());
  EXPECT_EQ(active.to_string(), det.to_string());
}

TEST_F(WavefunctionActiveSpaceConversionTest, SlaterDeterminantContainer) {
  // Test with SlaterDeterminantContainer (single determinant)
  auto wf = std::make_unique<Wavefunction>(
      std::make_unique<SlaterDeterminantContainer>(active_det1, orbitals));

  // Convert to total determinant
  Configuration total_det = wf->get_total_determinant(active_det1);
  EXPECT_EQ(total_det.to_string(), expected_total_det1.to_string());

  // Get all total determinants (should be one)
  auto total_dets = wf->get_total_determinants();
  EXPECT_EQ(total_dets.size(), 1);
  EXPECT_EQ(total_dets[0].to_string(), expected_total_det1.to_string());

  // Convert back to active
  Configuration recovered_active = wf->get_active_determinant(total_det);
  EXPECT_EQ(recovered_active.to_string(), active_det1.to_string());
}

TEST_F(WavefunctionCoreTest, DataTypeName) {
  // Test that Wavefunction has the correct data type name
  EXPECT_EQ(wf->get_data_type_name(), "wavefunction");
}
