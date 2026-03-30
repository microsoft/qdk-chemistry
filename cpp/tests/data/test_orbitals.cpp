// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class OrbitalsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean up any test files from previous runs
    std::filesystem::remove("test.orbitals.json");
    std::filesystem::remove("test.orbitals.h5");
  }

  void TearDown() override {
    // Clean up test files
    std::filesystem::remove("test.orbitals.json");
    std::filesystem::remove("test.orbitals.h5");
  }
};

TEST_F(OrbitalsTest, Constructors) {
  // Test construction with basic data
  const int n_basis = 4;
  const int n_orbitals = 3;

  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2;

  Eigen::VectorXd energies(n_orbitals);
  energies << -1.0, -0.5, 0.2;

  auto basis_set = testing::create_random_basis_set(n_basis);
  Orbitals orb1(coeffs, energies, std::nullopt, basis_set, std::nullopt);

  EXPECT_EQ(n_basis, orb1.get_num_atomic_orbitals());
  EXPECT_EQ(n_orbitals, orb1.get_num_molecular_orbitals());

  // Copy constructor
  Orbitals orb2(orb1);
  EXPECT_EQ(orb1.get_num_atomic_orbitals(), orb2.get_num_atomic_orbitals());
  EXPECT_EQ(orb1.get_num_molecular_orbitals(),
            orb2.get_num_molecular_orbitals());

  // Test constructor with restricted calculation
  Orbitals orb3(coeffs, energies, std::nullopt, basis_set, std::nullopt);
  EXPECT_EQ(n_basis, orb3.get_num_atomic_orbitals());
  EXPECT_EQ(n_orbitals, orb3.get_num_molecular_orbitals());
}

TEST_F(OrbitalsTest, CoefficientManagement) {
  const int n_basis = 3;
  const int n_orbitals = 2;

  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;

  auto basis_set = testing::create_random_basis_set(n_basis);
  Orbitals orb(coeffs, std::nullopt, std::nullopt, basis_set, std::nullopt);

  EXPECT_EQ(n_basis, orb.get_num_atomic_orbitals());
  EXPECT_EQ(n_orbitals, orb.get_num_molecular_orbitals());

  const auto& [alpha_coeffs, beta_coeffs] = orb.get_coefficients();
  EXPECT_EQ(coeffs.rows(), alpha_coeffs.rows());
  EXPECT_EQ(coeffs.cols(), alpha_coeffs.cols());

  for (int i = 0; i < coeffs.rows(); ++i) {
    for (int j = 0; j < coeffs.cols(); ++j) {
      EXPECT_NEAR(coeffs(i, j), alpha_coeffs(i, j),
                  testing::numerical_zero_tolerance);
      // For restricted calculation, alpha and beta should be identical
      EXPECT_NEAR(alpha_coeffs(i, j), beta_coeffs(i, j),
                  testing::numerical_zero_tolerance);
    }
  }
}

TEST_F(OrbitalsTest, EnergyManagement) {
  Eigen::VectorXd energies(3);
  energies << -2.0, -1.0, 0.5;

  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(3, 3);

  auto basis_set = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, energies, std::nullopt, basis_set, std::nullopt);

  const auto& [alpha_energies, beta_energies] = orb.get_energies();
  EXPECT_EQ(energies.size(), alpha_energies.size());
  EXPECT_EQ(energies.size(), beta_energies.size());

  for (int i = 0; i < energies.size(); ++i) {
    EXPECT_NEAR(energies(i), alpha_energies(i),
                testing::numerical_zero_tolerance);
    // For restricted calculation, alpha and beta should be identical
    EXPECT_NEAR(alpha_energies(i), beta_energies(i),
                testing::numerical_zero_tolerance);
  }
}

TEST_F(OrbitalsTest, AOOverlap) {
  const int n_basis = 3;
  Eigen::MatrixXd overlap(n_basis, n_basis);
  overlap << 1.0, 0.2, 0.1, 0.2, 1.0, 0.3, 0.1, 0.3, 1.0;

  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(n_basis, n_basis);

  auto basis_set = testing::create_random_basis_set(n_basis);
  Orbitals orb(coeffs, std::nullopt, overlap, basis_set, std::nullopt);

  const auto& retrieved_overlap = orb.get_overlap_matrix();
  EXPECT_EQ(overlap.rows(), retrieved_overlap.rows());
  EXPECT_EQ(overlap.cols(), retrieved_overlap.cols());

  for (int i = 0; i < overlap.rows(); ++i) {
    for (int j = 0; j < overlap.cols(); ++j) {
      EXPECT_NEAR(overlap(i, j), retrieved_overlap(i, j),
                  testing::numerical_zero_tolerance);
    }
  }

  // Test that overlap matrix is symmetric
  EXPECT_TRUE(orb.has_overlap_matrix());
  for (int i = 0; i < n_basis; ++i) {
    for (int j = 0; j < n_basis; ++j) {
      EXPECT_NEAR(retrieved_overlap(i, j), retrieved_overlap(j, i),
                  testing::numerical_zero_tolerance);
    }
  }
}

TEST_F(OrbitalsTest, SizeAndDimensionQueries) {
  // Test that empty matrices throw exception during construction
  Eigen::MatrixXd empty_coeffs(0, 0);
  auto empty_basis = testing::create_random_basis_set(1);  // Still need a basis
  EXPECT_THROW(Orbitals(empty_coeffs, std::nullopt, std::nullopt, empty_basis,
                        std::nullopt),
               std::runtime_error);

  // Set up data
  const int n_basis = 5;
  const int n_orbitals = 4;

  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs.setRandom();
  Eigen::VectorXd energies(n_orbitals);
  energies.setRandom();

  auto basis_set = testing::create_random_basis_set(n_basis);
  Orbitals orb(coeffs, energies, std::nullopt, basis_set, std::nullopt);

  EXPECT_EQ(n_basis, orb.get_num_atomic_orbitals());
  EXPECT_EQ(n_orbitals, orb.get_num_molecular_orbitals());

  // Test matrix dimensions
  const auto& [alpha_coeffs, beta_coeffs] = orb.get_coefficients();
  EXPECT_EQ(n_basis, alpha_coeffs.rows());
  EXPECT_EQ(n_orbitals, alpha_coeffs.cols());
  const auto& [alpha_energies, beta_energies] = orb.get_energies();
  EXPECT_EQ(n_orbitals, alpha_energies.size());
}

TEST_F(OrbitalsTest, OpenShellAndRestrictedQueries) {
  // Set up restricted (alpha = beta)
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  Eigen::VectorXd energies(2);
  energies << -1.0, 0.5;

  auto basis_set = testing::create_random_basis_set(2);
  Orbitals restricted_orb(coeffs, energies, std::nullopt, basis_set,
                          std::nullopt);

  // Should be restricted and closed shell
  EXPECT_TRUE(restricted_orb.is_restricted());

  // Test unrestricted
  Orbitals unrestricted_orb(coeffs, coeffs, energies, energies, std::nullopt,
                            basis_set);

  // Should now be open shell
  EXPECT_TRUE(unrestricted_orb.is_restricted());
}

TEST_F(OrbitalsTest, HasEnergies) {
  // Check has_energies() without energies
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(3, 3);

  auto basis_set = testing::create_random_basis_set(3);
  Orbitals orb_no_energies(coeffs, std::nullopt, std::nullopt, basis_set,
                           std::nullopt);

  EXPECT_FALSE(orb_no_energies.has_energies());

  // Set energies for restricted case
  Eigen::VectorXd energies(3);
  energies << -1.0, 0.0, 1.0;

  Orbitals orb_with_energies(coeffs, energies, std::nullopt, basis_set,
                             std::nullopt);

  // Check has_energies() after setting
  EXPECT_TRUE(orb_with_energies.has_energies());

  // Test with unrestricted energies
  Eigen::VectorXd alpha_energies(2), beta_energies(2);
  alpha_energies << -1.0, 0.5;
  beta_energies << -0.9, 0.6;
  Eigen::VectorXd beta_occ = Eigen::VectorXd::Ones(2);
  coeffs.resize(2, 2);
  coeffs.setIdentity();

  auto basis_set_2x2 = testing::create_random_basis_set(2);
  Orbitals orb2(coeffs, coeffs, alpha_energies, beta_energies, std::nullopt,
                basis_set_2x2, std::nullopt);

  // Check has_energies() after setting unrestricted energies
  EXPECT_TRUE(orb2.has_energies());
}

TEST_F(OrbitalsTest, Validation) {
  // Empty orbitals should throw exception during construction
  Eigen::MatrixXd empty_coeffs(0, 0);
  auto empty_basis_val = testing::create_random_basis_set(1);
  EXPECT_THROW(Orbitals(empty_coeffs, std::nullopt, std::nullopt,
                        empty_basis_val, std::nullopt),
               std::runtime_error);

  // Set minimal valid data
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  Eigen::VectorXd energies(2);
  energies << -1.0, 0.5;

  auto valid_basis = testing::create_random_basis_set(2);
  Orbitals valid_orb(coeffs, energies, std::nullopt, valid_basis, std::nullopt);
}

TEST_F(OrbitalsTest, JSONSerialization) {
  // Set up test data
  Eigen::MatrixXd coeffs(2, 2);
  coeffs << 0.1, 0.2, 0.3, 0.4;
  Eigen::VectorXd energies(2);
  energies << -1.0, 0.5;
  Eigen::VectorXd occupations(2);
  occupations << 2.0, 0.0;

  auto json_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, energies, std::nullopt, json_basis, std::nullopt);

  // Test JSON conversion
  auto json_data = orb.to_json();
  EXPECT_FALSE(json_data.empty());

  // Test file I/O with round-trip
  orb.to_json_file("test.orbitals.json");
  auto orb_json = Orbitals::from_json_file("test.orbitals.json");

  // Verify dimensions are preserved
  EXPECT_EQ(orb.get_num_atomic_orbitals(), orb_json->get_num_atomic_orbitals());
  EXPECT_EQ(orb.get_num_molecular_orbitals(),
            orb_json->get_num_molecular_orbitals());

  // Check coefficients are preserved
  auto [orig_coeffs_a, orig_coeffs_b] = orb.get_coefficients();
  auto [json_coeffs_a, json_coeffs_b] = orb_json->get_coefficients();
  EXPECT_TRUE(orig_coeffs_a.isApprox(json_coeffs_a, testing::json_tolerance));
  EXPECT_TRUE(orig_coeffs_b.isApprox(json_coeffs_b, testing::json_tolerance));

  // Check energies are preserved
  auto [orig_energies_a, orig_energies_b] = orb.get_energies();
  auto [json_energies_a, json_energies_b] = orb_json->get_energies();
  EXPECT_TRUE(
      orig_energies_a.isApprox(json_energies_a, testing::json_tolerance));
  EXPECT_TRUE(
      orig_energies_b.isApprox(json_energies_b, testing::json_tolerance));
}

TEST_F(OrbitalsTest, HDF5Serialization) {
  // Set up test data
  Eigen::MatrixXd coeffs(3, 3);
  coeffs.setRandom();
  Eigen::VectorXd energies(3);
  energies.setRandom();

  auto hdf5_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, energies, std::nullopt, hdf5_basis, std::nullopt);

  // Test HDF5 conversion - use correct filename format
  std::string hdf5_filename = "test.orbitals.h5";
  orb.to_hdf5_file(hdf5_filename);

  auto orb_from_file = Orbitals::from_hdf5_file(hdf5_filename);
  EXPECT_EQ(orb.get_num_atomic_orbitals(),
            orb_from_file->get_num_atomic_orbitals());
  EXPECT_EQ(orb.get_num_molecular_orbitals(),
            orb_from_file->get_num_molecular_orbitals());
}

TEST_F(OrbitalsTest, UnrestrictedCalculations) {
  const int n_basis = 3;
  const int n_orbitals = 2;

  // Set up different alpha and beta coefficients
  Eigen::MatrixXd alpha_coeffs(n_basis, n_orbitals);
  alpha_coeffs << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;

  Eigen::MatrixXd beta_coeffs(n_basis, n_orbitals);
  beta_coeffs << 0.11, 0.22, 0.33, 0.44, 0.55, 0.66;

  // Set different alpha and beta energies
  Eigen::VectorXd alpha_energies(n_orbitals);
  alpha_energies << -1.0, 0.5;
  Eigen::VectorXd beta_energies(n_orbitals);
  beta_energies << -1.1, 0.4;

  // Create the orbitals object with unrestricted data
  auto unrestricted_basis = testing::create_random_basis_set(n_basis);
  Orbitals orb(alpha_coeffs, beta_coeffs, alpha_energies, beta_energies,
               std::nullopt, unrestricted_basis, std::nullopt);

  // Verify unrestricted nature
  EXPECT_FALSE(orb.is_restricted());

  // Test coefficient retrieval
  const auto& [retrieved_alpha_coeffs, retrieved_beta_coeffs] =
      orb.get_coefficients();
  for (int i = 0; i < n_basis; ++i) {
    for (int j = 0; j < n_orbitals; ++j) {
      EXPECT_NEAR(alpha_coeffs(i, j), retrieved_alpha_coeffs(i, j),
                  testing::numerical_zero_tolerance);
      EXPECT_NEAR(beta_coeffs(i, j), retrieved_beta_coeffs(i, j),
                  testing::numerical_zero_tolerance);
    }
  }

  // Test energy retrieval
  const auto& [retrieved_alpha_energies, retrieved_beta_energies] =
      orb.get_energies();
  for (int i = 0; i < n_orbitals; ++i) {
    EXPECT_NEAR(alpha_energies(i), retrieved_alpha_energies(i),
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(beta_energies(i), retrieved_beta_energies(i),
                testing::numerical_zero_tolerance);
  }
}

TEST_F(OrbitalsTest, BasisSetManagement) {
  // Create basic orbital data for testing
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(2, 2);

  // Test that basis set is now required (should throw with nullptr)
  EXPECT_THROW(
      Orbitals(coeffs, std::nullopt, std::nullopt, nullptr, std::nullopt),
      std::runtime_error);

  // Create a minimal structure for the basis set
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure structure(coords, symbols);

  // Create a valid basis set with shells (empty basis sets are invalid)
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));
  auto basis = std::make_shared<BasisSet>("test", shells, structure);
  Orbitals orb_with_basis(coeffs, std::nullopt, std::nullopt, basis,
                          std::nullopt);
  EXPECT_TRUE(orb_with_basis.has_basis_set());

  // Test retrieval
  const auto& retrieved_basis = orb_with_basis.get_basis_set();
  // Basic test that we can retrieve it without throwing
}

TEST_F(OrbitalsTest, SummaryString) {
  // Set up minimal orbital data
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  Eigen::VectorXd energies(2);
  energies << -1.0, 0.5;

  auto summary_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, energies, std::nullopt, summary_basis, std::nullopt);

  // Test that summary string is non-empty and contains relevant information
  std::string summary = orb.get_summary();
  EXPECT_FALSE(summary.empty());
}

TEST_F(OrbitalsTest, ErrorHandling) {
  // Test that constructor throws for empty data
  Eigen::MatrixXd empty_coeffs(0, 0);
  EXPECT_THROW(
      Orbitals(empty_coeffs, std::nullopt, std::nullopt, nullptr, std::nullopt),
      std::runtime_error);

  // Create a valid orbital object without energies or overlap for testing
  // getter exceptions
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  auto error_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, std::nullopt, std::nullopt, error_basis, std::nullopt);

  // Test accessing missing data throws exceptions
  EXPECT_THROW(orb.get_energies(), std::runtime_error);
  EXPECT_THROW(orb.get_overlap_matrix(), std::runtime_error);

  // Test invalid file operations - these might need to be instance methods
  EXPECT_THROW(Orbitals::from_hdf5_file("nonexistent.orbitals.h5"),
               std::runtime_error);
  EXPECT_THROW(Orbitals::from_json_file("nonexistent.orbitals.json"),
               std::runtime_error);
}

TEST_F(OrbitalsTest, FileIOGeneric) {
  // Create a complete orbital set
  Eigen::MatrixXd coeffs(3, 2);
  coeffs << 0.9, 0.1, 0.1, -0.9, 0.0, 0.0;
  Eigen::VectorXd energies(2);
  energies << -1.0, 0.5;
  Eigen::MatrixXd overlap(3, 3);
  overlap.setIdentity();
  overlap(0, 1) = 0.1;
  overlap(1, 0) = 0.1;
  auto fileio_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, energies, overlap, fileio_basis);

  // Test JSON file I/O using generic methods
  orb.to_file("test.orbitals.json", "json");

  auto orb_json = Orbitals::from_file("test.orbitals.json", "json");

  EXPECT_EQ(orb_json->get_num_atomic_orbitals(), orb.get_num_atomic_orbitals());
  EXPECT_EQ(orb_json->get_num_molecular_orbitals(),
            orb.get_num_molecular_orbitals());

  // Check coefficients are preserved
  auto [orig_coeffs_a, orig_coeffs_b] = orb.get_coefficients();
  auto [json_coeffs_a, json_coeffs_b] = orb_json->get_coefficients();
  EXPECT_TRUE(orig_coeffs_a.isApprox(json_coeffs_a, testing::json_tolerance));
  EXPECT_TRUE(orig_coeffs_b.isApprox(json_coeffs_b, testing::json_tolerance));

  // Test HDF5 file I/O using generic methods
  orb.to_file("test.orbitals.h5", "hdf5");

  auto orb_hdf5 = Orbitals::from_file("test.orbitals.h5", "hdf5");

  EXPECT_EQ(orb_hdf5->get_num_atomic_orbitals(), orb.get_num_atomic_orbitals());
  EXPECT_EQ(orb_hdf5->get_num_molecular_orbitals(),
            orb.get_num_molecular_orbitals());

  // Check coefficients are preserved
  auto [hdf5_coeffs_a, hdf5_coeffs_b] = orb_hdf5->get_coefficients();
  EXPECT_TRUE(orig_coeffs_a.isApprox(hdf5_coeffs_a, testing::json_tolerance));
  EXPECT_TRUE(orig_coeffs_b.isApprox(hdf5_coeffs_b, testing::json_tolerance));

  // Test unsupported file type
  EXPECT_THROW(orb.to_file("test.orbitals.xyz", "xyz"), std::runtime_error);
  EXPECT_THROW(Orbitals::from_file("test.orbitals.xyz", "xyz"),
               std::runtime_error);
}

TEST_F(OrbitalsTest, FileIOSpecific) {
  // Create a complete orbital set
  Eigen::MatrixXd coeffs(3, 2);
  coeffs << 0.9, 0.1, 0.1, -0.9, 0.0, 0.0;
  Eigen::VectorXd energies(2);
  energies << -1.0, 0.5;
  Eigen::MatrixXd overlap(3, 3);
  overlap.setIdentity();
  overlap(0, 1) = 0.1;
  overlap(1, 0) = 0.1;
  auto hdf5_specific_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, energies, overlap, hdf5_specific_basis, std::nullopt);

  // Test HDF5 file I/O methods
  orb.to_hdf5_file("test.orbitals.h5");

  auto orb_hdf5 = Orbitals::from_hdf5_file("test.orbitals.h5");

  EXPECT_EQ(orb_hdf5->get_num_atomic_orbitals(), orb.get_num_atomic_orbitals());
  EXPECT_EQ(orb_hdf5->get_num_molecular_orbitals(),
            orb.get_num_molecular_orbitals());

  // Check all data is preserved
  auto [orig_coeffs_a, orig_coeffs_b] = orb.get_coefficients();
  auto [hdf5_coeffs_a, hdf5_coeffs_b] = orb_hdf5->get_coefficients();
  EXPECT_TRUE(orig_coeffs_a.isApprox(hdf5_coeffs_a, testing::json_tolerance));
  EXPECT_TRUE(orig_coeffs_b.isApprox(hdf5_coeffs_b, testing::json_tolerance));

  auto [orig_energies_a, orig_energies_b] = orb.get_energies();
  auto [hdf5_energies_a, hdf5_energies_b] = orb_hdf5->get_energies();
  EXPECT_TRUE(
      orig_energies_a.isApprox(hdf5_energies_a, testing::json_tolerance));
  EXPECT_TRUE(
      orig_energies_b.isApprox(hdf5_energies_b, testing::json_tolerance));

  EXPECT_TRUE(orb.get_overlap_matrix().isApprox(orb_hdf5->get_overlap_matrix(),
                                                testing::json_tolerance));

  // Test updated JSON file I/O methods
  orb.to_json_file("test.orbitals.json");

  auto orb_json = Orbitals::from_json_file("test.orbitals.json");

  EXPECT_EQ(orb_json->get_num_atomic_orbitals(), orb.get_num_atomic_orbitals());
  EXPECT_EQ(orb_json->get_num_molecular_orbitals(),
            orb.get_num_molecular_orbitals());

  // Check coefficients are preserved
  auto [json_coeffs_a, json_coeffs_b] = orb_json->get_coefficients();
  EXPECT_TRUE(orig_coeffs_a.isApprox(json_coeffs_a, testing::json_tolerance));
  EXPECT_TRUE(orig_coeffs_b.isApprox(json_coeffs_b, testing::json_tolerance));
}

TEST_F(OrbitalsTest, FileIOValidation) {
  // Create coefficients first
  // Set minimal valid data
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  auto validation_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, std::nullopt, std::nullopt, validation_basis,
               std::nullopt);

  // Test filename validation for JSON files
  EXPECT_THROW(orb.to_json_file("test.json"), std::invalid_argument);
  EXPECT_THROW(orb.from_json_file("test.json"), std::invalid_argument);

  // Test filename validation for HDF5 files
  EXPECT_THROW(orb.to_hdf5_file("test.h5"), std::invalid_argument);
  EXPECT_THROW(orb.from_hdf5_file("test.h5"), std::invalid_argument);

  // Test non-existent file
  EXPECT_THROW(orb.from_json_file("nonexistent.orbitals.json"),
               std::runtime_error);
  EXPECT_THROW(orb.from_hdf5_file("nonexistent.orbitals.h5"),
               std::runtime_error);
}

TEST_F(OrbitalsTest, ActiveSpaceManagement) {
  // Create coefficients first
  // Set up minimal valid data first
  Eigen::MatrixXd coeffs(4, 4);
  coeffs.setIdentity();
  Eigen::VectorXd energies(4);
  energies << -1.0, -0.5, 0.5, 1.0;
  std::vector<size_t> active_indices = {1, 2};
  auto active_basis = testing::create_random_basis_set(4);
  Orbitals orb(coeffs, energies, std::nullopt, active_basis,
               std::make_tuple(active_indices, std::vector<size_t>{}));

  // Check active space is set
  EXPECT_TRUE(orb.has_active_space());

  // Check active space indices are correctly stored
  // For restricted case, both alpha and beta indices should match input
  auto [alpha_indices, beta_indices] = orb.get_active_space_indices();
  EXPECT_EQ(active_indices, alpha_indices);
  EXPECT_EQ(active_indices, beta_indices);
}

TEST_F(OrbitalsTest, InactiveSpaceManagement) {
  // Create coefficients first
  // Set up minimal valid data first
  Eigen::MatrixXd coeffs(4, 4);
  coeffs.setIdentity();
  Eigen::VectorXd energies(4);
  energies << -1.0, -0.5, 0.5, 1.0;
  std::vector<size_t> inactive_indices = {0, 1};
  auto basis_set = testing::create_random_basis_set(4);
  Orbitals orb(coeffs, energies, std::nullopt, basis_set,
               std::make_tuple(std::vector<size_t>{}, inactive_indices));

  // Check inactive space is set
  EXPECT_TRUE(orb.has_inactive_space());

  // Check inactive space indices are correctly stored
  // For restricted case, both alpha and beta indices should match input
  auto [alpha_inactive_indices, beta_inactive_indices] =
      orb.get_inactive_space_indices();
  EXPECT_EQ(inactive_indices, alpha_inactive_indices);
  EXPECT_EQ(inactive_indices, beta_inactive_indices);
}

TEST_F(OrbitalsTest, ActiveSpaceSerialization) {
  // Create coefficients first
  Eigen::MatrixXd coeffs(3, 2);
  coeffs << 0.9, 0.1, 0.1, -0.9, 0.0, 0.0;
  Eigen::VectorXd energies(2);
  energies << -1.0, 0.5;
  std::vector<size_t> active_indices = {0, 1};
  auto active_serial_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, energies, std::nullopt, active_serial_basis,
               std::make_tuple(active_indices, std::vector<size_t>{}));

  // Test JSON serialization
  orb.to_json_file("test.orbitals.json");
  auto orb_json = Orbitals::from_json_file("test.orbitals.json");

  // Check active space data is preserved
  EXPECT_TRUE(orb_json->has_active_space());
  auto [json_alpha_indices, json_beta_indices] =
      orb_json->get_active_space_indices();
  EXPECT_EQ(active_indices, json_alpha_indices);
  EXPECT_EQ(active_indices, json_beta_indices);

  // Test HDF5 serialization
  orb.to_hdf5_file("test.orbitals.h5");
  auto orb_hdf5 = Orbitals::from_hdf5_file("test.orbitals.h5");

  // Check active space data is preserved
  EXPECT_TRUE(orb_hdf5->has_active_space());
  auto [hdf5_alpha_indices, hdf5_beta_indices] =
      orb_hdf5->get_active_space_indices();
  EXPECT_EQ(active_indices, hdf5_alpha_indices);
  EXPECT_EQ(active_indices, hdf5_beta_indices);
}

TEST_F(OrbitalsTest, UnrestrictedActiveSpaceSerialization) {
  // Create an unrestricted orbital set with different alpha/beta active spaces
  Eigen::MatrixXd coeffs_alpha(3, 2);
  coeffs_alpha << 0.9, 0.1, 0.1, -0.9, 0.0, 0.0;
  Eigen::MatrixXd coeffs_beta(3, 2);
  coeffs_beta << 0.8, 0.2, 0.2, -0.8, 0.1, 0.1;
  Eigen::VectorXd energies_alpha(2);
  energies_alpha << -1.0, 0.5;
  Eigen::VectorXd energies_beta(2);
  energies_beta << -0.9, 0.6;
  std::vector<size_t> alpha_active_indices = {0};
  std::vector<size_t> beta_active_indices = {1};
  auto unrestricted_active_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs_alpha, coeffs_beta, energies_alpha, energies_beta,
               std::nullopt, unrestricted_active_basis,
               std::make_tuple(alpha_active_indices, beta_active_indices,
                               std::vector<size_t>{}, std::vector<size_t>{}));

  // Test JSON round-trip
  orb.to_json_file("test.orbitals.json");
  auto orb_json = Orbitals::from_file("test.orbitals.json", "json");

  // Verify active space data
  EXPECT_TRUE(orb_json->has_active_space());
  auto [json_alpha_indices, json_beta_indices] =
      orb_json->get_active_space_indices();
  EXPECT_EQ(alpha_active_indices, json_alpha_indices);
  EXPECT_EQ(beta_active_indices, json_beta_indices);

  // Test HDF5 round-trip
  orb.to_hdf5_file("test.orbitals.h5");
  auto orb_hdf5 = Orbitals::from_file("test.orbitals.h5", "hdf5");

  // Verify active space data
  EXPECT_TRUE(orb_hdf5->has_active_space());
  auto [hdf5_alpha_indices, hdf5_beta_indices] =
      orb_hdf5->get_active_space_indices();
  EXPECT_EQ(alpha_active_indices, hdf5_alpha_indices);
  EXPECT_EQ(beta_active_indices, hdf5_beta_indices);
}

TEST_F(OrbitalsTest, CopyConstructorWithActiveSpace) {
  // Create coefficients first
  // Set up basic orbital data
  Eigen::MatrixXd coeffs(3, 2);
  coeffs << 0.9, 0.1, 0.1, -0.9, 0.0, 0.0;
  std::vector<size_t> active_indices = {0, 1};
  auto copy_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, Eigen::VectorXd::Random(2), std::nullopt, copy_basis,
               std::make_tuple(active_indices, std::vector<size_t>{}));

  // Create a copy via copy constructor
  Orbitals orb_copy(orb);

  // Verify active space data is copied
  EXPECT_TRUE(orb_copy.has_active_space());
  auto [copy_alpha_indices, copy_beta_indices] =
      orb_copy.get_active_space_indices();
  EXPECT_EQ(active_indices, copy_alpha_indices);
  EXPECT_EQ(active_indices, copy_beta_indices);

  // Test assignment operator
  Orbitals orb_assigned = orb;

  // Verify active space data is copied via assignment
  EXPECT_TRUE(orb_assigned.has_active_space());
  auto [assigned_alpha_indices, assigned_beta_indices] =
      orb_assigned.get_active_space_indices();
  EXPECT_EQ(active_indices, assigned_alpha_indices);
  EXPECT_EQ(active_indices, assigned_beta_indices);
}

TEST_F(OrbitalsTest, FileIORoundTrip) {
  // Create a complex orbital set with unrestricted calculation
  // Create coefficients first
  // Set alpha and beta coefficients (unrestricted)
  Eigen::MatrixXd coeffs_alpha(3, 2);
  coeffs_alpha << 0.9, 0.1, 0.1, -0.9, 0.0, 0.0;
  Eigen::MatrixXd coeffs_beta(3, 2);
  coeffs_beta << 0.8, 0.2, 0.2, -0.8, 0.1, 0.1;

  // Set alpha and beta energies
  Eigen::VectorXd energies_alpha(2);
  energies_alpha << -1.0, 0.5;
  Eigen::VectorXd energies_beta(2);
  energies_beta << -0.9, 0.6;

  // Set AO overlap
  Eigen::MatrixXd overlap(3, 3);
  overlap.setIdentity();
  overlap(0, 1) = 0.1;
  overlap(1, 0) = 0.1;

  auto roundtrip_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs_alpha, coeffs_beta, energies_alpha, energies_beta,
               overlap, roundtrip_basis, std::nullopt);

  // Test JSON round-trip
  orb.to_json_file("test.orbitals.json");
  auto orb_json = Orbitals::from_json_file("test.orbitals.json");

  // Check all properties are preserved
  EXPECT_EQ(orb_json->get_num_atomic_orbitals(), orb.get_num_atomic_orbitals());
  EXPECT_EQ(orb_json->get_num_molecular_orbitals(),
            orb.get_num_molecular_orbitals());

  // Check coefficients
  auto [orig_coeffs_a, orig_coeffs_b] = orb.get_coefficients();
  auto [json_coeffs_a, json_coeffs_b] = orb_json->get_coefficients();
  EXPECT_TRUE(orig_coeffs_a.isApprox(json_coeffs_a, testing::json_tolerance));
  EXPECT_TRUE(orig_coeffs_b.isApprox(json_coeffs_b, testing::json_tolerance));

  // Check energies
  auto [orig_energies_a, orig_energies_b] = orb.get_energies();
  auto [json_energies_a, json_energies_b] = orb_json->get_energies();
  EXPECT_TRUE(
      orig_energies_a.isApprox(json_energies_a, testing::json_tolerance));
  EXPECT_TRUE(
      orig_energies_b.isApprox(json_energies_b, testing::json_tolerance));

  // Check overlap
  EXPECT_TRUE(orb.get_overlap_matrix().isApprox(orb_json->get_overlap_matrix(),
                                                testing::json_tolerance));

  // Test HDF5 round-trip
  orb.to_hdf5_file("test.orbitals.h5");
  auto orb_hdf5 = Orbitals::from_hdf5_file("test.orbitals.h5");

  // Check all properties are preserved
  EXPECT_EQ(orb_hdf5->get_num_atomic_orbitals(), orb.get_num_atomic_orbitals());
  EXPECT_EQ(orb_hdf5->get_num_molecular_orbitals(),
            orb.get_num_molecular_orbitals());

  // Check coefficients
  auto [hdf5_coeffs_a, hdf5_coeffs_b] = orb_hdf5->get_coefficients();
  EXPECT_TRUE(orig_coeffs_a.isApprox(hdf5_coeffs_a, testing::json_tolerance));
  EXPECT_TRUE(orig_coeffs_b.isApprox(hdf5_coeffs_b, testing::json_tolerance));

  // Check energies
  auto [hdf5_energies_a, hdf5_energies_b] = orb_hdf5->get_energies();
  EXPECT_TRUE(
      orig_energies_a.isApprox(hdf5_energies_a, testing::json_tolerance));
  EXPECT_TRUE(
      orig_energies_b.isApprox(hdf5_energies_b, testing::json_tolerance));

  // Check overlap
  EXPECT_TRUE(orb.get_overlap_matrix().isApprox(orb_hdf5->get_overlap_matrix(),
                                                testing::json_tolerance));
}

TEST_F(OrbitalsTest, DataTypeName) {
  // Test that Orbitals has the correct data type name
  auto orbitals = testing::create_test_orbitals();
  EXPECT_EQ(orbitals->get_data_type_name(), "orbitals");
}

// -----------------------------------------------------------------------
// Tests merged from test_orbitals_edge_cases.cpp
// -----------------------------------------------------------------------
class OrbitalsEdgeCasesTest : public ::testing::Test {};

TEST_F(OrbitalsEdgeCasesTest, ErrorHandling) {
  // Set up some basic data
  Eigen::MatrixXd coeffs(3, 2);
  coeffs.setRandom();
  Eigen::VectorXd energies(2);
  energies.setRandom();

  // Create orbitals object with minimal required data
  auto error_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, energies, std::nullopt, error_basis, std::nullopt);

  // Test invalid JSON file
  EXPECT_THROW(Orbitals::from_json_file("nonexistent.orbitals.json"),
               std::runtime_error);

  // Test invalid HDF5 file
  EXPECT_THROW(Orbitals::from_hdf5_file("nonexistent.orbitals.h5"),
               std::runtime_error);
}

TEST_F(OrbitalsEdgeCasesTest, EmptyDataHandling) {
  // Test setting empty matrices - empty coefficients should throw in
  // constructor
  Eigen::MatrixXd empty_coeffs(0, 0);
  Eigen::VectorXd empty_energies(0);

  // Creating a basis set with 0 functions should be invalid now
  EXPECT_THROW(testing::create_random_basis_set(0), std::invalid_argument);

  // Test with valid basis set but empty coefficients
  auto valid_basis = testing::create_random_basis_set(1);
  EXPECT_THROW(Orbitals(empty_coeffs, std::make_optional(empty_energies),
                        std::nullopt, valid_basis, std::nullopt),
               std::runtime_error);
}

TEST_F(OrbitalsEdgeCasesTest, SingleOrbitalSingleBasis) {
  // Test minimal case: 1 atomic orbital, 1 orbital
  Eigen::MatrixXd coeffs(1, 1);
  coeffs(0, 0) = 1.0;
  Eigen::VectorXd energies(1);
  energies(0) = -1.0;

  auto single_basis = testing::create_random_basis_set(1);
  Orbitals orb(coeffs, std::make_optional(energies), std::nullopt, single_basis,
               std::nullopt);

  EXPECT_EQ(1, orb.get_num_atomic_orbitals());
  EXPECT_EQ(1, orb.get_num_molecular_orbitals());

  const auto& [alpha_coeffs, beta_coeffs] = orb.get_coefficients();
  EXPECT_NEAR(1.0, alpha_coeffs(0, 0), testing::numerical_zero_tolerance);
  const auto& [alpha_energies, beta_energies] = orb.get_energies();
  EXPECT_NEAR(-1.0, alpha_energies(0), testing::numerical_zero_tolerance);
}

TEST_F(OrbitalsEdgeCasesTest, AsymmetricDimensions) {
  // Test case: more atomic orbitals than orbitals
  const int n_basis = 10;
  const int n_orbitals = 3;

  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs.setRandom();
  Eigen::VectorXd energies(n_orbitals);
  energies.setRandom();

  auto asym_basis = testing::create_random_basis_set(n_basis);
  Orbitals orb(coeffs, std::make_optional(energies), std::nullopt, asym_basis,
               std::nullopt);

  EXPECT_EQ(n_basis, orb.get_num_atomic_orbitals());
  EXPECT_EQ(n_orbitals, orb.get_num_molecular_orbitals());

  // Test coefficient matrix dimensions
  const auto& [alpha_coeffs, beta_coeffs] = orb.get_coefficients();
  EXPECT_EQ(n_basis, alpha_coeffs.rows());
  EXPECT_EQ(n_orbitals, alpha_coeffs.cols());
}

TEST_F(OrbitalsEdgeCasesTest, ExtremeValues) {
  // Test with very large and very small values
  Eigen::MatrixXd coeffs(2, 2);
  coeffs << 1e-15, 1e15, -1e15, -1e-15;
  Eigen::VectorXd energies(2);
  energies << -1000.0, 1000.0;

  auto extreme_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, std::make_optional(energies), std::nullopt,
               extreme_basis, std::nullopt);

  // Test preservation of extreme values
  const auto& [alpha_coeffs, beta_coeffs] = orb.get_coefficients();
  EXPECT_NEAR(1e-15, alpha_coeffs(0, 0),
              testing::small_value_lower_bound_tolerance);
  EXPECT_NEAR(1e15, alpha_coeffs(0, 1),
              testing::small_value_upper_bound_tolerance);
  const auto& [alpha_energies, beta_energies] = orb.get_energies();
  EXPECT_NEAR(-1000.0, alpha_energies(0), testing::numerical_zero_tolerance);
  EXPECT_NEAR(1000.0, alpha_energies(1), testing::numerical_zero_tolerance);
}

TEST_F(OrbitalsEdgeCasesTest, SpecialMatrices) {
  // Test with identity matrix
  const int n = 4;
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(n, -2.0, 1.0);

  auto special_basis = testing::create_random_basis_set(n);
  Orbitals orb(identity, std::make_optional(energies), std::nullopt,
               special_basis, std::nullopt);

  // Check orthogonality
  const auto& [alpha_coeffs, beta_coeffs] = orb.get_coefficients();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == j) {
        EXPECT_NEAR(1.0, alpha_coeffs(i, j), testing::numerical_zero_tolerance);
      } else {
        EXPECT_NEAR(0.0, alpha_coeffs(i, j), testing::numerical_zero_tolerance);
      }
    }
  }
}

TEST_F(OrbitalsEdgeCasesTest, InconsistentData) {
  // Set coefficients and energies with different dimensions
  Eigen::MatrixXd coeffs(3, 2);
  coeffs.setRandom();
  Eigen::VectorXd energies(3);  // Wrong size!
  energies.setRandom();

  // Constructor should throw or create invalid object when dimensions mismatch
  Eigen::VectorXd occupations(3);  // Wrong size!
  occupations.setRandom();
  auto inconsistent_basis = testing::create_random_basis_set(3);
  EXPECT_THROW(Orbitals(coeffs, std::make_optional(energies), std::nullopt,
                        inconsistent_basis),
               std::runtime_error);

  // Fix the dimensions and create valid object
  Eigen::VectorXd correct_energies(2);
  correct_energies.setRandom();
  Orbitals valid_orb(coeffs, std::make_optional(correct_energies), std::nullopt,
                     inconsistent_basis);
}

TEST_F(OrbitalsEdgeCasesTest, LargeSystemPerformance) {
  // Test performance with larger orbital sets
  const int n_basis = 100;
  const int n_orbitals = 80;

  auto start = std::chrono::high_resolution_clock::now();

  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs.setRandom();
  Eigen::VectorXd energies(n_orbitals);
  energies.setRandom();

  auto large_basis = testing::create_random_basis_set(n_basis);
  Orbitals orb(coeffs, std::make_optional(energies), std::nullopt, large_basis,
               std::nullopt);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  EXPECT_EQ(n_basis, orb.get_num_atomic_orbitals());
  EXPECT_EQ(n_orbitals, orb.get_num_molecular_orbitals());

  // Performance should be reasonable (< 100ms for setup)
  EXPECT_LT(duration.count(), 100);
}

TEST_F(OrbitalsEdgeCasesTest, SerializationEdgeCases) {
  // Test serialization of orbital with special values - avoid inf/nan as they
  // cause issues
  Eigen::MatrixXd coeffs(2, 2);
  coeffs << 0.0, 1e-308, -1e-308, 1.0;  // Use very small but finite values
  Eigen::VectorXd energies(2);
  energies << std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max();

  auto serialization_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, std::make_optional(energies), std::nullopt,
               serialization_basis, std::nullopt);

  // JSON serialization should handle special values appropriately
  auto json_data = orb.to_json();
  EXPECT_FALSE(json_data.empty());

  // Note: Behavior with extreme values depends on JSON library
  // implementation This test mainly ensures no crashes occur
}

TEST_F(OrbitalsEdgeCasesTest, MemoryStress) {
  // Test creation and destruction of many orbital objects
  std::vector<std::unique_ptr<Orbitals>> orbital_objects;

  const int num_objects = 100;
  const int n_basis = 20;
  const int n_orbitals = 15;

  for (int i = 0; i < num_objects; ++i) {
    Eigen::MatrixXd coeffs(n_basis, n_orbitals);
    coeffs.setRandom();
    Eigen::VectorXd energies(n_orbitals);
    energies.setRandom();

    auto memory_basis = testing::create_random_basis_set(n_basis);
    auto orb =
        std::make_unique<Orbitals>(coeffs, std::make_optional(energies),
                                   std::nullopt, memory_basis, std::nullopt);
    orbital_objects.push_back(std::move(orb));
  }

  // Verify all objects are still valid
  for (const auto& orb : orbital_objects) {
    EXPECT_EQ(n_basis, orb->get_num_atomic_orbitals());
    EXPECT_EQ(n_orbitals, orb->get_num_molecular_orbitals());
  }

  // Objects will be automatically destroyed when vector goes out of scope
}

TEST_F(OrbitalsEdgeCasesTest, UnrestrictedEdgeCases) {
  // Test case: different dimensions for alpha and beta (should throw in
  // constructor)
  Eigen::MatrixXd alpha_coeffs(3, 2);
  alpha_coeffs.setRandom();
  Eigen::MatrixXd beta_coeffs(4, 2);  // Different number of atomic orbitals
  beta_coeffs.setRandom();

  // This should be rejected by the constructor
  auto basis = testing::create_random_basis_set(3);
  EXPECT_THROW(Orbitals(alpha_coeffs, beta_coeffs, std::nullopt, std::nullopt,
                        std::nullopt, basis, std::nullopt),
               std::runtime_error);
}

TEST_F(OrbitalsEdgeCasesTest, SpinComponentConsistency) {
  // Test that restricted calculations maintain alpha = beta consistency
  Eigen::MatrixXd coeffs(3, 2);
  coeffs.setRandom();
  Eigen::VectorXd energies(2);
  energies.setRandom();

  auto spin_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, std::make_optional(energies), std::nullopt, spin_basis,
               std::nullopt);

  // For restricted calculation, alpha and beta should be identical
  const auto& [alpha_coeffs, beta_coeffs] = orb.get_coefficients();
  const auto& [alpha_energies, beta_energies] = orb.get_energies();

  EXPECT_TRUE((alpha_coeffs.array() == beta_coeffs.array()).all());
  EXPECT_TRUE((alpha_energies.array() == beta_energies.array()).all());

  EXPECT_TRUE(orb.is_restricted());
}

TEST_F(OrbitalsEdgeCasesTest, EmptySpinChannels) {
  // Test with empty matrices for one spin channel
  Eigen::MatrixXd empty_matrix(0, 0);
  Eigen::VectorXd empty_vector(0);

  // This should either be handled gracefully or throw an exception
  auto basis = testing::create_random_basis_set(3);
  EXPECT_THROW(Orbitals(empty_matrix, empty_matrix, std::nullopt, std::nullopt,
                        std::nullopt, basis, std::nullopt),
               std::runtime_error);
}

TEST_F(OrbitalsEdgeCasesTest, CopyConstructorWithNullPointers) {
  // Test copying completely minimal orbitals
  Eigen::MatrixXd minimal_coeffs(1, 1);
  minimal_coeffs(0, 0) = 1.0;

  auto minimal_basis = testing::create_random_basis_set(1);
  Orbitals orb1(minimal_coeffs, std::nullopt, std::nullopt, minimal_basis,
                std::nullopt);

  // Test copying
  Orbitals orb2(orb1);
  EXPECT_EQ(1, orb2.get_num_atomic_orbitals());
  EXPECT_EQ(1, orb2.get_num_molecular_orbitals());

  // Test copying orbitals with coefficients but missing other data
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  auto copy_basis = testing::create_random_basis_set(2);
  Orbitals orb3(coeffs, std::nullopt, std::nullopt, copy_basis, std::nullopt);

  Orbitals orb4(orb3);
  EXPECT_EQ(2, orb4.get_num_atomic_orbitals());
  EXPECT_EQ(2, orb4.get_num_molecular_orbitals());
  // Should have coefficients but no energies
  const auto& [alpha_coeffs, beta_coeffs] = orb4.get_coefficients();
  EXPECT_TRUE(coeffs.isApprox(alpha_coeffs, testing::numerical_zero_tolerance));
  EXPECT_TRUE(coeffs.isApprox(beta_coeffs, testing::numerical_zero_tolerance));

  // Should throw for missing energies when requested
  EXPECT_THROW(orb4.get_energies(), std::runtime_error);
}

TEST_F(OrbitalsEdgeCasesTest, CopyConstructorUnrestrictedPaths) {
  // Test copying unrestricted calculation
  const int n_basis = 2;
  const int n_orbitals = 2;

  // Set up unrestricted data
  Eigen::MatrixXd alpha_coeffs(n_basis, n_orbitals);
  alpha_coeffs << 0.1, 0.2, 0.3, 0.4;
  Eigen::MatrixXd beta_coeffs(n_basis, n_orbitals);
  beta_coeffs << 0.11, 0.22, 0.33, 0.44;

  Eigen::VectorXd alpha_energies(n_orbitals);
  alpha_energies << -1.0, 0.5;
  Eigen::VectorXd beta_energies(n_orbitals);
  beta_energies << -1.1, 0.4;

  auto unrestricted_basis = testing::create_random_basis_set(n_basis);
  Orbitals orb1(alpha_coeffs, beta_coeffs, std::make_optional(alpha_energies),
                std::make_optional(beta_energies), std::nullopt,
                unrestricted_basis, std::nullopt);

  EXPECT_FALSE(orb1.is_restricted());

  // Copy the unrestricted orbital
  Orbitals orb2(orb1);

  // Verify the copy maintains unrestricted nature
  EXPECT_FALSE(orb2.is_restricted());

  // Verify all data copied correctly
  const auto& [copied_alpha_coeffs, copied_beta_coeffs] =
      orb2.get_coefficients();
  EXPECT_TRUE(alpha_coeffs.isApprox(copied_alpha_coeffs,
                                    testing::numerical_zero_tolerance));
  EXPECT_TRUE(beta_coeffs.isApprox(copied_beta_coeffs,
                                   testing::numerical_zero_tolerance));

  const auto& [copied_alpha_energies, copied_beta_energies] =
      orb2.get_energies();
  EXPECT_TRUE(alpha_energies.isApprox(copied_alpha_energies,
                                      testing::numerical_zero_tolerance));
  EXPECT_TRUE(beta_energies.isApprox(copied_beta_energies,
                                     testing::numerical_zero_tolerance));

  // Verify that modifications to original don't affect copy (deep copy test).
  // Since objects are immutable, we'll create a new object with modified data.
  alpha_coeffs(0, 0) = 999.0;
  Orbitals orb1_modified(alpha_coeffs, beta_coeffs,
                         std::make_optional(alpha_energies),
                         std::make_optional(beta_energies), std::nullopt,
                         unrestricted_basis, std::nullopt);

  const auto& [unchanged_alpha_coeffs, unchanged_beta_coeffs] =
      orb2.get_coefficients();
  EXPECT_NEAR(0.1, unchanged_alpha_coeffs(0, 0),
              testing::numerical_zero_tolerance);  // Should be unchanged
}

TEST_F(OrbitalsEdgeCasesTest, AssignmentOperatorWithNullPointers) {
  // Create minimal orbitals for testing assignment
  Eigen::MatrixXd minimal_coeffs(1, 1);
  minimal_coeffs(0, 0) = 1.0;

  auto assignment_basis = testing::create_random_basis_set(1);
  Orbitals orb1(minimal_coeffs, std::nullopt, std::nullopt, assignment_basis);
  Orbitals orb2(minimal_coeffs, std::nullopt, std::nullopt, assignment_basis);

  // Test self-assignment (should be no-op)
  orb1 = orb1;

  // Test assignment from one minimal orbital to another
  orb2 = orb1;

  // Test assignment with only partial data
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  auto partial_basis = testing::create_random_basis_set(2);
  Orbitals orb3(coeffs, std::nullopt, std::nullopt, partial_basis,
                std::nullopt);

  orb2 = orb3;
  EXPECT_EQ(2, orb2.get_num_atomic_orbitals());
  EXPECT_EQ(2, orb2.get_num_molecular_orbitals());

  // Should have coefficients but missing energies
  const auto& [alpha_coeffs, beta_coeffs] = orb2.get_coefficients();
  EXPECT_TRUE(coeffs.isApprox(alpha_coeffs, testing::numerical_zero_tolerance));
  EXPECT_THROW(orb2.get_energies(), std::runtime_error);
}

TEST_F(OrbitalsEdgeCasesTest, AssignmentOperatorUnrestrictedPaths) {
  // Test assignment with unrestricted calculation
  const int n_basis = 2;
  const int n_orbitals = 2;

  // Set up unrestricted data in orb1
  Eigen::MatrixXd alpha_coeffs(n_basis, n_orbitals);
  alpha_coeffs << 0.1, 0.2, 0.3, 0.4;
  Eigen::MatrixXd beta_coeffs(n_basis, n_orbitals);
  beta_coeffs << 0.15, 0.25, 0.35, 0.45;

  Eigen::VectorXd alpha_energies(n_orbitals);
  alpha_energies << -1.0, 0.5;
  Eigen::VectorXd beta_energies(n_orbitals);
  beta_energies << -1.1, 0.4;

  auto unrestricted_assign_basis = testing::create_random_basis_set(n_basis);
  Orbitals orb1(alpha_coeffs, beta_coeffs, std::make_optional(alpha_energies),
                std::make_optional(beta_energies), std::nullopt,
                unrestricted_assign_basis, std::nullopt);

  EXPECT_FALSE(orb1.is_restricted());

  // Assignment to another orbital
  Eigen::MatrixXd dummy_coeffs(1, 1);
  dummy_coeffs(0, 0) = 1.0;
  auto dummy_basis = testing::create_random_basis_set(1);
  Orbitals orb2(dummy_coeffs, std::nullopt, std::nullopt, dummy_basis);

  orb2 = orb1;

  // Verify the assignment maintained unrestricted nature
  EXPECT_FALSE(orb2.is_restricted());

  // Verify all data copied correctly
  const auto& [assigned_alpha_coeffs, assigned_beta_coeffs] =
      orb2.get_coefficients();
  EXPECT_TRUE(alpha_coeffs.isApprox(assigned_alpha_coeffs,
                                    testing::numerical_zero_tolerance));
  EXPECT_TRUE(beta_coeffs.isApprox(assigned_beta_coeffs,
                                   testing::numerical_zero_tolerance));

  const auto& [assigned_alpha_energies, assigned_beta_energies] =
      orb2.get_energies();
  EXPECT_TRUE(alpha_energies.isApprox(assigned_alpha_energies,
                                      testing::numerical_zero_tolerance));
  EXPECT_TRUE(beta_energies.isApprox(assigned_beta_energies,
                                     testing::numerical_zero_tolerance));

  // Test assignment from unrestricted to restricted (transition coverage)
  Eigen::MatrixXd restricted_coeffs(2, 2);
  restricted_coeffs.setIdentity();
  auto restricted_basis = testing::create_random_basis_set(2);
  Orbitals orb3(restricted_coeffs, std::nullopt, std::nullopt,
                restricted_basis);

  EXPECT_TRUE(orb3.is_restricted());

  // Assign unrestricted to previously restricted
  orb3 = orb1;
  EXPECT_FALSE(orb3.is_restricted());
}

TEST_F(OrbitalsEdgeCasesTest, AssignmentOperatorRestrictedToRestricted) {
  // Test assignment where both source and destination have restricted
  // calculations
  const int n_basis = 2;
  const int n_orbitals = 2;

  // Set up restricted data in orb1
  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs << 0.7, 0.1, 0.1, 0.7;
  Eigen::VectorXd energies(n_orbitals);
  energies << -1.5, 0.3;

  auto restricted_assign_basis = testing::create_random_basis_set(n_basis);
  Orbitals orb1(coeffs, std::make_optional(energies), std::nullopt,
                restricted_assign_basis);

  EXPECT_TRUE(orb1.is_restricted());

  // Create another orbital for assignment
  Eigen::MatrixXd dummy_coeffs(1, 1);
  dummy_coeffs(0, 0) = 1.0;
  auto dummy_restricted_basis = testing::create_random_basis_set(1);
  Orbitals orb2(dummy_coeffs, std::nullopt, std::nullopt,
                dummy_restricted_basis);

  // Assign to another orbital
  orb2 = orb1;

  // Verify restricted nature is preserved
  EXPECT_TRUE(orb2.is_restricted());

  // Verify data copied correctly
  const auto& [alpha_coeffs, beta_coeffs] = orb2.get_coefficients();
  EXPECT_TRUE(coeffs.isApprox(alpha_coeffs, testing::numerical_zero_tolerance));
  EXPECT_TRUE(alpha_coeffs.isApprox(
      beta_coeffs, testing::numerical_zero_tolerance));  // Should be identical
                                                         // for restricted

  const auto& [alpha_energies, beta_energies] = orb2.get_energies();
  EXPECT_TRUE(
      energies.isApprox(alpha_energies, testing::numerical_zero_tolerance));
  EXPECT_TRUE(alpha_energies.isApprox(
      beta_energies,
      testing::numerical_zero_tolerance));  // Should be identical for
                                            // restricted
}

TEST_F(OrbitalsEdgeCasesTest, AssignmentOperatorOptionalComponents) {
  // Test assignment with optional components (AO overlap, basis set)
  const int n_basis = 3;

  // Set up orb1 with all optional components
  Eigen::MatrixXd coeffs(n_basis, 2);
  coeffs.setRandom();

  // Add AO overlap
  Eigen::MatrixXd overlap(n_basis, n_basis);
  overlap.setIdentity();
  overlap(0, 1) = 0.1;
  overlap(1, 0) = 0.1;

  auto optional_basis = testing::create_random_basis_set(n_basis);
  Orbitals orb1(coeffs, std::nullopt, std::make_optional(overlap),
                optional_basis);

  // Create another orbital for testing
  Eigen::MatrixXd dummy_coeffs(1, 1);
  dummy_coeffs(0, 0) = 1.0;
  auto dummy_optional_basis = testing::create_random_basis_set(1);
  Orbitals orb2(dummy_coeffs, std::nullopt, std::nullopt, dummy_optional_basis);

  // Test assignment - should copy overlap
  orb2 = orb1;
  EXPECT_TRUE(orb2.has_overlap_matrix());
  const auto& copied_overlap = orb2.get_overlap_matrix();
  EXPECT_TRUE(
      overlap.isApprox(copied_overlap, testing::numerical_zero_tolerance));

  // Test assignment to orbital that already has overlap
  Eigen::MatrixXd different_overlap(n_basis, n_basis);
  different_overlap.setIdentity();
  different_overlap *= 2.0;
  Orbitals orb3(coeffs, std::nullopt, std::make_optional(different_overlap),
                optional_basis);

  orb3 = orb1;  // Should replace the existing overlap
  EXPECT_TRUE(orb3.has_overlap_matrix());
  const auto& replaced_overlap = orb3.get_overlap_matrix();
  EXPECT_TRUE(
      overlap.isApprox(replaced_overlap, testing::numerical_zero_tolerance));
  EXPECT_FALSE(different_overlap.isApprox(replaced_overlap,
                                          testing::numerical_zero_tolerance));

  // Test assignment from orbital without optional components
  Orbitals orb4(coeffs, std::nullopt, std::nullopt, optional_basis);

  orb2 = orb4;                              // orb2 had overlap, orb4 doesn't
  EXPECT_FALSE(orb2.has_overlap_matrix());  // Should be null now
  EXPECT_TRUE(orb2.has_basis_set());        // Basis set is always present
}

TEST_F(OrbitalsEdgeCasesTest, ValidationEdgeCases) {
  // Test with correct dimensions first
  const int n_basis = 3;
  const int n_orbitals = 2;

  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs.setRandom();
  Eigen::VectorXd energies(n_orbitals);
  energies.setRandom();

  auto validation_basis = testing::create_random_basis_set(n_basis);
  Orbitals valid_orb(coeffs, std::make_optional(energies), std::nullopt,
                     validation_basis);

  // Test with wrong AO overlap dimensions - should throw in constructor
  Eigen::MatrixXd wrong_sized_overlap(n_basis + 1, n_basis + 1);  // Wrong size!
  wrong_sized_overlap.setIdentity();

  EXPECT_THROW(
      Orbitals(coeffs, std::make_optional(energies),
               std::make_optional(wrong_sized_overlap), validation_basis),
      std::runtime_error);

  // Test with non-square AO overlap matrix - should throw in constructor
  Eigen::MatrixXd non_square_overlap(n_basis, n_basis + 1);  // Not square!
  non_square_overlap.setRandom();

  EXPECT_THROW(
      Orbitals(coeffs, std::make_optional(energies),
               std::make_optional(non_square_overlap), validation_basis),
      std::runtime_error);

  // Test with correct AO overlap dimensions
  Eigen::MatrixXd correct_overlap(n_basis, n_basis);
  correct_overlap.setIdentity();
  Orbitals valid_orb_with_overlap(coeffs, std::make_optional(energies),
                                  std::make_optional(correct_overlap),
                                  validation_basis);
}

TEST_F(OrbitalsEdgeCasesTest, FileIOErrorPaths) {
  // Set up valid orbital data
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  auto fileio_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, std::nullopt, std::nullopt, fileio_basis);

  // Test writing to invalid/protected path
  // Try to write to a directory that doesn't exist or is protected
  EXPECT_THROW(orb.to_json_file("/nonexistent_directory/test.orbitals.json"),
               std::runtime_error);

  // Test writing to read-only location (if possible)
  // Note: This might be system-dependent, so we'll use a more portable approach

  // Test reading from non-existent file (already covered in existing tests)
  EXPECT_THROW(
      Orbitals::from_json_file("definitely_nonexistent_file.orbitals.json"),
      std::runtime_error);

  // Test file write error scenarios by creating a valid file, then testing
  // corruption scenarios
  orb.to_json_file("temp_test.orbitals.json");

  // Create a file with invalid JSON to test read error
  {
    std::ofstream corrupt_file("corrupt_test.orbitals.json");
    corrupt_file << "{ invalid json content ";  // Intentionally malformed JSON
    // Don't close properly to potentially cause read issues
  }

  // Note: JSON parsing errors may throw nlohmann::json exceptions or
  // std::runtime_error depending on implementation, both are acceptable error
  // conditions
  EXPECT_THROW(Orbitals::from_json_file("corrupt_test.orbitals.json"),
               std::exception);

  // Clean up test files
  std::filesystem::remove("temp_test.orbitals.json");
  std::filesystem::remove("corrupt_test.orbitals.json");
}

TEST_F(OrbitalsEdgeCasesTest, HDF5BasisSetTemporaryFileOperations) {
  // Set up orbital data
  Eigen::MatrixXd coeffs(3, 2);
  coeffs.setRandom();
  Eigen::VectorXd energies(2);
  energies.setRandom();
  auto hdf5_basis = testing::create_random_basis_set(3);
  Orbitals orb(coeffs, std::make_optional(energies), std::nullopt, hdf5_basis);

  // Test HDF5 save/load operations
  // Create an HDF5 file with basis set data to test loading paths
  orb.to_hdf5_file("test_basis_temp.orbitals.h5");

  auto orb_load = Orbitals::from_hdf5_file("test_basis_temp.orbitals.h5");

  // Verify data loaded correctly
  const auto& [loaded_alpha_coeffs, loaded_beta_coeffs] =
      orb_load->get_coefficients();
  EXPECT_TRUE(
      coeffs.isApprox(loaded_alpha_coeffs, testing::numerical_zero_tolerance));

  const auto& [loaded_alpha_energies, loaded_beta_energies] =
      orb_load->get_energies();
  EXPECT_TRUE(energies.isApprox(loaded_alpha_energies,
                                testing::numerical_zero_tolerance));

  // Test temporary file cleanup scenarios
  // The temporary files should be automatically cleaned up
  EXPECT_FALSE(std::filesystem::exists("temp_basis_load.h5"));
  EXPECT_FALSE(std::filesystem::exists("temp_basis_save.h5"));

  // Clean up test file
  std::filesystem::remove("test_basis_temp.orbitals.h5");
}

TEST_F(OrbitalsEdgeCasesTest, HDF5ExceptionHandling) {
  // Set up valid orbital data
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  auto hdf5_exception_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, std::nullopt, std::nullopt, hdf5_exception_basis);

  // Test filename validation (throws std::invalid_argument)
  EXPECT_THROW(orb.to_hdf5_file("/invalid_path/nonexistent_dir/test.h5"),
               std::invalid_argument);

  // Test HDF5 exception handling during save operations
  // Use a valid filename but invalid path to trigger actual HDF5 errors
  EXPECT_THROW(orb.to_hdf5_file("/nonexistent_dir/test.orbitals.h5"),
               std::runtime_error);

  // Test HDF5 exception handling during load operations
  // Try to read a corrupted HDF5 file with proper naming
  {
    std::ofstream corrupt_hdf5("corrupt_test.orbitals.h5");
    corrupt_hdf5 << "This is not a valid HDF5 file content";
  }

  EXPECT_THROW(Orbitals::from_hdf5_file("corrupt_test.orbitals.h5"),
               std::runtime_error);

  // Clean up
  std::filesystem::remove("corrupt_test.orbitals.h5");
}

TEST_F(OrbitalsEdgeCasesTest, HDF5DatasetExistenceChecks) {
  // Set up minimal orbital data (only coefficients)
  Eigen::MatrixXd coeffs(2, 2);
  coeffs.setIdentity();
  auto hdf5_dataset_basis = testing::create_random_basis_set(2);
  Orbitals orb(coeffs, std::nullopt, std::nullopt, hdf5_dataset_basis);

  // Save minimal orbital (no energies, occupations, ao_overlap, basis_set)
  orb.to_hdf5_file("minimal_test.orbitals.h5");

  // Load and verify that missing datasets are handled gracefully
  auto orb_minimal = Orbitals::from_hdf5_file("minimal_test.orbitals.h5");

  // Should have coefficients and occupations (occupations are always saved)
  const auto& [alpha_coeffs, beta_coeffs] = orb_minimal->get_coefficients();
  EXPECT_TRUE(coeffs.isApprox(alpha_coeffs, testing::numerical_zero_tolerance));

  // Should throw for missing energies only
  EXPECT_THROW(orb_minimal->get_energies(), std::runtime_error);

  // Should not have optional components
  EXPECT_FALSE(orb_minimal->has_overlap_matrix());
  EXPECT_TRUE(orb_minimal->has_basis_set());

  // Clean up
  std::filesystem::remove("minimal_test.orbitals.h5");
}

TEST_F(OrbitalsEdgeCasesTest, ValidationInconsistentUnrestrictedData) {
  // Test validation with unrestricted data where alpha and beta have different
  // sizes - should throw in constructor
  const int n_basis = 3;
  const int n_orbitals_alpha = 2;
  const int n_orbitals_beta = 3;  // Different size

  Eigen::MatrixXd alpha_coeffs(n_basis, n_orbitals_alpha);
  alpha_coeffs.setRandom();
  Eigen::MatrixXd beta_coeffs(n_basis, n_orbitals_beta);
  beta_coeffs.setRandom();

  // Should be invalid due to dimension mismatch between alpha and beta
  auto basis = testing::create_random_basis_set(n_basis);
  EXPECT_THROW(Orbitals(alpha_coeffs, beta_coeffs, std::nullopt, std::nullopt,
                        std::nullopt, basis),
               std::runtime_error);
}

TEST_F(OrbitalsEdgeCasesTest, JSONDeserializationValidationErrors) {
  // Test missing coefficient data error
  {
    nlohmann::json j_missing_coeffs = {
        {"is_restricted", true}
        // Missing coefficients section
    };
    EXPECT_THROW(Orbitals::from_json(j_missing_coeffs), std::runtime_error);
  }

  {
    nlohmann::json j_missing_alpha = {{"is_restricted", true},
                                      {"coefficients",
                                       {// Missing alpha coefficients
                                        {"beta", {{1.0, 0.0}, {0.0, 1.0}}}}}};
    EXPECT_THROW(Orbitals::from_json(j_missing_alpha), std::runtime_error);
  }

  {
    nlohmann::json j_missing_beta = {{"is_restricted", true},
                                     {"coefficients",
                                      {
                                          {"alpha", {{1.0, 0.0}, {0.0, 1.0}}}
                                          // Missing beta coefficients
                                      }}};
    EXPECT_THROW(Orbitals::from_json(j_missing_beta), std::runtime_error);
  }

  // Test missing beta energies for unrestricted calculation
  {
    nlohmann::json j_missing_beta_energies = {
        {"is_restricted", false},
        {"coefficients",
         {{"alpha", {{1.0, 0.0}, {0.0, 1.0}}},
          {"beta", {{0.9, 0.1}, {0.1, 0.9}}}}},
        {"energies",
         {
             {"alpha", {-1.0, 0.5}}
             // Missing beta energies for unrestricted calculation
         }}};
    EXPECT_THROW(Orbitals::from_json(j_missing_beta_energies),
                 std::runtime_error);
  }
}

TEST_F(OrbitalsEdgeCasesTest, JSONParsingExceptionHandling) {
  // Test JSON parsing exception handling
  nlohmann::json j_invalid_structure = {
      {"is_restricted", true},
      {"coefficients",
       {{"alpha", "invalid_matrix_data"},  // Invalid data type
        {"beta", {{1.0, 0.0}, {0.0, 1.0}}}}}};

  // This should trigger JSON parsing errors
  EXPECT_THROW(Orbitals::from_json(j_invalid_structure), std::runtime_error);

  // Test with malformed JSON structure
  nlohmann::json j_type_mismatch = {
      {"is_restricted", "not_a_boolean"},  // Wrong type
      {"coefficients",
       {{"alpha", {{1.0, 0.0}, {0.0, 1.0}}},
        {"beta", {{1.0, 0.0}, {0.0, 1.0}}}}}};

  EXPECT_THROW(Orbitals::from_json(j_type_mismatch), std::runtime_error);
}
