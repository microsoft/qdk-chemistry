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
