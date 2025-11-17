// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class OrbitalsEdgeCasesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean up any test files from previous runs
    std::filesystem::remove("test.orbitals.json");
    std::filesystem::remove("test.orbitals.h5");
    std::filesystem::remove("large.orbitals.h5");
  }

  void TearDown() override {
    // Clean up test files
    std::filesystem::remove("test.orbitals.json");
    std::filesystem::remove("test.orbitals.h5");
    std::filesystem::remove("large.orbitals.h5");
  }
};

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
  // Test minimal case: 1 basis function, 1 orbital
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
  // Test case: more basis functions than orbitals
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
  Eigen::MatrixXd beta_coeffs(4, 2);  // Different number of basis functions
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
  EXPECT_TRUE(coeffs.isApprox(alpha_coeffs, 1e-12));
  EXPECT_TRUE(coeffs.isApprox(beta_coeffs, 1e-12));

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
  EXPECT_TRUE(alpha_coeffs.isApprox(copied_alpha_coeffs, 1e-12));
  EXPECT_TRUE(beta_coeffs.isApprox(copied_beta_coeffs, 1e-12));

  const auto& [copied_alpha_energies, copied_beta_energies] =
      orb2.get_energies();
  EXPECT_TRUE(alpha_energies.isApprox(copied_alpha_energies, 1e-12));
  EXPECT_TRUE(beta_energies.isApprox(copied_beta_energies, 1e-12));

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
  EXPECT_TRUE(coeffs.isApprox(alpha_coeffs, 1e-12));
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
  EXPECT_TRUE(alpha_coeffs.isApprox(assigned_alpha_coeffs, 1e-12));
  EXPECT_TRUE(beta_coeffs.isApprox(assigned_beta_coeffs, 1e-12));

  const auto& [assigned_alpha_energies, assigned_beta_energies] =
      orb2.get_energies();
  EXPECT_TRUE(alpha_energies.isApprox(assigned_alpha_energies, 1e-12));
  EXPECT_TRUE(beta_energies.isApprox(assigned_beta_energies, 1e-12));

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
  EXPECT_TRUE(coeffs.isApprox(alpha_coeffs, 1e-12));
  EXPECT_TRUE(alpha_coeffs.isApprox(
      beta_coeffs, 1e-12));  // Should be identical for restricted

  const auto& [alpha_energies, beta_energies] = orb2.get_energies();
  EXPECT_TRUE(energies.isApprox(alpha_energies, 1e-12));
  EXPECT_TRUE(alpha_energies.isApprox(
      beta_energies, 1e-12));  // Should be identical for restricted
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
  EXPECT_TRUE(overlap.isApprox(copied_overlap, 1e-12));

  // Test assignment to orbital that already has overlap
  Eigen::MatrixXd different_overlap(n_basis, n_basis);
  different_overlap.setIdentity();
  different_overlap *= 2.0;
  Orbitals orb3(coeffs, std::nullopt, std::make_optional(different_overlap),
                optional_basis);

  orb3 = orb1;  // Should replace the existing overlap
  EXPECT_TRUE(orb3.has_overlap_matrix());
  const auto& replaced_overlap = orb3.get_overlap_matrix();
  EXPECT_TRUE(overlap.isApprox(replaced_overlap, 1e-12));
  EXPECT_FALSE(different_overlap.isApprox(replaced_overlap, 1e-12));

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
  EXPECT_TRUE(coeffs.isApprox(loaded_alpha_coeffs, 1e-12));

  const auto& [loaded_alpha_energies, loaded_beta_energies] =
      orb_load->get_energies();
  EXPECT_TRUE(energies.isApprox(loaded_alpha_energies, 1e-12));

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
  EXPECT_TRUE(coeffs.isApprox(alpha_coeffs, 1e-12));

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
