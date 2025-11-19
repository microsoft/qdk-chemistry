// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/coupled_cluster.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class CoupledClusterAmplitudesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up basic orbital data
    const int n_basis = 4;
    const int n_orbitals = 3;

    Eigen::MatrixXd coeffs(n_basis, n_orbitals);
    coeffs << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2;

    Eigen::VectorXd energies(n_orbitals);
    energies << -1.0, -0.5, 0.2;

    // Set canonical occupations (0, 1, or 2 for restricted orbitals)
    Eigen::VectorXd occ(n_orbitals);
    occ << 2.0, 2.0,
        0.0;  // First two orbitals doubly occupied, last one virtual

    // Create orbitals with proper constructor (restricted)
    auto basis = testing::create_random_basis_set(3);
    orbitals = std::make_shared<Orbitals>(coeffs, std::make_optional(energies),
                                          std::nullopt, basis, std::nullopt);

    // Set up amplitudes for tests

    // T1 size should be no * nv = 2 * 1 = 2
    t1_amplitudes = Eigen::VectorXd(2);
    t1_amplitudes << 0.01, 0.02;

    // T2 size should be (no * nv)^2 = (2*1)^2 = 4
    t2_amplitudes = Eigen::VectorXd(4);
    for (int i = 0; i < 4; i++) {
      t2_amplitudes[i] = 0.001 * (i + 1);
    }
  }

  std::shared_ptr<Orbitals> orbitals;
  Eigen::VectorXd t1_amplitudes;
  Eigen::VectorXd t2_amplitudes;
};

TEST_F(CoupledClusterAmplitudesTest, Constructors) {
  // Default constructor
  CoupledClusterAmplitudes cc1;
  EXPECT_FALSE(cc1.has_t1_amplitudes());
  EXPECT_FALSE(cc1.has_t2_amplitudes());

  // Constructor with parameters
  CoupledClusterAmplitudes cc2(orbitals, t1_amplitudes, t2_amplitudes, 2, 2);
  EXPECT_TRUE(cc2.has_t1_amplitudes());
  EXPECT_TRUE(cc2.has_t2_amplitudes());

  // Copy constructor
  CoupledClusterAmplitudes cc3(cc2);
  EXPECT_TRUE(cc3.has_t1_amplitudes());
  EXPECT_TRUE(cc3.has_t2_amplitudes());

  // Check amplitudes were correctly copied
  EXPECT_TRUE(cc3.get_t1_amplitudes().isApprox(t1_amplitudes));
  EXPECT_TRUE(cc3.get_t2_amplitudes().isApprox(t2_amplitudes));
}

TEST_F(CoupledClusterAmplitudesTest, CopyAssignment) {
  CoupledClusterAmplitudes cc1(orbitals, t1_amplitudes, t2_amplitudes, 2, 2);

  // Test copy assignment
  CoupledClusterAmplitudes cc2;
  cc2 = cc1;

  EXPECT_TRUE(cc2.has_t1_amplitudes());
  EXPECT_TRUE(cc2.has_t2_amplitudes());
  EXPECT_TRUE(cc2.get_t1_amplitudes().isApprox(cc1.get_t1_amplitudes()));
  EXPECT_TRUE(cc2.get_t2_amplitudes().isApprox(cc1.get_t2_amplitudes()));
}

TEST_F(CoupledClusterAmplitudesTest, AmplitudeManagement) {
  CoupledClusterAmplitudes cc;

  // Test accessing amplitudes that aren't set
  EXPECT_FALSE(cc.has_t1_amplitudes());
  EXPECT_FALSE(cc.has_t2_amplitudes());
  EXPECT_THROW(cc.get_t1_amplitudes(), std::runtime_error);
  EXPECT_THROW(cc.get_t2_amplitudes(), std::runtime_error);

  // Set amplitudes via constructor
  CoupledClusterAmplitudes cc2(orbitals, t1_amplitudes, t2_amplitudes, 2, 2);

  // Test accessing amplitudes
  EXPECT_TRUE(cc2.has_t1_amplitudes());
  EXPECT_TRUE(cc2.has_t2_amplitudes());
  EXPECT_TRUE(cc2.get_t1_amplitudes().isApprox(t1_amplitudes));
  EXPECT_TRUE(cc2.get_t2_amplitudes().isApprox(t2_amplitudes));
}

TEST_F(CoupledClusterAmplitudesTest, CanonicalOrbitalsRequirement) {
  // Create orbitals with non-canonical occupations (fractional values)
  const int n_basis = 4;
  const int n_orbitals = 3;

  Eigen::MatrixXd coeffs(n_basis, n_orbitals);
  coeffs << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2;

  Eigen::VectorXd energies(n_orbitals);
  energies << -1.0, -0.5, 0.2;

  auto basis = testing::create_random_basis_set(3);
  auto non_canonical_orbitals = std::make_shared<Orbitals>(
      coeffs, std::make_optional(energies), std::nullopt, basis, std::nullopt);

  // Test that we can construct CoupledClusterAmplitudes with these orbitals
  EXPECT_NO_THROW(CoupledClusterAmplitudes cc(
      non_canonical_orbitals, t1_amplitudes, t2_amplitudes, 2, 2));

  // With canonical orbitals, it should work fine
  EXPECT_NO_THROW(CoupledClusterAmplitudes cc(orbitals, t1_amplitudes,
                                              t2_amplitudes, 2, 2));
}

TEST_F(CoupledClusterAmplitudesTest, OrbitalCounts) {
  // Create coupled cluster object with canonical orbitals
  CoupledClusterAmplitudes cc(orbitals, t1_amplitudes, t2_amplitudes, 2, 2);

  // Check that occupied counts are stored and accessible
  const auto [alpha_occ_count, beta_occ_count] = cc.get_num_occupied();
  EXPECT_EQ(alpha_occ_count, 2);  // First two orbitals occupied for alpha
  EXPECT_EQ(beta_occ_count,
            2);  // First two orbitals occupied for beta

  // Check that virtual counts are stored and accessible
  const auto [alpha_virt_count, beta_virt_count] = cc.get_num_virtual();
  EXPECT_EQ(alpha_virt_count, 1);  // Last orbital virtual for alpha
  EXPECT_EQ(beta_virt_count,
            1);  // Last orbital virtual for beta
}

TEST_F(CoupledClusterAmplitudesTest, ValidateIndicesAndEnergies) {
  // Test case 1: Non-adjacent indices
  // Create orbitals with a gap in orbital indices
  Eigen::VectorXd energies(4);
  energies << -1.0, -0.5, 0.2, 0.5;

  Eigen::MatrixXd coeffs(4, 4);
  coeffs << 0.1, 0.2, 0.0, 0.3, 0.4, 0.5, 0.0, 0.6, 0.7, 0.8, 0.0, 0.9, 1.0,
      1.1, 0.0, 1.2;

  Eigen::VectorXd occ(4);
  occ << 1.0, 1.0, 0.0,
      1.0;  // indices 0, 1, 3 occupied (not adjacent to virtual)

  auto basis = testing::create_random_basis_set(4);
  auto modified_orbitals = std::make_shared<Orbitals>(
      coeffs, std::make_optional(energies), std::nullopt, basis, std::nullopt);

  // Should throw due to non-adjacent indices
  EXPECT_THROW(CoupledClusterAmplitudes(modified_orbitals, t1_amplitudes,
                                        t2_amplitudes, 3, 3),
               std::invalid_argument);
}

TEST_F(CoupledClusterAmplitudesTest, ValidateAmplitudeDimensions) {
  // Test case 1: Invalid T1 amplitudes dimension
  // For restricted case with 2 occupied and 1 virtual orbitals
  // T1 size should be no * nv = 2 * 1 = 2
  Eigen::VectorXd invalid_t1(3);  // Wrong size (should be 2)
  for (int i = 0; i < 3; i++) {
    invalid_t1[i] = 0.01 * (i + 1);
  }

  EXPECT_THROW(
      CoupledClusterAmplitudes(orbitals, invalid_t1, t2_amplitudes, 2, 2),
      std::invalid_argument);

  // Test case 2: Invalid T2 amplitudes dimension
  // For restricted case with 2 occupied and 1 virtual orbitals
  // T2 size should be (no * nv)² = (2 * 1)² = 4
  Eigen::VectorXd invalid_t2(5);  // Wrong size (should be 4)
  for (int i = 0; i < 5; i++) {
    invalid_t2[i] = 0.001 * (i + 1);
  }

  EXPECT_THROW(
      CoupledClusterAmplitudes(orbitals, t1_amplitudes, invalid_t2, 2, 2),
      std::invalid_argument);

  // Test case 3: Both T1 and T2 with correct dimensions should work
  EXPECT_NO_THROW(
      CoupledClusterAmplitudes(orbitals, t1_amplitudes, t2_amplitudes, 2, 2));

  // Test case 4: Zero-length amplitudes (edge case)
  Eigen::VectorXd zero_length_vector(0);

  EXPECT_THROW(CoupledClusterAmplitudes(orbitals, zero_length_vector,
                                        t2_amplitudes, 2, 2),
               std::invalid_argument);

  EXPECT_THROW(CoupledClusterAmplitudes(orbitals, t1_amplitudes,
                                        zero_length_vector, 2, 2),
               std::invalid_argument);
}
