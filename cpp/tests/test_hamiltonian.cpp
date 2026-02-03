// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/density_fitted.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <string>

#include "../src/qdk/chemistry/algorithms/microsoft/mp2.hpp"
#include "ut_common.hpp"
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class HamiltonianTest : public ::testing::TestWithParam<std::string> {
 protected:
  void SetUp() override {
    // Create test data
    one_body = Eigen::MatrixXd::Identity(2, 2);
    one_body(0, 1) = 0.5;
    one_body(1, 0) = 0.5;

    two_body = 2 * Eigen::VectorXd::Ones(16);

    // Create a test Orbitals object using ModelOrbitals for model systems
    orbitals =
        std::make_shared<ModelOrbitals>(2, true);  // 2 orbitals, restricted

    num_electrons = 2;
    core_energy = 1.5;

    // Create inactive Fock matrix (empty for full space systems)
    inactive_fock = Eigen::MatrixXd::Zero(0, 0);

    // active space case:
    // inactive orbitals: 0, 1; active orbitals: 2, 3
    orbitals_with_inactive = std::make_shared<ModelOrbitals>(
        4,
        std::make_tuple(std::vector<size_t>{2, 3}, std::vector<size_t>{0, 1}));

    inactive_fock_non_empty = Eigen::MatrixXd::Random(2, 2);

    // For density-fitted: 3-center integrals [n_aux x n_orb^2]
    // Using 3 auxiliary basis functions for 2 orbitals (4 geminals)
    // These numbers are selected because they correspond to two_body that is
    // used for canonical four-center tests
    three_center = (Eigen::MatrixXd(3, 4) << 1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.6,
                    0.6, 0.8, 0.8, 0.8, 0.8)
                       .finished();

    container_type = GetParam();

    sample_n_aux = 3;

    sample_one_body_alpha = Eigen::MatrixXd::Identity(2, 2);
    sample_one_body_beta = Eigen::MatrixXd::Ones(2, 2);

    sample_two_body_aaaa = Eigen::VectorXd::Constant(16, 1.0);
    sample_two_body_aabb = Eigen::VectorXd::Constant(16, sqrt(3));
    sample_two_body_bbbb = Eigen::VectorXd::Constant(16, 3.0);

    // These numbers are selected because they correspond to
    // sample_two_body_xxxx that is used for canonical four-center tests
    sample_three_center_aa = sqrt(0.5) * three_center;
    sample_three_center_bb = sqrt(1.5) * three_center;

    sample_inactive_fock_alpha = Eigen::MatrixXd::Constant(2, 2, 4.0);
    sample_inactive_fock_beta = Eigen::MatrixXd::Constant(2, 2, 5.0);

    orbitals_unrestricted = std::make_shared<ModelOrbitals>(2, false);

    // Create restricted Hamiltonian with the parameterized container type
    hamiltonian_restricted = createHamiltonian(container_type);

    // Create unrestricted Hamiltonian
    hamiltonian_unrestricted = createUnrestrictedHamiltonian(container_type);
  }

  void TearDown() override {
    // Clean up any test files
    std::filesystem::remove("test.hamiltonian.json");
    std::filesystem::remove("test.hamiltonian.h5");
    std::filesystem::remove("test.hamiltonian.fcidump");
  }

  /**
   * @brief Factory method to create a Hamiltonian with the specified
   * container type.
   */
  std::shared_ptr<Hamiltonian> createHamiltonian(const std::string& type) {
    if (type == "canonical_four_center") {
      return std::make_shared<Hamiltonian>(
          std::make_unique<CanonicalFourCenterHamiltonianContainer>(
              one_body, two_body, orbitals, core_energy, inactive_fock));
    } else if (type == "density_fitted") {
      return std::make_shared<Hamiltonian>(
          std::make_unique<DensityFittedHamiltonianContainer>(
              one_body, three_center, orbitals, core_energy, inactive_fock));
    }
    throw std::runtime_error("Unknown container type: " + type);
  }

  /**
   * @brief Create an unrestricted Hamiltonian with the parameterized
   * container type.
   */
  std::shared_ptr<Hamiltonian> createUnrestrictedHamiltonian(
      const std::string& type) {
    if (type == "canonical_four_center") {
      return std::make_shared<Hamiltonian>(
          std::make_unique<CanonicalFourCenterHamiltonianContainer>(
              sample_one_body_alpha, sample_one_body_beta, sample_two_body_aaaa,
              sample_two_body_aabb, sample_two_body_bbbb, orbitals_unrestricted,
              core_energy, sample_inactive_fock_alpha,
              sample_inactive_fock_beta));
    } else if (type == "density_fitted") {
      return std::make_shared<Hamiltonian>(
          std::make_unique<DensityFittedHamiltonianContainer>(
              sample_one_body_alpha, sample_one_body_beta,
              sample_three_center_aa, sample_three_center_bb,
              orbitals_unrestricted, core_energy, sample_inactive_fock_alpha,
              sample_inactive_fock_beta));
    }
    throw std::runtime_error("Unknown container type: " + type);
  }

  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;
  Eigen::MatrixXd three_center;
  std::shared_ptr<Orbitals> orbitals;
  std::shared_ptr<Orbitals> orbitals_with_inactive;
  std::shared_ptr<Orbitals> orbitals_unrestricted;
  unsigned num_electrons;
  unsigned sample_n_aux;
  double core_energy;
  Eigen::MatrixXd inactive_fock;
  Eigen::MatrixXd inactive_fock_non_empty;

  Eigen::MatrixXd sample_one_body_alpha;
  Eigen::MatrixXd sample_one_body_beta;
  Eigen::VectorXd sample_two_body_aaaa;
  Eigen::VectorXd sample_two_body_aabb;
  Eigen::VectorXd sample_two_body_bbbb;
  Eigen::MatrixXd sample_three_center_aa;
  Eigen::MatrixXd sample_three_center_bb;
  Eigen::MatrixXd sample_inactive_fock_alpha;
  Eigen::MatrixXd sample_inactive_fock_beta;

  std::string container_type;
  std::shared_ptr<Hamiltonian> hamiltonian_restricted;
  std::shared_ptr<Hamiltonian> hamiltonian_unrestricted;
};

class HamiltonianConstructorTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class TestHamiltonianConstructor : public HamiltonianConstructor {
 public:
  std::string name() const override { return "test-hamiltonian_constructor"; }
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Orbitals> orbitals,
      OptionalAuxBasis aux_basis) const override {
    // Dummy implementation for testing
    Eigen::MatrixXd one_body = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd two_body = Eigen::VectorXd::Random(81);
    Eigen::MatrixXd f_inact = Eigen::MatrixXd::Identity(0, 0);
    return std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, orbitals, 0.0, f_inact));
  }
};

TEST_P(HamiltonianTest, Constructor) {
  // Test the constructor with all required data

  EXPECT_TRUE(hamiltonian_restricted->has_one_body_integrals());
  EXPECT_TRUE(hamiltonian_restricted->has_two_body_integrals());
  EXPECT_TRUE(hamiltonian_restricted->has_orbitals());
  EXPECT_EQ(
      hamiltonian_restricted->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(hamiltonian_restricted->get_core_energy(), 1.5);
  EXPECT_EQ(hamiltonian_restricted->get_container_type(), container_type);
  EXPECT_TRUE(hamiltonian_restricted->is_restricted());
  EXPECT_FALSE(hamiltonian_restricted->is_unrestricted());
  EXPECT_TRUE(hamiltonian_restricted->is_hermitian());

  EXPECT_TRUE(hamiltonian_unrestricted->has_one_body_integrals());
  EXPECT_TRUE(hamiltonian_unrestricted->has_two_body_integrals());
  EXPECT_TRUE(hamiltonian_unrestricted->has_orbitals());
  EXPECT_EQ(
      hamiltonian_unrestricted->get_orbitals()->get_num_molecular_orbitals(),
      2);
  EXPECT_EQ(hamiltonian_unrestricted->get_core_energy(), 1.5);
  EXPECT_EQ(hamiltonian_unrestricted->get_container_type(), container_type);
  EXPECT_FALSE(hamiltonian_unrestricted->is_restricted());
  EXPECT_TRUE(hamiltonian_unrestricted->is_unrestricted());
  EXPECT_TRUE(hamiltonian_unrestricted->is_hermitian());
}

TEST_P(HamiltonianTest, ConstructorWithInactiveFock) {
  // Test the constructor with inactive fock matrix
  // For this test specifically, create ModelOrbitals with inactive space

  std::string test_p = GetParam();

  // Create Hamiltonian using parameterized container type
  std::shared_ptr<Hamiltonian> h_active_space;
  if (test_p == "canonical_four_center") {
    h_active_space = std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, orbitals_with_inactive, core_energy,
            inactive_fock_non_empty));
  } else if (test_p == "density_fitted") {
    // For density fitted, need 3-center integrals
    Eigen::MatrixXd three_center_2x2 = Eigen::MatrixXd::Random(3, 4);
    h_active_space = std::make_shared<Hamiltonian>(
        std::make_unique<DensityFittedHamiltonianContainer>(
            one_body, three_center, orbitals_with_inactive, core_energy,
            inactive_fock_non_empty));
  }

  EXPECT_TRUE(h_active_space->has_one_body_integrals());
  EXPECT_TRUE(h_active_space->has_two_body_integrals());
  EXPECT_TRUE(h_active_space->has_orbitals());
  EXPECT_TRUE(h_active_space->has_inactive_fock_matrix());
  EXPECT_EQ(h_active_space->get_orbitals()->get_num_molecular_orbitals(), 4);
  EXPECT_EQ(h_active_space->get_core_energy(), 1.5);
}

TEST_P(HamiltonianTest, MoveConstructor) {
  Hamiltonian h2(std::move(*hamiltonian_unrestricted));

  EXPECT_TRUE(h2.has_one_body_integrals());
  EXPECT_TRUE(h2.has_two_body_integrals());
  EXPECT_TRUE(h2.has_orbitals());
  EXPECT_EQ(h2.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2.get_core_energy(), 1.5);
}

TEST_P(HamiltonianTest, CopyConstructorAndAssignment) {
  std::string test_p = GetParam();

  std::shared_ptr<Hamiltonian> h1;

  Eigen::MatrixXd inactive_fock = Eigen::MatrixXd::Random(2, 2);
  if (test_p == "canonical_four_center") {
    h1 = std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, orbitals, core_energy, inactive_fock));
  } else if (test_p == "density_fitted") {
    h1 = std::make_shared<Hamiltonian>(
        std::make_unique<DensityFittedHamiltonianContainer>(
            one_body, three_center, orbitals, core_energy, inactive_fock));
  }

  Hamiltonian h2(*h1);

  // Verify all data was copied correctly
  EXPECT_TRUE(h2.has_one_body_integrals());
  EXPECT_TRUE(h2.has_two_body_integrals());
  EXPECT_TRUE(h2.has_orbitals());
  EXPECT_TRUE(h2.has_inactive_fock_matrix());
  EXPECT_EQ(h2.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2.get_core_energy(), 1.5);
  EXPECT_TRUE(h2.is_restricted());

  // Verify one body integral copy
  auto [h1_one_alpha, h1_one_beta] = h1->get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2.get_one_body_integrals();
  EXPECT_TRUE(h1_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h1_one_beta.isApprox(h2_one_beta));

  // Compare each component of the two-body integrals tuple
  auto [h1_two_aaaa, h1_two_aabb, h1_two_bbbb] = h1->get_two_body_integrals();
  auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] = h2.get_two_body_integrals();
  EXPECT_TRUE(h1_two_aaaa.isApprox(h2_two_aaaa));
  EXPECT_TRUE(h1_two_aabb.isApprox(h2_two_aabb));
  EXPECT_TRUE(h1_two_bbbb.isApprox(h2_two_bbbb));
  EXPECT_TRUE(h1->get_inactive_fock_matrix().first.isApprox(
      h2.get_inactive_fock_matrix().first));

  // Test copy assignment using createHamiltonian with test parameter
  auto h3 = createHamiltonian(test_p);
  *h3 = *h1;

  // Verify assignment worked correctly
  EXPECT_TRUE(h3->has_one_body_integrals());
  EXPECT_TRUE(h3->has_two_body_integrals());
  EXPECT_TRUE(h3->has_orbitals());
  EXPECT_TRUE(h3->has_inactive_fock_matrix());
  EXPECT_EQ(h3->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h3->get_core_energy(), 1.5);
  EXPECT_TRUE(h3->is_restricted());

  // Test self-assignment (should be no-op)
  auto h4 = createHamiltonian(test_p);
  Hamiltonian* h4_ptr = h4.get();
  *h4 = *h4_ptr;  // Self-assignment

  // Should remain unchanged
  EXPECT_TRUE(h4->has_one_body_integrals());
  EXPECT_TRUE(h4->has_two_body_integrals());
  EXPECT_TRUE(h4->has_orbitals());
  EXPECT_EQ(h4->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h4->get_core_energy(), 1.5);

  // unrestricted Hamiltonian
  Hamiltonian h5(*hamiltonian_unrestricted);

  // Verify all data was copied correctly
  EXPECT_TRUE(h5.has_one_body_integrals());
  EXPECT_TRUE(h5.has_two_body_integrals());
  EXPECT_TRUE(h5.has_orbitals());
  EXPECT_TRUE(h5.has_inactive_fock_matrix());
  EXPECT_EQ(h5.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h5.get_core_energy(), 1.5);
  EXPECT_FALSE(h5.is_restricted());

  // Verify one body integral copy
  auto [hu_one_alpha, hu_one_beta] =
      hamiltonian_unrestricted->get_one_body_integrals();
  auto [h5_one_alpha, h5_one_beta] = h5.get_one_body_integrals();
  EXPECT_TRUE(hu_one_alpha.isApprox(h5_one_alpha));
  EXPECT_TRUE(hu_one_beta.isApprox(h5_one_beta));

  // Compare each component of the two-body integrals tuple
  auto [hu_two_aaaa, hu_two_aabb, hu_two_bbbb] =
      hamiltonian_unrestricted->get_two_body_integrals();
  auto [h5_two_aaaa, h5_two_aabb, h5_two_bbbb] = h5.get_two_body_integrals();
  EXPECT_TRUE(hu_two_aaaa.isApprox(h5_two_aaaa));
  EXPECT_TRUE(hu_two_aabb.isApprox(h5_two_aabb));
  EXPECT_TRUE(hu_two_bbbb.isApprox(h5_two_bbbb));
  EXPECT_TRUE(
      hamiltonian_unrestricted->get_inactive_fock_matrix().first.isApprox(
          h5.get_inactive_fock_matrix().first));
}

TEST_P(HamiltonianTest, TwoBodyElementAccess) {
  std::string test_p = GetParam();

  if (test_p == "canonical_four_center") {
    // Create a Hamiltonian with known two-body integrals
    Eigen::MatrixXd test_one_body = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd test_two_body = Eigen::VectorXd::Zero(16);  // 2^4 = 16

    // Set specific values we can test - these indices test the
    // get_two_body_index function
    test_two_body[0] = 1.0;   // (0,0,0,0) -> index 0*8 + 0*4 + 0*2 + 0 = 0
    test_two_body[1] = 2.0;   // (0,0,0,1) -> index 0*8 + 0*4 + 0*2 + 1 = 1
    test_two_body[5] = 3.0;   // (0,1,0,1) -> index 0*8 + 1*4 + 0*2 + 1 = 5
    test_two_body[15] = 4.0;  // (1,1,1,1) -> index 1*8 + 1*4 + 1*2 + 1 = 15
    test_two_body[10] = 5.0;  // (1,0,1,0) -> index 1*8 + 0*4 + 1*2 + 0 = 10
    test_two_body[7] = 6.0;   // (0,1,1,1) -> index 0*8 + 1*4 + 1*2 + 1 = 7

    Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
        test_one_body, test_two_body, orbitals, core_energy, inactive_fock));

    EXPECT_DOUBLE_EQ(h.get_one_body_element(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(1, 1), 1.0);

    // Test accessing specific elements to verify get_two_body_index
    // calculations
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0), 1.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 1), 2.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 1, 0, 1), 3.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 1, 1, 1), 4.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 0, 1, 0), 5.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 1, 1, 1), 6.0);

    // Test elements that should be zero
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 1, 0), 0.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 0, 0, 0), 0.0);

    // Test out-of-range access - this tests bounds checking in
    // get_two_body_index
    EXPECT_THROW(h.get_two_body_element(2, 0, 0, 0), std::out_of_range);
    EXPECT_THROW(h.get_two_body_element(0, 2, 0, 0), std::out_of_range);
    EXPECT_THROW(h.get_two_body_element(0, 0, 2, 0), std::out_of_range);
    EXPECT_THROW(h.get_two_body_element(0, 0, 0, 2), std::out_of_range);

    // Test with larger system to verify get_two_body_index scaling
    Eigen::MatrixXd large_inact_f = Eigen::MatrixXd::Identity(0, 0);
    Eigen::MatrixXd large_one_body = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd large_two_body = Eigen::VectorXd::Zero(81);  // 3^4 = 81

    // Test specific indices: (2,1,0,2) should give index 2*27 + 1*9 + 0*3 + 2 =
    // 54 + 9 + 0 + 2 = 65
    large_two_body[65] = 7.0;
    // Test (1,2,2,1) should give index 1*27 + 2*9 + 2*3 + 1 = 27 + 18 + 6 + 1 =
    // 52
    large_two_body[52] = 8.0;

    // Create orbitals for the larger system
    auto large_orbitals =
        std::make_shared<ModelOrbitals>(3, true);  // 3 orbitals, restricted

    Hamiltonian h_large(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            large_one_body, large_two_body, large_orbitals, 0.0,
            large_inact_f));

    EXPECT_DOUBLE_EQ(h_large.get_two_body_element(2, 1, 0, 2), 7.0);
    EXPECT_DOUBLE_EQ(h_large.get_two_body_element(1, 2, 2, 1), 8.0);
  } else if (test_p == "density_fitted") {
    Eigen::MatrixXd test_one_body = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd test_three_center = Eigen::MatrixXd::Zero(3, 4);

    // Set specific three center index values
    test_three_center(0, 0) = 1.0;  // (1, 0, 4, 0)
    test_three_center(1, 0) = 2.0;  // (2, 5, 0, 0)
    test_three_center(2, 0) = 3.0;  // (3, 0, 0, 6)
    test_three_center(0, 2) = 4.0;  //
    test_three_center(1, 1) = 5.0;  //
    test_three_center(2, 3) = 6.0;  //

    Hamiltonian h(std::make_unique<DensityFittedHamiltonianContainer>(
        test_one_body, test_three_center, orbitals, core_energy,
        inactive_fock));

    EXPECT_DOUBLE_EQ(h.get_one_body_element(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(1, 1), 1.0);

    // Test three center integral
    auto [three_c_aa, three_c_bb] =
        h.get_container<DensityFittedHamiltonianContainer>()
            .get_three_center_integrals();

    EXPECT_TRUE(three_c_aa.isApprox(test_three_center));
    EXPECT_TRUE(three_c_bb.isApprox(test_three_center));

    // Test accessing specific elements to verify get_two_body_index
    // calculations
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0), 14.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 1), 10.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 1, 0), 4.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 1, 0, 1), 25.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 0, 0, 0), 4.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 0, 1, 0), 16.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 1, 1, 1), 36.0);

    // Test elements that should be zero
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 1, 1, 1), 0.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 0, 0, 1), 0.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 1, 1, 0), 0.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 1, 0, 1), 0.0);

    // Test out-of-range access - this tests bounds checking in
    // get_two_body_index
    EXPECT_THROW(h.get_two_body_element(2, 0, 0, 0), std::out_of_range);
    EXPECT_THROW(h.get_two_body_element(0, 2, 0, 0), std::out_of_range);
    EXPECT_THROW(h.get_two_body_element(0, 0, 2, 0), std::out_of_range);
    EXPECT_THROW(h.get_two_body_element(0, 0, 0, 2), std::out_of_range);

    // Test with larger system to verify get_two_body_index scaling
    Eigen::MatrixXd large_inact_f = Eigen::MatrixXd::Identity(0, 0);
    Eigen::MatrixXd large_one_body = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd large_three_center =
        Eigen::MatrixXd::Zero(81, 9);  // 3^2 = 9

    // Test specific indices: (aux,1,2) should give index 1*3 + 2 =
    large_three_center(78, 5) = 7.0;
    // Test (aux,1,0) should give index 1*3 + 0 = 27 + 18 + 6 + 1 =
    // 52
    large_three_center(52, 3) = 8.0;

    // Create orbitals for the larger system
    auto large_orbitals =
        std::make_shared<ModelOrbitals>(3, true);  // 3 orbitals, restricted

    Hamiltonian h_large(std::make_unique<DensityFittedHamiltonianContainer>(
        large_one_body, large_three_center, large_orbitals, 0.0,
        large_inact_f));

    EXPECT_DOUBLE_EQ(h_large.get_two_body_element(1, 2, 1, 2), 49.0);
    EXPECT_DOUBLE_EQ(h_large.get_two_body_element(1, 0, 1, 0), 64.0);
  }
}

TEST_P(HamiltonianTest, JSONSerialization) {
  // Test JSON conversion using pre-built hamiltonian_restricted
  nlohmann::json j = hamiltonian_restricted->to_json();

  EXPECT_EQ(j["container"]["core_energy"], 1.5);
  EXPECT_TRUE(j["container"]["has_one_body_integrals"]);
  EXPECT_TRUE(j["container"]["has_two_body_integrals"]);
  EXPECT_TRUE(j["container"]["has_orbitals"]);

  // Test round-trip conversion
  auto h2 = Hamiltonian::from_json(j);

  EXPECT_EQ(h2->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2->get_core_energy(), 1.5);
  EXPECT_TRUE(h2->has_one_body_integrals());
  EXPECT_TRUE(h2->has_two_body_integrals());
  EXPECT_TRUE(h2->has_orbitals());

  // Check one body
  auto [h_one_alpha, h_one_beta] =
      hamiltonian_restricted->get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h_one_beta.isApprox(h2_one_beta));

  // Check two body
  auto [h_two_aaaa, h_two_aabb, h_two_bbbb] =
      hamiltonian_restricted->get_two_body_integrals();
  auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] = h2->get_two_body_integrals();
  EXPECT_TRUE(h_two_aaaa.isApprox(h2_two_aaaa));
  EXPECT_TRUE(h_two_aabb.isApprox(h2_two_aabb));
  EXPECT_TRUE(h_two_bbbb.isApprox(h2_two_bbbb));

  // Check they are still restricted
  EXPECT_TRUE(h2->is_restricted());
  EXPECT_FALSE(h2->is_unrestricted());
}

TEST_P(HamiltonianTest, JSONFileIO) {
  // Test file I/O using pre-built hamiltonian_restricted
  std::string filename = "test.hamiltonian.json";
  hamiltonian_restricted->to_json_file(filename);
  EXPECT_TRUE(std::filesystem::exists(filename));

  // Load from file
  auto h2 = Hamiltonian::from_json_file(filename);

  // Check loaded data
  EXPECT_EQ(h2->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2->get_core_energy(), 1.5);
  EXPECT_TRUE(h2->has_one_body_integrals());
  EXPECT_TRUE(h2->has_two_body_integrals());
  EXPECT_TRUE(h2->has_orbitals());

  // Check that matrices are approximately equal
  auto [h_one_alpha, h_one_beta] =
      hamiltonian_restricted->get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h_one_beta.isApprox(h2_one_beta));

  auto [h_two_aaaa, h_two_aabb, h_two_bbbb] =
      hamiltonian_restricted->get_two_body_integrals();
  auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] = h2->get_two_body_integrals();
  EXPECT_TRUE(h_two_aaaa.isApprox(h2_two_aaaa));
  EXPECT_TRUE(h_two_aabb.isApprox(h2_two_aabb));
  EXPECT_TRUE(h_two_bbbb.isApprox(h2_two_bbbb));

  // Check still restricted
  EXPECT_TRUE(h2->is_restricted());
  EXPECT_FALSE(h2->is_unrestricted());
}

TEST_P(HamiltonianTest, HDF5FileIO) {
  // Test file I/O using pre-built hamiltonian_restricted
  std::string filename = "test.hamiltonian.h5";
  hamiltonian_restricted->to_hdf5_file(filename);
  EXPECT_TRUE(std::filesystem::exists(filename));

  // Load from file
  auto h2 = Hamiltonian::from_hdf5_file(filename);

  // Check loaded data
  EXPECT_EQ(h2->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2->get_core_energy(), 1.5);
  EXPECT_TRUE(h2->has_one_body_integrals());
  EXPECT_TRUE(h2->has_two_body_integrals());
  EXPECT_TRUE(h2->has_orbitals());

  // Check that matrices are approximately equal
  auto [h_one_alpha, h_one_beta] =
      hamiltonian_restricted->get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h_one_beta.isApprox(h2_one_beta));

  auto [h_two_aaaa, h_two_aabb, h_two_bbbb] =
      hamiltonian_restricted->get_two_body_integrals();
  auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] = h2->get_two_body_integrals();
  EXPECT_TRUE(h_two_aaaa.isApprox(h2_two_aaaa));
  EXPECT_TRUE(h_two_aabb.isApprox(h2_two_aabb));
  EXPECT_TRUE(h_two_bbbb.isApprox(h2_two_bbbb));
}

TEST_P(HamiltonianTest, GenericFileIO) {
  // Test JSON via generic interface using pre-built hamiltonian_restricted
  std::string json_filename = "test.hamiltonian.json";
  hamiltonian_restricted->to_file(json_filename, "json");
  EXPECT_TRUE(std::filesystem::exists(json_filename));

  auto h2 = Hamiltonian::from_file(json_filename, "json");

  EXPECT_EQ(h2->get_orbitals()->get_num_molecular_orbitals(), 2);
  auto [h_one_alpha, h_one_beta] =
      hamiltonian_restricted->get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));

  // Test HDF5 via generic interface
  std::string hdf5_filename = "test.hamiltonian.h5";
  hamiltonian_restricted->to_file(hdf5_filename, "hdf5");
  EXPECT_TRUE(std::filesystem::exists(hdf5_filename));

  auto h3 = Hamiltonian::from_file(hdf5_filename, "hdf5");

  EXPECT_EQ(h3->get_orbitals()->get_num_molecular_orbitals(), 2);
  auto [h3_one_alpha, h3_one_beta] = h3->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h3_one_alpha));
}

TEST_P(HamiltonianTest, InvalidFileType) {
  // Test using pre-built hamiltonian_restricted
  EXPECT_THROW(hamiltonian_restricted->to_file("test.txt", "txt"),
               std::runtime_error);
  EXPECT_THROW(Hamiltonian::from_file("test.txt", "txt"), std::runtime_error);
}

TEST_P(HamiltonianTest, FileNotFound) {
  EXPECT_THROW(Hamiltonian::from_json_file("nonexistent.hamiltonian.json"),
               std::runtime_error);
  EXPECT_THROW(Hamiltonian::from_hdf5_file("nonexistent.hamiltonian.h5"),
               std::runtime_error);
}

TEST_P(HamiltonianTest, ValidationTests) {
  // Test validation of integral dimensions during construction
  std::string test_p = GetParam();

  if (test_p == "canonical_four_center") {
    // Mismatched dimensions should throw during construction
    Eigen::MatrixXd bad_one_body = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd bad_two_body =
        Eigen::VectorXd::Random(16);  // Should be 81 for 3x3

    EXPECT_THROW(
        Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            bad_one_body, bad_two_body, orbitals, core_energy, inactive_fock)),
        std::invalid_argument);

    // Test validation with non-square one-body matrix
    Eigen::MatrixXd non_square_one_body(2, 3);  // 2x3 non-square matrix
    non_square_one_body.setRandom();
    Eigen::VectorXd any_two_body = Eigen::VectorXd::Random(36);

    EXPECT_THROW(
        Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            non_square_one_body, any_two_body, orbitals, core_energy,
            inactive_fock)),
        std::invalid_argument);

    // Test validation passes with correct dimensions
    Eigen::MatrixXd correct_one_body = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd correct_two_body = Eigen::VectorXd::Random(16);  // 2^4 = 16

    EXPECT_NO_THROW(
        Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            correct_one_body, correct_two_body, orbitals, core_energy,
            inactive_fock)));
  } else if (test_p == "density_fitted") {
    // Mismatched dimensions should throw during construction
    Eigen::MatrixXd bad_one_body = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd bad_two_body = Eigen::MatrixXd::Random(
        5, 6);  // The number of columns hould be 9 for 3x3

    EXPECT_THROW(
        Hamiltonian(std::make_unique<DensityFittedHamiltonianContainer>(
            bad_one_body, bad_two_body, orbitals, core_energy, inactive_fock)),
        std::invalid_argument);

    // Test validation with non-square one-body matrix
    Eigen::MatrixXd non_square_one_body(2, 3);  // 2x3 non-square matrix
    non_square_one_body.setRandom();
    Eigen::MatrixXd any_two_body = Eigen::MatrixXd::Random(9, 9);

    EXPECT_THROW(
        Hamiltonian(std::make_unique<DensityFittedHamiltonianContainer>(
            non_square_one_body, any_two_body, orbitals, core_energy,
            inactive_fock)),
        std::invalid_argument);

    // Test validation passes with correct dimensions
    Eigen::MatrixXd correct_one_body = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd correct_two_body =
        Eigen::MatrixXd::Random(9, 4);  // 2*2 = 4
    EXPECT_NO_THROW(
        Hamiltonian(std::make_unique<DensityFittedHamiltonianContainer>(
            correct_one_body, correct_two_body, orbitals, core_energy,
            inactive_fock)));
  }
}

TEST_P(HamiltonianTest, ValidationEdgeCases) {
  // Test edge cases for validation during construction
  std::string test_p = GetParam();

  // Test with 1x1 matrices (smallest valid case)
  Eigen::MatrixXd tiny_one_body = Eigen::MatrixXd::Identity(1, 1);
  Eigen::VectorXd tiny_two_body = Eigen::VectorXd::Random(1);  // 1^4 = 1
  Eigen::MatrixXd tiny_three_center = Eigen::MatrixXd::Identity(1, 1);
  auto tiny_orbitals =
      std::make_shared<ModelOrbitals>(1, true);  // 1 orbital, restricted
  Eigen::MatrixXd tiny_inactive_fock = Eigen::MatrixXd::Zero(1, 1);

  if (test_p == "canonical_four_center") {
    EXPECT_NO_THROW(
        Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            tiny_one_body, tiny_two_body, tiny_orbitals, core_energy,
            tiny_inactive_fock)));
  } else if (test_p == "density_fitted") {
    EXPECT_NO_THROW(
        Hamiltonian(std::make_unique<DensityFittedHamiltonianContainer>(
            tiny_one_body, tiny_three_center, tiny_orbitals, core_energy,
            tiny_inactive_fock)));
  }

  // Test with large matrices (stress test)
  Eigen::MatrixXd large_one_body = Eigen::MatrixXd::Identity(10, 10);
  Eigen::VectorXd large_two_body =
      Eigen::VectorXd::Random(10000);  // 10^4 = 10000
  Eigen::MatrixXd large_three_center = Eigen::MatrixXd::Random(1000, 100);

  // Need orbitals that match the 10x10 size
  Eigen::MatrixXd large_coeffs = Eigen::MatrixXd::Identity(10, 10);

  auto large_orbitals =
      std::make_shared<ModelOrbitals>(10, true);  // 10 orbitals, restricted

  // Create a larger inactive_fock matrix for this test
  Eigen::MatrixXd large_inactive_fock = Eigen::MatrixXd::Zero(0, 0);

  if (test_p == "canonical_four_center") {
    EXPECT_NO_THROW(
        Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            large_one_body, large_two_body, large_orbitals, core_energy,
            large_inactive_fock)));
  } else if (test_p == "density_fitted") {
    EXPECT_NO_THROW(
        Hamiltonian(std::make_unique<DensityFittedHamiltonianContainer>(
            large_one_body, large_three_center, large_orbitals, core_energy,
            large_inactive_fock)));
  }

  // Test wrong size by one element
  Eigen::MatrixXd three_by_three = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd off_by_one_4c =
      Eigen::VectorXd::Random(80);  // Should be 81 for 3x3
  Eigen::MatrixXd off_by_one_3c = Eigen::MatrixXd::Random(1000, 8);

  if (test_p == "canonical_four_center") {
    EXPECT_THROW(
        Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            three_by_three, off_by_one_4c, orbitals, core_energy,
            inactive_fock)),
        std::invalid_argument);
  } else if (test_p == "density_fitted") {
    EXPECT_THROW(
        Hamiltonian(std::make_unique<DensityFittedHamiltonianContainer>(
            three_by_three, off_by_one_3c, orbitals, core_energy,
            inactive_fock)),
        std::invalid_argument);
  }
}

TEST_P(HamiltonianTest, UnrestrictedConstructor) {
  // Verify the unrestricted Hamiltonian was created successfully using
  // pre-built hamiltonian_unrestricted
  EXPECT_TRUE(hamiltonian_unrestricted->has_one_body_integrals());
  EXPECT_TRUE(hamiltonian_unrestricted->has_two_body_integrals());
  EXPECT_TRUE(hamiltonian_unrestricted->has_orbitals());
  EXPECT_TRUE(hamiltonian_unrestricted->has_inactive_fock_matrix());
  EXPECT_EQ(hamiltonian_unrestricted->get_core_energy(), core_energy);
  EXPECT_FALSE(hamiltonian_unrestricted->is_restricted());
  EXPECT_TRUE(hamiltonian_unrestricted->is_unrestricted());
}

TEST_P(HamiltonianTest, UnrestrictedAccessorMethods) {
  std::string test_p = GetParam();

  // Test alpha/beta one-body integral access using pre-built
  // hamiltonian_unrestricted
  auto [h_one_alpha, h_one_beta] =
      hamiltonian_unrestricted->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(sample_one_body_alpha));
  EXPECT_TRUE(h_one_beta.isApprox(sample_one_body_beta));

  // Test tuple access for two-body integrals
  auto [aaaa, aabb, bbbb] = hamiltonian_unrestricted->get_two_body_integrals();
  EXPECT_TRUE(aaaa.isApprox(sample_two_body_aaaa));
  EXPECT_TRUE(aabb.isApprox(sample_two_body_aabb));
  EXPECT_TRUE(bbbb.isApprox(sample_two_body_bbbb));
  // Test alpha/beta inactive Fock matrix access
  auto fock_matrices = hamiltonian_unrestricted->get_inactive_fock_matrix();
  EXPECT_TRUE(fock_matrices.first.isApprox(sample_inactive_fock_alpha));
  EXPECT_TRUE(fock_matrices.second.isApprox(sample_inactive_fock_beta));
}

TEST_P(HamiltonianTest, RestrictedVsUnrestrictedDetection) {
  // Test restricted detection using pre-built hamiltonian_restricted
  EXPECT_TRUE(hamiltonian_restricted->is_restricted());
  EXPECT_FALSE(hamiltonian_restricted->is_unrestricted());

  // Test unrestricted detection using pre-built hamiltonian_unrestricted
  EXPECT_FALSE(hamiltonian_unrestricted->is_restricted());
  EXPECT_TRUE(hamiltonian_unrestricted->is_unrestricted());
}

TEST_P(HamiltonianTest, UnrestrictedSpinChannelAccess) {
  // Create unrestricted orbitals for this test
  std::string test_p = GetParam();

  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(2, false);

  // Create unrestricted Hamiltonian with specific two-body integral values
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Identity(2, 2);

  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Zero(16);
  Eigen::VectorXd two_body_aabb = Eigen::VectorXd::Zero(16);
  Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Zero(16);

  Eigen::MatrixXd three_center_aa = Eigen::MatrixXd::Zero(3, 4);
  Eigen::MatrixXd three_center_bb = Eigen::MatrixXd::Zero(3, 4);

  // canonical four center case
  two_body_aaaa[0] = 1.0;   // (0,0,0,0) in aaaa channel
  two_body_aabb[5] = 2.0;   // (0,1,0,1) in aabb channel
  two_body_bbbb[15] = 3.0;  // (1,1,1,1) in bbbb channel

  // three center case
  // (a,a,a,a) (a,a,b,b)
  // (0,1,0,0) (0,0,0,0)
  // (0,0,0,0) (0,0,2,0)
  // (0,0,0,0) (0,0,0,0)
  three_center_aa(1, 0) = 1.0;
  three_center_bb(1, 2) = 2.0;

  Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);

  if (test_p == "canonical_four_center") {
    Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
        one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
        two_body_bbbb, unrestricted_orbitals, core_energy, empty_fock,
        empty_fock));

    EXPECT_DOUBLE_EQ(h.get_one_body_element(0, 0, SpinChannel::aa), 1.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(0, 1, SpinChannel::aa), 0.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(1, 1, SpinChannel::bb), 1.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(1, 0, SpinChannel::bb), 0.0);

    // Test accessing elements through different spin channels
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::aaaa),
                     1.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 1, 0, 1, SpinChannel::aabb),
                     2.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 1, 1, 1, SpinChannel::bbbb),
                     3.0);

    // Verify other elements are zero
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::aabb),
                     0.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::bbbb),
                     0.0);
  } else if (test_p == "density_fitted") {
    Hamiltonian h(std::make_unique<DensityFittedHamiltonianContainer>(
        one_body_alpha, one_body_beta, three_center_aa, three_center_bb,
        unrestricted_orbitals, core_energy, empty_fock, empty_fock));

    EXPECT_DOUBLE_EQ(h.get_one_body_element(0, 0, SpinChannel::aa), 1.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(0, 1, SpinChannel::aa), 0.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(1, 1, SpinChannel::bb), 1.0);
    EXPECT_DOUBLE_EQ(h.get_one_body_element(1, 0, SpinChannel::bb), 0.0);

    // Test accessing elements through different spin channels
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::aaaa),
                     1.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 1, 0, SpinChannel::aabb),
                     2.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 0, 1, 0, SpinChannel::bbbb),
                     4.0);

    // Verify other elements are zero
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::aabb),
                     0.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::aabb),
                     0.0);
    EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::bbbb),
                     0.0);
  }
}

TEST_P(HamiltonianTest, UnrestrictedJSONSerialization) {
  // Test JSON serialization round-trip using pre-built hamiltonian_unrestricted
  nlohmann::json j = hamiltonian_unrestricted->to_json();
  auto h_loaded = Hamiltonian::from_json(j);

  // Verify the loaded Hamiltonian matches the original
  EXPECT_EQ(h_loaded->get_core_energy(), core_energy);
  EXPECT_FALSE(h_loaded->is_restricted());
  EXPECT_TRUE(h_loaded->is_unrestricted());

  auto [orig_one_alpha, orig_one_beta] =
      hamiltonian_unrestricted->get_one_body_integrals();
  auto [loaded_one_alpha, loaded_one_beta] = h_loaded->get_one_body_integrals();
  EXPECT_TRUE(orig_one_alpha.isApprox(loaded_one_alpha));
  EXPECT_TRUE(orig_one_beta.isApprox(loaded_one_beta));

  auto [orig_two_aaaa, orig_two_aabb, orig_two_bbbb] =
      hamiltonian_unrestricted->get_two_body_integrals();
  auto [loaded_two_aaaa, loaded_two_aabb, loaded_two_bbbb] =
      h_loaded->get_two_body_integrals();
  EXPECT_TRUE(orig_two_aaaa.isApprox(loaded_two_aaaa));
  EXPECT_TRUE(orig_two_aabb.isApprox(loaded_two_aabb));
  EXPECT_TRUE(orig_two_bbbb.isApprox(loaded_two_bbbb));

  auto [h_orig_alpha, h_orig_beta] =
      hamiltonian_unrestricted->get_inactive_fock_matrix();
  auto [h_loaded_alpha, h_loaded_beta] = h_loaded->get_inactive_fock_matrix();
  EXPECT_TRUE(h_orig_alpha.isApprox(h_loaded_alpha));
  EXPECT_TRUE(h_orig_beta.isApprox(h_loaded_beta));
}

TEST_P(HamiltonianTest, UnrestrictedHDF5Serialization) {
  // Test HDF5 serialization round-trip using pre-built hamiltonian_unrestricted
  std::string filename = "test_unrestricted.hamiltonian.h5";
  hamiltonian_unrestricted->to_hdf5_file(filename);

  auto h_loaded = Hamiltonian::from_hdf5_file(filename);

  // Verify the loaded Hamiltonian matches the original
  EXPECT_EQ(h_loaded->get_core_energy(), core_energy);
  EXPECT_FALSE(h_loaded->is_restricted());
  EXPECT_TRUE(h_loaded->is_unrestricted());

  auto [orig_one_alpha, orig_one_beta] =
      hamiltonian_unrestricted->get_one_body_integrals();
  auto [loaded_one_alpha, loaded_one_beta] = h_loaded->get_one_body_integrals();
  EXPECT_TRUE(orig_one_alpha.isApprox(loaded_one_alpha));
  EXPECT_TRUE(orig_one_beta.isApprox(loaded_one_beta));

  auto [orig_two_aaaa, orig_two_aabb, orig_two_bbbb] =
      hamiltonian_unrestricted->get_two_body_integrals();
  auto [loaded_two_aaaa, loaded_two_aabb, loaded_two_bbbb] =
      h_loaded->get_two_body_integrals();
  EXPECT_TRUE(orig_two_aaaa.isApprox(loaded_two_aaaa));
  EXPECT_TRUE(orig_two_aabb.isApprox(loaded_two_aabb));
  EXPECT_TRUE(orig_two_bbbb.isApprox(loaded_two_bbbb));

  auto [h_orig_alpha, h_orig_beta] =
      hamiltonian_unrestricted->get_inactive_fock_matrix();
  auto [h_loaded_alpha, h_loaded_beta] = h_loaded->get_inactive_fock_matrix();
  EXPECT_TRUE(h_orig_alpha.isApprox(h_loaded_alpha));
  EXPECT_TRUE(h_orig_beta.isApprox(h_loaded_beta));

  // Clean up test file
  std::filesystem::remove(filename);
}

TEST_P(HamiltonianTest, FCIDUMPSerialization) {
  // Test FCIDUMP serialization using pre-built hamiltonian_restricted
  hamiltonian_restricted->to_fcidump_file("test.hamiltonian.fcidump", 1, 1);

  std::ifstream file("test.hamiltonian.fcidump");
  EXPECT_TRUE(file.is_open());

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string fcidump_content = buffer.str();

  // Check that the file matches the reference
  const std::string reference_fcidump_contents =
      "&FCI NORB=2, NELEC=2, MS2=0,\n"
      "ORBSYM=1,1,\n"
      "ISYM=1,\n"
      "&END\n"
      "      2.0000000000000000e+00    1    1    1    1\n"
      "      2.0000000000000000e+00    1    1    1    2\n"
      "      2.0000000000000000e+00    1    1    2    2\n"
      "      2.0000000000000000e+00    1    2    1    2\n"
      "      2.0000000000000000e+00    1    2    2    2\n"
      "      2.0000000000000000e+00    2    2    2    2\n"
      "      1.0000000000000000e+00    1    1    0    0\n"
      "      5.0000000000000000e-01    2    1    0    0\n"
      "      1.0000000000000000e+00    2    2    0    0\n"
      "      1.5000000000000000e+00    0    0    0    0";

  EXPECT_TRUE(fcidump_content == reference_fcidump_contents);
}

TEST_P(HamiltonianTest, FCIDUMPSerializationUnrestrictedThrowsError) {
  // Verify hamiltonian_unrestricted is actually unrestricted
  EXPECT_TRUE(hamiltonian_unrestricted->is_unrestricted());
  EXPECT_FALSE(hamiltonian_unrestricted->is_restricted());

  // Test that FCIDUMP serialization throws an error for unrestricted case
  EXPECT_THROW(hamiltonian_unrestricted->to_fcidump_file(
                   "test_unrestricted.hamiltonian.fcidump", 1, 1),
               std::runtime_error);
}

TEST_P(HamiltonianTest, FCIDUMPActiveSpaceConsistency) {
  std::string test_p = GetParam();

  // Create Hamiltonian using parameterized container type
  std::shared_ptr<Hamiltonian> h_active_space;
  if (test_p == "canonical_four_center") {
    h_active_space = std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, orbitals_with_inactive, core_energy,
            inactive_fock_non_empty));
  } else if (test_p == "density_fitted") {
    // For density fitted, need 3-center integrals
    Eigen::MatrixXd three_center_2x2 = Eigen::MatrixXd::Random(3, 4);
    h_active_space = std::make_shared<Hamiltonian>(
        std::make_unique<DensityFittedHamiltonianContainer>(
            one_body, three_center, orbitals_with_inactive, core_energy,
            inactive_fock_non_empty));
  }

  // Should successfully write FCIDUMP using active space dimensions
  EXPECT_NO_THROW({
    h_active_space->to_fcidump_file("test_active_space.hamiltonian.fcidump", 1,
                                    1);
  });

  // Verify file was created and has correct NORB (should be 2, not 3)
  std::ifstream file("test_active_space.hamiltonian.fcidump");
  EXPECT_TRUE(file.is_open());

  std::string first_line;
  std::getline(file, first_line);
  EXPECT_TRUE(first_line.find("NORB=2") != std::string::npos);

  // Clean up
  std::filesystem::remove("test_active_space.hamiltonian.fcidump");
}

TEST_P(HamiltonianTest, ErrorHandlingUnrestrictedMismatchedActiveSpace) {
  // Test error handling when alpha and beta active spaces have different sizes
  std::string test_p = GetParam();

  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(3, false);

  // Manually set different active space sizes for alpha and beta
  std::vector<size_t> alpha_active = {0, 1};  // 2 orbitals
  std::vector<size_t> beta_active = {0, 1,
                                     2};  // 3 orbitals - should cause error

  // Create matrices with mismatched dimensions
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Identity(3, 3);

  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Ones(16);  // 2^4
  Eigen::VectorXd two_body_aabb =
      Eigen::VectorXd::Ones(81);  // 3^4 - mismatched
  Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Ones(81);  // 3^4

  Eigen::MatrixXd three_center_aa = Eigen::MatrixXd::Random(3, 4);  // 2^4
  Eigen::MatrixXd three_center_bb =
      Eigen::MatrixXd::Random(3, 4);  //  - mismatched
  Eigen::MatrixXd three_center_bb1 = Eigen::MatrixXd::Random(3, 9);  // 3^4
  Eigen::MatrixXd three_center_bb2 =
      Eigen::MatrixXd::Random(4, 4);  // - mismatch

  Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);

  // This should throw during construction due to dimension mismatch
  if (test_p == "canonical_four_center") {
    EXPECT_THROW(
        {
          Hamiltonian h_mismatched(
              std::make_unique<CanonicalFourCenterHamiltonianContainer>(
                  one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
                  two_body_bbbb, unrestricted_orbitals, core_energy, empty_fock,
                  empty_fock));
        },
        std::invalid_argument);
  } else if (test_p == "density_fitted") {
    EXPECT_THROW(
        {
          // geminal dimension mismatch
          Hamiltonian h_mismatched(
              std::make_unique<DensityFittedHamiltonianContainer>(
                  one_body_alpha, one_body_beta, three_center_aa,
                  three_center_bb1, unrestricted_orbitals, core_energy,
                  empty_fock, empty_fock));
        },
        std::invalid_argument);
    EXPECT_THROW(
        {
          // aux basis number mismatch
          Hamiltonian h_mismatched(
              std::make_unique<DensityFittedHamiltonianContainer>(
                  one_body_alpha, one_body_beta, three_center_aa,
                  three_center_bb2, unrestricted_orbitals, core_energy,
                  empty_fock, empty_fock));
        },
        std::invalid_argument);
  }
}

TEST_P(HamiltonianTest, GetContainerTypedAccess) {
  std::string test_p = GetParam();

  // Test that get_container_type() returns the correct parameterized type
  EXPECT_EQ(hamiltonian_restricted->get_container_type(), test_p);

  // Test has_container_type for all known container types
  // The current container type should return true, others should return false
  bool is_canonical =
      hamiltonian_restricted
          ->has_container_type<CanonicalFourCenterHamiltonianContainer>();
  bool is_density_fitted =
      hamiltonian_restricted
          ->has_container_type<DensityFittedHamiltonianContainer>();

  // Exactly one should be true
  EXPECT_EQ(is_canonical, test_p == "canonical_four_center");
  EXPECT_EQ(is_density_fitted, test_p == "density_fitted");

  // Test that accessing with incorrect container type throws std::bad_cast
  // We test against all OTHER container types
  if (test_p != "canonical_four_center") {
    EXPECT_THROW(hamiltonian_restricted
                     ->get_container<CanonicalFourCenterHamiltonianContainer>(),
                 std::bad_cast);
  }
  if (test_p != "density_fitted") {
    EXPECT_THROW(hamiltonian_restricted
                     ->get_container<DensityFittedHamiltonianContainer>(),
                 std::bad_cast);
  }
}

TEST_P(HamiltonianTest, DataTypeName) {
  // Test that Hamiltonian has the correct data type name
  EXPECT_EQ(hamiltonian_restricted->get_data_type_name(), "hamiltonian");
}

// ============================================================================
// Integration Tests for Hamiltonian with real molecular calculations
// ============================================================================

// Helper lambda to run restricted O2 calculation
auto run_restricted_o2 = [](const std::string& factory_name = "qdk") {
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};

  // debugging with H2 molecule instead of O2
  //    std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0,
  //    0.0),
  //                                                Eigen::Vector3d(0.0,
  //                                                0.0, 1.0)};
  //    std::vector<std::string> symbols = {"H", "H"};
  Structure o2_structure(coordinates, symbols);

  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");

  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [rhf_energy, rhf_wavefunction] =
      scf_factory->run(o2_structure_ptr, 0, 1, "cc-pvdz");
  // scf_factory->run(o2_structure_ptr, 0, 1, "sto-3g");
  auto rhf_orbitals = rhf_wavefunction->get_orbitals();

  auto ham_factory = HamiltonianConstructorFactory::create(factory_name);

  auto rhf_hamiltonian =
      (factory_name == "qdk_density_fitted")
          ? ham_factory->run(rhf_orbitals, std::string("cc-pvdz-rifit"))
          : ham_factory->run(rhf_orbitals);

  return std::make_tuple(rhf_energy, rhf_hamiltonian, rhf_wavefunction);
};

// Helper lambda to run unrestricted O2 triplet calculation
auto run_unrestricted_o2 = [](const std::string& factory_name = "qdk") {
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");

  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [uhf_energy, uhf_wavefunction] =
      scf_factory->run(o2_structure_ptr, 0, 3, "cc-pvdz");
  auto uhf_orbitals = uhf_wavefunction->get_orbitals();

  auto ham_factory = HamiltonianConstructorFactory::create(factory_name);

  auto uhf_hamiltonian =
      (factory_name == "qdk_density_fitted")
          ? ham_factory->run(uhf_orbitals, std::string("cc-pvdz-rifit"))
          : ham_factory->run(uhf_orbitals);

  return std::make_tuple(uhf_energy, uhf_hamiltonian);
};

TEST_F(HamiltonianConstructorTest, Factory) {
  auto available_solvers = HamiltonianConstructorFactory::available();
  EXPECT_EQ(available_solvers.size(), 2);
  EXPECT_EQ(available_solvers[1], "qdk");
  EXPECT_EQ(available_solvers[0], "qdk_density_fitted");
  EXPECT_THROW(HamiltonianConstructorFactory::create("nonexistent_solver"),
               std::runtime_error);
  EXPECT_NO_THROW(HamiltonianConstructorFactory::register_instance(
      []() -> HamiltonianConstructorFactory::return_type {
        return std::make_unique<TestHamiltonianConstructor>();
      }));
  EXPECT_THROW(HamiltonianConstructorFactory::register_instance(
                   []() -> HamiltonianConstructorFactory::return_type {
                     return std::make_unique<TestHamiltonianConstructor>();
                   }),
               std::runtime_error);
  auto test_scf =
      HamiltonianConstructorFactory::create("test-hamiltonian_constructor");

  // Test unregister_instance
  // First test unregistering a non-existent key (should return false)
  EXPECT_FALSE(
      HamiltonianConstructorFactory::unregister_instance("nonexistent_key"));

  // Test unregistering an existing key (should return true)
  EXPECT_TRUE(HamiltonianConstructorFactory::unregister_instance(
      "test-hamiltonian_constructor"));

  // Test unregistering the same key again (should return false since it's
  // already removed)
  EXPECT_FALSE(HamiltonianConstructorFactory::unregister_instance(
      "test-hamiltonian_constructor"));
}

TEST_F(HamiltonianConstructorTest, Default_EdgeCases) {
  auto hc = HamiltonianConstructorFactory::create();

  // Create structure for basis set
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0)};
  std::vector<std::string> symbols = {"H"};
  Structure structure(coordinates, symbols);

  // Create basis set of appropriate size for tests
  std::vector<Shell> shells;
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector<double>{1.0},
                            std::vector<double>{1.0}));
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector<double>{1.0},
                            std::vector<double>{1.0}));
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector<double>{1.0},
                            std::vector<double>{1.0}));

  auto basis_set = std::make_shared<BasisSet>("test", shells, structure);

  // Throw if basis set is not set in orbitals
  EXPECT_THROW(
      {
        // Create model orbitals without basis set
        auto orbitals =
            std::make_shared<ModelOrbitals>(3, true);  // 3 orbitals, restricted
        hc->run(orbitals);
      },
      std::runtime_error);

  // Test that restricted orbitals throw when alpha active space is empty
  EXPECT_THROW(
      {
        Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(3, 3);
        std::vector<size_t> empty_active_indices{};  // Empty active space
        // Create restricted orbitals with no active space
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::nullopt, basis_set,
            std::make_tuple(std::move(empty_active_indices),
                            std::vector<size_t>{}));
        hc->run(orbitals);
      },
      std::runtime_error);

  // Test that unrestricted orbitals throw when alpha is empty
  EXPECT_THROW(({
                 Eigen::MatrixXd coeffs_alpha = Eigen::MatrixXd::Identity(3, 3);
                 Eigen::MatrixXd coeffs_beta = Eigen::MatrixXd::Identity(3, 3);
                 std::vector<size_t> alpha_active_indices{};  // Empty alpha
                 std::vector<size_t> beta_active_indices{0, 1};
                 std::vector<size_t> alpha_inactive_indices{};
                 std::vector<size_t> beta_inactive_indices{2};
                 // Create unrestricted orbitals with only beta active space
                 auto orbitals = std::make_shared<Orbitals>(
                     coeffs_alpha, coeffs_beta, std::nullopt, std::nullopt,
                     std::nullopt, basis_set,
                     std::make_tuple(std::move(alpha_active_indices),
                                     std::move(alpha_inactive_indices),
                                     std::move(beta_active_indices),
                                     std::move(beta_inactive_indices)));
                 hc->run(orbitals);
               }),
               std::runtime_error);

  // Test that unrestricted orbitals throw when beta is empty
  EXPECT_THROW(({
                 Eigen::MatrixXd coeffs_alpha = Eigen::MatrixXd::Identity(3, 3);
                 Eigen::MatrixXd coeffs_beta = Eigen::MatrixXd::Identity(3, 3);
                 std::vector<size_t> alpha_active_indices{0, 1};
                 std::vector<size_t> beta_active_indices{};  // Empty beta
                 std::vector<size_t> alpha_inactive_indices{2};
                 std::vector<size_t> beta_inactive_indices{};
                 // Create unrestricted orbitals with only alpha active space
                 auto orbitals = std::make_shared<Orbitals>(
                     coeffs_alpha, coeffs_beta, std::nullopt, std::nullopt,
                     std::nullopt, basis_set,
                     std::make_tuple(std::move(alpha_active_indices),
                                     std::move(alpha_inactive_indices),
                                     std::move(beta_active_indices),
                                     std::move(beta_inactive_indices)));
                 hc->run(orbitals);
               }),
               std::runtime_error);

  // Throw if the active space is larger than the MO set
  EXPECT_THROW(
      {
        Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(3, 3);
        std::vector<size_t> active_indices(
            {0, 1, 2, 3});  // 4 indices for 3x3 matrix
        // Create orbitals with invalid active space
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::nullopt, basis_set,
            std::make_tuple(std::move(active_indices), std::vector<size_t>{}));
        hc->run(orbitals);
      },
      std::invalid_argument);

  // Throw if there is an index out of bounds
  EXPECT_THROW(
      {
        Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(3, 3);
        std::vector<size_t> active_indices(
            {0, 3});  // Index 3 is out of bounds for 3x3 matrix
        // Create orbitals with out-of-bounds active space index
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::nullopt, basis_set,
            std::make_tuple(std::move(active_indices), std::vector<size_t>{}));
        hc->run(orbitals);
      },
      std::invalid_argument);

  // Throw if there are repeated indices in the active space
  EXPECT_THROW(
      {
        Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(3, 3);
        std::vector<size_t> active_indices({0, 0});  // Repeated index
        // Create orbitals with repeated active space indices
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::nullopt, basis_set,
            std::make_tuple(std::move(active_indices), std::vector<size_t>{}));
        hc->run(orbitals);
      },
      std::invalid_argument);

  // Throw if active space indices are not sorted
  EXPECT_THROW(
      {
        Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(3, 3);
        std::vector<size_t> active_indices({1, 0});  // Unsorted indices
        // Create orbitals with unsorted active space indices
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::nullopt, basis_set,
            std::make_tuple(std::move(active_indices), std::vector<size_t>{}));
        hc->run(orbitals);
      },
      std::runtime_error);

  // Throw if alpha and beta active spaces have different sizes
  EXPECT_THROW(
      {
        Eigen::MatrixXd coeffs_alpha = Eigen::MatrixXd::Identity(4, 4);
        Eigen::MatrixXd coeffs_beta = Eigen::MatrixXd::Identity(4, 4);
        std::vector<size_t> alpha_active_indices({0, 1});  // 2 orbitals
        std::vector<size_t> beta_active_indices({0, 1, 2});
        // Create unrestricted orbitals with different active space sizes
        std::vector<size_t> alpha_inactive_indices({2, 3});
        std::vector<size_t> beta_inactive_indices({3});
        auto orbitals = std::make_shared<Orbitals>(
            coeffs_alpha, coeffs_beta, std::nullopt, std::nullopt, std::nullopt,
            basis_set,
            std::make_tuple(std::move(alpha_active_indices),
                            std::move(alpha_inactive_indices),
                            std::move(beta_active_indices),
                            std::move(beta_inactive_indices)));
        hc->run(orbitals);
      },
      std::runtime_error);

  // Different alpha/beta indices with same size should work
  // Create structure for large basis set
  std::vector<Eigen::Vector3d> large_coordinates = {
      Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0)};
  std::vector<std::string> large_symbols = {"H", "H", "H", "H"};
  Structure large_structure(large_coordinates, large_symbols);

  EXPECT_NO_THROW({
    // Create basis set with enough shells for this test
    std::vector<Shell> large_shells;
    for (int i = 0; i < 4; ++i) {
      large_shells.emplace_back(Shell(i, OrbitalType::S,
                                      std::vector<double>{1.0},
                                      std::vector<double>{1.0}));
    }
    auto large_basis_set =
        std::make_shared<BasisSet>("test", large_shells, large_structure);

    Eigen::MatrixXd coeffs_alpha = Eigen::MatrixXd::Identity(4, 4);
    Eigen::MatrixXd coeffs_beta = Eigen::MatrixXd::Ones(4, 4);
    std::vector<size_t> alpha_active_indices({0, 1});  // Different indices
    std::vector<size_t> beta_active_indices({2, 3});   // but same size
    std::vector<size_t> alpha_inactive_indices(
        {2, 3});  // remaining orbitals for alpha
    std::vector<size_t> beta_inactive_indices(
        {0, 1});  // remaining orbitals for beta
    // Create unrestricted orbitals with different indices but same size
    auto orbitals = std::make_shared<Orbitals>(
        coeffs_alpha, coeffs_beta, std::nullopt, std::nullopt, std::nullopt,
        large_basis_set,
        std::make_tuple(
            std::move(alpha_active_indices), std::move(alpha_inactive_indices),
            std::move(beta_active_indices), std::move(beta_inactive_indices)));
    auto hamiltonian = hc->run(orbitals);
    EXPECT_TRUE(hamiltonian->has_one_body_integrals());
    EXPECT_TRUE(hamiltonian->has_two_body_integrals());
  });
}

TEST_F(HamiltonianConstructorTest, NonContiguousActiveSpace) {
  auto hc = HamiltonianConstructorFactory::create();

  // Create a structure for a simple molecule (e.g., H2)
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(0.0, 0.0, 1.4)};
  std::vector<std::string> symbols = {"H", "H"};
  Structure structure(coordinates, symbols);

  // Create basis set with enough shells for the test
  std::vector<Shell> shells;
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector<double>{1.0},
                            std::vector<double>{1.0}));
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector<double>{0.5},
                            std::vector<double>{1.0}));
  shells.emplace_back(Shell(1, OrbitalType::S, std::vector<double>{1.0},
                            std::vector<double>{1.0}));
  shells.emplace_back(Shell(1, OrbitalType::S, std::vector<double>{0.5},
                            std::vector<double>{1.0}));
  auto basis_set = std::make_shared<BasisSet>("test", shells, structure);

  // Create orbitals with non-contiguous active space indices
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(4, 4);

  // Set non-contiguous active space indices: 0, 2 (skipping 1)
  std::vector<size_t> active_indices = {0, 2};

  auto orbitals = std::make_shared<Orbitals>(
      coeffs, std::nullopt, std::nullopt, basis_set,
      std::make_tuple(std::vector<size_t>(active_indices),
                      std::vector<size_t>{}));
  // This should successfully construct the Hamiltonian
  // and exercise the non-contiguous active space code paths
  EXPECT_NO_THROW({
    auto hamiltonian = hc->run(orbitals);
    EXPECT_TRUE(hamiltonian->has_one_body_integrals());
    EXPECT_TRUE(hamiltonian->has_two_body_integrals());
  });
}

TEST_F(HamiltonianConstructorTest, NonContiguousInactiveSpace) {
  auto hc = HamiltonianConstructorFactory::create();

  // Create a structure for a molecule with enough electrons
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0)};
  std::vector<std::string> symbols = {"Li"};
  Structure structure(coordinates, symbols);

  // Create basis set with sufficient shells
  std::vector<Shell> shells;
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector<double>{2.0},
                            std::vector<double>{1.0}));
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector<double>{0.8},
                            std::vector<double>{1.0}));
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector<double>{0.3},
                            std::vector<double>{1.0}));
  shells.emplace_back(Shell(0, OrbitalType::P, std::vector<double>{1.0},
                            std::vector<double>{1.0}));
  auto basis_set = std::make_shared<BasisSet>("test", shells, structure);

  // Create orbitals with scenario that will create non-contiguous inactive
  // space
  Eigen::MatrixXd coeffs = Eigen::MatrixXd::Identity(
      6, 6);  // 1 s-shell + 1 s-shell + 1 s-shell + 3 p-shells = 6 orbitals

  // Set active space to include middle orbitals: 2, 3
  std::vector<size_t> active_indices = {2, 3};
  std::vector<size_t> inactive_indices = {0};

  auto orbitals = std::make_shared<Orbitals>(
      coeffs, std::nullopt, std::nullopt, basis_set,
      std::make_tuple(std::move(active_indices), std::move(inactive_indices)));
  EXPECT_NO_THROW({
    auto hamiltonian = hc->run(orbitals);
    EXPECT_TRUE(hamiltonian->has_one_body_integrals());
    EXPECT_TRUE(hamiltonian->has_two_body_integrals());
  });
}

// Cholesky Hamiltonian Constructor Tests
TEST_F(HamiltonianConstructorTest, DensityFittedFactoryRegistration) {
  // Test that qdk_cholesky is available
  auto available_solvers = HamiltonianConstructorFactory::available();
  EXPECT_GE(available_solvers.size(), 2);

  bool found_density_fitted = false;
  for (const auto& solver : available_solvers) {
    if (solver == "qdk_density_fitted") {
      found_density_fitted = true;
      break;
    }
  }
  EXPECT_TRUE(found_density_fitted)
      << "qdk_density_fitted not found in available constructors";

  // Test that we can create a density-fitted hamiltonian constructor
  EXPECT_NO_THROW(HamiltonianConstructorFactory::create("qdk_density_fitted"));
  auto density_fitted_hc =
      HamiltonianConstructorFactory::create("qdk_density_fitted");
  EXPECT_EQ(density_fitted_hc->name(), "qdk_density_fitted");
}

TEST_F(HamiltonianConstructorTest, DensityFittedRestrictedO2) {
  // Run restricted O2 with density-fitted
  auto [energy, hamiltonian, wfn] = run_restricted_o2("qdk_density_fitted");

  // Verify hamiltonian properties
  EXPECT_TRUE(hamiltonian->has_one_body_integrals());
  EXPECT_TRUE(hamiltonian->has_two_body_integrals());
  EXPECT_TRUE(hamiltonian->has_orbitals());
  EXPECT_TRUE(hamiltonian->is_restricted());
  EXPECT_EQ(hamiltonian->get_container_type(), "density_fitted");

  // Verify we can access the typed container
  EXPECT_TRUE(
      hamiltonian->has_container_type<DensityFittedHamiltonianContainer>());
  EXPECT_NO_THROW({
    const auto& container =
        hamiltonian->get_container<DensityFittedHamiltonianContainer>();
    EXPECT_EQ(container.get_container_type(), "density_fitted");
  });
}

TEST_F(HamiltonianConstructorTest, DensityFittedUnrestrictedO2) {
  // Run unrestricted O2 triplet with density-fitted
  auto [energy, hamiltonian] = run_unrestricted_o2("qdk_density_fitted");

  // Verify hamiltonian properties
  EXPECT_TRUE(hamiltonian->has_one_body_integrals());
  EXPECT_TRUE(hamiltonian->has_two_body_integrals());
  EXPECT_TRUE(hamiltonian->has_orbitals());
  EXPECT_TRUE(hamiltonian->is_unrestricted());
  EXPECT_EQ(hamiltonian->get_container_type(), "density_fitted");

  // Verify we can access the typed container
  EXPECT_TRUE(
      hamiltonian->has_container_type<DensityFittedHamiltonianContainer>());
  EXPECT_NO_THROW({
    const auto& container =
        hamiltonian->get_container<DensityFittedHamiltonianContainer>();
    EXPECT_EQ(container.get_container_type(), "density_fitted");
  });
}

// Helper class to force unrestricted behavior for closed-shell systems
class ForceUnrestrictedOrbitals : public Orbitals {
 public:
  ForceUnrestrictedOrbitals(
      const Eigen::MatrixXd& coeffs_alpha, const Eigen::MatrixXd& coeffs_beta,
      const std::optional<Eigen::VectorXd>& energies_alpha,
      const std::optional<Eigen::VectorXd>& energies_beta,
      const std::optional<Eigen::MatrixXd>& ao_overlap,
      std::shared_ptr<BasisSet> basis_set)
      : Orbitals(coeffs_alpha, coeffs_beta, energies_alpha, energies_beta,
                 ao_overlap, basis_set) {}

  bool is_restricted() const override { return false; }
  bool is_unrestricted() const override { return true; }

  // Add method to set active space
  void set_active_space(const std::vector<size_t>& alpha_active,
                        const std::vector<size_t>& beta_active) {
    _active_space_indices = {alpha_active, beta_active};
  }
};

class HamiltonianIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(HamiltonianIntegrationTest, IntegralSymmetriesEnergiesO2Singlet) {
  // Restricted and unrestricted calculations
  // should give identical results for closed-shell systems (o2 singlet)

  // Create o2 molecule structure
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Run restricted HF calculation
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");

  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [rhf_energy, rhf_wavefunction] =
      scf_factory->run(o2_structure_ptr, 0, 1, "cc-pvdz");
  auto rhf_orbitals = rhf_wavefunction->get_orbitals();

  // Create Hamiltonian from restricted orbitals
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto rhf_hamiltonian = ham_factory->run(rhf_orbitals);

  // Calculate restricted MP2 energy using factory
  auto rhf_ansatz =
      std::make_shared<Ansatz>(*rhf_hamiltonian, *rhf_wavefunction);
  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");
  auto [rmp2_energy, rhf_mp2_wavefunction] = mp2_calculator->run(rhf_ansatz);

  // Create unrestricted orbitals from restricted ones
  // Get restricted coefficients and energies
  auto [rhf_coeffs_alpha, rhf_coeffs_beta] = rhf_orbitals->get_coefficients();
  auto [rhf_energies_alpha, rhf_energies_beta] = rhf_orbitals->get_energies();

  // For closed shell: alpha = beta coefficients and energies
  // Create unrestricted orbitals with same alpha/beta data but force
  // unrestricted behavior
  auto unrestricted_orbitals = std::make_shared<ForceUnrestrictedOrbitals>(
      rhf_coeffs_alpha, rhf_coeffs_beta, rhf_energies_alpha, rhf_energies_beta,
      rhf_orbitals->get_overlap_matrix(), rhf_orbitals->get_basis_set());

  // Set active space if it exists in original orbitals
  if (rhf_orbitals->has_active_space()) {
    auto [alpha_active, beta_active] = rhf_orbitals->get_active_space_indices();
    unrestricted_orbitals->set_active_space(alpha_active, beta_active);
  }

  // Create unrestricted Hamiltonian
  auto uhf_hamiltonian = ham_factory->run(unrestricted_orbitals);

  // Calculate unrestricted MP2 energy using factory
  // Need to create a UHF wavefunction with the unrestricted orbitals
  // Get the determinant from the RHF wavefunction
  const auto& rhf_sd_container =
      rhf_wavefunction->get_container<SlaterDeterminantContainer>();
  const auto& rhf_determinants = rhf_sd_container.get_active_determinants();

  // Create a new SlaterDeterminantContainer with the same determinant but
  // unrestricted orbitals
  auto uhf_container = std::make_unique<SlaterDeterminantContainer>(
      rhf_determinants[0], unrestricted_orbitals);
  auto uhf_wavefunction =
      std::make_shared<Wavefunction>(std::move(uhf_container));

  auto uhf_ansatz =
      std::make_shared<Ansatz>(*uhf_hamiltonian, *uhf_wavefunction);
  auto [ump2_total_energy, uhf_mp2_wavefunction] =
      mp2_calculator->run(uhf_ansatz);
  double ump2_correlation = ump2_total_energy - rhf_energy;
  double ump2_energy = rhf_energy + ump2_correlation;

  // MP2 energies should be identical for RMP2/UMP2
  EXPECT_NEAR(rmp2_energy, ump2_energy, testing::scf_energy_tolerance)
      << "Restricted and unrestricted MP2 energies should be identical for "
         "closed-shell O2. "
      << "RMP2=" << rmp2_energy << ", UMP2=" << ump2_energy
      << ", diff=" << std::abs(rmp2_energy - ump2_energy);

  // Verify integral symmetries aaaa == bbbb
  const auto& [aaaa_integrals, aabb_integrals, bbbb_integrals] =
      uhf_hamiltonian->get_two_body_integrals();

  // Elementwise comparison for aaaa == bbbb integrals
  EXPECT_EQ(aaaa_integrals.size(), bbbb_integrals.size())
      << "Alpha-alpha and beta-beta integral sizes should match";
  for (int i = 0; i < aaaa_integrals.size(); ++i) {
    double diff = std::abs(aaaa_integrals[i] - bbbb_integrals[i]);
    EXPECT_LT(diff, std::numeric_limits<double>::epsilon())
        << "Alpha-alpha and beta-beta integrals should be identical."
           ". Difference: "
        << diff;
  }

  // Verify one-body integral symmetries alpha == beta
  const auto& [alpha_one_body, beta_one_body] =
      uhf_hamiltonian->get_one_body_integrals();

  // Elementwise comparison for alpha == beta one-body integrals
  EXPECT_EQ(alpha_one_body.rows(), beta_one_body.rows())
      << "Alpha and beta one-body integral dimensions should match";
  EXPECT_EQ(alpha_one_body.cols(), beta_one_body.cols())
      << "Alpha and beta one-body integral dimensions should match";
  for (int i = 0; i < alpha_one_body.rows(); ++i) {
    for (int j = 0; j < alpha_one_body.cols(); ++j) {
      double diff = std::abs(alpha_one_body(i, j) - beta_one_body(i, j));
      EXPECT_LT(diff, std::numeric_limits<double>::epsilon())
          << "Alpha and beta one-body integrals should be identical for "
             "closed-shell O2."
             "Difference: "
          << diff;
    }
  }

  // Verify that restricted and unrestricted Hamiltonians are consistent
  // The restricted integrals should match the aabb integrals
  const auto& [restricted_aaaa, restricted_aabb, restricted_bbbb] =
      rhf_hamiltonian->get_two_body_integrals();

  // Elementwise comparison for restricted aaaa == unrestricted aabb integrals
  EXPECT_EQ(restricted_aaaa.size(), aabb_integrals.size())
      << "Restricted aaaa and unrestricted aabb integral sizes should match";
  for (int i = 0; i < restricted_aaaa.size(); ++i) {
    double diff = std::abs(restricted_aaaa[i] - aabb_integrals[i]);
    EXPECT_LT(diff, std::numeric_limits<double>::epsilon())
        << "Integrals should be identical. "
           ". Difference: "
        << diff;
  }

  // Verify aabb == bbaa symmetry
  // Get active space size to determine integral tensor dimensions
  size_t active_space_size;
  auto [alpha_active, beta_active] =
      unrestricted_orbitals->get_active_space_indices();
  active_space_size = alpha_active.size();

  // Test aabb[i,j,k,l] == aabb[k,l,i,j] (particle exchange symmetry)
  auto get_integral_index = [active_space_size](size_t i, size_t j, size_t k,
                                                size_t l) -> size_t {
    return i * active_space_size * active_space_size * active_space_size +
           j * active_space_size * active_space_size + k * active_space_size +
           l;
  };

  for (size_t i = 0; i < active_space_size; i++) {
    for (size_t j = 0; j < active_space_size; j++) {
      for (size_t k = 0; k < active_space_size; k++) {
        for (size_t l = 0; l < active_space_size; l++) {
          double ijkl = aabb_integrals[get_integral_index(i, j, k, l)];
          double klij = aabb_integrals[get_integral_index(k, l, i, j)];
          double diff = std::abs(ijkl - klij);
          EXPECT_LT(diff, testing::integral_tolerance)
              << "Symmetry violation for particle exchange. "
              << "Difference: " << diff << " exceeds tolerance "
              << testing::integral_tolerance;
        }
      }
    }
  }
}

TEST_F(HamiltonianIntegrationTest, MixedIntegralSymmetriesO2Triplet) {
  // Test mixed integral symmetries for unrestricted O2 open shell
  // ijkl == jikl == ijlk == jilk

  // Create o2 molecule structure
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");

  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [energy, wavefunction] =
      scf_factory->run(o2_structure_ptr, 0, 3, "cc-pvdz");
  auto orbitals = wavefunction->get_orbitals();

  // Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto uhf_hamiltonian = ham_factory->run(orbitals);

  // Get aabb integrals
  const auto& [aaaa_integrals, aabb_integrals, bbbb_integrals] =
      uhf_hamiltonian->get_two_body_integrals();

  // Get active space size
  auto [alpha_active, beta_active] = orbitals->get_active_space_indices();
  size_t active_space_size = alpha_active.size();

  auto get_index = [active_space_size](size_t i, size_t j, size_t k,
                                       size_t l) -> size_t {
    return i * active_space_size * active_space_size * active_space_size +
           j * active_space_size * active_space_size + k * active_space_size +
           l;
  };

  // Test mixed integral symmetries: ijkl == jikl == ijlk == jilk
  for (size_t i = 0; i < active_space_size; i++) {
    for (size_t j = 0; j < active_space_size; j++) {
      for (size_t k = 0; k < active_space_size; k++) {
        for (size_t l = 0; l < active_space_size; l++) {
          // Get the four symmetry-related integrals
          double ijkl = aabb_integrals[get_index(i, j, k, l)];
          double jikl = aabb_integrals[get_index(j, i, k, l)];
          double ijlk = aabb_integrals[get_index(i, j, l, k)];
          double jilk = aabb_integrals[get_index(j, i, l, k)];

          // Test all symmetries
          double diff1 = std::abs(ijkl - jikl);
          double diff2 = std::abs(ijkl - ijlk);
          double diff3 = std::abs(ijkl - jilk);

          EXPECT_LT(diff1, testing::integral_tolerance)
              << "Symmetry violation for ijkl=jikl."
              << "Difference: " << diff1 << " exceeds tolerance "
              << testing::integral_tolerance;

          EXPECT_LT(diff2, testing::integral_tolerance)
              << "Symmetry violation for ijkl=ijlk."
              << "Difference: " << diff2 << " exceeds tolerance "
              << testing::integral_tolerance;

          EXPECT_LT(diff3, testing::integral_tolerance)
              << "Symmetry violation for ijkl=jikl."
              << "Difference: " << diff3 << " exceeds tolerance "
              << testing::integral_tolerance;
        }
      }
    }
  };
}

TEST_F(HamiltonianIntegrationTest,
       O2DeterministicBehaviorRestrictedUnrestricted) {
  // Test that repeated calculations give identical integral elements
  // for both restricted (singlet) and unrestricted (triplet) O2

  // Test restricted O2 deterministic behavior
  {
    auto [energy1, hamiltonian1, wfn1] = run_restricted_o2();
    auto [energy2, hamiltonian2, wfn2] = run_restricted_o2();

    // Energies should be identical
    EXPECT_DOUBLE_EQ(energy1, energy2)
        << "Restricted O2 energies should be identical across runs. "
        << "Energy1=" << energy1 << ", Energy2=" << energy2;

    // Core energies should be identical
    EXPECT_DOUBLE_EQ(hamiltonian1->get_core_energy(),
                     hamiltonian2->get_core_energy())
        << "Core energies should be identical across runs";

    // One-body integrals should be identical
    auto [h1_one_alpha, h1_one_beta] = hamiltonian1->get_one_body_integrals();
    auto [h2_one_alpha, h2_one_beta] = hamiltonian2->get_one_body_integrals();

    EXPECT_EQ(h1_one_alpha.rows(), h2_one_alpha.rows());
    EXPECT_EQ(h1_one_alpha.cols(), h2_one_alpha.cols());
    EXPECT_EQ(h1_one_beta.rows(), h2_one_beta.rows());
    EXPECT_EQ(h1_one_beta.cols(), h2_one_beta.cols());

    for (int i = 0; i < h1_one_alpha.rows(); ++i) {
      for (int j = 0; j < h1_one_alpha.cols(); ++j) {
        EXPECT_DOUBLE_EQ(h1_one_alpha(i, j), h2_one_alpha(i, j))
            << "Restricted O2 alpha one-body integral (" << i << "," << j
            << ") differs across runs";
        EXPECT_DOUBLE_EQ(h1_one_beta(i, j), h2_one_beta(i, j))
            << "Restricted O2 beta one-body integral (" << i << "," << j
            << ") differs across runs";
      }
    }

    // Two-body integrals should be identical
    auto [h1_two_aaaa, h1_two_aabb, h1_two_bbbb] =
        hamiltonian1->get_two_body_integrals();
    auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] =
        hamiltonian2->get_two_body_integrals();

    EXPECT_EQ(h1_two_aaaa.size(), h2_two_aaaa.size());
    EXPECT_EQ(h1_two_aabb.size(), h2_two_aabb.size());
    EXPECT_EQ(h1_two_bbbb.size(), h2_two_bbbb.size());

    for (size_t i = 0; i < h1_two_aaaa.size(); ++i) {
      EXPECT_DOUBLE_EQ(h1_two_aaaa[i], h2_two_aaaa[i])
          << "Restricted O2 aaaa two-body integral element " << i
          << " differs across runs";
    }
    for (size_t i = 0; i < h1_two_aabb.size(); ++i) {
      EXPECT_DOUBLE_EQ(h1_two_aabb[i], h2_two_aabb[i])
          << "Restricted O2 aabb two-body integral element " << i
          << " differs across runs";
    }
    for (size_t i = 0; i < h1_two_bbbb.size(); ++i) {
      EXPECT_DOUBLE_EQ(h1_two_bbbb[i], h2_two_bbbb[i])
          << "Restricted O2 bbbb two-body integral element " << i
          << " differs across runs";
    }
  }

  // Test unrestricted O2 triplet deterministic behavior
  {
    auto [energy1, hamiltonian1] = run_unrestricted_o2();
    auto [energy2, hamiltonian2] = run_unrestricted_o2();

    // Energies should be identical
    EXPECT_DOUBLE_EQ(energy1, energy2)
        << "Unrestricted O2 energies should be identical across runs. "
        << "Energy1=" << energy1 << ", Energy2=" << energy2;

    // Core energies should be identical
    EXPECT_DOUBLE_EQ(hamiltonian1->get_core_energy(),
                     hamiltonian2->get_core_energy())
        << "Core energies should be identical across runs";

    // One-body integrals should be identical
    auto [h1_one_alpha, h1_one_beta] = hamiltonian1->get_one_body_integrals();
    auto [h2_one_alpha, h2_one_beta] = hamiltonian2->get_one_body_integrals();

    EXPECT_EQ(h1_one_alpha.rows(), h2_one_alpha.rows());
    EXPECT_EQ(h1_one_alpha.cols(), h2_one_alpha.cols());
    EXPECT_EQ(h1_one_beta.rows(), h2_one_beta.rows());
    EXPECT_EQ(h1_one_beta.cols(), h2_one_beta.cols());

    for (int i = 0; i < h1_one_alpha.rows(); ++i) {
      for (int j = 0; j < h1_one_alpha.cols(); ++j) {
        EXPECT_DOUBLE_EQ(h1_one_alpha(i, j), h2_one_alpha(i, j))
            << "Unrestricted O2 alpha one-body integral (" << i << "," << j
            << ") differs across runs";
        EXPECT_DOUBLE_EQ(h1_one_beta(i, j), h2_one_beta(i, j))
            << "Unrestricted O2 beta one-body integral (" << i << "," << j
            << ") differs across runs";
      }
    }

    // Two-body integrals should be identical
    auto [h1_two_aaaa, h1_two_aabb, h1_two_bbbb] =
        hamiltonian1->get_two_body_integrals();
    auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] =
        hamiltonian2->get_two_body_integrals();

    EXPECT_EQ(h1_two_aaaa.size(), h2_two_aaaa.size());
    EXPECT_EQ(h1_two_aabb.size(), h2_two_aabb.size());
    EXPECT_EQ(h1_two_bbbb.size(), h2_two_bbbb.size());

    for (size_t i = 0; i < h1_two_aaaa.size(); ++i) {
      EXPECT_DOUBLE_EQ(h1_two_aaaa[i], h2_two_aaaa[i])
          << "Unrestricted O2 aaaa two-body integral element " << i
          << " differs across runs";
    }
    for (size_t i = 0; i < h1_two_aabb.size(); ++i) {
      EXPECT_DOUBLE_EQ(h1_two_aabb[i], h2_two_aabb[i])
          << "Unrestricted O2 aabb two-body integral element " << i
          << " differs across runs";
    }
    for (size_t i = 0; i < h1_two_bbbb.size(); ++i) {
      EXPECT_DOUBLE_EQ(h1_two_bbbb[i], h2_two_bbbb[i])
          << "Unrestricted O2 bbbb two-body integral element " << i
          << " differs across runs";
    }
  }
}

TEST_F(HamiltonianIntegrationTest, DensityFittedRestrictedO2MP2) {
  // Run restricted O2 with density-fitted
  auto [energy, df_hamiltonian, wfn] = run_restricted_o2("qdk_density_fitted");

  // Verify hamiltonian properties
  EXPECT_TRUE(df_hamiltonian->has_one_body_integrals());
  EXPECT_TRUE(df_hamiltonian->has_two_body_integrals());
  EXPECT_TRUE(df_hamiltonian->has_orbitals());
  EXPECT_TRUE(df_hamiltonian->is_restricted());
  EXPECT_EQ(df_hamiltonian->get_container_type(), "density_fitted");

  std::cout << std::setprecision(10) << "energy is: " << energy << std::endl;

  // Verify we can access the typed container
  auto [h_aa, h_bb] = df_hamiltonian->get_one_body_integrals();
  auto [eri_aaaa, eri_aabb, eri_bbbb] =
      df_hamiltonian->get_two_body_integrals();
  auto orbitals = df_hamiltonian->get_orbitals();
  double core_energy = df_hamiltonian->get_core_energy();

  //  auto [inactive_fock_aa, inactive_fock_bb] =
  //  df_hamiltonian->get_inactive_fock_matrix();

  // Calculate restricted MP2 energy using factory
  auto rhf_ansatz = std::make_shared<Ansatz>(*df_hamiltonian, *wfn);
  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");
  //   auto [rmp2_energy, rhf_mp2_wavefunction] =
  //   mp2_calculator->run(rhf_ansatz);

  auto [n_alpha, n_beta] = wfn->get_active_num_electrons();

  // Cast to MP2Calculator to access the specific method
  auto* mp2_calc_ptr =
      dynamic_cast<qdk::chemistry::algorithms::microsoft::MP2Calculator*>(
          mp2_calculator.get());
  auto rmp2_corr_energy = mp2_calc_ptr->calculate_restricted_mp2_energy(
      df_hamiltonian, orbitals, n_alpha);

  std::cout << "mp2 correlation energy" << rmp2_corr_energy << std::endl;

  EXPECT_NEAR(
      rmp2_corr_energy, -0.3843068379,
      testing::scf_energy_tolerance);  // Replace with actual expected value
}

// ============================================================================
// Instantiate parameterized tests for all container types
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    AllContainerTypes, HamiltonianTest,
    ::testing::Values("canonical_four_center", "density_fitted"),
    [](const ::testing::TestParamInfo<std::string>& info) {
      std::string name = info.param;
      std::replace(name.begin(), name.end(), '_', ' ');
      name.erase(std::remove(name.begin(), name.end(), ' '), name.end());
      return name;
    });
