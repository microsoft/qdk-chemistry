// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "ut_common.hpp"
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class HamiltonianTest : public ::testing::Test {
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

    // Create inactive Fock matrix (empty for restricted systems)
    inactive_fock = Eigen::MatrixXd::Zero(0, 0);
  }

  void TearDown() override {
    // Clean up any test files
    std::filesystem::remove("test.hamiltonian.json");
    std::filesystem::remove("test.hamiltonian.h5");
    std::filesystem::remove("test.hamiltonian.fcidump");
  }

  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;
  std::shared_ptr<Orbitals> orbitals;
  unsigned num_electrons;
  double core_energy;
  Eigen::MatrixXd inactive_fock;
};

class HamiltonianConstructorTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// Helper lambda to run restricted O2 calculation
auto run_restricted_o2 = []() {
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");

  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [rhf_energy, rhf_wavefunction] =
      scf_factory->run(o2_structure_ptr, 0, 1, "cc-pvdz");
  auto rhf_orbitals = rhf_wavefunction->get_orbitals();

  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto rhf_hamiltonian = ham_factory->run(rhf_orbitals);

  return std::make_tuple(rhf_energy, rhf_hamiltonian);
};

// Helper lambda to run unrestricted O2 triplet calculation
auto run_unrestricted_o2 = []() {
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

  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto uhf_hamiltonian = ham_factory->run(uhf_orbitals);

  return std::make_tuple(uhf_energy, uhf_hamiltonian);
};

class TestHamiltonianConstructor : public HamiltonianConstructor {
 public:
  std::string name() const override { return "test-hamiltonian_constructor"; }
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Orbitals> orbitals) const override {
    // Dummy implementation for testing
    Eigen::MatrixXd one_body = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd two_body = Eigen::VectorXd::Random(81);
    Eigen::MatrixXd f_inact = Eigen::MatrixXd::Identity(0, 0);
    return std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, orbitals, 0.0, f_inact));
  }
};

TEST_F(HamiltonianTest, Constructor) {
  // Test the constructor with all required data
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  EXPECT_TRUE(h.has_one_body_integrals());
  EXPECT_TRUE(h.has_two_body_integrals());
  EXPECT_TRUE(h.has_orbitals());
  EXPECT_EQ(h.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h.get_core_energy(), 1.5);
  EXPECT_EQ(h.get_container_type(), "canonical_four_center");
}

TEST_F(HamiltonianTest, ConstructorWithInactiveFock) {
  // Test the constructor with inactive fock matrix
  // For this test specifically, create ModelOrbitals with inactive space
  std::vector<size_t> active_indices = {1, 2};  // Only orbital 1 is active
  std::vector<size_t> inactive_indices = {0};   // Orbital 0 is inactive
  auto orbitals_with_inactive = std::make_shared<ModelOrbitals>(
      4,
      std::make_tuple(std::move(active_indices), std::move(inactive_indices)));

  // Create a non-empty inactive Fock matrix
  Eigen::MatrixXd non_empty_inactive_fock = Eigen::MatrixXd::Identity(2, 2);
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals_with_inactive, core_energy,
      non_empty_inactive_fock));

  EXPECT_TRUE(h.has_one_body_integrals());
  EXPECT_TRUE(h.has_two_body_integrals());
  EXPECT_TRUE(h.has_orbitals());
  EXPECT_TRUE(h.has_inactive_fock_matrix());
  EXPECT_EQ(h.get_orbitals()->get_num_molecular_orbitals(), 4);
  EXPECT_EQ(h.get_core_energy(), 1.5);
}

TEST_F(HamiltonianTest, MoveConstructor) {
  Hamiltonian h1(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));
  Hamiltonian h2(std::move(h1));

  EXPECT_TRUE(h2.has_one_body_integrals());
  EXPECT_TRUE(h2.has_two_body_integrals());
  EXPECT_TRUE(h2.has_orbitals());
  EXPECT_EQ(h2.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2.get_core_energy(), 1.5);
}

TEST_F(HamiltonianTest, CopyConstructorAndAssignment) {
  // Create source Hamiltonian with full data
  Eigen::MatrixXd inactive_fock = Eigen::MatrixXd::Random(2, 2);
  Hamiltonian h1(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  // Test copy constructor
  Hamiltonian h2(h1);

  // Verify all data was copied correctly
  EXPECT_TRUE(h2.has_one_body_integrals());
  EXPECT_TRUE(h2.has_two_body_integrals());
  EXPECT_TRUE(h2.has_orbitals());
  EXPECT_TRUE(h2.has_inactive_fock_matrix());
  EXPECT_EQ(h2.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2.get_core_energy(), 1.5);

  // Verify one body integral copy
  auto [h1_one_alpha, h1_one_beta] = h1.get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2.get_one_body_integrals();
  EXPECT_TRUE(h1_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h1_one_beta.isApprox(h2_one_beta));

  // Compare each component of the two-body integrals tuple
  auto [h1_two_aaaa, h1_two_aabb, h1_two_bbbb] = h1.get_two_body_integrals();
  auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] = h2.get_two_body_integrals();
  EXPECT_TRUE(h1_two_aaaa.isApprox(h2_two_aaaa));
  EXPECT_TRUE(h1_two_aabb.isApprox(h2_two_aabb));
  EXPECT_TRUE(h1_two_bbbb.isApprox(h2_two_bbbb));
  EXPECT_TRUE(h1.get_inactive_fock_matrix().first.isApprox(
      h2.get_inactive_fock_matrix().first));

  // Test copy assignment
  Hamiltonian h3(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));
  h3 = h1;

  // Verify assignment worked correctly
  EXPECT_TRUE(h3.has_one_body_integrals());
  EXPECT_TRUE(h3.has_two_body_integrals());
  EXPECT_TRUE(h3.has_orbitals());
  EXPECT_TRUE(h3.has_inactive_fock_matrix());
  EXPECT_EQ(h3.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h3.get_core_energy(), 1.5);

  // Test self-assignment (should be no-op)
  Hamiltonian h4(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));
  Hamiltonian* h4_ptr = &h4;
  h4 = *h4_ptr;  // Self-assignment

  // Should remain unchanged
  EXPECT_TRUE(h4.has_one_body_integrals());
  EXPECT_TRUE(h4.has_two_body_integrals());
  EXPECT_TRUE(h4.has_orbitals());
  EXPECT_EQ(h4.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h4.get_core_energy(), 1.5);
}

TEST_F(HamiltonianTest, TwoBodyElementAccess) {
  // Create a Hamiltonian with known two-body integrals
  Eigen::MatrixXd test_one_body = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd test_two_body = Eigen::VectorXd::Zero(16);  // 2^4 = 16

  // Set specific values we can test - these indices test the get_two_body_index
  // function
  test_two_body[0] = 1.0;   // (0,0,0,0) -> index 0*8 + 0*4 + 0*2 + 0 = 0
  test_two_body[1] = 2.0;   // (0,0,0,1) -> index 0*8 + 0*4 + 0*2 + 1 = 1
  test_two_body[5] = 3.0;   // (0,1,0,1) -> index 0*8 + 1*4 + 0*2 + 1 = 5
  test_two_body[15] = 4.0;  // (1,1,1,1) -> index 1*8 + 1*4 + 1*2 + 1 = 15
  test_two_body[10] = 5.0;  // (1,0,1,0) -> index 1*8 + 0*4 + 1*2 + 0 = 10
  test_two_body[7] = 6.0;   // (0,1,1,1) -> index 0*8 + 1*4 + 1*2 + 1 = 7

  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      test_one_body, test_two_body, orbitals, core_energy, inactive_fock));

  // Test accessing specific elements to verify get_two_body_index calculations
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 1), 2.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 1, 0, 1), 3.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 1, 1, 1), 4.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 0, 1, 0), 5.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 1, 1, 1), 6.0);

  // Test elements that should be zero
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 1, 0), 0.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 0, 0, 0), 0.0);

  // Test out-of-range access - this tests bounds checking in get_two_body_index
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

  Hamiltonian h_large(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      large_one_body, large_two_body, large_orbitals, 0.0, large_inact_f));

  EXPECT_DOUBLE_EQ(h_large.get_two_body_element(2, 1, 0, 2), 7.0);
  EXPECT_DOUBLE_EQ(h_large.get_two_body_element(1, 2, 2, 1), 8.0);
}

TEST_F(HamiltonianTest, JSONSerialization) {
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  // Test JSON conversion
  nlohmann::json j = h.to_json();

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
  auto [h_one_alpha, h_one_beta] = h.get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h_one_beta.isApprox(h2_one_beta));

  // Check two body
  auto [h_two_aaaa, h_two_aabb, h_two_bbbb] = h.get_two_body_integrals();
  auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] = h2->get_two_body_integrals();
  EXPECT_TRUE(h_two_aaaa.isApprox(h2_two_aaaa));
  EXPECT_TRUE(h_two_aabb.isApprox(h2_two_aabb));
  EXPECT_TRUE(h_two_bbbb.isApprox(h2_two_bbbb));

  // Check they are still restricted
  EXPECT_TRUE(h2->is_restricted());
  EXPECT_FALSE(h2->is_unrestricted());
}

TEST_F(HamiltonianTest, JSONFileIO) {
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  // Test file I/O
  std::string filename = "test.hamiltonian.json";
  h.to_json_file(filename);
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
  auto [h_one_alpha, h_one_beta] = h.get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h_one_beta.isApprox(h2_one_beta));

  auto [h_two_aaaa, h_two_aabb, h_two_bbbb] = h.get_two_body_integrals();
  auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] = h2->get_two_body_integrals();
  EXPECT_TRUE(h_two_aaaa.isApprox(h2_two_aaaa));
  EXPECT_TRUE(h_two_aabb.isApprox(h2_two_aabb));
  EXPECT_TRUE(h_two_bbbb.isApprox(h2_two_bbbb));

  // Check still restricted
  EXPECT_TRUE(h2->is_restricted());
  EXPECT_FALSE(h2->is_unrestricted());
}

TEST_F(HamiltonianTest, HDF5FileIO) {
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  // Test file I/O
  std::string filename = "test.hamiltonian.h5";
  h.to_hdf5_file(filename);
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
  auto [h_one_alpha, h_one_beta] = h.get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));
  EXPECT_TRUE(h_one_beta.isApprox(h2_one_beta));

  auto [h_two_aaaa, h_two_aabb, h_two_bbbb] = h.get_two_body_integrals();
  auto [h2_two_aaaa, h2_two_aabb, h2_two_bbbb] = h2->get_two_body_integrals();
  EXPECT_TRUE(h_two_aaaa.isApprox(h2_two_aaaa));
  EXPECT_TRUE(h_two_aabb.isApprox(h2_two_aabb));
  EXPECT_TRUE(h_two_bbbb.isApprox(h2_two_bbbb));
}

TEST_F(HamiltonianTest, GenericFileIO) {
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  // Test JSON via generic interface
  std::string json_filename = "test.hamiltonian.json";
  h.to_file(json_filename, "json");
  EXPECT_TRUE(std::filesystem::exists(json_filename));

  auto h2 = Hamiltonian::from_file(json_filename, "json");

  EXPECT_EQ(h2->get_orbitals()->get_num_molecular_orbitals(), 2);
  auto [h_one_alpha, h_one_beta] = h.get_one_body_integrals();
  auto [h2_one_alpha, h2_one_beta] = h2->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h2_one_alpha));

  // Test HDF5 via generic interface
  std::string hdf5_filename = "test.hamiltonian.h5";
  h.to_file(hdf5_filename, "hdf5");
  EXPECT_TRUE(std::filesystem::exists(hdf5_filename));

  auto h3 = Hamiltonian::from_file(hdf5_filename, "hdf5");

  EXPECT_EQ(h3->get_orbitals()->get_num_molecular_orbitals(), 2);
  auto [h3_one_alpha, h3_one_beta] = h3->get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(h3_one_alpha));
}

TEST_F(HamiltonianTest, InvalidFileType) {
  // Create a Hamiltonian for testing
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  EXPECT_THROW(h.to_file("test.txt", "txt"), std::runtime_error);
  EXPECT_THROW(Hamiltonian::from_file("test.txt", "txt"), std::runtime_error);
}

TEST_F(HamiltonianTest, FileNotFound) {
  EXPECT_THROW(Hamiltonian::from_json_file("nonexistent.hamiltonian.json"),
               std::runtime_error);
  EXPECT_THROW(Hamiltonian::from_hdf5_file("nonexistent.hamiltonian.h5"),
               std::runtime_error);
}

TEST_F(HamiltonianTest, ValidationTests) {
  // Test validation of integral dimensions during construction
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
}

TEST_F(HamiltonianTest, ValidationEdgeCases) {
  // Test edge cases for validation during construction

  // Test with 1x1 matrices (smallest valid case)
  Eigen::MatrixXd tiny_one_body = Eigen::MatrixXd::Identity(1, 1);
  Eigen::VectorXd tiny_two_body = Eigen::VectorXd::Random(1);  // 1^4 = 1
  auto tiny_orbitals =
      std::make_shared<ModelOrbitals>(1, true);  // 1 orbital, restricted
  Eigen::MatrixXd tiny_inactive_fock = Eigen::MatrixXd::Zero(1, 1);

  EXPECT_NO_THROW(
      Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          tiny_one_body, tiny_two_body, tiny_orbitals, core_energy,
          tiny_inactive_fock)));

  // Test with large matrices (stress test)
  Eigen::MatrixXd large_one_body = Eigen::MatrixXd::Identity(10, 10);
  Eigen::VectorXd large_two_body =
      Eigen::VectorXd::Random(10000);  // 10^4 = 10000

  // Need orbitals that match the 10x10 size
  Eigen::MatrixXd large_coeffs = Eigen::MatrixXd::Identity(10, 10);

  auto large_orbitals =
      std::make_shared<ModelOrbitals>(10, true);  // 10 orbitals, restricted

  // Create a larger inactive_fock matrix for this test
  Eigen::MatrixXd large_inactive_fock = Eigen::MatrixXd::Zero(0, 0);

  EXPECT_NO_THROW(
      Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          large_one_body, large_two_body, large_orbitals, core_energy,
          large_inactive_fock)));

  // Test wrong size by one element
  Eigen::MatrixXd three_by_three = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd off_by_one =
      Eigen::VectorXd::Random(80);  // Should be 81 for 3x3

  EXPECT_THROW(
      Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          three_by_three, off_by_one, orbitals, core_energy, inactive_fock)),
      std::invalid_argument);
}

TEST_F(HamiltonianConstructorTest, Factory) {
  auto available_solvers = HamiltonianConstructorFactory::available();
  EXPECT_EQ(available_solvers.size(), 1);
  EXPECT_EQ(available_solvers[0], "qdk");
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

TEST_F(HamiltonianTest, UnrestrictedConstructor) {
  // Create unrestricted orbitals for this test
  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(2, false);

  // Create different alpha and beta matrices to test unrestricted functionality
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Random(2, 2);

  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Random(16);
  Eigen::VectorXd two_body_aabb = Eigen::VectorXd::Random(16);
  Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Random(16);

  Eigen::MatrixXd inactive_fock_alpha = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd inactive_fock_beta = Eigen::MatrixXd::Random(2, 2);

  Hamiltonian h_unrestricted(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, unrestricted_orbitals, core_energy,
          inactive_fock_alpha, inactive_fock_beta));

  // Verify the unrestricted Hamiltonian was created successfully
  EXPECT_TRUE(h_unrestricted.has_one_body_integrals());
  EXPECT_TRUE(h_unrestricted.has_two_body_integrals());
  EXPECT_TRUE(h_unrestricted.has_orbitals());
  EXPECT_TRUE(h_unrestricted.has_inactive_fock_matrix());
  EXPECT_EQ(h_unrestricted.get_core_energy(), core_energy);
  EXPECT_FALSE(h_unrestricted.is_restricted());
  EXPECT_TRUE(h_unrestricted.is_unrestricted());
}

TEST_F(HamiltonianTest, UnrestrictedAccessorMethods) {
  // Create unrestricted orbitals for this test
  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(2, false);

  // Create different alpha and beta data
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Ones(2, 2);

  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Constant(16, 1.0);
  Eigen::VectorXd two_body_aabb = Eigen::VectorXd::Constant(16, 2.0);
  Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Constant(16, 3.0);

  Eigen::MatrixXd inactive_fock_alpha = Eigen::MatrixXd::Constant(2, 2, 4.0);
  Eigen::MatrixXd inactive_fock_beta = Eigen::MatrixXd::Constant(2, 2, 5.0);

  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
      two_body_bbbb, unrestricted_orbitals, core_energy, inactive_fock_alpha,
      inactive_fock_beta));

  // Test alpha/beta one-body integral access
  auto [h_one_alpha, h_one_beta] = h.get_one_body_integrals();
  EXPECT_TRUE(h_one_alpha.isApprox(one_body_alpha));
  EXPECT_TRUE(h_one_beta.isApprox(one_body_beta));

  // Test tuple access for two-body integrals
  auto [aaaa, aabb, bbbb] = h.get_two_body_integrals();
  EXPECT_TRUE(aaaa.isApprox(two_body_aaaa));
  EXPECT_TRUE(aabb.isApprox(two_body_aabb));
  EXPECT_TRUE(bbbb.isApprox(two_body_bbbb));
  // Test alpha/beta inactive Fock matrix access
  auto fock_matrices = h.get_inactive_fock_matrix();
  EXPECT_TRUE(fock_matrices.first.isApprox(inactive_fock_alpha));
  EXPECT_TRUE(fock_matrices.second.isApprox(inactive_fock_beta));
}

TEST_F(HamiltonianTest, RestrictedVsUnrestrictedDetection) {
  // Create restricted Hamiltonian using the first constructor
  Hamiltonian h_restricted(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body, two_body, orbitals, core_energy, inactive_fock));

  // Create unrestricted orbitals for the unrestricted test
  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(2, false);

  // Create unrestricted Hamiltonian with different alpha/beta data
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Ones(2, 2);
  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Constant(16, 1.0);
  Eigen::VectorXd two_body_aabb = Eigen::VectorXd::Constant(16, 2.0);
  Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Constant(16, 3.0);
  Eigen::MatrixXd inactive_fock_alpha = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd inactive_fock_beta = Eigen::MatrixXd::Ones(2, 2);

  Hamiltonian h_unrestricted(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, unrestricted_orbitals, core_energy,
          inactive_fock_alpha, inactive_fock_beta));

  // Test restricted detection
  EXPECT_TRUE(h_restricted.is_restricted());
  EXPECT_FALSE(h_restricted.is_unrestricted());

  // Test unrestricted detection
  EXPECT_FALSE(h_unrestricted.is_restricted());
  EXPECT_TRUE(h_unrestricted.is_unrestricted());
}

TEST_F(HamiltonianTest, UnrestrictedSpinChannelAccess) {
  // Create unrestricted orbitals for this test
  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(2, false);

  // Create unrestricted Hamiltonian with specific two-body integral values
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Identity(2, 2);

  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Zero(16);
  Eigen::VectorXd two_body_aabb = Eigen::VectorXd::Zero(16);
  Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Zero(16);

  // Set specific values for each spin channel
  two_body_aaaa[0] = 1.0;   // (0,0,0,0) in aaaa channel
  two_body_aabb[5] = 2.0;   // (0,1,0,1) in aabb channel
  two_body_bbbb[15] = 3.0;  // (1,1,1,1) in bbbb channel

  Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);

  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
      two_body_bbbb, unrestricted_orbitals, core_energy, empty_fock,
      empty_fock));

  // Test accessing elements through different spin channels
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::aaaa), 1.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 1, 0, 1, SpinChannel::aabb), 2.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(1, 1, 1, 1, SpinChannel::bbbb), 3.0);

  // Verify other elements are zero
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::aabb), 0.0);
  EXPECT_DOUBLE_EQ(h.get_two_body_element(0, 0, 0, 0, SpinChannel::bbbb), 0.0);
}

TEST_F(HamiltonianTest, UnrestrictedJSONSerialization) {
  // Create unrestricted orbitals for this test
  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(2, false);

  // Create unrestricted Hamiltonian
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Random(2, 2);
  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Random(16);
  Eigen::VectorXd two_body_aabb = Eigen::VectorXd::Random(16);
  Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Random(16);
  Eigen::MatrixXd inactive_fock_alpha = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd inactive_fock_beta = Eigen::MatrixXd::Random(2, 2);

  Hamiltonian h_orig(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
      two_body_bbbb, unrestricted_orbitals, core_energy, inactive_fock_alpha,
      inactive_fock_beta));

  // Test JSON serialization round-trip
  nlohmann::json j = h_orig.to_json();
  auto h_loaded = Hamiltonian::from_json(j);

  // Verify the loaded Hamiltonian matches the original
  EXPECT_EQ(h_loaded->get_core_energy(), core_energy);
  EXPECT_FALSE(h_loaded->is_restricted());
  EXPECT_TRUE(h_loaded->is_unrestricted());

  auto [orig_one_alpha, orig_one_beta] = h_orig.get_one_body_integrals();
  auto [loaded_one_alpha, loaded_one_beta] = h_loaded->get_one_body_integrals();
  EXPECT_TRUE(orig_one_alpha.isApprox(loaded_one_alpha));
  EXPECT_TRUE(orig_one_beta.isApprox(loaded_one_beta));

  auto [orig_two_aaaa, orig_two_aabb, orig_two_bbbb] =
      h_orig.get_two_body_integrals();
  auto [loaded_two_aaaa, loaded_two_aabb, loaded_two_bbbb] =
      h_loaded->get_two_body_integrals();
  EXPECT_TRUE(orig_two_aaaa.isApprox(loaded_two_aaaa));
  EXPECT_TRUE(orig_two_aabb.isApprox(loaded_two_aabb));
  EXPECT_TRUE(orig_two_bbbb.isApprox(loaded_two_bbbb));

  auto [h_orig_alpha, h_orig_beta] = h_orig.get_inactive_fock_matrix();
  auto [h_loaded_alpha, h_loaded_beta] = h_loaded->get_inactive_fock_matrix();
  EXPECT_TRUE(h_orig_alpha.isApprox(h_loaded_alpha));
  EXPECT_TRUE(h_orig_beta.isApprox(h_loaded_beta));
}

TEST_F(HamiltonianTest, UnrestrictedHDF5Serialization) {
  // Create unrestricted orbitals for this test
  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(2, false);

  // Create unrestricted Hamiltonian
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Random(2, 2);
  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Random(16);
  Eigen::VectorXd two_body_aabb = Eigen::VectorXd::Random(16);
  Eigen::VectorXd two_body_bbbb = Eigen::VectorXd::Random(16);
  Eigen::MatrixXd inactive_fock_alpha = Eigen::MatrixXd::Random(2, 2);
  Eigen::MatrixXd inactive_fock_beta = Eigen::MatrixXd::Random(2, 2);

  Hamiltonian h_orig(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
      two_body_bbbb, unrestricted_orbitals, core_energy, inactive_fock_alpha,
      inactive_fock_beta));

  // Test HDF5 serialization round-trip
  std::string filename = "test_unrestricted.hamiltonian.h5";
  h_orig.to_hdf5_file(filename);

  auto h_loaded = Hamiltonian::from_hdf5_file(filename);

  // Verify the loaded Hamiltonian matches the original
  EXPECT_EQ(h_loaded->get_core_energy(), core_energy);
  EXPECT_FALSE(h_loaded->is_restricted());
  EXPECT_TRUE(h_loaded->is_unrestricted());

  auto [orig_one_alpha, orig_one_beta] = h_orig.get_one_body_integrals();
  auto [loaded_one_alpha, loaded_one_beta] = h_loaded->get_one_body_integrals();
  EXPECT_TRUE(orig_one_alpha.isApprox(loaded_one_alpha));
  EXPECT_TRUE(orig_one_beta.isApprox(loaded_one_beta));

  auto [orig_two_aaaa, orig_two_aabb, orig_two_bbbb] =
      h_orig.get_two_body_integrals();
  auto [loaded_two_aaaa, loaded_two_aabb, loaded_two_bbbb] =
      h_loaded->get_two_body_integrals();
  EXPECT_TRUE(orig_two_aaaa.isApprox(loaded_two_aaaa));
  EXPECT_TRUE(orig_two_aabb.isApprox(loaded_two_aabb));
  EXPECT_TRUE(orig_two_bbbb.isApprox(loaded_two_bbbb));

  auto [h_orig_alpha, h_orig_beta] = h_orig.get_inactive_fock_matrix();
  auto [h_loaded_alpha, h_loaded_beta] = h_loaded->get_inactive_fock_matrix();
  EXPECT_TRUE(h_orig_alpha.isApprox(h_loaded_alpha));
  EXPECT_TRUE(h_orig_beta.isApprox(h_loaded_beta));
}

TEST_F(HamiltonianTest, FCIDUMPSerialization) {
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  // Test FCIDUMP serialization
  h.to_fcidump_file("test.hamiltonian.fcidump", 1, 1);

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

TEST_F(HamiltonianTest, FCIDUMPSerializationUnrestrictedThrowsError) {
  // Create unrestricted orbitals for this test
  auto unrestricted_orbitals = std::make_shared<ModelOrbitals>(2, false);

  // Create different alpha and beta matrices
  Eigen::MatrixXd one_body_alpha = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd one_body_beta = Eigen::MatrixXd::Ones(2, 2);

  Eigen::VectorXd two_body_aaaa = Eigen::VectorXd::Ones(16);
  Eigen::VectorXd two_body_aabb = 2 * Eigen::VectorXd::Ones(16);
  Eigen::VectorXd two_body_bbbb = 3 * Eigen::VectorXd::Ones(16);

  Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);

  // Create unrestricted Hamiltonian
  Hamiltonian h_unrestricted(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
          two_body_bbbb, unrestricted_orbitals, core_energy, empty_fock,
          empty_fock));

  // Verify it's actually unrestricted
  EXPECT_TRUE(h_unrestricted.is_unrestricted());
  EXPECT_FALSE(h_unrestricted.is_restricted());

  // Test that FCIDUMP serialization throws an error for unrestricted case
  EXPECT_THROW(h_unrestricted.to_fcidump_file(
                   "test_unrestricted.hamiltonian.fcidump", 1, 1),
               std::runtime_error);
}

TEST_F(HamiltonianTest, FCIDUMPActiveSpaceConsistency) {
  // Test that FCIDUMP correctly handles the active space indices properly
  // Create orbitals with a specific active space setup
  std::vector<size_t> active_indices = {0,
                                        1};    // Use first 2 orbitals as active
  std::vector<size_t> inactive_indices = {2};  // Third orbital is inactive

  auto orbitals_with_active_space = std::make_shared<ModelOrbitals>(
      3,
      std::make_tuple(std::move(active_indices), std::move(inactive_indices)));

  // Create 2x2 matrices for the active space
  Eigen::MatrixXd one_body_2x2 = Eigen::MatrixXd::Identity(2, 2);
  one_body_2x2(0, 1) = 0.5;
  one_body_2x2(1, 0) = 0.5;

  Eigen::VectorXd two_body_2x2 = 2 * Eigen::VectorXd::Ones(16);  // 2^4 = 16

  // Create appropriate inactive Fock matrix for the inactive space
  Eigen::MatrixXd inactive_fock_2x2 = Eigen::MatrixXd::Zero(2, 2);

  Hamiltonian h_active_space(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          one_body_2x2, two_body_2x2, orbitals_with_active_space, core_energy,
          inactive_fock_2x2));

  // Should successfully write FCIDUMP using active space dimensions
  EXPECT_NO_THROW({
    h_active_space.to_fcidump_file("test_active_space.hamiltonian.fcidump", 1,
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

TEST_F(HamiltonianTest, ErrorHandlingUnrestrictedMismatchedActiveSpace) {
  // Test error handling when alpha and beta active spaces have different sizes
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

  Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);

  // This should throw during construction due to dimension mismatch
  EXPECT_THROW(
      {
        Hamiltonian h_mismatched(
            std::make_unique<CanonicalFourCenterHamiltonianContainer>(
                one_body_alpha, one_body_beta, two_body_aaaa, two_body_aabb,
                two_body_bbbb, unrestricted_orbitals, core_energy, empty_fock,
                empty_fock));
      },
      std::invalid_argument);
}

TEST_F(HamiltonianTest, IntegralSymmetriesEnergiesO2Singlet) {
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

TEST_F(HamiltonianTest, MixedIntegralSymmetriesO2Triplet) {
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

TEST_F(HamiltonianTest, O2DeterministicBehaviorRestrictedUnrestricted) {
  // Test that repeated calculations give identical integral elements
  // for both restricted (singlet) and unrestricted (triplet) O2

  // Test restricted O2 deterministic behavior
  {
    auto [energy1, hamiltonian1] = run_restricted_o2();
    auto [energy2, hamiltonian2] = run_restricted_o2();

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

TEST_F(HamiltonianTest, IsValidComprehensive) {
  // Valid Hamiltonian with all required data
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  // Valid Hamiltonian with inactive Fock matrix
  Eigen::MatrixXd inactive_fock_matrix = Eigen::MatrixXd::Random(2, 2);
  Hamiltonian h2(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock_matrix));

  // Construction with mismatched dimensions should fail
  Eigen::MatrixXd wrong_one_body = Eigen::MatrixXd::Identity(3, 3);  // 3x3
  Eigen::VectorXd wrong_two_body = Eigen::VectorXd::Random(16);      // 2^4

  EXPECT_THROW(
      Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          wrong_one_body, wrong_two_body, orbitals, core_energy,
          inactive_fock)),
      std::invalid_argument);

  // Non-square one-body matrix should fail during construction
  Eigen::MatrixXd non_square(2, 3);  // 2x3 matrix
  non_square.setRandom();
  EXPECT_THROW(
      Hamiltonian(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          non_square, two_body, orbitals, core_energy, inactive_fock)),
      std::invalid_argument);
}

// Dummy container type for testing get_container bad_cast
class DummyHamiltonianContainer : public HamiltonianContainer {
 public:
  DummyHamiltonianContainer(const Eigen::MatrixXd& one_body,
                            std::shared_ptr<Orbitals> orbitals,
                            double core_energy,
                            const Eigen::MatrixXd& inactive_fock)
      : HamiltonianContainer(one_body, orbitals, core_energy, inactive_fock) {}

  std::unique_ptr<HamiltonianContainer> clone() const override {
    return std::make_unique<DummyHamiltonianContainer>(*this);
  }

  std::string get_container_type() const override { return "dummy"; }

  std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
             const Eigen::VectorXd&>
  get_two_body_integrals() const override {
    static Eigen::VectorXd empty;
    return {empty, empty, empty};
  }

  double get_two_body_element(unsigned, unsigned, unsigned, unsigned,
                              SpinChannel) const override {
    return 0.0;
  }

  bool has_two_body_integrals() const override { return false; }
  bool is_restricted() const override { return true; }
  nlohmann::json to_json() const override { return {}; }
  void to_hdf5(H5::Group&) const override {}
  void to_fcidump_file(const std::string&, size_t, size_t) const override {}
  bool is_valid() const override { return true; }
};

TEST_F(HamiltonianTest, GetContainerTypedAccess) {
  // Create a Hamiltonian with CanonicalFourCenterHamiltonianContainer container
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  // Test successful typed container access
  EXPECT_NO_THROW({
    const auto& container =
        h.get_container<CanonicalFourCenterHamiltonianContainer>();
    EXPECT_EQ(container.get_container_type(), "canonical_four_center");
  });

  // Verify has_container_type returns true for correct type
  EXPECT_TRUE(h.has_container_type<CanonicalFourCenterHamiltonianContainer>());

  // Verify has_container_type returns false for incorrect type
  EXPECT_FALSE(h.has_container_type<DummyHamiltonianContainer>());

  // Test that accessing with incorrect container type throws std::bad_cast
  EXPECT_THROW(h.get_container<DummyHamiltonianContainer>(), std::bad_cast);
}

TEST_F(HamiltonianTest, DataTypeName) {
  // Test that Hamiltonian has the correct data type name
  Hamiltonian h(std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      one_body, two_body, orbitals, core_energy, inactive_fock));

  EXPECT_EQ(h.get_data_type_name(), "hamiltonian");
}
