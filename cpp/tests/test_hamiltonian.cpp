// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>

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

class TestHamiltonianConstructor : public HamiltonianConstructor {
 public:
  std::string name() const override { return "test-hamiltonian_constructor"; }
  std::shared_ptr<Hamiltonian> _run_impl(
      std::shared_ptr<Orbitals> orbitals) const override {
    // Dummy implementation for testing
    Eigen::MatrixXd one_body = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd two_body = Eigen::VectorXd::Random(81);
    Eigen::MatrixXd f_inact = Eigen::MatrixXd::Identity(0, 0);
    return std::make_shared<Hamiltonian>(one_body, two_body, orbitals, 0.0,
                                         f_inact);
  }
};

TEST_F(HamiltonianTest, Constructor) {
  // Test the constructor with all required data
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

  EXPECT_TRUE(h.has_one_body_integrals());
  EXPECT_TRUE(h.has_two_body_integrals());
  EXPECT_TRUE(h.has_orbitals());
  EXPECT_EQ(h.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h.get_core_energy(), 1.5);
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
  Hamiltonian h(one_body, two_body, orbitals_with_inactive, core_energy,
                non_empty_inactive_fock);

  EXPECT_TRUE(h.has_one_body_integrals());
  EXPECT_TRUE(h.has_two_body_integrals());
  EXPECT_TRUE(h.has_orbitals());
  EXPECT_TRUE(h.has_inactive_fock_matrix());
  EXPECT_EQ(h.get_orbitals()->get_num_molecular_orbitals(), 4);
  EXPECT_EQ(h.get_core_energy(), 1.5);
}

TEST_F(HamiltonianTest, FullConstructor) {
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

  EXPECT_TRUE(h.has_one_body_integrals());
  EXPECT_TRUE(h.has_two_body_integrals());
  EXPECT_TRUE(h.has_orbitals());
  EXPECT_EQ(h.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h.get_core_energy(), 1.5);
}

TEST_F(HamiltonianTest, CopyConstructor) {
  Hamiltonian h1(one_body, two_body, orbitals, core_energy, inactive_fock);
  Hamiltonian h2(h1);

  EXPECT_TRUE(h2.has_one_body_integrals());
  EXPECT_TRUE(h2.has_two_body_integrals());
  EXPECT_TRUE(h2.has_orbitals());
  EXPECT_EQ(h2.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2.get_core_energy(), 1.5);
}

TEST_F(HamiltonianTest, MoveConstructor) {
  Hamiltonian h1(one_body, two_body, orbitals, core_energy, inactive_fock);
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
  Hamiltonian h1(one_body, two_body, orbitals, core_energy, inactive_fock);

  // Test copy constructor
  Hamiltonian h2(h1);

  // Verify all data was copied correctly
  EXPECT_TRUE(h2.has_one_body_integrals());
  EXPECT_TRUE(h2.has_two_body_integrals());
  EXPECT_TRUE(h2.has_orbitals());
  EXPECT_TRUE(h2.has_inactive_fock_matrix());
  EXPECT_EQ(h2.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2.get_core_energy(), 1.5);

  // Verify deep copying of matrices
  EXPECT_TRUE(
      h1.get_one_body_integrals().isApprox(h2.get_one_body_integrals()));
  EXPECT_TRUE(
      h1.get_two_body_integrals().isApprox(h2.get_two_body_integrals()));
  EXPECT_TRUE(h1.get_inactive_fock_matrix().first.isApprox(
      h2.get_inactive_fock_matrix().first));

  // Test copy assignment
  Hamiltonian h3(one_body, two_body, orbitals, core_energy, inactive_fock);
  h3 = h1;

  // Verify assignment worked correctly
  EXPECT_TRUE(h3.has_one_body_integrals());
  EXPECT_TRUE(h3.has_two_body_integrals());
  EXPECT_TRUE(h3.has_orbitals());
  EXPECT_TRUE(h3.has_inactive_fock_matrix());
  EXPECT_EQ(h3.get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h3.get_core_energy(), 1.5);

  // Test self-assignment (should be no-op)
  Hamiltonian h4(one_body, two_body, orbitals, core_energy, inactive_fock);
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

  Hamiltonian h(test_one_body, test_two_body, orbitals, core_energy,
                inactive_fock);

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

  Hamiltonian h_large(large_one_body, large_two_body, large_orbitals, 0.0,
                      large_inact_f);

  EXPECT_DOUBLE_EQ(h_large.get_two_body_element(2, 1, 0, 2), 7.0);
  EXPECT_DOUBLE_EQ(h_large.get_two_body_element(1, 2, 2, 1), 8.0);
}

TEST_F(HamiltonianTest, JSONSerialization) {
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

  // Test JSON conversion
  nlohmann::json j = h.to_json();

  EXPECT_EQ(j["core_energy"], 1.5);
  EXPECT_TRUE(j["has_one_body_integrals"]);
  EXPECT_TRUE(j["has_two_body_integrals"]);
  EXPECT_TRUE(j["has_orbitals"]);

  // Test round-trip conversion
  auto h2 = Hamiltonian::from_json(j);

  EXPECT_EQ(h2->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_EQ(h2->get_core_energy(), 1.5);
  EXPECT_TRUE(h2->has_one_body_integrals());
  EXPECT_TRUE(h2->has_two_body_integrals());
  EXPECT_TRUE(h2->has_orbitals());

  // Check that matrices are approximately equal
  EXPECT_TRUE(
      h.get_one_body_integrals().isApprox(h2->get_one_body_integrals()));
  EXPECT_TRUE(
      h.get_two_body_integrals().isApprox(h2->get_two_body_integrals()));
}

TEST_F(HamiltonianTest, JSONFileIO) {
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

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
  EXPECT_TRUE(
      h.get_one_body_integrals().isApprox(h2->get_one_body_integrals()));
  EXPECT_TRUE(
      h.get_two_body_integrals().isApprox(h2->get_two_body_integrals()));
}

TEST_F(HamiltonianTest, HDF5FileIO) {
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

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
  EXPECT_TRUE(
      h.get_one_body_integrals().isApprox(h2->get_one_body_integrals()));
  EXPECT_TRUE(
      h.get_two_body_integrals().isApprox(h2->get_two_body_integrals()));
}

TEST_F(HamiltonianTest, FCIDUMPSerialization) {
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

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

TEST_F(HamiltonianTest, GenericFileIO) {
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

  // Test JSON via generic interface
  std::string json_filename = "test.hamiltonian.json";
  h.to_file(json_filename, "json");
  EXPECT_TRUE(std::filesystem::exists(json_filename));

  auto h2 = Hamiltonian::from_file(json_filename, "json");

  EXPECT_EQ(h2->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_TRUE(
      h.get_one_body_integrals().isApprox(h2->get_one_body_integrals()));

  // Test HDF5 via generic interface
  std::string hdf5_filename = "test.hamiltonian.h5";
  h.to_file(hdf5_filename, "hdf5");
  EXPECT_TRUE(std::filesystem::exists(hdf5_filename));

  auto h3 = Hamiltonian::from_file(hdf5_filename, "hdf5");

  EXPECT_EQ(h3->get_orbitals()->get_num_molecular_orbitals(), 2);
  EXPECT_TRUE(
      h.get_one_body_integrals().isApprox(h3->get_one_body_integrals()));
}

TEST_F(HamiltonianTest, InvalidFileType) {
  // Create a Hamiltonian for testing
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

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

  EXPECT_THROW(Hamiltonian(bad_one_body, bad_two_body, orbitals, core_energy,
                           inactive_fock),
               std::invalid_argument);

  // Test validation with non-square one-body matrix
  Eigen::MatrixXd non_square_one_body(2, 3);  // 2x3 non-square matrix
  non_square_one_body.setRandom();
  Eigen::VectorXd any_two_body = Eigen::VectorXd::Random(36);

  EXPECT_THROW(Hamiltonian(non_square_one_body, any_two_body, orbitals,
                           core_energy, inactive_fock),
               std::invalid_argument);

  // Test validation passes with correct dimensions
  Eigen::MatrixXd correct_one_body = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd correct_two_body = Eigen::VectorXd::Random(16);  // 2^4 = 16

  EXPECT_NO_THROW(Hamiltonian(correct_one_body, correct_two_body, orbitals,
                              core_energy, inactive_fock));
}

TEST_F(HamiltonianTest, ValidationEdgeCases) {
  // Test edge cases for validation during construction

  // Test with 1x1 matrices (smallest valid case)
  Eigen::MatrixXd tiny_one_body = Eigen::MatrixXd::Identity(1, 1);
  Eigen::VectorXd tiny_two_body = Eigen::VectorXd::Random(1);  // 1^4 = 1
  auto tiny_orbitals =
      std::make_shared<ModelOrbitals>(1, true);  // 1 orbital, restricted
  Eigen::MatrixXd tiny_inactive_fock = Eigen::MatrixXd::Zero(1, 1);

  EXPECT_NO_THROW(Hamiltonian(tiny_one_body, tiny_two_body, tiny_orbitals,
                              core_energy, tiny_inactive_fock));

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

  EXPECT_NO_THROW(Hamiltonian(large_one_body, large_two_body, large_orbitals,
                              core_energy, large_inactive_fock));

  // Test wrong size by one element
  Eigen::MatrixXd three_by_three = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd off_by_one =
      Eigen::VectorXd::Random(80);  // Should be 81 for 3x3

  EXPECT_THROW(Hamiltonian(three_by_three, off_by_one, orbitals, core_energy,
                           inactive_fock),
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

  // Create basis set of appropriate size for tests
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}));

  auto basis_set = std::make_shared<BasisSet>("test", shells);

  // Throw if basis set is not set in orbitals
  EXPECT_THROW(
      {
        // Create model orbitals without basis set
        auto orbitals =
            std::make_shared<ModelOrbitals>(3, true);  // 3 orbitals, restricted
        hc->run(orbitals);
      },
      std::runtime_error);

  // 1P fails for unrestricted orbitals
  EXPECT_THROW(
      {
        Eigen::MatrixXd coeffs_alpha = Eigen::MatrixXd::Identity(3, 3);
        Eigen::MatrixXd coeffs_beta = Eigen::MatrixXd::Identity(3, 3);
        // Create unrestricted orbitals with basis set
        auto orbitals = std::make_shared<Orbitals>(
            coeffs_alpha, coeffs_beta, std::nullopt, std::nullopt, std::nullopt,
            basis_set, std::nullopt);
        hc->run(orbitals);
      },
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
}

TEST_F(HamiltonianTest, IsValidComprehensive) {
  // Test case 1: Valid Hamiltonian with all required data
  Hamiltonian h(one_body, two_body, orbitals, core_energy, inactive_fock);

  // Test case 2: Valid Hamiltonian with inactive Fock matrix
  Eigen::MatrixXd inactive_fock = Eigen::MatrixXd::Random(2, 2);
  Hamiltonian h2(one_body, two_body, orbitals, core_energy, inactive_fock);

  // Test case 3: Construction with mismatched dimensions should fail
  Eigen::MatrixXd wrong_one_body = Eigen::MatrixXd::Identity(3, 3);  // 3x3
  Eigen::VectorXd wrong_two_body = Eigen::VectorXd::Random(16);      // 2^4

  EXPECT_THROW(Hamiltonian(wrong_one_body, wrong_two_body, orbitals,
                           core_energy, inactive_fock),
               std::invalid_argument);

  // Test case 4: Non-square one-body matrix should fail during construction
  Eigen::MatrixXd non_square(2, 3);  // 2x3 matrix
  non_square.setRandom();
  EXPECT_THROW(
      Hamiltonian(non_square, two_body, orbitals, core_energy, inactive_fock),
      std::invalid_argument);
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
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{0.5}, std::vector{1.0}));
  shells.emplace_back(
      Shell(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0}));
  shells.emplace_back(
      Shell(1, OrbitalType::S, std::vector{0.5}, std::vector{1.0}));
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
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{2.0}, std::vector{1.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{0.8}, std::vector{1.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{0.3}, std::vector{1.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::P, std::vector{1.0}, std::vector{1.0}));
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
