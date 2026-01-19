// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "ut_common.hpp"

#define EXPECT_STRING_CONTAINS(str, query)                            \
  do {                                                                \
    std::string full_str(str);                                        \
    if (full_str.find(query) == std::string::npos) {                  \
      FAIL() << "Expected string to contain \"" << query              \
             << "\" but it was not found in: \"" << full_str << "\""; \
    }                                                                 \
  } while (0)

using namespace qdk::chemistry::data;

class BasisSetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean up any test files from previous runs
    std::error_code ec;
    std::filesystem::remove("test.basis_set.json", ec);
  }

  void TearDown() override {
    // Clean up test files
    std::error_code ec;
    std::filesystem::remove("test.basis_set.json", ec);
  }
};

TEST_F(BasisSetTest, ShellConstructors) {
  // Constructor with empty primitive data
  Eigen::VectorXd empty_exp(0);
  Eigen::VectorXd empty_coeff(0);
  Shell shell_basic(0, OrbitalType::S, empty_exp, empty_coeff);
  EXPECT_EQ(0u, shell_basic.atom_index);
  EXPECT_EQ(OrbitalType::S, shell_basic.orbital_type);
  EXPECT_TRUE(shell_basic.exponents.size() == 0);
  EXPECT_TRUE(shell_basic.coefficients.size() == 0);

  // Constructor with empty primitive data for P orbital
  Shell shell_with_params(1, OrbitalType::P, empty_exp, empty_coeff);
  EXPECT_EQ(1u, shell_with_params.atom_index);
  EXPECT_EQ(OrbitalType::P, shell_with_params.orbital_type);
  EXPECT_TRUE(shell_with_params.exponents.size() == 0);
  EXPECT_TRUE(shell_with_params.coefficients.size() == 0);

  // Constructor with primitive data
  Eigen::VectorXd exponents(3);
  exponents << 1.0, 2.0, 3.0;
  Eigen::VectorXd coefficients(3);
  coefficients << 0.5, 0.5, 0.5;

  Shell shell_with_data(2, OrbitalType::D, exponents, coefficients);
  EXPECT_EQ(2u, shell_with_data.atom_index);
  EXPECT_EQ(OrbitalType::D, shell_with_data.orbital_type);
  EXPECT_EQ(exponents.size(), shell_with_data.exponents.size());
  EXPECT_EQ(coefficients.size(), shell_with_data.coefficients.size());
}

TEST_F(BasisSetTest, Constructors) {
  // Create a simple structure for testing
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure structure(coords, symbols);

  // Constructor with empty name and structure
  EXPECT_THROW(BasisSet basis1("", structure), std::invalid_argument);

  // Constructor with name and structure should throw (empty basis invalid)
  EXPECT_THROW(BasisSet basis2("6-31G", structure), std::invalid_argument);

  // Constructor with name, structure and basis type should throw (empty basis
  // invalid)
  EXPECT_THROW(BasisSet basis3("6-31G", structure, AOType::Cartesian),
               std::invalid_argument);

  // Constructor with shells should work
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));
  BasisSet basis4("6-31G", shells, structure);
  EXPECT_EQ(std::string("6-31G"), basis4.get_name());
  EXPECT_EQ(AOType::Spherical, basis4.get_atomic_orbital_type());
  EXPECT_EQ(1u, basis4.get_num_shells());

  // Copy constructor
  BasisSet basis5(basis4);
  EXPECT_EQ(std::string("6-31G"), basis5.get_name());
  EXPECT_EQ(AOType::Spherical, basis5.get_atomic_orbital_type());
}

TEST_F(BasisSetTest, CopyConstructorAndAssignment) {
  // Create a structure
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure structure(coords, symbols);

  // Create shells
  std::vector<Shell> shells;

  // Create first shell with Eigen vectors
  Eigen::VectorXd s_exp(1), s_coeff(1);
  s_exp << 1.0;
  s_coeff << 2.0;
  shells.emplace_back(0, OrbitalType::S, s_exp, s_coeff);

  Eigen::VectorXd p_exp(1), p_coeff(1);
  p_exp << 0.5;
  p_coeff << 1.0;
  shells.emplace_back(0, OrbitalType::P, p_exp, p_coeff);

  // Create source basis set with data and structure
  BasisSet source("cc-pVDZ", shells, structure, AOType::Cartesian);

  // Test copy constructor
  BasisSet copy_constructed(source);
  EXPECT_EQ(std::string("cc-pVDZ"), copy_constructed.get_name());
  EXPECT_EQ(AOType::Cartesian, copy_constructed.get_atomic_orbital_type());
  EXPECT_EQ(2u, copy_constructed.get_num_shells());
  EXPECT_EQ(4u, copy_constructed.get_num_atomic_orbitals());  // 1 s + 3 p

  // Verify shells were copied
  const auto& atom_shells = copy_constructed.get_shells_for_atom(0);
  EXPECT_EQ(2u, atom_shells.size());
  EXPECT_EQ(OrbitalType::S, atom_shells[0].orbital_type);
  EXPECT_EQ(OrbitalType::P, atom_shells[1].orbital_type);

  // Verify structure was copied
  EXPECT_TRUE(copy_constructed.has_structure());
  auto copy_structure = copy_constructed.get_structure();
  EXPECT_EQ(1u, copy_structure->get_num_atoms());

  // Test copy assignment
  std::vector<Shell> different_shells;
  different_shells.emplace_back(
      Shell(1, OrbitalType::D, std::vector{2.0}, std::vector{3.0}));
  BasisSet target("STO-3G", different_shells, AOType::Spherical);

  target = source;

  // Verify all properties were copied correctly
  EXPECT_EQ(std::string("cc-pVDZ"), target.get_name());
  EXPECT_EQ(AOType::Cartesian, target.get_atomic_orbital_type());
  EXPECT_EQ(2u, target.get_num_shells());
  EXPECT_EQ(4u, target.get_num_atomic_orbitals());  // 1 s + 3 p

  // Test self-assignment (should be no-op)
  BasisSet& self_ref = target;
  self_ref = target;
  EXPECT_EQ(std::string("cc-pVDZ"), target.get_name());
  EXPECT_EQ(AOType::Cartesian, target.get_atomic_orbital_type());

  // Test assignment from basis set without structure
  std::vector<Shell> minimal_shells;
  std::vector<double> s_exp2 = {1.5};
  std::vector<double> s_coeff2 = {2.5};
  minimal_shells.emplace_back(0, OrbitalType::S, s_exp2, s_coeff2);
  BasisSet no_structure_basis("minimal", minimal_shells);

  target = no_structure_basis;
  EXPECT_EQ(std::string("minimal"), target.get_name());
  EXPECT_FALSE(target.has_structure());  // Structure should be reset
  EXPECT_EQ(1u, target.get_num_shells());
}

TEST_F(BasisSetTest, ShellMemberFunctions) {
  // Test shell with primitive data
  Eigen::VectorXd exponents(2);
  exponents << 1.0, 2.0;
  Eigen::VectorXd coefficients(2);
  coefficients << 0.5, 0.3;

  Shell shell(0, OrbitalType::S, exponents, coefficients);
  EXPECT_EQ(2u, shell.get_num_primitives());
  EXPECT_EQ(2u, shell.exponents.size());
  EXPECT_EQ(2u, shell.coefficients.size());
  EXPECT_DOUBLE_EQ(1.0, shell.exponents(0));
  EXPECT_DOUBLE_EQ(0.5, shell.coefficients(0));
  EXPECT_DOUBLE_EQ(2.0, shell.exponents(1));
  EXPECT_DOUBLE_EQ(0.3, shell.coefficients(1));
}

TEST_F(BasisSetTest, ShellManagement) {
  // Prepare shells
  std::vector<Shell> shells;
  std::vector<double> s_exp = {1.0};
  std::vector<double> s_coeff = {2.0};
  std::vector<double> p_exp = {0.5};
  std::vector<double> p_coeff = {1.0};
  shells.emplace_back(Shell(0, OrbitalType::S, s_exp, s_coeff));
  shells.emplace_back(Shell(0, OrbitalType::P, p_exp, p_coeff));

  // Create basis set with shells
  BasisSet basis("test", shells);
  EXPECT_EQ(2u, basis.get_num_shells());
  EXPECT_EQ(4u, basis.get_num_atomic_orbitals());  // 1 s + 3 p

  // Get shells for atom 0
  const auto& atom_shells = basis.get_shells_for_atom(0);
  EXPECT_EQ(2u, atom_shells.size());
  EXPECT_EQ(OrbitalType::S, atom_shells[0].orbital_type);
  EXPECT_EQ(OrbitalType::P, atom_shells[1].orbital_type);

  // Get specific shell
  const Shell& shell_0 = basis.get_shell(0);
  EXPECT_EQ(OrbitalType::S, shell_0.orbital_type);
  EXPECT_EQ(0u, shell_0.atom_index);
}

TEST_F(BasisSetTest, AOTypeManagement) {
  // Create a structure for testing
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure structure(coords, symbols);

  // Test default basis type (spherical) - empty basis sets are invalid
  EXPECT_THROW(BasisSet basis_spherical("test", structure),
               std::invalid_argument);

  // Create cartesian basis set
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::D, std::vector{1.0}, std::vector{2.0}));

  BasisSet basis_cartesian("test", shells, structure, AOType::Cartesian);
  EXPECT_EQ(AOType::Cartesian, basis_cartesian.get_atomic_orbital_type());

  // For cartesian d orbitals: 6 functions (dx2, dy2, dz2, dxy, dxz, dyz)
  EXPECT_EQ(6u, basis_cartesian.get_num_atomic_orbitals());

  // Create spherical basis set with same shell
  BasisSet basis_sph_test("test", shells, structure, AOType::Spherical);
  // For spherical d orbitals: 5 functions (d-2, d-1, d0, d1, d2)
  EXPECT_EQ(5u, basis_sph_test.get_num_atomic_orbitals());
}

TEST_F(BasisSetTest, ShellWithRawPrimitives) {
  // Create a shell with multiple primitives using raw data
  std::vector<double> exponents_vec = {3.425251, 0.623914, 0.168855};
  std::vector<double> coefficients_vec = {0.154329, 0.535328, 0.444635};

  // Convert to Eigen::VectorXd
  Eigen::VectorXd exponents =
      Eigen::Map<Eigen::VectorXd>(exponents_vec.data(), exponents_vec.size());
  Eigen::VectorXd coefficients = Eigen::Map<Eigen::VectorXd>(
      coefficients_vec.data(), coefficients_vec.size());

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, exponents, coefficients);

  BasisSet basis("test", shells);

  EXPECT_EQ(1u, basis.get_num_shells());
  const Shell& shell = basis.get_shell(0);
  EXPECT_EQ(3u, shell.get_num_primitives());
  EXPECT_EQ(3u, shell.exponents.size());
  EXPECT_EQ(3u, shell.coefficients.size());
  EXPECT_NEAR(3.425251, shell.exponents(0), testing::plain_text_tolerance);
  EXPECT_NEAR(0.154329, shell.coefficients(0), testing::plain_text_tolerance);
}

TEST_F(BasisSetTest, OrbitalTypeUtilities) {
  // Test orbital type sizes - spherical
  EXPECT_EQ(1u, BasisSet::get_num_orbitals_for_l(0, AOType::Spherical));
  EXPECT_EQ(3u, BasisSet::get_num_orbitals_for_l(1, AOType::Spherical));
  EXPECT_EQ(5u, BasisSet::get_num_orbitals_for_l(2, AOType::Spherical));
  EXPECT_EQ(7u, BasisSet::get_num_orbitals_for_l(3, AOType::Spherical));

  // Test orbital type sizes - cartesian
  EXPECT_EQ(1u, BasisSet::get_num_orbitals_for_l(0, AOType::Cartesian));
  EXPECT_EQ(3u, BasisSet::get_num_orbitals_for_l(1, AOType::Cartesian));
  EXPECT_EQ(6u, BasisSet::get_num_orbitals_for_l(2, AOType::Cartesian));
  EXPECT_EQ(10u, BasisSet::get_num_orbitals_for_l(3, AOType::Cartesian));

  // Test string conversion
  EXPECT_EQ("s", BasisSet::orbital_type_to_string(OrbitalType::S));
  EXPECT_EQ("p", BasisSet::orbital_type_to_string(OrbitalType::P));
  EXPECT_EQ("d", BasisSet::orbital_type_to_string(OrbitalType::D));
  EXPECT_EQ("f", BasisSet::orbital_type_to_string(OrbitalType::F));
  EXPECT_EQ("g", BasisSet::orbital_type_to_string(OrbitalType::G));
  EXPECT_EQ("h", BasisSet::orbital_type_to_string(OrbitalType::H));
  EXPECT_EQ("i", BasisSet::orbital_type_to_string(OrbitalType::I));

  // Test default case - this requires casting an invalid enum value
  EXPECT_EQ("unknown",
            BasisSet::orbital_type_to_string(static_cast<OrbitalType>(999)));

  // Test from string conversion
  EXPECT_EQ(OrbitalType::S, BasisSet::string_to_orbital_type("s"));
  EXPECT_EQ(OrbitalType::P, BasisSet::string_to_orbital_type("p"));
  EXPECT_EQ(OrbitalType::D, BasisSet::string_to_orbital_type("d"));
  EXPECT_EQ(OrbitalType::F, BasisSet::string_to_orbital_type("f"));

  // Test angular momentum conversion
  EXPECT_EQ(0, BasisSet::get_angular_momentum(OrbitalType::S));
  EXPECT_EQ(1, BasisSet::get_angular_momentum(OrbitalType::P));
  EXPECT_EQ(2, BasisSet::get_angular_momentum(OrbitalType::D));
  EXPECT_EQ(3, BasisSet::get_angular_momentum(OrbitalType::F));
  EXPECT_EQ(4, BasisSet::get_angular_momentum(OrbitalType::G));
  EXPECT_EQ(5, BasisSet::get_angular_momentum(OrbitalType::H));
  EXPECT_EQ(6, BasisSet::get_angular_momentum(OrbitalType::I));

  // Test basis type string conversion
  EXPECT_EQ("spherical",
            BasisSet::atomic_orbital_type_to_string(AOType::Spherical));
  EXPECT_EQ("cartesian",
            BasisSet::atomic_orbital_type_to_string(AOType::Cartesian));

  // Test default case - this requires casting an invalid enum value
  EXPECT_EQ("unknown",
            BasisSet::atomic_orbital_type_to_string(static_cast<AOType>(999)));

  EXPECT_EQ(AOType::Spherical,
            BasisSet::string_to_atomic_orbital_type("spherical"));
  EXPECT_EQ(AOType::Cartesian,
            BasisSet::string_to_atomic_orbital_type("cartesian"));
}

TEST_F(BasisSetTest, BasisFunctionQueries) {
  // Add shells for different atoms
  std::vector<Shell> shells;
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector{1.0},
                            std::vector{2.0}));  // atom 0: s
  shells.emplace_back(Shell(0, OrbitalType::P, std::vector{0.5},
                            std::vector{1.0}));  // atom 0: p
  shells.emplace_back(Shell(1, OrbitalType::S, std::vector{1.5},
                            std::vector{2.5}));  // atom 1: s

  BasisSet basis("test", shells);

  EXPECT_EQ(3u, basis.get_num_shells());
  EXPECT_EQ(5u, basis.get_num_atomic_orbitals());  // 1+3+1

  // Test atomic orbital info (now returns pair instead of BasisFunctionInfo)
  auto info_0 = basis.get_atomic_orbital_info(0);
  EXPECT_EQ(0u, info_0.first);  // shell index

  auto info_1 = basis.get_atomic_orbital_info(1);
  EXPECT_EQ(1u, info_1.first);  // shell index

  // Test basis indices for atoms
  auto indices_0 = basis.get_atomic_orbital_indices_for_atom(0);
  EXPECT_EQ(4u, indices_0.size());  // s + p = 1 + 3 = 4
  EXPECT_EQ(0u, indices_0[0]);
  EXPECT_EQ(3u, indices_0[3]);

  auto indices_1 = basis.get_atomic_orbital_indices_for_atom(1);
  EXPECT_EQ(1u, indices_1.size());  // s = 1
  EXPECT_EQ(4u, indices_1[0]);
}

TEST_F(BasisSetTest, StructureIntegration) {
  // Create a simple structure (H2)
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};

  std::vector<std::string> symbols = {"H", "H"};

  Structure structure(coords, symbols);

  // Add atomic orbitals for each hydrogen
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{3.425251}, std::vector{0.154329}));
  shells.emplace_back(
      Shell(1, OrbitalType::S, std::vector{3.425251}, std::vector{0.154329}));

  BasisSet basis("STO-3G", shells, structure);

  EXPECT_EQ(2u, basis.get_num_shells());
  EXPECT_EQ(2u, basis.get_num_atomic_orbitals());
}

TEST_F(BasisSetTest, Validation) {
  // Create a structure for testing
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure structure(coords, symbols);

  // Empty basis is invalid
  EXPECT_THROW(BasisSet empty_basis("test", structure), std::invalid_argument);

  // Add a shell
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));
  BasisSet basis("test", shells, structure);

  // Test validation with empty primitive data (create shell with empty vectors)
  Eigen::VectorXd empty_exponents(0);
  Eigen::VectorXd empty_coefficients(0);
  Shell invalid_shell(0, OrbitalType::S, empty_exponents, empty_coefficients);
  std::vector<Shell> invalid_shells;
  invalid_shells.emplace_back(invalid_shell);
  EXPECT_THROW(BasisSet invalid_basis("invalid", invalid_shells, structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, Summary) {
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::P, std::vector{0.5}, std::vector{1.0}));

  BasisSet basis("6-31G", shells);

  std::string summary = basis.get_summary();
  EXPECT_FALSE(summary.empty());
  EXPECT_NE(std::string::npos, summary.find("6-31G"));
  EXPECT_NE(std::string::npos, summary.find("2"));  // 2 shells
  EXPECT_NE(std::string::npos, summary.find("4"));  // 4 atomic orbitals
}

TEST_F(BasisSetTest, JSONSerialization) {
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));
  shells.emplace_back(
      Shell(0, OrbitalType::P, std::vector{0.5}, std::vector{1.0}));

  BasisSet basis("test-basis", shells);

  // Test JSON conversion
  auto json_data = basis.to_json();
  EXPECT_FALSE(json_data.empty());

  // Test round-trip
  auto basis2 = BasisSet::from_json(json_data);
  EXPECT_EQ(basis.get_name(), basis2->get_name());
  EXPECT_EQ(basis.get_num_shells(), basis2->get_num_shells());
  EXPECT_EQ(basis.get_num_atomic_orbitals(), basis2->get_num_atomic_orbitals());

  // Test file I/O
  basis.to_json_file("test.basis_set.json");
  auto basis3 = BasisSet::from_json_file("test.basis_set.json");
  EXPECT_EQ(basis.get_name(), basis3->get_name());
  EXPECT_EQ(basis.get_num_shells(), basis3->get_num_shells());

  // Cleanup
  std::remove("test.basis_set.json");
}

TEST_F(BasisSetTest, FileIO) {
  // Create shells with Eigen vectors
  std::vector<Shell> shells;

  Eigen::VectorXd s_exp(2), s_coeff(2);
  s_exp << 1.0, 2.0;
  s_coeff << 0.5, 0.5;
  shells.emplace_back(0, OrbitalType::S, s_exp, s_coeff);

  Eigen::VectorXd p_exp(1), p_coeff(1);
  p_exp << 3.0;
  p_coeff << 1.0;
  shells.emplace_back(0, OrbitalType::P, p_exp, p_coeff);

  // Create a basis set to test with
  BasisSet original_basis("test_basis", shells);

  // Create a temporary directory for test files
  std::string tmp_dir = "test_io_dir";
  std::error_code ec;
  std::filesystem::create_directories(tmp_dir, ec);

  // 1. Test generic to_file/from_file with JSON format
  std::string json_filename = tmp_dir + "/test_generic.basis_set.json";
  original_basis.to_file(json_filename, "json");

  auto loaded_json = BasisSet::from_file(json_filename, "json");
  EXPECT_EQ(original_basis.get_name(), loaded_json->get_name());
  EXPECT_EQ(original_basis.get_num_shells(), loaded_json->get_num_shells());

  // 2. Test generic to_file/from_file with HDF5 format
  std::string hdf5_filename = tmp_dir + "/test_generic.basis_set.h5";
  original_basis.to_file(hdf5_filename, "hdf5");

  auto loaded_hdf5 = BasisSet::from_file(hdf5_filename, "hdf5");
  EXPECT_EQ(original_basis.get_name(), loaded_hdf5->get_name());
  EXPECT_EQ(original_basis.get_num_shells(), loaded_hdf5->get_num_shells());

  // 3. Test specific HDF5 methods with filename validation
  std::string hdf5_direct = tmp_dir + "/test_direct.basis_set.h5";
  original_basis.to_hdf5_file(hdf5_direct);

  auto loaded_hdf5_direct = BasisSet::from_hdf5_file(hdf5_direct);
  EXPECT_EQ(original_basis.get_name(), loaded_hdf5_direct->get_name());

  // 4. Test error handling for unsupported file types
  EXPECT_THROW(original_basis.to_file("test.xyz", "unsupported"),
               std::runtime_error);

  EXPECT_THROW(BasisSet::from_file("test.xyz", "unsupported"),
               std::runtime_error);

  // 5. Test that error messages contain expected content for to_file
  try {
    original_basis.to_file("test.xyz", "xml");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    std::string error_msg(e.what());
    EXPECT_STRING_CONTAINS(error_msg, "Unsupported file type");
    EXPECT_STRING_CONTAINS(error_msg, "xml");
    EXPECT_STRING_CONTAINS(error_msg, "json, hdf5");
  }

  // 6. Test that error messages contain expected content for from_file
  try {
    auto test_basis = BasisSet::from_file("test.xyz", "binary");
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    std::string error_msg(e.what());
    EXPECT_STRING_CONTAINS(error_msg, "Unsupported file type");
    EXPECT_STRING_CONTAINS(error_msg, "binary");
    EXPECT_STRING_CONTAINS(error_msg, "json, hdf5");
  }

  // Clean up the temporary directory
  std::error_code ec2;
  std::filesystem::remove_all(tmp_dir, ec2);
}

// ============================================================================
// ECP (Effective Core Potential) Tests
// ============================================================================

TEST_F(BasisSetTest, ECPDefaultInitialization) {
  // Test that ECP is initialized with default values
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"O", "H"};
  Structure structure(coords, symbols);

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  BasisSet basis("test-basis", shells, structure);

  // Check default ECP values
  EXPECT_FALSE(basis.has_ecp_electrons());
  EXPECT_EQ("none", basis.get_ecp_name());
  EXPECT_EQ(2u, basis.get_ecp_electrons().size());
  EXPECT_EQ(0u, basis.get_ecp_electrons()[0]);
  EXPECT_EQ(0u, basis.get_ecp_electrons()[1]);

  // Check default ECP shell values
  EXPECT_FALSE(basis.has_ecp_shells());
  EXPECT_EQ(0u, basis.get_num_ecp_shells());
  EXPECT_TRUE(basis.get_ecp_shells().empty());
}

TEST_F(BasisSetTest, ECPGet) {
  // Test getting ECP data
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"Ag", "H"};
  Structure structure(coords, symbols);

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  // Create ECP shells with radial powers
  std::vector<Shell> ecp_shells;

  // ECP shell for Ag: s-type with r^0 and r^2 terms
  Eigen::VectorXd exp_s(2), coeff_s(2);
  Eigen::VectorXi rpow_s(2);
  exp_s << 10.0, 5.0;
  coeff_s << 50.0, 20.0;
  rpow_s << 0, 2;
  ecp_shells.emplace_back(0, OrbitalType::S, exp_s, coeff_s, rpow_s);

  // ECP shell for Ag: p-type with r^1 term
  Eigen::VectorXd exp_p(1), coeff_p(1);
  Eigen::VectorXi rpow_p(1);
  exp_p << 8.0;
  coeff_p << 30.0;
  rpow_p << 1;
  ecp_shells.emplace_back(0, OrbitalType::P, exp_p, coeff_p, rpow_p);

  // Create ECP data
  std::string ecp_name = "def2-tzvp";
  std::vector<size_t> ecp_electrons = {28, 0};

  // Create basis with ECP data using constructor
  BasisSet basis("test-basis", shells, ecp_name, ecp_shells, ecp_electrons,
                 structure);

  // Get ECP data
  EXPECT_TRUE(basis.has_ecp_electrons());
  EXPECT_EQ("def2-tzvp", basis.get_ecp_name());
  EXPECT_EQ(2u, basis.get_ecp_electrons().size());
  EXPECT_EQ(28u, basis.get_ecp_electrons()[0]);
  EXPECT_EQ(0u, basis.get_ecp_electrons()[1]);
}

TEST_F(BasisSetTest, ECPShellConstruction) {
  // Test creating ECP shells with radial powers
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"Ag", "H"};
  Structure structure(coords, symbols);

  // Create regular shells
  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  // Create ECP shells with radial powers
  std::vector<Shell> ecp_shells;

  // ECP shell for Ag: s-type with r^0 and r^2 terms
  Eigen::VectorXd exp_s(2), coeff_s(2);
  Eigen::VectorXi rpow_s(2);
  exp_s << 10.0, 5.0;
  coeff_s << 50.0, 20.0;
  rpow_s << 0, 2;
  ecp_shells.emplace_back(0, OrbitalType::S, exp_s, coeff_s, rpow_s);

  // ECP shell for Ag: p-type with r^1 term
  Eigen::VectorXd exp_p(1), coeff_p(1);
  Eigen::VectorXi rpow_p(1);
  exp_p << 8.0;
  coeff_p << 30.0;
  rpow_p << 1;
  ecp_shells.emplace_back(0, OrbitalType::P, exp_p, coeff_p, rpow_p);

  BasisSet basis("test-basis", shells, ecp_shells, structure);

  // Check ECP shell data
  EXPECT_TRUE(basis.has_ecp_shells());
  EXPECT_EQ(2u, basis.get_num_ecp_shells());

  // Check first ECP shell (s-type)
  const Shell& ecp_shell_s = basis.get_ecp_shell(0);
  EXPECT_EQ(0u, ecp_shell_s.atom_index);
  EXPECT_EQ(OrbitalType::S, ecp_shell_s.orbital_type);
  EXPECT_EQ(2u, ecp_shell_s.get_num_primitives());
  EXPECT_TRUE(ecp_shell_s.has_radial_powers());
  EXPECT_NEAR(10.0, ecp_shell_s.exponents(0), testing::plain_text_tolerance);
  EXPECT_NEAR(50.0, ecp_shell_s.coefficients(0), testing::plain_text_tolerance);
  EXPECT_EQ(0, ecp_shell_s.rpowers(0));
  EXPECT_EQ(2, ecp_shell_s.rpowers(1));

  // Check second ECP shell (p-type)
  const Shell& ecp_shell_p = basis.get_ecp_shell(1);
  EXPECT_EQ(0u, ecp_shell_p.atom_index);
  EXPECT_EQ(OrbitalType::P, ecp_shell_p.orbital_type);
  EXPECT_EQ(1u, ecp_shell_p.get_num_primitives());
  EXPECT_TRUE(ecp_shell_p.has_radial_powers());
  EXPECT_NEAR(8.0, ecp_shell_p.exponents(0), testing::plain_text_tolerance);
  EXPECT_NEAR(30.0, ecp_shell_p.coefficients(0), testing::plain_text_tolerance);
  EXPECT_EQ(1, ecp_shell_p.rpowers(0));

  // Check ECP shells for specific atom
  auto ecp_shells_atom0 = basis.get_ecp_shells_for_atom(0);
  EXPECT_EQ(2u, ecp_shells_atom0.size());

  auto ecp_shells_atom1 = basis.get_ecp_shells_for_atom(1);
  EXPECT_EQ(0u, ecp_shells_atom1.size());
}

TEST_F(BasisSetTest, ECPValidation) {
  // Test ECP validation and edge cases
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"Ag", "H"};
  Structure structure(coords, symbols);

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  // Create ECP shells with radial powers
  std::vector<Shell> ecp_shells;

  // ECP shell for Ag: s-type with r^0 and r^2 terms
  Eigen::VectorXd exp_s(2), coeff_s(2);
  Eigen::VectorXi rpow_s(2);
  exp_s << 10.0, 5.0;
  coeff_s << 50.0, 20.0;
  rpow_s << 0, 2;
  ecp_shells.emplace_back(0, OrbitalType::S, exp_s, coeff_s, rpow_s);

  // ECP shell for Ag: p-type with r^1 term
  Eigen::VectorXd exp_p(1), coeff_p(1);
  Eigen::VectorXi rpow_p(1);
  exp_p << 8.0;
  coeff_p << 30.0;
  rpow_p << 1;
  ecp_shells.emplace_back(0, OrbitalType::P, exp_p, coeff_p, rpow_p);

  // Test size validation - wrong size should throw
  std::vector<size_t> wrong_size_ecp = {28};  // Only 1 atom, but we have 2
  EXPECT_THROW(BasisSet("test-basis", shells, "test-ecp", ecp_shells,
                        wrong_size_ecp, structure),
               std::invalid_argument);

  // Correct size should work
  std::vector<size_t> correct_size_ecp = {28, 0};
  EXPECT_NO_THROW(BasisSet("test-basis", shells, "test-ecp", ecp_shells,
                           correct_size_ecp, structure));

  // Verify ECP electrons were set correctly
  BasisSet basis("test-basis", shells, "test-ecp", ecp_shells, correct_size_ecp,
                 structure);
  EXPECT_TRUE(basis.has_ecp_electrons());
  EXPECT_EQ("test-ecp", basis.get_ecp_name());
}

TEST_F(BasisSetTest, ECPCopyConstructorAndAssignment) {
  // Test that ECP data and ECP shells are copied correctly via both copy
  // constructor and assignment
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"Ag", "H"};
  Structure structure(coords, symbols);

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  // Create ECP shells
  std::vector<Shell> ecp_shells;
  Eigen::VectorXd exp(2), coeff(2);
  Eigen::VectorXi rpow(2);
  exp << 10.0, 5.0;
  coeff << 50.0, 20.0;
  rpow << 0, 2;
  ecp_shells.emplace_back(0, OrbitalType::S, exp, coeff, rpow);

  BasisSet basis1("test-basis", shells, "def2-tzvp", ecp_shells, {28, 0},
                  structure);

  // Test copy constructor
  BasisSet basis2(basis1);
  EXPECT_TRUE(basis2.has_ecp_electrons());
  EXPECT_EQ("def2-tzvp", basis2.get_ecp_name());
  EXPECT_EQ(28u, basis2.get_ecp_electrons()[0]);
  EXPECT_EQ(0u, basis2.get_ecp_electrons()[1]);
  EXPECT_TRUE(basis2.has_ecp_shells());
  EXPECT_EQ(1u, basis2.get_num_ecp_shells());
  EXPECT_EQ(2u, basis2.get_ecp_shell(0).get_num_primitives());

  // Test copy assignment
  BasisSet basis3("test-basis-3", shells, structure);
  basis3 = basis1;
  EXPECT_TRUE(basis3.has_ecp_electrons());
  EXPECT_EQ("def2-tzvp", basis3.get_ecp_name());
  EXPECT_EQ(28u, basis3.get_ecp_electrons()[0]);
  EXPECT_EQ(0u, basis3.get_ecp_electrons()[1]);
  EXPECT_TRUE(basis3.has_ecp_shells());
  EXPECT_EQ(1u, basis3.get_num_ecp_shells());
  EXPECT_EQ(2u, basis3.get_ecp_shell(0).get_num_primitives());
}

TEST_F(BasisSetTest, ECPJSONSerialization) {
  // Test comprehensive JSON serialization with and without ECP and ECP shells
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"Ag", "H"};
  Structure structure(coords, symbols);

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  // Create ECP shells with radial powers
  std::vector<Shell> ecp_shells;
  Eigen::VectorXd exp(2), coeff(2);
  Eigen::VectorXi rpow(2);
  exp << 10.0, 5.0;
  coeff << 50.0, 20.0;
  rpow << 0, 2;
  ecp_shells.emplace_back(0, OrbitalType::S, exp, coeff, rpow);

  // Test with ECP and ECP shells
  BasisSet basis_with_ecp("test-basis", shells, "def2-tzvp", ecp_shells,
                          {28, 0}, structure);

  // In-memory JSON round-trip
  auto json = basis_with_ecp.to_json();
  EXPECT_TRUE(json.contains("ecp_name"));
  EXPECT_TRUE(json.contains("ecp_electrons"));
  EXPECT_EQ("def2-tzvp", json["ecp_name"]);

  // Check that ECP shells are serialized in the atoms array
  EXPECT_TRUE(json.contains("atoms"));
  EXPECT_FALSE(json["atoms"].empty());
  bool found_ecp_shells = false;
  for (const auto& atom : json["atoms"]) {
    if (atom.contains("ecp_shells") && !atom["ecp_shells"].empty()) {
      found_ecp_shells = true;
      // Verify rpowers are present in ECP shell
      EXPECT_TRUE(atom["ecp_shells"][0].contains("rpowers"));
      break;
    }
  }
  EXPECT_TRUE(found_ecp_shells);

  auto loaded_basis = BasisSet::from_json(json);
  EXPECT_TRUE(loaded_basis->has_ecp_electrons());
  EXPECT_EQ("def2-tzvp", loaded_basis->get_ecp_name());
  EXPECT_EQ(28u, loaded_basis->get_ecp_electrons()[0]);
  EXPECT_EQ(0u, loaded_basis->get_ecp_electrons()[1]);

  // Verify ECP shells were preserved
  EXPECT_TRUE(loaded_basis->has_ecp_shells());
  EXPECT_EQ(1u, loaded_basis->get_num_ecp_shells());
  const Shell& loaded_ecp_shell = loaded_basis->get_ecp_shell(0);
  EXPECT_EQ(2u, loaded_ecp_shell.get_num_primitives());
  EXPECT_TRUE(loaded_ecp_shell.has_radial_powers());
  EXPECT_NEAR(10.0, loaded_ecp_shell.exponents(0),
              testing::plain_text_tolerance);
  EXPECT_NEAR(50.0, loaded_ecp_shell.coefficients(0),
              testing::plain_text_tolerance);
  EXPECT_EQ(0, loaded_ecp_shell.rpowers(0));
  EXPECT_EQ(2, loaded_ecp_shell.rpowers(1));

  // File-based JSON round-trip
  std::string filename = "test_ecp.basis_set.json";
  basis_with_ecp.to_json_file(filename);
  auto loaded_from_file = BasisSet::from_json_file(filename);
  EXPECT_TRUE(loaded_from_file->has_ecp_electrons());
  EXPECT_EQ("def2-tzvp", loaded_from_file->get_ecp_name());
  EXPECT_TRUE(loaded_from_file->has_ecp_shells());
  EXPECT_EQ(1u, loaded_from_file->get_num_ecp_shells());
  std::filesystem::remove(filename);

  // Test without ECP (verify default handling)
  BasisSet basis_without_ecp("test-basis", shells, structure);
  auto json_no_ecp = basis_without_ecp.to_json();
  auto loaded_no_ecp = BasisSet::from_json(json_no_ecp);
  EXPECT_FALSE(loaded_no_ecp->has_ecp_electrons());
  EXPECT_EQ("none", loaded_no_ecp->get_ecp_name());
  EXPECT_FALSE(loaded_no_ecp->has_ecp_shells());
  EXPECT_EQ(0u, loaded_no_ecp->get_num_ecp_shells());
}

TEST_F(BasisSetTest, ECPShellQueries) {
  // Test various query methods for ECP shells
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"Ag", "Au", "H"};
  Structure structure(coords, symbols);

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(2, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  // Create ECP shells for different atoms and angular momenta
  std::vector<Shell> ecp_shells;

  // Ag (atom 0): s-type and p-type ECP shells
  Eigen::VectorXd exp_s(1), coeff_s(1);
  Eigen::VectorXi rpow_s(1);
  exp_s << 10.0;
  coeff_s << 50.0;
  rpow_s << 0;
  ecp_shells.emplace_back(0, OrbitalType::S, exp_s, coeff_s, rpow_s);

  Eigen::VectorXd exp_p(1), coeff_p(1);
  Eigen::VectorXi rpow_p(1);
  exp_p << 8.0;
  coeff_p << 30.0;
  rpow_p << 1;
  ecp_shells.emplace_back(0, OrbitalType::P, exp_p, coeff_p, rpow_p);

  // Au (atom 1): d-type ECP shell
  Eigen::VectorXd exp_d(1), coeff_d(1);
  Eigen::VectorXi rpow_d(1);
  exp_d << 12.0;
  coeff_d << 40.0;
  rpow_d << 2;
  ecp_shells.emplace_back(1, OrbitalType::D, exp_d, coeff_d, rpow_d);

  BasisSet basis("test-basis", shells, ecp_shells, structure);

  // Test get_num_ecp_shells
  EXPECT_EQ(3u, basis.get_num_ecp_shells());

  // Test get_ecp_shells (all shells)
  auto all_ecp_shells = basis.get_ecp_shells();
  EXPECT_EQ(3u, all_ecp_shells.size());

  // Test get_ecp_shells_for_atom
  auto ecp_shells_atom0 = basis.get_ecp_shells_for_atom(0);
  EXPECT_EQ(2u, ecp_shells_atom0.size());
  EXPECT_EQ(OrbitalType::S, ecp_shells_atom0[0].orbital_type);
  EXPECT_EQ(OrbitalType::P, ecp_shells_atom0[1].orbital_type);

  auto ecp_shells_atom1 = basis.get_ecp_shells_for_atom(1);
  EXPECT_EQ(1u, ecp_shells_atom1.size());
  EXPECT_EQ(OrbitalType::D, ecp_shells_atom1[0].orbital_type);

  auto ecp_shells_atom2 = basis.get_ecp_shells_for_atom(2);
  EXPECT_EQ(0u, ecp_shells_atom2.size());

  // Test get_ecp_shell by index
  const Shell& shell0 = basis.get_ecp_shell(0);
  EXPECT_EQ(0u, shell0.atom_index);
  EXPECT_EQ(OrbitalType::S, shell0.orbital_type);

  const Shell& shell1 = basis.get_ecp_shell(1);
  EXPECT_EQ(0u, shell1.atom_index);
  EXPECT_EQ(OrbitalType::P, shell1.orbital_type);

  const Shell& shell2 = basis.get_ecp_shell(2);
  EXPECT_EQ(1u, shell2.atom_index);
  EXPECT_EQ(OrbitalType::D, shell2.orbital_type);

  // Test out-of-range access
  EXPECT_THROW(basis.get_ecp_shell(3), std::out_of_range);
}

TEST_F(BasisSetTest, ECPShellWithVectorConstructor) {
  // Test Shell construction using vector constructor with radial powers
  std::vector<double> exponents = {10.0, 5.0, 2.0};
  std::vector<double> coefficients = {50.0, 20.0, 10.0};
  std::vector<int> rpowers = {0, 1, 2};

  Shell ecp_shell(0, OrbitalType::S, exponents, coefficients, rpowers);

  EXPECT_EQ(0u, ecp_shell.atom_index);
  EXPECT_EQ(OrbitalType::S, ecp_shell.orbital_type);
  EXPECT_EQ(3u, ecp_shell.get_num_primitives());
  EXPECT_TRUE(ecp_shell.has_radial_powers());
  EXPECT_EQ(0, ecp_shell.get_angular_momentum());

  // Verify data
  EXPECT_NEAR(10.0, ecp_shell.exponents(0), testing::plain_text_tolerance);
  EXPECT_NEAR(50.0, ecp_shell.coefficients(0), testing::plain_text_tolerance);
  EXPECT_EQ(0, ecp_shell.rpowers(0));
  EXPECT_EQ(1, ecp_shell.rpowers(1));
  EXPECT_EQ(2, ecp_shell.rpowers(2));
}

TEST_F(BasisSetTest, ECPShellValidation) {
  // Test validation for ECP shell construction
  std::vector<double> exponents = {10.0, 5.0};
  std::vector<double> coefficients = {50.0, 20.0};
  std::vector<int> rpowers_wrong_size = {0};  // Wrong size
  std::vector<int> rpowers_correct = {0, 2};  // Correct size

  // Mismatched sizes should throw
  EXPECT_THROW(
      Shell(0, OrbitalType::S, exponents, coefficients, rpowers_wrong_size),
      std::invalid_argument);

  // Correct sizes should work
  EXPECT_NO_THROW(
      Shell(0, OrbitalType::S, exponents, coefficients, rpowers_correct));

  // Regular shell without rpowers should work
  EXPECT_NO_THROW(Shell(0, OrbitalType::S, exponents, coefficients));

  // Regular shell should not have radial powers
  Shell regular_shell(0, OrbitalType::S, exponents, coefficients);
  EXPECT_FALSE(regular_shell.has_radial_powers());
}

TEST_F(BasisSetTest, ECPShellsWithoutECPMetadata) {
  // Test that we can have ECP shells without setting ECP metadata
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"Ag"};
  Structure structure(coords, symbols);

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  std::vector<Shell> ecp_shells;
  Eigen::VectorXd exp(1), coeff(1);
  Eigen::VectorXi rpow(1);
  exp << 10.0;
  coeff << 50.0;
  rpow << 0;
  ecp_shells.emplace_back(0, OrbitalType::S, exp, coeff, rpow);

  BasisSet basis("test-basis", shells, ecp_shells, structure);

  // Should have ECP shells but no ECP metadata
  EXPECT_TRUE(basis.has_ecp_shells());
  EXPECT_EQ(1u, basis.get_num_ecp_shells());
  EXPECT_FALSE(basis.has_ecp_electrons());
  EXPECT_EQ("none", basis.get_ecp_name());
  EXPECT_EQ(0u, basis.get_ecp_electrons()[0]);
}

TEST_F(BasisSetTest, ECPHDF5Serialization) {
  // Test HDF5 serialization with ECP data
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"Ag", "H"};
  Structure structure(coords, symbols);

  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  // Create ECP shells with radial powers
  std::vector<Shell> ecp_shells;
  Eigen::VectorXd exp(2), coeff(2);
  Eigen::VectorXi rpow(2);
  exp << 10.0, 5.0;
  coeff << 50.0, 20.0;
  rpow << 0, 2;
  ecp_shells.emplace_back(0, OrbitalType::S, exp, coeff, rpow);

  BasisSet basis("test-basis", shells, "def2-tzvp", ecp_shells, {28, 0},
                 structure);

  // HDF5 file round-trip
  std::string filename = "test_ecp.basis_set.h5";
  basis.to_hdf5_file(filename);

  auto loaded_basis = BasisSet::from_hdf5_file(filename);
  EXPECT_TRUE(loaded_basis->has_ecp_electrons());
  EXPECT_EQ("def2-tzvp", loaded_basis->get_ecp_name());
  EXPECT_EQ(28u, loaded_basis->get_ecp_electrons()[0]);
  EXPECT_EQ(0u, loaded_basis->get_ecp_electrons()[1]);

  std::filesystem::remove(filename);
}

TEST_F(BasisSetTest, IndexConversion) {
  std::vector<Shell> shells;
  shells.emplace_back(Shell(0, OrbitalType::S, std::vector{1.0},
                            std::vector{2.0}));  // shell 0: 1 function
  shells.emplace_back(Shell(0, OrbitalType::P, std::vector{0.5},
                            std::vector{1.0}));  // shell 1: 3 functions
  shells.emplace_back(Shell(1, OrbitalType::S, std::vector{1.5},
                            std::vector{2.5}));  // shell 2: 1 function

  BasisSet basis("test", shells);

  // Test atomic orbital to shell conversion using new return type (pair)
  auto info_0 = basis.basis_to_shell_index(0);
  EXPECT_EQ(0u, info_0.first);  // shell index

  auto info_1 = basis.basis_to_shell_index(1);
  EXPECT_EQ(1u, info_1.first);  // shell index

  auto info_2 = basis.basis_to_shell_index(2);
  EXPECT_EQ(1u, info_2.first);  // shell index

  auto info_3 = basis.basis_to_shell_index(3);
  EXPECT_EQ(1u, info_3.first);  // shell index

  auto info_4 = basis.basis_to_shell_index(4);
  EXPECT_EQ(2u, info_4.first);  // shell index
}

TEST_F(BasisSetTest, ShellQueries) {
  // Create a multi-atom basis set with different shell types
  std::vector<Shell> shells;

  // Atom 0: S, P shells
  shells.push_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));  // shell 0
  shells.push_back(
      Shell(0, OrbitalType::P, std::vector{0.5}, std::vector{1.0}));  // shell 1

  // Atom 1: S, D shells
  shells.push_back(
      Shell(1, OrbitalType::S, std::vector{1.5}, std::vector{2.5}));  // shell 2
  shells.push_back(
      Shell(1, OrbitalType::D, std::vector{0.8}, std::vector{1.2}));  // shell 3

  // Atom 2: P shell
  shells.push_back(
      Shell(2, OrbitalType::P, std::vector{2.0}, std::vector{3.0}));  // shell 4

  BasisSet basis("test", shells);

  // Test get_shell_indices_for_atom
  auto atom0_shells = basis.get_shell_indices_for_atom(0);
  EXPECT_EQ(2u, atom0_shells.size());
  EXPECT_EQ(0u, atom0_shells[0]);  // First shell
  EXPECT_EQ(1u, atom0_shells[1]);  // Second shell

  auto atom1_shells = basis.get_shell_indices_for_atom(1);
  EXPECT_EQ(2u, atom1_shells.size());
  EXPECT_EQ(2u, atom1_shells[0]);  // Third shell overall
  EXPECT_EQ(3u, atom1_shells[1]);  // Fourth shell overall

  auto atom2_shells = basis.get_shell_indices_for_atom(2);
  EXPECT_EQ(1u, atom2_shells.size());
  EXPECT_EQ(4u, atom2_shells[0]);  // Fifth shell overall

  // Test get_num_atomic_orbitals_for_atom
  EXPECT_EQ(4u, basis.get_num_atomic_orbitals_for_atom(0));  // S(1) + P(3) = 4
  EXPECT_EQ(6u, basis.get_num_atomic_orbitals_for_atom(1));  // S(1) + D(5) = 6
  EXPECT_EQ(3u, basis.get_num_atomic_orbitals_for_atom(2));  // P(3) = 3
}

TEST_F(BasisSetTest, OrbitalTypeQueries) {
  // Create basis set with different orbital types
  std::vector<Shell> shells;
  shells.push_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));  // shell 0
  shells.push_back(
      Shell(0, OrbitalType::P, std::vector{0.5}, std::vector{1.0}));  // shell 1
  shells.push_back(
      Shell(1, OrbitalType::S, std::vector{1.5}, std::vector{2.5}));  // shell 2
  shells.push_back(
      Shell(1, OrbitalType::D, std::vector{0.8}, std::vector{1.2}));  // shell 3
  shells.push_back(
      Shell(2, OrbitalType::P, std::vector{2.0}, std::vector{3.0}));  // shell 4
  shells.push_back(
      Shell(2, OrbitalType::D, std::vector{1.8}, std::vector{2.2}));  // shell 5

  BasisSet basis("test", shells);

  // Test get_shell_indices_for_orbital_type
  auto s_shells = basis.get_shell_indices_for_orbital_type(OrbitalType::S);
  EXPECT_EQ(2u, s_shells.size());
  EXPECT_EQ(0u, s_shells[0]);  // First S shell
  EXPECT_EQ(2u, s_shells[1]);  // Second S shell

  auto p_shells = basis.get_shell_indices_for_orbital_type(OrbitalType::P);
  EXPECT_EQ(2u, p_shells.size());
  EXPECT_EQ(1u, p_shells[0]);  // First P shell
  EXPECT_EQ(4u, p_shells[1]);  // Second P shell

  auto d_shells = basis.get_shell_indices_for_orbital_type(OrbitalType::D);
  EXPECT_EQ(2u, d_shells.size());
  EXPECT_EQ(3u, d_shells[0]);  // First D shell
  EXPECT_EQ(5u, d_shells[1]);  // Second D shell

  // Test orbital type that doesn't exist
  auto f_shells = basis.get_shell_indices_for_orbital_type(OrbitalType::F);
  EXPECT_EQ(0u, f_shells.size());

  // Test get_num_atomic_orbitals_for_orbital_type (spherical basis)
  EXPECT_EQ(2u, basis.get_num_atomic_orbitals_for_orbital_type(
                    OrbitalType::S));  // 2 S shells, 1 function each
  EXPECT_EQ(6u, basis.get_num_atomic_orbitals_for_orbital_type(
                    OrbitalType::P));  // 2 P shells, 3 functions each
  EXPECT_EQ(10u, basis.get_num_atomic_orbitals_for_orbital_type(
                     OrbitalType::D));  // 2 D shells, 5 functions each
  EXPECT_EQ(0u, basis.get_num_atomic_orbitals_for_orbital_type(
                    OrbitalType::F));  // No F shells

  // Test with Cartesian basis type
  BasisSet cartesian_basis("test", shells, AOType::Cartesian);
  EXPECT_EQ(2u, cartesian_basis.get_num_atomic_orbitals_for_orbital_type(
                    OrbitalType::S));  // 2 S shells, 1 function each
  EXPECT_EQ(6u, cartesian_basis.get_num_atomic_orbitals_for_orbital_type(
                    OrbitalType::P));  // 2 P shells, 3 functions each
  EXPECT_EQ(12u,
            cartesian_basis.get_num_atomic_orbitals_for_orbital_type(
                OrbitalType::D));  // 2 D shells, 6 functions each (cartesian)
}

TEST_F(BasisSetTest, CombinedQueries) {
  // Create basis set with multiple atoms and orbital types
  std::vector<Shell> shells;

  // Atom 0: S, P, D
  shells.push_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));  // shell 0
  shells.push_back(
      Shell(0, OrbitalType::P, std::vector{0.5}, std::vector{1.0}));  // shell 1
  shells.push_back(
      Shell(0, OrbitalType::D, std::vector{0.3}, std::vector{0.8}));  // shell 2

  // Atom 1: S, P
  shells.push_back(
      Shell(1, OrbitalType::S, std::vector{1.5}, std::vector{2.5}));  // shell 3
  shells.push_back(
      Shell(1, OrbitalType::P, std::vector{0.8}, std::vector{1.2}));  // shell 4

  // Atom 2: D, F
  shells.push_back(
      Shell(2, OrbitalType::D, std::vector{1.8}, std::vector{2.2}));  // shell 5
  shells.push_back(
      Shell(2, OrbitalType::F, std::vector{1.1}, std::vector{1.5}));  // shell 6

  BasisSet basis("test", shells);

  // Test get_shell_indices_for_atom_and_orbital_type

  // Atom 0 queries
  auto atom0_s =
      basis.get_shell_indices_for_atom_and_orbital_type(0, OrbitalType::S);
  EXPECT_EQ(1u, atom0_s.size());
  EXPECT_EQ(0u, atom0_s[0]);

  auto atom0_p =
      basis.get_shell_indices_for_atom_and_orbital_type(0, OrbitalType::P);
  EXPECT_EQ(1u, atom0_p.size());
  EXPECT_EQ(1u, atom0_p[0]);

  auto atom0_d =
      basis.get_shell_indices_for_atom_and_orbital_type(0, OrbitalType::D);
  EXPECT_EQ(1u, atom0_d.size());
  EXPECT_EQ(2u, atom0_d[0]);

  auto atom0_f =
      basis.get_shell_indices_for_atom_and_orbital_type(0, OrbitalType::F);
  EXPECT_EQ(0u, atom0_f.size());  // No F shells on atom 0

  // Atom 1 queries
  auto atom1_s =
      basis.get_shell_indices_for_atom_and_orbital_type(1, OrbitalType::S);
  EXPECT_EQ(1u, atom1_s.size());
  EXPECT_EQ(3u, atom1_s[0]);

  auto atom1_p =
      basis.get_shell_indices_for_atom_and_orbital_type(1, OrbitalType::P);
  EXPECT_EQ(1u, atom1_p.size());
  EXPECT_EQ(4u, atom1_p[0]);

  auto atom1_d =
      basis.get_shell_indices_for_atom_and_orbital_type(1, OrbitalType::D);
  EXPECT_EQ(0u, atom1_d.size());  // No D shells on atom 1

  // Atom 2 queries
  auto atom2_s =
      basis.get_shell_indices_for_atom_and_orbital_type(2, OrbitalType::S);
  EXPECT_EQ(0u, atom2_s.size());  // No S shells on atom 2

  auto atom2_d =
      basis.get_shell_indices_for_atom_and_orbital_type(2, OrbitalType::D);
  EXPECT_EQ(1u, atom2_d.size());
  EXPECT_EQ(5u, atom2_d[0]);

  auto atom2_f =
      basis.get_shell_indices_for_atom_and_orbital_type(2, OrbitalType::F);
  EXPECT_EQ(1u, atom2_f.size());
  EXPECT_EQ(6u, atom2_f[0]);
}

TEST_F(BasisSetTest, ErrorHandling) {
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));

  BasisSet basis("test", shells);

  // Test out-of-bounds access
  EXPECT_THROW(basis.get_shell(10), std::out_of_range);
  EXPECT_THROW(basis.basis_to_shell_index(10), std::out_of_range);

  // Test invalid orbital type string
  EXPECT_THROW(BasisSet::string_to_orbital_type("X"), std::invalid_argument);

  // Test invalid basis type string
  EXPECT_THROW(BasisSet::string_to_atomic_orbital_type("unknown"),
               std::invalid_argument);

  // Test mismatched primitive sizes in Shell constructor
  std::vector<double> exp_vec = {1.0, 2.0};
  std::vector<double> coeff_vec = {1.0};  // Different size
  Eigen::VectorXd exp =
      Eigen::Map<Eigen::VectorXd>(exp_vec.data(), exp_vec.size());
  Eigen::VectorXd coeff =
      Eigen::Map<Eigen::VectorXd>(coeff_vec.data(), coeff_vec.size());
  EXPECT_THROW(Shell(0, OrbitalType::S, exp, coeff), std::invalid_argument);

  // Test invalid atom index for new query functions
  EXPECT_THROW(basis.get_shell_indices_for_atom(10), std::out_of_range);
  EXPECT_THROW(basis.get_num_atomic_orbitals_for_atom(10), std::out_of_range);
  EXPECT_THROW(
      basis.get_shell_indices_for_atom_and_orbital_type(10, OrbitalType::S),
      std::out_of_range);

  // Test _validate_atomic_orbital_index (via get_atomic_orbital_info)
  EXPECT_THROW(basis.get_atomic_orbital_info(100), std::out_of_range);

  // Test _validate_shell_index error message formatting
  try {
    basis.get_shell(50);
    FAIL() << "Expected std::out_of_range";
  } catch (const std::out_of_range& e) {
    std::string error_msg(e.what());
    EXPECT_STRING_CONTAINS(error_msg, "Shell index");
    EXPECT_STRING_CONTAINS(error_msg, "50");
    EXPECT_STRING_CONTAINS(error_msg, "out of range");
    EXPECT_STRING_CONTAINS(error_msg, "Maximum index");
  }

  // Test _validate_atom_index error message formatting
  try {
    basis.get_shell_indices_for_atom(25);
    FAIL() << "Expected std::out_of_range";
  } catch (const std::out_of_range& e) {
    std::string error_msg(e.what());
    EXPECT_STRING_CONTAINS(error_msg, "Atom index");
    EXPECT_STRING_CONTAINS(error_msg, "25");
    EXPECT_STRING_CONTAINS(error_msg, "out of range");
    EXPECT_STRING_CONTAINS(error_msg, "Maximum index");
  }

  // Test _validate_atomic_orbital_index error message formatting
  try {
    basis.get_atomic_orbital_info(75);
    FAIL() << "Expected std::out_of_range";
  } catch (const std::out_of_range& e) {
    std::string error_msg(e.what());
    EXPECT_STRING_CONTAINS(error_msg, "atomic orbital index");
    EXPECT_STRING_CONTAINS(error_msg, "75");
    EXPECT_STRING_CONTAINS(error_msg, "out of range");
    EXPECT_STRING_CONTAINS(error_msg, "Maximum index");
  }
}

TEST_F(BasisSetTest, HDF5Comprehensive) {
  // Create a temporary directory for test files
  std::string tmp_dir = "test_hdf5_comprehensive";
  std::error_code ec;
  std::filesystem::create_directories(tmp_dir, ec);

  // 1. Test comprehensive basis set with multiple atoms and shell types
  std::vector<Shell> complex_shells;

  // Add multiple shells per atom with various orbital types
  // Atom 0: S, S, P, D shells (like a real basis set)
  Eigen::VectorXd s1_exp(3), s1_coeff(3);
  s1_exp << 11720.0, 1759.0, 400.8;
  s1_coeff << 0.000710, 0.005470, 0.027837;
  complex_shells.emplace_back(0, OrbitalType::S, s1_exp, s1_coeff);

  Eigen::VectorXd s2_exp(2), s2_coeff(2);
  s2_exp << 55.44, 15.99;
  s2_coeff << -0.119325, 0.695736;
  complex_shells.emplace_back(0, OrbitalType::S, s2_exp, s2_coeff);

  Eigen::VectorXd p1_exp(3), p1_coeff(3);
  p1_exp << 17.7, 3.854, 1.046;
  p1_coeff << 0.143859, 0.624709, 0.440895;
  complex_shells.emplace_back(0, OrbitalType::P, p1_exp, p1_coeff);

  Eigen::VectorXd d1_exp(1), d1_coeff(1);
  d1_exp << 1.158;
  d1_coeff << 1.0;
  complex_shells.emplace_back(0, OrbitalType::D, d1_exp, d1_coeff);

  // Atom 1: Different shells
  complex_shells.emplace_back(1, OrbitalType::S, s1_exp, s1_coeff);
  complex_shells.emplace_back(1, OrbitalType::P, p1_exp, p1_coeff);

  // Atom 2: F orbital
  Eigen::VectorXd f1_exp(2), f1_coeff(2);
  f1_exp << 2.506, 0.808;
  f1_coeff << 0.123456, 0.654321;
  complex_shells.emplace_back(2, OrbitalType::F, f1_exp, f1_coeff);

  BasisSet complex_basis("cc-pVDZ", complex_shells, AOType::Spherical);

  std::string complex_filename = tmp_dir + "/complex.basis_set.h5";
  complex_basis.to_hdf5_file(complex_filename);

  auto loaded_complex = BasisSet::from_hdf5_file(complex_filename);

  // Verify comprehensive data integrity
  EXPECT_EQ(complex_basis.get_name(), loaded_complex->get_name());
  EXPECT_EQ(complex_basis.get_atomic_orbital_type(),
            loaded_complex->get_atomic_orbital_type());
  EXPECT_EQ(complex_basis.get_num_shells(), loaded_complex->get_num_shells());
  EXPECT_EQ(complex_basis.get_num_atomic_orbitals(),
            loaded_complex->get_num_atomic_orbitals());
  EXPECT_EQ(complex_basis.get_num_atoms(), loaded_complex->get_num_atoms());

  // Verify shell-by-shell data integrity
  for (size_t i = 0; i < complex_basis.get_num_shells(); ++i) {
    const Shell& orig_shell = complex_basis.get_shell(i);
    const Shell& loaded_shell = loaded_complex->get_shell(i);

    EXPECT_EQ(orig_shell.atom_index, loaded_shell.atom_index);
    EXPECT_EQ(orig_shell.orbital_type, loaded_shell.orbital_type);
    EXPECT_EQ(orig_shell.get_num_primitives(),
              loaded_shell.get_num_primitives());

    for (size_t j = 0; j < orig_shell.get_num_primitives(); ++j) {
      EXPECT_NEAR(orig_shell.exponents(j), loaded_shell.exponents(j),
                  testing::hdf5_tolerance);
      EXPECT_NEAR(orig_shell.coefficients(j), loaded_shell.coefficients(j),
                  testing::hdf5_tolerance);
    }
  }

  // 2. Test empty basis set - should be invalid
  std::vector<Shell> empty_shells;
  EXPECT_THROW(BasisSet empty_basis("empty", empty_shells),
               std::invalid_argument);

  // 3. Test basis set with single primitive per shell
  std::vector<Shell> single_shells;
  single_shells.push_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}));
  single_shells.push_back(
      Shell(0, OrbitalType::P, std::vector{2.0}, std::vector{0.5}));
  BasisSet single_prim("single", single_shells);

  std::string single_filename = tmp_dir + "/single.basis_set.h5";
  single_prim.to_hdf5_file(single_filename);

  auto loaded_single = BasisSet::from_hdf5_file(single_filename);

  EXPECT_EQ(single_prim.get_name(), loaded_single->get_name());
  EXPECT_EQ(2u, loaded_single->get_num_shells());
  EXPECT_EQ(4u, loaded_single->get_num_atomic_orbitals());

  // 4. Test HDF5 format structure verification - check metadata group exists
  std::vector<Shell> verify_shells;
  verify_shells.push_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0}));
  BasisSet verify_basis("verify", verify_shells);

  std::string verify_filename = tmp_dir + "/verify.basis_set.h5";
  verify_basis.to_hdf5_file(verify_filename);

  // Verify the file can be loaded
  auto loaded_verify = BasisSet::from_hdf5_file(verify_filename);
  EXPECT_EQ("verify", loaded_verify->get_name());

  // 5. Test cartesian vs spherical basis type preservation
  std::vector<Shell> cartesian_shells;
  cartesian_shells.push_back(
      Shell(0, OrbitalType::D, std::vector{1.0}, std::vector{1.0}));
  BasisSet cartesian_basis("cartesian_test", cartesian_shells,
                           AOType::Cartesian);

  std::string cartesian_filename = tmp_dir + "/cartesian.basis_set.h5";
  cartesian_basis.to_hdf5_file(cartesian_filename);

  auto loaded_cartesian = BasisSet::from_hdf5_file(cartesian_filename);

  // Verify that atomic_orbital_type is correctly preserved in HDF5
  EXPECT_EQ(AOType::Cartesian, loaded_cartesian->get_atomic_orbital_type());
  EXPECT_EQ(1u, loaded_cartesian->get_num_shells());

  // Clean up the temporary directory
  std::error_code ec2;
  std::filesystem::remove_all(tmp_dir, ec2);
}

TEST_F(BasisSetTest, SameBasisSetCheck) {
  // Compare energies from standard basis set string vs custom BasisSet object
  const std::string basis_set = "sto-3g";
  std::shared_ptr<Structure> structure = std::make_shared<Structure>(
      std::vector<Eigen::Vector3d>{
          {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
      std::vector<std::string>{"H", "H", "O"});

  // generate basis set from shells
  std::string manual_name = std::string(BasisSet::custom_name);
  std::vector<Shell> shells = {
      // H atom 0
      Shell(0, OrbitalType::S,
            // exponents
            std::vector<double>{0.3425250914E+01, 0.6239137298E+00,
                                0.1688554040E+00},
            // coefficients
            std::vector<double>{0.1543289673E+00, 0.5353281423E+00,
                                0.4446345422E+00}),
      // H atom 1
      Shell(1, OrbitalType::S,
            // exponents
            std::vector<double>{0.3425250914E+01, 0.6239137298E+00,
                                0.1688554040E+00},
            // coefficients
            std::vector<double>{0.1543289673E+00, 0.5353281423E+00,
                                0.4446345422E+00}),
      // O atom 2
      Shell(2, OrbitalType::S,
            // exponents
            std::vector<double>{0.1307093214E+03, 0.2380886605E+02,
                                0.6443608313E+01},
            // coefficients
            std::vector<double>{0.1543289673E+00, 0.5353281423E+00,
                                0.4446345422E+00}),
      Shell(2, OrbitalType::S,
            // exponents
            std::vector<double>{0.5033151319E+01, 0.1169596125E+01,
                                0.3803889600E+00},
            // coefficients
            std::vector<double>{-0.9996722919E-01, 0.3995128261E+00,
                                0.7001154689E+00}),
      Shell(2, OrbitalType::P,
            // exponents
            std::vector<double>{0.5033151319E+01, 0.1169596125E+01,
                                0.3803889600E+00},
            // coefficients
            std::vector<double>{0.1559162750E+00, 0.6076837186E+00,
                                0.3919573931E+00})};
  std::shared_ptr<BasisSet> manual_basis =
      std::make_shared<BasisSet>(manual_name, shells, structure);

  // create custom basis set object
  std::shared_ptr<BasisSet> basis =
      BasisSet::from_basis_name(basis_set, structure);

  // run hartree fock with both basis sets to ensure they are valid
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [e_scf_default, hf_det_default] =
      scf_solver->run(structure, 0, 1, basis_set);
  auto [e_scf_custom, hf_det_custom] = scf_solver->run(structure, 0, 1, basis);
  auto [e_scf_manual, hf_det_manual] =
      scf_solver->run(structure, 0, 1, manual_basis);

  EXPECT_NEAR(e_scf_custom, e_scf_default, testing::scf_energy_tolerance);
  EXPECT_NEAR(e_scf_manual, e_scf_default, testing::scf_energy_tolerance);
  EXPECT_NEAR(e_scf_custom, e_scf_manual, testing::scf_energy_tolerance);
}

TEST_F(BasisSetTest, SameBasisSetCheckWithEcp) {
  // Compare energies from standard basis set string vs custom BasisSet object
  const std::string basis_set = "def2-tzvp";
  std::shared_ptr<Structure> structure = std::make_shared<Structure>(
      std::vector<Eigen::Vector3d>{
          {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
      std::vector<std::string>{"H", "H", "Te"});

  // create custom basis set object
  std::shared_ptr<BasisSet> basis =
      BasisSet::from_basis_name(basis_set, structure);

  // run hartree fock with both basis sets to ensure they are valid
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [e_scf_default, hf_det_default] =
      scf_solver->run(structure, 0, 1, basis_set);
  auto [e_scf_custom, hf_det_custom] = scf_solver->run(structure, 0, 1, basis);

  EXPECT_NEAR(e_scf_custom, e_scf_default, testing::scf_energy_tolerance);
}

TEST_F(BasisSetTest, CustomBasisSetPerAtomCheck) {
  // Compare energies from standard basis set string vs custom BasisSet object
  std::string basis_set = "def2-SVP";
  auto structure = testing::create_water_structure();

  // create map of atoms with basis sets
  std::map<size_t, std::string> custom_basis_map;
  custom_basis_map[0] = basis_set;
  custom_basis_map[1] = basis_set;
  custom_basis_map[2] = basis_set;
  std::shared_ptr<BasisSet> basis =
      BasisSet::from_index_map(custom_basis_map, structure);

  // run hartree fock with both basis sets to ensure they are valid
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [e_scf_default, hf_det_default] =
      scf_solver->run(structure, 0, 1, basis_set);
  auto [e_scf_custom, hf_det_custom] = scf_solver->run(structure, 0, 1, basis);

  EXPECT_NEAR(e_scf_custom, e_scf_default, testing::scf_energy_tolerance);
}

TEST_F(BasisSetTest, CustomBasisSetAndEcpPerAtomCheck) {
  // Compare energies from standard basis set string vs custom BasisSet object
  std::string basis_set = "def2-SVP";
  auto structure = testing::create_agh_structure();

  // create map of atoms with basis sets
  std::map<size_t, std::string> custom_basis_map;
  custom_basis_map[0] = basis_set;
  custom_basis_map[1] = basis_set;
  std::shared_ptr<BasisSet> basis =
      BasisSet::from_index_map(custom_basis_map, structure);

  // run hartree fock with both basis sets to ensure they are valid
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [e_scf_custom, hf_det_custom] =
      scf_solver->run(structure, 0, 1, basis_set);
  auto [e_scf_default, hf_det_default] =
      scf_solver->run(structure, 0, 1, basis);

  EXPECT_NEAR(e_scf_custom, e_scf_default, testing::scf_energy_tolerance);
}

TEST_F(BasisSetTest, CustomBasisSetPerElementCheck) {
  // Compare energies from standard basis set string vs custom BasisSet object
  std::string basis_set = "def2-SVP";
  auto structure = testing::create_water_structure();

  // create map of atoms with basis sets
  std::map<std::string, std::string> custom_basis_map;
  custom_basis_map["H"] = basis_set;
  custom_basis_map["O"] = basis_set;

  std::shared_ptr<BasisSet> basis =
      BasisSet::from_element_map(custom_basis_map, structure);

  // run hartree fock with both basis sets to ensure they are valid
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  auto [e_scf_custom, hf_det_custom] =
      scf_solver->run(structure, 0, 1, basis_set);
  auto [e_scf_default, hf_det_default] =
      scf_solver->run(structure, 0, 1, basis);

  EXPECT_NEAR(e_scf_custom, e_scf_default, testing::scf_energy_tolerance);
}

TEST_F(BasisSetTest, CustomMixedBasisSetCheck) {
  // Compare energies from standard basis set string vs custom BasisSet object
  std::string basis_set = "sto-3g";
  auto structure = testing::create_water_structure();

  // create map of elements with basis sets
  std::map<std::string, std::string> custom_element_basis_map;
  custom_element_basis_map["H"] = "cc-pvdz";
  custom_element_basis_map["O"] = "sto-3g";

  // create map of atoms with basis sets
  std::map<size_t, std::string> custom_atom_basis_map;
  custom_atom_basis_map[0] = "cc-pvtz";
  custom_atom_basis_map[1] = "sto-3g";
  custom_atom_basis_map[2] = "def2-SVP";

  std::shared_ptr<BasisSet> element_basis =
      BasisSet::from_element_map(custom_element_basis_map, structure);
  std::shared_ptr<BasisSet> atom_basis =
      BasisSet::from_index_map(custom_atom_basis_map, structure);

  // run hartree fock with both basis sets to ensure they are valid
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();

  auto [e_scf_default, hf_det_default] =
      scf_solver->run(structure, 0, 1, basis_set);
  auto [e_scf_element, hf_det_element] =
      scf_solver->run(structure, 0, 1, element_basis);
  auto [e_scf_atom, hf_det_atom] = scf_solver->run(structure, 0, 1, atom_basis);

  // all three energies should be different
  EXPECT_FALSE(std::abs(e_scf_element - e_scf_default) <
               testing::scf_energy_tolerance);
  EXPECT_FALSE(std::abs(e_scf_atom - e_scf_default) <
               testing::scf_energy_tolerance);
  EXPECT_FALSE(std::abs(e_scf_atom - e_scf_element) <
               testing::scf_energy_tolerance);

  // check number of orbitals in determinant
  EXPECT_EQ(hf_det_default->get_orbitals()->get_num_molecular_orbitals(), 7);
  EXPECT_EQ(hf_det_element->get_orbitals()->get_num_molecular_orbitals(), 15);
  EXPECT_EQ(hf_det_atom->get_orbitals()->get_num_molecular_orbitals(), 36);
}

TEST_F(BasisSetTest, SupportedBasisSets) {
  auto supported_basis_sets = BasisSet::get_supported_basis_set_names();
  // Check that some known basis sets are in the supported list
  EXPECT_NE(std::find(supported_basis_sets.begin(), supported_basis_sets.end(),
                      "sto-3g"),
            supported_basis_sets.end());
  EXPECT_NE(std::find(supported_basis_sets.begin(), supported_basis_sets.end(),
                      "cc-pvdz"),
            supported_basis_sets.end());
  EXPECT_NE(std::find(supported_basis_sets.begin(), supported_basis_sets.end(),
                      "def2-tzvp"),
            supported_basis_sets.end());
}

TEST_F(BasisSetTest, SupportedElementsForBasisSet) {
  // Verify that supported elements for a given basis set are correct
  std::string basis_name = "aug-ano-pv5z";
  std::vector<Element> expected_elements = {Element::H,  Element::He,
                                            Element::Li, Element::Be,
                                            Element::Na, Element::Mg};

  auto supported_elements =
      BasisSet::get_supported_elements_for_basis_set(basis_name);

  EXPECT_EQ(expected_elements, supported_elements);
}

// Edge case tests for from_basis_name
TEST_F(BasisSetTest, FromBasisNameNullptrStructure) {
  std::shared_ptr<Structure> null_structure = nullptr;
  EXPECT_THROW(BasisSet::from_basis_name("sto-3g", null_structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, FromBasisNameInvalidBasisSet) {
  auto structure = testing::create_water_structure();
  EXPECT_THROW(BasisSet::from_basis_name("invalid-basis-set-name", structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, FromBasisNameEmptyBasisSet) {
  auto structure = testing::create_water_structure();
  EXPECT_THROW(BasisSet::from_basis_name("", structure), std::invalid_argument);
}

// Edge case tests for from_element_map
TEST_F(BasisSetTest, FromElementMapNullptrStructure) {
  std::map<std::string, std::string> element_map;
  element_map["H"] = "sto-3g";
  element_map["O"] = "sto-3g";
  std::shared_ptr<Structure> null_structure = nullptr;
  EXPECT_THROW(BasisSet::from_element_map(element_map, null_structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, FromElementMapMissingElement) {
  auto structure = testing::create_water_structure();
  std::map<std::string, std::string> element_map;
  // Only specify H, missing O
  element_map["H"] = "sto-3g";
  EXPECT_THROW(BasisSet::from_element_map(element_map, structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, FromElementMapEmpty) {
  auto structure = testing::create_water_structure();
  std::map<std::string, std::string> empty_map;
  EXPECT_THROW(BasisSet::from_element_map(empty_map, structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, FromElementMapInvalidBasisSet) {
  auto structure = testing::create_water_structure();
  std::map<std::string, std::string> element_map;
  element_map["H"] = "invalid-basis-set";
  element_map["O"] = "sto-3g";
  EXPECT_THROW(BasisSet::from_element_map(element_map, structure),
               std::invalid_argument);
}

// Edge case tests for from_index_map
TEST_F(BasisSetTest, FromIndexMapNullptrStructure) {
  std::map<size_t, std::string> index_map;
  index_map[0] = "sto-3g";
  index_map[1] = "sto-3g";
  index_map[2] = "sto-3g";
  std::shared_ptr<Structure> null_structure = nullptr;
  EXPECT_THROW(BasisSet::from_index_map(index_map, null_structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, FromIndexMapMissingAtomIndex) {
  auto structure = testing::create_water_structure();  // 3 atoms
  std::map<size_t, std::string> index_map;
  // Only specify atoms 0 and 1, missing atom 2
  index_map[0] = "sto-3g";
  index_map[1] = "sto-3g";
  EXPECT_THROW(BasisSet::from_index_map(index_map, structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, FromIndexMapEmpty) {
  auto structure = testing::create_water_structure();
  std::map<size_t, std::string> empty_map;
  EXPECT_THROW(BasisSet::from_index_map(empty_map, structure),
               std::invalid_argument);
}

TEST_F(BasisSetTest, FromIndexMapInvalidBasisSet) {
  auto structure = testing::create_water_structure();
  std::map<size_t, std::string> index_map;
  index_map[0] = "invalid-basis-set";
  index_map[1] = "sto-3g";
  index_map[2] = "sto-3g";
  EXPECT_THROW(BasisSet::from_index_map(index_map, structure),
               std::invalid_argument);
}

// Test basis set name normalization for filesystem safety
TEST_F(BasisSetTest, BasisSetNameNormalization) {
  // These functions are in the detail namespace and need to be declared
  // or we need to make them accessible for testing
  namespace detail = qdk::chemistry::data::detail;

  // Test normalization of special characters
  EXPECT_EQ("6-31g_st_", detail::normalize_basis_set_name("6-31g*"));
  EXPECT_EQ("6-31g_st__pl_", detail::normalize_basis_set_name("6-31g*+"));
  EXPECT_EQ("6-31g_st__pl__pl_", detail::normalize_basis_set_name("6-31g*++"));
  EXPECT_EQ("cc-pVTZ_sl_DK", detail::normalize_basis_set_name("cc-pVTZ/DK"));
  EXPECT_EQ("def2-TZVP_pl_", detail::normalize_basis_set_name("def2-TZVP+"));

  // Test denormalization reverses normalization
  EXPECT_EQ("6-31g*", detail::denormalize_basis_set_name("6-31g_st_"));
  EXPECT_EQ("6-31g*+", detail::denormalize_basis_set_name("6-31g_st__pl_"));
  EXPECT_EQ("6-31g*++",
            detail::denormalize_basis_set_name("6-31g_st__pl__pl_"));
  EXPECT_EQ("cc-pVTZ/DK", detail::denormalize_basis_set_name("cc-pVTZ_sl_DK"));
  EXPECT_EQ("def2-TZVP+", detail::denormalize_basis_set_name("def2-TZVP_pl_"));

  // Test round-trip conversion
  std::vector<std::string> test_names = {
      "6-31g*",    "6-31g**",    "6-31g*+",    "6-31g*++",
      "6-311+g*",  "6-311++g**", "cc-pVTZ/DK", "aug-cc-pVTZ/DK",
      "def2-TZVP", "def2-TZVP+", "def2-TZVPP"};

  for (const auto& name : test_names) {
    std::string normalized = detail::normalize_basis_set_name(name);
    std::string denormalized = detail::denormalize_basis_set_name(normalized);
    EXPECT_EQ(name, denormalized)
        << "Round-trip failed for basis set name: " << name;
  }

  // Test that normal names without special characters pass through unchanged
  EXPECT_EQ("sto-3g", detail::normalize_basis_set_name("sto-3g"));
  EXPECT_EQ("sto-3g", detail::denormalize_basis_set_name("sto-3g"));
  EXPECT_EQ("def2-TZVP", detail::normalize_basis_set_name("def2-TZVP"));
  EXPECT_EQ("def2-TZVP", detail::denormalize_basis_set_name("def2-TZVP"));
}

TEST_F(BasisSetTest, DataTypeName) {
  // Test that BasisSet has the correct data type name
  std::vector<Shell> shells;
  shells.emplace_back(
      Shell(0, OrbitalType::S, std::vector{1.0}, std::vector{2.0}));
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure structure(coords, symbols);
  BasisSet basis("6-31G", shells, structure);

  EXPECT_EQ(basis.get_data_type_name(), "basis_set");
}
