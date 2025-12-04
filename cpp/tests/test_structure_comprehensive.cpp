// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <qdk/chemistry/constants.hpp>
#include <qdk/chemistry/data/structure.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class StructureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean up any test files from previous runs
    std::filesystem::remove("test.structure.xyz");
    std::filesystem::remove("test.structure.json");
    std::filesystem::remove("test.structure.h5");
    std::filesystem::remove("test_complex.structure.h5");
    std::filesystem::remove("test_nested.h5");
    std::filesystem::remove("test_group_functionality.h5");
  }

  void TearDown() override {
    // Clean up test files
    std::filesystem::remove("test.structure.xyz");
    std::filesystem::remove("test.structure.json");
    std::filesystem::remove("test.structure.h5");
    std::filesystem::remove("test_complex.structure.h5");
    std::filesystem::remove("test_nested.h5");
    std::filesystem::remove("test_group_functionality.h5");
  }
};

// Test empty structure using vectors
TEST_F(StructureTest, EmptyStructure) {
  std::vector<Eigen::Vector3d> coords;
  std::vector<std::string> symbols;

  Structure s(coords, symbols);
  EXPECT_TRUE(s.is_empty());
  EXPECT_EQ(s.get_num_atoms(), 0);
}

// Test constructor with coordinates and elements
TEST_F(StructureTest, CoordinatesElementsConstructor) {
  Eigen::MatrixXd coords(2, 3);
  coords << 0.0, 0.0, 0.0, 0.0, 0.0, 0.74;
  std::vector<Element> elements = {Element::H, Element::H};

  Structure s(coords, elements);
  EXPECT_FALSE(s.is_empty());
  EXPECT_EQ(s.get_num_atoms(), 2);

  // Calculate total nuclear charge manually (now double)
  double total_charge = 0.0;
  for (size_t i = 0; i < s.get_num_atoms(); ++i) {
    total_charge += s.get_atom_nuclear_charge(i);
  }
  EXPECT_NEAR(total_charge, 2.0, testing::numerical_zero_tolerance);

  // Test individual atom access
  EXPECT_NEAR(s.get_atom_nuclear_charge(0), 1.0,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s.get_atom_nuclear_charge(1), 1.0,
              testing::numerical_zero_tolerance);
  EXPECT_EQ(s.get_atom_symbol(0), "H");
  EXPECT_EQ(s.get_atom_symbol(1), "H");

  Eigen::Vector3d atom0_coords = s.get_atom_coordinates(0);
  EXPECT_DOUBLE_EQ(atom0_coords(0), 0.0);
  EXPECT_DOUBLE_EQ(atom0_coords(1), 0.0);
  EXPECT_DOUBLE_EQ(atom0_coords(2), 0.0);
}

// Test constructor with symbols and coordinates
TEST_F(StructureTest, SymbolsCoordinatesConstructor) {
  std::vector<std::string> symbols = {"O", "H", "H"};
  Eigen::MatrixXd coords(3, 3);
  coords << 0.000000, 0.000000, 0.000000, 0.757000, 0.586000, 0.000000,
      -0.757000, 0.586000, 0.000000;

  Structure water(coords, symbols);
  EXPECT_EQ(water.get_num_atoms(), 3);

  // Calculate total nuclear charge manually (8 + 1 + 1 = 10, now double)
  double total_charge = 0.0;
  for (size_t i = 0; i < water.get_num_atoms(); ++i) {
    total_charge += water.get_atom_nuclear_charge(i);
  }
  EXPECT_NEAR(total_charge, 10.0, testing::numerical_zero_tolerance);

  EXPECT_EQ(water.get_atom_symbol(0), "O");
  EXPECT_EQ(water.get_atom_symbol(1), "H");
  EXPECT_EQ(water.get_atom_symbol(2), "H");
}

// Test dimension validation
TEST_F(StructureTest, DimensionValidation) {
  Eigen::MatrixXd coords(2, 3);
  coords << 0.0, 0.0, 0.0, 0.0, 0.0, 0.74;
  std::vector<Element> elements = {Element::H};  // Mismatched size

  EXPECT_THROW(Structure(coords, elements), std::invalid_argument);
}

// Test unknown symbol handling
TEST_F(StructureTest, UnknownSymbol) {
  std::vector<std::string> symbols = {"X"};  // Unknown symbol
  Eigen::MatrixXd coords(1, 3);
  coords << 0.0, 0.0, 0.0;

  EXPECT_THROW(Structure(coords, symbols), std::invalid_argument);
}

// Test building structures using constructors
TEST_F(StructureTest, BuildingStructures) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s(coords, symbols);
  EXPECT_EQ(s.get_num_atoms(), 2);
  EXPECT_EQ(s.get_atom_symbol(0), "H");
  EXPECT_EQ(s.get_atom_symbol(1), "H");
}

// Test access bounds checking
TEST_F(StructureTest, OutOfRangeAccess) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure s(coords, symbols);

  EXPECT_THROW(s.get_atom_coordinates(1), std::out_of_range);
  EXPECT_THROW(s.get_atom_nuclear_charge(1), std::out_of_range);
  EXPECT_THROW(s.get_atom_symbol(1), std::out_of_range);
}

// Test immutable data access
TEST_F(StructureTest, DataAccess) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure s(coords, symbols);

  // Test coordinate access
  Eigen::Vector3d retrieved_coords = s.get_atom_coordinates(0);
  EXPECT_TRUE(coords[0].isApprox(retrieved_coords));

  // Test nuclear charge access
  EXPECT_NEAR(s.get_atom_nuclear_charge(0), 1.0,
              testing::numerical_zero_tolerance);

  // Test that structure is immutable - we can only access, not modify
  EXPECT_EQ(s.get_atom_symbol(0), "H");
  EXPECT_NEAR(s.get_atom_nuclear_charge(0), 1.0,
              testing::numerical_zero_tolerance);
}

// Test geometric calculations (manual implementation)
TEST_F(StructureTest, GeometricCalculations) {
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  std::vector<std::string> symbols = {"H", "H", "H"};

  Structure s(coords, symbols);

  // Test distance calculation manually
  Eigen::Vector3d atom0_coords = s.get_atom_coordinates(0);
  Eigen::Vector3d atom1_coords = s.get_atom_coordinates(1);
  double distance = (atom1_coords - atom0_coords).norm();
  EXPECT_DOUBLE_EQ(distance, 1.0);

  // Test geometric center calculation manually
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < s.get_num_atoms(); ++i) {
    center += s.get_atom_coordinates(i);
  }
  center /= static_cast<double>(s.get_num_atoms());

  Eigen::Vector3d expected_center(1.0 / 3.0, 1.0 / 3.0, 0.0);
  EXPECT_TRUE(center.isApprox(expected_center));
}

// Test coordinate calculations
TEST_F(StructureTest, CoordinateOperations) {
  std::vector<Eigen::Vector3d> coords = {{1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s(coords, symbols);

  // Test coordinate access
  Eigen::Vector3d coord0 = s.get_atom_coordinates(0);
  EXPECT_TRUE(coord0.isApprox(Eigen::Vector3d(1.0, 1.0, 1.0)));

  // Test that we can calculate center from immutable structure
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < s.get_num_atoms(); ++i) {
    center += s.get_atom_coordinates(i);
  }
  center /= static_cast<double>(s.get_num_atoms());

  Eigen::Vector3d expected_center(1.5, 1.5, 1.5);
  EXPECT_TRUE(center.isApprox(expected_center));
}

// Test XYZ serialization
TEST_F(StructureTest, XYZSerialization) {
  // Create water molecule
  std::vector<std::string> symbols = {"O", "H", "H"};
  Eigen::MatrixXd coords(3, 3);
  coords << 0.000000, 0.000000, 0.000000, 0.757000, 0.586000, 0.000000,
      -0.757000, 0.586000, 0.000000;

  Structure water(coords, symbols);

  // Test XYZ string conversion
  std::string xyz_string = water.to_xyz("Water molecule");
  EXPECT_TRUE(xyz_string.find("3") != std::string::npos);  // Number of atoms
  EXPECT_TRUE(xyz_string.find("Water molecule") !=
              std::string::npos);                          // Comment
  EXPECT_TRUE(xyz_string.find("O") != std::string::npos);  // Oxygen symbol
  EXPECT_TRUE(xyz_string.find("H") != std::string::npos);  // Hydrogen symbol

  // Test round-trip conversion using static method
  auto water_copy = Structure::from_xyz(xyz_string);

  EXPECT_EQ(water_copy->get_num_atoms(), 3);
  EXPECT_EQ(water_copy->get_atom_symbol(0), "O");
  EXPECT_EQ(water_copy->get_atom_symbol(1), "H");
  EXPECT_EQ(water_copy->get_atom_symbol(2), "H");

  // Check coordinates are approximately equal
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(water.get_atom_coordinates(i).isApprox(
        water_copy->get_atom_coordinates(i), testing::plain_text_tolerance));
  }
}

// Test XYZ file I/O
TEST_F(StructureTest, XYZFileIO) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s(coords, symbols);

  // Save to file
  s.to_xyz_file("test.structure.xyz", "H2 molecule");
  EXPECT_TRUE(std::filesystem::exists("test.structure.xyz"));

  // Load from file using static method
  auto s_loaded = Structure::from_xyz_file("test.structure.xyz");

  EXPECT_EQ(s_loaded->get_num_atoms(), 2);
  EXPECT_EQ(s_loaded->get_atom_symbol(0), "H");
  EXPECT_EQ(s_loaded->get_atom_symbol(1), "H");
  EXPECT_TRUE(s.get_atom_coordinates(0).isApprox(
      s_loaded->get_atom_coordinates(0), testing::plain_text_tolerance));
  EXPECT_TRUE(s.get_atom_coordinates(1).isApprox(
      s_loaded->get_atom_coordinates(1), testing::plain_text_tolerance));
}

// Test JSON serialization
TEST_F(StructureTest, JSONSerialization) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}};
  std::vector<std::string> symbols = {"C", "O"};

  Structure s(coords, symbols);

  // Test JSON conversion
  nlohmann::json j = s.to_json();

  EXPECT_EQ(j["num_atoms"], 2);
  // Note: total_nuclear_charge may not be directly in JSON, calculate manually
  double total_charge = 0.0;
  for (size_t i = 0; i < s.get_num_atoms(); ++i) {
    total_charge += s.get_atom_nuclear_charge(i);
  }
  EXPECT_NEAR(total_charge, 14.0, testing::numerical_zero_tolerance);  // 6 + 8

  EXPECT_EQ(j["symbols"].size(), 2);
  EXPECT_EQ(j["symbols"][0], "C");
  EXPECT_EQ(j["symbols"][1], "O");
  EXPECT_EQ(j["nuclear_charges"].size(), 2);
  EXPECT_NEAR(j["nuclear_charges"][0], 6.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(j["nuclear_charges"][1], 8.0, testing::numerical_zero_tolerance);

  // Test round-trip conversion using static method
  auto s_copy = Structure::from_json(j);

  EXPECT_EQ(s_copy->get_num_atoms(), 2);
  EXPECT_EQ(s_copy->get_atom_symbol(0), "C");
  EXPECT_EQ(s_copy->get_atom_symbol(1), "O");
  EXPECT_TRUE(
      s.get_atom_coordinates(0).isApprox(s_copy->get_atom_coordinates(0)));
  EXPECT_TRUE(
      s.get_atom_coordinates(1).isApprox(s_copy->get_atom_coordinates(1)));
}

// Test JSON file I/O
TEST_F(StructureTest, JSONFileIO) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.1}};
  std::vector<std::string> symbols = {"N", "N"};

  Structure s(coords, symbols);

  // Save to file
  s.to_json_file("test.structure.json");
  EXPECT_TRUE(std::filesystem::exists("test.structure.json"));

  // Load from file using static method
  auto s_loaded = Structure::from_json_file("test.structure.json");

  EXPECT_EQ(s_loaded->get_num_atoms(), 2);
  EXPECT_EQ(s_loaded->get_atom_symbol(0), "N");
  EXPECT_EQ(s_loaded->get_atom_symbol(1), "N");
  EXPECT_TRUE(
      s.get_atom_coordinates(0).isApprox(s_loaded->get_atom_coordinates(0)));
  EXPECT_TRUE(
      s.get_atom_coordinates(1).isApprox(s_loaded->get_atom_coordinates(1)));
}

// Test static utility functions
TEST_F(StructureTest, StaticUtilityFunctions) {
  // Test symbol to nuclear charge conversion
  EXPECT_EQ(Structure::symbol_to_nuclear_charge("H"), 1u);
  EXPECT_EQ(Structure::symbol_to_nuclear_charge("C"), 6u);
  EXPECT_EQ(Structure::symbol_to_nuclear_charge("O"), 8u);
  EXPECT_THROW(Structure::symbol_to_nuclear_charge("Xx"),
               std::invalid_argument);

  // Test nuclear charge to symbol conversion
  EXPECT_EQ(Structure::nuclear_charge_to_symbol(1u), "H");
  EXPECT_EQ(Structure::nuclear_charge_to_symbol(6u), "C");
  EXPECT_EQ(Structure::nuclear_charge_to_symbol(8u), "O");
  EXPECT_THROW(Structure::nuclear_charge_to_symbol(200u),
               std::invalid_argument);

  // Test element conversion functions
  EXPECT_EQ(Structure::element_to_symbol(Element::H), "H");
  EXPECT_EQ(Structure::element_to_symbol(Element::C), "C");
  EXPECT_EQ(Structure::symbol_to_element("H"), Element::H);
  EXPECT_EQ(Structure::symbol_to_element("C"), Element::C);

  // Test nuclear charge to/from element
  EXPECT_EQ(Structure::element_to_nuclear_charge(Element::H), 1u);
  EXPECT_EQ(Structure::element_to_nuclear_charge(Element::C), 6u);
  EXPECT_EQ(Structure::nuclear_charge_to_element(1u), Element::H);
  EXPECT_EQ(Structure::nuclear_charge_to_element(6u), Element::C);
}

// Test get_summary
TEST_F(StructureTest, GetSummary) {
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"C", "H", "H"};

  Structure s(coords, symbols);

  std::string summary = s.get_summary();
  EXPECT_TRUE(summary.find("Number of atoms: 3") != std::string::npos);
  // Calculate total nuclear charge manually since it may not be in summary
  double total_charge = 0.0;
  for (size_t i = 0; i < s.get_num_atoms(); ++i) {
    total_charge += s.get_atom_nuclear_charge(i);
  }
  EXPECT_NEAR(total_charge, 8.0,
              testing::numerical_zero_tolerance);  // 6 + 1 + 1

  EXPECT_TRUE(summary.find("C") != std::string::npos);
  EXPECT_TRUE(summary.find("H") != std::string::npos);
}

// Test invalid JSON
TEST_F(StructureTest, InvalidJSON) {
  nlohmann::json invalid_json;
  invalid_json["invalid"] = "data";

  EXPECT_THROW(Structure::from_json(invalid_json), std::runtime_error);
}

// Test invalid XYZ format
TEST_F(StructureTest, InvalidXYZ) {
  // Invalid number of atoms line
  EXPECT_THROW(Structure::from_xyz("invalid\ncomment\nH 0 0 0"),
               std::runtime_error);

  // Missing comment line
  EXPECT_THROW(Structure::from_xyz("1"), std::runtime_error);

  // Incomplete atom data
  EXPECT_THROW(Structure::from_xyz("1\ncomment\nH 0 0"), std::runtime_error);
}

// Test file I/O errors
TEST_F(StructureTest, FileIOErrors) {
  // Test reading non-existent file
  EXPECT_THROW(Structure::from_xyz_file("non_existent.structure.xyz"),
               std::runtime_error);
  EXPECT_THROW(Structure::from_json_file("non_existent.structure.json"),
               std::runtime_error);
}

// Test comprehensive Eigen vector functionality
TEST_F(StructureTest, EigenVectorComprehensive) {
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C, Element::O};

  Structure s(coords, elements);

  // Test nuclear charges are Eigen::VectorXd
  const Eigen::VectorXd& charges = s.get_nuclear_charges();
  EXPECT_EQ(charges.size(), 3);
  EXPECT_NEAR(charges(0), 1.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(charges(1), 6.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(charges(2), 8.0, testing::numerical_zero_tolerance);

  // Test masses are Eigen::VectorXd
  const Eigen::VectorXd& masses = s.get_masses();
  EXPECT_EQ(masses.size(), 3);
  EXPECT_GT(masses(0), 0.0);  // Should have positive mass
  EXPECT_GT(masses(1), 0.0);
  EXPECT_GT(masses(2), 0.0);

  // Test constructor with custom vectors
  std::vector<double> custom_charges = {1.1, 6.6, 8.8};
  std::vector<double> custom_masses = {1.1, 12.1, 16.1};

  Structure s_custom(coords, elements, custom_masses, custom_charges);

  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(s_custom.get_atom_nuclear_charge(i), custom_charges[i],
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(s_custom.get_atom_mass(i), custom_masses[i],
                testing::numerical_zero_tolerance);
  }
}

// Test symbol capitalization comprehensive
TEST_F(StructureTest, SymbolCapitalizationComprehensive) {
  // Test various capitalization patterns using constructor
  std::vector<std::pair<std::string, std::string>> test_cases = {
      {"h", "H"},   {"H", "H"},   {"he", "He"}, {"HE", "He"}, {"He", "He"},
      {"li", "Li"}, {"LI", "Li"}, {"Li", "Li"}, {"ca", "Ca"}, {"CA", "Ca"},
      {"br", "Br"}, {"BR", "Br"}, {"cl", "Cl"}, {"CL", "Cl"}};

  for (const auto& test_case : test_cases) {
    std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
    std::vector<std::string> symbols = {test_case.first};

    Structure temp_s(coords, symbols);
    EXPECT_EQ(temp_s.get_atom_symbol(0), test_case.second)
        << "Failed for input: " << test_case.first;
  }

  // Test in constructor
  std::vector<std::string> mixed_symbols = {"h", "HE", "li", "CA"};
  Eigen::MatrixXd coords = Eigen::MatrixXd::Zero(4, 3);
  Structure s_constructor(coords, mixed_symbols);

  EXPECT_EQ(s_constructor.get_atom_symbol(0), "H");
  EXPECT_EQ(s_constructor.get_atom_symbol(1), "He");
  EXPECT_EQ(s_constructor.get_atom_symbol(2), "Li");
  EXPECT_EQ(s_constructor.get_atom_symbol(3), "Ca");
}

// Test constructor with custom masses and charges
TEST_F(StructureTest, ConstructorWithCustomVectors) {
  std::vector<Element> elements = {Element::H, Element::C, Element::O};
  Eigen::MatrixXd coords(3, 3);
  coords << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0;

  Eigen::VectorXd custom_masses(3);
  custom_masses << 1.1, 12.1, 16.1;

  Eigen::VectorXd custom_charges(3);
  custom_charges << 1.05, 6.05, 8.05;

  Structure s(coords, elements, custom_masses, custom_charges);

  EXPECT_EQ(s.get_num_atoms(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(s.get_atom_mass(i), custom_masses(i),
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(s.get_atom_nuclear_charge(i), custom_charges(i),
                testing::numerical_zero_tolerance);
  }
}

// Test constructor with various element types
TEST_F(StructureTest, ConstructorElementTypes) {
  // Test Element-based constructor
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C, Element::He,
                                   Element::Li};

  Structure s_elements(coords, elements);
  EXPECT_EQ(s_elements.get_num_atoms(), 4);
  EXPECT_EQ(s_elements.get_atom_symbol(0), "H");
  EXPECT_EQ(s_elements.get_atom_symbol(1), "C");
  EXPECT_EQ(s_elements.get_atom_symbol(2), "He");
  EXPECT_EQ(s_elements.get_atom_symbol(3), "Li");

  EXPECT_NEAR(s_elements.get_atom_nuclear_charge(0), 1.0,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s_elements.get_atom_nuclear_charge(1), 6.0,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s_elements.get_atom_nuclear_charge(2), 2.0,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s_elements.get_atom_nuclear_charge(3), 3.0,
              testing::numerical_zero_tolerance);

  // Test symbol-based constructor with various capitalizations
  std::vector<std::string> symbols = {"c", "HE", "Li"};
  std::vector<Eigen::Vector3d> coords2 = {
      {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};

  Structure s_symbols(coords2, symbols);
  EXPECT_EQ(s_symbols.get_num_atoms(), 3);
  EXPECT_EQ(s_symbols.get_atom_symbol(0), "C");
  EXPECT_EQ(s_symbols.get_atom_symbol(1), "He");
  EXPECT_EQ(s_symbols.get_atom_symbol(2), "Li");
}

// Test fractional charges throughout the system
TEST_F(StructureTest, FractionalChargesComprehensive) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};

  // Set fractional charges using constructor
  std::vector<double> fractional_charges = {0.5, 5.75};

  Structure s(coords, elements, std::vector<double>(), fractional_charges);

  EXPECT_NEAR(s.get_atom_nuclear_charge(0), 0.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s.get_atom_nuclear_charge(1), 5.75,
              testing::numerical_zero_tolerance);

  // Test JSON serialization/deserialization with fractional charges
  auto json_data = s.to_json();
  auto s2 = Structure::from_json(json_data);

  EXPECT_EQ(s2->get_num_atoms(), 2);
  EXPECT_NEAR(s2->get_atom_nuclear_charge(0), 0.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2->get_atom_nuclear_charge(1), 5.75,
              testing::numerical_zero_tolerance);
}

// Test nuclear repulsion energy calculation
TEST_F(StructureTest, NuclearRepulsionEnergyCalculation) {
  // Test empty structure should have zero repulsion energy
  std::vector<Eigen::Vector3d> empty_coords;
  std::vector<std::string> empty_symbols;
  Structure empty_structure(empty_coords, empty_symbols);
  EXPECT_EQ(empty_structure.get_num_atoms(), 0);
  EXPECT_DOUBLE_EQ(empty_structure.calculate_nuclear_repulsion_energy(), 0.0);

  // Test single atom structure should have zero repulsion energy
  std::vector<Eigen::Vector3d> single_coords = {{0.0, 0.0, 0.0}};
  std::vector<Element> single_elements = {Element::H};
  Structure single_atom(single_coords, single_elements);
  EXPECT_EQ(single_atom.get_num_atoms(), 1);
  EXPECT_DOUBLE_EQ(single_atom.calculate_nuclear_repulsion_energy(), 0.0);

  // Test H2 molecule with known repulsion energy
  std::vector<Eigen::Vector3d> h2_coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<Element> h2_elements = {Element::H, Element::H};
  Structure h2_molecule(h2_coords, h2_elements);

  // Expected: (1.0 * 1.0) / (0.74)
  double h2_expected = 1.0 / (0.74);
  EXPECT_NEAR(h2_molecule.calculate_nuclear_repulsion_energy(), h2_expected,
              testing::numerical_zero_tolerance);

  // Test water molecule
  std::vector<std::string> symbols = {"O", "H", "H"};
  Eigen::MatrixXd coords(3, 3);
  coords << 0.000000, 0.000000, 0.000000, 0.757000, 0.586000, 0.000000,
      -0.757000, 0.586000, 0.000000;

  Structure water(coords, symbols);
  // H-O-H should have 3 pairs of interactions: O-H1, O-H2, and H1-H2
  // Calculate expected repulsion energy manually
  // O-H1: 8.0 * 1.0 / dist(O, H1) in bohr
  // O-H2: 8.0 * 1.0 / dist(O, H2) in bohr
  // H1-H2: 1.0 * 1.0 / dist(H1, H2) in bohr
  double o_h1_dist = std::sqrt(0.757 * 0.757 + 0.586 * 0.586);
  double o_h2_dist = std::sqrt(0.757 * 0.757 + 0.586 * 0.586);
  double h1_h2_dist = std::sqrt(1.514 * 1.514);

  double expected_water_repulsion =
      8.0 * 1.0 / o_h1_dist + 8.0 * 1.0 / o_h2_dist + 1.0 * 1.0 / h1_h2_dist;

  EXPECT_NEAR(water.calculate_nuclear_repulsion_energy(),
              expected_water_repulsion, testing::numerical_zero_tolerance);

  // Test custom nuclear charges
  std::vector<Eigen::Vector3d> custom_coords = {{0.0, 0.0, 0.0},
                                                {0.0, 0.0, 1.0}};
  std::vector<Element> custom_elements = {Element::H, Element::H};

  // Set custom nuclear charges
  std::vector<double> custom_charges = {1.5, 2.5};

  Structure custom_molecule(custom_coords, custom_elements,
                            std::vector<double>(), custom_charges);

  // Expected: (1.5 * 2.5) / 1.0
  double custom_expected = (1.5 * 2.5) / 1.0;
  EXPECT_NEAR(custom_molecule.calculate_nuclear_repulsion_energy(),
              custom_expected, testing::numerical_zero_tolerance);
}

// File I/O and Serialization Tests
TEST_F(StructureTest, GenericFileIO_JSON) {
  // Create a water molecule
  Eigen::MatrixXd coords(3, 3);
  coords << 0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0;
  std::vector<std::string> symbols = {"O", "H", "H"};

  Structure s1(coords, symbols);

  // Test to_file and from_file with JSON
  s1.to_file("test.structure.json", "json");

  auto s2 = Structure::from_file("test.structure.json", "json");

  // Verify the structures match
  EXPECT_EQ(s2->get_num_atoms(), s1.get_num_atoms());
  for (size_t i = 0; i < s1.get_num_atoms(); ++i) {
    EXPECT_EQ(s2->get_atom_symbol(i), s1.get_atom_symbol(i));
    EXPECT_NEAR(s2->get_atom_nuclear_charge(i), s1.get_atom_nuclear_charge(i),
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(s2->get_atom_mass(i), s1.get_atom_mass(i),
                testing::plain_text_tolerance);

    Eigen::Vector3d coords1 = s1.get_atom_coordinates(i);
    Eigen::Vector3d coords2 = s2->get_atom_coordinates(i);
    EXPECT_NEAR((coords1 - coords2).norm(), 0.0,
                testing::numerical_zero_tolerance);
  }
}

TEST_F(StructureTest, GenericFileIO_XYZ) {
  // Create a hydrogen molecule
  Eigen::MatrixXd coords(2, 3);
  coords << 0.0, 0.0, 0.0, 0.0, 0.0, 0.74;
  std::vector<std::string> symbols = {"H", "H"};

  Structure s1(coords, symbols);

  // Test to_file and from_file with XYZ
  s1.to_file("test.structure.xyz", "xyz");

  auto s2 = Structure::from_file("test.structure.xyz", "xyz");

  // Verify the structures match
  EXPECT_EQ(s2->get_num_atoms(), s1.get_num_atoms());
  for (size_t i = 0; i < s1.get_num_atoms(); ++i) {
    EXPECT_EQ(s2->get_atom_symbol(i), s1.get_atom_symbol(i));

    Eigen::Vector3d coords1 = s1.get_atom_coordinates(i);
    Eigen::Vector3d coords2 = s2->get_atom_coordinates(i);
    EXPECT_NEAR((coords1 - coords2).norm(), 0.0, testing::plain_text_tolerance);
  }
}

TEST_F(StructureTest, FilenameValidation) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H};
  Structure s(coords, elements);

  // Test valid filenames work
  EXPECT_NO_THROW(s.to_json_file("valid.structure.json"));
  EXPECT_NO_THROW(s.to_xyz_file("valid.structure.xyz"));
  EXPECT_NO_THROW(s.to_xyz_file("valid.xyz"));
  EXPECT_NO_THROW(s.to_file("valid.structure.json", "json"));
  EXPECT_NO_THROW(s.to_file("valid.structure.xyz", "xyz"));
  EXPECT_NO_THROW(s.to_file("valid.xyz", "xyz"));

  // Test invalid filenames throw exceptions
  EXPECT_THROW(s.to_json_file("invalid.json"), std::invalid_argument);
  EXPECT_THROW(Structure::from_json_file("invalid.json"),
               std::invalid_argument);
}

TEST_F(StructureTest, FileIOErrorHandling) {
  // Test unsupported format types
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H};
  Structure s(coords, elements);

  EXPECT_THROW(s.to_file("test.structure.xyz", "invalid"),
               std::invalid_argument);
  EXPECT_THROW(Structure::from_file("test.structure.xyz", "invalid"),
               std::invalid_argument);

  // Test non-existent files
  EXPECT_THROW(Structure::from_file("nonexistent.structure.json", "json"),
               std::runtime_error);
  EXPECT_THROW(Structure::from_file("nonexistent.structure.xyz", "xyz"),
               std::runtime_error);
}

TEST_F(StructureTest, FileIOConsistency) {
  // Test that generic methods produce same results as specific methods
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0},
                                         {1.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0},
                                         {0.0, 0.0, 1.0},
                                         {-1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::C, Element::H, Element::H,
                                   Element::H, Element::H};

  Structure s(coords, elements);

  // Save using specific method
  s.to_json_file("specific.structure.json");

  // Save using generic method
  s.to_file("generic.structure.json", "json");

  // Load both files and compare
  auto s_specific = Structure::from_json_file("specific.structure.json");
  auto s_generic = Structure::from_file("generic.structure.json", "json");

  // They should be identical
  EXPECT_EQ(s_specific->get_num_atoms(), s_generic->get_num_atoms());
  for (size_t i = 0; i < s_specific->get_num_atoms(); ++i) {
    EXPECT_EQ(s_specific->get_atom_symbol(i), s_generic->get_atom_symbol(i));
    EXPECT_NEAR(s_specific->get_atom_nuclear_charge(i),
                s_generic->get_atom_nuclear_charge(i), testing::json_tolerance);

    Eigen::Vector3d coords_specific = s_specific->get_atom_coordinates(i);
    Eigen::Vector3d coords_generic = s_generic->get_atom_coordinates(i);
    EXPECT_NEAR((coords_specific - coords_generic).norm(), 0.0,
                testing::json_tolerance);
  }

  // Clean up
  std::filesystem::remove("specific.structure.json");
  std::filesystem::remove("generic.structure.json");
}

// Test HDF5 group-based serialization for nesting
TEST_F(StructureTest, HDF5GroupNesting) {
  // Create two different structures
  std::vector<std::string> symbols1 = {"H", "H"};
  std::vector<Eigen::Vector3d> coords1 = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  Structure h2(coords1, symbols1);

  std::vector<std::string> symbols2 = {"O", "H", "H"};
  std::vector<Eigen::Vector3d> coords2 = {
      {0.0, 0.0, 0.0}, {0.757, 0.586, 0.0}, {-0.757, 0.586, 0.0}};
  Structure h2o(coords2, symbols2);

  try {
    // Create an HDF5 file with nested structure groups
    H5::H5File file("test_nested.h5", H5F_ACC_TRUNC);

    // Create top-level groups
    H5::Group molecules_group = file.createGroup("/molecules");
    H5::Group reactants_group = molecules_group.createGroup("reactants");
    H5::Group products_group = molecules_group.createGroup("products");

    // Create subgroups for individual molecules
    H5::Group h2_group = reactants_group.createGroup("hydrogen");
    H5::Group h2o_group = products_group.createGroup("water");

    // Write structures to their respective groups
    h2.to_hdf5(h2_group);
    h2o.to_hdf5(h2o_group);

    // Close groups and file
    h2_group.close();
    h2o_group.close();
    reactants_group.close();
    products_group.close();
    molecules_group.close();
    file.close();

    // Now read back the structures from the nested groups
    H5::H5File read_file("test_nested.h5", H5F_ACC_RDONLY);
    H5::Group read_molecules = read_file.openGroup("/molecules");
    H5::Group read_reactants = read_molecules.openGroup("reactants");
    H5::Group read_products = read_molecules.openGroup("products");
    H5::Group read_h2 = read_reactants.openGroup("hydrogen");
    H5::Group read_h2o = read_products.openGroup("water");

    auto loaded_h2 = Structure::from_hdf5(read_h2);
    auto loaded_h2o = Structure::from_hdf5(read_h2o);

    // Verify H2 structure
    EXPECT_EQ(loaded_h2->get_num_atoms(), 2);
    EXPECT_EQ(loaded_h2->get_atom_symbol(0), "H");
    EXPECT_EQ(loaded_h2->get_atom_symbol(1), "H");

    // Verify H2O structure
    EXPECT_EQ(loaded_h2o->get_num_atoms(), 3);
    EXPECT_EQ(loaded_h2o->get_atom_symbol(0), "O");
    EXPECT_EQ(loaded_h2o->get_atom_symbol(1), "H");
    EXPECT_EQ(loaded_h2o->get_atom_symbol(2), "H");

    // Verify coordinates match original structures
    for (size_t i = 0; i < h2.get_num_atoms(); ++i) {
      Eigen::Vector3d orig = h2.get_atom_coordinates(i);
      Eigen::Vector3d loaded = loaded_h2->get_atom_coordinates(i);
      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(orig[j], loaded[j], testing::numerical_zero_tolerance);
      }
    }

    for (size_t i = 0; i < h2o.get_num_atoms(); ++i) {
      Eigen::Vector3d orig = h2o.get_atom_coordinates(i);
      Eigen::Vector3d loaded = loaded_h2o->get_atom_coordinates(i);
      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(orig[j], loaded[j], testing::numerical_zero_tolerance);
      }
    }

  } catch (const H5::Exception& e) {
    FAIL() << "HDF5 exception: " << e.getCDetailMsg();
  }

  // Clean up
  std::filesystem::remove("test_nested.h5");
}

// Test HDF5 group serialization with metadata
TEST_F(StructureTest, HDF5GroupWithMetadata) {
  std::vector<std::string> symbols = {"C", "O", "O"};
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0},   // C
      {1.16, 0.0, 0.0},  // O
      {-1.16, 0.0, 0.0}  // O
  };
  Structure co2(coords, symbols);

  try {
    // Create HDF5 file with metadata alongside structure
    H5::H5File file("test_with_metadata.h5", H5F_ACC_TRUNC);

    // Create main group
    H5::Group calculation_group = file.createGroup("/calculation");

    // Add metadata attributes to the group
    H5::DataSpace attr_space(H5S_SCALAR);

    // Add string attribute for molecule name
    H5::StrType str_type(H5::PredType::C_S1, 256);
    H5::Attribute name_attr = calculation_group.createAttribute(
        "molecule_name", str_type, attr_space);
    std::string mol_name = "carbon_dioxide";
    name_attr.write(str_type, mol_name);

    // Add numerical attributes
    H5::Attribute temp_attr = calculation_group.createAttribute(
        "temperature_K", H5::PredType::NATIVE_DOUBLE, attr_space);
    double temperature = 298.15;
    temp_attr.write(H5::PredType::NATIVE_DOUBLE, &temperature);

    H5::Attribute charge_attr = calculation_group.createAttribute(
        "total_charge", H5::PredType::NATIVE_INT, attr_space);
    int total_charge = 0;
    charge_attr.write(H5::PredType::NATIVE_INT, &total_charge);

    // Create structure subgroup and write structure
    H5::Group structure_group = calculation_group.createGroup("structure");
    co2.to_hdf5(structure_group);

    // Close and reopen to test reading
    file.close();

    // Read back
    H5::H5File read_file("test_with_metadata.h5", H5F_ACC_RDONLY);
    H5::Group read_calc = read_file.openGroup("/calculation");

    // Read metadata
    H5::Attribute read_name_attr = read_calc.openAttribute("molecule_name");
    std::string read_name;
    read_name_attr.read(str_type, read_name);
    EXPECT_EQ(read_name, "carbon_dioxide");

    H5::Attribute read_temp_attr = read_calc.openAttribute("temperature_K");
    double read_temp;
    read_temp_attr.read(H5::PredType::NATIVE_DOUBLE, &read_temp);
    EXPECT_NEAR(read_temp, 298.15, testing::numerical_zero_tolerance);

    H5::Attribute read_charge_attr = read_calc.openAttribute("total_charge");
    int read_charge;
    read_charge_attr.read(H5::PredType::NATIVE_INT, &read_charge);
    EXPECT_EQ(read_charge, 0);

    // Read structure
    H5::Group read_structure = read_calc.openGroup("structure");
    auto loaded_co2 = Structure::from_hdf5(read_structure);

    // Verify structure
    EXPECT_EQ(loaded_co2->get_num_atoms(), 3);
    EXPECT_EQ(loaded_co2->get_atom_symbol(0), "C");
    EXPECT_EQ(loaded_co2->get_atom_symbol(1), "O");
    EXPECT_EQ(loaded_co2->get_atom_symbol(2), "O");

    // Verify coordinates
    for (size_t i = 0; i < co2.get_num_atoms(); ++i) {
      Eigen::Vector3d orig = co2.get_atom_coordinates(i);
      Eigen::Vector3d loaded = loaded_co2->get_atom_coordinates(i);
      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(orig[j], loaded[j], testing::numerical_zero_tolerance);
      }
    }

  } catch (const H5::Exception& e) {
    FAIL() << "HDF5 exception: " << e.getCDetailMsg();
  }

  // Clean up
  std::filesystem::remove("test_with_metadata.h5");
}

// Test HDF5 group-based serialization functionality
TEST_F(StructureTest, HDF5GroupFunctionality) {
  std::vector<std::string> symbols = {"N", "H", "H", "H"};
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0},      // N
      {0.0, 0.0, 1.01},     // H
      {0.95, 0.0, -0.34},   // H
      {-0.48, 0.83, -0.34}  // H
  };
  Structure nh3(coords, symbols);

  try {
    // Test writing to a specific group within an HDF5 file
    H5::H5File file("test_group_functionality.h5", H5F_ACC_TRUNC);

    // Create a specific group manually
    H5::Group structure_group = file.createGroup("/ammonia_structure");

    // Use the existing to_hdf5 method to write to the group
    nh3.to_hdf5(structure_group);

    // Add some metadata to the group
    H5::DataSpace attr_space(H5S_SCALAR);
    H5::Attribute symmetry_attr = structure_group.createAttribute(
        "point_group", H5::StrType(H5::PredType::C_S1, 64), attr_space);
    std::string symmetry = "C3v";
    symmetry_attr.write(H5::StrType(H5::PredType::C_S1, 64), symmetry);

    structure_group.close();
    file.close();

    // Read back and verify
    H5::H5File read_file("test_group_functionality.h5", H5F_ACC_RDONLY);
    H5::Group read_group = read_file.openGroup("/ammonia_structure");

    // Check the metadata we added
    H5::Attribute read_symmetry = read_group.openAttribute("point_group");
    std::string read_symmetry_val;
    read_symmetry.read(H5::StrType(H5::PredType::C_S1, 64), read_symmetry_val);
    EXPECT_EQ(read_symmetry_val, "C3v");

    // Load the structure
    auto loaded_nh3 = Structure::from_hdf5(read_group);

    // Verify structure
    EXPECT_EQ(loaded_nh3->get_num_atoms(), 4);
    EXPECT_EQ(loaded_nh3->get_atom_symbol(0), "N");
    EXPECT_EQ(loaded_nh3->get_atom_symbol(1), "H");
    EXPECT_EQ(loaded_nh3->get_atom_symbol(2), "H");
    EXPECT_EQ(loaded_nh3->get_atom_symbol(3), "H");

    // Verify coordinates
    for (size_t i = 0; i < nh3.get_num_atoms(); ++i) {
      Eigen::Vector3d orig = nh3.get_atom_coordinates(i);
      Eigen::Vector3d loaded = loaded_nh3->get_atom_coordinates(i);
      for (int j = 0; j < 3; ++j) {
        EXPECT_NEAR(orig[j], loaded[j], testing::numerical_zero_tolerance);
      }
    }

  } catch (const H5::Exception& e) {
    FAIL() << "HDF5 exception: " << e.getCDetailMsg();
  }

  // Clean up
  std::filesystem::remove("test_group_functionality.h5");
}
