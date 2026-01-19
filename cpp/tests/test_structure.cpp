// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/data/structure.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class StructureBasicTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean up any test files from previous runs
    std::filesystem::remove("test.structure.xyz");
    std::filesystem::remove("test.structure.json");
    std::filesystem::remove("test.structure.h5");
    std::filesystem::remove("test_water.structure.h5");
    std::filesystem::remove("test_roundtrip.structure.h5");
  }

  void TearDown() override {
    // Clean up test files
    std::filesystem::remove("test.structure.xyz");
    std::filesystem::remove("test.structure.json");
    std::filesystem::remove("test.structure.h5");
    std::filesystem::remove("test_water.structure.h5");
    std::filesystem::remove("test_roundtrip.structure.h5");
  }
};

// Test basic construction
TEST_F(StructureBasicTest, BasicConstruction) {
  // Create a simple H2 structure to demonstrate immutable construction
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s1(coords, symbols);
  EXPECT_FALSE(s1.is_empty());
  EXPECT_EQ(s1.get_num_atoms(), 2);
}

// Test structure with predefined data
TEST_F(StructureBasicTest, StructureWithData) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s1(coords, symbols);
  EXPECT_EQ(s1.get_num_atoms(), 2);

  // Calculate total nuclear charge manually (now double)
  double total_charge = 0.0;
  for (size_t i = 0; i < s1.get_num_atoms(); ++i) {
    total_charge += s1.get_atom_nuclear_charge(i);
  }
  EXPECT_NEAR(total_charge, 2.0, testing::numerical_zero_tolerance);
}

// Test distance calculation (manual implementation)
TEST_F(StructureBasicTest, DistanceCalculation) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s1(coords, symbols);

  // Calculate distance manually since get_distance() doesn't exist
  Eigen::Vector3d atom1_coords = s1.get_atom_coordinates(0);
  Eigen::Vector3d atom2_coords = s1.get_atom_coordinates(1);
  double distance = (atom2_coords - atom1_coords).norm();
  EXPECT_NEAR(distance, 0.74, testing::numerical_zero_tolerance);
}

// Test XYZ serialization
TEST_F(StructureBasicTest, XYZSerialization) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s1(coords, symbols);

  std::string xyz = s1.to_xyz("H2 molecule");
  EXPECT_FALSE(xyz.empty());

  auto s2 = Structure::from_xyz(xyz);
  EXPECT_EQ(s2->get_num_atoms(), 2);
  EXPECT_EQ(s2->get_atom_symbol(0), "H");
  EXPECT_EQ(s2->get_atom_symbol(1), "H");
}

// Test JSON serialization
TEST_F(StructureBasicTest, JSONSerialization) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s1(coords, symbols);

  auto json_data = s1.to_json();
  EXPECT_FALSE(json_data.empty());

  auto s3 = Structure::from_json(json_data);
  EXPECT_EQ(s3->get_num_atoms(), 2);

  // Calculate total nuclear charge manually (now double)
  double total_charge = 0.0;
  for (size_t i = 0; i < s3->get_num_atoms(); ++i) {
    total_charge += s3->get_atom_nuclear_charge(i);
  }
  EXPECT_NEAR(total_charge, 2.0, testing::numerical_zero_tolerance);
}

// Test more complex molecule (water)
TEST_F(StructureBasicTest, ComplexMoleculeWater) {
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

  std::string water_xyz = water.to_xyz("Water molecule");
  EXPECT_FALSE(water_xyz.empty());
}

// Test basic coordinate operations
TEST_F(StructureBasicTest, CoordinateOperations) {
  std::vector<std::string> symbols = {"O", "H", "H"};
  Eigen::MatrixXd coords(3, 3);
  coords << 0.000000, 0.000000, 0.000000, 0.757000, 0.586000, 0.000000,
      -0.757000, 0.586000, 0.000000;

  Structure water(coords, symbols);

  // Calculate geometric center manually
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < water.get_num_atoms(); ++i) {
    center += water.get_atom_coordinates(i);
  }
  center /= static_cast<double>(water.get_num_atoms());

  // Test that we can access coordinates (but no longer modify after
  // construction)
  Eigen::Vector3d atom0_coords = water.get_atom_coordinates(0);

  // Since Structure is now immutable, we create a new structure with modified
  // coordinates
  std::vector<Eigen::Vector3d> modified_coords = {
      atom0_coords - center, water.get_atom_coordinates(1),
      water.get_atom_coordinates(2)};
  std::vector<std::string> modified_symbols = {"O", "H", "H"};

  Structure modified_water(modified_coords, modified_symbols);
  Eigen::Vector3d new_coords = modified_water.get_atom_coordinates(0);
  EXPECT_NEAR((new_coords - (atom0_coords - center)).norm(), 0.0,
              testing::numerical_zero_tolerance);
}

// Test file I/O
TEST_F(StructureBasicTest, FileIO) {
  std::vector<std::string> symbols = {"O", "H", "H"};
  Eigen::MatrixXd coords(3, 3);
  coords << 0.000000, 0.000000, 0.000000, 0.757000, 0.586000, 0.000000,
      -0.757000, 0.586000, 0.000000;

  Structure water(coords, symbols);
  water.to_xyz_file("test_water.structure.xyz", "Water molecule test");
  water.to_json_file("test_water.structure.json");

  auto water_from_file = Structure::from_xyz_file("test_water.structure.xyz");
  EXPECT_EQ(water_from_file->get_num_atoms(), 3);

  auto water_from_json = Structure::from_json_file("test_water.structure.json");
  EXPECT_EQ(water_from_json->get_num_atoms(), 3);

  // Clean up test files
  std::filesystem::remove("test_water.structure.xyz");
  std::filesystem::remove("test_water.structure.json");
}

// Test summary
TEST_F(StructureBasicTest, Summary) {
  std::vector<std::string> symbols = {"O", "H", "H"};
  Eigen::MatrixXd coords(3, 3);
  coords << 0.000000, 0.000000, 0.000000, 0.757000, 0.586000, 0.000000,
      -0.757000, 0.586000, 0.000000;

  Structure water(coords, symbols);
  std::string summary = water.get_summary();
  EXPECT_FALSE(summary.empty());
}

// Test symbol capitalization fixing
TEST_F(StructureBasicTest, SymbolCapitalizationFix) {
  // Test various incorrect capitalizations - functionality works through
  // constructors
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};

  std::vector<std::string> symbols = {
      "h", "HE", "li", "CA"};  // Various incorrect capitalizations

  Structure s(coords, symbols);

  EXPECT_EQ(s.get_num_atoms(), 4);
  EXPECT_EQ(s.get_atom_symbol(0), "H");
  EXPECT_EQ(s.get_atom_symbol(1), "He");
  EXPECT_EQ(s.get_atom_symbol(2), "Li");
  EXPECT_EQ(s.get_atom_symbol(3), "Ca");
}

// Test with custom masses and nuclear charges
TEST_F(StructureBasicTest, CustomMassesAndCharges) {
  std::vector<std::string> symbols = {"H", "C", "O"};
  Eigen::MatrixXd coords = Eigen::MatrixXd::Random(3, 3);

  // Custom masses
  Eigen::VectorXd custom_masses(3);
  custom_masses << 1.1, 12.2, 16.3;

  // Custom nuclear charges (non-integer)
  Eigen::VectorXd custom_charges(3);
  custom_charges << 1.1, 2.2, 3.3;

  // Test with both custom masses and charges
  Structure s1(coords, symbols, custom_masses, custom_charges);
  EXPECT_EQ(s1.get_num_atoms(), 3);
  EXPECT_NEAR(s1.get_atom_mass(0), 1.1, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s1.get_atom_mass(1), 12.2, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s1.get_atom_mass(2), 16.3, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s1.get_atom_nuclear_charge(0), 1.1,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s1.get_atom_nuclear_charge(1), 2.2,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s1.get_atom_nuclear_charge(2), 3.3,
              testing::numerical_zero_tolerance);

  // Test with only custom charges (use default masses)
  Structure s2(coords, symbols, Eigen::VectorXd(), custom_charges);
  EXPECT_EQ(s2.get_num_atoms(), 3);
  EXPECT_NEAR(s2.get_atom_nuclear_charge(0), 1.1,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2.get_atom_nuclear_charge(1), 2.2,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2.get_atom_nuclear_charge(2), 3.3,
              testing::numerical_zero_tolerance);
}

// Test nuclear charges and masses access
TEST_F(StructureBasicTest, EigenVectorProperties) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H", "C"};

  Structure s(coords, symbols);

  // Test that nuclear charges are Eigen::VectorXd
  const Eigen::VectorXd& charges = s.get_nuclear_charges();
  EXPECT_EQ(charges.size(), 2);
  EXPECT_NEAR(charges(0), 1.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(charges(1), 6.0, testing::numerical_zero_tolerance);

  // Test that masses are Eigen::VectorXd
  const Eigen::VectorXd& masses = s.get_masses();
  EXPECT_EQ(masses.size(), 2);
  EXPECT_NEAR(masses(0), 1.0080,
              testing::numerical_zero_tolerance);  // Should have positive mass
  EXPECT_NEAR(masses(1), 12.011,
              testing::numerical_zero_tolerance);  // Should have positive mass

  // Test constructor with custom charges
  std::vector<double> custom_charges = {1.5, 6.5};  // Fractional charges
  Structure s_custom(coords, symbols, std::vector<double>(), custom_charges);

  EXPECT_NEAR(s_custom.get_atom_nuclear_charge(0), 1.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s_custom.get_atom_nuclear_charge(1), 6.5,
              testing::numerical_zero_tolerance);
}

// Test constructor with symbols and optional charges
TEST_F(StructureBasicTest, ConstructorWithSymbolsAndCharges) {
  std::vector<std::string> symbols = {"H", "he", "LI"};  // Mixed capitalization
  Eigen::MatrixXd coords(3, 3);
  coords << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0;

  // Test constructor with symbols only (default masses and charges)
  Structure s1(coords, symbols);
  EXPECT_EQ(s1.get_num_atoms(), 3);
  EXPECT_EQ(s1.get_atom_symbol(0), "H");
  EXPECT_EQ(s1.get_atom_symbol(1), "He");
  EXPECT_EQ(s1.get_atom_symbol(2), "Li");
  EXPECT_NEAR(s1.get_atom_nuclear_charge(0), 1.0,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s1.get_atom_nuclear_charge(1), 2.0,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s1.get_atom_nuclear_charge(2), 3.0,
              testing::numerical_zero_tolerance);

  // Test constructor with custom charges
  Eigen::VectorXd custom_charges(3);
  custom_charges << 1.1, 2.2, 3.3;
  Structure s2(coords, symbols, Eigen::VectorXd(), custom_charges);
  EXPECT_EQ(s2.get_num_atoms(), 3);
  EXPECT_NEAR(s2.get_atom_nuclear_charge(0), 1.1,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2.get_atom_nuclear_charge(1), 2.2,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2.get_atom_nuclear_charge(2), 3.3,
              testing::numerical_zero_tolerance);

  // Test constructor with custom masses
  Eigen::VectorXd custom_masses(3);
  custom_masses << 1.5, 4.2, 7.8;
  Structure s3(coords, symbols, custom_masses, Eigen::VectorXd());
  EXPECT_EQ(s3.get_num_atoms(), 3);
  EXPECT_NEAR(s3.get_atom_mass(0), 1.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s3.get_atom_mass(1), 4.2, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s3.get_atom_mass(2), 7.8, testing::numerical_zero_tolerance);

  // Test constructor with both custom masses and charges
  Eigen::VectorXd both_masses(3);
  Eigen::VectorXd both_charges(3);
  both_masses << 2.1, 5.3, 8.9;
  both_charges << 1.5, 2.5, 3.5;
  Structure s4(coords, symbols, both_masses, both_charges);
  EXPECT_EQ(s4.get_num_atoms(), 3);
  EXPECT_NEAR(s4.get_atom_mass(0), 2.1, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s4.get_atom_mass(1), 5.3, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s4.get_atom_mass(2), 8.9, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s4.get_atom_nuclear_charge(0), 1.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s4.get_atom_nuclear_charge(1), 2.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s4.get_atom_nuclear_charge(2), 3.5,
              testing::numerical_zero_tolerance);
}

// Test JSON serialization with new types
TEST_F(StructureBasicTest, JSONSerializationNewTypes) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};

  // Set custom fractional charges
  std::vector<double> custom_charges = {1.1, 6.6};

  Structure s1(coords, elements, std::vector<double>(), custom_charges);

  // Serialize to JSON
  auto json_data = s1.to_json();

  // Deserialize from JSON
  auto s2 = Structure::from_json(json_data);

  EXPECT_EQ(s2->get_num_atoms(), 2);
  EXPECT_NEAR(s2->get_atom_nuclear_charge(0), 1.1,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2->get_atom_nuclear_charge(1), 6.6,
              testing::numerical_zero_tolerance);
}

// Test error handling for out-of-range atom access
TEST_F(StructureBasicTest, ErrorHandling) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};

  Structure s(coords, elements);

  EXPECT_EQ(s.get_num_atoms(), 2);

  // Test that accessing valid indices works
  EXPECT_NO_THROW(s.get_atom_coordinates(0));
  EXPECT_NO_THROW(s.get_atom_coordinates(1));
  EXPECT_NO_THROW(s.get_atom_element(0));
  EXPECT_NO_THROW(s.get_atom_element(1));
  EXPECT_NO_THROW(s.get_atom_mass(0));
  EXPECT_NO_THROW(s.get_atom_mass(1));
  EXPECT_NO_THROW(s.get_atom_nuclear_charge(0));
  EXPECT_NO_THROW(s.get_atom_nuclear_charge(1));
  EXPECT_NO_THROW(s.get_atom_symbol(0));
  EXPECT_NO_THROW(s.get_atom_symbol(1));

  // Test that accessing invalid indices throws out_of_range
  EXPECT_THROW(s.get_atom_coordinates(2), std::out_of_range);
  EXPECT_THROW(s.get_atom_coordinates(100), std::out_of_range);
  EXPECT_THROW(s.get_atom_element(2), std::out_of_range);
  EXPECT_THROW(s.get_atom_element(100), std::out_of_range);

  // throw std::out_of_range in get_atom_mass()
  EXPECT_THROW(s.get_atom_mass(2), std::out_of_range);
  EXPECT_THROW(s.get_atom_mass(100), std::out_of_range);

  EXPECT_THROW(s.get_atom_nuclear_charge(2), std::out_of_range);
  EXPECT_THROW(s.get_atom_nuclear_charge(100), std::out_of_range);
}

// Test immutable structure properties
TEST_F(StructureBasicTest, StructureProperties) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};

  Structure s(coords, elements);

  EXPECT_EQ(s.get_num_atoms(), 2);

  // Test that we can access coordinates and other properties
  Eigen::MatrixXd retrieved_coords = s.get_coordinates();
  EXPECT_EQ(retrieved_coords.rows(), 2);
  EXPECT_EQ(retrieved_coords.cols(), 3);

  // Test access to element and mass vectors
  const std::vector<Element>& retrieved_elements = s.get_elements();
  EXPECT_EQ(retrieved_elements.size(), 2);
  EXPECT_EQ(retrieved_elements[0], Element::H);
  EXPECT_EQ(retrieved_elements[1], Element::C);

  const Eigen::VectorXd& masses = s.get_masses();
  EXPECT_EQ(masses.size(), 2);
  EXPECT_NEAR(masses[0], 1.0080,
              testing::numerical_zero_tolerance);  // Should have positive mass
  EXPECT_NEAR(masses[1], 12.011,
              testing::numerical_zero_tolerance);  // Should have positive mass

  const Eigen::VectorXd& charges = s.get_nuclear_charges();
  EXPECT_EQ(charges.size(), 2);
  EXPECT_NEAR(charges[0], 1.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(charges[1], 6.0, testing::numerical_zero_tolerance);
}

// Test constructor variations and custom properties
TEST_F(StructureBasicTest, ConstructorVariationsAndCustomProperties) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::O, Element::N};

  // Test constructor with custom masses and charges
  std::vector<double> custom_masses = {15.999, 14.007};
  std::vector<double> custom_charges = {8.5, 7.5};

  Structure s(coords, elements, custom_masses, custom_charges);

  EXPECT_EQ(s.get_num_atoms(), 2);

  // Verify elements were set correctly
  EXPECT_EQ(s.get_atom_element(0), Element::O);
  EXPECT_EQ(s.get_atom_element(1), Element::N);
  EXPECT_EQ(s.get_atom_symbol(0), "O");
  EXPECT_EQ(s.get_atom_symbol(1), "N");

  // Verify masses were set correctly
  EXPECT_NEAR(s.get_atom_mass(0), 15.999, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s.get_atom_mass(1), 14.007, testing::numerical_zero_tolerance);

  // Verify nuclear charges were set correctly
  EXPECT_NEAR(s.get_atom_nuclear_charge(0), 8.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s.get_atom_nuclear_charge(1), 7.5,
              testing::numerical_zero_tolerance);

  // Test constructor with only custom coordinates (default masses/charges)
  std::vector<Eigen::Vector3d> coords2 = {{2.5, 3.5, 4.5}, {5.5, 6.5, 7.5}};
  Structure s2(coords2, elements);

  Eigen::Vector3d atom0_coords = s2.get_atom_coordinates(0);
  Eigen::Vector3d atom1_coords = s2.get_atom_coordinates(1);
  EXPECT_NEAR(atom0_coords[0], 2.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(atom0_coords[1], 3.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(atom0_coords[2], 4.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(atom1_coords[0], 5.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(atom1_coords[1], 6.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(atom1_coords[2], 7.5, testing::numerical_zero_tolerance);
}

// Test JSON deserialization edge cases and error handling
TEST_F(StructureBasicTest, JSONDeserializationEdgeCases) {
  // Test error: missing units (should throw)
  nlohmann::json json_no_units = {
      {"coordinates", {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}}},
      {"elements", {1, 6}}};
  EXPECT_THROW(Structure::from_json(json_no_units), std::runtime_error);

  // Test error: invalid units value
  nlohmann::json json_bad_units = {
      {"units", "invalid_unit"},
      {"coordinates", {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}}},
      {"elements", {1, 6}}};
  EXPECT_THROW(Structure::from_json(json_bad_units), std::runtime_error);

  // Test error: missing coordinates (should throw)
  nlohmann::json json_no_coords = {{"units", "bohr"}, {"elements", {1, 6}}};
  EXPECT_THROW(Structure::from_json(json_no_coords), std::runtime_error);

  // Test error: invalid coordinates format - not an array
  nlohmann::json json_bad_coords_format = {{"units", "bohr"},
                                           {"coordinates", "not_an_array"}};
  EXPECT_THROW(Structure::from_json(json_bad_coords_format),
               std::runtime_error);

  // Test error: invalid coordinate format for individual atom
  nlohmann::json json_bad_atom_coords = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 1.0, 2.0}, "not_an_array", {4.0, 5.0, 6.0}}},
      {"elements", {1, 6, 8}}};
  EXPECT_THROW(Structure::from_json(json_bad_atom_coords), std::runtime_error);

  // Test error: coordinate array with wrong size
  nlohmann::json json_wrong_coord_size = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 1.0, 2.0}, {3.0, 4.0}, {6.0, 7.0, 8.0}}},
      {"elements", {1, 6, 8}}};
  EXPECT_THROW(Structure::from_json(json_wrong_coord_size), std::runtime_error);

  // Test fallback to nuclear_charges when elements not present
  nlohmann::json json_nuclear_charges = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
      {"nuclear_charges", {1, 6}}  // H and C
  };
  auto s = Structure::from_json(json_nuclear_charges);
  EXPECT_EQ(s->get_num_atoms(), 2);
  EXPECT_EQ(s->get_atom_symbol(0), "H");
  EXPECT_EQ(s->get_atom_symbol(1), "C");

  // Test fallback to symbols when elements and nuclear_charges not present
  nlohmann::json json_symbols = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}}},
      {"symbols", {"H", "he", "LI"}}  // Mixed capitalization
  };
  auto s2 = Structure::from_json(json_symbols);
  EXPECT_EQ(s2->get_num_atoms(), 3);
  EXPECT_EQ(s2->get_atom_symbol(0), "H");
  EXPECT_EQ(s2->get_atom_symbol(1), "He");
  EXPECT_EQ(s2->get_atom_symbol(2), "Li");

  // Test error: missing all element information
  nlohmann::json json_no_elements = {
      {"units", "bohr"}, {"coordinates", {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}}};
  EXPECT_THROW(Structure::from_json(json_no_elements), std::runtime_error);

  // Test standard masses when masses not provided
  nlohmann::json json_no_masses = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
      {"elements", {1, 6}}};
  auto s3 = Structure::from_json(json_no_masses);
  EXPECT_EQ(s3->get_num_atoms(), 2);
  EXPECT_NEAR(
      s3->get_atom_mass(0), 1.0080,
      testing::numerical_zero_tolerance);  // Should have standard H mass
  EXPECT_NEAR(
      s3->get_atom_mass(1), 12.011,
      testing::numerical_zero_tolerance);  // Should have standard C mass

  // Test standard nuclear charges when not provided
  nlohmann::json json_no_nuclear_charges = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
      {"elements", {1, 6}},
      {"masses", {1.008, 12.011}}};
  auto s4 = Structure::from_json(json_no_nuclear_charges);
  EXPECT_EQ(s4->get_num_atoms(), 2);
  EXPECT_NEAR(s4->get_atom_nuclear_charge(0), 1.0,
              testing::numerical_zero_tolerance);  // Standard H charge
  EXPECT_NEAR(s4->get_atom_nuclear_charge(1), 6.0,
              testing::numerical_zero_tolerance);  // Standard C charge

  // Test custom masses and nuclear charges
  nlohmann::json json_custom_values = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
      {"elements", {1, 6}},
      {"masses", {2.014, 13.003}},
      {"nuclear_charges", {1.5, 6.5}}};
  auto s5 = Structure::from_json(json_custom_values);
  EXPECT_EQ(s5->get_num_atoms(), 2);
  EXPECT_NEAR(s5->get_atom_mass(0), 2.014, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s5->get_atom_mass(1), 13.003, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s5->get_atom_nuclear_charge(0), 1.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s5->get_atom_nuclear_charge(1), 6.5,
              testing::numerical_zero_tolerance);

  // Test unit conversion from angstrom to bohr
  nlohmann::json json_angstrom_units = {
      {"units", "angstrom"},
      {"coordinates",
       {{0.0, 0.0, 0.0},
        {qdk::chemistry::constants::bohr_to_angstrom, 0.0, 0.0}}},  // 1 bohr
      {"elements", {1, 1}}};
  auto s6 = Structure::from_json(json_angstrom_units);
  EXPECT_EQ(s6->get_num_atoms(), 2);
  Eigen::Vector3d atom1_coords = s6->get_atom_coordinates(1);
  EXPECT_NEAR(atom1_coords[0], 1.0,
              testing::numerical_zero_tolerance);  // Should be 1 bohr
  EXPECT_NEAR(atom1_coords[1], 0.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(atom1_coords[2], 0.0, testing::numerical_zero_tolerance);

  // Test JSON parsing error - this will test the catch block for
  // JSON exceptions
  nlohmann::json json_type_mismatch = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 0.0, 0.0}}},
      {"elements", "not_a_vector"}  // This should cause a type conversion error
  };
  EXPECT_THROW(Structure::from_json(json_type_mismatch), std::runtime_error);
}

// Test file I/O error handling and XYZ parsing errors
TEST_F(StructureBasicTest, FileIOErrorHandling) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};
  Structure s(coords, elements);

  // Test JSON file writing to invalid path
  EXPECT_THROW(s.to_json_file("/nonexistent_directory/test.structure.json"),
               std::runtime_error);

  // Test JSON file reading from nonexistent file
  EXPECT_THROW(Structure::from_json_file("nonexistent_file.structure.json"),
               std::runtime_error);

  // Test XYZ file writing to invalid path
  EXPECT_THROW(
      s.to_xyz_file("/nonexistent_directory/test.structure.xyz", "comment"),
      std::runtime_error);

  // Test XYZ file reading from nonexistent file
  EXPECT_THROW(Structure::from_xyz_file("nonexistent_file.structure.xyz"),
               std::runtime_error);

  // Test XYZ parsing errors - invalid format
  // Missing number of atoms
  std::string invalid_xyz1 = "";
  EXPECT_THROW(Structure::from_xyz(invalid_xyz1), std::runtime_error);

  // Invalid number of atoms
  std::string invalid_xyz2 = "not_a_number\nComment line\n";
  EXPECT_THROW(Structure::from_xyz(invalid_xyz2), std::runtime_error);

  // Missing comment (after valid atom count)
  std::string invalid_xyz3 = "2\n";  // Missing comment and atom data
  EXPECT_THROW(Structure::from_xyz(invalid_xyz3), std::runtime_error);

  // Missing atom data
  std::string invalid_xyz4 =
      "2\nComment line\nH 0.0 0.0 0.0\n";  // Missing second atom
  EXPECT_THROW(Structure::from_xyz(invalid_xyz4), std::runtime_error);

  // Invalid atom data format
  std::string invalid_xyz5 = "1\nComment line\nH invalid_coord 0.0 0.0\n";
  EXPECT_THROW(Structure::from_xyz(invalid_xyz5), std::runtime_error);

  // Test file write/read error scenarios by creating files with restricted
  // permissions Note: These tests may be platform-dependent

  // Create a test file and try to make it write-protected for testing read-only
  // scenarios
  std::string test_json_file = "test_readonly.structure.json";
  std::string test_xyz_file = "test_readonly.structure.xyz";

  // First write valid files
  s.to_json_file(test_json_file);
  s.to_xyz_file(test_xyz_file, "test comment");

  // Verify we can read them back normally
  EXPECT_NO_THROW(Structure::from_json_file(test_json_file));
  EXPECT_NO_THROW(Structure::from_xyz_file(test_xyz_file));

  // Clean up
  std::filesystem::remove(test_json_file);
  std::filesystem::remove(test_xyz_file);
}

// Test utility functions and edge cases
TEST_F(StructureBasicTest, UtilityFunctionsAndEdgeCases) {
  // Test nuclear_charge_to_element with invalid charges
  EXPECT_THROW(Structure::nuclear_charge_to_element(0),
               std::invalid_argument);  // Too low
  EXPECT_THROW(Structure::nuclear_charge_to_element(119),
               std::invalid_argument);  // Too high

  // Test valid nuclear charges work correctly
  EXPECT_NO_THROW(Structure::nuclear_charge_to_element(1));        // Hydrogen
  EXPECT_NO_THROW(Structure::nuclear_charge_to_element(118));      // Oganesson
  EXPECT_EQ(Structure::nuclear_charge_to_element(6), Element::C);  // Carbon

  // Test get_default_atomic_mass with invalid element
  // This is harder to test directly since Element enum only contains valid
  // elements But we can test it indirectly by using an extreme value
  Element invalid_element = static_cast<Element>(999);  // Invalid element
  EXPECT_THROW(Structure::get_default_atomic_mass(invalid_element),
               std::invalid_argument);

  // Test get_default_atomic_mass with string parameters
  EXPECT_THROW(Structure::get_default_atomic_mass("Zz120"),
               std::invalid_argument);  // Invalid atomic symbol
  EXPECT_THROW(Structure::get_default_atomic_mass("H999"),
               std::invalid_argument);  // Invalid mass number

  // Test get_default_nuclear_charge
  EXPECT_EQ(Structure::get_default_nuclear_charge(Element::H), 1);
  EXPECT_EQ(Structure::get_default_nuclear_charge(Element::C), 6);
  EXPECT_EQ(Structure::get_default_nuclear_charge(Element::O), 8);

  // Test _validate_dimensions with inconsistent empty structure
  // This function is private, so we test it indirectly through constructor

  // Create a structure with mismatched dimensions
  Eigen::MatrixXd coords(1, 3);
  coords << 0.0, 0.0, 0.0;
  std::vector<Element>
      empty_elements;  // Empty elements but non-empty coordinates

  // This should trigger validation error in constructor
  EXPECT_THROW(Structure invalid_structure(coords, empty_elements),
               std::invalid_argument);

  // Also test the case where we have elements but empty coordinates
  Eigen::MatrixXd empty_coords(0, 0);            // Empty coordinates
  std::vector<Element> elements = {Element::H};  // Non-empty elements
  EXPECT_THROW(Structure invalid_structure2(empty_coords, elements),
               std::invalid_argument);

  // Test symbol_to_element with empty string
  EXPECT_THROW(Structure::symbol_to_element(""), std::invalid_argument);

  // Test symbol conversions
  EXPECT_EQ(Structure::symbol_to_element("H"), Element::H);
  EXPECT_EQ(Structure::symbol_to_element("he"),
            Element::He);  // Case insensitive
  EXPECT_EQ(Structure::symbol_to_element("LI"),
            Element::Li);  // Case insensitive
  EXPECT_EQ(Structure::element_to_symbol(Element::C), "C");
  EXPECT_EQ(Structure::element_to_symbol(Element::O), "O");
}

// Test basic HDF5 serialization
TEST_F(StructureBasicTest, HDF5BasicSerialization) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s1(coords, symbols);

  // Test HDF5 file serialization
  s1.to_hdf5_file("test_h2.structure.h5");
  auto s2 = Structure::from_hdf5_file("test_h2.structure.h5");

  EXPECT_EQ(s2->get_num_atoms(), 2);
  EXPECT_EQ(s2->get_atom_symbol(0), "H");
  EXPECT_EQ(s2->get_atom_symbol(1), "H");

  // Verify coordinates match
  for (size_t i = 0; i < s2->get_num_atoms(); ++i) {
    Eigen::Vector3d orig_coords = s1.get_atom_coordinates(i);
    Eigen::Vector3d loaded_coords = s2->get_atom_coordinates(i);
    EXPECT_NEAR(orig_coords[0], loaded_coords[0],
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(orig_coords[1], loaded_coords[1],
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(orig_coords[2], loaded_coords[2],
                testing::numerical_zero_tolerance);
  }

  // Clean up
  std::filesystem::remove("test_h2.structure.h5");
}

// Test HDF5 serialization with custom masses and charges
TEST_F(StructureBasicTest, HDF5CustomProperties) {
  std::vector<std::string> symbols = {"O", "H", "H"};
  Eigen::MatrixXd coords(3, 3);
  coords << 0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0;

  // Custom properties
  Eigen::VectorXd custom_masses(3);
  custom_masses << 15.999, 1.008, 1.008;
  Eigen::VectorXd custom_charges(3);
  custom_charges << 8.5, 1.1, 1.1;

  Structure water(coords, symbols, custom_masses, custom_charges);

  // Test roundtrip through HDF5
  water.to_hdf5_file("test_water_custom.structure.h5");
  auto loaded_water =
      Structure::from_hdf5_file("test_water_custom.structure.h5");

  EXPECT_EQ(loaded_water->get_num_atoms(), 3);

  // Verify custom masses were preserved
  EXPECT_NEAR(loaded_water->get_atom_mass(0), 15.999,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(loaded_water->get_atom_mass(1), 1.008,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(loaded_water->get_atom_mass(2), 1.008,
              testing::numerical_zero_tolerance);

  // Verify custom charges were preserved
  EXPECT_NEAR(loaded_water->get_atom_nuclear_charge(0), 8.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(loaded_water->get_atom_nuclear_charge(1), 1.1,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(loaded_water->get_atom_nuclear_charge(2), 1.1,
              testing::numerical_zero_tolerance);

  // Clean up
  std::filesystem::remove("test_water_custom.structure.h5");
}

// Test HDF5 generic file I/O
TEST_F(StructureBasicTest, HDF5GenericFileIO) {
  std::vector<std::string> symbols = {"C", "H", "H", "H", "H"};
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0},          // C
      {1.089, 0.0, 0.0},        // H
      {-0.363, 1.026, 0.0},     // H
      {-0.363, -0.513, 0.889},  // H
      {-0.363, -0.513, -0.889}  // H
  };

  Structure methane(coords, symbols);

  // Test generic file I/O with HDF5 type
  methane.to_file("test_methane.structure.h5", "hdf5");
  auto loaded_methane =
      Structure::from_file("test_methane.structure.h5", "hdf5");

  EXPECT_EQ(loaded_methane->get_num_atoms(), 5);
  EXPECT_EQ(loaded_methane->get_atom_symbol(0), "C");
  for (size_t i = 1; i < 5; ++i) {
    EXPECT_EQ(loaded_methane->get_atom_symbol(i), "H");
  }

  // Verify coordinates match
  for (size_t i = 0; i < loaded_methane->get_num_atoms(); ++i) {
    Eigen::Vector3d orig_coords = methane.get_atom_coordinates(i);
    Eigen::Vector3d loaded_coords = loaded_methane->get_atom_coordinates(i);
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(orig_coords[j], loaded_coords[j],
                  testing::numerical_zero_tolerance);
    }
  }

  // Clean up
  std::filesystem::remove("test_methane.structure.h5");
}

// Test HDF5 empty structure serialization
TEST_F(StructureBasicTest, HDF5EmptyStructure) {
  std::vector<Eigen::Vector3d> coords;
  std::vector<std::string> symbols;

  Structure empty_structure(coords, symbols);
  EXPECT_TRUE(empty_structure.is_empty());

  // Test HDF5 serialization of empty structure
  empty_structure.to_hdf5_file("test_empty.structure.h5");
  auto loaded_empty = Structure::from_hdf5_file("test_empty.structure.h5");

  EXPECT_TRUE(loaded_empty->is_empty());
  EXPECT_EQ(loaded_empty->get_num_atoms(), 0);

  // Clean up
  std::filesystem::remove("test_empty.structure.h5");
}

// Test HDF5 error handling
TEST_F(StructureBasicTest, HDF5ErrorHandling) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure s(coords, symbols);

  // Test writing to invalid path
  EXPECT_THROW(s.to_hdf5_file("/nonexistent_directory/test.structure.h5"),
               std::runtime_error);

  // Test reading from nonexistent file
  EXPECT_THROW(Structure::from_hdf5_file("nonexistent_file.structure.h5"),
               std::runtime_error);

  // Test generic file I/O with invalid type
  EXPECT_THROW(s.to_file("test.structure.h5", "invalid_type"),
               std::invalid_argument);

  EXPECT_THROW(Structure::from_file("test.structure.h5", "invalid_type"),
               std::invalid_argument);
}

// Test compare HDF5 and JSON serialization results
TEST_F(StructureBasicTest, HDF5vsJSONComparison) {
  std::vector<std::string> symbols = {"O", "H", "H"};
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {0.757, 0.586, 0.0}, {-0.757, 0.586, 0.0}};

  Structure original(coords, symbols);

  // Serialize to both formats
  original.to_hdf5_file("test_comparison.structure.h5");
  original.to_json_file("test_comparison.structure.json");

  // Load from both formats
  auto from_hdf5 = Structure::from_hdf5_file("test_comparison.structure.h5");
  auto from_json = Structure::from_json_file("test_comparison.structure.json");

  // Verify both loaded structures match the original
  EXPECT_EQ(from_hdf5->get_num_atoms(), original.get_num_atoms());
  EXPECT_EQ(from_json->get_num_atoms(), original.get_num_atoms());

  for (size_t i = 0; i < original.get_num_atoms(); ++i) {
    // Check symbols
    EXPECT_EQ(from_hdf5->get_atom_symbol(i), original.get_atom_symbol(i));
    EXPECT_EQ(from_json->get_atom_symbol(i), original.get_atom_symbol(i));

    // Check coordinates
    Eigen::Vector3d orig_coords = original.get_atom_coordinates(i);
    Eigen::Vector3d hdf5_coords = from_hdf5->get_atom_coordinates(i);
    Eigen::Vector3d json_coords = from_json->get_atom_coordinates(i);

    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(hdf5_coords[j], orig_coords[j],
                  testing::numerical_zero_tolerance);
      EXPECT_NEAR(json_coords[j], orig_coords[j],
                  testing::numerical_zero_tolerance);
      EXPECT_NEAR(hdf5_coords[j], json_coords[j],
                  testing::numerical_zero_tolerance);
    }

    // Check masses and charges
    EXPECT_NEAR(from_hdf5->get_atom_mass(i), original.get_atom_mass(i),
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(from_json->get_atom_mass(i), original.get_atom_mass(i),
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(from_hdf5->get_atom_nuclear_charge(i),
                original.get_atom_nuclear_charge(i),
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(from_json->get_atom_nuclear_charge(i),
                original.get_atom_nuclear_charge(i),
                testing::numerical_zero_tolerance);
  }

  // Clean up
  std::filesystem::remove("test_comparison.structure.h5");
  std::filesystem::remove("test_comparison.structure.json");
}

// Test HDF5 large structure performance and data integrity
TEST_F(StructureBasicTest, HDF5LargeStructure) {
  // Create a larger structure to test performance and data integrity
  const size_t num_atoms = 100;
  std::vector<std::string> symbols;
  std::vector<Eigen::Vector3d> coords;

  // Create a repeating pattern of atoms
  std::vector<std::string> pattern = {"C", "H", "O", "N"};
  for (size_t i = 0; i < num_atoms; ++i) {
    symbols.push_back(pattern[i % pattern.size()]);

    // Create coordinates with some pattern
    double x = static_cast<double>(i) * 0.1;
    double y = std::sin(static_cast<double>(i) * 0.1) * 2.0;
    double z = std::cos(static_cast<double>(i) * 0.1) * 2.0;
    coords.push_back({x, y, z});
  }

  // Custom masses and charges with Eigen types
  std::vector<double> custom_masses_vec;
  std::vector<double> custom_charges_vec;
  for (size_t i = 0; i < num_atoms; ++i) {
    custom_masses_vec.push_back(10.0 + static_cast<double>(i) * 0.01);
    custom_charges_vec.push_back(1.0 + static_cast<double>(i % 10) * 0.1);
  }

  Structure large_structure(coords, symbols, custom_masses_vec,
                            custom_charges_vec);
  EXPECT_EQ(large_structure.get_num_atoms(), num_atoms);

  // Test HDF5 serialization
  large_structure.to_hdf5_file("test_large.structure.h5");
  auto loaded_large = Structure::from_hdf5_file("test_large.structure.h5");

  // Verify all data matches
  EXPECT_EQ(loaded_large->get_num_atoms(), num_atoms);

  for (size_t i = 0; i < num_atoms; ++i) {
    // Check symbols
    EXPECT_EQ(loaded_large->get_atom_symbol(i),
              large_structure.get_atom_symbol(i));

    // Check coordinates
    Eigen::Vector3d orig_coords = large_structure.get_atom_coordinates(i);
    Eigen::Vector3d loaded_coords = loaded_large->get_atom_coordinates(i);
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(loaded_coords[j], orig_coords[j],
                  testing::numerical_zero_tolerance);
    }

    // Check masses
    EXPECT_NEAR(loaded_large->get_atom_mass(i),
                large_structure.get_atom_mass(i),
                testing::numerical_zero_tolerance);

    // Check charges
    EXPECT_NEAR(loaded_large->get_atom_nuclear_charge(i),
                large_structure.get_atom_nuclear_charge(i),
                testing::numerical_zero_tolerance);
  }

  // Clean up
  std::filesystem::remove("test_large.structure.h5");
}

// Test HDF5 precision and numerical accuracy
TEST_F(StructureBasicTest, HDF5NumericalPrecision) {
  // Test with very small and very large coordinate values
  std::vector<std::string> symbols = {"H", "H", "H"};
  std::vector<Eigen::Vector3d> coords = {
      {1e-15, 2e-15, 3e-15},  // Very small coordinates
      {1e15, 2e15, 3e15},     // Very large coordinates
      {1.23456789012345, 2.98765432109876, 3.14159265358979}  // High precision
  };

  // Custom masses and charges with high precision
  std::vector<double> custom_masses = {1.0078250322, 1.0078250323,
                                       1.0078250324};
  std::vector<double> custom_charges = {1.0000000001, 1.0000000002,
                                        1.0000000003};

  Structure precision_test(coords, symbols, custom_masses, custom_charges);

  // Test HDF5 roundtrip
  precision_test.to_hdf5_file("test_precision.structure.h5");
  auto loaded_precision =
      Structure::from_hdf5_file("test_precision.structure.h5");

  EXPECT_EQ(loaded_precision->get_num_atoms(), 3);

  // Verify high precision values are preserved
  for (size_t i = 0; i < 3; ++i) {
    Eigen::Vector3d orig_coords = precision_test.get_atom_coordinates(i);
    Eigen::Vector3d loaded_coords = loaded_precision->get_atom_coordinates(i);

    for (int j = 0; j < 3; ++j) {
      // Use very tight tolerance for precision test
      EXPECT_NEAR(loaded_coords[j], orig_coords[j],
                  testing::numerical_zero_tolerance);
    }

    EXPECT_NEAR(loaded_precision->get_atom_mass(i),
                precision_test.get_atom_mass(i),
                testing::numerical_zero_tolerance);
    EXPECT_NEAR(loaded_precision->get_atom_nuclear_charge(i),
                precision_test.get_atom_nuclear_charge(i),
                testing::numerical_zero_tolerance);
  }

  // Clean up
  std::filesystem::remove("test_precision.structure.h5");
}

// Test get_default_atomic_mass with Element
TEST_F(StructureBasicTest, GetDefaultAtomicMassElement) {
  // Test some common elements
  double h_mass = Structure::get_default_atomic_mass(Element::H);
  double c_mass = Structure::get_default_atomic_mass(Element::C);

  EXPECT_NEAR(h_mass, 1.0080, testing::numerical_zero_tolerance);
  EXPECT_NEAR(c_mass, 12.011, testing::numerical_zero_tolerance);
}

// Test get_default_atomic_mass with string parameters
TEST_F(StructureBasicTest, GetDefaultAtomicMassString) {
  // Test standard atomic weights
  double h_mass = Structure::get_default_atomic_mass("H");
  double c_mass = Structure::get_default_atomic_mass("C");

  EXPECT_NEAR(h_mass, 1.0080, testing::numerical_zero_tolerance);
  EXPECT_NEAR(c_mass, 12.011, testing::numerical_zero_tolerance);

  // Test specific isotope masses
  double h1_mass = Structure::get_default_atomic_mass("H1");
  double h2_mass = Structure::get_default_atomic_mass("H2");
  double h3_mass = Structure::get_default_atomic_mass("H3");
  double c12_mass = Structure::get_default_atomic_mass("C12");
  double c13_mass = Structure::get_default_atomic_mass("C13");

  EXPECT_NEAR(h1_mass, 1.007825032, testing::numerical_zero_tolerance);
  EXPECT_NEAR(h2_mass, 2.014101778, testing::numerical_zero_tolerance);
  EXPECT_NEAR(h3_mass, 3.016049281, testing::numerical_zero_tolerance);
  EXPECT_NEAR(c12_mass, 12.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(c13_mass, 13.00335484, testing::numerical_zero_tolerance);

  // Test deuterium and tritium aliases
  double d_mass = Structure::get_default_atomic_mass("D");
  double t_mass = Structure::get_default_atomic_mass("T");
  EXPECT_NEAR(d_mass, h2_mass, testing::numerical_zero_tolerance);
  EXPECT_NEAR(t_mass, h3_mass, testing::numerical_zero_tolerance);
}

// Test Structure with isotope symbols in constructor
TEST_F(StructureBasicTest, ConstructorWithIsotopeSymbols) {
  // Test with isotope notation in symbols
  std::vector<std::string> symbols = {"H1", "D", "C12", "O16"};
  Eigen::MatrixXd coords(4, 3);
  coords << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0;

  Structure s(coords, symbols);

  EXPECT_EQ(s.get_num_atoms(), 4);

  // Check that isotope-specific masses are used
  EXPECT_NEAR(s.get_atom_mass(0), 1.007825032,
              testing::numerical_zero_tolerance);  // H-1
  EXPECT_NEAR(s.get_atom_mass(1), 2.014101778,
              testing::numerical_zero_tolerance);  // D
  EXPECT_NEAR(s.get_atom_mass(2), 12.0,
              testing::numerical_zero_tolerance);  // C-12
  EXPECT_NEAR(s.get_atom_mass(3), 15.99491462,
              testing::numerical_zero_tolerance);  // O-16

  // Check elements
  EXPECT_EQ(s.get_atom_element(0), Element::H);
  EXPECT_EQ(s.get_atom_element(1), Element::H);
  EXPECT_EQ(s.get_atom_element(2), Element::C);
  EXPECT_EQ(s.get_atom_element(3), Element::O);
}

// Test isotope extraction from symbols with mixed notation
TEST_F(StructureBasicTest, MixedSymbolNotation) {
  // Mix standard element symbols and isotope symbols
  std::vector<std::string> symbols = {"H", "D", "C12", "O"};
  Eigen::MatrixXd coords(4, 3);
  coords << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0;

  Structure s(coords, symbols);

  EXPECT_EQ(s.get_num_atoms(), 4);

  // H should use standard atomic weight
  EXPECT_NEAR(s.get_atom_mass(0), 1.0080, testing::numerical_zero_tolerance);
  // D should use deuterium mass
  EXPECT_NEAR(s.get_atom_mass(1), 2.014101778,
              testing::numerical_zero_tolerance);
  // C12 should use C-12 mass
  EXPECT_NEAR(s.get_atom_mass(2), 12.0, testing::numerical_zero_tolerance);
  // O should use standard atomic weight
  EXPECT_NEAR(s.get_atom_mass(3), 15.999, testing::numerical_zero_tolerance);
}

TEST_F(StructureBasicTest, DataTypeName) {
  // Test that Structure has the correct data type name
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};
  Structure s(coords, symbols);

  EXPECT_EQ(s.get_data_type_name(), "structure");
}
