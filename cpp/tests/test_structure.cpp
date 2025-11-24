// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/constants.hpp>
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
    std::filesystem::remove("test_nested.h5");
    std::filesystem::remove("test_with_metadata.h5");
    std::filesystem::remove("test_group_functionality.h5");
  }

  void TearDown() override {
    // Clean up test files
    std::filesystem::remove("test.structure.xyz");
    std::filesystem::remove("test.structure.json");
    std::filesystem::remove("test.structure.h5");
    std::filesystem::remove("test_water.structure.h5");
    std::filesystem::remove("test_roundtrip.structure.h5");
    std::filesystem::remove("test_nested.h5");
    std::filesystem::remove("test_with_metadata.h5");
    std::filesystem::remove("test_group_functionality.h5");
  }
};

// Basic Construction and Properties Tests
TEST_F(StructureBasicTest, BasicConstruction) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure s1(coords, symbols);
  EXPECT_FALSE(s1.is_empty());
  EXPECT_EQ(s1.get_num_atoms(), 2);

  // Calculate total nuclear charge
  double total_charge = 0.0;
  for (size_t i = 0; i < s1.get_num_atoms(); ++i) {
    total_charge += s1.get_atom_nuclear_charge(i);
  }
  EXPECT_NEAR(total_charge, 2.0, testing::numerical_zero_tolerance);

  // Test individual atom properties
  EXPECT_EQ(s1.get_atom_nuclear_charge(0), 1);
  EXPECT_EQ(s1.get_atom_nuclear_charge(1), 1);
  EXPECT_EQ(s1.get_atom_symbol(0), "H");
  EXPECT_EQ(s1.get_atom_symbol(1), "H");

  Eigen::Vector3d atom0_coords = s1.get_atom_coordinates(0);
  EXPECT_NEAR(atom0_coords[0], 0.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(atom0_coords[1], 0.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(atom0_coords[2], 0.0, testing::numerical_zero_tolerance);
}

// Different molecule test (water)
TEST_F(StructureBasicTest, WaterMolecule) {
  Eigen::MatrixXd coords(3, 3);
  coords << 0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0;
  std::vector<std::string> symbols = {"O", "H", "H"};
  Structure water(coords, symbols);

  EXPECT_EQ(water.get_num_atoms(), 3);

  // Total nuclear charge (8 + 1 + 1 = 10)
  double total_charge = 0.0;
  for (size_t i = 0; i < water.get_num_atoms(); ++i) {
    total_charge += water.get_atom_nuclear_charge(i);
  }
  EXPECT_NEAR(total_charge, 10.0, testing::numerical_zero_tolerance);

  EXPECT_EQ(water.get_atom_symbol(0), "O");
  EXPECT_EQ(water.get_atom_symbol(1), "H");
  EXPECT_EQ(water.get_atom_symbol(2), "H");
}

TEST_F(StructureBasicTest, ConstructorVariations) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};

  // Element-based constructor
  std::vector<Element> elements = {Element::O, Element::N};
  Structure s1(coords, elements);
  EXPECT_EQ(s1.get_atom_symbol(0), "O");
  EXPECT_EQ(s1.get_atom_symbol(1), "N");

  // Symbol-based constructor with custom masses and charges
  std::vector<std::string> symbols = {"O", "N"};
  std::vector<double> custom_masses = {15.999, 14.007};
  std::vector<double> custom_charges = {8.5, 7.5};
  Structure s2(coords, symbols, custom_masses, custom_charges);

  EXPECT_NEAR(s2.get_atom_mass(0), 15.999, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2.get_atom_mass(1), 14.007, testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2.get_atom_nuclear_charge(0), 8.5,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2.get_atom_nuclear_charge(1), 7.5,
              testing::numerical_zero_tolerance);
}

TEST_F(StructureBasicTest, SymbolCapitalization) {
  std::vector<std::pair<std::string, std::string>> test_cases = {
      {"h", "H"}, {"HE", "He"}, {"li", "Li"}, {"CA", "Ca"}};

  for (const auto& test_case : test_cases) {
    std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
    std::vector<std::string> symbols = {test_case.first};
    Structure s(coords, symbols);
    EXPECT_EQ(s.get_atom_symbol(0), test_case.second)
        << "Failed for input: " << test_case.first;
  }
}

TEST_F(StructureBasicTest, PropertyAccess) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};
  Structure s(coords, elements);

  // Coordinates
  Eigen::MatrixXd retrieved_coords = s.get_coordinates();
  EXPECT_EQ(retrieved_coords.rows(), 2);
  EXPECT_EQ(retrieved_coords.cols(), 3);

  // Elements
  const std::vector<Element>& retrieved_elements = s.get_elements();
  EXPECT_EQ(retrieved_elements.size(), 2);
  EXPECT_EQ(retrieved_elements[0], Element::H);
  EXPECT_EQ(retrieved_elements[1], Element::C);

  // Nuclear charges and masses as Eigen vectors
  const Eigen::VectorXd& charges = s.get_nuclear_charges();
  EXPECT_EQ(charges.size(), 2);
  EXPECT_NEAR(charges(0), 1.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(charges(1), 6.0, testing::numerical_zero_tolerance);

  const Eigen::VectorXd& masses = s.get_masses();
  EXPECT_EQ(masses.size(), 2);
  EXPECT_GT(masses(0), 0.0);
  EXPECT_GT(masses(1), 0.0);
}

TEST_F(StructureBasicTest, ErrorHandling) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};
  Structure s(coords, elements);

  // Valid indices work
  EXPECT_NO_THROW(s.get_atom_coordinates(0));
  EXPECT_NO_THROW(s.get_atom_element(1));

  // Invalid indices throw
  EXPECT_THROW(s.get_atom_coordinates(2), std::out_of_range);
  EXPECT_THROW(s.get_atom_element(100), std::out_of_range);
  EXPECT_THROW(s.get_atom_mass(2), std::out_of_range);
  EXPECT_THROW(s.get_atom_nuclear_charge(100), std::out_of_range);
}

// Static Utility Functions
TEST_F(StructureBasicTest, StaticUtilityFunctions) {
  // Symbol <-> nuclear charge
  EXPECT_EQ(Structure::symbol_to_nuclear_charge("H"), 1u);
  EXPECT_EQ(Structure::symbol_to_nuclear_charge("C"), 6u);
  EXPECT_THROW(Structure::symbol_to_nuclear_charge("Xx"),
               std::invalid_argument);

  EXPECT_EQ(Structure::nuclear_charge_to_symbol(1u), "H");
  EXPECT_EQ(Structure::nuclear_charge_to_symbol(6u), "C");
  EXPECT_THROW(Structure::nuclear_charge_to_symbol(200u),
               std::invalid_argument);

  // Element conversions
  EXPECT_EQ(Structure::element_to_symbol(Element::H), "H");
  EXPECT_EQ(Structure::symbol_to_element("H"), Element::H);
  EXPECT_EQ(Structure::element_to_nuclear_charge(Element::C), 6u);
  EXPECT_EQ(Structure::nuclear_charge_to_element(1u), Element::H);
}

// Serialization Tests
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

TEST_F(StructureBasicTest, JSONSerialization) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};

  Structure s1(coords, elements);
  auto json_data = s1.to_json();

  auto s2 = Structure::from_json(json_data);
  EXPECT_EQ(s2->get_num_atoms(), 2);
  EXPECT_NEAR(s2->get_atom_nuclear_charge(0), 1.0,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2->get_atom_nuclear_charge(1), 6.0,
              testing::numerical_zero_tolerance);
}

TEST_F(StructureBasicTest, JSONSerializationWithCustomCharges) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H, Element::C};
  std::vector<double> custom_charges = {1.1, 6.6};

  Structure s1(coords, elements, std::vector<double>(), custom_charges);
  auto json_data = s1.to_json();

  auto s2 = Structure::from_json(json_data);
  EXPECT_EQ(s2->get_num_atoms(), 2);
  EXPECT_NEAR(s2->get_atom_nuclear_charge(0), 1.1,
              testing::numerical_zero_tolerance);
  EXPECT_NEAR(s2->get_atom_nuclear_charge(1), 6.6,
              testing::numerical_zero_tolerance);
}

TEST_F(StructureBasicTest, JSONDeserializationEdgeCases) {
  // Missing units
  nlohmann::json json_no_units = {
      {"coordinates", {{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}}},
      {"elements", {1, 6}}};
  EXPECT_THROW(Structure::from_json(json_no_units), std::runtime_error);

  // Invalid units
  nlohmann::json json_bad_units = {{"units", "invalid_unit"},
                                   {"coordinates", {{0.0, 1.0, 2.0}}},
                                   {"elements", {1}}};
  EXPECT_THROW(Structure::from_json(json_bad_units), std::runtime_error);

  // Fallback to nuclear_charges
  nlohmann::json json_nuclear_charges = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
      {"nuclear_charges", {1, 6}}};
  auto s1 = Structure::from_json(json_nuclear_charges);
  EXPECT_EQ(s1->get_atom_symbol(0), "H");
  EXPECT_EQ(s1->get_atom_symbol(1), "C");

  // Fallback to symbols
  nlohmann::json json_symbols = {
      {"units", "bohr"},
      {"coordinates", {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}}},
      {"symbols", {"H", "he"}}};
  auto s2 = Structure::from_json(json_symbols);
  EXPECT_EQ(s2->get_atom_symbol(1), "He");

  // Unit conversion
  nlohmann::json json_angstrom = {
      {"units", "angstrom"},
      {"coordinates",
       {{0.0, 0.0, 0.0},
        {qdk::chemistry::constants::bohr_to_angstrom, 0.0, 0.0}}},
      {"elements", {1, 1}}};
  auto s3 = Structure::from_json(json_angstrom);
  EXPECT_NEAR(s3->get_atom_coordinates(1)[0], 1.0,
              testing::numerical_zero_tolerance);
}

// File I/O Tests
TEST_F(StructureBasicTest, FileIO) {
  Eigen::MatrixXd coords(3, 3);
  coords << 0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0;
  std::vector<std::string> symbols = {"O", "H", "H"};
  Structure water(coords, symbols);

  // XYZ file I/O
  water.to_xyz_file("test_water.structure.xyz", "Water molecule test");
  auto from_xyz = Structure::from_xyz_file("test_water.structure.xyz");
  EXPECT_EQ(from_xyz->get_num_atoms(), 3);

  // JSON file I/O
  water.to_json_file("test_water.structure.json");
  auto from_json = Structure::from_json_file("test_water.structure.json");
  EXPECT_EQ(from_json->get_num_atoms(), 3);

  // Generic file I/O
  water.to_file("test.structure.json", "json");
  auto from_generic = Structure::from_file("test.structure.json", "json");
  EXPECT_EQ(from_generic->get_num_atoms(), 3);

  // Clean up
  std::filesystem::remove("test_water.structure.xyz");
  std::filesystem::remove("test_water.structure.json");
}

TEST_F(StructureBasicTest, FilenameValidation) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H};
  Structure s(coords, elements);

  // Valid filenames
  EXPECT_NO_THROW(s.to_json_file("valid.structure.json"));
  EXPECT_NO_THROW(s.to_xyz_file("valid.structure.xyz"));

  // Invalid filenames
  EXPECT_THROW(s.to_json_file("invalid.json"), std::invalid_argument);
  EXPECT_THROW(s.to_xyz_file("invalid.xyz"), std::invalid_argument);

  std::filesystem::remove("valid.structure.json");
  std::filesystem::remove("valid.structure.xyz");
}

TEST_F(StructureBasicTest, FileIOErrorHandling) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<Element> elements = {Element::H};
  Structure s(coords, elements);

  // Invalid paths
  EXPECT_THROW(s.to_json_file("/nonexistent_directory/test.structure.json"),
               std::runtime_error);
  EXPECT_THROW(Structure::from_json_file("nonexistent.structure.json"),
               std::runtime_error);

  // XYZ parsing errors
  EXPECT_THROW(Structure::from_xyz(""), std::runtime_error);
  EXPECT_THROW(Structure::from_xyz("not_a_number\n"), std::runtime_error);
  EXPECT_THROW(Structure::from_xyz("2\nComment\nH 0.0 0.0 0.0\n"),
               std::runtime_error);
  EXPECT_THROW(Structure::from_xyz("1\nComment\nH invalid 0.0 0.0\n"),
               std::runtime_error);
}

// HDF5 Serialization Tests
TEST_F(StructureBasicTest, HDF5GroupNesting) {
  std::vector<std::string> symbols1 = {"H", "H"};
  std::vector<Eigen::Vector3d> coords1 = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  Structure h2(coords1, symbols1);

  std::vector<std::string> symbols2 = {"O", "H", "H"};
  std::vector<Eigen::Vector3d> coords2 = {
      {0.0, 0.0, 0.0}, {0.757, 0.586, 0.0}, {-0.757, 0.586, 0.0}};
  Structure h2o(coords2, symbols2);

  try {
    H5::H5File file("test_nested.h5", H5F_ACC_TRUNC);
    H5::Group molecules_group = file.createGroup("/molecules");
    H5::Group reactants_group = molecules_group.createGroup("reactants");
    H5::Group products_group = molecules_group.createGroup("products");
    H5::Group h2_group = reactants_group.createGroup("hydrogen");
    H5::Group h2o_group = products_group.createGroup("water");

    h2.to_hdf5(h2_group);
    h2o.to_hdf5(h2o_group);
    file.close();

    // Read back
    H5::H5File read_file("test_nested.h5", H5F_ACC_RDONLY);
    H5::Group h2_read_group =
        read_file.openGroup("/molecules/reactants/hydrogen");
    H5::Group water_read_group =
        read_file.openGroup("/molecules/products/water");

    auto loaded_h2 = Structure::from_hdf5(h2_read_group);
    auto loaded_h2o = Structure::from_hdf5(water_read_group);

    EXPECT_EQ(loaded_h2->get_num_atoms(), 2);
    EXPECT_EQ(loaded_h2o->get_num_atoms(), 3);
  } catch (const H5::Exception& e) {
    FAIL() << "HDF5 exception: " << e.getCDetailMsg();
  }
}

TEST_F(StructureBasicTest, HDF5WithMetadata) {
  std::vector<std::string> symbols = {"C", "O", "O"};
  std::vector<Eigen::Vector3d> coords = {
      {0.0, 0.0, 0.0}, {1.16, 0.0, 0.0}, {-1.16, 0.0, 0.0}};
  Structure co2(coords, symbols);

  try {
    H5::H5File file("test_with_metadata.h5", H5F_ACC_TRUNC);
    H5::Group calc_group = file.createGroup("/calculation");

    // Add metadata
    H5::DataSpace attr_space(H5S_SCALAR);
    H5::StrType str_type(H5::PredType::C_S1, 256);
    H5::Attribute name_attr =
        calc_group.createAttribute("molecule_name", str_type, attr_space);
    std::string mol_name = "carbon_dioxide";
    name_attr.write(str_type, mol_name);

    H5::Group structure_group = calc_group.createGroup("structure");
    co2.to_hdf5(structure_group);
    file.close();

    // Read back
    H5::H5File read_file("test_with_metadata.h5", H5F_ACC_RDONLY);
    H5::Group read_calc = read_file.openGroup("/calculation");

    H5::Attribute read_name_attr = read_calc.openAttribute("molecule_name");
    std::string read_name;
    read_name_attr.read(str_type, read_name);
    EXPECT_EQ(read_name, "carbon_dioxide");

    H5::Group read_structure = read_calc.openGroup("structure");
    auto loaded_co2 = Structure::from_hdf5(read_structure);
    EXPECT_EQ(loaded_co2->get_num_atoms(), 3);
  } catch (const H5::Exception& e) {
    FAIL() << "HDF5 exception: " << e.getCDetailMsg();
  }
}

// Energy tests
TEST_F(StructureBasicTest, NuclearRepulsionEnergy) {
  // Empty structure
  Structure empty(std::vector<Eigen::Vector3d>{}, std::vector<std::string>{});
  EXPECT_DOUBLE_EQ(empty.calculate_nuclear_repulsion_energy(), 0.0);

  // Single atom
  std::vector<Eigen::Vector3d> h_coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> h_symbol = {"H"};
  Structure single(h_coords, h_symbol);
  EXPECT_DOUBLE_EQ(single.calculate_nuclear_repulsion_energy(), 0.0);

  // H2 molecule
  std::vector<Eigen::Vector3d> h2_coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H", "H"};
  Structure h2(h2_coords, symbols);
  EXPECT_NEAR(h2.calculate_nuclear_repulsion_energy(), 1.0 / 0.74,
              testing::numerical_zero_tolerance);

  // Custom charges
  std::vector<double> custom_charges = {1.5, 2.5};
  Structure custom({{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}, {Element::H, Element::H},
                   {}, custom_charges);
  EXPECT_NEAR(custom.calculate_nuclear_repulsion_energy(), 3.75,
              testing::numerical_zero_tolerance);
}

TEST_F(StructureBasicTest, Summary) {
  Eigen::MatrixXd coords(3, 3);
  coords << 0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0;
  std::vector<std::string> symbols = {"O", "H", "H"};
  Structure water(coords, symbols);

  std::string summary = water.get_summary();
  EXPECT_FALSE(summary.empty());
  EXPECT_NE(summary.find("Number of atoms: 3"), std::string::npos);
  EXPECT_NE(summary.find("O"), std::string::npos);
  EXPECT_NE(summary.find("H"), std::string::npos);
}

TEST_F(StructureBasicTest, DimensionValidation) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.74}};
  std::vector<std::string> symbols = {"H"};  // Mismatched size

  EXPECT_THROW(Structure(coords, symbols), std::invalid_argument);
}

TEST_F(StructureBasicTest, UnknownSymbol) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"X"};  // Invalid symbol

  EXPECT_THROW(Structure(coords, symbols), std::invalid_argument);
}

TEST_F(StructureBasicTest, CoordinateMatrixOperations) {
  Eigen::MatrixXd new_coords(2, 3);
  new_coords << 0.0, 0.0, 0.0, 0.0, 0.0, 0.74;
  std::vector<std::string> symbols = {"H", "H"};

  Structure s(new_coords, symbols);

  Eigen::MatrixXd retrieved_coords = s.get_coordinates();
  EXPECT_EQ(retrieved_coords.rows(), 2);
  EXPECT_EQ(retrieved_coords.cols(), 3);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(retrieved_coords(i, j), new_coords(i, j),
                  testing::numerical_zero_tolerance);
    }
  }
}

TEST_F(StructureBasicTest, FileIOConsistency) {
  Eigen::MatrixXd coords(3, 3);
  coords << 0.0, 0.0, 0.0, 0.757, 0.586, 0.0, -0.757, 0.586, 0.0;
  std::vector<std::string> symbols = {"O", "H", "H"};
  Structure water(coords, symbols);

  // Test that generic methods produce same results as specific methods
  water.to_json_file("specific.structure.json");
  water.to_file("generic.structure.json", "json");

  auto s_specific = Structure::from_json_file("specific.structure.json");
  auto s_generic = Structure::from_file("generic.structure.json", "json");

  EXPECT_EQ(s_specific->get_num_atoms(), s_generic->get_num_atoms());
  for (size_t i = 0; i < s_specific->get_num_atoms(); ++i) {
    EXPECT_EQ(s_specific->get_atom_symbol(i), s_generic->get_atom_symbol(i));

    Eigen::Vector3d coords_specific = s_specific->get_atom_coordinates(i);
    Eigen::Vector3d coords_generic = s_generic->get_atom_coordinates(i);
    EXPECT_NEAR((coords_specific - coords_generic).norm(), 0.0,
                testing::json_tolerance);
  }

  std::filesystem::remove("specific.structure.json");
  std::filesystem::remove("generic.structure.json");

  // Test XYZ consistency
  water.to_xyz_file("specific.structure.xyz", "Water");
  water.to_file("generic.structure.xyz", "xyz");

  auto s_xyz_specific = Structure::from_xyz_file("specific.structure.xyz");
  auto s_xyz_generic = Structure::from_file("generic.structure.xyz", "xyz");

  EXPECT_EQ(s_xyz_specific->get_num_atoms(), s_xyz_generic->get_num_atoms());

  std::filesystem::remove("specific.structure.xyz");
  std::filesystem::remove("generic.structure.xyz");
}

TEST_F(StructureBasicTest, InvalidFileFormat) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure s(coords, symbols);

  EXPECT_THROW(s.to_file("test.structure.xyz", "invalid_format"),
               std::invalid_argument);
  EXPECT_THROW(Structure::from_file("test.structure.xyz", "invalid_format"),
               std::invalid_argument);
  EXPECT_THROW(Structure::from_file("non_existent.structure.json", "json"),
               std::runtime_error);
}

TEST_F(StructureBasicTest, FilenameValidationConsistency) {
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}};
  std::vector<std::string> symbols = {"H"};
  Structure s(coords, symbols);

  std::vector<std::string> invalid_filenames = {
      "test.json", "test.xyz", "test.structure", "structure.json",
      "structure.xyz"};

  for (const auto& filename : invalid_filenames) {
    EXPECT_THROW(s.to_json_file(filename), std::invalid_argument);
    EXPECT_THROW(Structure::from_json_file(filename), std::invalid_argument);
    EXPECT_THROW(s.to_xyz_file(filename), std::invalid_argument);
    EXPECT_THROW(Structure::from_xyz_file(filename), std::invalid_argument);
  }
}
