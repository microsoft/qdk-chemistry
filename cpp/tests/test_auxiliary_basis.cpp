// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <qdk/chemistry/data/auxiliary_basis.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qdk::chemistry::data;

class AuxiliaryBasisTest : public ::testing::Test {
 protected:
  static constexpr const char* json_filename = "test.auxiliary_basis.json";
  static constexpr const char* hdf5_filename = "test.auxiliary_basis.h5";
  static constexpr const char* collection_json_filename =
      "test.auxiliary_basis_collection.json";
  static constexpr const char* collection_hdf5_filename =
      "test.auxiliary_basis_collection.h5";

  void SetUp() override { remove_test_files(); }
  void TearDown() override { remove_test_files(); }

  static void remove_test_files() {
    std::error_code error;
    std::filesystem::remove(json_filename, error);
    std::filesystem::remove(hdf5_filename, error);
    std::filesystem::remove(collection_json_filename, error);
    std::filesystem::remove(collection_hdf5_filename, error);
  }

  static std::shared_ptr<Structure> make_structure(
      const std::vector<std::string>& symbols = {"H", "H", "H"}) {
    std::vector<Eigen::Vector3d> coordinates;
    coordinates.reserve(symbols.size());
    for (size_t atom_index = 0; atom_index < symbols.size(); ++atom_index) {
      coordinates.emplace_back(static_cast<double>(atom_index), 0.0, 0.0);
    }
    return std::make_shared<Structure>(coordinates, symbols);
  }

  static std::vector<Shell> make_shells() {
    std::vector<Shell> shells;
    shells.emplace_back(2, OrbitalType::P, std::vector{1.0}, std::vector{0.7});
    shells.emplace_back(0, OrbitalType::D, std::vector{3.0}, std::vector{0.8});
    shells.emplace_back(2, OrbitalType::S, std::vector{2.0}, std::vector{0.9});
    shells.emplace_back(0, OrbitalType::S, std::vector{0.5, 4.0},
                        std::vector{0.4, 0.6});
    return shells;
  }
};

TEST_F(AuxiliaryBasisTest, RoleNamesAndStrictParsing) {
  EXPECT_EQ("jfit", to_string(AuxiliaryBasisRole::JFit));
  EXPECT_EQ("jkfit", to_string(AuxiliaryBasisRole::JKFit));
  EXPECT_EQ("rifit", to_string(AuxiliaryBasisRole::RIFit));
  EXPECT_EQ("cabs", to_string(AuxiliaryBasisRole::CABS));

  EXPECT_EQ(AuxiliaryBasisRole::JFit, auxiliary_basis_role_from_string("jfit"));
  EXPECT_EQ(AuxiliaryBasisRole::JKFit,
            auxiliary_basis_role_from_string("jkfit"));
  EXPECT_EQ(AuxiliaryBasisRole::RIFit,
            auxiliary_basis_role_from_string("rifit"));
  EXPECT_EQ(AuxiliaryBasisRole::CABS, auxiliary_basis_role_from_string("cabs"));
  EXPECT_THROW(auxiliary_basis_role_from_string("MP2FIT"),
               std::invalid_argument);
  EXPECT_THROW(auxiliary_basis_role_from_string("OptRI+"),
               std::invalid_argument);
  EXPECT_THROW(auxiliary_basis_role_from_string("unknown"),
               std::invalid_argument);
}

TEST_F(AuxiliaryBasisTest, CustomAndNamedConstruction) {
  auto structure = make_structure();
  AuxiliaryBasis custom(make_shells(), structure);
  AuxiliaryBasis named("density-fit", make_shells(), structure);

  EXPECT_EQ(std::string(AuxiliaryBasis::custom_name), custom.get_name());
  EXPECT_EQ("density-fit", named.get_name());
  EXPECT_EQ(AOType::Spherical, named.get_atomic_orbital_type());
  EXPECT_EQ(structure, named.get_structure());
  EXPECT_EQ(3u, named.get_num_atoms());
  EXPECT_EQ(4u, named.get_num_shells());
}

TEST_F(AuxiliaryBasisTest, CanonicalOrderingAndEmptyAtomLookup) {
  AuxiliaryBasis basis("ordered", make_shells(), make_structure());
  const auto shells = basis.get_shells();

  ASSERT_EQ(4u, shells.size());
  EXPECT_EQ(0u, shells[0].atom_index);
  EXPECT_EQ(OrbitalType::S, shells[0].orbital_type);
  EXPECT_DOUBLE_EQ(4.0, shells[0].exponents[0]);
  EXPECT_EQ(OrbitalType::D, shells[1].orbital_type);
  EXPECT_EQ(2u, shells[2].atom_index);
  EXPECT_EQ(OrbitalType::S, shells[2].orbital_type);
  EXPECT_EQ(OrbitalType::P, shells[3].orbital_type);
  EXPECT_TRUE(basis.get_shells_for_atom(1).empty());
}

TEST_F(AuxiliaryBasisTest, SphericalAndCartesianOrbitalCounts) {
  auto structure = make_structure();
  AuxiliaryBasis spherical("spherical", make_shells(), structure,
                           AOType::Spherical);
  AuxiliaryBasis cartesian("cartesian", make_shells(), structure,
                           AOType::Cartesian);

  EXPECT_EQ(10u, spherical.get_num_auxiliary_orbitals());
  EXPECT_EQ(11u, cartesian.get_num_auxiliary_orbitals());
  EXPECT_EQ(AOType::Cartesian, cartesian.get_atomic_orbital_type());
}

TEST_F(AuxiliaryBasisTest, FactoriesLoadByNameElementAndIndex) {
  auto structure = make_structure({"O", "H", "H"});
  const std::string basis_name = "def2-universal-jfit";

  auto by_name = AuxiliaryBasis::from_basis_name(basis_name, structure);
  auto by_element = AuxiliaryBasis::from_element_map(
      {{"H", basis_name}, {"O", basis_name}}, structure);
  auto by_index = AuxiliaryBasis::from_index_map(
      {{0, basis_name}, {1, basis_name}, {2, basis_name}}, structure);

  EXPECT_EQ(basis_name, by_name->get_name());
  EXPECT_EQ(std::string(AuxiliaryBasis::custom_name), by_element->get_name());
  EXPECT_EQ(std::string(AuxiliaryBasis::custom_name), by_index->get_name());
  EXPECT_EQ(3u, by_name->get_num_atoms());
  EXPECT_GT(by_name->get_num_shells(), 0u);
  EXPECT_GT(by_element->get_num_shells(), 0u);
  EXPECT_GT(by_index->get_num_shells(), 0u);
}

TEST_F(AuxiliaryBasisTest, ConstructorValidation) {
  auto structure = make_structure({"H"});
  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector{1.0}, std::vector{1.0});

  EXPECT_THROW(AuxiliaryBasis(shells, nullptr), std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis("", shells, structure), std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis({}, structure), std::invalid_argument);

  std::vector<Shell> radial_shells;
  radial_shells.emplace_back(0, OrbitalType::S, std::vector{2.0},
                             std::vector{1.0}, std::vector{0});
  EXPECT_THROW(AuxiliaryBasis(radial_shells, structure), std::invalid_argument);

  std::vector<Shell> local_potential_shells;
  local_potential_shells.emplace_back(0, OrbitalType::UL, std::vector{2.0},
                                      std::vector{1.0});
  EXPECT_THROW(AuxiliaryBasis(local_potential_shells, structure),
               std::invalid_argument);

  shells[0].atom_index = 1;
  EXPECT_THROW(AuxiliaryBasis(shells, structure), std::invalid_argument);
}

TEST_F(AuxiliaryBasisTest, FactoryValidationRejectsBadMaps) {
  auto structure = make_structure({"O", "H", "H"});
  const std::string basis_name = "def2-universal-jfit";

  EXPECT_THROW(AuxiliaryBasis::from_element_map({{"H", basis_name}}, structure),
               std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis::from_index_map(
                   {{0, basis_name}, {1, basis_name}}, structure),
               std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis::from_element_map(
                   {{"H", basis_name}, {"O", "invalid-basis-set"}}, structure),
               std::invalid_argument);

  auto partially_supported = make_structure({"H", "F"});
  EXPECT_THROW(AuxiliaryBasis::from_basis_name("6-31g-j", partially_supported),
               std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis::from_element_map(
                   {{"H", "6-31g-j"}, {"F", "6-31g-j"}}, partially_supported),
               std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis::from_index_map({{0, "6-31g-j"}, {1, "6-31g-j"}},
                                              partially_supported),
               std::invalid_argument);
}

TEST_F(AuxiliaryBasisTest, ConstructorRejectsNonfiniteShellData) {
  auto structure = make_structure({"H"});
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  EXPECT_THROW(AuxiliaryBasis({Shell(0, OrbitalType::S, std::vector{nan},
                                     std::vector{1.0})},
                              structure),
               std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis({Shell(0, OrbitalType::S, std::vector{infinity},
                                     std::vector{1.0})},
                              structure),
               std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis({Shell(0, OrbitalType::S, std::vector{1.0},
                                     std::vector{nan})},
                              structure),
               std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasis({Shell(0, OrbitalType::S, std::vector{0.0},
                                     std::vector{1.0})},
                              structure),
               std::invalid_argument);
}

TEST_F(AuxiliaryBasisTest, ContentHashTracksNameShellsAndAOType) {
  auto structure = make_structure();
  auto shells = make_shells();
  AuxiliaryBasis baseline("aux", shells, structure);
  AuxiliaryBasis same("aux", shells, structure);
  AuxiliaryBasis different_name("other", shells, structure);
  AuxiliaryBasis different_type("aux", shells, structure, AOType::Cartesian);

  shells[0].coefficients[0] = 0.5;
  AuxiliaryBasis different_shell("aux", shells, structure);

  EXPECT_EQ(baseline.content_hash(), same.content_hash());
  EXPECT_NE(baseline.content_hash(), different_name.content_hash());
  EXPECT_NE(baseline.content_hash(), different_type.content_hash());
  EXPECT_NE(baseline.content_hash(), different_shell.content_hash());
}

TEST_F(AuxiliaryBasisTest, SummaryAndDataType) {
  AuxiliaryBasis basis("summary-aux", make_shells(), make_structure());

  EXPECT_EQ("auxiliary_basis", basis.get_data_type_name());
  EXPECT_NE(std::string::npos,
            basis.get_summary().find("AuxiliaryBasis: summary-aux"));
  EXPECT_NE(std::string::npos,
            basis.get_summary().find("Number of auxiliary orbitals: 10"));
}

TEST_F(AuxiliaryBasisTest, JSONRoundTripsInMemoryAndFile) {
  AuxiliaryBasis basis("json-aux", make_shells(), make_structure(),
                       AOType::Cartesian);

  auto from_memory = AuxiliaryBasis::from_json(basis.to_json());
  EXPECT_EQ(basis.content_hash(), from_memory->content_hash());
  EXPECT_EQ(AOType::Cartesian, from_memory->get_atomic_orbital_type());

  basis.to_json_file(json_filename);
  auto from_file = AuxiliaryBasis::from_json_file(json_filename);
  EXPECT_EQ(basis.content_hash(), from_file->content_hash());
  EXPECT_EQ(3u, from_file->get_num_atoms());
}

TEST_F(AuxiliaryBasisTest, HDF5FileRoundTrip) {
  AuxiliaryBasis basis("hdf5-aux", make_shells(), make_structure());

  basis.to_hdf5_file(hdf5_filename);
  auto loaded = AuxiliaryBasis::from_hdf5_file(hdf5_filename);

  EXPECT_EQ(basis.content_hash(), loaded->content_hash());
  EXPECT_EQ("hdf5-aux", loaded->get_name());
  EXPECT_EQ(4u, loaded->get_num_shells());
}

TEST_F(AuxiliaryBasisTest, ImmutableCollectionLookupAndRoundTrips) {
  auto structure = make_structure();
  auto jfit =
      std::make_shared<AuxiliaryBasis>("jfit-basis", make_shells(), structure);
  auto jkfit =
      std::make_shared<AuxiliaryBasis>("jkfit-basis", make_shells(), structure);
  AuxiliaryBasisCollection empty;

  auto jk_collection =
      with_auxiliary_basis(empty, AuxiliaryBasisRole::JKFit, jkfit);
  EXPECT_FALSE(empty.has_auxiliary_basis(AuxiliaryBasisRole::JKFit));
  EXPECT_EQ(jkfit,
            jk_collection->resolve_auxiliary_basis(AuxiliaryBasisRole::JFit));

  auto j_collection =
      with_auxiliary_basis(empty, AuxiliaryBasisRole::JFit, jfit);
  EXPECT_THROW(j_collection->resolve_auxiliary_basis(AuxiliaryBasisRole::JKFit),
               std::out_of_range);

  auto collection =
      with_auxiliary_basis(*jk_collection, AuxiliaryBasisRole::JFit, jfit);
  EXPECT_EQ(2u, collection->get_auxiliary_bases().size());
  EXPECT_EQ(jfit, collection->get_auxiliary_basis(AuxiliaryBasisRole::JFit));
  EXPECT_EQ(jfit,
            collection->resolve_auxiliary_basis(AuxiliaryBasisRole::JFit));
  EXPECT_THROW(collection->resolve_auxiliary_basis(AuxiliaryBasisRole::RIFit),
               std::out_of_range);
  EXPECT_NE(empty.content_hash(), collection->content_hash());

  auto replacement = std::make_shared<AuxiliaryBasis>("replacement-jfit",
                                                      make_shells(), structure);
  auto replaced =
      with_auxiliary_basis(*collection, AuxiliaryBasisRole::JFit, replacement);
  EXPECT_EQ(jfit, collection->get_auxiliary_basis(AuxiliaryBasisRole::JFit));
  EXPECT_EQ(replacement,
            replaced->get_auxiliary_basis(AuxiliaryBasisRole::JFit));

  const auto json = collection->to_json();
  EXPECT_EQ("0.1.0", json.at("version"));
  EXPECT_TRUE(json.at("auxiliary_bases").contains("jfit"));
  EXPECT_TRUE(json.at("auxiliary_bases").contains("jkfit"));
  auto from_json = AuxiliaryBasisCollection::from_json(json);
  EXPECT_EQ(collection->content_hash(), from_json->content_hash());

  collection->to_json_file(collection_json_filename);
  auto from_json_file =
      AuxiliaryBasisCollection::from_json_file(collection_json_filename);
  EXPECT_EQ(collection->content_hash(), from_json_file->content_hash());

  collection->to_hdf5_file(collection_hdf5_filename);
  auto from_hdf5 =
      AuxiliaryBasisCollection::from_hdf5_file(collection_hdf5_filename);
  EXPECT_EQ(collection->content_hash(), from_hdf5->content_hash());
  EXPECT_EQ(
      "jfit-basis",
      from_hdf5->get_auxiliary_basis(AuxiliaryBasisRole::JFit)->get_name());
}

TEST_F(AuxiliaryBasisTest, CollectionRejectsInvalidEntries) {
  auto structure = make_structure();
  auto matching =
      std::make_shared<AuxiliaryBasis>("matching", make_shells(), structure);
  auto different_structure = make_structure({"H", "He", "H"});
  auto mismatched = std::make_shared<AuxiliaryBasis>(
      "mismatched", make_shells(), different_structure);

  EXPECT_THROW(
      AuxiliaryBasisCollection({{AuxiliaryBasisRole::JFit, matching},
                                {AuxiliaryBasisRole::RIFit, mismatched}}),
      std::invalid_argument);
  EXPECT_THROW(AuxiliaryBasisCollection({{AuxiliaryBasisRole::JFit, nullptr}}),
               std::invalid_argument);

  AuxiliaryBasisCollection collection({{AuxiliaryBasisRole::JFit, matching}});
  EXPECT_THROW(
      with_auxiliary_basis(collection, AuxiliaryBasisRole::RIFit, mismatched),
      std::invalid_argument);
}

TEST_F(AuxiliaryBasisTest, CollectionIsIndependentOfPrimaryBasis) {
  auto structure = make_structure();
  BasisSet primary("primary", make_shells(), structure);
  const auto primary_hash = primary.content_hash();
  auto auxiliary =
      std::make_shared<AuxiliaryBasis>("jfit", make_shells(), structure);

  AuxiliaryBasisCollection collection({{AuxiliaryBasisRole::JFit, auxiliary}});

  EXPECT_EQ(primary_hash, primary.content_hash());
  EXPECT_FALSE(primary.to_json().contains("auxiliary_bases"));
  EXPECT_EQ(auxiliary,
            collection.get_auxiliary_basis(AuxiliaryBasisRole::JFit));
}

TEST_F(AuxiliaryBasisTest, ShellAccessAndErrorPaths) {
  AuxiliaryBasis basis("access", make_shells(), make_structure());

  EXPECT_EQ(OrbitalType::S, basis.get_shell(0).orbital_type);
  ASSERT_EQ(2u, basis.get_shells_for_atom(0).size());
  EXPECT_THROW(basis.get_shell(basis.get_num_shells()), std::out_of_range);
  EXPECT_THROW(basis.get_shells_for_atom(basis.get_num_atoms()),
               std::out_of_range);
}
