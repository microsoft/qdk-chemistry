// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <bitset>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/configuration_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class ConfigurationTest : public ::testing::Test {
 protected:
  // Empty test fixture - we'll create configurations directly in each test
};

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

TEST_F(ConfigurationTest, DeprecatedConstructorsForwardToFactories) {
  Configuration from_string("2du0");
  std::bitset<8> occupations("01010011");
  Configuration from_bitset(occupations, 4);

  EXPECT_EQ(from_string, Configuration::from_spin_half_string("2du0"));
  EXPECT_EQ(from_bitset, Configuration::from_spin_half_bitset(occupations, 4));
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

// Test construction from string
TEST_F(ConfigurationTest, ConstructFromString) {
  // Test that construction works with valid strings
  EXPECT_NO_THROW(Configuration::from_spin_half_string("0"));
  EXPECT_NO_THROW(Configuration::from_spin_half_string("u"));
  EXPECT_NO_THROW(Configuration::from_spin_half_string("d"));
  EXPECT_NO_THROW(Configuration::from_spin_half_string("2"));
  EXPECT_NO_THROW(Configuration::from_spin_half_string("2du0"));
  EXPECT_NO_THROW(Configuration::from_spin_half_string("2duu0d20"));

  // Test that invalid characters throw an exception
  EXPECT_THROW(Configuration::from_spin_half_string("4"),
               std::invalid_argument);
  EXPECT_THROW(Configuration::from_spin_half_string("A"),
               std::invalid_argument);
  EXPECT_THROW(Configuration::from_spin_half_string("2dux"),
               std::invalid_argument);
}

// Test conversion to string
TEST_F(ConfigurationTest, ToStringConversion) {
  auto basic_config = Configuration::from_spin_half_string("2du0");
  auto complex_config = Configuration::from_spin_half_string("22dduudd00");

  EXPECT_EQ(basic_config.to_string(), "2du0");
  EXPECT_EQ(complex_config.to_string().substr(0, 10), "22dduudd00");
}

// Test conversion to alpha, beta strings
TEST_F(ConfigurationTest, ToBinaryStrings) {
  auto basic_config = Configuration::from_spin_half_string("2du0");

  auto [alpha_string, beta_string] = basic_config.to_binary_strings(4);

  EXPECT_EQ(alpha_string, "1010");
  EXPECT_EQ(beta_string, "1100");

  // Active space - check only the first two orbitals
  auto [alpha_string_red, beta_string_red] = basic_config.to_binary_strings(2);

  EXPECT_EQ(alpha_string_red, "10");
  EXPECT_EQ(beta_string_red, "11");

  // Throw if we ask for too many orbitals
  EXPECT_THROW(basic_config.to_binary_strings(5), std::runtime_error);
}

// Test conversion from alpha, beta binary strings
TEST_F(ConfigurationTest, FromBinaryStrings) {
  std::string alpha_string = "1010";
  std::string beta_string = "1100";

  Configuration basic_config =
      Configuration::from_binary_strings(alpha_string, beta_string);

  // should be 2du0
  EXPECT_EQ(basic_config.to_string(), "2du0");
}

// Test construction from bitset and conversion to bitset
TEST_F(ConfigurationTest, BitsetConversion) {
  // Test with a 8-bit bitset (4 spatial orbitals)
  // Bits: 0101|0011 (beta|alpha) little-endian
  std::bitset<8> test_bitset("01010011");
  auto config_from_bitset =
      Configuration::from_spin_half_bitset(test_bitset, 4);

  EXPECT_EQ(config_from_bitset.to_string().substr(0, 4), "2ud0");

  auto result_bitset = config_from_bitset.to_bitset<8>();
  // bitsets have different sizes, so correct size is ensured at compile time
  EXPECT_EQ(result_bitset, test_bitset);
}

TEST_F(ConfigurationTest, ElectronCounting) {
  auto basic_config = Configuration::from_spin_half_string("2du0");
  auto [alpha_basic, beta_basic] = basic_config.get_n_electrons();
  EXPECT_EQ(alpha_basic, 2);
  EXPECT_EQ(beta_basic, 2);

  auto complex_config = Configuration::from_spin_half_string("22dduudd00");
  auto [alpha_complex, beta_complex] = complex_config.get_n_electrons();
  EXPECT_EQ(alpha_complex, 4);
  EXPECT_EQ(beta_complex, 6);
}

TEST_F(ConfigurationTest, EqualityComparison) {
  auto basic_config = Configuration::from_spin_half_string("2du0");
  auto same_as_basic = Configuration::from_spin_half_string("2du0");
  auto different_from_basic = Configuration::from_spin_half_string("d2u0");
  auto complex_config = Configuration::from_spin_half_string("22dduudd00");

  // Same configuration should be equal
  EXPECT_EQ(basic_config, same_as_basic);
  EXPECT_TRUE(basic_config == same_as_basic);

  // Different configurations should not be equal
  EXPECT_NE(basic_config, different_from_basic);
  EXPECT_FALSE(basic_config == different_from_basic);

  // Different length configurations should not be equal
  EXPECT_NE(basic_config, complex_config);
  EXPECT_FALSE(basic_config == complex_config);
}

TEST_F(ConfigurationTest, InequalityComparison) {
  auto basic_config = Configuration::from_spin_half_string("2du0");
  auto same_as_basic = Configuration::from_spin_half_string("2du0");
  auto different_from_basic = Configuration::from_spin_half_string("d2u0");
  auto complex_config = Configuration::from_spin_half_string("22dduudd00");

  // Same configuration should not be unequal
  EXPECT_FALSE(basic_config != same_as_basic);

  // Different configurations should be unequal
  EXPECT_TRUE(basic_config != different_from_basic);

  // Different length configurations should be unequal
  EXPECT_TRUE(basic_config != complex_config);
}

TEST_F(ConfigurationTest, EdgeCases) {
  // Test a very long string to ensure proper resizing of internal storage
  std::string long_string(100, '2');  // 100 doubly occupied orbitals
  auto long_config = Configuration::from_spin_half_string(long_string);
  auto [alpha_long, beta_long] = long_config.get_n_electrons();
  EXPECT_EQ(alpha_long, 100);
  EXPECT_EQ(beta_long, 100);
}

TEST_F(ConfigurationTest, BitPacking) {
  // Create a configuration with enough orbitals to span multiple bytes
  auto config = Configuration::from_spin_half_string("22dduuud00222dddduu");

  // Convert to string and check it's preserved
  EXPECT_EQ(config.to_string().substr(0, 19), "22dduuud00222dddduu");

  // Check electron count
  auto [alpha, beta] = config.get_n_electrons();
  EXPECT_EQ(alpha, 10);  // 5 "2"s and 5 "u"s
  EXPECT_EQ(beta, 12);   // 5 "2"s and 7 "d"s
}

TEST_F(ConfigurationTest, DifferentLengthString) {
  auto conf = Configuration::from_spin_half_string("u2d00d");
  auto [alpha1, beta1] = conf.get_n_electrons();
  EXPECT_EQ(alpha1, 2);
  EXPECT_EQ(beta1, 3);

  EXPECT_EQ(conf.to_string().substr(0, 6), "u2d00d");
}

TEST_F(ConfigurationTest, DefaultConstructor) {
  Configuration default_config;

  // Default constructor should create a 0-orbital configuration
  EXPECT_EQ(default_config.to_string().length(), 0);
  EXPECT_EQ(default_config.to_string(), "");

  auto [alpha, beta] = default_config.get_n_electrons();
  EXPECT_EQ(alpha, 0);
  EXPECT_EQ(beta, 0);
}

TEST_F(ConfigurationTest, GetNumOrbitals) {
  auto config1 = Configuration::from_spin_half_string("2du0");
  auto config2 = Configuration::from_spin_half_string("22dduudd00");

  EXPECT_GE(config1.to_string().length(), 4);
  EXPECT_GE(config2.to_string().length(), 10);
}

TEST_F(ConfigurationTest, BitsetConstructorWithDifferentSizes) {
  // Test with 8-bit bitset but only use 2 orbitals
  std::bitset<8> test_bitset("10010100");  // beta: 1001, alpha: 0100
  auto config = Configuration::from_spin_half_bitset(test_bitset, 2);

  EXPECT_GE(config.to_string().length(), 2);
  EXPECT_EQ(config.to_string().substr(0, 2), "d0");

  auto [alpha, beta] = config.get_n_electrons();
  EXPECT_EQ(alpha, 0);
  EXPECT_EQ(beta, 1);
}

// Test JSON serialization
TEST_F(ConfigurationTest, JsonSerialization) {
  auto original = Configuration::from_spin_half_string("22uud0");

  // Serialize to JSON
  nlohmann::json j = original.to_json();

  // Deserialize from JSON
  Configuration restored = Configuration::from_json(j);

  // Verify they are equal
  EXPECT_EQ(original, restored);
}

// Test HDF5 serialization
TEST_F(ConfigurationTest, Hdf5Serialization) {
  auto original = Configuration::from_spin_half_string("22uud0");

  // Create temporary HDF5 file
  std::string filename = "test_config_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    // Serialize to HDF5
    original.to_hdf5(root);

    // Deserialize from HDF5
    Configuration restored = Configuration::from_hdf5(root);

    // Verify they are equal
    EXPECT_EQ(original, restored);
    EXPECT_EQ(original.to_string(), restored.to_string());

    file.close();
  }

  // Clean up
  std::remove(filename.c_str());
}

// Test fixture for ConfigurationSet validation
class ConfigurationSetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create orbitals with active space defined
    // Total: 10 orbitals (3 inactive, 4 active, 3 virtual)
    size_t num_molecular_orbitals = 10;

    // Define active space: orbitals 3, 4, 5, 6 (0-indexed)
    std::vector<size_t> active_indices = {3, 4, 5, 6};

    // Define inactive space: orbitals 0, 1, 2
    std::vector<size_t> inactive_indices = {0, 1, 2};

    // Create base orbitals and then add active space
    auto base_orbitals_with_active = testing::create_test_orbitals(
        num_molecular_orbitals, num_molecular_orbitals, true);
    orbitals_with_active = testing::with_active_space(
        base_orbitals_with_active, active_indices, inactive_indices);

    // Create orbitals without active space
    orbitals_without_active = testing::create_test_orbitals(
        num_molecular_orbitals, num_molecular_orbitals, true);
  }

  std::shared_ptr<Orbitals> orbitals_with_active;
  std::shared_ptr<Orbitals> orbitals_without_active;
};

TEST_F(ConfigurationSetTest, ValidActiveSpaceConfigurations) {
  // Create valid active space configurations (4 orbitals)
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0"),  // 4 electrons
      Configuration::from_spin_half_string("u2d0"),  // 4 electrons
      Configuration::from_spin_half_string("ud20")   // 4 electrons
  };

  // Should not throw - all have same orbital capacity and electron count
  EXPECT_NO_THROW(ConfigurationSet(configs, orbitals_with_active,
                                   Wavefunction::DEFAULT_SECTOR));
}

TEST_F(ConfigurationSetTest, RejectDifferentOrbitalCapacity) {
  // Configurations with different orbital capacities
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0"),  // 4 orbitals
      Configuration::from_spin_half_string("2ud00")  // 5 orbitals - different!
  };

  // Should throw - different orbital capacities
  EXPECT_THROW(ConfigurationSet(configs, orbitals_with_active,
                                Wavefunction::DEFAULT_SECTOR),
               std::invalid_argument);
}

TEST_F(ConfigurationSetTest, AcceptDifferentElectronCount) {
  // Particle-number conservation is a many-body symmetry constraint, not a
  // structural one. ConfigurationSet accepts mixed-N expansions; conservation
  // will be enforced by an optional many-body SymmetryProduct in a future PR.
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0"),  // 4 electrons
      Configuration::from_spin_half_string("2u00")   // 3 electrons
  };

  EXPECT_NO_THROW(ConfigurationSet(configs, orbitals_with_active,
                                   Wavefunction::DEFAULT_SECTOR));
}

TEST_F(ConfigurationSetTest, RejectOverhangingElectrons) {
  // Configurations with electrons beyond active space size
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string(
          "2ud0u")  // 5 orbitals, but 5th has electron (beyond active space)
  };

  // Should throw - electrons in overhanging orbitals
  EXPECT_THROW(ConfigurationSet(configs, orbitals_with_active,
                                Wavefunction::DEFAULT_SECTOR),
               std::invalid_argument);
}

TEST_F(ConfigurationSetTest, AllowOverhangingUnoccupied) {
  // Configurations with extra unoccupied orbitals beyond active space
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string(
          "2ud00"),  // 5 orbitals, 5th is unoccupied - OK
      Configuration::from_spin_half_string(
          "u2d00")  // 5 orbitals, 5th is unoccupied - OK
  };

  // Should not throw - overhanging orbitals are unoccupied
  EXPECT_NO_THROW(ConfigurationSet(configs, orbitals_with_active,
                                   Wavefunction::DEFAULT_SECTOR));
}

TEST_F(ConfigurationSetTest, EmptyConfigurationSet) {
  // Empty configuration set should be valid
  std::vector<Configuration> configs;
  EXPECT_NO_THROW(ConfigurationSet(configs, orbitals_with_active,
                                   Wavefunction::DEFAULT_SECTOR));
}

TEST_F(ConfigurationSetTest, SingleConfiguration) {
  // Single configuration should be valid
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0")};
  EXPECT_NO_THROW(ConfigurationSet(configs, orbitals_with_active,
                                   Wavefunction::DEFAULT_SECTOR));

  ConfigurationSet config_set(configs, orbitals_with_active,
                              Wavefunction::DEFAULT_SECTOR);
  EXPECT_EQ(config_set.size(), 1);
  EXPECT_FALSE(config_set.empty());
}

TEST_F(ConfigurationSetTest, NoActiveSpaceNoValidation) {
  // Without active space, only basic validation applies
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0"),
      Configuration::from_spin_half_string("u2d0")};

  // Should not throw even without active space
  EXPECT_THROW(ConfigurationSet(configs, orbitals_without_active,
                                Wavefunction::DEFAULT_SECTOR),
               std::invalid_argument);
}

TEST_F(ConfigurationSetTest, AccessOperators) {
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0"),
      Configuration::from_spin_half_string("u2d0"),
      Configuration::from_spin_half_string("ud20")};

  ConfigurationSet config_set(configs, orbitals_with_active,
                              Wavefunction::DEFAULT_SECTOR);

  // Test operator[]
  EXPECT_EQ(config_set[0].to_string(), "2ud0");
  EXPECT_EQ(config_set[1].to_string(), "u2d0");
  EXPECT_EQ(config_set[2].to_string(), "ud20");

  // Test at() with valid index
  EXPECT_EQ(config_set.at(0).to_string(), "2ud0");

  // Test at() with invalid index
  EXPECT_THROW(config_set.at(3), std::out_of_range);
}

TEST_F(ConfigurationSetTest, Iteration) {
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0"),
      Configuration::from_spin_half_string("u2d0"),
      Configuration::from_spin_half_string("ud20")};

  ConfigurationSet config_set(configs, orbitals_with_active,
                              Wavefunction::DEFAULT_SECTOR);

  // Test iteration
  size_t count = 0;
  for (const auto& config : config_set) {
    EXPECT_EQ(config.to_string(), configs[count].to_string());
    ++count;
  }
  EXPECT_EQ(count, 3);
}

TEST_F(ConfigurationSetTest, Equality) {
  std::vector<Configuration> configs1 = {
      Configuration::from_spin_half_string("2ud0"),
      Configuration::from_spin_half_string("u2d0")};

  std::vector<Configuration> configs2 = {
      Configuration::from_spin_half_string("2ud0"),
      Configuration::from_spin_half_string("u2d0")};

  std::vector<Configuration> configs3 = {
      Configuration::from_spin_half_string("2ud0"),
      Configuration::from_spin_half_string("ud20")  // Different
  };

  ConfigurationSet set1(configs1, orbitals_with_active,
                        Wavefunction::DEFAULT_SECTOR);
  ConfigurationSet set2(configs2, orbitals_with_active,
                        Wavefunction::DEFAULT_SECTOR);
  ConfigurationSet set3(configs3, orbitals_with_active,
                        Wavefunction::DEFAULT_SECTOR);

  // Same configurations and orbitals
  EXPECT_TRUE(set1 == set2);
  EXPECT_FALSE(set1 != set2);

  // Different configurations
  EXPECT_FALSE(set1 == set3);
  EXPECT_TRUE(set1 != set3);
}

TEST_F(ConfigurationSetTest, GetSummary) {
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0"),
      Configuration::from_spin_half_string("u2d0"),
      Configuration::from_spin_half_string("ud20")};

  ConfigurationSet config_set(configs, orbitals_with_active,
                              Wavefunction::DEFAULT_SECTOR);

  std::string summary = config_set.get_summary();

  // Verify summary contains key information
  EXPECT_NE(summary.find("ConfigurationSet"), std::string::npos);
  EXPECT_NE(summary.find("3"), std::string::npos);  // Number of configurations
  EXPECT_NE(summary.find("4"), std::string::npos);  // Orbital capacity
}

TEST_F(ConfigurationSetTest, MoveSemantics) {
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0"),
      Configuration::from_spin_half_string("u2d0")};

  // Test move constructor
  EXPECT_NO_THROW(ConfigurationSet(std::move(configs), orbitals_with_active,
                                   Wavefunction::DEFAULT_SECTOR));
}

TEST_F(ConfigurationSetTest, NullOrbitalsRejected) {
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0")};

  // Should throw - null orbitals pointer
  EXPECT_THROW(ConfigurationSet(configs, nullptr, Wavefunction::DEFAULT_SECTOR),
               std::invalid_argument);
}

TEST_F(ConfigurationTest, DataTypeName) {
  // Test that Configuration has the correct data type name
  auto config = Configuration::from_spin_half_string("ud0000");
  EXPECT_EQ(config.get_data_type_name(), "configuration");
}

TEST_F(ConfigurationSetTest, DataTypeName) {
  // Test that ConfigurationSet has the correct data type name
  std::vector<Configuration> configs = {
      Configuration::from_spin_half_string("2ud0")};
  ConfigurationSet config_set(configs, orbitals_with_active,
                              Wavefunction::DEFAULT_SECTOR);

  EXPECT_EQ(config_set.get_data_type_name(), "configuration_set");
}
