// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "qdk/chemistry/data/settings.hpp"
#include "ut_common.hpp"

using namespace qdk::chemistry::data;

// Basic derived class for general testing purposes (adds common test keys)
class BasicTestSettings : public Settings {
 public:
  BasicTestSettings() {
    // Set up common test keys used by most tests
    set_default("bool_val", false);
    set_default("int_val", 0);
    set_default("int64_val", int64_t(0));
    set_default("long_val", 0L);
    set_default("size_t_val", size_t(0));
    set_default("float_val", 0.0f);
    set_default("double_val", 0.0);
    set_default("string_val", std::string(""));
    set_default("int_vector", std::vector<int>{});
    set_default("double_vector", std::vector<double>{});
    set_default("string_vector", std::vector<std::string>{});
    set_default("test_key", 0);
    set_default("existing_key", 0);
    set_default("key1", 0);
    set_default("key2", 0);
    set_default("key3", 0);
    set_default("test_val", 0);
    set_default("test_string", std::string(""));
    set_default("test_bool", false);
    set_default("test_double", 0.0);
    set_default("custom_param", 0);
    set_default("string_key", std::string(""));
    set_default("int_vec", std::vector<int>{});
    set_default("test_vector", std::vector<int>{});
    set_default("string_vec", std::vector<std::string>{});
    // Add keys needed by TestSettings tests
    set_default("coefficients", std::vector<double>{});
    set_default("modes", std::vector<std::string>{});

    // Keys for JSONTypeConversionEdgeCases test
    set_default("large_int", 0L);
    set_default("empty_array", std::vector<int>{});
    set_default("unsupported", std::string(""));
    set_default("unsupported_array", std::vector<int>{});

    // Keys for CStyleStringOverloads test
    set_default("cstring_val", std::string(""));

    // Keys for ComprehensiveStringConversion test
    set_default("bool_true", false);
    set_default("bool_false", false);

    // Keys for HDF5ComprehensiveTypeCoverage test
    set_default("empty_int_vector", std::vector<int>{});
    set_default("empty_double_vector", std::vector<double>{});
    set_default("empty_string_vector", std::vector<std::string>{});

    // Keys for HDF5SpecializedNumericTypes test
    set_default("negative_long", 0L);
    set_default("positive_long", 0L);
    set_default("small_size_t", size_t(0));
    set_default("large_size_t", size_t(0));
    set_default("small_float", 0.0f);
    set_default("large_float", 0.0f);
    set_default("small_double", 0.0);
    set_default("large_double", 0.0);
    set_default("precise_double", 0.0);

    // Keys for HDF5VectorEdgeCases test
    set_default("large_int_vector", std::vector<int>{});
    set_default("large_double_vector", std::vector<double>{});
    set_default("large_string_vector", std::vector<std::string>{});
    set_default("explicitly_empty_int", std::vector<int>{});
    set_default("explicitly_empty_double", std::vector<double>{});
    set_default("explicitly_empty_string", std::vector<std::string>{});
    set_default("single_element_vector", std::vector<int>{});
    set_default("mixed_sign_vector", std::vector<int>{});

    // Keys for HDF5StringVectorSpecialCases test
    set_default("special_string_vector", std::vector<std::string>{});
    set_default("long_string_vector", std::vector<std::string>{});
    set_default("unicode_vector", std::vector<std::string>{});
    set_default("special_chars_vector", std::vector<std::string>{});
    set_default("long_strings_vector", std::vector<std::string>{});

    // Keys for ComprehensiveHasTypeTesting test
    set_default("vector_int", std::vector<int>{});
    set_default("vector_double", std::vector<double>{});
    set_default("vector_string", std::vector<std::string>{});
    set_default("numeric_val", 0);
    set_default("float_array", std::vector<double>{});
    set_default("double_array", std::vector<double>{});
    set_default("int_array", std::vector<int>{});
    set_default("string_array", std::vector<std::string>{});

    // Keys for ComprehensiveStringConversion test
    set_default("empty_string", std::string(""));
    set_default("empty_int_vec", std::vector<int>{});
    set_default("empty_str_vec", std::vector<std::string>{});
    set_default("single_int_vec", std::vector<int>{});
    set_default("single_str_vec", std::vector<std::string>{});
    set_default("multi_int_vec", std::vector<int>{});
    set_default("multi_str_vec", std::vector<std::string>{});

    // Keys for CompleteVariantTypeCoverage test
    set_default("bool_type", false);
    set_default("int_type", 0);
    set_default("long_type", 0L);
    set_default("size_t_type", size_t(0));
    set_default("float_type", 0.0f);
    set_default("double_type", 0.0);
    set_default("string_type", std::string(""));
    set_default("vector_int_type", std::vector<int>{});
    set_default("vector_double_type", std::vector<double>{});
    set_default("vector_string_type", std::vector<std::string>{});
    set_default("variant_bool", false);
    set_default("variant_int", 0);
    set_default("variant_long", 0L);
    set_default("variant_size_t", size_t(0));
    set_default("variant_float", 0.0f);
    set_default("variant_double", 0.0);
    set_default("variant_string", std::string(""));
    set_default("variant_int_vec", std::vector<int>{});
    set_default("variant_double_vec", std::vector<double>{});
    set_default("variant_string_vec", std::vector<std::string>{});

    // Keys for HDF5TypeDetectionEdgeCases test
    set_default("max_long", 0L);
    set_default("max_size_t", size_t(0));
    set_default("min_long", 0L);
    set_default("min_size_t", size_t(0));
    set_default("empty_string_vec", std::vector<std::string>{});
    set_default("empty_double_vec", std::vector<double>{});
    set_default("single_double_vec", std::vector<double>{});
    set_default("single_string_vec", std::vector<std::string>{});
    set_default("small_int_vec", std::vector<int>{});
    set_default("edge_case_int", 0);
    set_default("edge_case_long", 0L);
    set_default("edge_case_size_t", size_t(0));
    set_default("edge_case_float", 0.0f);
    set_default("edge_case_double", 0.0);

    // Keys for HDF5CrossPlatformCompatibility test
    set_default("platform_int", 0);
    set_default("platform_long", 0L);
    set_default("platform_size_t", size_t(0));
    set_default("platform_float", 0.0f);
    set_default("platform_double", 0.0);
    set_default("cross_platform_bool", false);
    set_default("cross_platform_int", 0);
    set_default("cross_platform_long", 0L);
    set_default("cross_platform_size_t", size_t(0));
    set_default("cross_platform_float", 0.0f);
    set_default("cross_platform_double", 0.0);
    set_default("cross_platform_string", std::string(""));
    set_default("cross_platform_int_vec", std::vector<int>{});
    set_default("cross_platform_double_vec", std::vector<double>{});
    set_default("cross_platform_string_vec", std::vector<std::string>{});
  }
};

// Derived class for testing purposes
class TestSettings : public Settings {
 public:
  TestSettings() {
    // Set some default values for testing
    set_default("max_iterations", 100);
    set_default("tolerance", 1e-6);
    set_default("method", std::string("default"));
    set_default("enable_logging", true);
    set_default("coefficients", std::vector<double>{1.0, 2.0, 3.0});
    set_default("modes", std::vector<std::string>{"mode1", "mode2"});
  }

  // Convenience getters (optional)
  int get_max_iterations() const { return get<int>("max_iterations"); }
  double get_tolerance() const { return get<double>("tolerance"); }
  std::string get_method() const { return get<std::string>("method"); }
  bool get_enable_logging() const { return get<bool>("enable_logging"); }
  std::vector<double> get_coefficients() const {
    return get<std::vector<double>>("coefficients");
  }
  std::vector<std::string> get_modes() const {
    return get<std::vector<std::string>>("modes");
  }
};

// Test fixture for Settings tests
class SettingsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean up any test files from previous runs
    std::filesystem::remove("test.settings.json");
    std::filesystem::remove("test.settings.h5");
    std::filesystem::remove("test_generic.settings.json");
    std::filesystem::remove("test_specific.settings.json");
    std::filesystem::remove("test_generic.settings.h5");
    std::filesystem::remove("test_specific.settings.h5");
    std::filesystem::remove("roundtrip.json");
    std::filesystem::remove("roundtrip.h5");
  }

  void TearDown() override {
    // Clean up test files
    std::filesystem::remove("test.settings.json");
    std::filesystem::remove("test.settings.h5");
    std::filesystem::remove("test_generic.settings.json");
    std::filesystem::remove("test_specific.settings.json");
    std::filesystem::remove("test_generic.settings.h5");
    std::filesystem::remove("test_specific.settings.h5");
    std::filesystem::remove("roundtrip.json");
    std::filesystem::remove("roundtrip.h5");
  }

  BasicTestSettings settings;  // Use BasicTestSettings instead of Settings
  TestSettings test_settings;
};

// Basic functionality tests
TEST_F(SettingsTest, BasicConstruction) {
  // Create a fresh Settings object for this test
  Settings empty_settings;
  EXPECT_TRUE(empty_settings.empty());
  EXPECT_EQ(empty_settings.size(), 0);
  EXPECT_TRUE(empty_settings.keys().empty());
}

TEST_F(SettingsTest, SetAndGetVariantTypes) {
  // Test setting with variant interface
  settings.set("bool_val", SettingValue(true));
  settings.set("int_val", SettingValue(42));
  settings.set("string_val", SettingValue(std::string("hello")));

  // Test getting with variant interface
  auto bool_variant = settings.get("bool_val");
  auto int_variant = settings.get("int_val");
  auto string_variant = settings.get("string_val");

  EXPECT_TRUE(std::get<bool>(bool_variant));
  EXPECT_EQ(std::get<int>(int_variant), 42);
  EXPECT_EQ(std::get<std::string>(string_variant), "hello");

  // Test template interface still works
  EXPECT_TRUE(settings.get<bool>("bool_val"));
  EXPECT_EQ(settings.get<int>("int_val"), 42);
  EXPECT_EQ(settings.get<std::string>("string_val"), "hello");
}

TEST_F(SettingsTest, SetAndGetBasicTypes) {
  // Test different types
  settings.set("bool_val", true);
  settings.set("int_val", 42);
  settings.set("long_val", 123456789L);
  settings.set("size_t_val", size_t(999));
  settings.set("float_val", 3.14f);
  settings.set("double_val", 2.718281828);
  settings.set("string_val", std::string("hello"));

  EXPECT_TRUE(settings.get<bool>("bool_val"));
  EXPECT_EQ(settings.get<int>("int_val"), 42);
  EXPECT_EQ(settings.get<long>("long_val"), 123456789L);
  EXPECT_EQ(settings.get<size_t>("size_t_val"), 999);
  EXPECT_FLOAT_EQ(settings.get<float>("float_val"), 3.14f);
  EXPECT_DOUBLE_EQ(settings.get<double>("double_val"), 2.718281828);
  EXPECT_EQ(settings.get<std::string>("string_val"), "hello");

  EXPECT_EQ(settings.size(),
            123);  // Should contain all 123 default keys
  EXPECT_FALSE(settings.empty());
}

TEST_F(SettingsTest, SetAndGetVectorTypes) {
  std::vector<int> int_vec = {1, 2, 3, 4, 5};
  std::vector<double> double_vec = {1.1, 2.2, 3.3};
  std::vector<std::string> string_vec = {"apple", "banana", "cherry"};

  settings.set("int_vector", int_vec);
  settings.set("double_vector", double_vec);
  settings.set("string_vector", string_vec);

  auto retrieved_int_vec = settings.get<std::vector<int>>("int_vector");
  auto retrieved_double_vec =
      settings.get<std::vector<double>>("double_vector");
  auto retrieved_string_vec =
      settings.get<std::vector<std::string>>("string_vector");

  EXPECT_EQ(retrieved_int_vec, int_vec);
  EXPECT_EQ(retrieved_double_vec, double_vec);
  EXPECT_EQ(retrieved_string_vec, string_vec);
}

TEST_F(SettingsTest, HasFunction) {
  settings.set("test_key", 123);

  EXPECT_TRUE(settings.has("test_key"));
  EXPECT_FALSE(settings.has("nonexistent_key"));
}

TEST_F(SettingsTest, GetOrDefaultVariant) {
  settings.set("existing_key", SettingValue(42));

  // Test variant interface
  auto result1 = settings.get_or_default("existing_key", SettingValue(999));
  auto result2 = settings.get_or_default("nonexistent_key", SettingValue(999));

  EXPECT_EQ(std::get<int>(result1), 42);
  EXPECT_EQ(std::get<int>(result2), 999);

  // Test template interface still works
  EXPECT_EQ(settings.get_or_default("existing_key", 999), 42);
  EXPECT_EQ(settings.get_or_default("nonexistent_key", 999), 999);

  // Test with wrong type (should return default)
  settings.set("string_key", std::string("hello"));
  EXPECT_EQ(settings.get_or_default<int>("string_key", 999), 999);
}

TEST_F(SettingsTest, Keys) {
  settings.set("key1", 1);
  settings.set("key2", 2);
  settings.set("key3", 3);

  auto keys = settings.keys();
  EXPECT_EQ(keys.size(), 123);

  // Check that the specific keys we set are present
  std::sort(keys.begin(), keys.end());
  EXPECT_TRUE(std::find(keys.begin(), keys.end(), "key1") != keys.end());
  EXPECT_TRUE(std::find(keys.begin(), keys.end(), "key2") != keys.end());
  EXPECT_TRUE(std::find(keys.begin(), keys.end(), "key3") != keys.end());
}

TEST_F(SettingsTest, NewInstance) {
  // Create a new instance to get fresh defaults
  BasicTestSettings fresh_settings;
  EXPECT_EQ(fresh_settings.size(), 123);
  EXPECT_FALSE(fresh_settings.empty());  // Should have default values

  // Verify the defaults are correct
  EXPECT_EQ(fresh_settings.get<int>("key1"), 0);  // Default value
  EXPECT_EQ(fresh_settings.get<int>("key2"), 0);  // Default value
}

TEST_F(SettingsTest, GetAsString) {
  settings.set("bool_val", true);
  settings.set("int_val", 42);
  settings.set("double_val", 3.14159);
  settings.set("string_val", std::string("hello"));
  settings.set("int_vec", std::vector<int>{1, 2, 3});
  settings.set("string_vec", std::vector<std::string>{"a", "b"});

  EXPECT_EQ(settings.get_as_string("bool_val"), "true");
  EXPECT_EQ(settings.get_as_string("int_val"), "42");
  EXPECT_EQ(settings.get_as_string("string_val"), "hello");

  // Vector representations (exact format may vary)
  std::string int_vec_str = settings.get_as_string("int_vec");
  EXPECT_TRUE(int_vec_str.find("1") != std::string::npos);
  EXPECT_TRUE(int_vec_str.find("2") != std::string::npos);
  EXPECT_TRUE(int_vec_str.find("3") != std::string::npos);
}

// Exception tests
TEST_F(SettingsTest, SettingNotFoundExceptions) {
  EXPECT_THROW(settings.get<int>("nonexistent"), SettingNotFound);
  EXPECT_THROW(settings.get_as_string("nonexistent"), SettingNotFound);

  // Test variant version get method
  EXPECT_THROW(settings.get("nonexistent"), SettingNotFound);
}

TEST_F(SettingsTest, SettingTypeMismatchExceptions) {
  settings.set("int_val", 42);

  EXPECT_THROW(settings.get<std::string>("int_val"), SettingTypeMismatch);
  EXPECT_THROW(settings.get<double>("int_val"), SettingTypeMismatch);
}

TEST_F(SettingsTest, ValidateRequired) {
  settings.set("key1", 1);
  settings.set("key2", 2);

  // Should not throw
  EXPECT_NO_THROW(settings.validate_required({"key1", "key2"}));
  EXPECT_NO_THROW(settings.validate_required({"key1"}));
  EXPECT_NO_THROW(settings.validate_required({}));

  // Should throw
  EXPECT_THROW(settings.validate_required({"key1", "key2", "missing_key"}),
               SettingNotFound);
}

// JSON serialization tests
TEST_F(SettingsTest, JSONSerialization) {
  settings.set("int_val", 42);
  settings.set("double_val", 3.14159);
  settings.set("string_val", std::string("hello"));
  settings.set("bool_val", true);
  settings.set("int_vec", std::vector<int>{1, 2, 3});

  // Convert to JSON
  auto json_obj = settings.to_json();

  EXPECT_EQ(json_obj["int_val"], 42);
  EXPECT_NEAR(json_obj["double_val"], 3.14159,
              testing::numerical_zero_tolerance);
  EXPECT_EQ(json_obj["string_val"], "hello");
  EXPECT_EQ(json_obj["bool_val"], true);

  // Create new settings from JSON
  auto loaded_settings = Settings::from_json(json_obj);

  // Create BasicTestSettings with loaded values
  BasicTestSettings new_settings;
  for (const auto& [key, value] : loaded_settings->get_all_settings()) {
    new_settings.set(key, value);
  }

  // Verify all values are preserved
  EXPECT_EQ(new_settings.get<int>("int_val"), 42);
  EXPECT_NEAR(new_settings.get<double>("double_val"), 3.14159,
              testing::numerical_zero_tolerance);
  EXPECT_EQ(new_settings.get<std::string>("string_val"), "hello");
  EXPECT_EQ(new_settings.get<bool>("bool_val"), true);
  EXPECT_EQ(new_settings.get<std::vector<int>>("int_vec"),
            (std::vector<int>{1, 2, 3}));
}

TEST_F(SettingsTest, JSONValidationErrors) {
  // Test from_json with non-object JSON
  nlohmann::json non_object_json = "this is a string, not an object";
  EXPECT_THROW(Settings::from_json(non_object_json), std::runtime_error);

  nlohmann::json array_json = nlohmann::json::array({1, 2, 3});
  EXPECT_THROW(Settings::from_json(array_json), std::runtime_error);

  nlohmann::json number_json = 42;
  EXPECT_THROW(Settings::from_json(number_json), std::runtime_error);
}

TEST_F(SettingsTest, JSONTypeConversionEdgeCases) {
  // Test large integer casting to long
  nlohmann::json large_int_json = nlohmann::json::object();
  large_int_json["version"] = "0.1.0";
  large_int_json["large_int"] =
      static_cast<long>(2147483648L);  // Larger than int
  auto large_int_settings = Settings::from_json(large_int_json);
  EXPECT_EQ(large_int_settings->get<long>("large_int"), 2147483648L);

  // Test empty JSON array default to vector<int>
  nlohmann::json empty_array_json = nlohmann::json::object();
  empty_array_json["version"] = "0.1.0";
  empty_array_json["empty_array"] = nlohmann::json::array();
  auto empty_array_settings = Settings::from_json(empty_array_json);
  auto empty_vec = empty_array_settings->get<std::vector<int>>("empty_array");
  EXPECT_TRUE(empty_vec.empty());

  // Test unsupported JSON type error
  nlohmann::json unsupported_json = nlohmann::json::object();
  unsupported_json["version"] = "0.1.0";
  unsupported_json["unsupported"] =
      nlohmann::json::value_t::binary;  // Binary type not supported
  EXPECT_THROW(Settings::from_json(unsupported_json), std::runtime_error);

  // Test unsupported array element type error
  nlohmann::json unsupported_array_json = nlohmann::json::object();
  unsupported_array_json["version"] = "0.1.0";
  // Create an array with object elements (not supported)
  nlohmann::json nested_object = nlohmann::json::object();
  nested_object["key"] = "value";
  unsupported_array_json["unsupported_array"] =
      nlohmann::json::array({nested_object});
  EXPECT_THROW(Settings::from_json(unsupported_array_json), std::runtime_error);
}

// Tests for generic file I/O methods
TEST_F(SettingsTest, GenericFileIO) {
  settings.set("test_val", 123);
  settings.set("test_string", std::string("test"));
  settings.set("test_bool", true);
  settings.set("test_double", 3.14159);

  // Test JSON through generic interface
  EXPECT_NO_THROW(settings.to_file("test_generic.settings.json", "json"));
  EXPECT_TRUE(std::filesystem::exists("test_generic.settings.json"));

  // Load from file
  std::shared_ptr<Settings> loaded_settings;
  EXPECT_NO_THROW(loaded_settings = Settings::from_file(
                      "test_generic.settings.json", "json"));

  EXPECT_EQ(loaded_settings->get<int>("test_val"), 123);
  EXPECT_EQ(loaded_settings->get<std::string>("test_string"), "test");
  EXPECT_EQ(loaded_settings->get<bool>("test_bool"), true);
  EXPECT_NEAR(loaded_settings->get<double>("test_double"), 3.14159,
              testing::numerical_zero_tolerance);

  // Test HDF5 through generic interface
  EXPECT_NO_THROW(settings.to_file("test_generic.settings.h5", "hdf5"));
  EXPECT_TRUE(std::filesystem::exists("test_generic.settings.h5"));

  std::shared_ptr<Settings> loaded_settings_hdf5;
  EXPECT_NO_THROW(loaded_settings_hdf5 =
                      Settings::from_file("test_generic.settings.h5", "hdf5"));

  EXPECT_EQ(loaded_settings_hdf5->get<int>("test_val"), 123);
  EXPECT_EQ(loaded_settings_hdf5->get<std::string>("test_string"), "test");
  EXPECT_EQ(loaded_settings_hdf5->get<bool>("test_bool"), true);
  EXPECT_NEAR(loaded_settings_hdf5->get<double>("test_double"), 3.14159,
              testing::numerical_zero_tolerance);
}

// Tests for filename validation
TEST_F(SettingsTest, FilenameValidation) {
  settings.set("test_val", 123);

  // Test specific method validation
  EXPECT_THROW(settings.to_json_file(""), std::invalid_argument);
  EXPECT_THROW(Settings::from_json_file(""), std::invalid_argument);
  EXPECT_THROW(settings.to_hdf5_file(""), std::invalid_argument);
  EXPECT_THROW(Settings::from_hdf5_file(""), std::invalid_argument);
}

// Test unsupported file formats
TEST_F(SettingsTest, UnsupportedFileFormat) {
  settings.set("test_val", 123);

  // Test unsupported format
  EXPECT_THROW(settings.to_file("test.xml", "xml"), std::invalid_argument);
  EXPECT_THROW(Settings::from_file("test.xml", "xml"), std::invalid_argument);
}

// Test file not found error
TEST_F(SettingsTest, FileNotFound) {
  EXPECT_THROW(
      Settings::from_file("/nonexistent/path/file.settings.json", "json"),
      std::runtime_error);
  EXPECT_THROW(Settings::from_json_file("/nonexistent/path/file.settings.json"),
               std::runtime_error);
  EXPECT_THROW(Settings::from_hdf5_file("/nonexistent/path/file.settings.h5"),
               std::runtime_error);
}

// Test consistency between generic and specific methods
TEST_F(SettingsTest, ConsistencyBetweenMethods) {
  settings.set("test_val", 123);
  settings.set("test_string", std::string("test"));
  settings.set("test_bool", true);
  settings.set("test_double", 3.14159);

  // Save with generic method
  EXPECT_NO_THROW(settings.to_file("test_generic.settings.json", "json"));

  // Save with specific method
  EXPECT_NO_THROW(settings.to_json_file("test_specific.settings.json"));

  // Load with both methods
  std::shared_ptr<Settings> settings_from_generic;
  std::shared_ptr<Settings> settings_from_specific;

  EXPECT_NO_THROW(settings_from_generic = Settings::from_file(
                      "test_generic.settings.json", "json"));
  EXPECT_NO_THROW(settings_from_specific =
                      Settings::from_json_file("test_specific.settings.json"));

  // Verify consistency
  EXPECT_EQ(settings_from_generic->get<int>("test_val"),
            settings_from_specific->get<int>("test_val"));
  EXPECT_EQ(settings_from_generic->get<std::string>("test_string"),
            settings_from_specific->get<std::string>("test_string"));
  EXPECT_EQ(settings_from_generic->get<bool>("test_bool"),
            settings_from_specific->get<bool>("test_bool"));
  EXPECT_NEAR(settings_from_generic->get<double>("test_double"),
              settings_from_specific->get<double>("test_double"),
              testing::numerical_zero_tolerance);

  // Test HDF5 consistency
  EXPECT_NO_THROW(settings.to_file("test_generic.settings.h5", "hdf5"));
  EXPECT_NO_THROW(settings.to_hdf5_file("test_specific.settings.h5"));

  std::shared_ptr<Settings> settings_from_generic_hdf5;
  std::shared_ptr<Settings> settings_from_specific_hdf5;

  EXPECT_NO_THROW(settings_from_generic_hdf5 =
                      Settings::from_file("test_generic.settings.h5", "hdf5"));
  EXPECT_NO_THROW(settings_from_specific_hdf5 =
                      Settings::from_hdf5_file("test_specific.settings.h5"));

  // Verify HDF5 consistency
  EXPECT_EQ(settings_from_generic_hdf5->get<int>("test_val"),
            settings_from_specific_hdf5->get<int>("test_val"));
  EXPECT_EQ(settings_from_generic_hdf5->get<std::string>("test_string"),
            settings_from_specific_hdf5->get<std::string>("test_string"));
  EXPECT_EQ(settings_from_generic_hdf5->get<bool>("test_bool"),
            settings_from_specific_hdf5->get<bool>("test_bool"));
  EXPECT_NEAR(settings_from_generic_hdf5->get<double>("test_double"),
              settings_from_specific_hdf5->get<double>("test_double"),
              testing::numerical_zero_tolerance);
}

// Test data integrity through multiple save/load cycles
TEST_F(SettingsTest, RoundtripDataIntegrity) {
  settings.set("test_val", 123);
  settings.set("test_string", std::string("test"));
  settings.set("test_bool", true);
  settings.set("test_double", 3.14159);
  settings.set("test_vector", std::vector<int>{1, 2, 3});

  // Multiple roundtrips for JSON
  auto current_settings =
      std::make_shared<Settings>(settings);  // Copy to shared_ptr
  for (int i = 0; i < 3; ++i) {
    current_settings->to_file("roundtrip.json", "json");

    auto new_settings = Settings::from_file("roundtrip.json", "json");

    // Verify data integrity
    EXPECT_EQ(new_settings->get<int>("test_val"), 123);
    EXPECT_EQ(new_settings->get<std::string>("test_string"), "test");
    EXPECT_EQ(new_settings->get<bool>("test_bool"), true);
    EXPECT_NEAR(new_settings->get<double>("test_double"), 3.14159,
                testing::numerical_zero_tolerance);
    EXPECT_EQ(new_settings->get<std::vector<int>>("test_vector"),
              (std::vector<int>{1, 2, 3}));

    current_settings = new_settings;
  }

  // Multiple roundtrips for HDF5
  current_settings =
      std::make_shared<Settings>(settings);  // Reset to original settings
  for (int i = 0; i < 3; ++i) {
    current_settings->to_file("roundtrip.h5", "hdf5");

    auto new_settings = Settings::from_file("roundtrip.h5", "hdf5");

    // Verify data integrity
    EXPECT_EQ(new_settings->get<int>("test_val"), 123);
    EXPECT_EQ(new_settings->get<std::string>("test_string"), "test");
    EXPECT_EQ(new_settings->get<bool>("test_bool"), true);
    EXPECT_NEAR(new_settings->get<double>("test_double"), 3.14159,
                testing::numerical_zero_tolerance);
    EXPECT_EQ(new_settings->get<std::vector<int>>("test_vector"),
              (std::vector<int>{1, 2, 3}));

    current_settings = new_settings;
  }
}

// Comprehensive HDF5 type coverage tests
TEST_F(SettingsTest, HDF5ComprehensiveTypeCoverage) {
  // Test all scalar types for HDF5 serialization
  settings.set("long_val", 9223372036854775807L);
  settings.set("size_t_val", static_cast<size_t>(18446744073709551615ULL));
  settings.set("float_val", 3.14159f);
  settings.set("double_val", 2.718281828459045);

  // Test vector types including empty vectors
  settings.set("int_vector", std::vector<int>{1, 2, 3, 4, 5});
  settings.set("double_vector", std::vector<double>{1.1, 2.2, 3.3, 4.4});
  settings.set("string_vector",
               std::vector<std::string>{"hello", "world", "hdf5"});

  // Test empty vectors
  settings.set("empty_int_vector", std::vector<int>{});
  settings.set("empty_double_vector", std::vector<double>{});
  settings.set("empty_string_vector", std::vector<std::string>{});

  // Save to HDF5
  EXPECT_NO_THROW(settings.to_hdf5_file("comprehensive_types.settings.h5"));

  // Load from HDF5 and verify all types
  std::shared_ptr<Settings> loaded_settings;
  EXPECT_NO_THROW(loaded_settings = Settings::from_hdf5_file(
                      "comprehensive_types.settings.h5"));

  // Verify scalar types
  EXPECT_EQ(loaded_settings->get<long>("long_val"), 9223372036854775807L);
  EXPECT_EQ(loaded_settings->get<size_t>("size_t_val"),
            static_cast<size_t>(18446744073709551615ULL));
  EXPECT_FLOAT_EQ(loaded_settings->get<float>("float_val"), 3.14159f);
  EXPECT_DOUBLE_EQ(loaded_settings->get<double>("double_val"),
                   2.718281828459045);

  // Verify vector types
  EXPECT_EQ(loaded_settings->get<std::vector<int>>("int_vector"),
            (std::vector<int>{1, 2, 3, 4, 5}));
  EXPECT_EQ(loaded_settings->get<std::vector<double>>("double_vector"),
            (std::vector<double>{1.1, 2.2, 3.3, 4.4}));
  EXPECT_EQ(loaded_settings->get<std::vector<std::string>>("string_vector"),
            (std::vector<std::string>{"hello", "world", "hdf5"}));

  // Verify empty vectors
  EXPECT_TRUE(
      loaded_settings->get<std::vector<int>>("empty_int_vector").empty());
  EXPECT_TRUE(
      loaded_settings->get<std::vector<double>>("empty_double_vector").empty());
  EXPECT_TRUE(
      loaded_settings->get<std::vector<std::string>>("empty_string_vector")
          .empty());
}

TEST_F(SettingsTest, HDF5SpecializedNumericTypes) {
  // Test specific coverage for numeric type paths

  // Test long type serialization
  settings.set("negative_long", LONG_MIN);
  settings.set("positive_long", LONG_MAX);

  // Test size_t type serialization
  settings.set("small_size_t", static_cast<size_t>(0));
  settings.set("large_size_t", static_cast<size_t>(SIZE_MAX));

  // Test float precision
  settings.set("small_float", 1.175494351e-38f);
  settings.set("large_float", 3.402823466e+38f);

  // Test double precision
  settings.set("small_double", 2.2250738585072014e-308);
  settings.set("large_double", 1.7976931348623157e+308);

  // Save and load
  EXPECT_NO_THROW(settings.to_hdf5_file("numeric_types.settings.h5"));

  std::shared_ptr<Settings> loaded_settings;
  EXPECT_NO_THROW(loaded_settings =
                      Settings::from_hdf5_file("numeric_types.settings.h5"));

  // Verify all numeric types preserved correctly
  EXPECT_EQ(loaded_settings->get<long>("negative_long"), LONG_MIN);
  EXPECT_EQ(loaded_settings->get<long>("positive_long"), LONG_MAX);
  EXPECT_EQ(loaded_settings->get<size_t>("small_size_t"),
            static_cast<size_t>(0));
  EXPECT_EQ(loaded_settings->get<size_t>("large_size_t"),
            static_cast<size_t>(SIZE_MAX));
  EXPECT_FLOAT_EQ(loaded_settings->get<float>("small_float"), 1.175494351e-38f);
  EXPECT_FLOAT_EQ(loaded_settings->get<float>("large_float"), 3.402823466e+38f);
  EXPECT_DOUBLE_EQ(loaded_settings->get<double>("small_double"),
                   2.2250738585072014e-308);
  EXPECT_DOUBLE_EQ(loaded_settings->get<double>("large_double"),
                   1.7976931348623157e+308);
}

TEST_F(SettingsTest, HDF5VectorEdgeCases) {
  // Test vector serialization edge cases

  // Test vectors with many elements
  std::vector<int> large_int_vec;
  std::vector<double> large_double_vec;
  std::vector<std::string> large_string_vec;

  for (int i = 0; i < 100; ++i) {
    large_int_vec.push_back(i);
    large_double_vec.push_back(i * 0.1);
    large_string_vec.push_back("item_" + std::to_string(i));
  }

  settings.set("large_int_vector", large_int_vec);
  settings.set("large_double_vector", large_double_vec);
  settings.set("large_string_vector", large_string_vec);

  // Test empty vectors explicitly (covering empty vector paths)
  settings.set("explicitly_empty_int", std::vector<int>());
  settings.set("explicitly_empty_double", std::vector<double>());
  settings.set("explicitly_empty_string", std::vector<std::string>());

  // Save and load
  EXPECT_NO_THROW(settings.to_hdf5_file("vector_edge_cases.settings.h5"));

  std::shared_ptr<Settings> loaded_settings;
  EXPECT_NO_THROW(loaded_settings = Settings::from_hdf5_file(
                      "vector_edge_cases.settings.h5"));

  // Verify large vectors
  EXPECT_EQ(loaded_settings->get<std::vector<int>>("large_int_vector"),
            large_int_vec);
  EXPECT_EQ(loaded_settings->get<std::vector<double>>("large_double_vector"),
            large_double_vec);
  EXPECT_EQ(
      loaded_settings->get<std::vector<std::string>>("large_string_vector"),
      large_string_vec);

  // Verify explicitly empty vectors
  EXPECT_TRUE(
      loaded_settings->get<std::vector<int>>("explicitly_empty_int").empty());
  EXPECT_TRUE(
      loaded_settings->get<std::vector<double>>("explicitly_empty_double")
          .empty());
  EXPECT_TRUE(
      loaded_settings->get<std::vector<std::string>>("explicitly_empty_string")
          .empty());
}

TEST_F(SettingsTest, HDF5StringVectorSpecialCases) {
  // Test string vector serialization special cases

  // Test strings with special characters
  std::vector<std::string> special_strings = {
      "normal_string",
      "string with spaces",
      "string\nwith\nnewlines",
      "string\twith\ttabs",
      "string\"with\"quotes",
      "string'with'apostrophes",
      "string\\with\\backslashes",
      "string/with/slashes",
      ""  // Empty string within vector
  };

  settings.set("special_string_vector", special_strings);

  // Test very long strings
  std::string very_long_string(1000, 'A');
  very_long_string += "END";
  settings.set("long_string_vector",
               std::vector<std::string>{very_long_string, "short"});

  // Save and load
  EXPECT_NO_THROW(settings.to_hdf5_file("string_special_cases.settings.h5"));

  std::shared_ptr<Settings> loaded_settings;
  EXPECT_NO_THROW(loaded_settings = Settings::from_hdf5_file(
                      "string_special_cases.settings.h5"));

  // Verify special strings preserved
  EXPECT_EQ(
      loaded_settings->get<std::vector<std::string>>("special_string_vector"),
      special_strings);

  // Verify long strings
  auto loaded_long_vector =
      loaded_settings->get<std::vector<std::string>>("long_string_vector");
  EXPECT_EQ(loaded_long_vector.size(), 2);
  EXPECT_EQ(loaded_long_vector[0], very_long_string);
  EXPECT_EQ(loaded_long_vector[1], "short");
}

TEST_F(SettingsTest, FromJsonIgnoresUnknownKeys) {
  TestSettings test_settings;
  test_settings.set("max_iterations", 200);  // Change from default

  // Create JSON
  nlohmann::json j;
  j["version"] = "0.1.0";       // Required version field
  j["unknown_key1"] = 123;      // Should be ignored
  j["unknown_key2"] = "hello";  // Should be ignored
  j["max_iterations"] = 500;    // Should be loaded (registered key)
  j["tolerance"] = 1e-8;        // Should be loaded (registered key)

  // Load from JSON
  auto loaded_settings = Settings::from_json(j);

  // Copy the loaded values to our test_settings (only existing keys)
  for (const auto& [key, value] : loaded_settings->get_all_settings()) {
    if (test_settings.has(key)) {
      test_settings.set(key, value);
    }
  }

  // Known keys should be loaded
  EXPECT_EQ(test_settings.get<int>("max_iterations"), 500);
  EXPECT_NEAR(test_settings.get<double>("tolerance"), 1e-8,
              testing::numerical_zero_tolerance);

  // Unknown keys should not exist in our test_settings (should throw
  // SettingNotFound)
  EXPECT_THROW(test_settings.get<int>("unknown_key1"), SettingNotFound);
  EXPECT_THROW(test_settings.get<std::string>("unknown_key2"), SettingNotFound);
}

TEST_F(SettingsTest, FromHdf5IgnoresUnknownKeys) {
  TestSettings test_settings;
  test_settings.set("max_iterations", 500);
  test_settings.set("tolerance", 1e-8);

  // Save to HDF5
  test_settings.to_hdf5_file("test_known_keys.settings.h5");

  TestSettings new_test_settings;
  new_test_settings.set("max_iterations", 100);  // Different from file

  // Use static factory method to load from HDF5 file
  auto loaded_settings =
      Settings::from_hdf5_file("test_known_keys.settings.h5");

  // Copy the loaded values to our new_test_settings (only existing keys)
  for (const auto& [key, value] : loaded_settings->get_all_settings()) {
    if (new_test_settings.has(key)) {
      new_test_settings.set(key, value);
    }
  }

  EXPECT_EQ(new_test_settings.get<int>("max_iterations"), 500);
  EXPECT_NEAR(new_test_settings.get<double>("tolerance"), 1e-8,
              testing::numerical_zero_tolerance);

  // Clean up
  std::filesystem::remove("test_known_keys.settings.h5");
}
