// Settings usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// start-cell-1
// Get the settings object
auto& settings = algorithm->settings();

// Set a parameter
settings.set("parameter_name", value);

// Get a parameter
auto value = settings.get<ValueType>("parameter_name");
// end-cell-1

// start-cell-2
// Set a string value
settings.set("basis_set", "def2-tzvp");

// Set a numeric value
settings.set("convergence_threshold", 1.0e-8);

// Set a boolean value
settings.set("density_fitting", true);

// Set an array value
settings.set("active_orbitals", std::vector<int>{4, 5, 6, 7});
// end-cell-2

// start-cell-3
// Get a string value
std::string basis = settings.get<std::string>("basis_set");

// Get a numeric value
double threshold = settings.get<double>("convergence_threshold");

// Get a boolean value
bool use_df = settings.get<bool>("density_fitting");

// Get an array value
auto active_orbitals = settings.get<std::vector<int>>("active_orbitals");

// Get a value with default fallback
auto max_iter = settings.get_or_default<int>("max_iterations", 100);
// end-cell-3

// start-cell-4
// Check if a setting exists
if (settings.has("basis_set")) {
  // Use the setting
}

// Check if a setting exists with the expected type
if (settings.has_type<double>("convergence_threshold")) {
  // Use the setting
}

// Try to get a value (returns std::optional)
auto maybe_value = settings.try_get<double>("convergence_threshold");
if (maybe_value) {
  double value = *maybe_value;
  // Use the value
}
// end-cell-4

// start-cell-5
// Get all setting keys
auto keys = settings.keys();

// Get the number of settings
size_t count = settings.size();

// Check if settings are empty
bool is_empty = settings.empty();

// Clear all settings
settings.clear();

// Validate that required settings exist
settings.validate_required({"basis_set", "convergence_threshold"});

// Get a setting as a string representation
std::string value_str = settings.get_as_string("convergence_threshold");

// Merge settings from another settings object
Settings other_settings;
settings.merge(other_settings, true);  // true to overwrite existing

// Update an existing setting (throws if key doesn't exist)
settings.update("convergence_threshold", 1.0e-9);

// Get the type name of a setting
std::string type = settings.get_type_name("convergence_threshold");
// end-cell-5

// start-cell-6
// Save settings to JSON file
settings.to_json_file("configuration.settings.json");

// Load settings from JSON file
auto settings_from_json_file =
    Settings::from_json_file("configuration.settings.json");

// Save settings to HDF5 file
settings.to_hdf5("configuration.settings.h5");

// Load settings from HDF5 file
auto settings_from_hdf5 = Settings::from_hdf5("configuration.settings.h5");

// Generic file I/O with specified format
settings.to_file("configuration", "json");
auto settings_from_file = Settings::from_file("configuration", "hdf5");

// Convert to JSON object
auto json_data = settings.to_json();

// Load from JSON object
auto settings_from_json = Settings::from_json(json_data);
// end-cell-6

// start-cell-7
class MySettings : public Settings {
 public:
  MySettings() {
    // Can only call set_default during construction
    set_default("max_iterations", 100);
    set_default("tolerance", 1e-6);
    set_default("method", std::string("default"));
  }
};
// end-cell-7

// start-cell-8
try {
  auto value = settings.get<double>("non_existent_setting");
} catch (const qdk::chemistry::data::SettingNotFound& e) {
  std::cerr << e.what()
            << std::endl;  // "Setting not found: non_existent_setting"
}

try {
  auto value =
      settings.get<int>("string_setting");  // where string_setting is a string
} catch (const qdk::chemistry::data::SettingTypeMismatch& e) {
  std::cerr << e.what() << std::endl;  // "Type mismatch for setting
                                       // 'string_setting'. Expected: int"
}
// end-cell-8
