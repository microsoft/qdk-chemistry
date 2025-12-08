// Settings usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

#include <iostream>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/settings.hpp>

using namespace qdk::chemistry;

// --------------------------------------------------------------------------------------------
// start-cell-get-settings
// Create an algorithm
auto scf_solver = algorithms::ScfSolverFactory::create();

// Get the settings object for that algorithm
auto& settings = scf_solver->settings();

// Get a parameter
auto max_iter = settings.get<int64_t>("max_iterations");
std::cout << "Max iterations: " << max_iter << std::endl;
// end-cell-get-settings
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-set-settings
// Get the settings object
auto& settings = scf_solver->settings();

// Set an integer value
settings.set("max_iterations", 100);

// Set a string value
settings.set("basis_set", "def2-tzvp");

// Set a numeric value
settings.set("convergence_threshold", 1.0e-8);
// end-cell-set-settings
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-misc-settings
// Check if a setting exists
if (settings.has("basis_set")) {
  std::cout << "Basis set is configured: "
            << settings.get<std::string>("basis_set") << std::endl;
}

// Get with default fallback
auto custom_param = settings.get_or_default<int>("my_custom_param", 42);
std::cout << "Custom parameter (with default): " << custom_param << std::endl;

// Get all setting keys
auto keys = settings.keys();

// Get the number of settings
size_t count = settings.size();

// Check if settings are empty
bool is_empty = settings.empty();

// Validate that required settings exist
settings.validate_required({"basis_set", "convergence_threshold"});

// Get a setting as a string representation
std::string value_str = settings.get_as_string("convergence_threshold");

// Update an existing setting (throws if key doesn't exist)
settings.update("convergence_threshold", 1.0e-9);

// Get the type name of a setting
std::string type_name = settings.get_type_name("convergence_threshold");
// end-cell-misc-settings
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-serialization
// Save settings to JSON file
settings.to_json_file("configuration.settings.json");

// Load settings from JSON file
auto settings_from_json_file =
    data::Settings::from_json_file("configuration.settings.json");

// Generic file I/O with specified format
settings.to_file("configuration.settings.json", "json");
auto settings_from_file =
    data::Settings::from_file("configuration.settings.json", "json");

// Convert to JSON object
auto json_data = settings.to_json();

// Load from JSON object
auto settings_from_json = data::Settings::from_json(json_data);
// end-cell-serialization
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-extend-settings
class MySettings : public data::Settings {
 public:
  MySettings() {
    // Set default values during construction
    set_default("max_iterations", 100);
    set_default("convergence_threshold", 1e-6);
    set_default("method", std::string("default"));
  }
};
// end-cell-extend-settings
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-settings-errors
// Error handling example
try {
  auto value = settings.get<double>("non_existent_setting");
} catch (const data::SettingNotFound& e) {
  std::cerr << e.what()
            << std::endl;  // "Setting not found: non_existent_setting"
  // Use a fallback and continue execution
  auto value = settings.get_or_default<double>("non_existent_setting", 0.0);
}
// end-cell-settings-errors
// --------------------------------------------------------------------------------------------
