"""Settings configuration examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-get-settings
import os
import tempfile
import qdk_chemistry
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Settings

# Create an algorithm
scf_solver = create("scf_solver")

# Get the settings object for that algorithm
settings = scf_solver.settings()

# Get a parameter
max_iter = settings.get("max_iterations")
print(f"Max iterations: {max_iter}")

# end-cell-get-settings
################################################################################

################################################################################
# start-cell-set-settings
# Create an algorithm
scf_solver = create("scf_solver")

# Get the settings object
settings = scf_solver.settings()

# Set a integer value
settings.set("max_iterations", 100)

# Set a string value
settings.set("method", "B3LYP")

# Set a numeric value
settings.set("convergence_threshold", 1.0e-8)
# end-cell-set-settings
################################################################################

################################################################################
# start-cell-get-settings
# View all settings
# List available implementations for each algorithm type
for algorithm_name in available():
    print(f"{algorithm_name} has methods:")
    for method_name in available(algorithm_name):
        print(f"  {method_name} has settings:")
        method_ = create(algorithm_name, method_name)
        settings_ = method_.settings()
        for key, value in settings_.items():
            print(f"    {key}: {value}")
# end-cell-get-settings
################################################################################

################################################################################
# start-cell-misc-settings
# Check if a setting exists
if settings.has("method"):
    # Use the setting
    print(f"Method is selected: {settings.get('method')}")

# Check if a setting exists (Python duck typing, no type check needed)
if settings.has("convergence_threshold"):
    # Use the setting
    print(f"Convergence threshold: {settings.get('convergence_threshold')}")

# List the available settings
print("Available settings:", settings)

# Try to get a value (Python uses get_or_default or try/except)
try:
    value = settings.get("convergence_threshold")
    # Use the value
    print(f"Got convergence threshold: {value}")
except KeyError:
    print("convergence_threshold not found")

# Check if settings exist
if settings.has("max_iterations"):
    print("max_iterations setting exists")

# Get with default fallback
custom_param = settings.get_or_default("my_custom_param", 42)
print(f"Custom parameter (with default): {custom_param}")

# Get all setting keys
keys = settings.keys()

# Get the number of settings
count = settings.size()

# Check if settings are empty
is_empty = settings.empty()

# Clear all settings
settings = scf_solver.settings()  # Re-initialize to clear

# Validate that required settings exist
settings.validate_required(["convergence_threshold"])

# Get a setting as a string representation
value_str = settings.get_as_string("convergence_threshold")

# Update an existing setting (throws if key doesn't exist)
settings.update("convergence_threshold", 1.0e-9)

# Get the type name of a setting
type_name = settings.get_type_name("convergence_threshold")
# end-cell-misc-settings
################################################################################

################################################################################
# start-cell-serialization
# Serialization
tmpdir = tempfile.mkdtemp()
os.chdir(tmpdir)
# Save settings to JSON file
settings.to_json_file("configuration.settings.json")

# Load settings from JSON file
settings_from_json_file = settings.from_json_file("configuration.settings.json")

# Generic file I/O with JSON format
settings.to_file("configuration.settings.json", "json")
settings_from_file = settings.from_file("configuration.settings.json", "json")

# Convert to JSON object
json_data = settings.to_json()

# Load from JSON object
settings_from_json = settings.from_json(json_data)
# end-cell-serialization
################################################################################


################################################################################
# start-cell-extend-settings
## Extending Settings class
class MySettings(Settings):
    def __init__(self):
        super().__init__()
        # Set default values during initialization
        self.set_default("max_iterations", 100)
        self.set_default("convergence_threshold", 1e-6)
        self.set_default("method", "default")


# end-cell-extend-settings
################################################################################

################################################################################
# start-cell-settings-errors
# Error handling example
try:
    value = settings.get("non_existent_setting")
except qdk_chemistry.data.SettingNotFound as e:
    print(e)  # "Setting not found: non_existent_setting"
    # Don't exit; use a fallback and continue execution
    value = settings.get_or_default("non_existent_setting", None)
# end-cell-settings-errors
################################################################################
