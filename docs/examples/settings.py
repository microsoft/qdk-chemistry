"""Settings configuration examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import tempfile

import qdk_chemistry
from qdk_chemistry._core.data import Settings
from qdk_chemistry.algorithms import create

# Create an algorithm
scf_solver = create("scf_solver")

# Get the settings object
settings = scf_solver.settings()

# Set a parameter
settings.set("max_iterations", 100)

# Get a parameter
max_iter = settings.get("max_iterations")

# Set various parameter types
# Set a string value
settings.set("basis_set", "def2-tzvp")

# Set a numeric value
settings.set("tolerance", 1.0e-8)

# Set a boolean value
# settings.set("density_fitting", True)

# Set an array value
# settings.set("active_orbitals", [4, 5, 6, 7])

# Get various parameter types
# Get a string value
basis = settings.get("basis_set")

# Get a numeric value
threshold = settings.get("tolerance")

# Get a boolean value
# use_df = settings.get("density_fitting")

# Get an array value
# active_orbitals = settings.get("active_orbitals")

# Get a value with default fallback
max_iter_with_default = settings.get_or_default("max_iterations", 100)


print(f"Max iterations: {max_iter}")

# Check if a setting exists
if settings.has("basis_set"):
    # Use the setting
    print(f"Basis set is configured: {settings.get('basis_set')}")

# Check if a setting exists (Python duck typing, no type check needed)
if settings.has("tolerance"):
    # Use the setting
    print(f"Convergence threshold: {settings.get('tolerance')}")

# Try to get a value (Python uses get_or_default or try/except)
try:
    value = settings.get("tolerance")
    # Use the value
    print(f"Got convergence threshold: {value}")
except KeyError:
    print("tolerance not found")

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
settings.validate_required(["basis_set", "tolerance"])

# Get a setting as a string representation
value_str = settings.get_as_string("tolerance")

# Update an existing setting (throws if key doesn't exist)
settings.update("tolerance", 1.0e-9)

# Get the type name of a setting
type_name = settings.get_type_name("tolerance")

# Serialization
tmpdir = tempfile.mkdtemp()
os.chdir(tmpdir)
# Save settings to JSON file
settings.to_json_file("configuration.settings.json")

# Load settings from JSON file
settings_from_json_file = settings.from_json_file("configuration.settings.json")

# TODO: HDF5 serialization has bugs
# settings.to_hdf5_file("configuration.settings.h5")
# settings_from_hdf5 = settings.from_hdf5_file("configuration.settings.h5")

# Generic file I/O with JSON format
settings.to_file("configuration.settings.json", "json")
settings_from_file = settings.from_file("configuration.settings.json", "json")

# Convert to JSON object
json_data = settings.to_json()

# Load from JSON object
settings_from_json = settings.from_json(json_data)


## Extending Settings class
class MySettings(Settings):
    def __init__(self):
        super().__init__()
        # Set default values during initialization
        self.set_default("max_iterations", 100)
        self.set_default("tolerance", 1e-6)
        self.set_default("method", "default")


# Error handling example
try:
    value = settings.get("non_existent_setting")
except qdk_chemistry._core.data.SettingNotFound as e:
    print(e)  # "Setting not found: non_existent_setting"
    # Don't exit; use a fallback and continue execution
    value = settings.get_or_default("non_existent_setting", None)
