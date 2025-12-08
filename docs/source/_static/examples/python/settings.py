"""Settings configuration examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-get-settings
from qdk_chemistry.algorithms import create
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
# Get the settings object
settings = scf_solver.settings()

# Set an integer value
settings.set("max_iterations", 100)

# Set a string value
settings.set("basis_set", "def2-tzvp")

# Set a numeric value
settings.set("convergence_threshold", 1.0e-8)
# end-cell-set-settings
################################################################################

################################################################################
# start-cell-factory-settings
from qdk_chemistry.algorithms import create  # noqa E402

# Pass settings directly to create() as keyword arguments
scf_solver = create(
    "scf_solver",
    max_iterations=100,
    basis_set="def2-tzvp",
    convergence_threshold=1.0e-8,
)

# This is equivalent to:
# scf_solver = create("scf_solver")
# scf_solver.settings().set("max_iterations", 100)
# scf_solver.settings().set("basis_set", "def2-tzvp")
# scf_solver.settings().set("convergence_threshold", 1.0e-8)
# end-cell-factory-settings
################################################################################

################################################################################
# start-cell-misc-settings
# Check if a setting exists
if settings.has("basis_set"):
    print(f"Basis set is configured: {settings.get('basis_set')}")

# Get with default fallback
custom_param = settings.get_or_default("my_custom_param", 42)
print(f"Custom parameter (with default): {custom_param}")

# Get all setting keys
keys = settings.keys()

# Get the number of settings
count = settings.size()

# Check if settings are empty
is_empty = settings.empty()

# Validate that required settings exist
settings.validate_required(["basis_set", "convergence_threshold"])

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
import os  # noqa E402
import tempfile  # noqa E402

tmpdir = tempfile.mkdtemp()
os.chdir(tmpdir)

# Save settings to JSON file
settings.to_json_file("configuration.settings.json")

# Load settings from JSON file
settings_from_json_file = Settings.from_json_file("configuration.settings.json")

# Generic file I/O with specified format
settings.to_file("configuration.settings.json", "json")
settings_from_file = Settings.from_file("configuration.settings.json", "json")

# Convert to JSON string
json_data = settings.to_json()

# Load from JSON string
settings_from_json = Settings.from_json(json_data)
# end-cell-serialization
################################################################################


################################################################################
# start-cell-extend-settings
class MySettings(Settings):
    def __init__(self):
        super().__init__()
        # Set default values during construction
        self._set_default("max_iterations", "int", 100)
        self._set_default("convergence_threshold", "double", 1e-6)
        self._set_default("method", "string", "default")


# end-cell-extend-settings
################################################################################

################################################################################
# start-cell-settings-errors
import qdk_chemistry  # noqa E402

# Error handling example
try:
    value = settings.get("non_existent_setting")
except qdk_chemistry.data.SettingNotFound as e:
    print(e)  # "Setting not found: non_existent_setting"
    # Use a fallback and continue execution
    value = settings.get_or_default("non_existent_setting", 0.0)
# end-cell-settings-errors
################################################################################
