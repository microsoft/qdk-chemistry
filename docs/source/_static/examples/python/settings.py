"""Settings configuration examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-discover-settings
from qdk_chemistry.algorithms import available, create, inspect_settings, print_settings

# List all algorithm types and their implementations
for algo_type, implementations in available().items():
    print(f"{algo_type}: {implementations}")

# Display a formatted settings table for a specific implementation
print_settings("scf_solver", "qdk")

# Inspect settings programmatically
for name, type_name, default, description, limits in inspect_settings(
    "scf_solver", "qdk"
):
    print(f"{name} ({type_name}): {default}")

# Create an instance and iterate over its settings
scf = create("scf_solver")
for key, value in scf.settings().items():
    print(f"  {key}: {value}")
# end-cell-discover-settings
################################################################################

################################################################################
# start-cell-get-settings
from qdk_chemistry.algorithms import create  # noqa: E402
from qdk_chemistry.data import Settings  # noqa: E402

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
settings.set("method", "B3LYP")

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
    convergence_threshold=1.0e-8,
)

# This is equivalent to:
# scf_solver = create("scf_solver")
# scf_solver.settings().set("max_iterations", 100)
# scf_solver.settings().set("convergence_threshold", 1.0e-8)
# end-cell-factory-settings
################################################################################

################################################################################
# start-cell-misc-settings
# Check if a setting exists
if settings.has("method"):
    # Use the setting
    print(f"Method is selected: {settings.get('method')}")

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
import numpy as np  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

################################################################################
# start-cell-settings-locked
scf = create("scf_solver")
scf.settings().set("method", "hf")
energy, wfn = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")  # type: ignore[name-defined]  # noqa: F821

# Settings are now locked - this raises SettingsAreLocked:
# scf.settings().set("method", "b3lyp")

# Create a new instance for different settings
scf2 = create("scf_solver")
scf2.settings().set("method", "b3lyp")
energy2, wfn2 = scf2.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)  # type: ignore[name-defined]  # noqa: F821
# end-cell-settings-locked
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

################################################################################
# start-cell-get-expected-type
expected_type = settings.get_expected_python_type("convergence_threshold")
print(expected_type)  # "float"

expected_type = settings.get_expected_python_type("method")
print(expected_type)  # "str"
# end-cell-get-expected-type
################################################################################

################################################################################
# start-cell-inspect-constraints
if settings.has_limits("max_iterations"):
    limits = settings.get_limits("max_iterations")
    # Returns (min, max) tuple for bounds, or list for allowed values
    print(f"Allowed range: {limits}")  # e.g., (1, 1000)

if settings.has_limits("method"):
    allowed = settings.get_limits("method")
    print(f"Allowed values: {allowed}")  # e.g., ['hf', 'dft']
# end-cell-inspect-constraints
################################################################################
