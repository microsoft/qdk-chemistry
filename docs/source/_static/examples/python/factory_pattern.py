"""Factory pattern usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-scf-localizer
from pathlib import Path
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# Load H2 molecule from XYZ file
structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

# Create a SCF solver using the default implementation
scf_solver = create("scf_solver")

# Create an orbital localizer using a specific implementation
localizer = create("orbital_localizer", "qdk_pipek_mezey")

# Configure the SCF solver and run
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)
# end-cell-scf-localizer
################################################################################

################################################################################
# start-cell-list-algorithms
from qdk_chemistry.algorithms import registry  # noqa: E402

# List all algorithm types and their implementations
all_algorithms = registry.available()
print(all_algorithms)
# Output: {'scf_solver': ['qdk', 'pyscf'], 'orbital_localizer': ['qdk_pipek_mezey', 'pyscf'], ...}

# List implementations for a specific algorithm type
scf_methods = registry.available("scf_solver")
print("Available SCF solvers:", scf_methods)
# Output: ['qdk', 'pyscf']

localizer_methods = registry.available("orbital_localizer")
print("Available localizers:", localizer_methods)
# Output: ['qdk_pipek_mezey', 'pyscf', ...]

# Show default implementations for each algorithm type
defaults = registry.show_default()
print("Defaults:", defaults)
# Output: {'scf_solver': 'qdk', 'orbital_localizer': 'qdk_pipek_mezey', ...}
# end-cell-list-algorithms
################################################################################

################################################################################
# start-cell-inspect-settings
from qdk_chemistry.algorithms import registry  # noqa: E402

# Create a SCF solver and inspect its settings
scf = registry.create("scf_solver", "qdk")

# Print settings as a formatted table
registry.print_settings("scf_solver", "qdk")

# Or iterate over individual settings
for key in scf.settings().keys():
    print(f"{key}: {scf.settings().get(key)}")
# end-cell-inspect-settings
################################################################################
