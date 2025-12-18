"""Active space selection examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry import algorithms, data

# Create a molecular structure (water molecule)
coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8897], [1.7802, 0.0, -0.4738]]
structure = data.Structure(coords, ["O", "H", "H"])

# First, run SCF to get molecular orbitals
scf_solver = algorithms.create("scf_solver")
scf_energy, scf_wavefunction = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="6-31g"
)

# Create an active space selector using the default implementation
active_space_selector = algorithms.create("active_space_selector")
print(f"Default active space selector: {active_space_selector.name()}")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the selector for valence-based selection
# Select 4 electrons in 4 orbitals (oxygen 2p and bonding/antibonding combinations)
valence_selector = algorithms.create("active_space_selector", "qdk_valence")
valence_selector.settings().set("num_active_electrons", 4)
valence_selector.settings().set("num_active_orbitals", 4)

# Alternative: occupation-based automatic selection
occupation_selector = algorithms.create("active_space_selector", "qdk_occupation")
occupation_selector.settings().set("occupation_threshold", 0.1)

# Alternative: entropy-based automatic selection
autocas_selector = algorithms.create("active_space_selector", "qdk_autocas")
autocas_selector.settings().set("min_plateau_size", 2)
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
# Run active space selection with valence selector
active_wavefunction = valence_selector.run(scf_wavefunction)
active_orbitals = active_wavefunction.get_orbitals()

print(f"Active orbitals summary:\n{active_orbitals.get_summary()}")
# The active space can now be used for multireference calculations

# end-cell-run
################################################################################
