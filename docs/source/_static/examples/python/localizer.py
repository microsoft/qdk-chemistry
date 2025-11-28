"""Orbital localizer usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Structure

# First run an SCF calculation to get a wavefunction
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# List available orbital localizer implementations
available_localizers = available("orbital_localizer")
print(f"Available orbital localizers: {available_localizers}")

# Create a Pipek-Mezey localizer
localizer = create("orbital_localizer", "qdk_pipek_mezey")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure localizer settings
localizer.settings().set("tolerance", 1.0e-6)
localizer.settings().set("max_iterations", 100)

# View available settings
print(f"Localizer settings: {localizer.settings().keys()}")
# end-cell-configure
################################################################################

################################################################################
# start-cell-localize
# Create indices for orbitals to localize
# For H2 with sto-3g, we have 2 molecular orbitals
num_mos = wfn.get_orbitals().get_num_molecular_orbitals()
loc_indices = list(range(num_mos))  # Localize all orbitals

# Localize the specified orbitals
# For restricted orbitals, alpha and beta indices must be the same
localized_wfn = localizer.run(wfn, loc_indices, loc_indices)

localized_orbitals = localized_wfn.get_orbitals()
print(f"Localized {num_mos} orbitals")
print(localized_orbitals.get_summary())
# end-cell-localize
################################################################################
