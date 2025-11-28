"""Orbitals and model orbitals usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure, ModelOrbitals

# =============================================================================
# Orbitals usage example
# =============================================================================

# Create H2 molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Obtain orbitals from an SCF calculation
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)
orbitals = wfn.get_orbitals()

# Access orbital coefficients (returns tuple of alpha/beta matrices)
coeffs_alpha, coeffs_beta = orbitals.get_coefficients()

# Access orbital energies (returns tuple of alpha/beta vectors)
energies_alpha, energies_beta = orbitals.get_energies()

# Access atomic orbital overlap matrix
ao_overlap = orbitals.get_overlap_matrix()

# =============================================================================
# Model Orbitals creation example
# =============================================================================

# Set basis set size 
basis_size = 6 
# Set active orbitals
alpha_active = [1,2]
beta_active = [2,3,4]
alpha_inactive = [0,3,4,5]
beta_inactive = [0,1,5]

model_orbitals = ModelOrbitals(basis_size, (alpha_active, beta_active, alpha_inactive, beta_inactive))

# we can then pass this object to a custom Hamiltonian constructor