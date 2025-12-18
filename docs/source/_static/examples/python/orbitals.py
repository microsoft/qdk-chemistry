"""Orbitals and model orbitals usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure, ModelOrbitals

################################################################################
# start-cell-create
# Create H2 molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Obtain orbitals from an SCF calculation
scf_solver = create("scf_solver")
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
orbitals = wfn.get_orbitals()
# end-cell-create
################################################################################

################################################################################
# start-cell-model-orbitals-create
# Set basis set size
basis_size = 6

# Set active orbitals
alpha_active = [1, 2]
beta_active = [2, 3, 4]
alpha_inactive = [0, 3, 4, 5]
beta_inactive = [0, 1, 5]

model_orbitals = ModelOrbitals(
    basis_size, (alpha_active, beta_active, alpha_inactive, beta_inactive)
)

# We can then pass this object to a custom Hamiltonian constructor
# end-cell-model-orbitals-create
################################################################################

################################################################################
# start-cell-access
# Access orbital coefficients (alpha, beta)
coeffs_alpha, coeffs_beta = orbitals.get_coefficients()

# Access orbital energies
energies_alpha, energies_beta = orbitals.get_energies()
# Get active space indices
active_indices_alpha, active_indices_beta = orbitals.get_active_space_indices()

# Access atomic orbital overlap matrix
ao_overlap = orbitals.get_overlap_matrix()

# Access basis set information
basis_set = orbitals.get_basis_set()

# Check calculation type
is_restricted = orbitals.is_restricted()

# Get size information
num_molecular_orbitals = orbitals.get_num_molecular_orbitals()
num_atomic_orbitals = orbitals.get_num_atomic_orbitals()

summary = orbitals.get_summary()
print(summary)
# end-cell-access
################################################################################
