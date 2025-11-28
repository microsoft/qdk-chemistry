"""Basic Wavefunction creation and manipulation example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    Structure,
)

# =============================================================================
# Typical wavefunction creation via SCF calculation
# =============================================================================

# Create H2 molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Run SCF calculation to get wavefunction
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
E_scf, wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# Access wavefunction properties
orbitals = wavefunction.get_orbitals()
container_type = (
    wavefunction.get_container_type()
)  # Returns "sd" for Slater determinant
wf_type = wavefunction.get_type()  # WavefunctionType.SelfDual

# Get determinant information
determinants = wavefunction.get_active_determinants()
coefficients = wavefunction.get_coefficients()
num_dets = wavefunction.size()

# Get electron counts
n_alpha_total, n_beta_total = wavefunction.get_total_num_electrons()
n_alpha_active, n_beta_active = wavefunction.get_active_num_electrons()

# Get RDMs for active orbitals
one_rdm = wavefunction.get_active_one_rdm_spin_traced()
two_rdm = wavefunction.get_active_two_rdm_spin_traced()
