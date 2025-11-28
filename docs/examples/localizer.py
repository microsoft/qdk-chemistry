"""Orbital localizer usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# =============================================================================
# Basic orbital localization example
# =============================================================================

# Create H2O molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.757, 0.587], [0.0, -0.757, 0.587]])
symbols = ["O", "H", "H"]
structure = Structure(coords, symbols=symbols)

# Obtain orbitals from an SCF calculation
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
E_scf, wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)
orbitals = wavefunction.get_orbitals()

# Create Pipek-Mezey localizer
pm_localizer = create("orbital_localizer", "qdk_pipek_mezey")

# Localize occupied orbitals (indices 0, 1, 2, 3, 4 for H2O with STO-3G)
occupied_indices = [0, 1, 2, 3, 4]  # 5 occupied orbitals for H2O
localized_orbitals = pm_localizer.run(orbitals, occupied_indices, occupied_indices)
