"""Complete SCF workflow example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# =============================================================================
# Default SCF solver
# =============================================================================

# Create a Structure 
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Create the default ScfSolver instance (HF)
scf_solver = create("scf_solver")

# Set the basis set
scf_solver.settings().set("basis_set", "sto-3g")

# Run the SCF calculation
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# =============================================================================
# DFT SCF solver
# =============================================================================

# Create the default ScfSolver instance
scf_solver_dft = create("scf_solver")

# Set DFT method via functional name
scf_solver_dft.settings().set("method", "b3lyp")

# Run the SCF calculation
E_dft, wfn_dft = scf_solver_dft.run(structure, charge=0, spin_multiplicity=1)

# =============================================================================
# Pyscf SCF solver
# =============================================================================

# This one is python-only
scf_solver_pyscf = create("scf_solver", "pyscf")

# Set the basis set
scf_solver_pyscf.settings().set("basis_set", "sto-3g")

E_pyscf, wfn_pyscf = scf_solver_pyscf.run(structure, charge=0, spin_multiplicity=1)