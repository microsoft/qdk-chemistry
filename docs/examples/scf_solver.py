"""Complete SCF workflow example with settings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# Create a Structure (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Create the default ScfSolver instance
scf_solver = create("scf_solver")

# Run the SCF calculation
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_information="sto-3g"
)
scf_orbitals = wfn.get_orbitals()

print(f"SCF Energy: {E_scf}")
print(f"Number of molecular orbitals: {scf_orbitals.get_num_molecular_orbitals()}")
