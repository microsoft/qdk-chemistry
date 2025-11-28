"""Complete SCF workflow example with settings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Structure

# Create a Structure (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# start-cell-1
# List available SCF solver implementations
available_solvers = available("scf_solver")
print(f"Available SCF solvers: {available_solvers}")

# Create the default ScfSolver instance
scf_solver = create("scf_solver")
# end-cell-1

# start-cell-2
# Configure the SCF solver using the settings interface
scf_solver.settings().set("basis_set", "sto-3g")
scf_solver.settings().set("method", "hf")
scf_solver.settings().set("max_iterations", 100)
scf_solver.settings().set("tolerance", 1.0e-8)
# end-cell-2

# start-cell-3
# Run the SCF calculation
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)
scf_orbitals = wfn.get_orbitals()

print(f"SCF Energy: {E_scf:.10f} Hartree")
print(f"Number of molecular orbitals: {scf_orbitals.get_num_molecular_orbitals()}")
print(f"Is restricted: {scf_orbitals.is_restricted()}")
# end-cell-3
