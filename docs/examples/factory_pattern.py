"""Factory pattern usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Structure

# Create a simple molecule for testing
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
molecule = Structure(coords, ["H", "H"])

# Create algorithms using the factory
scf_solver = create("scf_solver")
hamiltonian_constructor = create("hamiltonian_constructor")
orbital_localizer = create("orbital_localizer")

# List available algorithms
print("Available SCF solvers:", available("scf_solver"))
print("Available Hamiltonian constructors:", available("hamiltonian_constructor"))
print("Available orbital localizers:", available("orbital_localizer"))

# Example of creating with custom settings
scf_solver = create("scf_solver")
scf_solver.settings().set("max_iterations", 100)
scf_solver.settings().set("basis_set", "sto-3g")
print(f"Set max_iterations to: {scf_solver.settings().get('max_iterations')}")

# Run the SCF solver with the molecule (requires resources directory)
# E_scf, wfn = scf_solver.run(molecule, charge=0, spin_multiplicity=1)
# print(f"SCF Energy: {E_scf}")
