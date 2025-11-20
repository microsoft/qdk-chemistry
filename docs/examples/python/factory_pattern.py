"""Factory pattern usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.data import Structure

# Create a simple molecule for testing
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Create algorithms using the factory
# start-cell-1
from qdk_chemistry.algorithms import create

# Create default implementation
scf_solver = create("scf_solver")

# Create specific implementation by name
localizer = create("orbital_localizer", "qdk_pipek_mezey")

# Configure and use the instance
scf_solver.settings().set("basis_set", "cc-pvdz")
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)
# end-cell-1

# List available algorithms
# print("Available SCF solvers:", available("scf_solver"))
# print("Available Hamiltonian constructors:", available("hamiltonian_constructor"))
# print("Available orbital localizers:", available("orbital_localizer"))

# # Example of creating with custom settings
# scf_solver = create("scf_solver")
# scf_solver.settings().set("max_iterations", 100)
# scf_solver.settings().set("basis_set", "sto-3g")
# print(f"Set max_iterations to: {scf_solver.settings().get('max_iterations')}")

# # Run the SCF solver with the molecule (requires resources directory)
# # E_scf, wfn = scf_solver.run(molecule, charge=0, spin_multiplicity=1)
# # print(f"SCF Energy: {E_scf}")
