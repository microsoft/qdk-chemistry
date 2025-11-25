"""Example demonstrating algorithm interface usage."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.data import Structure

coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# start-cell-1
from qdk_chemistry.algorithms import create

# Create an SCF solver using the factory
scf_solver = create("scf_solver")

# Configure it using the standard settings interface
scf_solver.settings().set("basis_set", "cc-pvdz")
scf_solver.settings().set("method", "hf")

# Run calculation - returns (energy, wavefunction)
energy, wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)
orbitals = wavefunction.get_orbitals()

print(f"SCF Energy: {energy:.10f} Hartree")
# end-cell-1


# start-cell-2
from qdk_chemistry.algorithms import available

# List available implementations for each algorithm type
print("Available SCF solvers:", available("scf_solver"))
print("Available Hamiltonian constructors:", available("hamiltonian_constructor"))
print("Available orbital localizers:", available("orbital_localizer"))
print("Available MC calculators:", available("multi_configuration_calculator"))
# end-cell-2


# start-cell-4
# All algorithms use a consistent settings interface
scf = create("scf_solver")

# Set general options that work across implementations
scf.settings().set("basis_set", "sto-3g")
scf.settings().set("max_iterations", 100)
scf.settings().set("tolerance", 1.0e-8)

# Query available settings for an algorithm
print(f"SCF settings: {scf.settings().keys()}")

# Get a setting value
max_iter = scf.settings().get("max_iterations")
print(f"Max iterations: {max_iter}")
# end-cell-4


