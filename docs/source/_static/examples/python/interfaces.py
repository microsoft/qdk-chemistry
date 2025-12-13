"""Example demonstrating algorithm interface usage."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-scf
import numpy as np
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Structure

coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Create a SCF solver using the factory
scf_solver = create("scf_solver", "pyscf")

# Configure it using the standard settings interface
scf_solver.settings().set("basis_set", "cc-pvdz")
scf_solver.settings().set("method", "hf")

# Run calculation - returns (energy, wavefunction)
energy, wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)
orbitals = wavefunction.get_orbitals()

print(f"SCF Energy: {energy:.10f} Hartree")
# end-cell-scf
################################################################################

################################################################################
# start-cell-list-methods
# List available implementations for each algorithm type
for algorithm_name in available():
    print(f"{algorithm_name} has methods:")
    for method_name in available(algorithm_name):
        print(f"  {method_name} has settings:")
        method = create(algorithm_name, method_name)
        settings = method.settings()
        for key, value in settings.items():
            print(f"    {key}: {value}")
# end-cell-list-methods
################################################################################

################################################################################
# start-cell-discover-implementations
from qdk_chemistry.algorithms import ScfSolver  # noqa: E402

# List all registered SCF solver implementations
print(ScfSolver.available())  # ['qdk', 'pyscf']

# Create a specific implementation
solver = ScfSolver.create("pyscf")

# Inspect available settings
solver.print_settings()
# end-cell-discover-implementations
################################################################################

################################################################################
# start-cell-settings
# All algorithms use a consistent settings interface
scf = create("scf_solver")

# Set general options that work across implementations
scf.settings().set("basis_set", "sto-3g")
scf.settings().set("max_iterations", 100)
scf.settings().set("convergence_threshold", 1e-7)

# Query available settings for an algorithm
print(f"SCF settings: {scf.settings().keys()}")

# Get a setting value
max_iter = scf.settings().get("max_iterations")
print(f"Max iterations: {max_iter}")
# end-cell-settings
################################################################################
