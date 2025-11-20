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

# Create an SCF solver that uses the QDK/Chemistry library as solver
scf_solver = create("scf_solver")

# Configure it using the standard settings interface
settings = scf_solver.settings()
settings.set("basis_set", "cc-pvdz")
settings.set("method", "hf")

# Run calculation
scf_solver.run(structure, charge=0, spin_multiplicity=1)
# end-cell-1


# start-cell-2
from qdk_chemistry.algorithms import available

# Get a list of available SCF solver implementations
available_solvers = available("scf_solver")
print(f"Available SCF solvers: {available_solvers}")
# end-cell-2


