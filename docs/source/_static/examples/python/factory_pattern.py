"""Factory pattern usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-scf-localizer
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# Create a simple molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Create a SCF solver using the default implementation
scf_solver = create("scf_solver")

# Create an orbital localizer using a specific implementation
localizer = create("orbital_localizer", "qdk_pipek_mezey")

# Configure the SCF solver and run
scf_solver.settings().set("basis_set", "cc-pvdz")
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)
# end-cell-scf-localizer
################################################################################
