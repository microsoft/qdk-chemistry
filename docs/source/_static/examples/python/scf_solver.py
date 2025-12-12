"""Complete SCF workflow example with settings."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# Create the default ScfSolver instance
scf_solver = create("scf_solver")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the SCF solver using the settings interface
# Note that the following line is optional, since hf is the default method
scf_solver.settings().set("method", "hf")

# end-cell-configure
################################################################################

################################################################################
# start-cell-run
# Specify a structure
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Run scf
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="def2-tzvpp"
)
scf_orbitals = wfn.get_orbitals()

print(f"SCF Energy: {E_scf:.10f} Hartree")
# end-cell-run
################################################################################
