"""Factory pattern usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-scf-localizer
import numpy as np
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.data import Structure

# Create a simple molecule for testing
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Create default implementation
print(f"Available SCF solver methods: {available('scf_solver')}")
scf_solver = create("scf_solver", "qdk")

# Create specific implementation by name
print(f"Available orbital localizer methods: {available('orbital_localizer')}")
localizer = create("orbital_localizer", "qdk_pipek_mezey")

# Configure and use the instance
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)
# end-cell-scf-localizer
################################################################################
