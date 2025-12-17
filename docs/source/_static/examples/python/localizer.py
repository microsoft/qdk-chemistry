"""Orbital localizer usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# Create a Pipek-Mezey localizer
localizer = create("orbital_localizer", "qdk_pipek_mezey")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure localizer settings
localizer.settings().set("tolerance", 1.0e-6)
localizer.settings().set("max_iterations", 100)

# View available settings
print(f"Localizer settings: {localizer.settings().keys()}")
# end-cell-configure
################################################################################

################################################################################
# start-cell-localize
# Create H2O molecule
coords = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.43052268, 1.10926924], [0.0, -1.43052268, 1.10926924]]
)
symbols = ["O", "H", "H"]
structure = Structure(coords, symbols=symbols)

# Obtain orbitals from SCF
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# Create indices for orbitals to localize
loc_indices = [0, 1, 2, 3]

# Localize the specified orbitals
localized_wfn = localizer.run(wfn, loc_indices, loc_indices)

localized_orbitals = localized_wfn.get_orbitals()
print(localized_orbitals.get_summary())
# end-cell-localize
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("orbital_localizer"))
# ['pyscf_multi', 'qdk_vvhv', 'qdk_mp2_natural_orbitals', 'qdk_pipek_mezey']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-pyscf-multi-example
# Using PySCF's multi-method localizer
localizer = create("orbital_localizer", "pyscf_multi")
localizer.settings().set("method", "foster-boys")

# Localize occupied orbitals
n_occupied = 5
occ_indices = list(range(n_occupied))
localized_orbs = localizer.run(wfn, occ_indices, occ_indices)
# end-cell-pyscf-multi-example
################################################################################
