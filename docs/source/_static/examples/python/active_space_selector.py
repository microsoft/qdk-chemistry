"""Active space selection examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create the default ActiveSpaceSelector instance
active_space_selector = create("active_space_selector")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the selector using the settings interface
# Set the number of electrons and orbitals for the active space
active_space_selector.settings().set("num_active_electrons", 4)
active_space_selector.settings().set("num_active_orbitals", 4)

# end-cell-configure
################################################################################

################################################################################
# start-cell-run
import numpy as np  # noqa: E402
from qdk_chemistry.data import Structure  # noqa: E402

# Create a molecular structure (water molecule)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8897], [1.7802, 0.0, -0.4738]])
structure = Structure(coords, symbols=["O", "H", "H"])

# First, run SCF to get molecular orbitals
scf_solver = create("scf_solver")
scf_energy, scf_wavefunction = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="6-31g"
)

# Run active space selection
active_wavefunction = active_space_selector.run(scf_wavefunction)
active_orbitals = active_wavefunction.get_orbitals()

print(f"SCF Energy: {scf_energy:.10f} Hartree")
print(f"Active orbitals summary:\n{active_orbitals.get_summary()}")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("active_space_selector"))
# ['pyscf_avas', 'qdk_occupation', 'qdk_autocas_eos', 'qdk_autocas', 'qdk_valence']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-avas-example
from qdk_chemistry.algorithms import create  # noqa: E402

avas = create("active_space_selector", "pyscf_avas")
avas.settings().set("ao_labels", ["Fe 3d", "Fe 4d"])
avas.settings().set("canonicalize", True)

active_wavefunction = avas.run(scf_wavefunction)
# end-cell-avas-example
################################################################################
