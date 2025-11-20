"""Hamiltonian usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------
import tempfile
from pathlib import Path

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Hamiltonian, Structure

coords = np.array(
    [
        [0.000000, 0.000000, 0.000000],
        [1.43233673, 0.000000, 0.000000],
        [-0.44604614, 1.09629126, 0.000000],
    ]
)
structure = Structure(coords, ["O", "H", "H"])
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)
orbitals = wfn.get_orbitals()

# start-cell-1
# Create a Hamiltonian constructor
hamiltonian_constructor = create("hamiltonian_constructor")

# Construct the Hamiltonian from orbitals
hamiltonian = hamiltonian_constructor.run(orbitals)
# end-cell-1
i, j, k, l = 0, 1, 2, 3  # Example indices for two-body element access
# start-cell-2
# Access one-electron integrals
h1 = hamiltonian.get_one_body_integrals()

# Access two-electron integrals
h2 = hamiltonian.get_two_body_integrals()

# Access a specific two-electron integral <ij|kl>
element = hamiltonian.get_two_body_element(i, j, k, l)

# Get core energy (nuclear repulsion + inactive orbital energy)
core_energy = hamiltonian.get_core_energy()

# Get orbital data
orbitals = hamiltonian.get_orbitals()
# end-cell-2

temp_dir = Path(tempfile.gettempdir())
# start-cell-3
# Serialize to JSON file
hamiltonian.to_json_file(temp_dir / "molecule.hamiltonian.json")

# Deserialize from JSON file
hamiltonian_from_json_file = Hamiltonian.from_json_file(
    temp_dir / "molecule.hamiltonian.json"
)

# Serialize to HDF5 file (TODO: bugs to be fixed)
# hamiltonian.to_hdf5_file("molecule.hamiltonian.h5")
# hamiltonian_from_hdf5_file = Hamiltonian.from_hdf5_file("molecule.hamiltonian.h5")

# Generic file I/O based on type parameter
hamiltonian.to_file(temp_dir / "molecule.hamiltonian.json", "json")
hamiltonian_loaded = Hamiltonian.from_file(
    temp_dir / "molecule.hamiltonian.json", "json"
)
# end-cell-3

# start-cell-4
# Check if specific components are available
has_one_body = hamiltonian.has_one_body_integrals()
has_two_body = hamiltonian.has_two_body_integrals()
has_orbitals = hamiltonian.has_orbitals()
# end-cell-4
