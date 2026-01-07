"""Serialization examples for QDK Chemistry objects."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-json
import os
from pathlib import Path

import numpy as np

from qdk_chemistry.data import Structure, Hamiltonian, ModelOrbitals

# Load structure from XYZ file (the file uses Angstrom, which is converted to Bohr internally)
structure = Structure.from_xyz_file(Path(__file__).parent / "../data/h2.structure.xyz")

# For demonstration: create a structure with custom masses and charges
# (requires explicit coordinates, here in Bohr)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])  # Bohr
symbols = ["H", "H"]
custom_masses = [1.001, 0.999]
custom_charges = [0.9, 1.1]
structure_custom = Structure(
    coords, symbols=symbols, masses=custom_masses, nuclear_charges=custom_charges
)

# Serialize to JSON object
structure_data = structure_custom.to_json()

# Deserialize from JSON object
# "Structure" is the data type to de-serialize into (will throw, if it doesn't match)
structure_from_json = Structure.from_json(structure_data)

# Write to json file
tmpfile = "example.structure.json"
structure.to_json_file(tmpfile)

# Read from json file
structure_from_json_file = Structure.from_json_file(tmpfile)

os.remove(tmpfile)
# end-cell-json
################################################################################

################################################################################
# start-cell-hdf5
# Hamiltonian data class example
# Create dummy data for Hamiltonian class
one_body = np.identity(2)
two_body = 2 * np.ones((16,))
orbitals = ModelOrbitals(2, True)  # 2 orbitals, restricted
core_energy = 1.5
inactive_fock = np.zeros((0, 0))

h2_example = Hamiltonian(one_body, two_body, orbitals, core_energy, inactive_fock)

h2_example.to_hdf5_file("h2_example.hamiltonian.h5")

# Deserialize from HDF5 file
h2_example_from_hdf5_file = Hamiltonian.from_hdf5_file("h2_example.hamiltonian.h5")
os.remove("h2_example.hamiltonian.h5")
# end-cell-hdf5
################################################################################
