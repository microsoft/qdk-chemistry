"""Basic Structure creation and manipulation example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.data import Structure

# Create the Structure manually (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
elements = ["H", "H"]
structure = Structure(coords, elements)

print(f"Created structure with {structure.get_num_atoms()} atoms")
print(f"Elements: {structure.get_elements()}")

# Load from XYZ file
# structure = Structure.from_xyz_file("molecule.structure.xyz")  # Required .structure.xyz suffix

# Load from JSON file
# structure = Structure.from_json_file("molecule.structure.json")  # Required .structure.json suffix

# Add an atom with coordinates and element
# structure.add_atom([1.0, 0.0, 0.0], "O")  # Add an oxygen atom

# Remove an atom
# structure.remove_atom(2)  # Remove the third atom

# Create a structure with coordinates in Angstroms (default)
# structure_angstrom = Structure(coords_matrix, elements_list)

# Explicitly specify units as Angstrom
# structure_angstrom_explicit = Structure(coords_matrix, elements_list, "angstrom")
