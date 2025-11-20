"""Basic Structure creation and manipulation example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.data import Structure

# start-cell-1
# Create the Structure manually (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
elements = ["H", "H"]
structure = Structure(coords, elements)

print(f"Created structure with {structure.get_num_atoms()} atoms")
print(f"Elements: {structure.get_elements()}")
# end-cell-1

# start-cell-2
# Load from XYZ file
# structure = Structure.from_xyz_file("molecule.structure.xyz")  # Required .structure.xyz suffix

# Load from JSON file
# structure = Structure.from_json_file("molecule.structure.json")  # Required .structure.json suffix
# end-cell-2

# start-cell-3
# Get coordinates of a specific atom
# coords = structure.get_atom_coordinates(0)  # First atom

# Get element of a specific atom
# element = structure.get_atom_element(0)  # First atom

# Get all coordinates as a matrix
# all_coords = structure.get_coordinates()

# Get all elements as a list
# all_elements = structure.get_elements()
# end-cell-3

# start-cell-5
# Add an atom with coordinates and element
# structure.add_atom([1.0, 0.0, 0.0], "O")  # Add an oxygen atom

# Remove an atom
# structure.remove_atom(2)  # Remove the third atom
# end-cell-5
