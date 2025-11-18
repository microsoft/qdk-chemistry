"""Example showing design principles: immutability and data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.data import Structure

# Data classes in QDK/Chemistry are immutable by design (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Access data through getter methods
num_atoms = structure.get_num_atoms()
elements = structure.get_elements()
coords_retrieved = structure.get_coordinates()

print(f"Number of atoms: {num_atoms}")
print(f"Elements: {elements}")
print(f"Coordinates shape: {coords_retrieved.shape}")

# Immutability: To "modify" a structure, create a new one
new_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]])
new_structure = Structure(new_coords, ["H", "H"])
print("New structure created with modified coordinates")
