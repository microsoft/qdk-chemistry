"""Basic Structure creation and manipulation example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import numpy as np
from qdk_chemistry.data import Structure

# start-cell-1
# Create a Structure manually (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
elements = ["H", "H"]
structure = Structure(coords, elements)

print(f"Created structure with {structure.get_num_atoms()} atoms")
print(f"Elements: {structure.get_elements()}")
# end-cell-1

# start-cell-2
# Load from XYZ file
# structure = Structure.from_xyz_file("molecule.structure.xyz")

# Load from JSON file
# structure = Structure.from_json_file("molecule.structure.json")
# end-cell-2

# start-cell-3
# Get coordinates of a specific atom
atom_coords = structure.get_atom_coordinates(0)  # First atom
print(f"First atom coordinates: {atom_coords}")

# Get element of a specific atom
element = structure.get_atom_element(0)  # First atom
print(f"First atom element: {element}")

# Get all coordinates as a matrix
all_coords = structure.get_coordinates()
print(f"All coordinates shape: {all_coords.shape}")

# Get all elements as a list
all_elements = structure.get_elements()
print(f"All elements: {all_elements}")

# Get nuclear repulsion energy
nuc_repulsion = structure.calculate_nuclear_repulsion_energy()
print(f"Nuclear repulsion energy: {nuc_repulsion:.6f} Hartree")
# end-cell-3

# Use a temporary directory for file I/O examples
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    json_file = tmpdir_path / "molecule.structure.json"
    xyz_file = tmpdir_path / "molecule.structure.xyz"

    # start-cell-4
    # Serialize to JSON object
    json_data = structure.to_json()

    # Deserialize from JSON object
    structure_from_json = Structure.from_json(json_data)

    # Serialize to JSON file
    structure.to_json_file(str(json_file))

    # Deserialize from JSON file
    structure_from_json_file = Structure.from_json_file(str(json_file))

    # Get XYZ format as string
    xyz_string = structure.to_xyz()
    print(f"XYZ format:\n{xyz_string}")

    # Serialize to XYZ file
    structure.to_xyz_file(str(xyz_file))

    # Load from XYZ file
    structure_from_xyz_file = Structure.from_xyz_file(str(xyz_file))
    # end-cell-4

# start-cell-5
# Structure is immutable - create a new structure to modify
# new_coords = np.vstack([structure.get_coordinates(), [[1.0, 0.0, 0.0]]])
# new_elements = list(structure.get_elements()) + ["O"]
# modified_structure = Structure(new_coords, new_elements)
# end-cell-5
