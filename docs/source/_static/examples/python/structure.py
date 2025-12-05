"""Basic Structure creation and manipulation example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.data import Structure, Element

# Specify a structure using coordinates, and either symbols or elements
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])  # Bohr
symbols = ["H", "H"]

structure = Structure(coords, symbols=symbols)
# Equivalent to 'structure', but uses 'elements' instead of 'symbols'
elements = [Element.H, Element.H]
structure_alternative = Structure(coords, elements)

# Another variation on construction: can specify custom masses and/or charges
custom_masses = [1.001, 0.999]
custom_charges = [0.9, 1.1]
structure_custom = Structure(
    coords, elements=elements, masses=custom_masses, nuclear_charges=custom_charges
)
# end-cell-create
################################################################################

################################################################################
# start-cell-data
# Get coordinates of a specific atom
atom_coords = structure.get_atom_coordinates(0)  # First atom

# Get element of a specific atom
element = structure.get_atom_element(0)  # First atom

# Get all coordinates as a matrix
all_coords = structure.get_coordinates()

# Get all elements as a list
all_elements = structure.get_elements()

# Get nuclear repulsion energy
nuc_repulsion = structure.calculate_nuclear_repulsion_energy()
# end-cell-data
################################################################################
