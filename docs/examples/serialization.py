"""Serialization examples for QDK Chemistry objects."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import os
import shutil
import tempfile

import numpy as np
from qdk_chemistry.data import Structure

# =============================================================================
# Serialization for structure objects
# =============================================================================

# Create a structure (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
custom_masses = [1.001, 0.999]
custom_charges = [0.9, 1.1]
structure = Structure(coords, symbols=symbols, masses=custom_masses, nuclear_charges=custom_charges)

# Serialize to HDF5 file
structure.to_hdf5_file("h2_molecule.structure.h5")

# Deserialize from HDF5 file
structure_from_hdf5 = Structure.from_hdf5_file("h2_molecule.structure.h5")

# TODO 