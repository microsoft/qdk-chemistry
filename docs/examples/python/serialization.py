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

tmpdir = tempfile.mkdtemp()
json_file = os.path.join(tmpdir, "molecule.structure.json")

# Create a structure (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])
# start-cell-1
# Serialize to JSON object
json_data = structure.to_json()

# Deserialize from JSON object
structure_from_json = Structure.from_json(json_data)

# Serialize to JSON file
structure.to_json_file(json_file)

# Deserialize from JSON file
structure_from_file = Structure.from_json_file(json_file)
# end-cell-1

hdf5_file = os.path.join(tmpdir, "molecule.structure.h5")
# start-cell-2
# Serialize to HDF5 file (TODO: HDF5 not yet implemented for Structure)
# structure.to_hdf5_file(hdf5_file)

# Deserialize from HDF5 file
# structure_from_hdf5 = Structure.from_hdf5_file(hdf5_file)
# end-cell-2

# Clean up
shutil.rmtree(tmpdir)
