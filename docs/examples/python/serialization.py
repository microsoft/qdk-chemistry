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

# Create a structure (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Serialize to JSON object
json_data = structure.to_json()
print("Serialized to JSON:", type(json_data))

# Deserialize from JSON object
structure_from_json = Structure.from_json(json_data)
print(f"Deserialized structure has {structure_from_json.get_num_atoms()} atoms")

# Serialize to/from file using temporary directory
tmpdir = tempfile.mkdtemp()
json_file = os.path.join(tmpdir, "molecule.structure.json")

# Serialize to JSON file
structure.to_json_file(json_file)
print(f"Saved to {json_file}")

# Deserialize from JSON file
structure_from_file = Structure.from_json_file(json_file)
print(f"Loaded structure from file: {structure_from_file.get_num_atoms()} atoms")

hdf5_file = os.path.join(tmpdir, "molecule.structure.h5")
# Serialize to HDF5 file
# structure.to_hdf5_file(hdf5_file)
# print(f"Saved to {hdf5_file}")

# Deserialize from HDF5 file
# structure_from_hdf5 = Structure.from_hdf5_file(hdf5_file)
# print(f"Loaded structure from HDF5: {structure_from_hdf5.get_num_atoms()} atoms")

# Clean up
shutil.rmtree(tmpdir)
