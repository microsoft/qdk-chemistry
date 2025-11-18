"""Basis set usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# import numpy as np
# from qdk_chemistry.data import BasisSet, BasisType, OrbitalType, Structure

# # Create an empty basis set with a name
# basis_set = BasisSet("6-31G", BasisType.SPHERICAL)

# # Add a shell with multiple primitives
# atom_index = 0  # First atom
# orbital_type = OrbitalType.P  # p orbital
# exponents = np.array([1.0, 0.5])
# coefficients = np.array([0.6, 0.4])
# basis_set.add_shell(atom_index, orbital_type, exponents, coefficients)

# # Add a shell with single primitive (convenience)
# basis_set.add_shell(1, OrbitalType.S, 0.5, 1.0)

# # Set structure associated with the basis set
# coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
# structure = Structure(coords, ["H", "H"])
# basis_set.set_structure(structure)

# # Access basic properties
# basis_type = basis_set.get_basis_type()
# name = basis_set.get_name()

# # Access shell information
# all_shells = basis_set.get_shells()
# shells_for_atom = basis_set.get_shells_for_atom(0)
# specific_shell = basis_set.get_shell(0)

# # Get counts
# num_shells = basis_set.get_num_shells()
# num_basis_functions = basis_set.get_num_basis_functions()
# num_atoms = basis_set.get_num_atoms()

# # Get mapping information
# shell_index, m_quantum_number = basis_set.get_basis_function_info(0)
# atom_index = basis_set.get_atom_index_for_basis_function(0)

# # Get indices for specific atoms/orbitals
# basis_indices = basis_set.get_basis_function_indices_for_atom(0)
# shell_indices = basis_set.get_shell_indices_for_orbital_type(OrbitalType.P)

# # Check validity
# is_valid = basis_set.is_valid()

# # Access individual shell properties
# shell = basis_set.get_shell(0)
# shell_atom_idx = shell.get_atom_index()
# shell_orb_type = shell.get_orbital_type()
# shell_l = shell.get_l()
# shell_num_primitives = shell.get_num_primitives()
# shell_num_basis_funcs = shell.get_num_basis_functions()

# # Access shell primitive data
# shell_exponents = shell.get_exponents()
# shell_coefficients = shell.get_coefficients()
# single_exponent = shell.get_exponent(0)
# single_coeff = shell.get_coefficient(0)

# # Serialize to JSON
# json_str = basis_set.to_json()
# basis_set.to_json_file("molecule.basis.json")

# # Deserialize from JSON
# basis_set_from_json_file = BasisSet.from_json_file("molecule.basis.json")

# # Direct JSON conversion
# j = basis_set.to_json()
# basis_set_from_json = BasisSet.from_json(j)

# # HDF5 serialization (commented out - has bugs)
# # basis_set.to_hdf5_file("molecule.basis.h5")
# # basis_set.from_hdf5_file("molecule.basis.h5")

# # Create a basis set from a predefined library
# basis_set_from_lib = BasisSet.create("6-31G")

# # List all available basis sets
# available_basis_sets = BasisSet.get_available_basis_sets()

# # Static utility functions for type conversions
# orbital_str = BasisSet.orbital_type_to_string(OrbitalType.D)  # "d"
# orbital_type = BasisSet.string_to_orbital_type("f")  # OrbitalType.F

# l_value = BasisSet.get_angular_momentum(OrbitalType.P)  # 1
# num_orbitals = BasisSet.get_num_orbitals_for_l(2, BasisType.SPHERICAL)  # 5

# basis_str = BasisSet.basis_type_to_string(BasisType.CARTESIAN)  # "cartesian"
# basis_type = BasisSet.string_to_basis_type("spherical")  # BasisType.SPHERICAL
