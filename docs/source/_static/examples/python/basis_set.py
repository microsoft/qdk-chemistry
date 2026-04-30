"""Basis set usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-loading
from pathlib import Path

import numpy as np
from qdk_chemistry.data import AOType, BasisSet, OrbitalType, Shell, Structure

# Load a water molecule structure from XYZ file
structure = Structure.from_xyz_file(
    Path(__file__).parent / "../data/water.structure.xyz"
)

# Create basis sets from the library using basis set name
basis_from_name = BasisSet.from_basis_name("sto-3g", structure)

# Create basis sets from the library using element-based mapping
basis_map = {"H": "sto-3g", "O": "def2-svp"}
basis_from_element = BasisSet.from_element_map(basis_map, structure)

# Create basis sets from the library using index-based mapping
index_basis_map = {0: "def2-svp", 1: "sto-3g", 2: "sto-3g"}  # O at 0, H at 1 and 2
basis_from_index = BasisSet.from_index_map(index_basis_map, structure)
# end-cell-loading
################################################################################

################################################################################
# start-cell-loading-with-aux
# Load a primary basis set with an auxiliary basis set for density fitting
basis_with_aux = BasisSet.from_basis_name("def2-svp", "def2-universal-jfit", structure)
print(f"Primary shells: {basis_with_aux.get_num_shells()}")
print(f"Auxiliary shells: {basis_with_aux.get_num_aux_shells()}")
print(f"Auxiliary name: {basis_with_aux.get_aux_name()}")
# end-cell-loading-with-aux
################################################################################

################################################################################
# start-cell-create
# Create a shell with multiple primitives
atom_index = 0  # First atom
orbital_type = OrbitalType.P  # p orbital
exponents = np.array([0.16871439, 0.62391373])
coefficients = np.array([0.43394573, 0.56604777])
shell1 = Shell(atom_index, orbital_type, exponents, coefficients)

# Create a shell with a single primitive
shell2 = Shell(1, OrbitalType.S, [0.5], [1.0])

# Create a basis set from the shells
basis_set = BasisSet(
    "6-31G", [shell1, shell2], structure=structure, atomic_orbital_type=AOType.Spherical
)
# end-cell-create
################################################################################

################################################################################
# start-cell-access
# Get basis set type and name (returns AOType)
basis_atomic_orbital_type = basis_set.get_atomic_orbital_type()
# Get basis set name (returns str)
basis_name = basis_set.get_name()

# Get all shells (returns list[Shell])
all_shells = basis_set.get_shells()
# Get shells for specific atom (returns list[Shell])
shells_for_atom = basis_set.get_shells_for_atom(0)
# Get specific shell by index (returns Shell)
specific_shell = basis_set.get_shell(1)

# Get counts
num_shells = basis_set.get_num_shells()
num_atomic_orbitals = basis_set.get_num_atomic_orbitals()
num_atoms = basis_set.get_num_atoms()

# Get atomic orbital information (returns tuple[int, int])
shell_index, m_quantum_number = basis_set.get_atomic_orbital_info(2)
atom_index = basis_set.get_atom_index_for_atomic_orbital(2)

# Get indices for specific atoms or orbital types
# Returns list[int]
atomic_orbital_indices = basis_set.get_atomic_orbital_indices_for_atom(1)
# Returns list[int]
shell_indices = basis_set.get_shell_indices_for_orbital_type(OrbitalType.P)
# Returns list[int]
shell_indices_specific = basis_set.get_shell_indices_for_atom_and_orbital_type(
    0, OrbitalType.D
)
# end-cell-access
################################################################################

################################################################################
# start-cell-shells
# Get shell by index (returns Shell)
shell = basis_set.get_shell(0)
atom_idx = shell.atom_index
orb_type = shell.orbital_type
# Get exponents (returns np.ndarray)
exps = shell.exponents
# Get coefficients (returns np.ndarray)
coeffs = shell.coefficients

# Get information from shell
num_primitives = shell.get_num_primitives()
num_aos = shell.get_num_atomic_orbitals(AOType.Spherical)
angular_momentum = shell.get_angular_momentum()
# end-cell-shells
################################################################################

################################################################################
# start-cell-serialization
# Generic serialization with format specification
basis_set.to_file("molecule.basis_set.json", "json")
basis_set = BasisSet.from_file("molecule.basis_set.json", "json")

# JSON serialization
basis_set.to_json_file("molecule.basis_set.json")
basis_set = BasisSet.from_json_file("molecule.basis_set.json")
# Direct JSON conversion
j = basis_set.to_json()
basis_set = BasisSet.from_json(j)

# HDF5 serialization
basis_set.to_hdf5_file("molecule.basis_set.h5")
basis_set = BasisSet.from_hdf5_file("molecule.basis_set.h5")
# end-cell-serialization
Path("molecule.basis_set.json").unlink()
Path("molecule.basis_set.h5").unlink()
################################################################################

################################################################################
# start-cell-ecp
# Create an ECP shell with radial powers (r^n terms)
ecp_exponents = np.array([10.0, 5.0])
ecp_coefficients = np.array([50.0, 20.0])
ecp_rpowers = np.array([0, 2], dtype=np.int32)
ecp_shell = Shell(0, OrbitalType.S, ecp_exponents, ecp_coefficients, ecp_rpowers)

# Create a basis set with ECP data
ecp_shells = [ecp_shell]
ecp_electrons = [28, 0, 0]  # 28 core electrons replaced on the first atom
basis_with_ecp = BasisSet(
    "my-basis",
    [shell1, shell2],
    ecp_name="my-ecp",
    ecp_shells=ecp_shells,
    ecp_electrons=ecp_electrons,
    structure=structure,
)

# Query ECP data
print(f"Has ECP shells: {basis_with_ecp.has_ecp_shells()}")
print(f"ECP name: {basis_with_ecp.get_ecp_name()}")
print(f"ECP electrons: {list(basis_with_ecp.get_ecp_electrons())}")
print(f"Num ECP shells: {basis_with_ecp.get_num_ecp_shells()}")
# end-cell-ecp
################################################################################

################################################################################
# start-cell-auxiliary
# Create auxiliary shells for density fitting
aux_shells = [
    Shell(0, OrbitalType.S, np.array([5.0]), np.array([2.0])),
    Shell(1, OrbitalType.S, np.array([4.0]), np.array([1.5])),
]

# Construct a basis set with a named auxiliary basis
basis_with_aux_manual = BasisSet(
    "my-basis",
    [shell1, shell2],
    aux_name="my-aux-fit",
    aux_shells=aux_shells,
    structure=structure,
)

# Query auxiliary data
print(f"Has auxiliary: {basis_with_aux_manual.has_aux_basis()}")
print(f"Auxiliary name: {basis_with_aux_manual.get_aux_name()}")
print(f"Num aux shells: {basis_with_aux_manual.get_num_aux_shells()}")

# Retrieve auxiliary shell data
for i in range(basis_with_aux_manual.get_num_aux_shells()):
    s = basis_with_aux_manual.get_aux_shell(i)
    print(f"  Aux shell {i}: atom={s.atom_index}, type={s.orbital_type}")
# end-cell-auxiliary
################################################################################

################################################################################
# start-cell-utility-functions
# Convert orbital type to string (returns str)
orbital_str = BasisSet.orbital_type_to_string(OrbitalType.D)  # "d"
# Convert string to orbital type (returns OrbitalType)
orbital_type = BasisSet.string_to_orbital_type("f")  # OrbitalType.F

# Get angular momentum (returns int)
l_value = BasisSet.get_angular_momentum(OrbitalType.P)  # 1
# Get number of orbitals for angular momentum (returns int)
num_orbitals = BasisSet.get_num_orbitals_for_l(2, AOType.Spherical)  # 5

# Convert basis type to string (returns str)
basis_str = BasisSet.atomic_orbital_type_to_string(AOType.Cartesian)  # "cartesian"
# Convert string to basis type (returns AOType)
atomic_orbital_type = BasisSet.string_to_atomic_orbital_type(
    "spherical"
)  # AOType.Spherical
# end-cell-utility-functions
################################################################################

################################################################################
# start-cell-library
# Check supported basis sets
supported_basis_sets = BasisSet.get_supported_basis_set_names()

# Check supported elements for basis set
supported_elements = BasisSet.get_supported_elements_for_basis_set("sto-3g")
# end-cell-library
################################################################################
