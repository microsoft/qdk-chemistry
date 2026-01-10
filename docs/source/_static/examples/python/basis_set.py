"""Basis set usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-basis-set-create
import tempfile
from pathlib import Path
from qdk_chemistry.data import AOType, BasisSet, OrbitalType, Shell, Structure

# Create a spherical basis set from a single S shell
shell = Shell(0, OrbitalType.S, [1.0], [1.0])
basis_spherical = BasisSet("spherical example", [shell])

# Create a Cartesian basis set
basis_cartesian = BasisSet("6-31G", [shell], AOType.Cartesian)

# end-cell-basis-set-create
################################################################################

################################################################################
# start-cell-basis-set-get
# Load a water molecule structure from XYZ file
structure = Structure.from_xyz_file(
    Path(__file__).parent / "../data/water.structure.xyz"
)

# Create shells consistent with structure
shells = [
    Shell(0, OrbitalType.S, [1.0], [1.0]),  # O 1s
    Shell(1, OrbitalType.S, [1.0], [1.0]),  # H 1s
    Shell(2, OrbitalType.S, [1.0], [1.0]),  # H 1s
]
basis = BasisSet("STO-3G", shells, structure)

# Access basic properties
print(f"Basis set name: {basis.get_name()}")
print(f"Atomic orbital type: {basis.get_atomic_orbital_type()}")

# Access shell information
print(f"Number of atomic orbitals: {basis.get_num_atomic_orbitals()}")
print(f"Number of shells: {basis.get_num_shells()}")
for ishell, shell in enumerate(basis.get_shells()):
    print(
        f"Shell {ishell}: type={shell.orbital_type}, l={shell.get_angular_momentum()}, "
        f"num_primitives={shell.get_num_primitives()}"
    )
print(f"Shells for atom 0: {basis.get_shells_for_atom(0)}")
print(f"Specific shell = {basis.get_shell(0)}")

# Get counts
print(f"num_shells = {basis.get_num_shells()}")
print(f"num_atomic_orbitals = {basis.get_num_atomic_orbitals()}")
print(f"num_atoms = {basis.get_num_atoms()}")

# Get mapping information
print(f"shell_index, m_quantum_number = {basis.get_atomic_orbital_info(0)}")
print(f"atom_index = {basis.get_atom_index_for_atomic_orbital(0)}")

# Get indices for specific atoms/orbitals
print(f"atomic_orbital_indices = {basis.get_atomic_orbital_indices_for_atom(0)}")
print(f"shell_indices = {basis.get_shell_indices_for_orbital_type(OrbitalType.P)}")
# end-cell-basis-set-get
################################################################################

################################################################################
# start-cell-shells
# Access individual shell properties
print(f"shell_atom_idx = {shell.atom_index}")
print(f"shell_orb_type = {shell.orbital_type}")
shell = basis.get_shell(0)
print(f"shell_l = {shell.get_angular_momentum()}")
print(f"shell_num_primitives = {shell.get_num_primitives()}")
print(f"shell_num_atomic_orbitals = {shell.get_num_atomic_orbitals()}")

# Access shell primitive data
print(f"shell_exponents = {shell.exponents}")
print(f"shell_coefficients = {shell.coefficients}")
# end-cell-shells
################################################################################

################################################################################
# start-cell-serialization
# Create a new basis set with data
shells = [
    Shell(0, OrbitalType.S, [1.0], [1.0]),
    Shell(0, OrbitalType.P, [0.5], [1.0]),
]
basis_out = BasisSet("STO-3G", shells)

# Demonstrate and test JSON conversion
json_data = basis_out.to_json()
assert isinstance(json_data, str)
assert "STO-3G" in json_data

basis_in = BasisSet.from_json(json_data)
assert basis_in.get_name() == "STO-3G"
assert basis_in.get_num_shells() == 2
assert basis_in.get_num_atomic_orbitals() == 4

with tempfile.NamedTemporaryFile(
    suffix=".basis_set.json", mode="w", delete=False
) as tmp:
    filename = tmp.name
    basis_out.to_json_file(filename)
    basis_file = BasisSet.from_json_file(filename)
    assert basis_file.get_name() == "STO-3G"
    assert basis_file.get_num_shells() == 2
    assert basis_file.get_num_atomic_orbitals() == 4

# Demonstrate and test HDF5 conversion
with tempfile.NamedTemporaryFile(suffix=".basis_set.h5", delete=False) as tmp:
    filename = tmp.name
    basis_out.to_hdf5_file(filename)
    basis_in = BasisSet.from_hdf5_file(filename)
    assert basis_in.get_name() == "STO-3G"
    assert basis_in.get_num_shells() == 2
    assert basis_in.get_num_atomic_orbitals() == 4
# end-cell-serialization
################################################################################

################################################################################
# start-cell-utility-functions
# Static utility functions for type conversions
print(f"orbital_str = {BasisSet.orbital_type_to_string(OrbitalType.D)}")
print(f"orbital_type = {BasisSet.string_to_orbital_type('f')}")
print(f"l_value = {BasisSet.get_angular_momentum(OrbitalType.P)}")
print(f"num_orbitals = {BasisSet.get_num_orbitals_for_l(2, AOType.Spherical)}")
print(f"basis_str = {BasisSet.atomic_orbital_type_to_string(AOType.Cartesian)}")
print(f"atomic_orbital_type = {BasisSet.string_to_atomic_orbital_type('spherical')}")
# end-cell-utility-functions
################################################################################

################################################################################
# start-cell-library

# TODO: add example of listing available basis sets in the library and loading one

# end-cell-library
################################################################################
