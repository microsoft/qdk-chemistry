"""Orbitals usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import tempfile
from pathlib import Path

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Orbitals, Structure

# Create H2 molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Obtain orbitals from an SCF calculation
scf_solver = create("scf_solver")
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_information="sto-3g"
)
orbitals = wfn.get_orbitals()

print(f"SCF Energy: {E_scf:.6f} Hartree")
# end-cell-create
################################################################################

################################################################################
# start-cell-access
# Access orbital coefficients (returns tuple of alpha/beta matrices)
coeffs_alpha, coeffs_beta = orbitals.get_coefficients()
print(f"Orbital coefficients shape: {coeffs_alpha.shape}")

# Access orbital energies (returns tuple of alpha/beta vectors)
energies_alpha, energies_beta = orbitals.get_energies()
print(f"Orbital energies: {energies_alpha}")

# Access atomic orbital overlap matrix
ao_overlap = orbitals.get_overlap_matrix()
print(f"AO overlap matrix shape: {ao_overlap.shape}")

# Access basis set information
basis_set = orbitals.get_basis_set()
print(f"Basis set: {basis_set.get_name()}")

# Check calculation type
is_restricted = orbitals.is_restricted()
print(f"Is restricted: {is_restricted}")

# Get size information
num_mos = orbitals.get_num_molecular_orbitals()
num_aos = orbitals.get_num_atomic_orbitals()
print(f"MOs: {num_mos}, AOs: {num_aos}")

# Get summary
print(orbitals.get_summary())
# end-cell-access
################################################################################

################################################################################
# start-cell-serialization
# Use a temporary directory for file I/O
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    orbitals_file = tmpdir_path / "molecule.orbitals.json"

    # Generic serialization with format specification
    orbitals.to_file(str(orbitals_file), "json")
    orbitals_from_file = Orbitals.from_file(str(orbitals_file), "json")

    # JSON serialization
    orbitals.to_json_file(str(orbitals_file))
    orbitals_from_json_file = Orbitals.from_json_file(str(orbitals_file))

    # Direct JSON conversion
    j = orbitals.to_json()
    orbitals_from_json = Orbitals.from_json(j)

    # HDF5 serialization (if available)
    # orbitals.to_hdf5_file(str(tmpdir_path / "molecule.orbitals.h5"))
    # orbitals_from_hdf5 = Orbitals.from_hdf5_file(str(tmpdir_path / "molecule.orbitals.h5"))
# end-cell-serialization
################################################################################
