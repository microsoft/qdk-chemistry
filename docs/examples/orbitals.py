"""Orbitals usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import tempfile
from pathlib import Path

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# Create H2 molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Obtain orbitals from an SCF calculation
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
E_scf, wfn = scf_solver.run(structure, charge=0, spin_multiplicity=1)
orbitals = wfn.get_orbitals()

print(f"SCF Energy: {E_scf:.6f} Hartree")

# set coefficients manually example (restricted)
# orbs_manual = Orbitals()
# coeffs = # coefficient matrix
# orbs_manual.set_coefficients(coeffs)            # Same for alpha and beta

# set coefficients manually example (unrestricted)
# orbs_unrestricted = Orbitals()
# coeffs_alpha = # alpha coefficients
# coeffs_beta = # beta coefficients
# orbs_unrestricted.set_coefficients(coeffs_alpha, coeffs_beta)

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

# Use a temporary directory for any file I/O
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    orbitals_file = tmpdir_path / "molecule.orbitals.json"

    # Generic serialization with format specification
    orbitals.to_file(str(orbitals_file), "json")
    orbitals_from_file = orbitals.from_file(str(orbitals_file), "json")

    # JSON serialization
    orbitals.to_json_file(str(orbitals_file))
    orbitals_from_json_file = orbitals.from_json_file(str(orbitals_file))

# Direct JSON conversion
j = orbitals.to_json()
orbitals_from_json = orbitals.from_json(j)
# HDF5 serialization (commented out - has bugs)
# orbitals.to_hdf5_file("molecule.orbitals.h5")
# orbitals_from_hdf5 = Orbitals.from_hdf5_file("molecule.orbitals.h5")
