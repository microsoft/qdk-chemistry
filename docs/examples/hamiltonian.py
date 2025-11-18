"""Hamiltonian usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create

# Create a Hamiltonian constructor
hamiltonian_constructor = create("hamiltonian_constructor")

# Set active orbitals if needed
# active_orbitals = [4, 5, 6, 7]  # Example indices
# hamiltonian_constructor.settings().set("active_orbitals", active_orbitals)

# Construct the Hamiltonian from orbitals (requires orbitals object)
# hamiltonian = hamiltonian_constructor.run(orbitals)

# Alternatively, create a Hamiltonian directly
# direct_hamiltonian = Hamiltonian(one_body_integrals, two_body_integrals, orbitals,
#                                selected_orbital_indices, num_electrons, core_energy)

# Access one-electron integrals
# h1 = hamiltonian.get_one_body_integrals()

# Access two-electron integrals
# h2 = hamiltonian.get_two_body_integrals()

# Access a specific two-electron integral <ij|kl>
# element = hamiltonian.get_two_body_element(i, j, k, l)

# Get core energy (nuclear repulsion + inactive orbital energy)
# core_energy = hamiltonian.get_core_energy()

# Get inactive Fock matrix (if available)
# if hamiltonian.has_inactive_fock_matrix():
#     fock_matrix = hamiltonian.get_inactive_fock_matrix()

# Serialize to JSON file
# hamiltonian.to_json_file("molecule.hamiltonian.json")

# Deserialize from JSON file
# hamiltonian_from_json_file = Hamiltonian.from_json_file("molecule.hamiltonian.json")

# Serialize to HDF5 file (commented out - has bugs)
# hamiltonian.to_hdf5_file("molecule.hamiltonian.h5")
# hamiltonian_from_hdf5_file = Hamiltonian.from_hdf5_file("molecule.hamiltonian.h5")

# Generic file I/O based on type parameter
# hamiltonian.to_file("molecule.hamiltonian.json", "json")
# hamiltonian_loaded = Hamiltonian.from_file("molecule.hamiltonian.json", "json")

# Check if the Hamiltonian data is complete and consistent
# valid = hamiltonian.is_valid()

# Check if specific components are available
# has_one_body = hamiltonian.has_one_body_integrals()
# has_two_body = hamiltonian.has_two_body_integrals()
# has_orbitals = hamiltonian.has_orbitals()
# has_inactive_fock = hamiltonian.has_inactive_fock_matrix()
