"""Hamiltonian usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Hamiltonian, BasisSet, Orbitals, OrbitalType, Shell, SpinChannel

# =============================================================================
# Creating a Hamiltonian object
# =============================================================================

# Create a Hamiltonian constructor
hamiltonian_constructor = create("hamiltonian_constructor")

# Set active orbitals if needed
active_orbitals = [4, 5, 6, 7]  # Example indices
hamiltonian_constructor.settings().set("active_orbitals", active_orbitals)

# Construct the Hamiltonian from orbitals
# (assuming 'orbitals' object exists from prior calculation)
# hamiltonian = hamiltonian_constructor.run(orbitals)

# =============================================================================
# Creating an unrestricted Hamiltonian
# =============================================================================

# Create test unrestricted orbitals
num_orbitals = 2

# Create basis set
shells = []
for i in range(num_orbitals):
    shell = Shell(0, OrbitalType.S, np.array([1.0]), np.array([1.0]))
    shells.append(shell)
basis_set = BasisSet("test", shells)

# Create unrestricted orbitals with different alpha and beta coefficients
coeffs_alpha = np.eye(num_orbitals)
coeffs_beta = np.array([[0.8, 0.6], [0.6, -0.8]])
unrestricted_orbitals = Orbitals(coeffs_alpha, coeffs_beta, None, None, None, basis_set)

# Create unrestricted integral data (alpha and beta are different)
one_body_alpha = np.array([[1.0, 0.2], [0.2, 1.5]])
one_body_beta = np.array([[1.1, 0.3], [0.3, 1.6]])

# Create spin-separated two-body integrals
rng = np.random.default_rng(42)
two_body_aaaa = rng.random(num_orbitals**4)
two_body_aabb = rng.random(num_orbitals**4)
two_body_bbbb = rng.random(num_orbitals**4)

# Inactive Fock matrices (also spin-separated)
inactive_fock_alpha = np.array([[0.5, 0.1], [0.1, 0.7]])
inactive_fock_beta = np.array([[0.6, 0.2], [0.2, 0.8]])

# Construct unrestricted Hamiltonian directly
h_unrestricted = Hamiltonian(
    one_body_alpha,
    one_body_beta,
    two_body_aaaa,
    two_body_aabb,
    two_body_bbbb,
    unrestricted_orbitals,
    core_energy=2.0,
    inactive_fock_matrix_alpha=inactive_fock_alpha,
    inactive_fock_matrix_beta=inactive_fock_beta
)

# Check if Hamiltonian is unrestricted
is_unrestricted = h_unrestricted.is_unrestricted()
is_restricted = h_unrestricted.is_restricted()

# =============================================================================
# Accessing Hamiltonian data
# =============================================================================

# Access one-electron integrals, returns tuple of numpy arrays
# For restricted hamiltonians, these point to the same data
# h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()

# Access two-electron integrals, returns triple of numpy arrays
# For restricted hamiltonians, these point to the same data
# h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()

# Access a specific two-electron integral <ij|kl>
# element = hamiltonian.get_two_body_element(i, j, k, l)

# Get core energy (nuclear repulsion + inactive orbital energy)
# core_energy = hamiltonian.get_core_energy()

# Get inactive Fock matrix (if available)
# if hamiltonian.has_inactive_fock_matrix():
#     inactive_fock_alpha, inactive_fock_beta = hamiltonian.get_inactive_fock_matrix()

# Get orbital data
# orbitals = hamiltonian.get_orbitals()

# Get active space information
# active_indices = hamiltonian.get_selected_orbital_indices()
# num_electrons = hamiltonian.get_num_electrons()
# num_orbitals = hamiltonian.get_num_orbitals()

# For unrestricted Hamiltonians, access specific one-electron integral channels
integral_aa = h_unrestricted.get_one_body_element(0, 0, SpinChannel.aa)
integral_bb = h_unrestricted.get_one_body_element(0, 0, SpinChannel.bb)

# For unrestricted Hamiltonians, access specific two-electron integral channels
integral_aaaa = h_unrestricted.get_two_body_element(0, 0, 0, 0, SpinChannel.aaaa)
integral_aabb = h_unrestricted.get_two_body_element(0, 0, 0, 0, SpinChannel.aabb)
integral_bbbb = h_unrestricted.get_two_body_element(0, 0, 0, 0, SpinChannel.bbbb)

# Access fock matrices for alpha and beta
fock_alpha, fock_beta = h_unrestricted.get_inactive_fock_matrix()

# Get orbital data
orbitals = h_unrestricted.get_orbitals()

# Get active space information
active_indices = h_unrestricted.get_selected_orbital_indices()
num_electrons = h_unrestricted.get_num_electrons()
num_orbitals = h_unrestricted.get_num_orbitals()

# =============================================================================
# File formats (Serialization)
# =============================================================================

# Serialize to JSON file
# hamiltonian.to_json_file("molecule.hamiltonian.json")

# Deserialize from JSON file
# from qdk_chemistry.data import Hamiltonian
# hamiltonian_from_json_file = Hamiltonian.from_json_file("molecule.hamiltonian.json")

# Serialize to HDF5 file
# hamiltonian.to_hdf5_file("molecule.hamiltonian.h5")

# Deserialize from HDF5 file
# hamiltonian_from_hdf5_file = Hamiltonian.from_hdf5_file("molecule.hamiltonian.h5")

# Generic file I/O based on type parameter
# hamiltonian.to_file("molecule.hamiltonian.json", "json")
# hamiltonian_from_file = Hamiltonian.from_file("molecule.hamiltonian.h5", "hdf5")

# Convert to/from JSON in Python
# import json
# j = hamiltonian.to_json()
# j_str = json.dumps(j)
# hamiltonian_from_json = Hamiltonian.from_json(json.loads(j_str))

# =============================================================================
# Validation methods
# =============================================================================

# Check if the Hamiltonian data is complete and consistent
# valid = hamiltonian.is_valid()

# Check if specific components are available
# has_one_body = hamiltonian.has_one_body_integrals()
# has_two_body = hamiltonian.has_two_body_integrals()
# has_orbitals = hamiltonian.has_orbitals()
# has_inactive_fock = hamiltonian.has_inactive_fock_matrix()
