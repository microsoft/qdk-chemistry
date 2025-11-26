"""Hamiltonian usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    Hamiltonian,
    BasisSet,
    Orbitals,
    OrbitalType,
    Shell,
    SpinChannel,
    Structure,
    Element,
)

# =============================================================================
# Creating a Hamiltonian object
# =============================================================================

# Create a simple structure
coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
elements = [Element.H, Element.H]
structure = Structure(coords, elements=elements)

# Run initial SCF
scf_solver = create("scf_solver")
E_hf, wfn_hf = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# Create a Hamiltonian constructor
hamiltonian_constructor = create("hamiltonian_constructor")

# Construct Hamiltonian from orbitals
hamiltonian = hamiltonian_constructor.run(wfn_hf.get_orbitals())

# =============================================================================
# Creating an unrestricted Hamiltonian object
# =============================================================================

# Create O2 (spin and multiplicity are defined below)
coords = np.array([[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]])
elements = [Element.O, Element.O]
structure = Structure(coords, elements=elements)

# Run initial SCF
scf_solver = create("scf_solver")
E_uhf, wfn_uhf = scf_solver.run(
    structure, charge=0, spin_multiplicity=3
)  # open shell: this will run UHF

# Create a Hamiltonian constructor
hamiltonian_constructor = create("hamiltonian_constructor")

# Construct Hamiltonian from orbitals
hamiltonian = hamiltonian_constructor.run(wfn_uhf.get_orbitals())
# Here, the Hamiltonian will be unrestricted by default and use the UHF orbitals

# can double-check it is unrestricted like:
is_unrestricted = hamiltonian.is_unrestricted()


# =============================================================================
# Custom construction (restricted)
# =============================================================================

# Create restricted orbitals
num_orbitals = 2

# Create basis set
shells = []
for i in range(num_orbitals):
    shell = Shell(0, OrbitalType.S, np.array([1.0]), np.array([1.0]))
    shells.append(shell)
basis_set = BasisSet("test", shells)

# Create restricted orbitals
coeffs = np.eye(num_orbitals)
restricted_orbitals = Orbitals(coeffs, None, None, basis_set)

# Create one-body integrals
one_body = np.array([[1.0, 0.2], [0.2, 1.5]])

# Create two-body integrals
rng = np.random.default_rng(42)
two_body = rng.random(num_orbitals**4)

# Inactive Fock matrix
inactive_fock = np.array([[0.5, 0.1], [0.1, 0.7]])

# Construct Hamiltonian directly
h_restricted = Hamiltonian(
    one_body,
    two_body,
    restricted_orbitals,
    core_energy=2.0,
    inactive_fock_matrix=inactive_fock,
)

# =============================================================================
# Custom construction (unrestricted)
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
    inactive_fock_matrix_beta=inactive_fock_beta,
)

# Check if Hamiltonian is unrestricted
is_unrestricted = h_unrestricted.is_unrestricted()

# =============================================================================
# Accessing Hamiltonian data
# =============================================================================

# Access one-electron integrals, returns tuple of numpy arrays
# For restricted hamiltonians, these point to the same data
h1, _ = hamiltonian.get_one_body_integrals()

# Access specific one-electron integral element <ij>
element_one = hamiltonian.get_one_body_element(0, 0)

# Access two-electron integrals, returns triple of numpy arrays
# For restricted hamiltonians, these point to the same data
h2, _, _ = hamiltonian.get_two_body_integrals()

# Access a specific two-electron integral <ij|kl>
element = hamiltonian.get_two_body_element(0, 0, 0, 0)

# Get core energy (nuclear repulsion + inactive orbital energy)
core_energy = hamiltonian.get_core_energy()

# Get inactive Fock matrix (if available)
if hamiltonian.has_inactive_fock_matrix():
    inactive_fock, _ = hamiltonian.get_inactive_fock_matrix()

# Get orbital data
orbitals = hamiltonian.get_orbitals()

# =============================================================================
# Accessing Hamiltonian data (unrestricted)
# =============================================================================

# Access one-electron integrals, returns tuple of numpy arrays
h1_alpha, h1_beta = hamiltonian.get_one_body_integrals()

# Access specific elements of one-electron integrals
element_one_aa = hamiltonian.get_one_body_element(0, 0, SpinChannel.aa)
element_one_bb = hamiltonian.get_one_body_element(0, 0, SpinChannel.bb)

# Access two-electron integrals, returns triple of numpy arrays
h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()

# Access a specific two-electron integral <ij|kl>
element_aaaa = hamiltonian.get_two_body_element(0, 0, 0, 0, SpinChannel.aaaa)
element_aabb = hamiltonian.get_two_body_element(0, 0, 0, 0, SpinChannel.aabb)

# Get inactive Fock matrix (if available)
if hamiltonian.has_inactive_fock_matrix():
    inactive_fock_alpha, inactive_fock_beta = hamiltonian.get_inactive_fock_matrix()
