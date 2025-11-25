"""Hamiltonian usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Hamiltonian, Structure, BasisSet, Orbitals, OrbitalType, Shell, SpinChannel

# Example: Restricted Hamiltonian - Closed-shell system (H2 singlet)
print("="*80)
print("Example: Restricted Hamiltonian (H2 singlet)")
print("="*80)

# Create a molecular structure (H2 molecule, coordinates in Bohr)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Run restricted Hartree-Fock (RHF) calculation
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "sto-3g")
scf_energy, scf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)

print(f"RHF Energy: {scf_energy:.8f} Hartree")

# Get orbitals from the wavefunction
orbitals = scf_wavefunction.get_orbitals()
print(f"Number of molecular orbitals: {orbitals.get_num_molecular_orbitals()}")
print(f"Orbitals are restricted: {orbitals.is_restricted()}")

# Construct the restricted Hamiltonian
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(orbitals)

# Access one-electron integrals
h1, _ = hamiltonian.get_one_body_integrals()
print(f"One-body integrals shape: {h1.shape}")
one_body_integral_element = hamiltonian.get_one_body_element(0,0)
print(f"One body integral element 0,0: {one_body_integral_element}")

# Access two-electron integrals
h2_integrals, _, _ = hamiltonian.get_two_body_integrals()
print(f"Two-body integrals shape: {h2_integrals.shape}")
# Element i,j,k,l
two_body_integral_element = hamiltonian.get_two_body_element(0,0,0,0)
print(f"Two body integral element 0,0,0,0: {two_body_integral_element}")

# Get core energy
core_energy = hamiltonian.get_core_energy()
print(f"Core energy: {core_energy:.8f} Hartree")

# Example: Unrestricted Hamiltonian - Open-shell system (O2 triplet)
print("\n" + "="*80)
print("Unrestricted Hamiltonian (O2 triplet)")
print("="*80)

# Create O2 molecule (ground state triplet)
coords_o2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.282]])
o2_structure = Structure(coords_o2, ["O", "O"])

# Run unrestricted Hartree-Fock (UHF) calculation with triplet spin state
scf_solver_uhf = create("scf_solver")
scf_solver_uhf.settings().set("basis_set", "sto-3g")

# spin_multiplicity = 3 for triplet
uhf_energy, uhf_wavefunction = scf_solver_uhf.run(o2_structure, charge=0, spin_multiplicity=3)

print(f"UHF Energy: {uhf_energy:.8f} Hartree")

# Get unrestricted orbitals
uhf_orbitals = uhf_wavefunction.get_orbitals()
print(f"Number of molecular orbitals: {uhf_orbitals.get_num_molecular_orbitals()}")
print(f"Orbitals are unrestricted: {uhf_orbitals.is_unrestricted()}")

# Construct the unrestricted Hamiltonian
hamiltonian_constructor_uhf = create("hamiltonian_constructor")
hamiltonian_uhf = hamiltonian_constructor_uhf.run(uhf_orbitals)

# Access one-electron integrals (alpha and beta are different for unrestricted)
h1_alpha_uhf, h1_beta_uhf = hamiltonian_uhf.get_one_body_integrals()
print(f"\nOne-body integrals shape (alpha): {h1_alpha_uhf.shape}")
print(f"One-body integrals shape (beta): {h1_beta_uhf.shape}")
print(f"Alpha and beta are different: {not np.array_equal(h1_alpha_uhf, h1_beta_uhf)}")
alpha_one_body_integral_element = hamiltonian_uhf.get_one_body_element(0, 0, SpinChannel.aa)
beta_one_body_integral_element = hamiltonian_uhf.get_one_body_element(0, 0, SpinChannel.bb)
print("Element (0,0) of alpha one body integrals", alpha_one_body_integral_element)
print("Element (0,0) of beta one body integral element", beta_one_body_integral_element)

# Access two-electron integrals (aaaa, aabb, bbbb are all different for unrestricted)
h2_aaaa_uhf, h2_aabb_uhf, h2_bbbb_uhf = hamiltonian_uhf.get_two_body_integrals()
print(f"\nTwo-body integrals:")
print(f"  aaaa size: {len(h2_aaaa_uhf)}")
print(f"  aabb size: {len(h2_aabb_uhf)}")
print(f"  bbbb size: {len(h2_bbbb_uhf)}")
alpha_two_body_integral_element = hamiltonian_uhf.get_two_body_element(0, 0, 0, 0, SpinChannel.aaaa)
mixed_two_body_integral_element = hamiltonian_uhf.get_two_body_element(0, 0, 0, 0, SpinChannel.aabb)
beta_two_body_integral_element = hamiltonian_uhf.get_two_body_element(0, 0, 0, 0, SpinChannel.bbbb)
print("Element (0,0,0,0) of alpha two body integrals", alpha_two_body_integral_element)
print("Element (0,0,0,0) of alpha-beta two body integrals", mixed_two_body_integral_element)
print("Element (0,0,0,0) of beta two body integrals", beta_two_body_integral_element)

# Get core energy
core_energy_uhf = hamiltonian_uhf.get_core_energy()
print(f"Core energy: {core_energy_uhf:.8f} Hartree")

# Example: Direct construction of unrestricted Hamiltonian
print("\n" + "="*80)
print("Example: Direct unrestricted Hamiltonian construction")
print("="*80)

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

print(f"Created unrestricted orbitals: {unrestricted_orbitals.is_unrestricted()}")

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
direct_unrestricted_hamiltonian = Hamiltonian(
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

print(f"\nDirect unrestricted Hamiltonian:")
print(f"  is_unrestricted(): {direct_unrestricted_hamiltonian.is_unrestricted()}")
print(f"  Core energy: {direct_unrestricted_hamiltonian.get_core_energy():.8f}")

# Verify separate alpha/beta components
h1_a, h1_b = direct_unrestricted_hamiltonian.get_one_body_integrals()
print(f"  Alpha one-body integrals:\n{h1_a}")
print(f"  Beta one-body integrals:\n{h1_b}")