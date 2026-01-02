"""Hamiltonian usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-hamiltonian-creation
from pathlib import Path
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure, SpinChannel

# Load a structure from XYZ file
structure = Structure.from_xyz_file(
    Path(__file__).parent / "../data/water.structure.xyz"
)

# Run initial SCF to get orbitals
scf_solver = create("scf_solver")
E_scf, wfn = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)
orbitals = wfn.get_orbitals()

# Create a Hamiltonian constructor
hamiltonian_constructor = create("hamiltonian_constructor")

# Construct the Hamiltonian from orbitals
hamiltonian = hamiltonian_constructor.run(orbitals)
# end-cell-hamiltonian-creation
################################################################################

################################################################################
# start-cell-properties
# Example indices for one- and two-electron integral access
i_int, j_int, k_int, l_int = 0, 1, 2, 3

# Access one-electron integrals (both spin channels)
h1_a, h1_b = hamiltonian.get_one_body_integrals()

# Access two-electron integrals (both spin channels)
h2_aaaa, h2_aabb, h2_bbbb = hamiltonian.get_two_body_integrals()

# Access a specific one-electron integral <ij> (for aa spin channel)
one_body_element = hamiltonian.get_one_body_element(i_int, j_int, SpinChannel.aa)

# Access a specific two-electron integral <ij|kl> (for aaaa spin channel)
two_body_element = hamiltonian.get_two_body_element(
    i_int, j_int, k_int, l_int, SpinChannel.aaaa
)

# Get core energy (nuclear repulsion + inactive orbital energy)
core_energy = hamiltonian.get_core_energy()

# Get orbital data
orbitals = hamiltonian.get_orbitals()
# end-cell-properties
################################################################################

################################################################################
# start-cell-validation
# Check if specific components are available
has_one_body = hamiltonian.has_one_body_integrals()
has_two_body = hamiltonian.has_two_body_integrals()
has_orbitals = hamiltonian.has_orbitals()
# end-cell-validation
################################################################################
