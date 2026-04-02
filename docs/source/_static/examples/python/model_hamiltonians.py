"""Model Hamiltonian construction examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
from qdk_chemistry import algorithms
from qdk_chemistry.data import LatticeGraph
from qdk_chemistry.utils.model_hamiltonians import (
    create_heisenberg_hamiltonian,
    create_hubbard_hamiltonian,
    create_huckel_hamiltonian,
    create_ising_hamiltonian,
    create_ppp_hamiltonian,
    mataga_nishimoto_potential,
    ohno_potential,
    pairwise_potential,
)

################################################################################
# start-cell-create-huckel
# Create a 6-site chain for the Hückel model
lattice = LatticeGraph.chain(6)

# Uniform parameters: all sites have the same on-site energy and hopping
hamiltonian = create_huckel_hamiltonian(lattice, epsilon=0.0, t=1.0)

# Print the one-body integrals
print(f"Has one-body integrals: {hamiltonian.has_one_body_integrals()}")
print(f"Has two-body integrals: {hamiltonian.has_two_body_integrals()}")
# end-cell-create-huckel
################################################################################

################################################################################
# start-cell-create-hubbard
# Create a 4-site Hubbard chain
lattice = LatticeGraph.chain(4)
hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=4.0)

print(f"Has one-body integrals: {hamiltonian.has_one_body_integrals()}")
print(f"Has two-body integrals: {hamiltonian.has_two_body_integrals()}")

# Verify the on-site repulsion
for i in range(lattice.num_sites):
    print(f"  U({i},{i},{i},{i}) = {hamiltonian.get_two_body_element(i, i, i, i):.1f}")
# end-cell-create-hubbard
################################################################################

################################################################################
# start-cell-create-hubbard-2d
# Create a 4x4 square lattice Hubbard model with periodic boundaries
lattice = LatticeGraph.square(4, 4, periodic_x=True, periodic_y=True)
hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=4.0)

print(f"Has two-body integrals: {hamiltonian.has_two_body_integrals()}")
# end-cell-create-hubbard-2d
################################################################################

################################################################################
# start-cell-create-ppp
# Create a 6-site ring for the PPP model (benzene-like)
lattice = LatticeGraph.chain(6, periodic=True)

# Compute interpair Coulomb repulsion with the Ohno potential
V = ohno_potential(lattice, U=0.414, R=2.65)

# Create the PPP Hamiltonian
hamiltonian = create_ppp_hamiltonian(
    lattice,
    epsilon=0.0,
    t=0.088,
    U=0.414,
    V=V,
    z=1.0,
)

print(f"Has one-body integrals: {hamiltonian.has_one_body_integrals()}")
print(f"Has two-body integrals: {hamiltonian.has_two_body_integrals()}")
print(f"Core energy: {hamiltonian.get_core_energy():.6f}")
# end-cell-create-ppp
################################################################################

################################################################################
# start-cell-create-ising
# Create a transverse-field Ising model on a chain
lattice = LatticeGraph.chain(6)
qubit_hamiltonian = create_ising_hamiltonian(lattice, j=1.0, h=0.5)

print(f"Ising Hamiltonian ({lattice.num_sites} qubits):")
print(f"  Number of Pauli terms: {len(qubit_hamiltonian.pauli_strings)}")
print(f"  Is Hermitian: {qubit_hamiltonian.is_hermitian()}")
# end-cell-create-ising
################################################################################

################################################################################
# start-cell-create-heisenberg
# Create an anisotropic Heisenberg model on a square lattice
lattice = LatticeGraph.square(3, 3)
qubit_hamiltonian = create_heisenberg_hamiltonian(
    lattice,
    jx=1.0,
    jy=1.0,
    jz=1.0,
    hz=0.5,  # External longitudinal field
)

print(f"Heisenberg Hamiltonian ({lattice.num_sites} qubits):")
print(f"  Number of Pauli terms: {len(qubit_hamiltonian.pauli_strings)}")
print(f"  Is Hermitian: {qubit_hamiltonian.is_hermitian()}")
# end-cell-create-heisenberg
################################################################################

################################################################################
# start-cell-site-dependent
# Site-dependent Hubbard model: different U on each site
lattice = LatticeGraph.chain(4)
U_values = np.array([2.0, 4.0, 4.0, 2.0])  # Weaker repulsion at edges
hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=U_values)

# Bond-dependent Ising model: different coupling on each bond
j_matrix = np.ones((4, 4))
j_matrix[0, 1] = 2.0  # Stronger coupling on first bond
h_fields = np.array([0.5, 0.3, 0.3, 0.5])  # Inhomogeneous field
qh = create_ising_hamiltonian(lattice, j=j_matrix, h=h_fields)
# end-cell-site-dependent
################################################################################

################################################################################
# start-cell-solve-hubbard
# Create a 4-site half-filled Hubbard chain
lattice = LatticeGraph.chain(4)
hamiltonian = create_hubbard_hamiltonian(lattice, epsilon=0.0, t=1.0, U=4.0)

# Run exact diagonalization (CASCI) with half filling (2 alpha + 2 beta electrons)
mc_calculator = algorithms.create("multi_configuration_calculator", "macis_cas")
energy, wavefunction = mc_calculator.run(hamiltonian, 2, 2)

print(f"Ground state energy: {energy:.6f} a.u.")
# end-cell-solve-hubbard
################################################################################

################################################################################
# start-cell-potentials
lattice = LatticeGraph.chain(4)

# Ohno potential: V_ij = U_ij / sqrt(1 + (U_ij * epsilon_r * R_ij)^2)
V_ohno = ohno_potential(lattice, U=0.414, R=2.65, epsilon_r=1.0)

# Mataga-Nishimoto potential: V_ij = U_ij / (1 + U_ij * epsilon_r * R_ij)
V_mn = mataga_nishimoto_potential(lattice, U=0.414, R=2.65, epsilon_r=1.0)

# Custom pairwise potential using a user-defined function
V_custom = pairwise_potential(
    lattice,
    U=0.414,
    R=2.65,
    func=lambda _i, _j, uij, rij: uij / (1.0 + rij),
)

print(f"Ohno V(0,1) = {V_ohno[0, 1]:.4f}")
print(f"Mataga-Nishimoto V(0,1) = {V_mn[0, 1]:.4f}")
print(f"Custom V(0,1) = {V_custom[0, 1]:.4f}")
# end-cell-potentials
################################################################################

################################################################################
# start-cell-solve-ising
# Create a transverse-field Ising model on a 6-site chain
lattice = LatticeGraph.chain(6)
qubit_hamiltonian = create_ising_hamiltonian(lattice, j=1.0, h=0.5)

# Solve for the ground state energy using exact diagonalization
q_solver = algorithms.create("qubit_hamiltonian_solver", "qdk_sparse_matrix_solver")
energy, ground_state = q_solver.run(qubit_hamiltonian)

print(f"Ising ground state energy: {energy:.6f}")
print(f"Number of qubits: {qubit_hamiltonian.num_qubits}")
# end-cell-solve-ising
################################################################################
