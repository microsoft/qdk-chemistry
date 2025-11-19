"""Model Hamiltonian general example."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

# start-cell-1
import numpy as np
import qdk_chemistry.algorithms as algorithms
from qdk_chemistry.data import Hamiltonian, ModelOrbitals

# Define a 4-site Hubbard chain with restricted "orbitals"
num_sites = 4
model_orbitals = ModelOrbitals(num_sites, restricted=True)

# Define model parameters
t = -1.0  # Hopping parameter
U = 4.0  # On-site interaction strength

# Construct one-body integrals (hopping terms)
# For a 1D chain with nearest-neighbor hopping
one_body = np.zeros((num_sites, num_sites))
for i in range(num_sites - 1):
    one_body[i, i + 1] = t
    one_body[i + 1, i] = t

# Construct two-body integrals (on-site repulsion)
# Two-body integrals are stored as a flattened vector
two_body = np.zeros(num_sites**4)
for i in range(num_sites):
    # On-site repulsion: U * n_i↑ * n_i↓
    # In physicist's notation: <ii|ii>
    idx = i * num_sites**3 + i * num_sites**2 + i * num_sites + i
    two_body[idx] = U

# Core energy (usually 0 for model Hamiltonians)
core_energy = 0.0

# Create empty inactive Fock matrix
inactive_fock = np.zeros((0, 0))

# Create the Hamiltonian
hamiltonian = Hamiltonian(
    one_body, two_body, model_orbitals, core_energy, inactive_fock
)
# end-cell-1
# start-cell-2
# Create a multi-configuration calculator
mc_calculator = algorithms.create("multi_configuration_calculator", "macis_cas")

# Run calculation for 2 alpha and 2 beta electrons
energy, wavefunction = mc_calculator.run(hamiltonian, 2, 2)

print(f"Ground state energy: {energy} a.u.")
# end-cell-2
