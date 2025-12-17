"""Dynamical correlation examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-mp2-example
from qdk_chemistry.data import Structure, Ansatz
from qdk_chemistry.algorithms import create
import numpy as np

# Create a simple structure
coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Run initial SCF
scf_solver = create("scf_solver")
E_hf, wfn_hf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="def2-svp"
)

# Create hamiltonian constructor
hamiltonian_constructor = create("hamiltonian_constructor")

# Construct Hamiltonian from orbitals
hamiltonian = hamiltonian_constructor.run(wfn_hf.get_orbitals())

# Create ansatz for Mp2 calculation
ansatz = Ansatz(hamiltonian, wfn_hf)

# Run MP2
mp2_calculator = create("dynamical_correlation_calculator")

# Get energies
mp2_total_energy, final_wavefunction = mp2_calculator.run(ansatz)

# If desired we can extract only the correlation energy
mp2_corr_energy = mp2_total_energy - E_hf
# end-cell-mp2-example
