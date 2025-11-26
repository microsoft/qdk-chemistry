"""Reference-derived calculator examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure, Ansatz, Element
import numpy as np

# Create a simple structure
coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
elements = [Element.H, Element.H]
structure = Structure(coords, elements=elements)

# Run SCF calculation
scf_solver = create("scf_solver")
hf_energy, hf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# Build Hamiltonian
orbitals = hf_wavefunction.get_orbitals()
ham_constructor = create("hamiltonian_constructor")
hamiltonian = ham_constructor.run(orbitals)

# Create ansatz for MP2 calculation
ansatz = Ansatz(hamiltonian, hf_wavefunction)

# Run MP2 calculation
mp2_calculator = create("reference_derived_calculator", "microsoft_mp2_calculator")
mp2_energy, mp2_wavefunction = mp2_calculator.run(ansatz)

# If desired, we can extract only the correlation energy
mp2_corr_energy = mp2_energy - hf_energy
