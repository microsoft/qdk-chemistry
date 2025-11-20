"""Example showing design principles: immutability and data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms import create

# start-cell-1
scf_solver = create("scf_solver")
# end-cell-1

# start-cell-2
scf_solver.settings().set("max_iterations", 100)
# end-cell-2


# start-cell-3
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# Create a Structure (coordinates in Bohr/atomic units) or read from file
# Data classes in QDK/Chemistry are immutable by design (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Configure and run SCF calculation
scf_solver = create("scf_solver")
scf_solver.settings().set("basis_set", "cc-pvdz")
scf_energy, scf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# Select active space orbitals
# TODO: Fix active_space_selector settings names
# active_space_selector = create("active_space_selector")
# active_space_selector.settings().set("num_active_orbitals", 6)
# active_space_selector.settings().set("num_active_electrons", 6)
# active_orbitals = active_space_selector.run(scf_wavefunction)
# active_indices = active_orbitals.get_active_orbital_indices()

# Create Hamiltonian with active space
ham_constructor = create("hamiltonian_constructor")
# ham_constructor.settings().set("active_orbitals", active_indices)
hamiltonian = ham_constructor.run(scf_wavefunction.get_orbitals())

# Run multi-configuration calculation
# TODO: Fix mc_solver once active space is working
# mc_solver = create("mc_solver")
# mc_energy, mc_wavefunction = mc_solver.run(hamiltonian)
# end-cell-3
