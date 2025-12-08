"""Example showing design principles: immutability and data classes."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-scf-create
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

scf_solver = create("scf_solver")
# end-cell-scf-create
################################################################################

################################################################################
# start-cell-scf-settings
print(f"Available settings: {scf_solver.settings().items()}")
scf_solver.settings().set("max_iterations", 100)
# end-cell-scf-settings
################################################################################

################################################################################
# start-cell-data-flow
# Create a Structure (coordinates in Bohr/atomic units) or read from file
# Data classes in QDK/Chemistry are immutable by design (coordinates in Bohr/atomic units)
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, ["H", "H"])

# Configure and run SCF calculation
scf_solver = create("scf_solver")
print(f"Available SCF settings: {scf_solver.settings().items()}")
scf_solver.settings().set("basis_set", "cc-pvdz")
scf_energy, scf_wavefunction = scf_solver.run(structure, charge=0, spin_multiplicity=1)
print(f"Total number of electrons: {scf_wavefunction.get_total_num_electrons()}")
print(f"Orbital occupations: {scf_wavefunction.get_total_orbital_occupations()}")

# Select active space orbitals
active_space_selector = create(
    "active_space_selector",
    algorithm_name="qdk_valence",
)
print(
    f"Available active space selector settings: {active_space_selector.settings().items()}"
)
active_space_selector.settings().set("num_active_orbitals", 2)
active_space_selector.settings().set("num_active_electrons", 2)
active_wfn = active_space_selector.run(scf_wavefunction)
active_orbitals = active_wfn.get_orbitals()
print(f"Active orbitals: {active_orbitals}")

# Create Hamiltonian with active space
ham_constructor = create("hamiltonian_constructor")
print(
    f"Available Hamiltonian constructor settings: {ham_constructor.settings().items()}"
)
ham_constructor.settings().set("eri_method", "incore")
hamiltonian = ham_constructor.run(active_orbitals)
print("Active Space Hamiltonian:\n", hamiltonian.get_summary())

mc = create("multi_configuration_calculator")
print(f"Available multi-configuration calculator settings: {mc.settings().items()}")
mc.settings().set("davidson_iterations", 300)
E_cas, wfn_cas = mc.run(
    hamiltonian, n_active_alpha_electrons=1, n_active_beta_electrons=1
)
print(
    f"CASCI energy is {E_cas:.3f} Hartree, and the electron correlation energy is {E_cas - scf_energy:.3f} Hartree"
)
# end-cell-data-flow
################################################################################
