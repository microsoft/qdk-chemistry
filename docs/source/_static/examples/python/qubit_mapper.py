"""Qubit mapper usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from pathlib import Path
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

mapper = create("qubit_mapper", "qiskit", encoding="jordan-wigner")
# end-cell-create
################################################################################

################################################################################
# start-cell-example
# Read a molecular structure from XYZ file
structure = Structure.from_xyz_file(Path(".") / "../data/water.structure.xyz")

# Perform an SCF calculation to generate initial orbitals
scf_solver = create("scf_solver")
_, wfn_hf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)
print(f"Orbital occupancies: {wfn_hf.get_total_orbital_occupations()}")

# Select an active space
active_space_selector = create(
    "active_space_selector",
    algorithm_name="qdk_valence",
    num_active_electrons=4,
    num_active_orbitals=6,
)
active_wfn = active_space_selector.run(wfn_hf)
active_orbitals = active_wfn.get_orbitals()
print(active_orbitals.get_summary())

# Construct Hamiltonian in the active space and print its summary
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(active_orbitals)
print("Active space Hamiltonian:\n", hamiltonian.get_summary())

# Create the qubit Hamiltonian with the mapper
qubit_mapper = create("qubit_mapper", algorithm_name="qiskit", encoding="jordan-wigner")
qubit_hamiltonian = qubit_mapper.run(hamiltonian)
# end-cell-example
################################################################################
