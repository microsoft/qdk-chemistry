"""Qubit mapper usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a QubitMapper instance
qubit_mapper = create("qubit_mapper", "qiskit")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Configure the encoding strategy
qubit_mapper.settings().set("encoding", "jordan-wigner")
# end-cell-configure
################################################################################

################################################################################
# start-cell-run
from pathlib import Path
from qdk_chemistry.data import Structure

# Read a molecular structure from XYZ file
structure = Structure.from_xyz_file(Path(".") / "../data/water.structure.xyz")

# Perform an SCF calculation to generate initial orbitals
scf_solver = create("scf_solver", basis_set="cc-pvdz")
_, wfn_hf = scf_solver.run(structure, charge=0, spin_multiplicity=1)

# Select an active space
active_space_selector = create(
    "active_space_selector",
    algorithm_name="qdk_valence",
    num_active_electrons=4,
    num_active_orbitals=6,
)
active_wfn = active_space_selector.run(wfn_hf)
active_orbitals = active_wfn.get_orbitals()

# Construct Hamiltonian in the active space
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(active_orbitals)

# Map the fermionic Hamiltonian to a qubit Hamiltonian
qubit_hamiltonian = qubit_mapper.run(hamiltonian)
print(f"Qubit Hamiltonian has {qubit_hamiltonian.num_qubits()} qubits")
# end-cell-run
################################################################################
