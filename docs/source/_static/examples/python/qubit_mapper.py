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
scf_solver = create("scf_solver")
_, wfn_hf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="cc-pvdz"
)

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
print(f"Qubit Hamiltonian has {qubit_hamiltonian.num_qubits} qubits")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

print(registry.available("qubit_mapper"))
# ['qdk', 'qiskit']
# end-cell-list-implementations
################################################################################

################################################################################
# start-cell-qdk-mapper
from qdk_chemistry.algorithms import create as create_algorithm

# Create a native QDK QubitMapper instance
qdk_mapper = create_algorithm("qubit_mapper", "qdk")

# Configure the encoding (jordan-wigner or bravyi-kitaev)
qdk_mapper.settings().set("encoding", "jordan-wigner")

# Optional: configure thresholds for numerical precision
qdk_mapper.settings().set("threshold", 1e-12)
qdk_mapper.settings().set("integral_threshold", 1e-12)

# Map the fermionic Hamiltonian to a qubit Hamiltonian
qdk_qubit_hamiltonian = qdk_mapper.run(hamiltonian)
print(f"QDK mapper produced {len(qdk_qubit_hamiltonian.pauli_strings)} Pauli terms")
# end-cell-qdk-mapper
################################################################################

################################################################################
# start-cell-scbk-mapper
from qdk_chemistry.data import Symmetries

# Create an OpenFermion mapper with SCBK encoding
scbk_mapper = create_algorithm(
    "qubit_mapper", "openfermion", encoding="symmetry-conserving-bravyi-kitaev"
)

# Provide symmetries: the SCBK encoding needs active electron counts
# Option 1: construct directly
symmetries = Symmetries(n_alpha=2, n_beta=2)
scbk_hamiltonian = scbk_mapper.run(hamiltonian, symmetries)

# Option 2: derive from a wavefunction
symmetries = Symmetries.from_wavefunction(active_wfn)
scbk_hamiltonian = scbk_mapper.run(hamiltonian, symmetries)

# SCBK reduces the qubit count by 2 compared to JW or BK
print(
    f"SCBK mapper: {scbk_hamiltonian.num_qubits} qubits "
    f"(vs {qdk_qubit_hamiltonian.num_qubits} for JW)"
)
# end-cell-scbk-mapper
################################################################################
