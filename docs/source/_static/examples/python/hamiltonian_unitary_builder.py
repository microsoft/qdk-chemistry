"""Time evolution builder usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create a time evolution builder
trotter = create("hamiltonian_unitary_builder", "trotter")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure-trotter
# Configure a second-order Trotter builder with automatic step count
trotter = create("hamiltonian_unitary_builder", "trotter")
trotter.settings().set("order", 2)
trotter.settings().set("target_accuracy", 1e-3)
trotter.settings().set("error_bound", "commutator")
# end-cell-configure-trotter
################################################################################

################################################################################
# start-cell-configure-qdrift
# Configure qDRIFT with a fixed seed for reproducibility
qdrift = create("hamiltonian_unitary_builder", "qdrift")
qdrift.settings().set("num_samples", 500)
qdrift.settings().set("seed", 42)
qdrift.settings().set("merge_duplicate_terms", True)
qdrift.settings().set("commutation_type", "qubit_wise")
# end-cell-configure-qdrift
################################################################################

################################################################################
# start-cell-configure-pr
# Configure partially randomized builder
pr = create("hamiltonian_unitary_builder", "partially_randomized")
pr.settings().set("weight_threshold", 0.1)
pr.settings().set("trotter_order", 2)
pr.settings().set("num_random_samples", 200)
pr.settings().set("seed", 42)
# end-cell-configure-pr
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# 1. Setup molecule
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# 2. SCF
scf_solver = create("scf_solver")
E_scf, wfn_scf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# 3. Hamiltonian construction
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())

# 4. Qubit mapping
qubit_mapper = create("qubit_mapper", encoding="jordan-wigner")
qubit_ham = qubit_mapper.run(hamiltonian)

# 5. Build time evolution unitary
trotter = create("hamiltonian_unitary_builder", "trotter", order=2)
evolution = trotter.run(qubit_ham, time=0.1)

print(f"Container type: {evolution.get_container_type()}")
print(f"Number of qubits: {evolution.get_num_qubits()}")
print(evolution.get_summary())
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

# List all registered time evolution builder implementations
implementations = registry.available("hamiltonian_unitary_builder")
print(implementations)  # e.g. ['trotter', 'qdrift', 'partially_randomized']
# end-cell-list-implementations
################################################################################
