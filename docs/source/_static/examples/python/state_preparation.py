"""State preparation examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef

# Create a StatePreparation instance
sparse_prep = create("state_prep", "sparse_isometry")
dense_prep = create("state_prep", "dense_pure_state")
regular_prep = create("state_prep", "qiskit_regular_isometry")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Transpilation is configured on the algorithm that emits the Qiskit circuit.
regular_prep.settings().set("transpile", True)
regular_prep.settings().set("basis_gates", ["rz", "cz", "sdg", "h"])
regular_prep.settings().set("transpile_optimization_level", 3)

# Select other dense preparation method for sparse isometry
sparse_isometry_with_qiskit_dense_prep = create(
    "state_prep",
    "sparse_isometry",
    dense_state_prep=AlgorithmRef(
        "state_prep",
        "qiskit_regular_isometry",
        transpile=True,
        basis_gates=["rz", "cz", "sdg", "h"],
        transpile_optimization_level=3,
    ),
)
# end-cell-configure
################################################################################

################################################################################
# start-cell-configure-binary-encoding
# Compress the reduced subspace with binary encoding
binary_encoded_prep = create("state_prep", "sparse_isometry", binary_encoding=True)

# Select the algorithm that prepares the compressed dense register
binary_encoded_prep.settings().set(
    "dense_state_prep", AlgorithmRef("state_prep", "dense_pure_state")
)
# end-cell-configure-binary-encoding
################################################################################

################################################################################
# start-cell-run
import numpy as np
from qdk_chemistry.data import Structure

# Specify a structure
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
symbols = ["H", "H"]
structure = Structure(coords, symbols=symbols)

# Run scf
scf_solver = create("scf_solver")
E_scf, wfn_scf = scf_solver.run(
    structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g"
)

# Compute the Hamiltonian
hamiltonian_constructor = create("hamiltonian_constructor")
hamiltonian = hamiltonian_constructor.run(wfn_scf.get_orbitals())

# Compute CAS wavefunction
cas_solver = create("multi_configuration_calculator", "macis_cas")
E_cas, wfn_cas = cas_solver.run(hamiltonian, 1, 1)

# Construct the circuit
regular_circuit = regular_prep.run(wfn_cas)
sparse_circuit = sparse_prep.run(wfn_cas)
dense_circuit = dense_prep.run(wfn_cas)
print(f"Regular isometry circuit:\n{regular_circuit.get_qiskit_circuit()}")
print(f"Sparse isometry circuit:\n{sparse_circuit.get_qsharp_circuit()}")
print(f"Dense pure state circuit:\n{dense_circuit.get_qsharp_circuit()}")
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

print(registry.available("state_prep"))
# ['dense_pure_state', 'sparse_isometry', 'alias_sampling', 'qrom', 'qiskit_regular_isometry']
# The order follows registration order, and 'qiskit_regular_isometry' only appears
# when the Qiskit interop plugin is installed.
# end-cell-list-implementations
################################################################################
