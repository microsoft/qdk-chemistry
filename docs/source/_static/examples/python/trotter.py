"""Trotter builder usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import QubitHamiltonian

# Build a simple qubit Hamiltonian as a sum of 5 Pauli strings
qubit_hamiltonian = QubitHamiltonian(
    pauli_strings=["ZZI", "XXI", "YYI", "ZII", "IXZ"],
    coefficients=np.array([0.5, 0.3, 0.2, 0.1, -0.4]),
)
print(f"Qubit Hamiltonian has {qubit_hamiltonian.num_qubits} qubits")
print("Pauli strings:", qubit_hamiltonian.pauli_strings)
# end-cell-create
################################################################################

################################################################################
# start-cell-trotter
# Create and run a second-order Trotter builder
trotter = create("hamiltonian_unitary_builder", "trotter", order=2, time=1.0)
unitary_rep = trotter.run(qubit_hamiltonian)
container = unitary_rep.get_container()
print(f"Number of step terms: {len(container.step_terms)}")
print(f"Step repetitions: {container.step_reps}")
# end-cell-trotter
################################################################################
