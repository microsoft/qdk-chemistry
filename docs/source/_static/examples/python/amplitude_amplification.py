"""Amplitude amplification usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Number of Grover iterates. Choose it from an estimate of the overlap a,
# the success probability after k rounds is sin^2((2k+1) arcsin(sqrt(a))).
amplitude_amplification = create("amplitude_amplification", "base", rounds=2)

# end-cell-create
################################################################################

################################################################################
# start-cell-run
import math

import numpy as np
from qdk_chemistry.algorithms import create, phase_marking_oracle
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

# 1. A two-qubit Hamiltonian
qubit_hamiltonian = QubitOperator(
    pauli_strings=["ZI", "IZ"], coefficients=np.array([math.pi / 4.0, math.pi / 4.0])
)

# 2. A guiding state with 0.3 amplitude on the target eigenvector |11>
state_vector = [0.0, 0.0, 0.0, 0.0]
state_vector[3] = 0.3
state_vector[0] = math.sqrt(1.0 - 0.3**2)
prep_parameters = {
    "rowMap": [1, 0],
    "stateVector": state_vector,
    "expansionOps": [],
    "numQubits": 2,
}
state_preparation = Circuit(
    qsharp_factory=QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
        parameter=prep_parameters,
    ),
    qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(prep_parameters),
)

# 3. Build a measurement-free QPE circuit
num_bits = 4
unitary = AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True)
builder = create("qpe_circuit_builder", "qdk_standard")
builder.settings().update("num_bits", num_bits)
builder.settings().update(
    "controlled_circuit_mapper",
    AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
)
builder.settings().update("unitary_builder", unitary)
builder.settings().update("measure_phase", False)
state_prep_oracle = builder.run(
    state_preparation=state_preparation, qubit_hamiltonian=qubit_hamiltonian
)[0]

# 4. Derive the register layout.
params = state_prep_oracle._qsharp_factory.parameter
num_system_qubits = len(params["systems"])
num_ancilla_qubits = params["numAncillaQubits"]
num_qubits = params["numBits"] + num_system_qubits + num_ancilla_qubits

# 5. Mark the phase bin of the target eigenvector and require clean signal ancillas.
accepted = (8, 9)
good_state_oracle = phase_marking_oracle(
    num_bits,
    accepted,
    list(range(num_system_qubits, num_system_qubits + num_ancilla_qubits)),
)

# 6. Amplify, then execute
amplitude_amplification = create("amplitude_amplification", "base", rounds=2)
circuit = amplitude_amplification.run(
    state_prep_oracle, good_state_oracle, num_qubits=num_qubits
)

executor = create("circuit_executor", "qdk_sparse_state_simulator")
shots = 400
counts = executor.run(circuit, shots=shots).bitstring_counts

# end-cell-run
################################################################################
