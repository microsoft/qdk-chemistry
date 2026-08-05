"""Amplitude amplification usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create the default amplitude amplification algorithm
amplitude_amplification = create("amplitude_amplification", "qdk")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure
# Number of Grover iterates. Choose it from an estimate of the overlap a:
# the success probability after k rounds is sin^2((2k+1) arcsin(sqrt(a))).
amplitude_amplification = create("amplitude_amplification", "qdk", rounds=2)

print("rounds:", amplitude_amplification.settings().get("rounds"))
# end-cell-configure
################################################################################

################################################################################
# start-cell-oracle
from qdk_chemistry.algorithms import phase_marking_oracle

# Mark a half-open range of phase bins on an 8-qubit phase register
selected_bins = phase_marking_oracle(8, (12, 15))
lower_bins = phase_marking_oracle(8, (0, 32))
upper_bins = phase_marking_oracle(8, (224, 256))
# end-cell-oracle
################################################################################

################################################################################
# start-cell-trusted-bins
# Also require that the block-encoding signal ancillas return to |0>
trusted_bins = phase_marking_oracle(8, (12, 15), [2, 3])
# end-cell-trusted-bins
################################################################################

################################################################################
# start-cell-run
import math

import numpy as np
from qdk_chemistry.algorithms import create, phase_marking_oracle
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

# 1. A two-qubit Hamiltonian H = (pi/4) ZI + (pi/4) IZ, with |11> at energy -pi/2
qubit_hamiltonian = QubitOperator(
    pauli_strings=["ZI", "IZ"], coefficients=np.array([math.pi / 4.0, math.pi / 4.0])
)

# 2. A guiding state with only 0.3 amplitude on the target eigenvector |11>
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

# 3. Build a measurement-free QPE circuit: this is the state preparation to amplify
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

# 4. Work out the register layout: phase qubits, then system qubits, then walk ancillas
unitary_algorithm = create(
    unitary.algorithm_type, unitary.algorithm_name, **unitary.settings
)
num_system_qubits = qubit_hamiltonian.num_qubits
num_ancilla_qubits = (
    unitary_algorithm.run(qubit_hamiltonian).get_num_qubits() - num_system_qubits
)
num_qubits = num_bits + num_system_qubits + num_ancilla_qubits

# 5. Mark the phase bin of the target eigenvector, requiring clean signal ancillas.
#    The walk maps the eigenvalue -pi/2 to phase 1/2, which is bin 8 of 16.
accepted = (8, 9)
good_state_oracle = phase_marking_oracle(
    num_bits,
    accepted,
    list(range(num_system_qubits, num_system_qubits + num_ancilla_qubits)),
)

# 6. Amplify, then execute
amplitude_amplification = create("amplitude_amplification", "qdk", rounds=2)
circuit = amplitude_amplification.run(
    state_prep_oracle, good_state_oracle, num_qubits=num_qubits
)

executor = create("circuit_executor", "qdk_sparse_state_simulator")
shots = 400
counts = executor.run(circuit, shots=shots).bitstring_counts

# 7. Accept a shot when the ancillas are |0> and the phase lands in the window.
#    The whole register is measured and the executor reverses the Q# results, so
#    the phase register reads MSB first at the end of the key.
accepted_counts: dict[str, int] = {}
for bitstring, count in counts.items():
    phase_bits, ancilla_bits = bitstring[-num_bits:], bitstring[:num_ancilla_qubits]
    if any(bit != "0" for bit in ancilla_bits):
        continue
    if not accepted[0] <= int(phase_bits, 2) < accepted[1]:
        continue
    accepted_counts[phase_bits] = accepted_counts.get(phase_bits, 0) + count

print("accepted phase counts:", accepted_counts)
print("acceptance probability:", sum(accepted_counts.values()) / shots)
# end-cell-run
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

# List all registered amplitude amplification implementations
implementations = registry.available("amplitude_amplification")
print(implementations)  # e.g. ['qdk']
# end-cell-list-implementations
################################################################################
