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
amplitude_amplification = create("amplitude_amplification", "qdk_base", rounds=2)

# end-cell-create
################################################################################

################################################################################
# start-cell-run
import math

import numpy as np
from qdk_chemistry.algorithms import create
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

# 3. Build a measurement-free QPE circuit. This whole circuit is the preparation that
# gets amplified, so the phase register and its ancillas stay inside the amplified
# register and every round reflects about the full prepared state.
num_bits = 4
walk = AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True)
builder = create(
    "qpe_circuit_builder",
    "qdk_standard",
    num_bits=num_bits,
    controlled_circuit_mapper=AlgorithmRef(
        "controlled_circuit_mapper", "prepare_select_prepare"
    ),
    unitary_builder=walk,
    measure_phase=False,
)
state_prep_oracle = builder.run(
    state_preparation=state_preparation, qubit_hamiltonian=qubit_hamiltonian
)[0]
# The same walk, as a unitary representation: this is what the QPE circuit estimates.
unitary_representation = create(
    "hamiltonian_unitary_builder", "lcu", quantum_walk=True
).run(qubit_hamiltonian)

# 4. Mark the phase bins holding the target eigenvalue, naming it by energy. The oracle
# reads the register width and the post-processing equation phases are read with off that
# unitary, and inverts the equation to turn the energy back into a phase. Here the walk
# gives E = lambda cos(2 pi phi), so the target eigenvector |11> at E = -pi/4 - pi/4 =
# -lambda sits at phi = 0.5, that is bin 0.5 * 16 = 8. Both signs of the phase are marked,
# because the walk has eigenvalues exp(+-i arccos(E / lambda)); here bin 8 is its own mirror.
good_state_oracle = create(
    "subspace_oracle",
    "qdk_qpe_subspace",
    target_energy=-qubit_hamiltonian.schatten_norm,
).run(state_prep_oracle, unitary_representation)

# 5. Amplify, then execute. The overlap is a = 0.3**2 = 0.09, so
# arcsin(sqrt(a)) = 0.3047 and 2 rounds put (2k+1) arcsin(sqrt(a)) at 1.523, just under
# pi/2: the probability of landing in bin 8 rises from 0.09 to sin^2(1.523) = 0.998.
# A third round would overshoot back down to 0.79.
amplitude_amplification = create("amplitude_amplification", "qdk_base", rounds=2)
circuit = amplitude_amplification.run(state_prep_oracle, good_state_oracle)

executor = create("circuit_executor", "qdk_sparse_state_simulator")
shots = 400
# Keys are big-endian over the whole register, so the phase register is the last
# num_bits characters: almost every shot reads "1000", the binary form of bin 8.
counts = executor.run(circuit, shots=shots).bitstring_counts

# end-cell-run
################################################################################
