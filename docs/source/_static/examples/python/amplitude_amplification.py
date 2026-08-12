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
from qdk_chemistry.data import (
    AlgorithmRef,
    Configuration,
    ModelOrbitals,
    QubitOperator,
    StateVectorContainer,
    Wavefunction,
)

# 1. A two-qubit Hamiltonian. Its spectrum is {+lambda, 0, 0, -lambda}, with |11> on top.
qubit_hamiltonian = QubitOperator(
    pauli_strings=["ZI", "IZ"], coefficients=np.array([-math.pi / 4.0, -math.pi / 4.0])
)

# 2. A guiding state with 0.3 amplitude on the target eigenvector |11>.
amplitude = 0.3
guiding_state = Wavefunction(
    StateVectorContainer(
        np.array([math.sqrt(1.0 - amplitude**2), amplitude]),
        [Configuration.from_bitstring("00"), Configuration.from_bitstring("11")],
        ModelOrbitals(2),
    )
)
state_preparation = create("state_prep", "dense_pure_state").run(guiding_state)

# 3. To mark the target state, a QPE is run on the prepared register, and a flag
# is flipped when the QPE phase lands in the desired range. Like any qpe_circuit_builder,
# run returns a list of circuits; this one always holds exactly the oracle.
good_state_oracle = create(
    "qpe_circuit_builder",
    "qdk_qpe_subspace",
    num_bits=4,
    unitary_builder=AlgorithmRef(
        "hamiltonian_unitary_builder", "lcu", quantum_walk=True
    ),
    controlled_circuit_mapper=AlgorithmRef(
        "controlled_circuit_mapper", "prepare_select_prepare"
    ),
    target_energy=qubit_hamiltonian.schatten_norm / 2,
).run(state_preparation, qubit_hamiltonian)[0]

# 4. Amplify the initial state against the qpe subspace marking oracle.
amplitude_amplification = create("amplitude_amplification", "qdk_base", rounds=2)
circuit = amplitude_amplification.run(state_preparation, good_state_oracle)

# 5. Run the circuit and measure. The |11> state should be amplified.
executor = create("circuit_executor", "qdk_sparse_state_simulator")
shots = 400
counts = executor.run(circuit, shots=shots).bitstring_counts

# end-cell-run
################################################################################
