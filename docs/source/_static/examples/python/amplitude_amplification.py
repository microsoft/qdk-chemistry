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

# 1. A two-qubit Hamiltonian
qubit_hamiltonian = QubitOperator(
    pauli_strings=["ZI", "IZ"], coefficients=np.array([math.pi / 4.0, math.pi / 4.0])
)

# 2. A guiding state with 0.3 amplitude on the target eigenvector |11>
amplitude = 0.3
guiding_state = Wavefunction(
    StateVectorContainer(
        np.array([math.sqrt(1.0 - amplitude**2), amplitude]),
        [Configuration.from_bitstring("00"), Configuration.from_bitstring("11")],
        ModelOrbitals(2),
    )
)
state_preparation = create("state_prep", "dense_pure_state").run(guiding_state)

# 3. The good state oracle runs a QPE of the Hamiltonian on the prepared register, flips a
# flag when the phase lands in a bin whose energy is at most the target, then undoes the QPE.
# It is configured like a QPE circuit builder, plus the energy bound. Put the bound between
# the eigenvalue to amplify and the next one up: the spectrum here is {-lambda, 0, 0,
# +lambda}, so -lambda/2 keeps |11> and drops the rest. The bins come from the equation QPE
# results are read with, here E = lambda cos(2 pi phi), which puts the accepted band at
# phi in [1/3, 2/3], that is bins 6 through 10 of 16.
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
    target_energy=-qubit_hamiltonian.schatten_norm / 2,
).run(qubit_hamiltonian)

# 4. Amplify the state preparation against that oracle, then execute. The overlap is
# a = 0.3**2 = 0.09, so arcsin(sqrt(a)) = 0.3047 and 2 rounds put (2k+1) arcsin(sqrt(a)) at
# 1.523, just under pi/2: the probability of the target eigenvector rises from 0.09 to
# sin^2(1.523) = 0.998. A third round would overshoot back down to 0.79.
amplitude_amplification = create("amplitude_amplification", "qdk_base", rounds=2)
circuit = amplitude_amplification.run(state_preparation, good_state_oracle)

executor = create("circuit_executor", "qdk_sparse_state_simulator")
shots = 400
# Only the two system qubits are measured; the phase register is uncomputed inside the
# oracle. Almost every shot reads "11".
counts = executor.run(circuit, shots=shots).bitstring_counts

# end-cell-run
################################################################################
