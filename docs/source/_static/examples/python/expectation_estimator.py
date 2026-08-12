"""Expectation estimator usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QuantumErrorProfile,
    QubitOperator,
)

# Create expectation estimator using Qsharp simulator as backend
qdk_estimator = create("expectation_estimator", "qdk")
# end-cell-create
################################################################################

################################################################################
# start-cell-qdk
# Define a simple quantum circuit in QASM and a qubit operator
circuit = Circuit(
    qasm="""
    include "stdgates.inc";
    qubit[2] q;
    rz(pi) q[0];
    x q[0];
    cx q[0], q[1];
    """
)
qubit_hamiltonian = QubitOperator(["ZZ"], np.array([1.0]))

# Run expectation estimation using Qsharp simulator without noise
qdk_estimator = create(
    "expectation_estimator",
    "qdk",
    circuit_executor=AlgorithmRef("circuit_executor", "qdk_sparse_state_simulator"),
)
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit, qubit_hamiltonian, total_shots=1000
)
print(
    "Energy expectation value from noiseless QDK Simulator: "
    f"{energy_expectation_results.energy_expectation_value}"
)

# Create expectation estimator using Qsharp simulator with depolarizing noise
noise_model = QuantumErrorProfile(
    name="noise model",
    description="Noise model for QDK full state simulator",
    errors={
        "rz": {"depolarizing_error": 0.005},
        "h": {"depolarizing_error": 0.005},
        "s": {"depolarizing_error": 0.005},
        "cx": {"depolarizing_error": 0.007},
    },
)
qdk_estimator = create(
    "expectation_estimator",
    "qdk",
    circuit_executor=AlgorithmRef(
        "circuit_executor", "qdk_full_state_simulator", type="cpu"
    ),
)
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit,
    qubit_hamiltonian,
    total_shots=1000,
    noise_model=noise_model,
)
print(
    "Expectation value from QDK Simulator with depolarizing noise: "
    f"{energy_expectation_results.energy_expectation_value}"
)
# end-cell-qdk
################################################################################

################################################################################
# start-cell-qiskit
# Run expectation estimation using Qiskit Aer simulator without noise
qdk_estimator = create(
    "expectation_estimator",
    "qdk",
    circuit_executor=AlgorithmRef("circuit_executor", "qiskit_aer_simulator"),
)
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit, qubit_hamiltonian, total_shots=1000
)
print(
    f"Energy expectation value from Qiskit Aer Simulator: {energy_expectation_results.energy_expectation_value}"
)

# Create expectation estimator using Qiskit Aer simulator with noise model

noise_model = QuantumErrorProfile(
    name="noise model",
    description="Noise model for Qiskit Aer simulator",
    errors={
        "rz": {"depolarizing_error": 0.005},
        "sx": {"depolarizing_error": 0.005},
        "cx": {"depolarizing_error": 0.007},
    },
)
qdk_estimator = create(
    "expectation_estimator",
    "qdk",
    circuit_executor=AlgorithmRef("circuit_executor", "qiskit_aer_simulator"),
)
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit,
    qubit_hamiltonian,
    total_shots=1000,
    noise_model=noise_model,
)
print(
    "Expectation value from Qiskit Aer Simulator with noise: "
    f"{energy_expectation_results.energy_expectation_value}"
)
# end-cell-qiskit
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

print(registry.available("expectation_estimator"))
# ['qdk']
print(registry.available("circuit_executor"))
# ['qdk_full_state_simulator', 'qdk_sparse_state_simulator', 'qiskit_aer_simulator']
# end-cell-list-implementations
################################################################################
################################################################################
