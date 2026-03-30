"""Energy estimator usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Circuit, QuantumErrorProfile, QubitHamiltonian

# Create energy estimator using Qsharp simulator as backend
qdk_estimator = create("energy_estimator", "qdk")
# end-cell-create
################################################################################

################################################################################
# start-cell-qdk
# Define a simple quantum circuit in QASM and a qubit Hamiltonian
circuit = Circuit(
    qasm="""
    include "stdgates.inc";
    qubit[2] q;
    rz(pi) q[0];
    x q[0];
    cx q[0], q[1];
    """
)
qubit_hamiltonian = QubitHamiltonian(["ZZ"], np.array([1.0]))

# Run energy estimation using Qsharp simulator without noise
circuit_executor = create("circuit_executor", "qdk_sparse_state_simulator")
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit, qubit_hamiltonian, circuit_executor, total_shots=1000
)
print(
    "Energy expectation value from noiseless QDK Simulator: "
    f"{energy_expectation_results.energy_expectation_value}"
)

# Create energy estimator using Qsharp simulator with depolarizing noise
noise_model = QuantumErrorProfile(
    name="noise model",
    description="Noise model for QDK full state simulator",
    errors={
        "rz": {
            "type": "depolarizing_error",
            "rate": 0.005,
            "num_qubits": 1,
        },
        "h": {
            "type": "depolarizing_error",
            "rate": 0.005,
            "num_qubits": 1,
        },
        "s": {
            "type": "depolarizing_error",
            "rate": 0.005,
            "num_qubits": 1,
        },
        "cx": {
            "type": "depolarizing_error",
            "rate": 0.007,
            "num_qubits": 2,
        },
    },
)
circuit_executor = create("circuit_executor", "qdk_full_state_simulator", type="cpu")
qdk_estimator = create("energy_estimator", "qdk")
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit,
    qubit_hamiltonian,
    circuit_executor,
    total_shots=1000,
    noise_model=noise_model,
)
print(
    "Energy expectation value from QDK Simulator with depolarizing noise: "
    f"{energy_expectation_results.energy_expectation_value}"
)
# end-cell-qdk
################################################################################

################################################################################
# start-cell-qiskit
# Run energy estimation using Qiskit Aer simulator without noise
qiskit_aer_simulator = create("circuit_executor", "qiskit_aer_simulator")
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit, qubit_hamiltonian, qiskit_aer_simulator, total_shots=1000
)
print(
    f"Energy expectation value from Qiskit Aer Simulator: {energy_expectation_results.energy_expectation_value}"
)

# Create energy estimator using Qiskit Aer simulator with noise model

noise_model = QuantumErrorProfile(
    name="noise model",
    description="Noise model for Qiskit Aer simulator",
    errors={
        "rz": {
            "type": "depolarizing_error",
            "rate": 0.005,
            "num_qubits": 1,
        },
        "sx": {
            "type": "depolarizing_error",
            "rate": 0.005,
            "num_qubits": 1,
        },
        "cx": {
            "type": "depolarizing_error",
            "rate": 0.007,
            "num_qubits": 2,
        },
    },
)
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit,
    qubit_hamiltonian,
    qiskit_aer_simulator,
    total_shots=1000,
    noise_model=noise_model,
)
print(
    "Energy expectation value from Qiskit Aer Simulator with noise: "
    f"{energy_expectation_results.energy_expectation_value}"
)
# end-cell-qiskit
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry  # noqa: E402

print(registry.available("energy_estimator"))
# ['qdk']
print(registry.available("circuit_executor"))
# ['qdk_full_state_simulator', 'qdk_sparse_state_simulator', 'qiskit_aer_simulator']
# end-cell-list-implementations
################################################################################
################################################################################
