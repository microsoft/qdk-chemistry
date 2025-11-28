"""Energy estimator usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

# Import noise models for QDK simulator examples
from qsharp import DepolarizingNoise

# Import noise models for Qiskit Aer simulator examples
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.energy_estimator.qsharp import QDKEnergyEstimator
from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.plugins.qiskit.energy_estimator import QiskitEnergyEstimator

# Define a simple quantum circuit in QASM and a qubit Hamiltonian
circuit_qasm = """
    include "stdgates.inc";
    qubit[2] q;
    rz(pi) q[0];
    x q[0];
    cx q[0], q[1];
    """
qubit_hamiltonians = [QubitHamiltonian(["ZZ"], np.array([1.0]))]

# Create energy estimator using Qsharp simulator as backend
qdk_estimator = create("energy_estimator", "qdk_base_simulator")
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit_qasm, qubit_hamiltonians, total_shots=1000
)
print(
    f"Energy expectation value from QDK Simulator: {energy_expectation_results.energy_expectation_value}"
)

# Create energy estimator using Qsharp simulator with depolarizing noise
noise_model = DepolarizingNoise(0.01)
qdk_estimator = QDKEnergyEstimator(noise_model=noise_model)
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit_qasm, qubit_hamiltonians, total_shots=1000
)
print(
    "Energy expectation value from QDK Simulator with depolarizing noise: "
    f"{energy_expectation_results.energy_expectation_value}"
)

# Create energy estimator using Qsharp simulator with qubit loss
qdk_estimator = QDKEnergyEstimator(qubit_loss=0.05)
energy_expectation_results, measurement_data = qdk_estimator.run(
    circuit_qasm, qubit_hamiltonians, total_shots=1000
)
print(
    "Energy expectation value from QDK Simulator with qubit loss: "
    f"{energy_expectation_results.energy_expectation_value}"
)

# Create energy estimator using Qiskit Aer simulator as backend
qiskit_estimator = create("energy_estimator", "qiskit_aer_simulator")
energy_expectation_results, measurement_data = qiskit_estimator.run(
    circuit_qasm, qubit_hamiltonians, total_shots=1000
)
print(
    f"Energy expectation value from Qiskit Aer Simulator: {energy_expectation_results.energy_expectation_value}"
)

# Create energy estimator using Qiskit Aer simulator with noise model
noise_model = NoiseModel(basis_gates=["rz", "sx", "cx", "measure"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.005, 1), ["rz", "sx"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.007, 2), ["cx"])
backend_options = {"noise_model": noise_model}

qiskit_estimator = QiskitEnergyEstimator.from_backend_options(
    backend_options=backend_options
)
energy_expectation_results, measurement_data = qiskit_estimator.run(
    circuit_qasm, qubit_hamiltonians, total_shots=1000
)
print(
    "Energy expectation value from Qiskit Aer Simulator with noise: "
    f"{energy_expectation_results.energy_expectation_value}"
)
