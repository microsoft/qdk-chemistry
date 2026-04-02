"""Circuit executor usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create
from qdk_chemistry.algorithms import create

# Create the default executor (QDK sparse-state simulator)
executor = create("circuit_executor")

# Or select a specific implementation
full_state = create("circuit_executor", "qdk_full_state_simulator")
sparse_state = create("circuit_executor", "qdk_sparse_state_simulator")
aer = create("circuit_executor", "qiskit_aer_simulator")
# end-cell-create
################################################################################

################################################################################
# start-cell-configure-qdk
# Configure the QDK full-state simulator
executor = create("circuit_executor", "qdk_full_state_simulator")
executor.settings().set("type", "cpu")
executor.settings().set("seed", 42)
# end-cell-configure-qdk
################################################################################

################################################################################
# start-cell-configure-qiskit
# Configure the Qiskit Aer simulator
executor = create("circuit_executor", "qiskit_aer_simulator")
executor.settings().set("method", "statevector")
executor.settings().set("seed", 42)
executor.settings().set("transpile_optimization_level", 0)
# end-cell-configure-qiskit
################################################################################

################################################################################
# start-cell-run
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Circuit

# Define a circuit in OpenQASM
circuit = Circuit(
    qasm="""
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    x q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """
)

# Execute with the QDK sparse-state simulator
executor = create("circuit_executor", "qdk_sparse_state_simulator")
result = executor.run(circuit, shots=1000)
print(f"Bitstring counts: {result.bitstring_counts}")
print(f"Total shots: {result.total_shots}")
# end-cell-run
################################################################################

################################################################################
# start-cell-noise
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Circuit, QuantumErrorProfile

circuit = Circuit(
    qasm="""
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    x q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """
)

# Define a noise model
noise_model = QuantumErrorProfile(
    name="depolarizing",
    description="Simple depolarizing noise model",
    errors={
        "x": {"type": "depolarizing_error", "rate": 0.005, "num_qubits": 1},
        "cx": {"type": "depolarizing_error", "rate": 0.007, "num_qubits": 2},
    },
)

# Execute with noise
executor = create("circuit_executor", "qdk_full_state_simulator", type="cpu")
result = executor.run(circuit, shots=1000, noise=noise_model)
print(f"Noisy bitstring counts: {result.bitstring_counts}")
# end-cell-noise
################################################################################

################################################################################
# start-cell-list-implementations
from qdk_chemistry.algorithms import registry

# List all registered circuit executor implementations
implementations = registry.available("circuit_executor")
print(implementations)  # e.g. ['qdk_sparse_state_simulator', 'qdk_full_state_simulator', 'qiskit_aer_simulator']
# end-cell-list-implementations
################################################################################
