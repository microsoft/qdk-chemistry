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
# start-cell-configure-azure-quantum
# Configure the Azure Quantum backend. The connection settings must be provided.
import json

executor = create("circuit_executor", "azure_quantum_backend")
executor.settings().set("subscription_id", "my-subscription-id")
executor.settings().set("resource_group", "my-resource-group")
executor.settings().set("workspace_name", "my-workspace")
executor.settings().set("location", "my-location")
executor.settings().set("target_name", "my.backend.target")
executor.settings().set("auth_mode", "azure-cli")

# Target-specific job parameters, passed through as-is
executor.settings().set("input_params", json.dumps({"seed": 42}))

# Keep job artifacts locally
executor.settings().set("output_dir", "./job_artifacts")
executor.settings().set("attachments", ["output"])
# end-cell-configure-azure-quantum
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
        "x": {"depolarizing_error": 0.005},
        "cx": {"depolarizing_error": 0.007},
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
print(implementations)
# e.g. ['qdk_full_state_simulator', 'qdk_sparse_state_simulator',
#       'qiskit_aer_simulator', 'azure_quantum_backend']
# end-cell-list-implementations
################################################################################
