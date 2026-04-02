"""Circuit data class usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-create-qasm
from qdk_chemistry.data import Circuit

# Create a circuit from an OpenQASM string
circuit = Circuit(
    qasm="""
    include "stdgates.inc";
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
    """
)
# end-cell-create-qasm
################################################################################

################################################################################
# start-cell-get-qasm
# Retrieve the OpenQASM representation
qasm_str = circuit.get_qasm()
print(qasm_str)
# end-cell-get-qasm
################################################################################

################################################################################
# start-cell-get-qsharp
# Retrieve the Q# circuit representation
qsharp_circuit = circuit.get_qsharp_circuit()
print(qsharp_circuit)

# Optionally prune unused classical qubits
qsharp_circuit_pruned = circuit.get_qsharp_circuit(prune_classical_qubits=True)
# end-cell-get-qsharp
################################################################################

################################################################################
# start-cell-get-qir
# Compile to QIR (Quantum Intermediate Representation)
qir = circuit.get_qir()
# end-cell-get-qir
################################################################################

################################################################################
# start-cell-get-qiskit
# Convert to a Qiskit QuantumCircuit (requires qiskit installed)
qiskit_circuit = circuit.get_qiskit_circuit()
print(qiskit_circuit)
# end-cell-get-qiskit
################################################################################

################################################################################
# start-cell-qsharp-workflow
import qsharp
from qdk_chemistry.data import Circuit

# 1. Define a Q# operation from string
qsharp.init()
qsharp.eval(
    """
    operation BellPair() : (Result, Result) {
        use (q0, q1) = (Qubit(), Qubit());
        H(q0);
        CNOT(q0, q1);
        let (r0, r1) = (M(q0), M(q1));
        Reset(q0);
        Reset(q1);
        return (r0, r1);
    }
    """
)

# 2. Get a Q# circuit object and wrap it — keep QASM too for estimate()
qsharp_circuit = qsharp.circuit("BellPair()")
bell_qasm = """
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""
circuit = Circuit(qsharp=qsharp_circuit, qasm=bell_qasm)

# 3. Inspect the circuit — prints an ASCII diagram
#    q_0  ── H ──── ● ──── M ──── |0〉 ──
#                   │      ╘════════════
#    q_1  ───────── X ──── M ──── |0〉 ──
#                          ╘════════════
print(circuit.get_qsharp_circuit())

# 4. Access the gate-level JSON structure
import json

circuit_json = json.loads(qsharp_circuit.json())
print(f"Qubits: {len(circuit_json['qubits'])}")

# 5. Resource estimation
estimate_result = circuit.estimate()
formatted = estimate_result["physicalCountsFormatted"]
print(f"Physical qubits: {formatted['physicalQubits']}")
print(f"Runtime: {formatted['runtime']}")
# end-cell-qsharp-workflow
################################################################################

################################################################################
# start-cell-serialization
import os
import tempfile

tmpdir = tempfile.mkdtemp()

# Save to JSON
circuit.to_json_file(os.path.join(tmpdir, "my.circuit.json"))

# Load from JSON
loaded = Circuit.from_json_file(os.path.join(tmpdir, "my.circuit.json"))

# Save to HDF5
circuit.to_hdf5_file(os.path.join(tmpdir, "my.circuit.h5"))

# Load from HDF5
loaded_h5 = Circuit.from_hdf5_file(os.path.join(tmpdir, "my.circuit.h5"))
# end-cell-serialization
################################################################################
