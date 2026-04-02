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

# 2. Get a Q# circuit object and wrap it
qsharp_circuit = qsharp.circuit("BellPair()")
circuit = Circuit(qsharp=qsharp_circuit)

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
# start-cell-qsharp-native
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Structure

# When algorithms produce circuits, they carry native Q# operations internally.
# This enables end-to-end Q# composition without format conversions.
coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
structure = Structure(coords, symbols=["H", "H"])

scf = create("scf_solver")
_, wfn = scf.run(structure, charge=0, spin_multiplicity=1, basis_or_guess="sto-3g")
ham = create("hamiltonian_constructor").run(wfn.get_orbitals())
_, wfn_cas = create("multi_configuration_calculator").run(ham, 1, 1)

# StatePreparation produces a Circuit with a native Q# factory
state_prep = create("state_prep", "sparse_isometry_gf2x")
circuit = state_prep.run(wfn_cas)

# Inspect the Q# circuit (prune unused qubits for clarity)
print(circuit.get_qsharp_circuit(prune_classical_qubits=True))

# Resource estimation — deferred QIR compilation happens here automatically
print(circuit.estimate())

# Export to OpenQASM or Qiskit when needed
qasm_str = circuit.get_qasm()
# qiskit_circuit = circuit.get_qiskit_circuit()  # requires qiskit installed
# end-cell-qsharp-native
################################################################################

################################################################################
# start-cell-serialization
# Save to JSON
circuit.to_json_file("circuit.json")

# Load from JSON
loaded = Circuit.from_json_file("circuit.json")

# Save to HDF5
circuit.to_hdf5_file("circuit.h5")

# Load from HDF5
loaded_h5 = Circuit.from_hdf5_file("circuit.h5")
# end-cell-serialization
################################################################################
