"""Circuit data class usage examples."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

################################################################################
# start-cell-qsharp-workflow
import json

import qdk
import qsharp
from qdk_chemistry.data import Circuit
from qdk_chemistry.data.circuit import QsharpFactoryData

# 1. Define a Q# operation from string
qsharp.eval(
    """
    operation GHZSample(n: Int) : Result[] {
    use qs = Qubit[n];

    H(qs[0]);
    ApplyToEach(CNOT(qs[0], _), qs[1...]);

    let results = MeasureEachZ(qs);
    ResetAll(qs);
    return results;
    }
    """
)

# 2. Get a Q# circuit object and wrap it
qsharp_factory = QsharpFactoryData(
    program=qdk.code.GHZSample,
    parameter={"n": 3},
)
circuit = Circuit(qsharp_factory=qsharp_factory)

# 3. Inspect the circuit
qsharp_circuit = circuit.get_qsharp_circuit()
print(qsharp_circuit)
# Return an ASCII diagram of the circuit
#    q_0    ── H ──── ● ──── ● ──── M ──── |0〉 ──
#                     │      │      ╘════════════
#    q_1    ───────── X ─────┼───── M ──── |0〉 ──
#                            │      ╘════════════
#    q_2    ──────────────── X ──── M ──── |0〉 ──
#                                   ╘════════════

# 4. Access the gate-level JSON structure
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
# start-cell-qsharp-harness
# Prepare single-reference state with Q# factory parameters
from qdk_chemistry.data import Circuit
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

# 1. Build a parameter object (a dictionary or a Q# structured parameter class)
bitstring = [1, 1, 0, 0]
params = QSHARP_UTILS.StatePreparation.SingleReferenceParams(bitStrings=bitstring, numQubits=len(bitstring))

# 2. Create the factory — vars() converts the dataclass to a dict
factory = QsharpFactoryData(
    program=QSHARP_UTILS.StatePreparation.MakeSingleReferenceStateCircuit,
    parameter=vars(params),
)

# 3. Wrap in a Circuit — nothing is compiled yet
circuit = Circuit(qsharp_factory=factory, encoding="jordan-wigner")

# 4. Compilation happens on demand:
circuit.get_qsharp_circuit()  # → qsharp.circuit(program, **params)
circuit.get_qir()  # → qsharp.compile(program, **params)
# end-cell-qsharp-harness
################################################################################

################################################################################
# start-cell-conversion
import numpy as np
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Circuit, Structure

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
pruned_qsharp_circuit = circuit.get_qsharp_circuit(prune_classical_qubits=True)
print(pruned_qsharp_circuit)

# Get the QIR representation
qir = circuit.get_qir()
print(qir)

# Export to OpenQASM or Qiskit when needed (if qiskit is installed)
qasm_str = circuit.get_qasm()
print(qasm_str)

qiskit_circuit = circuit.get_qiskit_circuit()
print(qiskit_circuit)

# end-cell-conversion
################################################################################

################################################################################
# start-cell-serialization
# Save to JSON
circuit.to_json_file("example_circuit.circuit.json")

# Load from JSON
loaded = Circuit.from_json_file("example_circuit.circuit.json")

# Save to HDF5
circuit.to_hdf5_file("example_circuit.circuit.h5")

# Load from HDF5
loaded_h5 = Circuit.from_hdf5_file("example_circuit.circuit.h5")
# end-cell-serialization
################################################################################
