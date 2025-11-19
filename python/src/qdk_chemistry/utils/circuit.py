"""Utilities for visualizing circuits with QDK widgets."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import logging

from qiskit import QuantumCircuit, qasm3
from qsharp._native import Circuit
from qsharp.openqasm import circuit

from qdk_chemistry.plugins.qiskit._interop.circuit import analyze_qubit_status

_LOGGER = logging.getLogger(__name__)


def _trim_circuit(circuit_qasm: str, remove_idle_qubits: bool = True, remove_classical_qubits: bool = True) -> str:
    """Trim the quantum circuit by removing idle and classical qubits.

    Args:
        circuit_qasm: The quantum circuit in QASM format.
        remove_idle_qubits: If True, remove qubits that are idle (no gates applied).
        remove_classical_qubits: If True, remove qubits with gates but bitstring outputs are deterministic (0 or 1).

    Returns:
        A trimmed circuit in QASM format.

    """
    try:
        qc = qasm3.loads(circuit_qasm)
    except Exception as e:
        raise ValueError("Invalid QASM3 syntax provided.") from e

    status = analyze_qubit_status(qc)
    remove_status = []
    if remove_idle_qubits:
        remove_status.append("idle")
    if remove_classical_qubits:
        remove_status.append("classical")
        _LOGGER.info(
            "Removing classical qubits will also remove any control operations sourced from them "
            "and measurements involving them."
        )

    kept_qubit_indices = [q for q, role in status.items() if role not in remove_status]
    if not kept_qubit_indices:
        raise ValueError("No qubits remain after filtering. Try relaxing filters.")

    # Check measurement operations
    kept_measurements: list[tuple[int, int]] = []
    for inst in qc.data:
        if inst.operation.name == "measure":
            qidx = qc.find_bit(inst.qubits[0]).index
            cidx = qc.find_bit(inst.clbits[0]).index
            if qidx in kept_qubit_indices:
                kept_measurements.append((qidx, cidx))

    if remove_classical_qubits:
        kept_clbit_indices = sorted({cidx for _, cidx in kept_measurements})
    else:
        kept_clbit_indices = list(range(len(qc.clbits)))

    if not kept_clbit_indices and len(qc.clbits) > 0:
        _LOGGER.warning("All measurements are dropped, no classical bits remain.")

    new_qc = QuantumCircuit(len(kept_qubit_indices), len(kept_clbit_indices))
    qubit_map = {qc.qubits[i]: new_qc.qubits[new_i] for new_i, i in enumerate(kept_qubit_indices)}
    clbit_map = {qc.clbits[i]: new_qc.clbits[new_i] for new_i, i in enumerate(kept_clbit_indices)}

    for inst in qc.data:
        qargs = [qubit_map[q] for q in inst.qubits if q in qubit_map]
        cargs = [clbit_map[c] for c in inst.clbits if c in clbit_map]
        if len(qargs) != len(inst.qubits) or len(cargs) != len(inst.clbits):
            continue
        new_qc.append(inst.operation, qargs, cargs)

    return qasm3.dumps(new_qc)


def qasm_to_qdk_circuit(
    circuit_qasm: str, remove_idle_qubits: bool = True, remove_classical_qubits: bool = True
) -> Circuit:
    """Parse a QASM circuit into a QDK Circuit object with trimming options.

    Args:
        circuit_qasm: The quantum circuit to visualize.
        remove_idle_qubits: If True, remove qubits that are idle (no gates applied).
        remove_classical_qubits: If True, remove qubits with gates but bitstring outputs are deterministic (0 or 1).

    Returns:
        A QDK Circuit object representing the trimmed circuit.

    """
    circuit_to_visualize = _trim_circuit(circuit_qasm, remove_idle_qubits, remove_classical_qubits)

    return circuit(circuit_to_visualize)
