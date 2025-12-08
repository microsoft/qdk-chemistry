"""Utilities to analyze and plot qiskit quantum circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections import Counter
from dataclasses import dataclass, field
from math import inf

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import circuit_drawer

from qdk_chemistry.definitions import (
    BI_DIRECTIONAL_2Q_GATES,
    NON_CLIFFORD_GATES,
    SINGLE_QUBIT_CLIFFORD_GATES,
    SUPERPOSITION_1Q_GATES,
    TWO_QUBIT_CLIFFORD_GATES,
    UNI_DIRECTIONAL_2Q_CLIFFORD_GATES,
)
from qdk_chemistry.utils import Logger

__all__ = ["CircuitInfo", "analyze_qubit_status", "plot_circuit_diagram"]


def analyze_qubit_status(circuit: QuantumCircuit) -> dict[int, str]:
    """Analyze the status of qubits in a quantum circuit.

    Note: The gate classification logic depends on the settings defined in
    definitions.py. Please modify gate sets to ensure gate consistency.

    This function inspects the quantum circuit to determine the role of each qubit:

    * "quantum": has a quantum gate applied
    * "classical": touched by gates but bitstring outputs are deterministic
    * "idle": has no gates applied

    Args:
        circuit: The quantum circuit to analyze.

    Returns:
        A summary of qubit roles indexed by qubit index.

    """
    Logger.trace_entering()
    dag = circuit_to_dag(circuit)

    # Setup data structures to track qubit status and two-qubit gates for propagation.
    # Two-qubit gates track store as: (time, gate_name, control, target, edge_type),
    # edge_type: {"bidirectional", "unidirectional"}.
    n = circuit.num_qubits
    has_gate = [False] * n
    quantum_time = [inf] * n  # Earliest time the qubit became quantum; inf means not quantum
    two_q_edges: list[tuple[int, str, int, int, str]] = []

    # Walk through the circuit with a time index.
    for time, node in enumerate(dag.topological_op_nodes()):
        gate = node.op.name.lower()
        qargs = node.qargs
        indices = [circuit.find_bit(q).index for q in qargs]

        # Mark touched qubits
        for q in indices:
            if gate not in {"id", "barrier"}:
                has_gate[q] = True

        if len(indices) == 1:
            q = indices[0]
            # Seed quantum if the gate creates superposition
            if gate in SUPERPOSITION_1Q_GATES:
                quantum_time[q] = min(quantum_time[q], time)

        elif len(indices) == 2:
            control, target = indices

            if gate in BI_DIRECTIONAL_2Q_GATES:
                two_q_edges.append((time, gate, control, target, "bidirectional"))

            elif gate in UNI_DIRECTIONAL_2Q_CLIFFORD_GATES:
                two_q_edges.append((time, gate, control, target, "unidirectional"))

    # Process edges in time order so that only sources that are already quantum by that time can propagate.
    two_q_edges.sort(key=lambda x: x[0])
    for time, _gate, control, target, edge_type in two_q_edges:
        if edge_type == "bidirectional":
            # If either endpoint is quantum by time, the other becomes quantum at time.
            if quantum_time[control] <= time and quantum_time[target] > time:
                quantum_time[target] = time
            if quantum_time[target] <= time and quantum_time[control] > time:
                quantum_time[control] = time

        elif edge_type == "unidirectional":
            if quantum_time[control] <= time and quantum_time[target] > time:
                quantum_time[target] = time

    summary: dict[int, str] = {}
    for q in range(n):
        if quantum_time[q] < inf:
            role = "quantum"
        elif has_gate[q]:
            role = "classical"
        else:
            role = "idle"
        summary[q] = role

    return summary


@dataclass
class CircuitInfo:
    """Data class to store information of a quantum circuit.

    This class provides methods to analyze the circuit and summarize its properties.
    """

    circuit: QuantumCircuit
    """The quantum circuit to analyze."""

    num_qubits: int = field(init=False)
    """Number of qubits in the circuit."""

    depth: int = field(init=False)
    """Depth of the circuit."""

    num_gates: int = field(init=False)
    """Total number of gates in the circuit."""

    gate_counts: Counter = field(init=False)
    """Counts of each type of gate in the circuit."""

    def __post_init__(self):
        """Post-initialization to compute circuit properties."""
        Logger.trace_entering()
        self.num_qubits = self.circuit.num_qubits
        self.depth = self.circuit.depth()
        self.gate_counts = Counter(self.circuit.count_ops())
        self.num_gates = sum(self.gate_counts.values())

    def count_gate_category(self, gate_list: frozenset[str]) -> int:
        """Return the number of gates in the circuit that belong to a list."""
        Logger.trace_entering()
        return sum(self.gate_counts.get(g, 0) for g in gate_list)

    def count_gate(self, gate_name: str) -> int:
        """Return the number of times a specific gate appears."""
        Logger.trace_entering()
        return self.gate_counts.get(gate_name.lower(), 0)

    @property
    def num_single_qubit_clifford(self) -> int:
        """Return the number of single-qubit Clifford gates in the circuit."""
        return self.count_gate_category(SINGLE_QUBIT_CLIFFORD_GATES)

    @property
    def num_two_qubit_clifford(self) -> int:
        """Return the number of two-qubit Clifford gates in the circuit."""
        return self.count_gate_category(TWO_QUBIT_CLIFFORD_GATES)

    @property
    def num_non_clifford(self) -> int:
        """Return the number of non-Clifford gates in the circuit."""
        return self.count_gate_category(NON_CLIFFORD_GATES)

    def summary(self) -> dict:
        Logger.trace_entering()
        """Return a summary of the circuit information."""
        return {
            "num_qubits": self.num_qubits,
            "depth": self.depth,
            "total_gates": self.num_gates,
            "single_qubit_clifford": self.num_single_qubit_clifford,
            "two_qubit_clifford": self.num_two_qubit_clifford,
            "non_clifford": self.num_non_clifford,
        }

    def __str__(self) -> str:
        """Nicely formatted summary for printing."""
        Logger.trace_entering()
        s = self.summary()
        return (
            f"Circuit info summary:\n"
            f"  Qubits: {s['num_qubits']}\n"
            f"  Depth: {s['depth']}\n"
            f"  Total Gates: {s['total_gates']}\n"
            f"  Single-Qubit Clifford Gates: {s['single_qubit_clifford']}\n"
            f"  Two-Qubit Clifford Gates: {s['two_qubit_clifford']}\n"
            f"  Non-Clifford Gates: {s['non_clifford']}"
        )


def plot_circuit_diagram(
    circuit: QuantumCircuit,
    remove_idle_qubits: bool = True,
    remove_classical_qubits: bool = True,
    output_file: str | None = None,
    **draw_kwargs,
):
    """Plots a simplified circuit diagram, removing idle or classical qubits safely.

    Ensures measurement targets and classical registers remain consistent.
    """
    Logger.trace_entering()
    status = analyze_qubit_status(circuit)
    remove_status = []
    if remove_idle_qubits:
        remove_status.append("idle")
    if remove_classical_qubits:
        remove_status.append("classical")
        Logger.info(
            "Removing classical qubits will also remove any control operations sourced from them "
            "and measurements involving them."
        )

    kept_qubit_indices = [q for q, role in status.items() if role not in remove_status]
    if not kept_qubit_indices:
        raise ValueError("No qubits remain after filtering. Try relaxing filters.")

    # Check measurement operations
    kept_measurements: list[tuple[int, int]] = []
    for inst in circuit.data:
        if inst.operation.name == "measure":
            qidx = circuit.find_bit(inst.qubits[0]).index
            cidx = circuit.find_bit(inst.clbits[0]).index
            if qidx in kept_qubit_indices:
                kept_measurements.append((qidx, cidx))

    if remove_classical_qubits:
        kept_clbit_indices = sorted({cidx for _, cidx in kept_measurements})
    else:
        kept_clbit_indices = list(range(len(circuit.clbits)))

    if not kept_clbit_indices and len(circuit.clbits) > 0:
        Logger.warn("All measurements are dropped, no classical bits remain.")

    new_qc = QuantumCircuit(len(kept_qubit_indices), len(kept_clbit_indices))
    qubit_map = {circuit.qubits[i]: new_qc.qubits[new_i] for new_i, i in enumerate(kept_qubit_indices)}
    clbit_map = {circuit.clbits[i]: new_qc.clbits[new_i] for new_i, i in enumerate(kept_clbit_indices)}

    for inst in circuit.data:
        qargs = [qubit_map[q] for q in inst.qubits if q in qubit_map]
        cargs = [clbit_map[c] for c in inst.clbits if c in clbit_map]
        if len(qargs) != len(inst.qubits) or len(cargs) != len(inst.clbits):
            continue
        new_qc.append(inst.operation, qargs, cargs)

    if output_file:
        try:
            circuit_drawer(new_qc, output="mpl", filename=output_file, **draw_kwargs)
            return None
        except MemoryError:
            Logger.warn("MemoryError: Failed to save circuit diagram.")
            return None
    else:
        return new_qc.draw("mpl", **draw_kwargs)
