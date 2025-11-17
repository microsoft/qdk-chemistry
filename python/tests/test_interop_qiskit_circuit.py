"""Test for circuit utilities in QDK/Chemistry qiskit interop."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from unittest.mock import patch

import pytest
from qiskit import QuantumCircuit

from qdk_chemistry.plugins.qiskit._interop.circuit import (
    CircuitInfo,
    analyze_qubit_status,
    plot_circuit_diagram,
)


@pytest.fixture
def simple_circuit():
    """Fixture to create a simple quantum circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def test_analyze_qubit_status(simple_circuit):
    """Test that analyze_qubit_status correctly classifies qubit types.

    Identify idle and classical qubits in the circuit.
    """
    # Analyze the qubit status of a simple circuit
    status = analyze_qubit_status(simple_circuit)

    # Check that all qubits are marked as quantum
    assert all(qubit_status == "quantum" for qubit_status in status.values())
    # There should be no classical or idle qubits
    assert not any(qubit_status == "classical" for qubit_status in status.values())
    assert not any(qubit_status == "idle" for qubit_status in status.values())


def test_circuit_info(simple_circuit):
    """Test that CircuitInfo extracts and summarizes circuit metrics correctly."""
    # Create a CircuitInfo instance
    info = CircuitInfo(simple_circuit)
    summary = info.summary()
    all_gates = [instr.operation.name for instr in simple_circuit.data]

    # Check that the info contains expected attributes
    assert summary["num_qubits"] == simple_circuit.num_qubits
    assert summary["depth"] == simple_circuit.depth()
    assert summary["total_gates"] == len(all_gates)
    assert summary["single_qubit_clifford"] == 1
    assert summary["two_qubit_clifford"] == 1
    assert summary["non_clifford"] == 0


def test_circuit_info_str_method(simple_circuit):
    """Test CircuitInfo __str__ method for string representation."""
    info = CircuitInfo(simple_circuit)
    str_repr = str(info)

    # Check that the string contains expected content
    assert "Circuit info summary:" in str_repr
    assert "Qubits:" in str_repr
    assert "Depth:" in str_repr
    assert "Total Gates:" in str_repr


def assert_qubit_status(result, expected):
    """Helper to assert qubit status dictionaries are equal."""
    assert set(result.keys()) == set(expected.keys())
    for q, role in expected.items():
        assert result[q] == role, f"Qubit {q}: expected {role}, got {result[q]}"


def test_analyze_qubit_status_with_idle_qubits():
    """Test analyze_qubit_status with a circuit containing idle qubits."""
    # Create a circuit with idle qubits
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    # Qubits 2 and 3 remain idle

    status = analyze_qubit_status(circuit)
    assert_qubit_status(status, {0: "quantum", 1: "quantum", 2: "idle", 3: "idle"})


def test_analyze_qubit_status_with_classical_operations():
    """Test analyze_qubit_status with classical-only operations."""
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    status = analyze_qubit_status(circuit)
    # All should be classified as classical since only X, CX, and measurement gates
    assert_qubit_status(status, {0: "classical", 1: "classical", 2: "classical"})

    circuit_1 = QuantumCircuit(3)
    circuit_1.x(0)
    circuit_1.cx(0, 1)
    circuit_1.barrier()  # barrier should not affect classification
    status_1 = analyze_qubit_status(circuit_1)
    assert_qubit_status(status_1, {0: "classical", 1: "classical", 2: "idle"})


def test_analyze_qubit_status_control_from_classical_qubits():
    """Test analyze_qubit_status with control qubit has classical gate."""
    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.cx(0, 1)  # control qubit with CX gate but it does not control any quantum gate
    circuit.h(1)
    circuit.x(2)  # classical qubit
    status = analyze_qubit_status(circuit)
    assert_qubit_status(status, {0: "classical", 1: "quantum", 2: "classical"})

    circuit_1 = QuantumCircuit(3)
    circuit_1.h(0)
    circuit_1.cx(0, 1)
    circuit_1.x(2)
    status_1 = analyze_qubit_status(circuit_1)
    assert_qubit_status(status_1, {0: "quantum", 1: "quantum", 2: "classical"})


def test_analyze_qubit_status_with_swap():
    """Test analyze_qubit_status with a circuit containing a swap gate."""
    qc = QuantumCircuit(4)
    qc.h(2)
    qc.cx(2, 3)
    qc.swap(0, 2)
    status = analyze_qubit_status(qc)
    assert_qubit_status(status, {0: "quantum", 1: "idle", 2: "quantum", 3: "quantum"})

    qc2 = QuantumCircuit(4)
    qc2.x(2)
    qc2.cx(2, 3)
    qc2.swap(0, 2)
    status = analyze_qubit_status(qc2)
    assert_qubit_status(status, {0: "classical", 1: "idle", 2: "classical", 3: "classical"})

    qc3 = QuantumCircuit(4)
    qc3.h(0)
    qc3.x(2)
    qc3.cx(2, 3)  # qubit 3 controlled by classical qubit 2
    qc3.swap(0, 2)  # qubit 2 becomes quantum via swap
    status = analyze_qubit_status(qc3)
    assert_qubit_status(status, {0: "quantum", 1: "idle", 2: "quantum", 3: "classical"})

    qc4 = QuantumCircuit(4)
    qc4.h(0)
    qc4.x(2)
    qc4.swap(0, 2)
    qc4.cx(2, 3)  # qubit 3 controlled by quantum qubit 2
    status = analyze_qubit_status(qc4)
    assert_qubit_status(status, {0: "quantum", 1: "idle", 2: "quantum", 3: "quantum"})


def test_analyze_qubit_status_complex_entanglement():
    """Test analyze_qubit_status with complex entanglement patterns."""
    # Create a circuit with complex entanglement that exercises more code paths
    circuit = QuantumCircuit(6)
    circuit.h(0)
    circuit.cx(0, 2)
    circuit.x(1)  # classical qubit
    circuit.cx(1, 3)  # control qubit 3 from classical qubit
    circuit.h(1)  # make qubit 1 after CX
    circuit.cx(2, 4)
    circuit.x(5)
    status = analyze_qubit_status(circuit)

    assert_qubit_status(
        status, {0: "quantum", 1: "quantum", 2: "quantum", 3: "classical", 4: "quantum", 5: "classical"}
    )


def test_circuit_info_count_gate_methods(simple_circuit):
    """Test CircuitInfo gate counting methods."""
    info = CircuitInfo(simple_circuit)

    # Test count_gate method
    h_count = info.count_gate("h")
    assert h_count == 1

    # Test count_gate with non-existent gate
    non_existent_count = info.count_gate("non_existent_gate")
    assert non_existent_count == 0

    # Test count_gate_category with empty list
    empty_category_count = info.count_gate_category([])
    assert empty_category_count == 0


def test_plot_circuit_diagram(tmp_path, simple_circuit):
    """Test plotting a simple quantum circuit diagram and save the output."""
    # Plot with default settings
    plot_circuit_diagram(simple_circuit, output_file=tmp_path / "circuit.png")
    assert (tmp_path / "circuit.png").exists()


def test_plot_circuit_diagram_remove_idle_classical_with_measurements():
    """Test plotting a circuit diagram with idle and classical qubits removed."""
    # Create a circuit with idle and classical qubits
    qc = QuantumCircuit(5, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.x(3)  # classical qubit
    qc.h(4)
    qc.measure([0, 2, 3], [0, 1, 2])

    fig = plot_circuit_diagram(qc, remove_idle_qubits=True, remove_classical_qubits=True)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert fig is not None
    assert "${q}_{4}$" not in texts  # classical qubit removed, 4 qubits remain
    assert "X" not in texts  # X gate on classical qubit removed
    assert "3" not in texts  # 2 measurements remain, not 3


def test_plot_circuit_diagram_clbits_index():
    """Test plotting a circuit diagram with classical bits and their indices."""
    # Create a circuit with classical bits
    qc = QuantumCircuit(3, 1)
    qc.h(1)
    qc.h(2)
    qc.measure(1, 0)

    fig = plot_circuit_diagram(qc)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert fig is not None
    assert "${q}_{2}$" not in texts  # 1 idle qubit removed
    assert "1" in texts  # classical bit 1 present
    assert "0" in texts  # measurement index 0 present


def test_clbit_register_handling():
    """Test plotting a circuit diagram with classical registers."""
    # Create a circuit with classical registers
    qc = QuantumCircuit(3, 3)
    qc.h(1)
    qc.h(2)

    fig = plot_circuit_diagram(qc, remove_classical_qubits=False)
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert fig is not None
    assert "${q}_{2}$" not in texts  # 1 idle qubit removed
    assert "c" in texts  # classical register present
    assert "3" in texts  # 3 clibits registered


def test_plot_circuit_diagram_logging_and_warning(caplog):
    # Circuit with one qubit and one classical bit
    qc = QuantumCircuit(2, 1)
    qc.h(1)
    qc.measure(0, 0)

    # Capture logs
    with caplog.at_level("WARNING"):
        plot_circuit_diagram(qc, remove_classical_qubits=True)

    # Verify the warning was logged
    assert any("All measurements are dropped" in message for message in caplog.messages), (
        f"Expected warning not found in logs: {caplog.messages}"
    )

    with caplog.at_level("INFO"):
        plot_circuit_diagram(qc, remove_classical_qubits=True)
    assert any(
        "Removing classical qubits will also remove any control operations sourced from them" in message
        for message in caplog.messages
    ), f"Expected info not found in logs: {caplog.messages}"


def test_circuit_no_qubits():
    """Test no qubits remain error."""
    circuit = QuantumCircuit(1)  # No gates = idle qubit
    with pytest.raises(ValueError, match="No qubits remain after filtering"):
        plot_circuit_diagram(circuit, remove_idle_qubits=True)


def test_circuit_memory_error():
    """Test MemoryError handling."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    with (
        patch("qdk_chemistry.plugins.qiskit._interop.circuit.circuit_drawer", side_effect=MemoryError()),
        patch("qdk_chemistry.plugins.qiskit._interop.circuit._LOGGER") as mock_logger,
    ):
        plot_circuit_diagram(circuit, output_file="test.png")
        mock_logger.warning.assert_called_once()
