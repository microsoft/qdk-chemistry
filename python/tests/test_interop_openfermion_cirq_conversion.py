"""Test Cirq circuit conversion utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import importlib.util

import pytest

CIRQ_AVAILABLE = importlib.util.find_spec("cirq") is not None

if CIRQ_AVAILABLE:
    import cirq

    from qdk_chemistry.data import Circuit
    from qdk_chemistry.plugins.openfermion._interop.cirq_conversion import (
        _qasm2_to_qasm3,
        _qasm3_to_qasm2,
        cirq_circuit_to_qasm3,
        cirq_circuit_to_qdk_circuit,
        qdk_circuit_to_cirq_circuit,
    )

pytestmark = pytest.mark.skipif(not CIRQ_AVAILABLE, reason="Cirq not available")


# -------------------------------------------------------------------------------------
# QASM 2 ↔ QASM 3 syntax conversion
# -------------------------------------------------------------------------------------


def test_qasm2_to_qasm3_header():
    """Test that QASM 2 header is upgraded to QASM 3."""
    qasm2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];'
    qasm3 = _qasm2_to_qasm3(qasm2)
    assert "OPENQASM 3.0;" in qasm3
    assert 'include "stdgates.inc";' in qasm3
    assert "qubit[2] q;" in qasm3
    assert "bit[2] c;" in qasm3


def test_qasm2_to_qasm3_measure():
    """Test that QASM 2 measurement syntax is converted to QASM 3."""
    qasm2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];'
    qasm3 = _qasm2_to_qasm3(qasm2)
    assert "c[0] = measure q[0];" in qasm3


def test_qasm2_to_qasm3_barrier_removed():
    """Test that barrier statements are removed in QASM 3 output."""
    qasm2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nbarrier q[0],q[1];\nh q[0];'
    qasm3 = _qasm2_to_qasm3(qasm2)
    assert "barrier" not in qasm3
    assert "h q[0];" in qasm3


def test_qasm3_to_qasm2_roundtrip():
    """Test that QASM 2 → 3 → 2 round-trip preserves structure."""
    original = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
        "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];"
    )
    qasm3 = _qasm2_to_qasm3(original)
    recovered = _qasm3_to_qasm2(qasm3)

    # Verify key structural elements are preserved
    assert "OPENQASM 2.0;" in recovered
    assert 'include "qelib1.inc";' in recovered
    assert "qreg q[2];" in recovered
    assert "creg c[2];" in recovered
    assert "h q[0];" in recovered
    assert "cx q[0],q[1];" in recovered
    assert "measure q[0] -> c[0];" in recovered


def test_qasm2_to_qasm3_gate_preservation():
    """Test that gate statements are preserved during QASM upgrade."""
    qasm2 = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\nrz(0.5) q[0];'
    qasm3 = _qasm2_to_qasm3(qasm2)
    assert "h q[0];" in qasm3
    assert "cx q[0],q[1];" in qasm3
    assert "rz(0.5) q[0];" in qasm3


# -------------------------------------------------------------------------------------
# Cirq → QASM 3 / QDK Circuit
# -------------------------------------------------------------------------------------


def test_cirq_circuit_to_qasm3_basic():
    """Test converting a simple Cirq circuit to QASM 3."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

    qasm3_str = cirq_circuit_to_qasm3(circuit)
    assert "OPENQASM 3.0;" in qasm3_str
    assert "qubit" in qasm3_str


def test_cirq_circuit_to_qdk_circuit():
    """Test converting a Cirq circuit to a QDK Circuit object."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

    qdk_circuit = cirq_circuit_to_qdk_circuit(circuit, encoding="jordan-wigner")
    assert isinstance(qdk_circuit, Circuit)
    assert qdk_circuit.encoding == "jordan-wigner"
    assert "OPENQASM 3.0;" in qdk_circuit.get_qasm()


def test_cirq_circuit_to_qasm3_with_measurements():
    """Test that Cirq measurements are correctly converted to QASM 3 syntax."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        [
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="result"),
        ]
    )

    qasm3_str = cirq_circuit_to_qasm3(circuit)
    assert "OPENQASM 3.0;" in qasm3_str
    # Should have measurement in QASM 3 syntax (= measure)
    assert "= measure" in qasm3_str


# -------------------------------------------------------------------------------------
# QDK Circuit → Cirq round-trip
# -------------------------------------------------------------------------------------


def test_cirq_round_trip():
    """Test Cirq → QDK Circuit → Cirq preserves circuit structure."""
    q0, q1 = cirq.LineQubit.range(2)
    original = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])

    qdk_circuit = cirq_circuit_to_qdk_circuit(original)
    recovered = qdk_circuit_to_cirq_circuit(qdk_circuit)

    assert isinstance(recovered, cirq.Circuit)
    # Both should produce the same unitary
    original_unitary = cirq.unitary(original)
    recovered_unitary = cirq.unitary(recovered)
    assert original_unitary.shape == recovered_unitary.shape
