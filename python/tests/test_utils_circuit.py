"""Test for circuit utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
import re

import pytest

from qdk_chemistry.utils.circuit import _trim_circuit, qasm_to_qdk_circuit


def strip_ws(s: str) -> str:
    """Normalize whitespace to make string matching more robust."""
    return re.sub(r"\s+", " ", s).strip()


def test_trim_removes_idle_qubits_string():
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    bit[1] c;
    h q[0];
    cx q[0], q[2];
    c[0] = measure q[0];
    """
    trimmed = _trim_circuit(qasm, remove_idle_qubits=True, remove_classical_qubits=False)

    # idle qubit q[1] removed → new indices → q[0], q[1]
    norm = strip_ws(trimmed)

    assert "h q[0];" in norm
    assert "cx q[0], q[1];" in norm
    assert "c[0] = measure q[0];" in norm

    # ensure q[1] never appears alone (idle qubit eliminated)
    assert "q[2]" not in norm


def test_trim_removes_classical_qubits_string():
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    bit[3] c;
    h q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """
    trimmed = _trim_circuit(qasm, remove_idle_qubits=True, remove_classical_qubits=True)
    norm = strip_ws(trimmed)

    # classical qubit q[0] removed
    assert "h q[0];" in norm  # reindexed: old q[1] → new q[0]
    assert "c[0] = measure q[0];" in norm

    # everything referencing q[2] or old q[0] is gone
    assert "q[2]" not in norm
    assert "measure q[1]" not in norm


def test_measurement_dropped_if_qubit_filtered():
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[1];
    """
    trimmed = _trim_circuit(qasm, True, True)
    norm = strip_ws(trimmed)

    assert "h q[0];" in norm
    assert "measure" not in norm  # measurement removed


def test_control_gate_removed_if_control_is_classical():
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    bit[1] c;
    c[0] = measure q[0];
    h q[1];
    cx q[0], q[2];
    """
    trimmed = _trim_circuit(qasm, False, True)
    norm = strip_ws(trimmed)

    assert "h q[0];" in norm
    assert "cx" not in norm  # control removed
    assert "measure" not in norm  # measure removed because q0 removed entirely


def test_trim_reindexing_correct_string():
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[4] q;
    bit[4] c;
    h q[1];
    cx q[1], q[2];
    c[3] = measure q[3];
    """
    trimmed = _trim_circuit(qasm, True, True)
    norm = strip_ws(trimmed)

    # only q1 and q2 survive -> new q[0], q[1]
    assert "h q[0];" in norm
    assert "cx q[0], q[1];" in norm
    assert "measure" not in norm


def test_raises_if_all_removed():
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    c[0] = measure q[0];
    """
    with pytest.raises(ValueError, match="No qubits remain after filtering"):
        _trim_circuit(qasm, True, True)


def test_no_removal_returns_same_string():
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    bit[1] c;
    qubit[2] q;
    h q[0];
    c[0] = measure q[0];
    x q[1];
    """
    trimmed = _trim_circuit(qasm, False, False)

    # Do whitespace normalization and compare
    assert strip_ws(trimmed) == strip_ws(qasm)


def test_qasm_to_qdk_circuit():
    qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    x q[2];
    """
    circuit = qasm_to_qdk_circuit(qasm, remove_idle_qubits=True, remove_classical_qubits=True)
    circuit_info = json.loads(circuit.json())

    # The resulting circuit should have 2 qubits (q0 and q1)
    assert len(circuit_info["qubits"]) == 2
