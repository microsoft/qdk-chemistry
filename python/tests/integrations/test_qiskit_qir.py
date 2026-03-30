"""Tests for QIR to Qiskit conversion."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pytest
from qsharp.openqasm import compile as compile_qasm_to_qir

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit

    from qdk_chemistry.plugins.qiskit._interop.qir import qir_ir_to_qiskit


pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")


def test_qir_to_qiskit_conversion():
    """Test conversion of QIR to Qiskit."""
    qasm_str = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    cx q[0], q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    """

    qir = compile_qasm_to_qir(qasm_str)
    circuit = qir_ir_to_qiskit(str(qir))
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2
    assert circuit.num_clbits == 2
    assert circuit.count_ops() == {"h": 1, "cx": 1, "measure": 2}

    qasm_str_2 = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    bit[3] c;
    h q[0];
    x q[1];
    y q[2];
    cx q[0], q[1];
    cz q[1], q[2];
    t q[0];
    s q[1];
    c[0] = measure q[0];
    c[1] = measure q[1];
    c[2] = measure q[2];
    """
    qir_2 = compile_qasm_to_qir(qasm_str_2)
    circuit2 = qir_ir_to_qiskit(str(qir_2))
    assert isinstance(circuit2, QuantumCircuit)
    assert circuit2.num_qubits == 3
    assert circuit2.num_clbits == 3
    assert circuit2.count_ops() == {"h": 1, "x": 1, "y": 1, "cx": 1, "cz": 1, "t": 1, "s": 1, "measure": 3}
