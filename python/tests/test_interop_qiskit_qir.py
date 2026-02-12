"""Tests for QIR to Qiskit conversion."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import pyqir
import pytest

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit

    from qdk_chemistry.plugins.qiskit._interop.qir import qir_bitcode_to_qiskit


pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")


def test_qir_to_qiskit_conversion():
    """Test conversion of QIR to Qiskit."""
    module = pyqir.SimpleModule("bell_state", 2, 2)
    qis = pyqir.BasicQisBuilder(module.builder)
    qis.h(module.qubits[0])
    qis.cx(module.qubits[0], module.qubits[1])
    qis.mz(module.qubits[0], module.results[0])
    qis.mz(module.qubits[1], module.results[1])

    circuit = qir_bitcode_to_qiskit(module.bitcode())
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2
    assert circuit.num_clbits == 2
    assert circuit.count_ops() == {"h": 1, "cx": 1, "measure": 2}

    module2 = pyqir.SimpleModule("complex", 3, 3)
    qis2 = pyqir.BasicQisBuilder(module2.builder)
    qis2.h(module2.qubits[0])
    qis2.x(module2.qubits[1])
    qis2.y(module2.qubits[2])
    qis2.cx(module2.qubits[0], module2.qubits[1])
    qis2.cz(module2.qubits[1], module2.qubits[2])
    qis2.t(module2.qubits[0])
    qis2.s(module2.qubits[1])
    qis2.mz(module2.qubits[0], module2.results[0])
    qis2.mz(module2.qubits[1], module2.results[1])
    qis2.mz(module2.qubits[2], module2.results[2])

    circuit2 = qir_bitcode_to_qiskit(module2.bitcode())
    assert isinstance(circuit2, QuantumCircuit)
    assert circuit2.num_qubits == 3
    assert circuit2.num_clbits == 3
    assert circuit2.count_ops() == {"h": 1, "x": 1, "y": 1, "cx": 1, "cz": 1, "t": 1, "s": 1, "measure": 3}
