"""Tests for QIR to Qiskit conversion."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk import TargetProfile
from qdk.openqasm import compile as compile_qasm_to_qir

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit

    from qdk_chemistry.plugins.qiskit._interop.qir import UnsupportedQIROperationError, qir_ir_to_qiskit


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

    qir = compile_qasm_to_qir(qasm_str, target_profile=TargetProfile.Base)
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
    qir_2 = compile_qasm_to_qir(qasm_str_2, target_profile=TargetProfile.Base)
    circuit2 = qir_ir_to_qiskit(str(qir_2))
    assert isinstance(circuit2, QuantumCircuit)
    assert circuit2.num_qubits == 3
    assert circuit2.num_clbits == 3
    assert circuit2.count_ops() == {"h": 1, "x": 1, "y": 1, "cx": 1, "cz": 1, "t": 1, "s": 1, "measure": 3}


@pytest.fixture(scope="module")
def adaptive_context():
    """Branching on a measurement needs an adaptive profile; the shared context defaults to Base."""
    from qdk_chemistry.utils.qsharp import create_qsharp_context  # noqa: PLC0415

    return create_qsharp_context(target_profile=TargetProfile.Adaptive_RIF)


def _qsharp_qir(context, source: str, entry: str) -> str:
    """Compile a Q# snippet to QIR through *context*."""
    context.eval(source)
    namespace, operation = entry.rsplit(".", 1)
    program = getattr(getattr(context.code, namespace), operation)
    return str(context.compile(program))


def test_qir_measurement_conditioned_gate_becomes_if_block(adaptive_context):
    """A gate guarded by a measurement result must convert to a Qiskit if-block, not a bare gate."""
    qir = _qsharp_qir(
        adaptive_context,
        """
        namespace QirCondTest {
            operation Guarded() : Unit {
                use qs = Qubit[2];
                H(qs[0]);
                if MResetZ(qs[0]) == One {
                    X(qs[1]);
                }
                ResetAll(qs);
            }
        }
        """,
        "QirCondTest.Guarded",
    )
    circuit = qir_ir_to_qiskit(qir)
    ops = circuit.count_ops()
    assert ops.get("if_else") == 1, f"conditional gate was flattened into the main body: {ops}"
    assert "x" not in ops, f"conditioned X must live inside the if-block, not the main body: {ops}"

    body = next(instr.operation.blocks[0] for instr in circuit.data if instr.operation.name == "if_else")
    assert body.count_ops() == {"x": 1}


def test_qir_conditioned_gate_only_fires_on_matching_outcome(adaptive_context):
    """Simulate the branch to confirm the condition is honored rather than applied unconditionally."""
    aer = pytest.importorskip("qiskit_aer")

    # q[0] stays in |0>, so the measurement is deterministically Zero and the body must not run.
    qir = _qsharp_qir(
        adaptive_context,
        """
        namespace QirCondSimTest {
            operation NeverTaken() : Unit {
                use qs = Qubit[2];
                if MResetZ(qs[0]) == One {
                    X(qs[1]);
                }
            }
        }
        """,
        "QirCondSimTest.NeverTaken",
    )
    circuit = qir_ir_to_qiskit(qir).copy()
    circuit.save_statevector()

    result = aer.AerSimulator(method="statevector").run(circuit, shots=1, seed_simulator=1).result()
    probabilities = np.asarray(result.get_statevector()).conj()
    probabilities = np.abs(probabilities) ** 2
    # |00> must retain all the probability; a flattened X would move it to |10>.
    assert np.isclose(
        probabilities[0], 1.0, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
    ), f"unexpected statevector: {probabilities}"


def test_qir_unsupported_branch_condition_raises():
    """Branch conditions the converter cannot resolve must fail loudly instead of being dropped."""
    qasm_str = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[1] c;
    h q[0];
    c[0] = measure q[0];
    if (c[0] == 1) {
        x q[1];
    }
    """
    qir = str(compile_qasm_to_qir(qasm_str, target_profile=TargetProfile.Adaptive_RIF))
    with pytest.raises(UnsupportedQIROperationError, match="read a measurement result"):
        qir_ir_to_qiskit(qir)
