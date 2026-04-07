"""Test for transpiler utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import IGate, SdgGate, SGate, XGate, YGate, ZGate
    from qiskit.transpiler import PassManager

    from qdk_chemistry.plugins.qiskit._interop.transpiler import (
        FactorCliffordFromRz,
        FactorPauliFromRotation,
        MergeZBasisRotations,
        RemoveZBasisOnZeroState,
        SubstituteCliffordRz,
    )
else:
    # Define placeholders for type checking when Qiskit is not available
    QuantumCircuit = object
    Parameter = object
    IGate = object
    SdgGate = object
    SGate = object
    XGate = object
    YGate = object
    ZGate = object
    PassManager = object
    MergeZBasisRotations = object
    RemoveZBasisOnZeroState = object
    SubstituteCliffordRz = object
    FactorCliffordFromRz = object
    FactorPauliFromRotation = object

pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")


def _run_pass(pass_class, circuit):
    """Helper to apply TransformationPass to a QuantumCircuit."""
    pm = PassManager([pass_class])
    return pm.run(circuit)


def test_merge_z_basis_rotations_simple():
    """Test MergeZBasisRotations merges consecutive Z-basis gates correctly."""
    qc = QuantumCircuit(1, 1)
    qc.s(0)
    qc.id(0)
    qc.rz(np.pi / 4, 0)
    qc.sdg(0)
    qc.z(0)
    qc.h(0)
    qc.rz(np.pi / 4, 0)

    result = _run_pass(MergeZBasisRotations(), qc)

    assert "rz" in result.count_ops()
    assert "s" not in result.count_ops()
    assert "sdg" not in result.count_ops()
    assert "z" not in result.count_ops()
    assert "id" not in result.count_ops()


@pytest.mark.parametrize(
    ("angle", "expected_gate"),
    [
        (0, IGate),
        (np.pi / 2, SGate),
        (np.pi, ZGate),
        (-np.pi / 2, SdgGate),
    ],
)
def test_substitute_clifford_rz(angle, expected_gate):
    """Test SubstituteCliffordRz substitutes Rz(θ) with correct Clifford gate."""
    qc = QuantumCircuit(1)
    qc.rz(angle, 0)

    result = _run_pass(SubstituteCliffordRz(), qc)

    ops = [instr.operation for instr in result.data]
    assert isinstance(ops[0], expected_gate)


def test_substitute_clifford_rz_parameterized():
    """Test SubstituteCliffordRz leaves parameterized Rz untouched."""
    qc = QuantumCircuit(1)
    qc.rz(5.5, 0)

    result = _run_pass(SubstituteCliffordRz(), qc)

    assert "rz" in result.count_ops()
    assert result.data[0].operation.params[0] == 5.5

    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.rz(theta, 0)
    result = _run_pass(SubstituteCliffordRz(), qc)
    assert "rz" in result.count_ops()


def test_substitute_clifford_rz_initialization():
    """Test SubstituteCliffordRz initialization."""
    with pytest.raises(TypeError):
        SubstituteCliffordRz(equivalent_gate_set="z")

    # Add "id" to equivalent gate set
    qc = QuantumCircuit(1)
    qc.rz(5.5, 0)
    pass_scr = SubstituteCliffordRz(equivalent_gate_set=["z", "s", "sdg"])
    _ = _run_pass(pass_scr, qc)
    assert "id" in pass_scr.settings().get("equivalent_gate_set")


def test_remove_z_basis_on_zero_state_removes_redundant_gates():
    """Test RemoveZBasisOnZeroState removes Z-basis gates on |0⟩."""
    qc = QuantumCircuit(1)
    qc.rz(np.pi / 3, 0)
    qc.s(0)
    qc.z(0)

    result = _run_pass(RemoveZBasisOnZeroState(), qc)

    # All gates removed because qubit remains in |0⟩
    assert result.size() == 0


def test_remove_z_basis_on_zero_state_preserves_after_x():
    """Test RemoveZBasisOnZeroState keeps gates after qubit leaves |0⟩."""
    qc = QuantumCircuit(1)
    qc.rz(np.pi / 3, 0)  # Should be removed
    qc.x(0)  # Qubit now in |1⟩
    qc.s(0)  # Should remain

    result = _run_pass(RemoveZBasisOnZeroState(), qc)

    assert "s" in result.count_ops()
    assert "rz" not in result.count_ops()


# ---------------------------------------------------------------------------
# FactorCliffordFromRz tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("angle", "expected_clifford"),
    [
        (0, IGate),
        (np.pi / 2, SGate),
        (np.pi, ZGate),
        (-np.pi / 2, SdgGate),
    ],
)
def test_factor_clifford_exact_angle(angle, expected_clifford):
    """Exact Clifford angles should produce a single Clifford gate with no residual Rz."""
    qc = QuantumCircuit(1)
    qc.rz(angle, 0)

    result = _run_pass(FactorCliffordFromRz(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert isinstance(ops[0], expected_clifford)


def test_factor_clifford_near_s():
    """An angle near π/2 should produce S followed by a small residual Rz."""
    angle = np.pi / 2 + 0.1
    qc = QuantumCircuit(1)
    qc.rz(angle, 0)

    result = _run_pass(FactorCliffordFromRz(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 2
    assert isinstance(ops[0], SGate)
    assert ops[1].name == "rz"
    assert np.isclose(ops[1].params[0], 0.1, atol=1e-10)


def test_factor_clifford_near_zero():
    """An angle near 0 should produce only the residual Rz (identity is dropped)."""
    angle = 0.05
    qc = QuantumCircuit(1)
    qc.rz(angle, 0)

    result = _run_pass(FactorCliffordFromRz(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert ops[0].name == "rz"
    assert np.isclose(ops[0].params[0], 0.05, atol=1e-10)


def test_factor_clifford_negative_angle():
    """A negative angle near -π/2 should factor out Sdg."""
    angle = -np.pi / 2 - 0.2
    qc = QuantumCircuit(1)
    qc.rz(angle, 0)

    result = _run_pass(FactorCliffordFromRz(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 2
    assert isinstance(ops[0], SdgGate)
    assert ops[0].name == "sdg"
    assert ops[1].name == "rz"
    assert np.isclose(ops[1].params[0], -0.2, atol=1e-10)


def test_factor_clifford_parameterized_untouched():
    """Parameterized Rz gates should be left untouched."""
    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.rz(theta, 0)

    result = _run_pass(FactorCliffordFromRz(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert ops[0].name == "rz"


# ---------------------------------------------------------------------------
# FactorPauliFromRotation tests
# ---------------------------------------------------------------------------


def test_factor_pauli_rz_exact_pi():
    """Rz(π) should become a single Z gate."""
    qc = QuantumCircuit(1)
    qc.rz(np.pi, 0)

    result = _run_pass(FactorPauliFromRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert isinstance(ops[0], ZGate)


def test_factor_pauli_rz_exact_2pi():
    """Rz(2π) is identity — the gate should be removed."""
    qc = QuantumCircuit(1)
    qc.rz(2 * np.pi, 0)

    result = _run_pass(FactorPauliFromRotation(), qc)

    assert result.size() == 0


def test_factor_pauli_rz_near_pi():
    """Rz(π + 0.1) should become Z followed by Rz(0.1)."""
    qc = QuantumCircuit(1)
    qc.rz(np.pi + 0.1, 0)

    result = _run_pass(FactorPauliFromRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 2
    assert isinstance(ops[0], ZGate)
    assert ops[1].name == "rz"
    assert np.isclose(ops[1].params[0], 0.1, atol=1e-10)


def test_factor_pauli_rz_small_angle():
    """Rz(0.05) is near 0·π, so just the reduced Rz(0.05) should remain."""
    qc = QuantumCircuit(1)
    qc.rz(0.05, 0)

    result = _run_pass(FactorPauliFromRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert ops[0].name == "rz"
    assert np.isclose(ops[0].params[0], 0.05, atol=1e-10)


def test_factor_pauli_rx_exact_pi():
    """Rx(π) should become a single X gate."""
    qc = QuantumCircuit(1)
    qc.rx(np.pi, 0)

    result = _run_pass(FactorPauliFromRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert isinstance(ops[0], XGate)


def test_factor_pauli_ry_near_pi():
    """Ry(π - 0.2) should become Y followed by Ry(-0.2)."""
    qc = QuantumCircuit(1)
    qc.ry(np.pi - 0.2, 0)

    result = _run_pass(FactorPauliFromRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 2
    assert isinstance(ops[0], YGate)
    assert ops[1].name == "ry"
    assert np.isclose(ops[1].params[0], -0.2, atol=1e-10)


def test_factor_pauli_rzz_exact_pi():
    """Rzz(π) should become Z on each qubit."""
    qc = QuantumCircuit(2)
    qc.rzz(np.pi, 0, 1)

    result = _run_pass(FactorPauliFromRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 2
    assert all(isinstance(op, ZGate) for op in ops)


def test_factor_pauli_rxx_near_pi():
    """Rxx(π + 0.1) should become X·X followed by Rxx(0.1)."""
    qc = QuantumCircuit(2)
    qc.rxx(np.pi + 0.1, 0, 1)

    result = _run_pass(FactorPauliFromRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 3
    assert isinstance(ops[0], XGate)
    assert isinstance(ops[1], XGate)
    assert ops[2].name == "rxx"
    assert np.isclose(ops[2].params[0], 0.1, atol=1e-10)


def test_factor_pauli_ryy_exact_2pi():
    """Ryy(2π) is identity — the gate should be removed."""
    qc = QuantumCircuit(2)
    qc.ryy(2 * np.pi, 0, 1)

    result = _run_pass(FactorPauliFromRotation(), qc)

    assert result.size() == 0


def test_factor_pauli_parameterized_untouched():
    """Parameterized rotation gates should be left untouched."""
    theta = Parameter("θ")
    qc = QuantumCircuit(2)
    qc.rzz(theta, 0, 1)

    result = _run_pass(FactorPauliFromRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert ops[0].name == "rzz"
