"""Test for transpiler utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

from .reference_tolerances import float_comparison_absolute_tolerance

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import IGate, SdgGate, SGate, ZGate
    from qiskit.transpiler import PassManager

    from qdk_chemistry.plugins.qiskit._interop.transpiler import (
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
    ZGate = object
    PassManager = object
    MergeZBasisRotations = object
    RemoveZBasisOnZeroState = object
    SubstituteCliffordRz = object

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


def test_merge_z_basis_rotations_rz_gates():
    """Test MergeZBasisRotations correctly handles rz gates (not just s/z/sdg)."""
    qc = QuantumCircuit(1)
    qc.rz(np.pi / 4, 0)
    qc.rz(np.pi / 4, 0)

    result = _run_pass(MergeZBasisRotations(), qc)

    # Two rz(π/4) should merge into one rz(π/2)
    ops = result.count_ops()
    assert ops.get("rz", 0) == 1
    angle = result.data[0].operation.params[0]
    assert abs(angle - np.pi / 2) < float_comparison_absolute_tolerance


def test_merge_z_basis_rotations_net_zero():
    """Test MergeZBasisRotations removes gates when net rotation is zero."""
    qc = QuantumCircuit(1)
    qc.s(0)  # +π/2
    qc.sdg(0)  # -π/2  → net = 0

    result = _run_pass(MergeZBasisRotations(), qc)

    assert result.size() == 0


def test_merge_z_basis_rotations_parameterized_boundary():
    """Test MergeZBasisRotations flushes accumulator at parameterized Rz boundary."""
    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.s(0)  # π/2 accumulated
    qc.rz(theta, 0)  # boundary: flush π/2 as rz, keep parameterized gate
    qc.z(0)  # new accumulation: π

    result = _run_pass(MergeZBasisRotations(), qc)

    ops = result.count_ops()
    # Should have: rz(π/2), rz(θ), rz(π) — three separate rz gates
    assert ops.get("rz", 0) == 3
    assert "s" not in ops
    assert "z" not in ops


def test_merge_z_basis_rotations_multi_qubit():
    """Test MergeZBasisRotations handles multiple qubits independently."""
    qc = QuantumCircuit(2)
    qc.s(0)
    qc.z(1)
    qc.sdg(0)
    qc.s(1)
    qc.cx(0, 1)  # boundary for both qubits
    qc.rz(np.pi / 4, 0)

    result = _run_pass(MergeZBasisRotations(), qc)

    ops = result.count_ops()
    assert "s" not in ops
    assert "sdg" not in ops
    assert "z" not in ops
    # q0: s + sdg = 0 → removed; then rz(π/4) after cx
    # q1: z + s = 3π/2 → single rz before cx
    assert ops.get("rz", 0) == 2
    assert ops.get("cx", 0) == 1


def test_merge_z_basis_rotations_single_gate_unchanged():
    """Test MergeZBasisRotations leaves a single rz gate unchanged."""
    qc = QuantumCircuit(1)
    qc.rz(np.pi / 3, 0)

    result = _run_pass(MergeZBasisRotations(), qc)

    assert result.count_ops().get("rz", 0) == 1
    assert abs(result.data[0].operation.params[0] - np.pi / 3) < float_comparison_absolute_tolerance


def test_merge_z_basis_rotations_id_removal():
    """Test MergeZBasisRotations removes identity gates."""
    qc = QuantumCircuit(1)
    qc.id(0)
    qc.id(0)
    qc.h(0)

    result = _run_pass(MergeZBasisRotations(), qc)

    assert "id" not in result.count_ops()
    assert result.count_ops().get("h", 0) == 1


def test_substitute_clifford_rz_settings_update():
    """Test SubstituteCliffordRzSettings.update ensures 'id' in gate set."""
    scr = SubstituteCliffordRz(equivalent_gate_set=["z"])
    scr.settings().update({"equivalent_gate_set": ["s", "sdg"]})
    gate_set = scr.settings().get("equivalent_gate_set")
    assert "id" in gate_set


def test_remove_z_basis_on_zero_state_diagonal_gate_preserves_zero():
    """Test RemoveZBasisOnZeroState treats diagonal gates as preserving |0⟩."""
    qc = QuantumCircuit(1)
    qc.rz(np.pi / 3, 0)  # diagonal + |0⟩ → removed, qubit stays |0⟩
    qc.s(0)  # still |0⟩ → removed

    result = _run_pass(RemoveZBasisOnZeroState(), qc)

    assert result.size() == 0


def test_remove_z_basis_on_zero_state_multi_qubit():
    """Test RemoveZBasisOnZeroState handles multiple qubits independently."""
    qc = QuantumCircuit(2)
    qc.s(0)  # q0 |0⟩ → removed
    qc.h(0)  # q0 leaves |0⟩
    qc.z(1)  # q1 still |0⟩ → removed
    qc.rz(0.5, 0)  # q0 no longer |0⟩ → kept

    result = _run_pass(RemoveZBasisOnZeroState(), qc)

    ops = result.count_ops()
    assert "s" not in ops
    assert "z" not in ops
    assert ops.get("h", 0) == 1
    assert ops.get("rz", 0) == 1
