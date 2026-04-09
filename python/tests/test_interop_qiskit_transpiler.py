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
        MergeZBasisRotations,
        RemoveZBasisOnZeroState,
        SubstituteCliffordRz,
        SubstitutePauli1QRotation,
        SubstitutePauli2QRotation,
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
    SubstitutePauli1QRotation = object
    SubstitutePauli2QRotation = object


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
# SubstitutePauli1QRotation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gate_method", "angle", "expected_gate"),
    [
        ("rx", np.pi, XGate),
        ("ry", np.pi, YGate),
        ("rz", np.pi, ZGate),
        ("rx", 3 * np.pi, XGate),
        ("rz", -np.pi, ZGate),
    ],
)
def test_substitute_pauli_1q_odd_pi(gate_method, angle, expected_gate):
    """Odd multiples of π should be substituted with the Pauli gate."""
    qc = QuantumCircuit(1)
    getattr(qc, gate_method)(angle, 0)

    result = _run_pass(SubstitutePauli1QRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert isinstance(ops[0], expected_gate)


@pytest.mark.parametrize(
    ("gate_method", "angle"),
    [
        ("rx", 0.0),
        ("ry", 2 * np.pi),
        ("rz", 4 * np.pi),
    ],
)
def test_substitute_pauli_1q_even_pi(gate_method, angle):
    """Even multiples of π (including 0) should be removed (identity)."""
    qc = QuantumCircuit(1)
    getattr(qc, gate_method)(angle, 0)

    result = _run_pass(SubstitutePauli1QRotation(), qc)

    assert result.size() == 0


@pytest.mark.parametrize(
    ("gate_method", "angle"),
    [
        ("rx", 0.5),
        ("ry", np.pi / 2),
        ("rz", np.pi + 0.1),
    ],
)
def test_substitute_pauli_1q_non_pi_unchanged(gate_method, angle):
    """Angles that are not integer multiples of π should be left untouched."""
    qc = QuantumCircuit(1)
    getattr(qc, gate_method)(angle, 0)

    result = _run_pass(SubstitutePauli1QRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert ops[0].name == gate_method


def test_substitute_pauli_1q_parameterized_untouched():
    """Parameterized rotation gates should be left untouched."""
    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.rz(theta, 0)

    result = _run_pass(SubstitutePauli1QRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert ops[0].name == "rz"


def test_substitute_pauli_1q_mixed_circuit():
    """A circuit with multiple rotation gates should have each handled independently."""
    qc = QuantumCircuit(1)
    qc.rx(np.pi, 0)  # odd → X
    qc.ry(2 * np.pi, 0)  # even → removed
    qc.rz(np.pi, 0)  # odd → Z
    qc.rx(0.5, 0)  # not a multiple → kept

    result = _run_pass(SubstitutePauli1QRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 3
    assert isinstance(ops[0], XGate)
    assert isinstance(ops[1], ZGate)
    assert ops[2].name == "rx"


def test_substitute_pauli_1q_selective_gate_set():
    """Only Pauli gates in equivalent_gate_set should be substituted."""
    qc = QuantumCircuit(1)
    qc.rx(np.pi, 0)  # would become X
    qc.rz(np.pi, 0)  # would become Z

    # Only allow X substitution, not Z
    result = _run_pass(SubstitutePauli1QRotation(equivalent_gate_set=["id", "x"]), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 2
    assert isinstance(ops[0], XGate)
    assert ops[1].name == "rz"  # Z not in gate set, so kept as rz


def test_substitute_pauli_1q_settings():
    """Test settings initialization and 'id' auto-inclusion."""
    p = SubstitutePauli1QRotation(equivalent_gate_set=["x"])
    assert "id" in p.settings().get("equivalent_gate_set")

    with pytest.raises(TypeError):
        SubstitutePauli1QRotation(equivalent_gate_set="x")


# ---------------------------------------------------------------------------
# SubstitutePauli2QRotation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gate_method", "angle", "expected_gate"),
    [
        ("rxx", np.pi, XGate),
        ("ryy", np.pi, YGate),
        ("rzz", np.pi, ZGate),
        ("rzz", 3 * np.pi, ZGate),
    ],
)
def test_substitute_pauli_2q_odd_pi(gate_method, angle, expected_gate):
    """Odd multiples of π should produce the Pauli on each qubit."""
    qc = QuantumCircuit(2)
    getattr(qc, gate_method)(angle, 0, 1)

    result = _run_pass(SubstitutePauli2QRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 2
    assert all(isinstance(op, expected_gate) for op in ops)


@pytest.mark.parametrize(
    ("gate_method", "angle"),
    [
        ("rxx", 2 * np.pi),
        ("ryy", 0.0),
        ("rzz", 4 * np.pi),
    ],
)
def test_substitute_pauli_2q_even_pi(gate_method, angle):
    """Even multiples of π (including 0) should be removed (identity)."""
    qc = QuantumCircuit(2)
    getattr(qc, gate_method)(angle, 0, 1)

    result = _run_pass(SubstitutePauli2QRotation(), qc)

    assert result.size() == 0


@pytest.mark.parametrize(
    ("gate_method", "angle"),
    [
        ("rxx", np.pi + 0.1),
        ("ryy", 0.5),
        ("rzz", np.pi / 2),
    ],
)
def test_substitute_pauli_2q_non_pi_unchanged(gate_method, angle):
    """Non-integer-π angles should be left untouched."""
    qc = QuantumCircuit(2)
    getattr(qc, gate_method)(angle, 0, 1)

    result = _run_pass(SubstitutePauli2QRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert ops[0].name == gate_method


def test_substitute_pauli_2q_parameterized_untouched():
    """Parameterized 2-qubit rotation gates should be left untouched."""
    theta = Parameter("θ")
    qc = QuantumCircuit(2)
    qc.rzz(theta, 0, 1)

    result = _run_pass(SubstitutePauli2QRotation(), qc)

    ops = [instr.operation for instr in result.data]
    assert len(ops) == 1
    assert ops[0].name == "rzz"


def test_substitute_pauli_2q_mixed_circuit():
    """A circuit with multiple 2Q rotation gates should have each handled independently."""
    qc = QuantumCircuit(2)
    qc.rxx(np.pi, 0, 1)  # odd → X⊗X
    qc.ryy(2 * np.pi, 0, 1)  # even → removed
    qc.rzz(3 * np.pi, 0, 1)  # odd → Z⊗Z

    result = _run_pass(SubstitutePauli2QRotation(), qc)

    ops = [instr.operation for instr in result.data]
    names = [op.name for op in ops]
    assert len(ops) == 4
    assert names.count("x") == 2
    assert names.count("z") == 2
