"""Test for transpiler utilities in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import IGate, SdgGate, SGate, ZGate
from qiskit.transpiler import PassManager

from qdk_chemistry.plugins.qiskit._interop.transpiler import (
    MergeZBasisRotations,
    RemoveZBasisOnZeroState,
    SubstituteCliffordRz,
)


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
