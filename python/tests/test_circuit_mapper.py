"""Tests for the non-controlled PauliSequenceMapper in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

import numpy as np
import pytest
import scipy
from qdk import TargetProfile

try:
    from qdk._native import Circuit as QdkCircuitType
except ImportError:
    from qsharp._native import Circuit as QdkCircuitType


from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.circuit_mapper.pauli_sequence_mapper import (
    PauliSequenceMapper,
)
from qdk_chemistry.data import SparsePauliProductFormulaContainer, SparsePauliTerms
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.qsharp import create_qsharp_context, use_qsharp_context

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator


@pytest.fixture
def simple_unitary() -> UnitaryRepresentation:
    """Create a simple UnitaryRepresentation for testing."""
    container = PauliProductFormulaContainer(
        step_terms=[
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
            ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.25),
        ],
        step_reps=2,
        num_qubits=2,
    )
    return UnitaryRepresentation(container=container)


class TestPauliSequenceMapperNonControlled:
    """Tests for the non-controlled PauliSequenceMapper class."""

    def test_name_and_type_name(self):
        """Test mapper identity methods."""
        mapper = PauliSequenceMapper()

        assert mapper.name() == "pauli_sequence"
        assert mapper.type_name() == "circuit_mapper"

    def test_run_builds_regular_unitary_circuit(self, simple_unitary):
        """Test run() builds a regular (non-controlled) unitary circuit."""
        mapper = PauliSequenceMapper()

        circuit = mapper.run(simple_unitary)

        assert isinstance(circuit, Circuit)
        assert isinstance(circuit.get_qsharp_circuit(), QdkCircuitType)

        qsc_json = json.loads(circuit.get_qsharp_circuit().json())
        num_qubits = len(qsc_json["qubits"])
        assert num_qubits == 2

    def test_run_builds_sparse_term_circuit(self):
        container = SparsePauliProductFormulaContainer.from_sparse_terms(
            SparsePauliTerms(2, [{0: "X"}, {1: "Z"}]),
            [0.5, 0.25],
            step_reps=2,
        )
        circuit = PauliSequenceMapper().run(UnitaryRepresentation(container=container))

        assert isinstance(circuit.get_qsharp_circuit(), QdkCircuitType)
        assert len(json.loads(circuit.get_qsharp_circuit().json())["qubits"]) == 2

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_unitary_circuit_matrix(self, simple_unitary):
        """Test that the constructed unitary circuit has the expected matrix."""
        mapper = PauliSequenceMapper()
        circuit = mapper.run(simple_unitary)

        container = simple_unitary.get_container()
        angle_x = container.step_terms[0].angle
        angle_z = container.step_terms[1].angle

        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        x_0 = np.kron(identity, pauli_x)
        z_1 = np.kron(pauli_z, identity)

        u_step = scipy.linalg.expm(-1j * angle_z * z_1) @ scipy.linalg.expm(-1j * angle_x * x_0)
        expected_matrix = np.linalg.matrix_power(u_step, container.step_reps)

        qc = circuit.get_qiskit_circuit()
        actual_matrix = Operator(qc).data

        assert np.allclose(
            actual_matrix,
            expected_matrix,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
@pytest.mark.parametrize("profile", [TargetProfile.Base, TargetProfile.Adaptive_RIF])
@pytest.mark.parametrize("variant", ["uncontrolled", "pauli_sequence"])
def test_fused_formula_matrix(profile: TargetProfile, variant: str) -> None:
    """Fused endpoints execute once and retain identity phases under control."""
    xx = ExponentiatedPauliTerm({0: "X", 1: "X"}, 0.13)
    yy = ExponentiatedPauliTerm({0: "Y", 1: "Y"}, -0.21)
    phase = ExponentiatedPauliTerm({}, 0.07)
    terms = [xx, yy, phase, ExponentiatedPauliTerm({0: "Z"}, 0.32), phase, yy, xx]
    formula = PauliProductFormulaContainer(terms, 3, 2, group_offsets=(0, 3, 4, 7), layer_offsets=(0, 1, 3, 4, 6, 7))
    with use_qsharp_context(create_qsharp_context(profile)):
        mapper = (
            create("circuit_mapper", "pauli_sequence")
            if variant == "uncontrolled"
            else create("controlled_circuit_mapper", variant, control_indices=[2], target_indices=[0, 1])
        )
        actual = Operator(mapper.run(UnitaryRepresentation(formula.combine(atol=0.0))).get_qiskit_circuit()).data
    axes = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    expected = np.eye(4 if variant == "uncontrolled" else 8, dtype=complex)
    for term in terms * 3:
        generator = np.kron(axes[term.pauli_term.get(1, "I")], axes[term.pauli_term.get(0, "I")])
        if variant != "uncontrolled":
            generator = np.kron(np.diag([0, 1]), generator)
        expected = scipy.linalg.expm(-1j * term.angle * generator) @ expected
    if variant == "uncontrolled":
        pivot = np.unravel_index(np.argmax(np.abs(expected)), expected.shape)
        actual *= expected[pivot] / actual[pivot]  # QIR may omit the circuit-global phase.
    else:
        actual /= actual[0, 0]  # Control-off amplitude fixes global phase, not relative phase.
    np.testing.assert_allclose(
        actual, expected, atol=float_comparison_absolute_tolerance, rtol=float_comparison_relative_tolerance
    )
