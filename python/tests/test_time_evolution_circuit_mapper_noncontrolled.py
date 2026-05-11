"""Tests for the non-controlled PauliSequenceMapper in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

import numpy as np
import pytest
import qsharp
import scipy

from qdk_chemistry.algorithms.time_evolution.circuit_mapper.pauli_sequence_mapper import PauliSequenceMapper
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

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
        assert mapper.type_name() == "evolution_circuit_mapper"

    def test_run_builds_regular_unitary_circuit(self, simple_unitary):
        """Test run() builds a regular (non-controlled) unitary circuit."""
        mapper = PauliSequenceMapper()

        circuit = mapper.run(simple_unitary)

        assert isinstance(circuit, Circuit)
        assert isinstance(circuit.get_qsharp_circuit(), qsharp._native.Circuit)

        qsc_json = json.loads(circuit.get_qsharp_circuit().json())
        num_qubits = len(qsc_json["qubits"])
        assert num_qubits == 2

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
