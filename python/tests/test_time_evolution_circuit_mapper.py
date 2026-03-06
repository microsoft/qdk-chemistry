"""Tests for the PauliSequenceMapper and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import numpy as np
import pytest
import scipy
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator

from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.pauli_sequence_mapper import (
    PauliSequenceMapper,
    _append_controlled_pauli_rotation,
    append_controlled_time_evolution,
)
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.data.time_evolution.controlled_time_evolution import (
    ControlledTimeEvolutionUnitary,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


@pytest.fixture
def simple_ppf_container():
    """Create a simple PauliProductFormulaContainer for testing."""
    terms = [
        ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
        ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.25),
    ]

    return PauliProductFormulaContainer(
        step_terms=terms,
        step_reps=1,
        num_qubits=2,
    )


@pytest.fixture
def controlled_unitary(simple_ppf_container):
    """Create a ControlledTimeEvolutionUnitary for testing."""
    teu = TimeEvolutionUnitary(container=simple_ppf_container)
    return ControlledTimeEvolutionUnitary(
        time_evolution_unitary=teu,
        control_indices=[2],
    )


class TestPauliSequenceMapper:
    """Tests for the PauliSequenceMapper class."""

    def test_name(self):
        """Test that the name method returns the correct algorithm name."""
        mapper = PauliSequenceMapper()
        assert mapper.name() == "pauli_sequence"

    def test_basic_mapping(self, controlled_unitary):
        """Test basic mapping of ControlledTimeEvolutionUnitary to Circuit."""
        mapper = PauliSequenceMapper(power=1)

        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert isinstance(circuit.qasm, str)
        assert "OPENQASM" in circuit.qasm

    def test_default_target_indices(self, controlled_unitary):
        """Test that default target indices are used when none are provided."""
        mapper = PauliSequenceMapper()

        circuit = mapper.run(controlled_unitary)

        # control qubit is at index 2, so target qubits should be [0, 1]
        assert re.search(r"crz\s*\([^)]*\)\s+_gate_q_2\s*,\s*_gate_q_", circuit.qasm)

    def test_invalid_container_type_raises(self):
        """Test that an invalid container type raises a ValueError."""

        # Create a new TimeEvolutionUnitary with invalid container type
        class MockContainer:
            """Mock container class."""

            @property
            def type(self):
                """Return mock container type."""
                return "mock_container"

        invalid_teu = TimeEvolutionUnitary(container=MockContainer())
        invalid_controlled = ControlledTimeEvolutionUnitary(
            time_evolution_unitary=invalid_teu,
            control_indices=[2],
        )

        mapper = PauliSequenceMapper()

        with pytest.raises(ValueError, match="not supported"):
            mapper.run(invalid_controlled)

    def test_rotation_parameters(self, controlled_unitary):
        """Test that rotation parameters are correctly set in the mapped circuit."""
        mapper = PauliSequenceMapper(power=1)

        circuit = mapper.run(controlled_unitary)

        # Check that the angles in the CRZ gates are correctly set
        crz_angles = re.findall(r"crz\s*\(\s*([^\)]+)\s*\)", circuit.qasm)
        container = controlled_unitary.time_evolution_unitary.get_container()
        expected_angles = [
            f"{2 * container.step_terms[0].angle:.1f}",
            f"{2 * container.step_terms[1].angle:.1f}",
        ]

        assert crz_angles == expected_angles

    def test_controlled_u_circuit_matrix(self, controlled_unitary):
        """Test that the constructed controlled-U circuit has the expected matrix."""
        mapper = PauliSequenceMapper(power=1)
        circuit = mapper.run(controlled_unitary)

        # Extract angles from the container
        container = controlled_unitary.time_evolution_unitary.get_container()
        angle_x = container.step_terms[0].angle
        angle_z = container.step_terms[1].angle

        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        x_0 = np.kron(identity, pauli_x)
        z_1 = np.kron(pauli_z, identity)
        u_1 = scipy.linalg.expm(-1j * angle_x * x_0)
        u_2 = scipy.linalg.expm(-1j * angle_z * z_1)
        u = u_2 @ u_1

        # CU = (|0><0| ⊗ I₄) + (|1><1| ⊗ U)
        p_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        p_1 = np.array([[0, 0], [0, 1]], dtype=complex)
        i_4 = np.eye(4, dtype=complex)
        expected_matrix = np.kron(p_0, i_4) + np.kron(p_1, u)

        # Get actual matrix from the circuit
        qc = qasm3.loads(circuit.qasm)
        actual_matrix = Operator(qc).data

        assert np.allclose(
            actual_matrix,
            expected_matrix,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )


class TestAppendControlledTimeEvolution:
    """Tests for the append_controlled_time_evolution function."""

    def test_power_validation(self, controlled_unitary):
        """Test that invalid power raises a ValueError."""
        qc = QuantumCircuit(3)
        container = controlled_unitary.time_evolution_unitary.get_container()

        with pytest.raises(ValueError, match="power must be at least 1"):
            append_controlled_time_evolution(
                qc,
                exponential_terms=container.step_terms,
                reps=container.step_reps,
                control_qubit=controlled_unitary.control_indices[0],
                target_qubits=[0, 1],
                power=0,
            )

    def test_appends_operations(self, controlled_unitary):
        """Test that controlled time evolution operations are appended to the circuit."""
        qc = QuantumCircuit(3)
        container = controlled_unitary.time_evolution_unitary.get_container()

        append_controlled_time_evolution(
            qc,
            exponential_terms=container.step_terms,
            reps=container.step_reps,
            control_qubit=controlled_unitary.control_indices[0],
            target_qubits=[0, 1],
            power=2,
        )
        ops = qc.count_ops()
        assert ops == {"ctrl_time_evol_power_2": 1}

    def test_skips_zero_angle_terms(self):
        """Test that terms with zero angle are skipped."""
        qc = QuantumCircuit(2)

        append_controlled_time_evolution(
            qc,
            exponential_terms=[ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.0)],
            reps=1,
            control_qubit=1,
            target_qubits=[0],
            power=1,
        )

        # No CRZ should be added
        cir_qasm = qasm3.dumps(qc)
        assert "crz" not in cir_qasm


class TestAppendControlledPauliRotation:
    """Tests for the _append_controlled_pauli_rotation helper function."""

    def test_identity_term_adds_phase(self):
        """Test that identity terms add a controlled phase gate."""
        qc = QuantumCircuit(1)
        term = ExponentiatedPauliTerm(pauli_term={}, angle=0.3)

        _append_controlled_pauli_rotation(
            qc,
            control_qubit=0,
            target_qubits=[],
            term=term,
        )

        assert qc.count_ops().get("p", 0) == 1

    def test_single_pauli_rotation(self):
        """Test appending a single-qubit controlled Pauli rotation."""
        qc = QuantumCircuit(2)
        term = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.2)

        _append_controlled_pauli_rotation(
            qc,
            control_qubit=1,
            target_qubits=[0],
            term=term,
        )

        assert qc.count_ops().get("crz", 0) == 1

    def test_multi_qubit_pauli_chain(self):
        """Test appending a multi-qubit controlled Pauli rotation."""
        qc = QuantumCircuit(3)
        term = ExponentiatedPauliTerm(pauli_term={0: "X", 1: "Y"}, angle=0.4)

        _append_controlled_pauli_rotation(
            qc,
            control_qubit=2,
            target_qubits=[0, 1],
            term=term,
        )

        ops = qc.count_ops()
        assert ops.get("cx", 0) >= 2
        assert ops.get("crz", 0) == 1
