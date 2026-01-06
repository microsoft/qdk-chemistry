"""Tests for the ChainStructureMapper and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import re

import pytest
from qiskit import QuantumCircuit

from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.chain_structure_mapper import (
    ChainStructureMapper,
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
        control_index=2,
    )


class TestChainStructureMapper:
    """Tests for the ChainStructureMapper class."""

    def test_name(self):
        """Test that the name method returns the correct algorithm name."""
        mapper = ChainStructureMapper()
        assert mapper.name() == "chain_structure"

    def test_basic_mapping(self, controlled_unitary):
        """Test basic mapping of ControlledTimeEvolutionUnitary to Circuit."""
        mapper = ChainStructureMapper(power=1)

        circuit = mapper.run(controlled_unitary)

        assert isinstance(circuit, Circuit)
        assert isinstance(circuit.qasm, str)
        assert "OPENQASM" in circuit.qasm

    def test_default_system_indices(self, controlled_unitary):
        """Test that default system indices are used when none are provided."""
        mapper = ChainStructureMapper()

        circuit = mapper.run(controlled_unitary)

        assert re.search(r"crz\s*\([^)]*\)\s+q\[2\]\s*,\s*q\[\d+\]", circuit.qasm)

    def test_invalid_container_type_raises(self):
        """Test that an invalid container type raises a ValueError."""

        # Create a new TimeEvolutionUnitary with invalid container type
        class MockContainer:
            """Mock container class."""

            def type(self):
                """Return mock container type."""
                return "mock_container"

        invalid_teu = TimeEvolutionUnitary(container=MockContainer())
        invalid_controlled = ControlledTimeEvolutionUnitary(
            time_evolution_unitary=invalid_teu,
            control_index=2,
        )

        mapper = ChainStructureMapper()

        with pytest.raises(ValueError, match="not supported"):
            mapper.run(invalid_controlled)


class TestAppendControlledTimeEvolution:
    """Tests for the append_controlled_time_evolution function."""

    def test_power_validation(self, controlled_unitary):
        """Test that invalid power raises a ValueError."""
        qc = QuantumCircuit(3)

        with pytest.raises(ValueError, match="power must be at least 1"):
            append_controlled_time_evolution(
                qc,
                controlled_unitary,
                system_indices=[0, 1],
                power=0,
            )

    def test_appends_operations(self, controlled_unitary):
        """Test that controlled time evolution operations are appended to the circuit."""
        qc = QuantumCircuit(3)

        append_controlled_time_evolution(
            qc,
            controlled_unitary,
            system_indices=[0, 1],
            power=2,
        )

        assert qc.count_ops().get("crz", 0) == 4

    def test_skips_zero_angle_terms(self):
        """Test that terms with zero angle are skipped."""
        container = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.0),
            ],
            step_reps=1,
            num_qubits=1,
        )
        teu = TimeEvolutionUnitary(container=container)
        controlled = ControlledTimeEvolutionUnitary(teu, control_index=1)

        qc = QuantumCircuit(2)

        append_controlled_time_evolution(
            qc,
            controlled,
            system_indices=[0],
            power=1,
        )

        # No CRZ should be added
        assert "crz" not in qc.count_ops()


class TestAppendControlledPauliRotation:
    """Tests for the _append_controlled_pauli_rotation helper function."""

    def test_identity_term_adds_phase(self):
        """Test that identity terms add a controlled phase gate."""
        qc = QuantumCircuit(1)
        term = ExponentiatedPauliTerm(pauli_term={}, angle=0.3)

        _append_controlled_pauli_rotation(
            qc,
            control_qubit=0,
            system_qubits=[],
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
            system_qubits=[0],
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
            system_qubits=[0, 1],
            term=term,
        )

        ops = qc.count_ops()
        assert ops.get("cx", 0) >= 2
        assert ops.get("crz", 0) == 1
