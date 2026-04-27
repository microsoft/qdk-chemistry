"""Test time evolution container functionality in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

import h5py
import numpy as np
import pytest

from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance


@pytest.fixture
def step_terms():
    """Create a list of ExponentiatedPauliTerm instances for testing."""
    return [
        ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
        ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=1.2),
        ExponentiatedPauliTerm(pauli_term={0: "Y", 1: "X"}, angle=0.3),
    ]


@pytest.fixture
def container(step_terms):
    """Create a PauliProductFormulaContainer instance for testing."""
    return PauliProductFormulaContainer(
        step_terms=step_terms,
        step_reps=4,
        num_qubits=2,
    )


class TestExponentiatedPauliTerm:
    """Tests for the ExponentiatedPauliTerm dataclass."""

    def test_attributes(self):
        """Test the attributes of ExponentiatedPauliTerm."""
        term = ExponentiatedPauliTerm(pauli_term={0: "X", 2: "Z"}, angle=1.57)

        assert term.pauli_term == {0: "X", 2: "Z"}
        assert np.isclose(
            term.angle, 1.57, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
        )

    def test_frozen(self):
        """Test that ExponentiatedPauliTerm is immutable."""
        term = ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1)
        with pytest.raises(Exception, match="cannot assign to field 'angle'"):
            term.angle = 0.2


class TestPauliProductFormulaContainer:
    """Tests for the PauliProductFormulaContainer class."""

    def test_basic_properties(self, container):
        """Test basic properties of the container."""
        assert container.type == "pauli_product_formula"
        assert container.num_qubits == 2
        assert container.step_reps == 4
        assert len(container.step_terms) == 3

    def test_update_ordering(self, container):
        """Test setting a new valid evolution ordering."""
        updated_container = container.reorder_terms([1, 2, 0])

        assert updated_container.step_terms[0] == container.step_terms[1]
        assert updated_container.step_terms[1] == container.step_terms[2]
        assert updated_container.step_terms[2] == container.step_terms[0]

    def test_update_ordering_invalid(self, container):
        """Test setting an invalid evolution ordering."""
        with pytest.raises(ValueError, match="must match the number of terms"):
            container.reorder_terms([0, 1])

        with pytest.raises(ValueError, match="Invalid permutation"):
            container.reorder_terms([0, 1, 3])

    def test_to_json_roundtrip(self, container):
        """Test JSON serialization and deserialization roundtrip."""
        json_data = container.to_json()
        restored = PauliProductFormulaContainer.from_json(json_data)

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps

        for t1, t2 in zip(restored.step_terms, container.step_terms, strict=True):
            assert t1.pauli_term == t2.pauli_term
            assert np.isclose(
                t1.angle, t2.angle, rtol=float_comparison_relative_tolerance, atol=float_comparison_absolute_tolerance
            )

    def test_from_json_pauli_term_keys_are_int(self, container):
        """Regression: JSON keys are strings, but pauli_term keys must be int after deserialization."""
        json_data = container.to_json()
        # Simulate a real JSON roundtrip where all dict keys become strings
        json_string = json.dumps(json_data)
        parsed = json.loads(json_string)

        restored = PauliProductFormulaContainer.from_json(parsed)

        for term in restored.step_terms:
            for key in term.pauli_term:
                assert isinstance(key, int), f"pauli_term key {key!r} should be int, got {type(key).__name__}"

    def test_to_hdf5_roundtrip(self, container, tmp_path):
        """Test HDF5 serialization and deserialization roundtrip."""
        file_path = tmp_path / "ppf_container.h5"

        with h5py.File(file_path, "w") as f:
            grp = f.create_group("container")
            container.to_hdf5(grp)

        with h5py.File(file_path, "r") as f:
            restored = PauliProductFormulaContainer.from_hdf5(f["container"])

        assert restored.type == container.type
        assert restored.num_qubits == container.num_qubits
        assert restored.step_reps == container.step_reps
        assert len(restored.step_terms) == len(container.step_terms)

    def test_summary(self, container):
        """Test the summary generation of the container."""
        summary = container.get_summary()

        assert "Pauli Product Formula Container" in summary
        assert "Number of qubits: 2" in summary
        assert "Number of step terms: 3" in summary
        assert "Step repetitions: 4" in summary


class TestPauliProductFormulaCirqConversion:
    """Test Cirq circuit conversion from PauliProductFormulaContainer."""

    def test_to_circuit_returns_circuit(self, step_terms):
        """Test that to_circuit returns a Circuit object with a Cirq representation."""
        import cirq  # noqa: PLC0415

        container = PauliProductFormulaContainer(step_terms=step_terms, step_reps=1, num_qubits=2)
        circuit = container.to_circuit()
        assert circuit is not None
        cirq_circuit = circuit.get_cirq_circuit()
        assert isinstance(cirq_circuit, cirq.AbstractCircuit)

    def test_to_circuit_single_qubit_terms(self):
        """Test conversion with single-qubit Pauli terms."""
        import cirq  # noqa: PLC0415

        terms = [ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5)]
        container = PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=1)
        cirq_circuit = container.to_circuit().get_cirq_circuit()
        ops = list(cirq_circuit.all_operations())
        assert len(ops) == 1
        assert isinstance(ops[0], cirq.PauliStringPhasor)

    def test_to_circuit_two_qubit_terms(self):
        """Test conversion with two-qubit Pauli terms."""
        terms = [ExponentiatedPauliTerm(pauli_term={0: "Z", 1: "Z"}, angle=0.25)]
        container = PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=2)
        cirq_circuit = container.to_circuit().get_cirq_circuit()
        ops = list(cirq_circuit.all_operations())
        assert len(ops) == 1

    def test_to_circuit_step_reps(self, step_terms):
        """Test that step_reps > 1 produces a CircuitOperation with repetitions."""
        import cirq  # noqa: PLC0415

        container = PauliProductFormulaContainer(step_terms=step_terms, step_reps=5, num_qubits=2)
        cirq_circuit = container.to_circuit().get_cirq_circuit()
        ops = list(cirq_circuit.all_operations())
        assert len(ops) == 1
        assert isinstance(ops[0], cirq.CircuitOperation)
        assert ops[0].repetitions == 5

    def test_to_circuit_empty_terms(self):
        """Test conversion with empty step terms."""
        container = PauliProductFormulaContainer(step_terms=[], step_reps=1, num_qubits=2)
        cirq_circuit = container.to_circuit().get_cirq_circuit()
        assert len(list(cirq_circuit.all_operations())) == 0

    def test_to_circuit_angle_mapping(self):
        """Test that the angle is correctly mapped to exponent_neg."""
        import math  # noqa: PLC0415

        import cirq  # noqa: PLC0415

        angle = 0.7
        terms = [ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=angle)]
        container = PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=1)
        cirq_circuit = container.to_circuit().get_cirq_circuit()
        op = next(iter(cirq_circuit.all_operations()))
        assert isinstance(op, cirq.PauliStringPhasor)
        assert abs(op.exponent_neg - angle / math.pi) < 1e-12
        assert op.exponent_pos == 0

    def test_to_circuit_preserves_term_order(self):
        """Test that the order of terms in the circuit matches the container."""
        import cirq  # noqa: PLC0415

        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
            ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=0.2),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.3),
        ]
        container = PauliProductFormulaContainer(step_terms=terms, step_reps=1, num_qubits=1)
        cirq_circuit = container.to_circuit().get_cirq_circuit()
        ops = list(cirq_circuit.all_operations())
        assert len(ops) == 3
        q = cirq.LineQubit(0)
        for op, expected_pauli in zip(ops, [cirq.X, cirq.Y, cirq.Z], strict=True):
            ps = op.pauli_string
            assert ps[q] == expected_pauli


class TestTimeEvolutionUnitaryToCircuit:
    """Test TimeEvolutionUnitary.to_circuit() delegation."""

    def test_to_circuit_delegates_to_container(self, step_terms):
        """Test that TimeEvolutionUnitary.to_circuit() works end-to-end."""
        from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitary  # noqa: PLC0415

        container = PauliProductFormulaContainer(step_terms=step_terms, step_reps=2, num_qubits=2)
        evolution = TimeEvolutionUnitary(container=container)
        circuit = evolution.to_circuit()
        assert circuit is not None
        cirq_circuit = circuit.get_cirq_circuit()
        assert cirq_circuit is not None
