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

    def test_combine_no_adjacent_identical(self):
        """Test combine when no adjacent terms share the same Pauli string."""
        a = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
                ExponentiatedPauliTerm(pauli_term={1: "Z"}, angle=0.2),
            ],
            step_reps=2,
            num_qubits=2,
        )
        b = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=0.3),
                ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.4),
            ],
            step_reps=2,
            num_qubits=2,
        )
        result = a.combine(b)

        # a expanded: [X, Z, X, Z], b expanded: [Y, X, Y, X]
        # No adjacent duplicates anywhere, so all 8 terms survive.
        assert result.step_reps == 1
        assert len(result.step_terms) == 8
        expected_angles = [0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.3, 0.4]
        for term, expected in zip(result.step_terms, expected_angles, strict=True):
            assert np.isclose(term.angle, expected, atol=1e-14)

    def test_combine_with_adjacent_identical(self):
        """Test combine where adjacent identical Pauli terms get merged."""
        a = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=1.5),
                ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5),
            ],
            step_reps=2,
            num_qubits=1,
        )
        b = PauliProductFormulaContainer(
            step_terms=[
                ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.7),
                ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=1.5),
            ],
            step_reps=1,
            num_qubits=1,
        )
        result = a.combine(b)

        # a expanded: [Y(1.5), X(0.5), Y(1.5), X(0.5)], b expanded: [X(0.7), Z(1.5)]
        # Only the two adjacent X terms at the boundary are merged into X(1.2)
        assert result.step_reps == 1
        assert len(result.step_terms) == 5

        assert result.step_terms[0].pauli_term == {0: "Y"}
        assert np.isclose(result.step_terms[0].angle, 1.5, atol=1e-14)
        assert result.step_terms[1].pauli_term == {0: "X"}
        assert np.isclose(result.step_terms[1].angle, 0.5, atol=1e-14)
        assert result.step_terms[2].pauli_term == {0: "Y"}
        assert np.isclose(result.step_terms[2].angle, 1.5, atol=1e-14)
        assert result.step_terms[3].pauli_term == {0: "X"}
        assert np.isclose(result.step_terms[3].angle, 1.2, atol=1e-14)
        assert result.step_terms[4].pauli_term == {0: "Z"}
        assert np.isclose(result.step_terms[4].angle, 1.5, atol=1e-14)

    def test_summary(self, container):
        """Test the summary generation of the container."""
        summary = container.get_summary()

        assert "Pauli Product Formula Container" in summary
        assert "Number of qubits: 2" in summary
        assert "Number of step terms: 3" in summary
        assert "Step repetitions: 4" in summary


class TestCirqPauliStringMapper:
    """Test CirqPauliStringMapper circuit conversion."""

    def _map(self, step_terms, step_reps, num_qubits):
        """Helper: build a TimeEvolutionUnitary and map it to a Circuit via CirqPauliStringMapper."""
        from qdk_chemistry.algorithms.time_evolution.circuit_mapper.cirq_pauli_string_mapper import (  # noqa: PLC0415
            CirqPauliStringMapper,
        )
        from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitary  # noqa: PLC0415

        container = PauliProductFormulaContainer(step_terms=step_terms, step_reps=step_reps, num_qubits=num_qubits)
        evolution = TimeEvolutionUnitary(container=container)
        mapper = CirqPauliStringMapper()
        return mapper.run(evolution)

    def test_returns_circuit_with_cirq(self, step_terms):
        """Test that the mapper returns a Circuit with a Cirq representation."""
        import cirq  # noqa: PLC0415

        circuit = self._map(step_terms, step_reps=1, num_qubits=2)
        assert circuit is not None
        cirq_circuit = circuit.get_cirq_circuit()
        assert isinstance(cirq_circuit, cirq.AbstractCircuit)

    def test_single_qubit_terms(self):
        """Test conversion with single-qubit Pauli terms."""
        import cirq  # noqa: PLC0415

        terms = [ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.5)]
        cirq_circuit = self._map(terms, 1, 1).get_cirq_circuit()
        ops = list(cirq_circuit.all_operations())
        assert len(ops) == 1
        assert isinstance(ops[0], cirq.PauliStringPhasor)

    def test_two_qubit_terms(self):
        """Test conversion with two-qubit Pauli terms."""
        terms = [ExponentiatedPauliTerm(pauli_term={0: "Z", 1: "Z"}, angle=0.25)]
        cirq_circuit = self._map(terms, 1, 2).get_cirq_circuit()
        ops = list(cirq_circuit.all_operations())
        assert len(ops) == 1

    def test_step_reps(self, step_terms):
        """Test that step_reps > 1 produces a CircuitOperation with repetitions."""
        import cirq  # noqa: PLC0415

        cirq_circuit = self._map(step_terms, step_reps=5, num_qubits=2).get_cirq_circuit()
        ops = list(cirq_circuit.all_operations())
        assert len(ops) == 1
        assert isinstance(ops[0], cirq.CircuitOperation)
        assert ops[0].repetitions == 5

    def test_empty_terms(self):
        """Test conversion with empty step terms."""
        cirq_circuit = self._map([], 1, 2).get_cirq_circuit()
        assert len(list(cirq_circuit.all_operations())) == 0

    def test_angle_mapping(self):
        """Test that the angle is correctly mapped to exponent_neg = 2*angle/pi."""
        import math  # noqa: PLC0415

        import cirq  # noqa: PLC0415

        angle = 0.7
        terms = [ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=angle)]
        cirq_circuit = self._map(terms, 1, 1).get_cirq_circuit()
        op = next(iter(cirq_circuit.all_operations()))
        assert isinstance(op, cirq.PauliStringPhasor)
        assert abs(op.exponent_neg - 2 * angle / math.pi) < 1e-12
        assert op.exponent_pos == 0

    def test_preserves_term_order(self):
        """Test that the order of terms in the circuit matches the container."""
        import cirq  # noqa: PLC0415

        terms = [
            ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.1),
            ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=0.2),
            ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.3),
        ]
        cirq_circuit = self._map(terms, 1, 1).get_cirq_circuit()
        ops = list(cirq_circuit.all_operations())
        assert len(ops) == 3
        q = cirq.LineQubit(0)
        for op, expected_pauli in zip(ops, [cirq.X, cirq.Y, cirq.Z], strict=True):
            ps = op.pauli_string
            assert ps[q] == expected_pauli

    def test_end_to_end_with_trotter(self, step_terms):
        """Test mapper works end-to-end from TimeEvolutionUnitary."""
        from qdk_chemistry.algorithms.time_evolution.circuit_mapper.cirq_pauli_string_mapper import (  # noqa: PLC0415
            CirqPauliStringMapper,
        )
        from qdk_chemistry.data.time_evolution.base import TimeEvolutionUnitary  # noqa: PLC0415

        container = PauliProductFormulaContainer(step_terms=step_terms, step_reps=2, num_qubits=2)
        evolution = TimeEvolutionUnitary(container=container)
        mapper = CirqPauliStringMapper()
        circuit = mapper.run(evolution)
        assert circuit is not None
        cirq_circuit = circuit.get_cirq_circuit()
        assert cirq_circuit is not None
