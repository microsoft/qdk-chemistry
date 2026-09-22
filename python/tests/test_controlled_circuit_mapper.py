"""Tests for the PauliSequenceMapper and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from typing import ClassVar

import numpy as np
import pytest
import scipy
from qdk import qsharp

try:
    from qdk._native import Circuit as QdkCircuitType
except ImportError:
    from qsharp._native import Circuit as QdkCircuitType


from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.controlled_circuit_mapper.controlled_pauli_sequence_mapper import (
    ControlledPauliSequenceMapper,
)
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import dense_matrix

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator

#: ``dump_operation_on_state`` rounds to about six decimals, so exact agreement lands near 1e-6.
_TOL = 1e-5


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
def unitary_rep(simple_ppf_container):
    """Create a UnitaryRepresentation for testing."""
    return UnitaryRepresentation(container=simple_ppf_container)


class TestPauliSequenceMapper:
    """Tests for the PauliSequenceMapper class."""

    def test_name(self):
        """Test that the name method returns the correct algorithm name."""
        mapper = ControlledPauliSequenceMapper()
        assert mapper.name() == "pauli_sequence"

    def test_basic_mapping(self, unitary_rep):
        """Test basic mapping of unitary to Circuit."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])

        circuit = mapper.run(unitary_rep)

        assert isinstance(circuit, Circuit)
        assert isinstance(circuit.get_qsharp_circuit(), QdkCircuitType)

    def test_sparse_encoding_carries_only_non_identity_positions(self, unitary_rep):
        """The controlled mapper must reuse the sparse repeated-evolution parameters."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])

        circuit = mapper.run(unitary_rep)
        evo_params = vars(circuit._qsharp_factory.parameter["params"])

        assert "pauliExponents" not in evo_params
        assert evo_params["pauliIndices"] == [[0], [1]]
        assert evo_params["pauliOps"] == [[qsharp.Pauli.X], [qsharp.Pauli.Z]]

    def test_default_target_indices(self, unitary_rep):
        """Test that default target indices are used when none are provided."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])

        circuit = mapper.run(unitary_rep)
        qsc_json = json.loads(circuit.get_qsharp_circuit().json())
        num_qubits = len(qsc_json["qubits"])  # 2 system qubits + 1 control qubit
        assert num_qubits == 3

        def _find_control_qubits(node):
            """Recursively collect control-qubit indices for X gates."""
            control_qubits = []
            if isinstance(node, dict):
                if node.get("gate") == "X" and "controls" in node:
                    for ctrl in node["controls"]:
                        qubit_idx = ctrl.get("qubit")
                        if qubit_idx is not None:
                            control_qubits.append(qubit_idx)
                for value in node.values():
                    control_qubits.extend(_find_control_qubits(value))
            elif isinstance(node, list):
                for item in node:
                    control_qubits.extend(_find_control_qubits(item))
            return control_qubits

        # Check that there is at least one X gate controlled by qubit 2
        control_qubits = _find_control_qubits(qsc_json.get("componentGrid", []))

        assert set(control_qubits) == {2}

    def test_explicit_target_indices(self, unitary_rep):
        """Test that explicit target indices are used when provided."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])
        mapper.settings().set("target_indices", [0, 1])

        circuit = mapper.run(unitary_rep)
        assert isinstance(circuit, Circuit)

        mapper2 = ControlledPauliSequenceMapper()
        mapper2.settings().set("control_indices", [2])
        mapper2.settings().set("target_indices", [3, 4])

        circuit2 = mapper2.run(unitary_rep)
        assert isinstance(circuit2, Circuit)

    def test_invalid_container_type_raises(self):
        """Test that an invalid container type raises a ValueError."""

        # Create a new UnitaryRepresentation with invalid container type
        class MockContainer:
            """Mock container class."""

            @property
            def type(self):
                """Return mock container type."""
                return "mock_container"

        invalid_teu = UnitaryRepresentation(container=MockContainer())

        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])

        with pytest.raises(ValueError, match="not supported"):
            mapper.run(invalid_teu)

    def test_rotation_parameters(self, unitary_rep):
        """Test that rotation parameters are correctly set in the mapped circuit."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])

        circuit = mapper.run(unitary_rep)

        qsc_json = json.loads(circuit.get_qsharp_circuit().json())
        num_qubits = len(qsc_json["qubits"])  # 2 system qubits + 1 control qubit
        assert num_qubits == 3
        operations = qsc_json["componentGrid"][0]["components"][0]["children"][0]["components"][0]["children"]
        # Check that "X0" on qubit 0 and "Z1" on qubit 1 are present in the circuit with correct parameters
        for op in operations:
            for component in op["components"]:
                if component["gate"] == "Rz":
                    params = float(component["args"][0])
                    target_qubit = component["targets"][0]["qubit"]
                    if target_qubit == 0:
                        assert np.isclose(
                            abs(params),
                            0.5,
                            rtol=float_comparison_relative_tolerance,
                            atol=float_comparison_absolute_tolerance,
                        )  # X on qubit 0
                    elif target_qubit == 1:
                        assert np.isclose(
                            abs(params),
                            0.25,
                            rtol=float_comparison_relative_tolerance,
                            atol=float_comparison_absolute_tolerance,
                        )  # Z on qubit 1

    @pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
    def test_controlled_u_circuit_matrix(self, unitary_rep, simple_ppf_container):
        """Test that the constructed controlled-U circuit has the expected matrix."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])
        circuit = mapper.run(unitary_rep)

        # Extract angles from the container
        angle_x = simple_ppf_container.step_terms[0].angle
        angle_z = simple_ppf_container.step_terms[1].angle

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

        qc = circuit.get_qiskit_circuit()
        actual_matrix = Operator(qc).data

        assert np.allclose(
            actual_matrix,
            expected_matrix,
            atol=float_comparison_absolute_tolerance,
            rtol=float_comparison_relative_tolerance,
        )

    def test_duplicate_control_indices_raises(self, unitary_rep):
        """Test that duplicate control indices raise a ValueError."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2, 2])

        with pytest.raises(ValueError, match="duplicates"):
            mapper.run(unitary_rep)

    def test_duplicate_target_indices_raises(self, unitary_rep):
        """Test that duplicate target indices raise a ValueError."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [2])
        mapper.settings().set("target_indices", [0, 0])

        with pytest.raises(ValueError, match="duplicates"):
            mapper.run(unitary_rep)

    def test_overlapping_control_and_target_indices_raises(self, unitary_rep):
        """Test that overlapping control and target indices raise a ValueError."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [1])
        mapper.settings().set("target_indices", [0, 1])

        with pytest.raises(ValueError, match="overlap"):
            mapper.run(unitary_rep)

    def test_wrong_target_indices_length_raises(self, unitary_rep):
        """Test that target_indices length mismatching unitary qubit count raises ValueError."""
        mapper = ControlledPauliSequenceMapper()
        mapper.settings().set("control_indices", [3])
        mapper.settings().set("target_indices", [0, 1, 2])  # unitary has 2 qubits, not 3

        with pytest.raises(ValueError, match="length"):
            mapper.run(unitary_rep)


def _sparse_controlled_op(terms, *, repetitions=1):
    """Build controlled sparse evolution using typed Q# parameters."""
    params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(
        pauliIndices=[term["qubits"] for term in terms],
        pauliOps=[[getattr(qsharp.Pauli, axis) for axis in term["axes"]] for term in terms],
        pauliCoefficients=[term["angle"] for term in terms],
        repetitions=repetitions,
        beginning=0,
        end=0,
    )
    op = QSHARP_UTILS.PauliExp.MakeSparseRepPauliExpAdjCtlOp(params)
    return QSHARP_UTILS.CircuitComposition.MakeControlledOnFirstQubitOp(op)


def _dense_controlled_op(terms, num_qubits, *, repetitions=1):
    """Build controlled dense evolution using typed Q# parameters."""
    rows = []
    for term in terms:
        axes = [qsharp.Pauli.I] * num_qubits
        for qubit, axis in zip(term["qubits"], term["axes"], strict=True):
            axes[qubit] = getattr(qsharp.Pauli, axis)
        rows.append(axes)
    params = QSHARP_UTILS.PauliExp.RepPauliExpParams(
        pauliExponents=rows,
        pauliCoefficients=[term["angle"] for term in terms],
        repetitions=repetitions,
    )
    op = QSHARP_UTILS.PauliExp.MakeRepPauliExpAdjCtlOp(params)
    return QSHARP_UTILS.CircuitComposition.MakeControlledOnFirstQubitOp(op)


class TestSparseControlledEvolution:
    """Sparse controlled evolution must match control of the dense representation."""

    CASES: ClassVar[dict] = {
        "single-qubit mixed axes": (
            3,
            [
                {"qubits": [0], "axes": "X", "angle": 0.4},
                {"qubits": [1], "axes": "Y", "angle": -0.9},
                {"qubits": [2], "axes": "Z", "angle": 0.15},
            ],
        ),
        "two-qubit adjacent": (
            3,
            [
                {"qubits": [0, 1], "axes": "ZZ", "angle": 0.37},
                {"qubits": [1, 2], "axes": "XX", "angle": -0.5},
            ],
        ),
        "two-qubit non-adjacent, unsorted indices": (
            4,
            [
                {"qubits": [3, 0], "axes": "XY", "angle": 0.62},
                {"qubits": [2, 1], "axes": "YZ", "angle": -0.24},
            ],
        ),
        "an identity term among real ones": (
            2,
            [
                {"qubits": [0], "axes": "X", "angle": 0.5},
                {"qubits": [], "axes": "", "angle": 0.8},
                {"qubits": [1], "axes": "Z", "angle": -0.3},
            ],
        ),
        "identity": (2, [{"qubits": [], "axes": "", "angle": 0.7}]),
        "empty": (2, []),
    }

    @pytest.mark.parametrize("name", list(CASES))
    @pytest.mark.parametrize("repetitions", [1, 2])
    def test_sparse_matches_dense(self, name, repetitions):
        """Switching to sparse encoding must not change controlled evolution."""
        num_qubits, terms = self.CASES[name]
        got = dense_matrix(_sparse_controlled_op(terms, repetitions=repetitions), num_qubits + 1)
        want = dense_matrix(_dense_controlled_op(terms, num_qubits, repetitions=repetitions), num_qubits + 1)
        assert np.max(np.abs(got - want)) < _TOL

    @pytest.mark.parametrize("name", list(CASES))
    def test_control_off_branch_is_the_identity(self, name):
        """With the control off, sparse evolution must leave the targets unchanged."""
        num_qubits, terms = self.CASES[name]
        got = dense_matrix(_sparse_controlled_op(terms), num_qubits + 1)
        control_off_size = 2**num_qubits
        assert np.max(np.abs(got[:control_off_size, :control_off_size] - np.eye(control_off_size))) < _TOL


def test_declared_layers_reduce_rotation_depth() -> None:
    """The standard mapper preserves declared layers, reducing rotation rounds rather than gate counts."""
    assert create("controlled_circuit_mapper").name() == "pauli_sequence"
    terms = [ExponentiatedPauliTerm({2 * i: "X", 2 * i + 1: "Y"}, 0.123) for i in range(6)]
    counts = []
    for layer_offsets in (None, (0, 6)):
        unitary = UnitaryRepresentation(
            container=PauliProductFormulaContainer(terms, 2, 12, layer_offsets=layer_offsets)
        )
        mapper = create("controlled_circuit_mapper", "pauli_sequence")
        circuit = mapper.run(unitary)
        payload = circuit._qsharp_factory.parameter
        params = vars(payload["params"])
        assert params["repetitions"] == 2
        assert sum(map(len, params["pauliIndices"])) == 12
        assert payload["layerOffsets"] == list(layer_offsets or ())
        application = circuit.get_qre_application()
        counts.append(dict(get_qsharp_context().logical_counts(application.entry_expr, *application.args)))
    assert counts[0]["numQubits"] == counts[1]["numQubits"] == 13
    assert counts[0]["rotationCount"] == counts[1]["rotationCount"] == 24
    assert counts[1]["rotationDepth"] == 4
    assert counts[1]["rotationDepth"] < counts[0]["rotationDepth"]


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
@pytest.mark.parametrize("repetitions", [1, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_declared_layers_full_matrix(repetitions: int, empty: bool) -> None:
    """Preserve mixed supports, noncommuting order, identity phases, and permuted target wires."""
    terms = [
        ExponentiatedPauliTerm({3: "Y", 0: "X"}, -0.31),
        ExponentiatedPauliTerm({2: "Z", 1: "Y"}, 0.27),
        ExponentiatedPauliTerm({0: "Y"}, -0.19),
        ExponentiatedPauliTerm({}, 0.11),
        ExponentiatedPauliTerm({0: "I", 2: "Z"}, -0.23),
    ]
    if empty:
        terms = []
    targets = [3, 0, 2, 1]
    mapper = create("controlled_circuit_mapper", "pauli_sequence", control_indices=[4], target_indices=targets)
    container = PauliProductFormulaContainer(terms, repetitions, 4, layer_offsets=(0,) if empty else (0, 2, 5))
    circuit = mapper.run(UnitaryRepresentation(container=container))
    paulis = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    step = np.eye(16, dtype=complex)
    for term in terms:
        factors = {targets[q]: paulis[axis] for q, axis in term.pauli_term.items()}
        generator = np.ones((1, 1), dtype=complex)
        for q in reversed(range(4)):
            generator = np.kron(generator, factors.get(q, paulis["I"]))
        step = scipy.linalg.expm(-1j * term.angle * generator) @ step
    zero = np.zeros_like(step)
    expected = np.block([[np.eye(16), zero], [zero, np.linalg.matrix_power(step, repetitions)]])
    actual = Operator(circuit.get_qiskit_circuit()).data
    # Fix only the circuit-global phase QIR can elide, using the known control-off branch.
    actual /= actual[0, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
