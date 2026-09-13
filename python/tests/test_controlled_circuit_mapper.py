"""Tests for the PauliSequenceMapper and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json

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
from qdk_chemistry.utils.qsharp import get_qsharp_context

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit.quantum_info import Operator


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


def _map_sparse_formula(
    terms: list[ExponentiatedPauliTerm],
    packed: bool,
    repetitions: int,
    num_qubits: int = 2,
    targets: list[int] | None = None,
    *,
    variant: str = "pauli_sequence",
) -> Circuit:
    """Map equivalent legacy or packed terms without widening their support."""
    if packed:
        offsets, indices, codes = [0], [], []
        for term in terms:
            for index, pauli in term.pauli_term.items():
                if pauli != "I":  # Packed storage represents identity words by empty support.
                    indices.append(index)
                    codes.append("IXYZ".index(pauli))
            offsets.append(len(indices))
        container = PauliProductFormulaContainer.from_sparse_arrays(
            np.array(offsets),
            np.array(indices),
            np.array(codes),
            np.array([term.angle for term in terms]),
            step_reps=repetitions,
            num_qubits=num_qubits,
        )
    else:
        container = PauliProductFormulaContainer(terms, step_reps=repetitions, num_qubits=num_qubits)
    mapper = create("controlled_circuit_mapper", variant)
    mapper.settings().set("control_indices", [num_qubits])
    if targets is not None:
        mapper.settings().set("target_indices", targets)
    return mapper.run(UnitaryRepresentation(container=container))


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
@pytest.mark.parametrize("variant", ["pauli_sequence", "batched_pauli_sequence"])
@pytest.mark.parametrize("packed", [False, True], ids=["legacy", "packed"])
@pytest.mark.parametrize("repetitions", [1, 3])
@pytest.mark.parametrize(
    ("empty", "targets"),
    [(False, [0, 1]), (True, [0, 1]), (False, [4, 1])],
    ids=["mixed", "empty", "reordered-noncontiguous"],
)
def test_sparse_controlled_matrix(
    packed: bool, repetitions: int, empty: bool, targets: list[int], variant: str
) -> None:
    """Preserve order, sign, identity-relative phases, and spectator qubits exactly."""
    terms = [
        ExponentiatedPauliTerm({0: "X"}, -0.31),
        ExponentiatedPauliTerm({1: "Z"}, 0.13),
        ExponentiatedPauliTerm({0: "Y"}, 0.27),
        ExponentiatedPauliTerm({1: "X", 0: "Z"}, -0.19),
        ExponentiatedPauliTerm({}, 0.11),
        ExponentiatedPauliTerm({0: "I", 1: "I"}, -0.23),
    ]
    if empty:
        terms = []
    circuit = _map_sparse_formula(terms, packed, repetitions, targets=targets, variant=variant)
    width = max(2, *targets) + 1
    assert len(json.loads(circuit.get_qsharp_circuit().json())["qubits"]) == width
    paulis = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]),
    }
    step = np.eye(2**width, dtype=complex)
    for term in terms:
        factors = {targets[index]: paulis[pauli] for index, pauli in term.pauli_term.items()}
        factors[2] = np.diag([0, 1])  # exp(-i angle |1><1|_control tensor P).
        generator = np.ones((1, 1), dtype=complex)
        for qubit in reversed(range(width)):
            generator = np.kron(generator, factors.get(qubit, paulis["I"]))
        step = scipy.linalg.expm(-1j * term.angle * generator) @ step
    expected = np.linalg.matrix_power(step, repetitions)
    actual = Operator(circuit.get_qiskit_circuit()).data
    # QIR elides circuit-global phase. Fix it using the control-off amplitude;
    # the observable phase between control branches must still match exactly.
    actual /= actual[0, 0]
    np.testing.assert_allclose(
        actual,
        expected,
        atol=float_comparison_absolute_tolerance,
        rtol=float_comparison_relative_tolerance,
    )


@pytest.mark.parametrize("packed", [False, True], ids=["legacy", "packed"])
@pytest.mark.parametrize("variant", ["pauli_sequence", "batched_pauli_sequence"])
def test_wide_controlled_transport_stays_sparse(packed: bool, variant: str) -> None:
    """Transport support-sized lists and scalar repetitions, never expanded evolution."""
    num_qubits = 40_000
    terms = [
        ExponentiatedPauliTerm({num_qubits - 1: "Y", 0: "X"}, -0.31),
        ExponentiatedPauliTerm({num_qubits // 2: "Z"}, 0.27),
        ExponentiatedPauliTerm({}, -0.19),
    ]
    circuit = _map_sparse_formula(terms, packed, 1_000_000, num_qubits, variant=variant)
    assert circuit._qsharp_factory is not None
    payload = circuit._qsharp_factory.parameter
    assert payload == {
        "termOffsets": [0, 2, 3, 3],
        "qubitIndices": [39_999, 0, 20_000],
        "paulis": [qsharp.Pauli.Y, qsharp.Pauli.X, qsharp.Pauli.Z],
        "pauliCoefficients": [-0.31, 0.27, -0.19],
        "repetitions": 1_000_000,
        "control": num_qubits,
        "systems": list(range(num_qubits)),
        **({"batchOffsets": [0, 2, 3]} if variant == "batched_pauli_sequence" else {}),
    }
    assert isinstance(payload["repetitions"], int)


@pytest.mark.parametrize("variant", ["pauli_sequence", "batched_pauli_sequence"])
def test_packed_controlled_transport_preserves_duplicate_dict_semantics(variant: str) -> None:
    """Repeated packed indices keep the last Pauli, as the lazy term dictionary does."""
    container = PauliProductFormulaContainer.from_sparse_arrays(
        np.array([0, 3]),
        np.array([1, 0, 1]),
        np.array([1, 3, 2]),
        np.array([-0.2]),
        step_reps=1,
        num_qubits=2,
    )
    mapper = create("controlled_circuit_mapper", variant)
    mapper.settings().set("control_indices", [2])
    circuit = mapper.run(UnitaryRepresentation(container=container))
    assert circuit._qsharp_factory is not None
    payload = circuit._qsharp_factory.parameter
    assert payload["termOffsets"] == [0, 2]
    assert payload["qubitIndices"] == [1, 0]
    assert payload["paulis"] == [qsharp.Pauli.Y, qsharp.Pauli.Z]


def test_batched_variant_reduces_rotation_depth_without_changing_default() -> None:
    """Batching is opt-in and reduces rotation rounds without adding qubits or rotations."""
    assert create("controlled_circuit_mapper").name() == "pauli_sequence"
    assert create("controlled_circuit_mapper", "batched_pauli_sequence").name() == "batched_pauli_sequence"
    terms = [ExponentiatedPauliTerm({2 * i: "X", 2 * i + 1: "Y"}, 0.123) for i in range(6)]
    counts = []
    for variant in ("pauli_sequence", "batched_pauli_sequence"):
        circuit = _map_sparse_formula(terms, True, 2, num_qubits=12, variant=variant)
        application = circuit.get_qre_application()
        counts.append(dict(get_qsharp_context().logical_counts(application.entry_expr, *application.args)))
    assert counts[0]["numQubits"] == counts[1]["numQubits"] == 13
    assert counts[0]["rotationCount"] == counts[1]["rotationCount"] == 24
    assert counts[1]["rotationDepth"] == 4
    assert counts[1]["rotationDepth"] < counts[0]["rotationDepth"]
