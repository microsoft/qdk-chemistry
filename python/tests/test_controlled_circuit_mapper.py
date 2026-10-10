"""Tests for the PauliSequenceMapper and its helper functions in QDK/Chemistry."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import itertools
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
    ControlledPauliSequenceMapperSettings,
)
from qdk_chemistry.data import LatticeGraph
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.model_hamiltonians import create_heisenberg_hamiltonian, create_ising_hamiltonian
from qdk_chemistry.utils.qsharp import get_qsharp_context

from .reference_tolerances import float_comparison_absolute_tolerance, float_comparison_relative_tolerance
from .test_helpers import (
    apply_controlled_operation,
    assert_states_match_up_to_global_phase,
    controlled_product_formula_state,
    random_sparse_state,
)

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
    repetitions: int,
    num_qubits: int = 2,
    targets: list[int] | None = None,
    *,
    layer_offsets: tuple[int, ...] | None = None,
) -> Circuit:
    """Map a formula without widening its support."""
    container = PauliProductFormulaContainer(
        terms, step_reps=repetitions, num_qubits=num_qubits, layer_offsets=layer_offsets
    )
    mapper = create("controlled_circuit_mapper", "pauli_sequence")
    mapper.settings().set("control_indices", [num_qubits])
    if targets is not None:
        mapper.settings().set("target_indices", targets)
    return mapper.run(UnitaryRepresentation(container=container))


@pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available.")
@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("repetitions", [1, 3])
@pytest.mark.parametrize(
    ("empty", "targets"),
    [(False, [0, 1]), (True, [0, 1]), (False, [4, 1])],
    ids=["mixed", "empty", "reordered-noncontiguous"],
)
def test_sparse_controlled_matrix(repetitions: int, empty: bool, targets: list[int], layered: bool) -> None:
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
    layer_offsets = ((0,) if empty else (0, 2, 3, 6)) if layered else None
    circuit = _map_sparse_formula(terms, repetitions, targets=targets, layer_offsets=layer_offsets)
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


@pytest.mark.parametrize("layered", [False, True])
def test_wide_controlled_transport_stays_sparse(layered: bool) -> None:
    """Transport support-sized lists and scalar repetitions, never expanded evolution."""
    num_qubits = 40_000
    terms = [
        ExponentiatedPauliTerm({num_qubits - 1: "Y", 0: "X"}, -0.31),
        ExponentiatedPauliTerm({num_qubits // 2: "Z"}, 0.27),
        ExponentiatedPauliTerm({}, -0.19),
    ]
    circuit = _map_sparse_formula(terms, 1_000_000, num_qubits, layer_offsets=(0, 2, 3) if layered else None)
    assert circuit._qsharp_factory is not None
    payload = circuit._qsharp_factory.parameter
    params = vars(payload["params"])
    assert params == {
        "pauliIndices": [[39_999, 0], [20_000], []],
        "pauliOps": [[qsharp.Pauli.Y, qsharp.Pauli.X], [qsharp.Pauli.Z], []],
        "pauliCoefficients": [-0.31, 0.27, -0.19],
        "repetitions": 1_000_000,
        "numPrefixTerms": 0,
        "numSuffixTerms": 0,
    }
    assert isinstance(params["repetitions"], int)
    assert payload["control"] == num_qubits
    assert payload["systems"] == list(range(num_qubits))
    assert payload["layerOffsets"] == ([0, 2, 3] if layered else [])


def test_declared_layers_reduce_rotation_depth_without_changing_default() -> None:
    """Declared layers reduce rotation rounds without adding qubits or rotations."""
    assert create("controlled_circuit_mapper").name() == "pauli_sequence"
    terms = [ExponentiatedPauliTerm({2 * i: "X", 2 * i + 1: "Y"}, 0.123) for i in range(6)]
    counts = []
    for layer_offsets in (None, (0, 6)):
        circuit = _map_sparse_formula(terms, 2, num_qubits=12, layer_offsets=layer_offsets)
        application = circuit.get_qre_application()
        counts.append(dict(get_qsharp_context().logical_counts(application.entry_expr, *application.args)))
    assert counts[0]["numQubits"] == counts[1]["numQubits"] == 13
    assert counts[0]["rotationCount"] == counts[1]["rotationCount"] == 24
    assert counts[1]["rotationDepth"] == 4
    assert counts[1]["rotationDepth"] < counts[0]["rotationDepth"]


def test_identity_terms_share_one_control_phase_per_layer() -> None:
    """Identity terms in one declared layer cost the same rotations and depth as their merged term."""
    system = [ExponentiatedPauliTerm({0: "X"}, 0.123), ExponentiatedPauliTerm({1: "Z"}, 0.321)]
    counts = []
    for identities in ([0.2, -0.05], [0.15]):
        terms = [system[0], *(ExponentiatedPauliTerm({}, angle) for angle in identities), system[1]]
        circuit = _map_sparse_formula(terms, 1, num_qubits=2, layer_offsets=(0, len(terms)))
        application = circuit.get_qre_application()
        logical = get_qsharp_context().logical_counts(application.entry_expr, *application.args)
        counts.append((logical["rotationCount"], logical["rotationDepth"]))
    assert counts[0] == counts[1]


#: Absolute tolerance on simulated amplitudes after a few hundred gates.
_STATE_TOLERANCE = 1e-10


def _trotter(hamiltonian, *, steps: int = 1) -> UnitaryRepresentation:
    """Return the first-order Trotter evolution of ``hamiltonian``, which declares its disjoint layers."""
    builder = create("hamiltonian_unitary_builder", "trotter")
    builder.settings().update({"order": 1, "num_divisions": steps, "time": 0.3 * steps})
    return builder.run(hamiltonian)


def _map_with_cap(unitary: UnitaryRepresentation, cap: int) -> Circuit:
    """Map ``unitary`` with control 0 and the given Hamming-weight phasing batch cap."""
    return create("controlled_circuit_mapper", "pauli_sequence", max_hamming_weight_phasing_batch_size=cap).run(unitary)


def _logical_counts(circuit: Circuit) -> dict[str, int]:
    """Return the logical resource counts of ``circuit``."""
    application = circuit.get_qre_application()
    return dict(get_qsharp_context().logical_counts(application.entry_expr, *application.args))


def _toffolis(counts: dict[str, int]) -> int:
    """Return the Toffoli-class gates of ``counts``, which only the Hamming-weight adder trees use here."""
    return counts.get("cczCount", 0) + counts.get("ccixCount", 0)


def _z_layers(angles: list[list[float]]) -> UnitaryRepresentation:
    """Return single-Z terms on fresh qubits, one declared layer per row of ``angles``."""
    terms, offsets, qubit = [], [0], 0
    for layer in angles:
        for angle in layer:
            terms.append(ExponentiatedPauliTerm({qubit: "Z"}, angle))
            qubit += 1
        offsets.append(len(terms))
    return UnitaryRepresentation(PauliProductFormulaContainer(terms, 1, qubit, layer_offsets=offsets))


class TestHammingWeightPhasing:
    """Equal-angle towers of a declared layer are phased through Hamming-weight registers."""

    def test_default_setting_and_payload(self) -> None:
        """The cap defaults to 1, leaving phasing off, and reaches the Q# factory and operation."""
        mapper = create("controlled_circuit_mapper", "pauli_sequence")
        assert isinstance(mapper.settings(), ControlledPauliSequenceMapperSettings)
        assert mapper.settings().get("max_hamming_weight_phasing_batch_size") == 1
        tower = _z_layers([[0.1] * 8])
        default = mapper.run(tower)
        assert default._qsharp_factory is not None
        assert default._qsharp_factory.parameter["maxBatchSize"] == 1
        assert _toffolis(_logical_counts(default)) == 0
        circuit = _map_with_cap(tower, 4)
        assert circuit._qsharp_factory is not None
        assert list(circuit._qsharp_factory.parameter) == [
            "params",
            "layerOffsets",
            "maxBatchSize",
            "control",
            "systems",
        ]
        assert circuit._qsharp_factory.parameter["maxBatchSize"] == 4

    @pytest.mark.parametrize("cap", [0, -2])
    def test_invalid_cap_raises(self, cap: int) -> None:
        """A cap must be -1 or positive."""
        with pytest.raises(ValueError, match="max_hamming_weight_phasing_batch_size must be -1 or a positive"):
            _map_with_cap(_z_layers([[0.1] * 8]), cap)

    @pytest.mark.parametrize(
        ("name", "hamiltonian", "support"),
        [
            # The 8-site field layer is an X tower; its 4-bond layers stay plain rotations.
            ("transverse-ising", create_ising_hamiltonian(LatticeGraph.chain(8, periodic=True), j=1.0, h=0.5), 512),
            # Each bond color of the 16-site ring is a tower of 8 ZZ terms.
            ("ising-bonds", create_ising_hamiltonian(LatticeGraph.chain(16, periodic=True), j=1.0, h=0.0), 16),
            # XX, YY and ZZ towers of 8, each mapped onto Z by its own basis change.
            (
                "heisenberg",
                create_heisenberg_hamiltonian(
                    LatticeGraph.chain(16, periodic=True), jx=1.0, jy=1.0, jz=0.7, hx=0.0, hy=0.0, hz=0.0
                ),
                2,
            ),
        ],
    )
    @pytest.mark.parametrize("cap", [-1, 1])
    def test_spin_chain_matches_the_controlled_trotter_product(self, name, hamiltonian, support, cap) -> None:
        """Phased and plain towers both apply exactly the controlled product of their rotations."""
        unitary = _trotter(hamiltonian, steps=2)
        container = unitary.get_container()
        assert any(b - a >= 8 for a, b in itertools.pairwise(container.layer_offsets)), name
        circuit = _map_with_cap(unitary, cap)
        state = random_sparse_state(container.num_qubits + 1, support, seed=len(name))
        assert_states_match_up_to_global_phase(
            apply_controlled_operation(circuit._qsharp_op, state),
            controlled_product_formula_state(container, state),
            _STATE_TOLERANCE,
        )
        counts = _logical_counts(circuit)
        assert (_toffolis(counts) > 0) == (cap == -1)

    def test_phasing_starts_at_eight_equal_angles(self) -> None:
        """Open Ising chains of 15 and 17 sites have bond layers of 7 and 8 equal-angle terms."""
        for sites, phased in ((15, False), (17, True)):
            unitary = _trotter(create_ising_hamiltonian(LatticeGraph.chain(sites), j=1.0, h=0.0))
            container = unitary.get_container()
            assert {b - a for a, b in itertools.pairwise(container.layer_offsets)} == {sites // 2}
            uncapped, plain = (_logical_counts(_map_with_cap(unitary, cap)) for cap in (-1, 1))
            assert _toffolis(plain) == 0
            if phased:
                assert _toffolis(uncapped) > 0
                assert uncapped["rotationCount"] < plain["rotationCount"]
            else:
                assert uncapped == plain

    def test_only_exactly_equal_angles_share_a_tower(self) -> None:
        """Interleaved angles are grouped by value, and nearly equal angles are not grouped at all."""
        tower, other = 0.2, -0.35
        interleaved = _z_layers([[tower, other] * 7 + [tower]])
        separated = _z_layers([[tower] * 8, [other] * 7])
        assert (
            _logical_counts(_map_with_cap(interleaved, -1))["rotationCount"]
            == (_logical_counts(_map_with_cap(separated, -1))["rotationCount"])
        )
        assert _toffolis(_logical_counts(_map_with_cap(interleaved, -1))) > 0
        nearly_equal = _z_layers([[tower + 1e-9 * k for k in range(8)]])
        assert _toffolis(_logical_counts(_map_with_cap(nearly_equal, -1))) == 0

    @pytest.mark.parametrize("cap", [-1, 12, 8, 7, 1])
    def test_batch_cap_splits_a_tower_exactly(self, cap: int) -> None:
        """A 16-term tower phased whole, as 12 + 4, as 8 + 8, or not at all applies the same unitary."""
        unitary = _z_layers([[0.37] * 16])
        circuit = _map_with_cap(unitary, cap)
        container = unitary.get_container()
        state = random_sparse_state(container.num_qubits + 1, 16, seed=cap + 2)
        assert_states_match_up_to_global_phase(
            apply_controlled_operation(circuit._qsharp_op, state),
            controlled_product_formula_state(container, state),
            _STATE_TOLERANCE,
        )
        counts = _logical_counts(circuit)
        assert (_toffolis(counts) > 0) == (cap == -1 or cap >= 8)

    def test_smaller_batches_use_fewer_qubits(self) -> None:
        """Splitting a tower releases each batch's adder scratch before the next allocates."""
        unitary = _z_layers([[0.37] * 32])
        whole, halves = (_logical_counts(_map_with_cap(unitary, cap)) for cap in (-1, 16))
        assert halves["numQubits"] < whole["numQubits"]
