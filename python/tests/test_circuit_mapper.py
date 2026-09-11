"""Tests for the non-controlled PauliSequenceMapper in QDK/Chemistry."""

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
from qdk.test_utils import dump_operation_on_state

try:
    from qdk._native import Circuit as QdkCircuitType
    from qdk._native import QSharpError
except ImportError:
    from qsharp._native import Circuit as QdkCircuitType
    from qsharp._native import QSharpError


from qdk_chemistry.algorithms.circuit_mapper.pauli_sequence_mapper import (
    PauliSequenceMapper,
)
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

    def test_sparse_encoding_carries_only_non_identity_positions(self):
        """The Q# adapter preserves caller factor order, skips explicit identities, and retains empty rows."""
        word = {2: "I", 1: "Z", 0: "X"}
        container = PauliProductFormulaContainer(
            [
                ExponentiatedPauliTerm(word, 0.5),
                ExponentiatedPauliTerm({0: "I"}, 0.25),
                ExponentiatedPauliTerm(word, -0.2),
            ],
            2,
            3,
        )
        circuit = PauliSequenceMapper().run(UnitaryRepresentation(container))
        evo_params = circuit._qsharp_factory.parameter["evo_params"]

        assert "pauliExponents" not in evo_params
        assert "batchIds" not in evo_params
        assert evo_params["pauliIndices"] == [[1, 0], [], [1, 0]]
        assert evo_params["pauliOps"] == [[qsharp.Pauli.Z, qsharp.Pauli.X], [], [qsharp.Pauli.Z, qsharp.Pauli.X]]
        assert evo_params["pauliCoefficients"] == [0.5, 0.25, -0.2]

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


def _sparse_op(terms, *, repetitions=1):
    """Build sparse evolution using typed Q# parameters."""
    params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(
        pauliIndices=[term["qubits"] for term in terms],
        pauliOps=[[getattr(qsharp.Pauli, axis) for axis in term["axes"]] for term in terms],
        pauliCoefficients=[term["angle"] for term in terms],
        repetitions=repetitions,
    )
    return QSHARP_UTILS.PauliExp.MakeSparseRepPauliExpOp(params)


def _dense_op(terms, num_qubits, *, repetitions=1):
    """Build dense evolution using typed Q# parameters."""
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
    return QSHARP_UTILS.PauliExp.MakeRepPauliExpOp(params)


class TestSparseUncontrolledEvolution:
    """The sparse dispatch must be indistinguishable from the dense one it replaces."""

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
        """Switching to the sparse encoding must not change the unitary."""
        num_qubits, terms = self.CASES[name]
        got = dense_matrix(_sparse_op(terms, repetitions=repetitions), num_qubits)
        want = dense_matrix(_dense_op(terms, num_qubits, repetitions=repetitions), num_qubits)
        if not any(term["qubits"] for term in terms):
            np.testing.assert_allclose(
                want, np.exp(-1j * repetitions * sum(t["angle"] for t in terms)) * np.eye(4), atol=_TOL
            )
        assert np.max(np.abs(got - want)) < _TOL

    @pytest.mark.parametrize(
        ("indices", "ops", "coefficients"),
        [
            pytest.param([[0]], [[qsharp.Pauli.X]], [], id="coefficients-shorter"),
            pytest.param([], [[qsharp.Pauli.X]], [0.1], id="indices-shorter"),
            pytest.param([[0]], [], [0.1], id="ops-shorter"),
        ],
    )
    def test_rejects_mismatched_term_array_lengths(self, indices, ops, coefficients):
        """Sparse evolution requires one support, axis row and angle per term."""
        params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(
            pauliIndices=indices,
            pauliOps=ops,
            pauliCoefficients=coefficients,
            repetitions=1,
        )
        op = QSHARP_UTILS.PauliExp.MakeSparseRepPauliExpOp(params)

        with pytest.raises(QSharpError, match="inconsistent array lengths"):
            dump_operation_on_state(op, 1, [1.0, 0.0], context=get_qsharp_context())

    def test_sparse_encoding_applies_the_container_sign_convention(self):
        """A single Z rotation must realise exp(-i theta Z), not its conjugate."""
        angle = 0.37
        got = dense_matrix(_sparse_op([{"qubits": [0], "axes": "Z", "angle": angle}]), 1)
        want = scipy.linalg.expm(-1j * angle * np.array([[1, 0], [0, -1]], dtype=complex))
        assert np.max(np.abs(got - want)) < _TOL


@pytest.fixture(params=[PauliSequenceMapper, ControlledPauliSequenceMapper], ids=["regular", "controlled"])
def sparse_mapper(request):
    """Exercise both production mappers with the same sparse cases."""
    return request.param()


class TestSparsePauliMappers:
    """The existing sparse payload stays independent of register width and repetitions."""

    def test_resource_estimates_include_all_repetitions(self, sparse_mapper):
        """Resource estimates count symbolic repetitions without adding qubits."""
        counts = []
        for repetitions in (1, 17):
            container = PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X", 1: "Z"}, 0.137)], repetitions, 2)
            circuit = sparse_mapper.run(UnitaryRepresentation(container))
            counts.append(circuit.estimate()["logicalCounts"])
        assert counts[0]["rotationCount"] > 0
        assert counts[1]["rotationCount"] == 17 * counts[0]["rotationCount"]
        assert counts[1]["numQubits"] == counts[0]["numQubits"]
        assert isinstance(circuit.get_qsharp_circuit(), QdkCircuitType)
        assert "define" in str(circuit.get_qir())

    def test_payload_stays_sparse_and_repetitions_stay_symbolic(self, sparse_mapper):
        """Large registers retain only non-identity support and symbolic repetitions."""
        container = PauliProductFormulaContainer(
            [
                ExponentiatedPauliTerm({0: "Y"}, 0.2),
                ExponentiatedPauliTerm({}, 0.4),
                ExponentiatedPauliTerm({1: "X", 80_799: "Z"}, -0.7),
            ],
            1_000_000_000,
            80_800,
        )
        circuit = sparse_mapper.run(UnitaryRepresentation(container))
        factory = circuit._qsharp_factory.parameter
        params = (
            vars(factory["params"])
            if isinstance(sparse_mapper, ControlledPauliSequenceMapper)
            else factory["evo_params"]
        )
        assert params == {
            "pauliIndices": [[0], [], [1, 80_799]],
            "pauliOps": [[qsharp.Pauli.Y], [], [qsharp.Pauli.X, qsharp.Pauli.Z]],
            "pauliCoefficients": [0.2, 0.4, -0.7],
            "repetitions": 1_000_000_000,
        }
