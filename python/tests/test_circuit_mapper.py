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

    def test_sparse_encoding_carries_only_non_identity_positions(self, simple_unitary):
        """The Q# parameters must list Pauli positions, not one Pauli per system qubit."""
        circuit = PauliSequenceMapper().run(simple_unitary)
        evo_params = circuit._qsharp_factory.parameter["evo_params"]

        assert "pauliExponents" not in evo_params
        assert "batchIds" not in evo_params
        assert evo_params["termOffsets"] == [0, 1, 2]
        assert evo_params["qubitIndices"] == [0, 1]
        assert evo_params["pauliCodes"] == [1, 3]

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
        termOffsets=np.cumsum([0, *(len(term["qubits"]) for term in terms)]).tolist(),
        qubitIndices=[index for term in terms for index in term["qubits"]],
        pauliCodes=["IXYZ".index(axis) for term in terms for axis in term["axes"]],
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
    }

    @pytest.mark.parametrize("name", list(CASES))
    @pytest.mark.parametrize("repetitions", [1, 2])
    def test_sparse_matches_dense(self, name, repetitions):
        """Switching to the sparse encoding must not change the unitary."""
        num_qubits, terms = self.CASES[name]
        got = dense_matrix(_sparse_op(terms, repetitions=repetitions), num_qubits)
        want = dense_matrix(_dense_op(terms, num_qubits, repetitions=repetitions), num_qubits)
        assert np.max(np.abs(got - want)) < _TOL

    @pytest.mark.parametrize(
        ("offsets", "indices", "codes", "coefficients"),
        [
            pytest.param([0, 1], [0], [1], [], id="coefficients-shorter"),
            pytest.param([0, 1], [], [1], [0.1], id="indices-shorter"),
            pytest.param([0, 1], [0], [], [0.1], id="codes-shorter"),
        ],
    )
    def test_rejects_mismatched_term_array_lengths(self, offsets, indices, codes, coefficients):
        """Sparse evolution requires consistent factor arrays and term boundaries."""
        params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(
            termOffsets=offsets,
            qubitIndices=indices,
            pauliCodes=codes,
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
def packed_mapper(request):
    """Exercise both production mappers with the same packed cases."""
    return request.param()


@pytest.fixture(scope="module")
def packed_controlled_on_register():
    """Adapt the production single-control callable to the matrix helper."""
    context = get_qsharp_context()
    context.eval(
        """
        function PackedMapperOnRegister(op : ((Qubit, Qubit[]) => Unit is Adj + Ctl))
            : (Qubit[] => Unit is Adj + Ctl) {
            qs => op(qs[0], qs[1...])
        }
        """
    )
    return context.code.PackedMapperOnRegister


class TestPackedPauliMappers:
    """Packed mapping preserves the unitary and stays compact in both control modes."""

    def test_resource_estimates_include_all_repetitions(self, packed_mapper):
        """Resource estimates count symbolic repetitions without adding qubits."""
        counts = []
        for repetitions in (1, 17):
            container = PauliProductFormulaContainer.from_sparse_arrays(
                [0, 2], [0, 1], [1, 3], [0.137], step_reps=repetitions, num_qubits=2
            )
            circuit = packed_mapper.run(UnitaryRepresentation(container))
            counts.append(circuit.estimate()["logicalCounts"])
        assert counts[0]["rotationCount"] > 0
        assert counts[1]["rotationCount"] == 17 * counts[0]["rotationCount"]
        assert counts[1]["numQubits"] == counts[0]["numQubits"]

    def test_payload_stays_flat_and_repetitions_stay_symbolic(self, packed_mapper, monkeypatch):
        """Large registers and repetition counts must not expand term dictionaries or labels."""
        container = PauliProductFormulaContainer.from_sparse_arrays(
            [0, 1, 1, 3],
            [0, 1, 80_799],
            [2, 1, 3],
            [0.2, 0.4, -0.7],
            step_reps=1_000_000_000,
            num_qubits=80_800,
        )

        def no_term_objects(*_args):
            raise AssertionError("Packed mapping materialized term dictionaries")

        monkeypatch.setattr(type(container.step_terms), "__getitem__", no_term_objects)
        controlled = isinstance(packed_mapper, ControlledPauliSequenceMapper)
        if controlled:
            packed_mapper.settings().set("control_indices", [80_800])
            packed_mapper.settings().set("target_indices", list(reversed(range(80_800))))
        circuit = packed_mapper.run(UnitaryRepresentation(container))
        factory = circuit._qsharp_factory.parameter
        params = vars(factory["params"]) if controlled else factory["evo_params"]
        assert params == {
            "termOffsets": [0, 1, 1, 3],
            "qubitIndices": [0, 1, 80_799],
            "pauliCodes": [2, 1, 3],
            "pauliCoefficients": [0.2, 0.4, -0.7],
            "repetitions": 1_000_000_000,
        }
        if controlled:
            assert factory["control"] == 80_800
            assert factory["systems"] == list(reversed(range(80_800)))
        else:
            assert circuit.num_qubits == 80_800

    @pytest.mark.parametrize(
        ("offsets", "indices", "codes", "angles"),
        [
            pytest.param([0, 1, 3, 3, 4], [0, 0, 1, 1], [1, 3, 2, 3], [0.2, -0.45, 0.7, 0.1], id="mixed"),
            pytest.param([0, 0], [], [], [0.7], id="identity"),
            pytest.param([0], [], [], [], id="empty"),
        ],
    )
    def test_packed_matches_legacy_including_global_phase(
        self, offsets, indices, codes, angles, packed_mapper, packed_controlled_on_register
    ):
        """Preserve mixed terms, empty formulas, controlled identity phases, and QIR lowering."""
        container = PauliProductFormulaContainer.from_sparse_arrays(
            offsets, indices, codes, angles, step_reps=3, num_qubits=2
        )
        legacy = PauliProductFormulaContainer(list(container.step_terms), 3, 2)
        packed_circuit = packed_mapper.run(UnitaryRepresentation(container))
        legacy_circuit = PauliSequenceMapper().run(UnitaryRepresentation(legacy))
        expected = dense_matrix(legacy_circuit._qsharp_op, 2)
        if not indices:
            np.testing.assert_allclose(expected, np.exp(-3j * sum(angles)) * np.eye(4), atol=_TOL)
        if isinstance(packed_mapper, ControlledPauliSequenceMapper):
            actual = dense_matrix(packed_controlled_on_register(packed_circuit._qsharp_op), 3)
            expected = scipy.linalg.block_diag(np.eye(4), expected)
        else:
            actual = dense_matrix(packed_circuit._qsharp_op, 2)
        np.testing.assert_allclose(actual, expected, atol=_TOL)
        if indices:
            assert isinstance(packed_circuit.get_qsharp_circuit(), QdkCircuitType)
            assert "define" in str(packed_circuit.get_qir())

    def test_qsharp_rejects_malformed_packed_payload(self):
        """The Q# entry point validates offsets even when Python construction is bypassed."""
        params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(
            termOffsets=[0, 2, 1], qubitIndices=[0], pauliCodes=[1], pauliCoefficients=[0.2, 0.3], repetitions=1
        )
        op = QSHARP_UTILS.PauliExp.MakeSparseRepPauliExpOp(params)
        with pytest.raises(QSharpError, match="SparsePauliExp"):
            dump_operation_on_state(op, 1, [1.0, 0.0], context=get_qsharp_context())
