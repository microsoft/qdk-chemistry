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
        assert evo_params["pauliIndices"] == [[0], [1]]
        assert [[str(p) for p in ops] for ops in evo_params["pauliOps"]] == [["Pauli.X"], ["Pauli.Z"]]
        # An unbatched container passes empty flag arrays rather than all-default ones,
        # so the Q# skips both the exemption and the batching walks entirely.
        assert evo_params["needsControl"] == []
        assert evo_params["batchIds"] == []

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

    def test_one_time_boundaries_surround_the_repeated_step(self):
        """A noncommuting prefix and suffix execute once, on the correct sides of the loop."""
        before = ExponentiatedPauliTerm(pauli_term={0: "X"}, angle=0.2)
        repeated = ExponentiatedPauliTerm(pauli_term={0: "Z"}, angle=0.3)
        after = ExponentiatedPauliTerm(pauli_term={0: "Y"}, angle=-0.4)
        container = PauliProductFormulaContainer(
            step_terms=[repeated],
            step_reps=3,
            num_qubits=1,
            before_repeated_terms=[before],
            after_repeated_terms=[after],
        )

        actual = dense_matrix(PauliSequenceMapper().run(UnitaryRepresentation(container))._qsharp_op, 1)
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        expected = (
            scipy.linalg.expm(-1j * after.angle * y)
            @ np.linalg.matrix_power(scipy.linalg.expm(-1j * repeated.angle * z), container.step_reps)
            @ scipy.linalg.expm(-1j * before.angle * x)
        )

        assert np.max(np.abs(actual - expected)) < _TOL


def _sparse_op(terms, *, repetitions=1):
    """Build sparse evolution using typed Q# parameters."""
    params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(
        pauliIndices=[term["qubits"] for term in terms],
        pauliOps=[[getattr(qsharp.Pauli, axis) for axis in term["axes"]] for term in terms],
        pauliCoefficients=[term["angle"] for term in terms],
        needsControl=[],
        batchIds=[],
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
        ("pauli_indices", "pauli_ops", "pauli_coefficients"),
        [
            pytest.param([[0]], [[qsharp.Pauli.X]], [], id="coefficients-shorter"),
            pytest.param([], [[qsharp.Pauli.X]], [0.1], id="indices-shorter"),
            pytest.param([[0]], [], [0.1], id="ops-shorter"),
        ],
    )
    def test_rejects_mismatched_term_array_lengths(
        self,
        pauli_indices: list[list[int]],
        pauli_ops: list[list[qsharp.Pauli]],
        pauli_coefficients: list[float],
    ):
        """Sparse evolution requires one index row, Pauli row, and coefficient per term."""
        params = QSHARP_UTILS.PauliExp.SparseRepPauliExpParams(
            pauliIndices=pauli_indices,
            pauliOps=pauli_ops,
            pauliCoefficients=pauli_coefficients,
            needsControl=[],
            batchIds=[],
            repetitions=1,
        )
        op = QSHARP_UTILS.PauliExp.MakeSparseRepPauliExpOp(params)

        with pytest.raises(QSharpError, match="must have the same length"):
            dump_operation_on_state(op, 1, [1.0, 0.0], context=get_qsharp_context())

    def test_sparse_encoding_applies_the_container_sign_convention(self):
        """A single Z rotation must realise exp(-i theta Z), not its conjugate."""
        angle = 0.37
        got = dense_matrix(_sparse_op([{"qubits": [0], "axes": "Z", "angle": angle}]), 1)
        want = scipy.linalg.expm(-1j * angle * np.array([[1, 0], [0, -1]], dtype=complex))
        assert np.max(np.abs(got - want)) < _TOL
