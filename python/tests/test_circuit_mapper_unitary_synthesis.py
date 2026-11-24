"""Tests for the unitary synthesis circuit mappers and their Q# building blocks."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from unittest.mock import Mock

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state
from scipy import sparse
from scipy.stats import ortho_group

from qdk_chemistry._core.utils.unitary_synthesis import block_sparse_isometry_synthesis, dense_isometry_right_factor
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.circuit_mapper import unitary_synthesis_mapper
from qdk_chemistry.data import UnitaryMatrixContainer, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, use_qsharp_context

from .helpers import apply_operation, assert_states_match_up_to_global_phase

_DENSE, _BLOCK_SPARSE = "unitary_synthesis_dense", "unitary_synthesis_block_sparse"
# At the default 10 rotation bits every quantized angle is within 2^-10 of its target.
_INFIDELITY_TOLERANCE = 1e-3


def _bit_reversal(num_qubits: int) -> np.ndarray:
    """Return the permutation that maps little-endian basis indices to dump (big-endian) indices."""
    indices = np.arange(1 << num_qubits)
    reversed_indices = np.zeros_like(indices)
    for bit in range(num_qubits):
        reversed_indices |= ((indices >> bit) & 1) << (num_qubits - 1 - bit)
    return reversed_indices


def _representation(matrix) -> UnitaryRepresentation:
    """Wrap a matrix in a unitary representation."""
    return UnitaryRepresentation(UnitaryMatrixContainer(matrix))


def _block_sparse(block_sizes: list[int], seed: int) -> sparse.csc_array:
    """Return a row- and column-permuted block-diagonal orthogonal matrix."""
    rng = np.random.default_rng(seed)
    blocks = [
        sparse.coo_array(ortho_group.rvs(size, random_state=rng) if size > 1 else [[-1.0]]) for size in block_sizes
    ]
    matrix = sparse.block_diag(blocks, format="csc")
    dimension = matrix.shape[0]
    return sparse.csc_array(matrix[rng.permutation(dimension)][:, rng.permutation(dimension)])


def _assert_realizes(circuit, matrix, seed: int, preimage=None) -> None:
    """Assert that ``circuit`` maps ``preimage @ a`` to ``matrix @ a`` up to a global phase for a random unit ``a``."""
    num_qubits = matrix.shape[0].bit_length() - 1
    coefficients = np.random.default_rng(seed).normal(size=matrix.shape[1])
    coefficients /= np.linalg.norm(coefficients)
    circuit_input = coefficients if preimage is None else preimage @ coefficients
    op = circuit._qsharp_op
    num_ancillas = circuit.metadata.num_phase_gradient_ancillas
    if num_ancillas:
        op = QSHARP_UTILS.CircuitComposition.MakeSharedAncillaOp(
            op, QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState, num_ancillas
        )
    assert circuit.num_qubits == num_qubits + num_ancillas
    reverse = _bit_reversal(num_qubits)
    initial = np.zeros(1 << num_qubits)
    initial[reverse[: len(circuit_input)]] = circuit_input
    ancilla_zero = np.eye(1, 1 << num_ancillas).ravel()
    output = apply_operation(op, np.kron(initial, ancilla_zero)).reshape(1 << num_qubits, 1 << num_ancillas)
    np.testing.assert_allclose(output[:, 1:], 0.0, atol=1e-8, err_msg="phase gradient was not restored")
    # Measurement-based unlookups leave a random global sign.
    infidelity = 1.0 - abs(np.vdot(matrix @ coefficients, output[reverse, 0])) ** 2
    assert infidelity < _INFIDELITY_TOLERANCE, f"infidelity {infidelity:.2e}"


@pytest.mark.parametrize(
    ("name", "matrix"),
    [
        *[
            (_DENSE, ortho_group.rvs(1 << n, random_state=10 * n + m)[:, :m])
            for n, m in [(1, 1), (2, 1), (2, 3), (3, 2), (3, 5), (4, 4), (4, 16)]
        ],
        (_DENSE, _block_sparse([4, 2, 1, 1], seed=4)[:, :3]),
        *[
            (_BLOCK_SPARSE, _block_sparse(sizes, seed=len(sizes) + m)[:, :m])
            for sizes, m in [([2, 4, 1, 1], 8), ([2, 4, 1, 1], 3), ([1] * 8, 8), ([4, 4, 3, 2, 1, 1, 1], 5), ([2], 1)]
        ],
        (_BLOCK_SPARSE, _block_sparse([4, 4, 3, 2, 1, 1, 1], seed=16).toarray()),
        (_BLOCK_SPARSE, ortho_group.rvs(4, random_state=8)),
    ],
)
def test_realizes_matrix(name, matrix):
    """Isometries and unitaries in dense or sparse storage are realized on the leading columns."""
    _assert_realizes(create("circuit_mapper", name).run(_representation(matrix)), matrix, seed=matrix.shape[1])


@pytest.mark.parametrize(
    ("name", "num_input_qubits"), [(_DENSE, 2), (_DENSE, 4), (_BLOCK_SPARSE, 1), (_BLOCK_SPARSE, 3)]
)
def test_explicit_input_register(name, num_input_qubits):
    """Any valid input width realizes the matrix, and the block-sparse mapper may add more than two qubits."""
    matrix = _block_sparse([4, 4, 3, 2, 1, 1, 1], seed=6)[:, :2]
    circuit = create("circuit_mapper", name, num_input_qubits=num_input_qubits).run(_representation(matrix))
    assert circuit._qsharp_factory.parameter["numAddedQubits"] == 4 - num_input_qubits
    _assert_realizes(circuit, matrix, seed=num_input_qubits)


@pytest.mark.parametrize("name", [_DENSE, _BLOCK_SPARSE])
def test_shared_phase_gradient(name):
    """Without its own gradient the operation acts on [target | gradient] and restores the gradient."""
    matrix = _block_sparse([2, 4, 1, 1], seed=11)[:, :2]
    mapper = create("circuit_mapper", name, allocate_phase_gradient=False, rotation_bit_precision=8)
    circuit = mapper.run(_representation(matrix))
    assert circuit.metadata.num_phase_gradient_ancillas == 8
    _assert_realizes(circuit, matrix, seed=11)


@pytest.mark.parametrize(
    "matrix",
    [
        ortho_group.rvs(4, random_state=4)[:, :2],
        ortho_group.rvs(8, random_state=12)[:, :4],
        ortho_group.rvs(16, random_state=12)[:, :3],
        np.array([[-0.6], [0.8]]),
        np.array([[-0.5], [0.5], [0.5], [0.5]]),
    ],
)
def test_omit_right_factor(matrix):
    """The dense circuit realizes ``M``, and without ``V`` it maps ``V a`` to ``M a``."""
    representation = _representation(matrix)
    right_factor = create("circuit_mapper", _DENSE).right_factor(representation)
    np.testing.assert_allclose(right_factor.T @ right_factor, np.eye(matrix.shape[1]), atol=1e-12)
    _assert_realizes(create("circuit_mapper", _DENSE).run(representation), matrix, seed=0)
    mapper = create("circuit_mapper", _DENSE, omit_right_factor=True)
    assert mapper.synthesize(representation)["rightFactorGivens"] == []
    _assert_realizes(mapper.run(representation), matrix, seed=1, preimage=right_factor)


def test_right_factor_reuses_last_run(monkeypatch):
    """After ``run`` the dense mapper returns a copy of the synthesized ``V`` without factorizing again."""
    spy = Mock(wraps=dense_isometry_right_factor)
    monkeypatch.setattr(unitary_synthesis_mapper, "dense_isometry_right_factor", spy)
    matrix, other = (ortho_group.rvs(16, random_state=seed)[:, :4] for seed in (7, 8))
    mapper = create("circuit_mapper", _DENSE, omit_right_factor=True)
    mapper.run(_representation(matrix))
    mapper.right_factor(_representation(matrix.copy()))[:] = 0.0
    np.testing.assert_array_equal(mapper.right_factor(_representation(matrix)), dense_isometry_right_factor(matrix, 4))
    spy.assert_not_called()
    np.testing.assert_array_equal(mapper.right_factor(_representation(other)), dense_isometry_right_factor(other, 4))
    spy.assert_called_once()


def test_signed_permutation_needs_no_rotations():
    """One-by-one blocks leave only the permutations and the signs."""
    synthesis = block_sparse_isometry_synthesis(sparse.csc_array(np.diag([1.0, -1.0, 1.0, -1.0])[[2, 0, 3, 1]]))
    assert synthesis.block_givens.layer_angles == []
    assert len(synthesis.column_permutation) == len(synthesis.row_permutation) == 4


@pytest.mark.parametrize("duplicates", ["split", "cancelling"])
def test_duplicate_entries_are_summed(duplicates):
    """Duplicates sum as in SciPy: ``2v`` and ``-v`` give ``v``, and ``+1``, ``-1`` pairs couple no blocks."""
    matrix = _block_sparse([2, 4, 1, 1], seed=4)
    entries = matrix.tocoo()
    rows, columns, data = entries.row, entries.col, entries.data
    if duplicates == "split":
        rows, columns, data = np.tile(rows, 2), np.tile(columns, 2), np.concatenate([2.0 * data, -data])
    else:
        lone = np.flatnonzero(np.diff(matrix.indptr) == 1)[0]
        pairs = np.repeat(np.delete(np.arange(matrix.shape[1]), lone), 2)
        rows = np.concatenate([rows, np.full_like(pairs, matrix.indices[matrix.indptr[lone]])])
        columns = np.concatenate([columns, pairs])
        data = np.concatenate([data, np.tile([1.0, -1.0], matrix.shape[1] - 1)])
    order = np.lexsort((rows, columns))
    indptr = np.searchsorted(columns[order], np.arange(matrix.shape[1] + 1))
    stored = sparse.csc_array((data[order], rows[order], indptr), shape=matrix.shape)
    assert not stored.has_canonical_format
    expected, actual = block_sparse_isometry_synthesis(matrix), block_sparse_isometry_synthesis(stored)
    assert actual.row_permutation == expected.row_permutation
    assert actual.column_permutation == expected.column_permutation
    assert actual.block_givens.layer_angles == expected.block_givens.layer_angles
    assert actual.block_givens.phases == expected.block_givens.phases


@pytest.mark.parametrize(
    "call",
    [
        lambda unitary: create("circuit_mapper", _DENSE).run(unitary),
        lambda unitary: create("circuit_mapper", _DENSE).right_factor(unitary),
        lambda unitary: create("circuit_mapper", _BLOCK_SPARSE).run(unitary),
    ],
    ids=["dense-run", "dense-right-factor", "block-sparse-run"],
)
def test_rejects_unsupported_containers(call):
    """Only real unitary matrix containers can be synthesized."""
    with pytest.raises(ValueError, match="unitary_matrix"):
        call(UnitaryRepresentation(PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, 0.1)], 1, 1)))
    with pytest.raises(ValueError, match="requires a real matrix"):
        call(_representation(np.diag([1.0, 1j])))


@pytest.mark.parametrize(
    ("name", "num_qubits", "num_columns", "num_input_qubits"),
    [(_DENSE, 3, 5, 2), (_DENSE, 4, 2, 1), (_DENSE, 2, 2, 3), (_BLOCK_SPARSE, 3, 5, 2), (_BLOCK_SPARSE, 2, 2, 3)],
)
def test_rejects_invalid_input_register(name, num_qubits, num_columns, num_input_qubits):
    """The input register must hold the columns and fit the target, and the dense mapper adds at most two qubits."""
    mapper = create("circuit_mapper", name, num_input_qubits=num_input_qubits)
    with pytest.raises(ValueError, match="num_input_qubits"):
        mapper.run(_representation(np.eye(1 << num_qubits)[:, :num_columns]))


def _ry_product_state(angles: list[float]) -> np.ndarray:
    """Return the big-endian state of Ry(angle) on each qubit, first qubit most significant."""
    state = np.ones(1)
    for angle in angles:
        state = np.kron(state, [np.cos(angle / 2), np.sin(angle / 2)])
    return state


class TestGivensNetwork:
    """``ApplyRealUnitaryViaGivens`` applies rotations, signs and controls."""

    # 6 rotation bits resolve each angle to O(2^-6), bounding the infidelity at O(2^-12).
    _ROTATION_BITS = 6

    def _run(
        self, qsharp_test_context, qsharp_test_utils, givens, prep_angles, controls, inverse=False, bits=_ROTATION_BITS
    ):
        op = qsharp_test_utils.UnitarySynthesisTests.TestMakeGivensOp(
            givens, prep_angles, list(controls), bits, inverse
        )
        num_controls, num_target = len(controls), len(prep_angles)
        state = np.array(dump_operation_on_state(op, num_controls + num_target, context=qsharp_test_context))
        control_value = sum(1 << (num_controls - 1 - k) for k, bit in enumerate(controls) if bit)
        state = state.reshape(1 << num_controls, 1 << num_target)
        np.testing.assert_allclose(np.delete(state, control_value, axis=0), 0.0, atol=1e-10)
        return state[control_value]

    @pytest.mark.parametrize(
        ("givens", "prep_angles", "controls", "inverse", "bits"),
        [
            (
                {"layerAngles": [[0.4], [0.9]], "layerShifted": [False, True], "phases": [True, False, True, False]},
                [1.1, 0.7],
                (),
                False,
                _ROTATION_BITS,
            ),
            (
                {"layerAngles": [[0.6]], "layerShifted": [True], "phases": [False] * 4},
                [0.8, 1.3],
                (True,),
                False,
                _ROTATION_BITS,
            ),
            (
                {"layerAngles": [[0.6]], "layerShifted": [True], "phases": [False] * 4},
                [0.8, 1.3],
                (False,),
                False,
                _ROTATION_BITS,
            ),
            (
                {
                    "layerAngles": [[0.4, 0.2], [0.9], [0.3, 0.7], [0.5]],
                    "layerShifted": [False, True, False, True],
                    "phases": [True, False, False, True, False, True, True, False],
                },
                [1.1, 0.7, 0.4],
                (),
                True,
                _ROTATION_BITS,
            ),
            # 5-bit angles are exact, and 32-word layers load them through a swap network.
            (
                {
                    "layerAngles": [
                        (np.pi / 16 * ((5 * np.arange(32) + 3) % 32)).tolist(),
                        (np.pi / 16 * ((7 * np.arange(31) + 1) % 32)).tolist(),
                    ],
                    "layerShifted": [False, True],
                    "phases": (np.arange(64) % 3 == 0).tolist(),
                },
                [1.1, 0.7, 0.4, 0.9, 1.3, 0.2],
                (),
                False,
                5,
            ),
        ],
        ids=["signed-two-layer", "shifted-control-on", "shifted-control-off", "adjoint", "swap-network"],
    )
    def test_matches_reconstruction(
        self, qsharp_test_context, qsharp_test_utils, givens, prep_angles, controls, inverse, bits
    ):
        """The network, or its adjoint, matches its dense reconstruction exactly when every control is set."""
        state = self._run(qsharp_test_context, qsharp_test_utils, givens, prep_angles, controls, inverse, bits)
        # Reconstruct D L_{m-1} ... L_0, most significant qubit first.
        unitary = np.eye(len(givens["phases"]))
        for angles, shifted in zip(givens["layerAngles"], givens["layerShifted"], strict=True):
            upper = int(shifted) + 2 * np.arange(len(angles))
            cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
            first, second = unitary[upper], unitary[upper + 1]
            unitary[upper], unitary[upper + 1] = cosine * first - sine * second, sine * first + cosine * second
        unitary *= np.where(givens["phases"], -1.0, 1.0)[:, None]
        initial = _ry_product_state(prep_angles)
        expected = (unitary.T if inverse else unitary) @ initial if all(controls) else initial
        assert abs(np.vdot(expected, state)) ** 2 > 1 - 1e-3

    @pytest.mark.parametrize(
        ("prep_angles", "controls"),
        [
            ([0.8, 1.3], ()),
            ([0.8, 1.3], (True,)),
            ([0.8], (False,)),
            ([0.8], (True,) * 20),
            ([0.8], (True,) * 19 + (False,)),
        ],
        ids=["uncontrolled", "control-on", "control-off", "20-controls-on", "last-of-20-off"],
    )
    def test_global_sign(self, qsharp_test_context, qsharp_test_utils, prep_angles, controls):
        """``D = -I`` negates every amplitude, with the exact global phase, whenever every control is set."""
        givens = {"layerAngles": [], "layerShifted": [], "phases": [True] * (1 << len(prep_angles))}
        state = self._run(qsharp_test_context, qsharp_test_utils, givens, prep_angles, controls)
        initial = _ry_product_state(prep_angles)
        np.testing.assert_allclose(state, -initial if all(controls) else initial, atol=1e-10)

    @pytest.mark.parametrize(("num_register", "num_controls"), [(0, 1), (2, 3), (5, 0), (5, 1), (4, 3)])
    def test_controlled_phase_polynomial(self, qsharp_test_context, qsharp_test_utils, num_register, num_controls):
        """Controlled and padded ``ApplyPhasePolynomial`` flip the signs of ``AND(controls) * f(x)`` exactly."""
        tests = qsharp_test_utils.UnitarySynthesisTests
        rng = np.random.default_rng(10 * num_register + num_controls)
        phases = [True, *rng.integers(0, 2, (1 << num_register) - 1).astype(bool).tolist()]
        num_qubits = num_register + num_controls
        initial = rng.normal(size=1 << num_qubits)
        initial /= np.linalg.norm(initial)

        # Big-endian over [register | controls], with the register little-endian.
        index = np.arange(1 << num_qubits)
        register = sum(((index >> (num_qubits - 1 - q)) & 1) << q for q in range(num_register))
        controls_set = (index & ((1 << num_controls) - 1)) == (1 << num_controls) - 1
        expected = np.where(controls_set & np.asarray(phases)[register], -initial, initial)
        for padded in (True, False):
            op = tests.TestMakeControlledPhasePolynomialOp(phases, num_register, padded)
            state = np.array(dump_operation_on_state(op, num_qubits, list(initial), context=qsharp_test_context))
            np.testing.assert_allclose(state, expected, atol=1e-10)


class TestMultiplexedRy:
    """``ApplyControlledMultiplexedRy`` loads short and full tables and erases them by measurement."""

    _BITS = 6
    _ADDRESS = 7

    @pytest.mark.parametrize(
        ("num_words", "num_controls", "max_swap_bits"),
        [(20, 0, -1), (20, 2, -1), (20, 0, 2), (100, 1, 1), (128, 2, 0)],
    )
    def test_matches_reference(self, qsharp_test_context, qsharp_test_utils, num_words, num_controls, max_swap_bits):
        """Each address below the table length rotates by its word; later addresses and unset controls do not."""
        rng = np.random.default_rng(num_words + 10 * num_controls)
        values = rng.integers(1, 1 << self._BITS, num_words)
        words = [[bool((value >> bit) & 1) for bit in range(self._BITS)] for value in values]
        num_qubits = num_controls + self._ADDRESS + 1
        initial = rng.normal(size=1 << num_qubits)
        initial /= np.linalg.norm(initial)
        op = qsharp_test_utils.UnitarySynthesisTests.TestMakeMultiplexedRyOp(
            words, max_swap_bits, num_controls, self._ADDRESS
        )
        with use_qsharp_context(qsharp_test_context):
            state = apply_operation(op, initial)

        # Big-endian over [controls | address | target], each register little-endian.
        expected = initial.reshape(-1, 2).copy()
        for index in range(1 << (num_qubits - 1)):
            bits = [(index >> (num_qubits - 2 - q)) & 1 for q in range(num_qubits - 1)]
            address = sum(bit << k for k, bit in enumerate(bits[num_controls:]))
            if all(bits[:num_controls]) and address < num_words:
                angle = 4 * np.pi * values[address] / (1 << self._BITS)
                cosine, sine = np.cos(angle / 2), np.sin(angle / 2)
                expected[index] = [[cosine, -sine], [sine, cosine]] @ expected[index]
        assert abs(np.vdot(expected.ravel(), state)) ** 2 == pytest.approx(1.0, abs=1e-10)


class TestAddConstant:
    """``AddConstant`` adds a signed constant modulo the register size."""

    @pytest.mark.parametrize("width", [1, 3])
    @pytest.mark.parametrize("constant", [0, 1, -5])
    @pytest.mark.parametrize("inverse", [False, True], ids=["forward", "adjoint"])
    def test_adds_constant_modulo_width(self, qsharp_test_utils, width, constant, inverse):
        """Every little-endian starting value maps to ``value ± constant`` modulo ``2**width``."""
        results = qsharp_test_utils.ArithmeticTests.TestAddConstantTable(constant, width, inverse)
        sign = -1 if inverse else 1
        assert list(results) == [(value + sign * constant) % (1 << width) for value in range(1 << width)]


class TestPermutationViaQROAM:
    """``PermutationViaQROAM`` permutes basis states coherently."""

    @pytest.mark.parametrize(
        ("num_bits", "max_swap_bits"), [(1, -1), (2, -1), (3, -1), (4, -1), (7, -1), (7, 1), (7, 0)]
    )
    @pytest.mark.parametrize("inverse", [False, True], ids=["forward", "adjoint"])
    def test_permutes_superposition(self, qsharp_test_context, qsharp_test_utils, num_bits, max_swap_bits, inverse):
        """The lookup, swap and measurement-based unlookup apply the permutation, or its inverse, exactly."""
        rng = np.random.default_rng(num_bits)
        reverse = _bit_reversal(num_bits)
        # Each repetition exercises the phase fix-up on new unlookup measurement outcomes.
        for _ in range(4):
            permutation = rng.permutation(1 << num_bits).tolist()
            angles = rng.uniform(0.2, np.pi - 0.2, num_bits).tolist()
            amplitudes = _ry_product_state(angles[::-1])
            expected = np.zeros_like(amplitudes)
            if inverse:
                expected = amplitudes[permutation]
            else:
                expected[permutation] = amplitudes

            op = qsharp_test_utils.IsometrySynthesisTests.TestMakePermutationOp(
                permutation, angles, max_swap_bits, inverse
            )
            state = np.array(dump_operation_on_state(op, num_bits, context=qsharp_test_context))[reverse]
            assert_states_match_up_to_global_phase(state, expected, atol=1e-10)
