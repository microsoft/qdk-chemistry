"""Tests for the unitary synthesis circuit mapper and its Q# building blocks."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest
from qdk.test_utils import dump_operation_on_state
from scipy import sparse
from scipy.stats import ortho_group

from qdk_chemistry._core.utils.unitary_synthesis import (
    block_sparse_isometry_synthesis,
    dense_isometry_right_factor,
    dense_isometry_synthesis,
)
from qdk_chemistry.algorithms import available, create
from qdk_chemistry.algorithms.circuit_mapper import UnitarySynthesisMapper
from qdk_chemistry.data import RealUnitaryContainer, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .helpers import apply_operation, assert_states_match_up_to_global_phase

# At the default 10 rotation bits every quantized angle is within 2^-10 of its target; the
# observed infidelities stay below 1e-4 for these widths.
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
    return UnitaryRepresentation(RealUnitaryContainer(matrix))


def _block_sparse(block_sizes: list[int], seed: int) -> sparse.csc_array:
    """Return a row- and column-permuted block-diagonal orthogonal matrix."""
    rng = np.random.default_rng(seed)
    blocks = [
        sparse.coo_array(ortho_group.rvs(size, random_state=rng) if size > 1 else [[-1.0]]) for size in block_sizes
    ]
    matrix = sparse.block_diag(blocks, format="csc")
    dimension = matrix.shape[0]
    return sparse.csc_array(matrix[rng.permutation(dimension)][:, rng.permutation(dimension)])


def _run_on_columns(circuit, num_qubits: int, coefficients: np.ndarray) -> np.ndarray:
    r"""Return the little-endian state ``circuit`` produces from :math:`\sum_c` ``coefficients[c]`` :math:`|c\rangle`.

    A circuit that acts on a caller-owned phase gradient is wrapped so the gradient is prepared
    around it, and the gradient register must come back in :math:`|0\rangle`.
    """
    op = circuit._qsharp_op
    num_ancillas = circuit.metadata.num_phase_gradient_ancillas
    if num_ancillas:
        op = QSHARP_UTILS.CircuitComposition.MakeSharedAncillaOp(
            op, QSHARP_UTILS.PhaseGradient.PreparePhaseGradientState, num_ancillas
        )
    assert circuit.num_qubits == num_qubits + num_ancillas
    reverse = _bit_reversal(num_qubits)
    initial = np.zeros(1 << num_qubits)
    initial[reverse[: len(coefficients)]] = coefficients
    ancilla_zero = np.eye(1, 1 << num_ancillas).ravel()
    output = apply_operation(op, np.kron(initial, ancilla_zero)).reshape(1 << num_qubits, 1 << num_ancillas)
    np.testing.assert_allclose(output[:, 1:], 0.0, atol=1e-8, err_msg="phase gradient was not restored")
    return output[reverse, 0]


def _assert_realizes(circuit, matrix: np.ndarray, seed: int, preimage=None) -> None:
    """Assert that ``circuit`` maps a random combination of the leading columns through ``matrix``.

    The comparison ignores the global phase: the measurement-based unlookups of the angle and
    permutation tables leave a random global sign on each run. ``preimage`` maps the
    coefficients to the circuit input; it defaults to the identity.
    """
    num_qubits = matrix.shape[0].bit_length() - 1
    coefficients = np.random.default_rng(seed).normal(size=matrix.shape[1])
    coefficients /= np.linalg.norm(coefficients)
    circuit_input = coefficients if preimage is None else preimage @ coefficients
    actual = _run_on_columns(circuit, num_qubits, circuit_input)
    expected = matrix @ coefficients
    infidelity = 1.0 - abs(np.vdot(expected, actual)) ** 2
    assert infidelity < _INFIDELITY_TOLERANCE, f"infidelity {infidelity:.2e}"


class TestRegistration:
    """The mapper is registered under ``circuit_mapper``."""

    def test_create(self):
        """``create`` returns the mapper with its default settings."""
        mapper = create("circuit_mapper", "unitary_synthesis")
        assert isinstance(mapper, UnitarySynthesisMapper)
        assert mapper.name() == "unitary_synthesis"
        assert mapper.type_name() == "circuit_mapper"
        assert "unitary_synthesis" in available("circuit_mapper")
        settings = mapper.settings()
        assert settings.get("method") == "dense"
        assert settings.get("rotation_bit_precision") == 10
        assert settings.get("allocate_phase_gradient") is True
        assert settings.get("omit_right_factor") is False
        assert settings.get("num_input_qubits") == -1


class TestDense:
    """The cosine-sine decomposition circuit realizes the matrix on its leading columns."""

    @pytest.mark.parametrize(
        ("num_qubits", "num_columns"),
        [(1, 1), (1, 2), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 5), (3, 8), (4, 4), (4, 16)],
    )
    def test_fidelity(self, num_qubits, num_columns):
        """Random isometries and orthogonal matrices, with zero, one or two added qubits."""
        matrix = ortho_group.rvs(1 << num_qubits, random_state=10 * num_qubits + num_columns)[:, :num_columns]
        circuit = create("circuit_mapper", "unitary_synthesis").run(_representation(matrix))
        _assert_realizes(circuit, matrix, seed=num_columns)

    @pytest.mark.parametrize("num_input_qubits", [1, 2, 3])
    def test_explicit_input_register(self, num_input_qubits):
        """Any valid input register width realizes the same isometry."""
        matrix = ortho_group.rvs(8, random_state=3)[:, :2]
        mapper = create("circuit_mapper", "unitary_synthesis", num_input_qubits=num_input_qubits)
        circuit = mapper.run(_representation(matrix))
        assert circuit._qsharp_factory.parameter["numInputQubits"] == num_input_qubits
        assert circuit._qsharp_factory.parameter["numAddedQubits"] == 3 - num_input_qubits
        _assert_realizes(circuit, matrix, seed=num_input_qubits)

    @pytest.mark.parametrize("column", [[-0.6, 0.8], [-0.5, 0.5, 0.5, 0.5]], ids=["one-added", "two-added"])
    def test_negative_scalar_right_factor(self, column):
        """Without input qubits ``V`` is a 1 x 1 sign, applied as a global sign when it is -1."""
        matrix = np.array(column)[:, None]
        mapper = create("circuit_mapper", "unitary_synthesis")
        np.testing.assert_array_equal(mapper.right_factor(_representation(matrix)), [[-1.0]])
        _assert_realizes(mapper.run(_representation(matrix)), matrix, seed=0)

    def test_sparse_storage(self):
        """A sparse container is densified for the dense method."""
        matrix = _block_sparse([4, 2, 1, 1], seed=4)[:, :3]
        circuit = create("circuit_mapper", "unitary_synthesis").run(_representation(matrix))
        _assert_realizes(circuit, matrix.toarray(), seed=4)

    def test_shared_phase_gradient(self):
        """Without its own gradient the operation acts on [target | gradient] and leaves the gradient intact."""
        matrix = ortho_group.rvs(8, random_state=5)[:, :2]
        mapper = create("circuit_mapper", "unitary_synthesis", allocate_phase_gradient=False, rotation_bit_precision=8)
        circuit = mapper.run(_representation(matrix))
        assert circuit.num_qubits == 3 + 8
        assert circuit.metadata.num_phase_gradient_ancillas == 8
        _assert_realizes(circuit, matrix, seed=5)


class TestRightFactor:
    """``omit_right_factor`` drops exactly the ``V`` that the helper returns."""

    @pytest.mark.parametrize(("num_qubits", "num_columns"), [(2, 2), (3, 2), (3, 4), (4, 3)])
    def test_omitted_circuit_and_helper_reconstruct_matrix(self, num_qubits, num_columns):
        """The omitted circuit realizes ``M V^T``, so feeding it ``V a`` gives ``M a``."""
        matrix = ortho_group.rvs(1 << num_qubits, random_state=num_qubits * num_columns)[:, :num_columns]
        representation = _representation(matrix)
        mapper = create("circuit_mapper", "unitary_synthesis", omit_right_factor=True)
        right_factor = mapper.right_factor(representation)
        num_input_states = 1 << max((num_columns - 1).bit_length(), num_qubits - 2)

        assert right_factor.shape == (num_columns, num_columns)
        np.testing.assert_allclose(right_factor.T @ right_factor, np.eye(num_columns), atol=1e-12)
        np.testing.assert_array_equal(right_factor, dense_isometry_right_factor(matrix, num_input_states))
        np.testing.assert_array_equal(right_factor, dense_isometry_synthesis(matrix, num_input_states).right_factor)
        omitted = dense_isometry_synthesis(matrix, num_input_states, omit_right_factor=True)
        np.testing.assert_array_equal(omitted.right_factor, right_factor)
        assert omitted.right_factor_givens is None

        circuit = mapper.run(representation)
        assert circuit._qsharp_factory.parameter["synthesis"]["rightFactorGivens"] == []
        _assert_realizes(circuit, matrix, seed=num_columns, preimage=right_factor)

    def test_without_added_qubits_is_identity(self):
        """With the whole register as input there is no first decomposition and ``V = I``."""
        matrix = ortho_group.rvs(4, random_state=1)
        mapper = create("circuit_mapper", "unitary_synthesis", num_input_qubits=2)
        np.testing.assert_array_equal(mapper.right_factor(_representation(matrix)), np.eye(4))

    def test_omitting_saves_gates(self):
        """Leaving ``V`` out lowers the Toffoli count of the resource estimate."""
        representation = _representation(ortho_group.rvs(8, random_state=3)[:, :2])
        counts = [
            create("circuit_mapper", "unitary_synthesis", omit_right_factor=omit)
            .run(representation)
            .estimate()["logicalCounts"]["cczCount"]
            for omit in (False, True)
        ]
        assert counts[1] < counts[0]


class TestBlockSparse:
    """The permuted block-diagonal circuit realizes the matrix on its leading columns."""

    @pytest.mark.parametrize(
        ("block_sizes", "num_columns", "stored_sparse"),
        [
            ([2, 4, 1, 1], 8, True),
            ([2, 4, 1, 1], 3, True),
            ([2, 4, 1, 1], 8, False),
            ([1] * 8, 8, True),
            ([4, 4, 3, 2, 1, 1, 1], 16, True),
            ([4, 4, 3, 2, 1, 1, 1], 5, True),
            ([2], 1, True),
        ],
        ids=[
            "three-blocks",
            "isometry",
            "dense-storage",
            "signed-permutation",
            "sixteen",
            "sixteen-isometry",
            "one-qubit",
        ],
    )
    def test_fidelity(self, block_sizes, num_columns, stored_sparse):
        """Matrices with several coupled blocks, permuted rows and columns, and their leading columns."""
        matrix = _block_sparse(block_sizes, seed=len(block_sizes) + num_columns)[:, :num_columns]
        container_matrix = matrix if stored_sparse else matrix.toarray()
        circuit = create("circuit_mapper", "unitary_synthesis", method="block_sparse").run(
            _representation(container_matrix)
        )
        _assert_realizes(circuit, matrix.toarray(), seed=num_columns)

    def test_dense_matrix(self):
        """A matrix without sparsity is a single block."""
        matrix = ortho_group.rvs(4, random_state=8)
        circuit = create("circuit_mapper", "unitary_synthesis", method="block_sparse").run(_representation(matrix))
        _assert_realizes(circuit, matrix, seed=8)

    def test_signed_permutation_needs_no_rotations(self):
        """One-by-one blocks leave only the permutations and the signs."""
        matrix = sparse.csc_array(np.diag([1.0, -1.0, 1.0, -1.0])[[2, 0, 3, 1]])
        synthesis = block_sparse_isometry_synthesis(matrix)
        assert synthesis.block_givens.layer_angles == []
        assert len(synthesis.column_permutation) == len(synthesis.row_permutation) == 4

    def test_shared_phase_gradient(self):
        """The block-sparse operation also composes with a caller-owned gradient."""
        matrix = _block_sparse([2, 4, 1, 1], seed=11)
        mapper = create(
            "circuit_mapper",
            "unitary_synthesis",
            method="block_sparse",
            allocate_phase_gradient=False,
            rotation_bit_precision=8,
        )
        circuit = mapper.run(_representation(matrix))
        assert circuit.metadata.num_phase_gradient_ancillas == 8
        _assert_realizes(circuit, matrix.toarray(), seed=11)


class TestResourceEstimate:
    """Resource estimation runs on the exported circuit of each method."""

    @pytest.mark.parametrize("method", ["dense", "block_sparse"])
    def test_estimate(self, method):
        """The estimate counts the target, the gradient and the lookup scratch."""
        matrix = _block_sparse([2, 4, 1, 1], seed=2)[:, :2]
        circuit = create("circuit_mapper", "unitary_synthesis", method=method, rotation_bit_precision=6).run(
            _representation(matrix)
        )
        counts = circuit.estimate()["logicalCounts"]
        assert counts["numQubits"] > 3 + 6
        assert counts["cczCount"] > 0


class TestErrors:
    """Unsupported inputs and inconsistent settings raise ``ValueError``."""

    def test_rejects_other_containers(self):
        """Only real unitary containers can be synthesized."""
        representation = UnitaryRepresentation(
            PauliProductFormulaContainer([ExponentiatedPauliTerm({0: "X"}, 0.1)], 1, 1)
        )
        mapper = create("circuit_mapper", "unitary_synthesis")
        with pytest.raises(ValueError, match="real_unitary"):
            mapper.run(representation)
        with pytest.raises(ValueError, match="real_unitary"):
            mapper.right_factor(representation)

    def test_block_sparse_has_no_right_factor(self):
        """``omit_right_factor`` and ``right_factor`` are specific to the dense method."""
        representation = _representation(np.eye(4))
        mapper = create("circuit_mapper", "unitary_synthesis", method="block_sparse", omit_right_factor=True)
        with pytest.raises(ValueError, match="only to the dense method"):
            mapper.run(representation)
        with pytest.raises(ValueError, match="Only the dense method"):
            mapper.right_factor(representation)

    @pytest.mark.parametrize(
        ("num_qubits", "num_columns", "num_input_qubits"),
        [(3, 5, 2), (4, 2, 1), (2, 2, 3)],
        ids=["too-few-for-columns", "too-many-added", "wider-than-register"],
    )
    def test_invalid_input_register(self, num_qubits, num_columns, num_input_qubits):
        """The input register must hold the columns, add at most two qubits and fit the register."""
        representation = _representation(np.eye(1 << num_qubits)[:, :num_columns])
        mapper = create("circuit_mapper", "unitary_synthesis", num_input_qubits=num_input_qubits)
        with pytest.raises(ValueError, match="num_input_qubits"):
            mapper.run(representation)


def _reconstruct_givens(givens: dict) -> np.ndarray:
    """Return ``D L_{m-1} ... L_0`` of a ``GivensDecomposition`` dict, most significant qubit first."""
    result = np.eye(len(givens["phases"]))
    for angles, shifted in zip(givens["layerAngles"], givens["layerShifted"], strict=True):
        upper = (1 if shifted else 0) + 2 * np.arange(len(angles))
        cosine, sine = np.cos(angles)[:, None], np.sin(angles)[:, None]
        first, second = result[upper], result[upper + 1]
        result[upper], result[upper + 1] = cosine * first - sine * second, sine * first + cosine * second
    return np.where(np.asarray(givens["phases"], dtype=bool), -1.0, 1.0)[:, None] * result


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

    def _run(self, qsharp_test_context, qsharp_test_utils, givens, prep_angles, controls):
        op = qsharp_test_utils.UnitarySynthesisTests.TestMakeGivensOp(
            givens, prep_angles, list(controls), self._ROTATION_BITS
        )
        num_controls, num_target = len(controls), len(prep_angles)
        state = np.array(dump_operation_on_state(op, num_controls + num_target, context=qsharp_test_context))
        control_value = sum(1 << (num_controls - 1 - k) for k, bit in enumerate(controls) if bit)
        state = state.reshape(1 << num_controls, 1 << num_target)
        np.testing.assert_allclose(np.delete(state, control_value, axis=0), 0.0, atol=1e-10)
        return state[control_value]

    def test_rotation_and_sign(self, qsharp_test_context, qsharp_test_utils):
        """A two-layer, signed network matches its dense reconstruction."""
        givens = {"layerAngles": [[0.4], [0.9]], "layerShifted": [False, True], "phases": [True, False, True, False]}
        prep_angles = [1.1, 0.7]
        state = self._run(qsharp_test_context, qsharp_test_utils, givens, prep_angles, ())
        expected = _reconstruct_givens(givens) @ _ry_product_state(prep_angles)
        assert abs(np.vdot(expected, state)) ** 2 > 1 - 1e-3

    @pytest.mark.parametrize("controls", [(), (True,), (False,)], ids=["uncontrolled", "control-on", "control-off"])
    @pytest.mark.parametrize("prep_angles", [[0.8], [0.8, 1.3]], ids=["one-qubit", "two-qubit"])
    def test_global_sign(self, qsharp_test_context, qsharp_test_utils, controls, prep_angles):
        """``D = -I`` negates every amplitude whenever the network applies.

        Without layers there is no measurement-based unlookup, so the global phase is exact.
        """
        givens = {"layerAngles": [], "layerShifted": [], "phases": [True] * (1 << len(prep_angles))}
        state = self._run(qsharp_test_context, qsharp_test_utils, givens, prep_angles, controls)
        initial = _ry_product_state(prep_angles)
        np.testing.assert_allclose(state, initial if controls == (False,) else -initial, atol=1e-10)

    @pytest.mark.parametrize("control", [True, False], ids=["control-on", "control-off"])
    def test_shifted_layer_under_control(self, qsharp_test_context, qsharp_test_utils, control):
        """A shifted layer applies only when the control is set."""
        givens = {"layerAngles": [[0.6]], "layerShifted": [True], "phases": [False, False, False, False]}
        prep_angles = [0.8, 1.3]
        state = self._run(qsharp_test_context, qsharp_test_utils, givens, prep_angles, (control,))
        initial = _ry_product_state(prep_angles)
        expected = _reconstruct_givens(givens) @ initial if control else initial
        assert abs(np.vdot(expected, state)) ** 2 > 1 - 1e-3

    def test_estimate_caching_distinguishes_precision(self, qsharp_test_context, qsharp_test_utils):
        """Networks that differ only in rotation precision are costed separately in one estimate."""
        givens = {"layerAngles": [[0.4], [0.9]], "layerShifted": [False, True], "phases": [True, False, True, False]}
        probe = qsharp_test_utils.UnitarySynthesisTests.TestGivensAtTwoPrecisions
        counts = {
            bits: qsharp_test_context.logical_counts(probe, givens, 2, *bits) for bits in [(6, 6), (9, 9), (6, 9)]
        }
        for key in ("cczCount", "measurementCount"):
            assert counts[(6, 6)][key] != counts[(9, 9)][key]
            assert 2 * counts[(6, 9)][key] == counts[(6, 6)][key] + counts[(9, 9)][key], key


class TestAddConstant:
    """``AddConstant`` adds a signed constant modulo the register size."""

    @pytest.mark.parametrize("width", [1, 2, 3, 5])
    @pytest.mark.parametrize("constant", [0, 1, -1, 3, -5])
    @pytest.mark.parametrize("inverse", [False, True], ids=["forward", "adjoint"])
    def test_adds_constant_modulo_width(self, qsharp_test_utils, width, constant, inverse):
        """Every little-endian starting value maps to ``value ± constant`` modulo ``2**width``."""
        results = qsharp_test_utils.ArithmeticTests.TestAddConstantTable(constant, width, inverse)
        sign = -1 if inverse else 1
        assert list(results) == [(value + sign * constant) % (1 << width) for value in range(1 << width)]


class TestPermutationViaQROAM:
    """``PermutationViaQROAM`` permutes basis states coherently."""

    @pytest.mark.parametrize("num_bits", [1, 2, 3, 4])
    def test_permutes_superposition(self, qsharp_test_context, qsharp_test_utils, num_bits):
        """The lookup, swap and measurement-based unlookup apply the permutation exactly.

        Each repetition samples new X-basis outcomes in the unlookup, so the phase fix-up is
        exercised on different measurement records.
        """
        rng = np.random.default_rng(num_bits)
        reverse = _bit_reversal(num_bits)
        for _ in range(4):
            permutation = rng.permutation(1 << num_bits).tolist()
            angles = rng.uniform(0.2, np.pi - 0.2, num_bits).tolist()
            amplitudes = _ry_product_state(angles[::-1])
            expected = np.zeros_like(amplitudes)
            expected[permutation] = amplitudes

            op = qsharp_test_utils.IsometrySynthesisTests.TestMakePermutationOp(permutation, angles)
            state = np.array(dump_operation_on_state(op, num_bits, context=qsharp_test_context))[reverse]
            assert_states_match_up_to_global_phase(state, expected, atol=1e-10)
