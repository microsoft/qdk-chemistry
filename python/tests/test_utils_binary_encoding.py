"""Tests for the binary encoding utils."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np
import pytest

from qdk_chemistry.utils.binary_encoding import (
    BinaryEncodingSynthesizer,
    MatrixCompressionType,
    NotRrefError,
    RrefTableau,
    _bits_to_int,
    _check_rref,
    _dense_qubits_size,
    _int_to_bits,
    _is_diagonal_reduction_shape,
    _lookup_select,
)


class TestDenseQubitsSize:
    """Tests for _dense_qubits_size."""

    @pytest.mark.parametrize(
        ("num_cols", "expected"),
        [
            (1, 1),
            (2, 1),
            (3, 2),
            (4, 2),
            (5, 3),
            (8, 3),
            (16, 4),
            (1000, 10),
        ],
    )
    def test_dense_qubits_size(self, num_cols, expected):
        """ceil(log2(num_cols)) must equal the expected dense register width."""
        assert _dense_qubits_size(num_cols) == expected


class TestIntToBits:
    """Tests for _int_to_bits."""

    @pytest.mark.parametrize(
        ("val", "nbits", "expected"),
        [
            (0, 4, [False, False, False, False]),
            (1, 4, [False, False, False, True]),
            (15, 4, [True, True, True, True]),
            (5, 3, [True, False, True]),
            (0, 1, [False]),
            (1, 1, [True]),
            (3, 5, [False, False, False, True, True]),
        ],
    )
    def test_int_to_bits(self, val, nbits, expected):
        """Integer must convert to the expected big-endian bit list."""
        assert _int_to_bits(val, nbits) == expected


class TestBitsToInt:
    """Tests for _bits_to_int."""

    @pytest.mark.parametrize(
        ("bits", "expected"),
        [
            ([0, 0, 0], 0),
            ([1, 1, 1, 1], 15),
            ([1, 0, 1], 5),
            ([1], 1),
            ([0], 0),
            ([True, False, True], 5),
        ],
    )
    def test_bits_to_int(self, bits, expected):
        """Big-endian bit list must convert to the expected integer."""
        assert _bits_to_int(bits) == expected

    def test_roundtrip(self):
        """Int -> bits -> int must be the identity for all 4-bit values."""
        for val in range(16):
            assert _bits_to_int(_int_to_bits(val, 4)) == val


class TestCheckRref:
    """Tests for _check_rref RREF validation."""

    def test_identity_is_rref(self):
        """Identity matrix is a valid RREF."""
        _check_rref(np.eye(3, dtype=np.int8))

    def test_valid_rref_non_square(self):
        """Non-square matrix with trailing zero row is valid RREF."""
        mat = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]], dtype=np.int8)
        _check_rref(mat)

    def test_valid_rref_with_trailing_zeros(self):
        """RREF with a trailing all-zero row is accepted."""
        mat = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int8)
        _check_rref(mat)

    def test_empty_matrix(self):
        """All-zero matrix is trivially in RREF."""
        _check_rref(np.zeros((3, 3), dtype=np.int8))

    def test_non_rref_pivots_not_increasing(self):
        """Pivots must appear in strictly increasing column order."""
        mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8)
        with pytest.raises(NotRrefError, match="not strictly to the right"):
            _check_rref(mat)

    def test_non_rref_pivot_col_not_unique(self):
        """Pivot column with two non-zero entries is rejected."""
        mat = np.array([[1, 0], [1, 1]], dtype=np.int8)
        with pytest.raises(NotRrefError, match="non-zero entries"):
            _check_rref(mat)

    def test_non_rref_nonzero_after_zero_row(self):
        """Non-zero row appearing after an all-zero row is rejected."""
        mat = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.int8)
        with pytest.raises(NotRrefError, match="after an all-zero row"):
            _check_rref(mat)


class TestIsDiagonalReductionShape:
    """Tests for _is_diagonal_reduction_shape."""

    def test_identity_is_not_staircase(self):
        """Identity matrix has no upper-triangular fill, so it is not staircase."""
        assert not _is_diagonal_reduction_shape(np.eye(3, dtype=np.int8))

    def test_upper_triangular_ones(self):
        """Upper-triangular matrix with fill above the diagonal is staircase."""
        mat = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.int8)
        assert _is_diagonal_reduction_shape(mat)

    def test_all_zeros(self):
        """All-zero matrix is not staircase-shaped."""
        assert not _is_diagonal_reduction_shape(np.zeros((3, 3), dtype=np.int8))


class TestRrefTableau:
    """Tests for RrefTableau construction and gate operations."""

    def _make_rref(self, n_pivots: int, n_extra_cols: int) -> RrefTableau:
        """Build a realistic RREF tableau with fill in non-pivot columns.

        The pivot block is an identity matrix.  Non-pivot columns get
        alternating 0/1 entries (a common pattern after Gaussian
        elimination).

        Args:
            n_pivots: Number of pivot columns (and rows with leading 1s).
            n_extra_cols: Additional non-pivot columns to add after the pivots.

        Returns:
            RrefTableau with the specified shape and pivot structure.

        """
        num_cols = n_pivots + n_extra_cols
        dense_size = _dense_qubits_size(num_cols)
        num_rows = max(n_pivots, dense_size + 1)
        mat = np.zeros((num_rows, num_cols), dtype=np.int8)
        mat[:n_pivots, :n_pivots] = np.eye(n_pivots, dtype=np.int8)
        for c in range(n_pivots, num_cols):
            for r in range(n_pivots):
                mat[r, c] = (r + c) % 2
        return RrefTableau(mat)

    def test_construction_from_rref(self):
        """Valid RREF matrix produces a tableau with correct dimensions and pivots."""
        t = self._make_rref(3, 2)
        assert t.num_rows == 4
        assert t.num_cols == 5
        assert t.dense_size == _dense_qubits_size(5)
        assert len(t.pivots) == 3

    def test_construction_rejects_non_rref(self):
        """Non-RREF matrix must raise NotRrefError."""
        mat = np.array([[0, 1], [1, 0]], dtype=np.int8)
        with pytest.raises(NotRrefError):
            RrefTableau(mat)

    def test_get_and_get_col(self):
        """Element access and column extraction return correct values."""
        t = self._make_rref(3, 0)
        assert t.get(0, 0) is True
        assert t.get(0, 1) is False
        col = t.get_col(1)
        np.testing.assert_array_equal(col, [0, 1, 0])

    def test_row_is_zero(self):
        """Pivot rows are non-zero; trailing rows below rank are zero."""
        t = self._make_rref(3, 2)
        assert not t.row_is_zero(0)
        assert not t.row_is_zero(2)
        assert t.row_is_zero(t.num_rows - 1)

    def test_cx_operation(self):
        """CX XORs the control row into the target row."""
        t = self._make_rref(3, 0)
        t.cx(0, 1)
        np.testing.assert_array_equal(t.data[1], [1, 1, 0])
        np.testing.assert_array_equal(t.data[0], [1, 0, 0])

    def test_swap_operation(self):
        """SWAP exchanges two rows."""
        t = self._make_rref(3, 0)
        t.swap(0, 2)
        np.testing.assert_array_equal(t.data[0], [0, 0, 1])
        np.testing.assert_array_equal(t.data[2], [1, 0, 0])

    def test_x_operation(self):
        """X flips every bit in the target row."""
        t = self._make_rref(3, 0)
        t.x(0)
        np.testing.assert_array_equal(t.data[0], [0, 1, 1])

    def test_toffoli_both_positive(self):
        """Toffoli with both controls positive ANDs the two rows into the target."""
        mat = np.array([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.int8)
        t = RrefTableau(mat)
        t.toffoli(2, (0, True), (1, True))
        np.testing.assert_array_equal(t.data[2], [0, 0, 1, 0])

    def test_toffoli_negative_control(self):
        """Toffoli with a negated control ANDs row0 with ~row1 into the target."""
        mat = np.array([[1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.int8)
        t = RrefTableau(mat)
        t.toffoli(2, (0, True), (1, False))
        np.testing.assert_array_equal(t.data[2], [1, 0, 0, 1])

    def test_identify_rref_pivots(self):
        """Pivot detection returns (row, col) pairs for each leading 1."""
        mat = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0]], dtype=np.int8)
        t = RrefTableau(mat)
        assert t.pivots == [(0, 0), (1, 1)]

    def test_permute_columns(self):
        """Column permutation reorders all rows accordingly."""
        t = self._make_rref(3, 0)
        t.permute_columns([2, 1, 0])
        np.testing.assert_array_equal(t.data[0], [0, 0, 1])
        np.testing.assert_array_equal(t.data[2], [1, 0, 0])

    def test_toffoli_pui_fixed_and_rest(self):
        """Test PUI initialization and per-branch application."""
        mat = np.array(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        t = RrefTableau(mat)
        # Fixed control: row 0 must be 1
        t.toffoli_pui_fixed([(0, True)])
        # _tmp_row should be row0 = [1, 0, 1, 0]
        np.testing.assert_array_equal(t._tmp_row, [1, 0, 1, 0])

        # Apply rest: target offset 0 (row dense_size + 0), control: row 1 True
        # mask = _tmp_row & row1 = [1,0,1,0] & [0,1,0,1] = [0,0,0,0]
        t.toffoli_pui_rest(0, [(1, True)])
        # Row dense_size+0 should be unchanged (XOR with zeros)
        np.testing.assert_array_equal(t.data[t.dense_size], [0, 0, 0, 0])


class TestBinaryEncodingSynthesizerBasic:
    """Basic construction and property tests."""

    def test_from_matrix_identity(self):
        """Identity RREF matrix should produce a valid synthesiser."""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int8)
        synth = BinaryEncodingSynthesizer.from_matrix(mat)
        assert synth.dense_size == _dense_qubits_size(4)
        assert len(synth.bijection) == 4

    def test_from_matrix_rejects_non_rref(self):
        """Non-RREF input must raise NotRrefError."""
        mat = np.array([[0, 1], [1, 0]], dtype=np.int8)
        with pytest.raises(NotRrefError):
            BinaryEncodingSynthesizer.from_matrix(mat)

    def test_max_batch_size_power_of_two(self):
        """max_batch_size must return a positive power of two."""
        mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.int8,
        )
        synth = BinaryEncodingSynthesizer(RrefTableau(mat))
        mbs = synth.max_batch_size()
        assert mbs > 0
        assert mbs & (mbs - 1) == 0  # power of two

    def test_measurement_based_uncompute_flag(self):
        """measurement_based_uncompute flag is stored on the synthesizer."""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int8)
        synth = BinaryEncodingSynthesizer.from_matrix(mat, measurement_based_uncompute=True)
        assert synth.measurement_based_uncompute is True

    def test_include_negative_controls_flag(self):
        """include_negative_controls flag is stored on the synthesizer."""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int8)
        synth = BinaryEncodingSynthesizer.from_matrix(mat, include_negative_controls=False)
        assert synth.include_negative_controls is False

    def test_include_negative_controls_preserves_bijection_semantics(self):
        """Both include_negative_controls settings must produce valid bijections."""
        mat = np.array(
            [[1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]],
            dtype=np.int8,
        )
        for inc_neg in (True, False):
            synth = BinaryEncodingSynthesizer.from_matrix(mat, include_negative_controls=inc_neg)
            assert len(synth.bijection) == 6
            ds = synth.dense_size
            for dv, c in synth.bijection:
                assert _bits_to_int(synth.tableau.data[:ds, c]) == dv


class TestBinaryEncodingSynthesizerBijection:
    """End-to-end compression correctness for BinaryEncodingSynthesizer.

    Each parametrized RREF matrix is fed through from_matrix(); the tests
    verify that the bijection faithfully represents the compressed output.
    """

    @pytest.fixture(
        params=[
            "identity_3x4",
            "identity_4x5",
            "rref_with_fill",
            "wide_rref",
            "minimal_3x3",
            "staircase_4x5",
            "all_pivot_4x5",
            "many_non_pivot_5x8",
        ]
    )
    def rref_matrix(self, request) -> np.ndarray:
        """Parametrized RREF matrices covering various shapes."""
        matrices = {
            # 3 pivots, 1 non-pivot, 1 trailing zero row
            "identity_3x4": np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                dtype=np.int8,
            ),
            # 4 pivots, 1 non-pivot
            "identity_4x5": np.array(
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                dtype=np.int8,
            ),
            # 4 pivots, 2 non-pivot columns with fill
            "rref_with_fill": np.array(
                [
                    [1, 0, 0, 0, 1, 1],
                    [0, 1, 0, 0, 0, 1],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 1, 1, 1],
                ],
                dtype=np.int8,
            ),
            # 5 pivots, 3 non-pivot columns (wide matrix, dense_size=3)
            "wide_rref": np.array(
                [
                    [1, 0, 0, 0, 0, 1, 1, 0],
                    [0, 1, 0, 0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 0, 0, 1],
                ],
                dtype=np.int8,
            ),
            # Minimal: 2 pivots, 1 non-pivot, 1 trailing zero row
            "minimal_3x3": np.array(
                [[1, 0, 1], [0, 1, 1], [0, 0, 0]],
                dtype=np.int8,
            ),
            # Upper-staircase (diagonal-reduced) shape
            "staircase_4x5": np.array(
                [
                    [1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 0],
                ],
                dtype=np.int8,
            ),
            # All pivots, single zero non-pivot column (stage-2 trivial)
            "all_pivot_4x5": np.array(
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]],
                dtype=np.int8,
            ),
            # 4 pivots, 4 non-pivot columns — exercises batch flushing
            "many_non_pivot_5x8": np.array(
                [
                    [1, 0, 0, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 0, 1, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
        }
        return matrices[request.param]

    def test_bijection_covers_all_columns(self, rref_matrix):
        """Every column must appear exactly once in the bijection."""
        synth = BinaryEncodingSynthesizer.from_matrix(rref_matrix)
        cols = [c for _, c in synth.bijection]
        assert sorted(cols) == list(range(rref_matrix.shape[1]))

    def test_bijection_dense_labels_unique(self, rref_matrix):
        """Dense labels must be unique."""
        synth = BinaryEncodingSynthesizer.from_matrix(rref_matrix)
        dense_vals = [dv for dv, _ in synth.bijection]
        assert len(set(dense_vals)) == len(dense_vals)

    def test_bijection_dense_labels_fit_in_register(self, rref_matrix):
        """All dense labels must fit in the dense register."""
        synth = BinaryEncodingSynthesizer.from_matrix(rref_matrix)
        max_label = (1 << synth.dense_size) - 1
        for dv, _ in synth.bijection:
            assert 0 <= dv <= max_label

    def test_sparse_rows_zeroed_after_synthesis(self, rref_matrix):
        """After synthesis, all sparse rows should be all-zero."""
        synth = BinaryEncodingSynthesizer.from_matrix(rref_matrix)
        for row in range(synth.dense_size, synth.tableau.num_rows):
            assert synth.tableau.row_is_zero(row)

    def test_dense_register_matches_bijection(self, rref_matrix):
        """Reading dense rows of each column must reproduce the bijection label.

        This is the core compression-correctness check: the synthesizer
        transforms the original RREF matrix so that the top ``dense_size``
        rows encode a binary label for every column, and that label matches
        what the bijection records.
        """
        synth = BinaryEncodingSynthesizer.from_matrix(rref_matrix)
        ds = synth.dense_size
        for dense_val, col in synth.bijection:
            actual = _bits_to_int(synth.tableau.data[:ds, col])
            assert actual == dense_val, f"Column {col}: bijection says {dense_val}, but dense register reads {actual}"


class TestBinaryEncodingSynthesizerCircuit:
    """Tests on the recorded circuit structure."""

    def test_circuit_nonempty(self):
        """Synthesis must produce at least one circuit operation."""
        mat = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int8)
        synth = BinaryEncodingSynthesizer.from_matrix(mat)
        assert len(synth.circuit) > 0

    def test_circuit_op_types_valid(self):
        """Every circuit entry must be a MatrixCompressionType variant."""
        mat = np.array(
            [[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]],
            dtype=np.int8,
        )
        synth = BinaryEncodingSynthesizer.from_matrix(mat)
        for operation_type, _ in synth.circuit:
            assert isinstance(operation_type, MatrixCompressionType)

    def test_stage1_starts_with_cx_or_x(self):
        """Stage 1 always begins with CX or X (unary staircase)."""
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int8)
        synth = BinaryEncodingSynthesizer.from_matrix(mat)
        first_operation_type = synth.circuit[0][0]
        assert first_operation_type in (MatrixCompressionType.CX, MatrixCompressionType.X)


class TestBinaryEncodingSynthesizerReplay:
    """Verify that replaying the circuit on the original matrix produces the final tableau."""

    def test_replay_matches_final_state(self):
        """Manually replaying circuit ops on the original matrix must match final tableau."""
        mat = np.array(
            [[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1], [0, 0, 0, 0, 0]],
            dtype=np.int8,
        )
        synth = BinaryEncodingSynthesizer.from_matrix(mat)

        # Reconstruct by creating a fresh tableau and replaying
        replay = RrefTableau(mat.copy())

        # Permute columns the same way the solver did
        pivot_cols = [p[1] for p in replay.pivots]
        pivot_set = set(pivot_cols)
        non_pivot = [c for c in range(replay.num_cols) if c not in pivot_set]
        col_perm = pivot_cols + non_pivot
        replay.permute_columns(col_perm)

        # Replay all operations
        for operation_type, payload in synth.circuit:
            if operation_type is MatrixCompressionType.CX:
                replay.cx(*payload)
            elif operation_type is MatrixCompressionType.SWAP:
                replay.swap(*payload)
            elif operation_type is MatrixCompressionType.TOFFOLI:
                tgt, ctrl_pos, ctrl_row, ctrl_val = payload
                replay.toffoli(tgt, (ctrl_pos, True), (ctrl_row, ctrl_val))
            elif operation_type is MatrixCompressionType.X:
                replay.x(*payload)
            elif operation_type is MatrixCompressionType.PUI_BLOCK:
                fixed_controls, rest_entries = payload
                replay.toffoli_pui_fixed(fixed_controls)
                for off, ctrls in rest_entries:
                    replay.toffoli_pui_rest(off, ctrls)

        # Undo column permutation
        inv_perm = [0] * len(col_perm)
        for new_idx, old_idx in enumerate(col_perm):
            inv_perm[old_idx] = new_idx
        replay.permute_columns(inv_perm)

        # Final state should match
        np.testing.assert_array_equal(replay.data, synth.tableau.data)


class TestToGf2xOperations:
    """Tests for operation export."""

    def test_returns_ops_and_ancilla_count(self):
        """to_gf2x_operations returns an op list."""
        mat = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int8)
        synth = BinaryEncodingSynthesizer.from_matrix(mat)
        ops = synth.to_gf2x_operations(num_local_qubits=3)
        assert isinstance(ops, list)

    def test_op_names_are_strings(self):
        """All emitted op names must belong to the known gate vocabulary."""
        mat = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.int8)
        synth = BinaryEncodingSynthesizer.from_matrix(mat)
        ops = synth.to_gf2x_operations(num_local_qubits=3)
        valid_names = {"cx", "swap", "ccx", "x", "mcx", "select", "select_and"}
        for name, _ in ops:
            assert name in valid_names, f"Unexpected op name: {name}"

    def test_translate_ops_identity_mapping(self):
        """When active_qubit_indices is identity, ops stay the same."""
        mat = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int8)
        synth = BinaryEncodingSynthesizer.from_matrix(mat)
        ops_raw = synth.to_gf2x_operations(num_local_qubits=3)
        ops_xlat = synth.to_gf2x_operations(
            num_local_qubits=3,
            active_qubit_indices=[0, 1, 2],
            ancilla_start=3,
        )
        # With identity mapping and ancilla_start = num_local, should be equivalent
        assert len(ops_raw) == len(ops_xlat)

    def test_translate_ops_remaps_indices(self):
        """Translation must remap qubit indices through the provided map."""
        ops = [("cx", (0, 1)), ("x", 2)]
        translated = BinaryEncodingSynthesizer._translate_ops(
            ops,
            num_local_qubits=3,
            active_qubit_indices=[10, 20, 30],
            ancilla_start=100,
        )
        assert translated[0] == ("cx", (10, 20))
        assert translated[1] == ("x", 30)

    def test_translate_ops_remaps_ancilla(self):
        """Indices >= num_local_qubits should map to ancilla space."""
        ops = [("cx", (0, 3))]
        translated = BinaryEncodingSynthesizer._translate_ops(
            ops,
            num_local_qubits=3,
            active_qubit_indices=[10, 20, 30],
            ancilla_start=100,
        )
        # Index 3 >= num_local_qubits=3, so maps to ancilla_start + (3 - 3) = 100
        assert translated[0] == ("cx", (10, 100))

    def test_translate_ops_ccx(self):
        """CCX indices are remapped through active_qubit_indices."""
        ops = [("ccx", (0, 1, 2))]
        translated = BinaryEncodingSynthesizer._translate_ops(
            ops,
            num_local_qubits=3,
            active_qubit_indices=[5, 6, 7],
            ancilla_start=10,
        )
        assert translated[0] == ("ccx", (5, 6, 7))

    def test_translate_ops_select(self):
        """Select ops remap address and data qubit indices, keeping the table."""
        data_table = [[True, False], [False, True]]
        ops = [("select", (data_table, [0, 1], [2, 3]))]
        translated = BinaryEncodingSynthesizer._translate_ops(
            ops,
            num_local_qubits=4,
            active_qubit_indices=[10, 11, 12, 13],
            ancilla_start=100,
        )
        _, (dt, addr, dat) = translated[0]
        assert dt is data_table
        assert addr == [10, 11]
        assert dat == [12, 13]

    def test_translate_ops_mcx(self):
        """MCX remaps control and target indices, preserving control states."""
        ops = [("mcx", ([0, 1], [True, False], 2))]
        translated = BinaryEncodingSynthesizer._translate_ops(
            ops,
            num_local_qubits=3,
            active_qubit_indices=[5, 6, 7],
            ancilla_start=10,
        )
        _, (ctrls, state, tgt) = translated[0]
        assert ctrls == [5, 6]
        assert state == [True, False]
        assert tgt == 7

    def test_measurement_based_uses_select_and(self):
        """With measurement_based_uncompute, PUI blocks should emit select_and."""
        mat = np.array(
            [[1, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 1, 1]],
            dtype=np.int8,
        )
        synth = BinaryEncodingSynthesizer.from_matrix(mat, measurement_based_uncompute=True)
        ops = synth.to_gf2x_operations(num_local_qubits=4)
        select_names = {name for name, _ in ops if "select" in name}
        if select_names:
            assert "select_and" in select_names


class TestLookupSelect:
    """Tests for the sparse-to-dense lookup table synthesiser."""

    def test_empty_table(self):
        """Empty truth table produces no ops."""
        ops = _lookup_select({}, [0], [1])
        assert ops == []

    def test_single_entry(self):
        """Single-entry table emits one select op."""
        table = {(1,): (1,)}
        ops = _lookup_select(table, [0], [1])
        assert len(ops) == 1
        assert ops[0][0] == "select"

    def test_two_address_bits(self):
        """Two address bits produce a 2^2 = 4 entry dense data table."""
        table = {(0, 1): (1,), (1, 0): (1,)}
        ops = _lookup_select(table, [0, 1], [2])
        assert len(ops) == 1
        name, (data_table, addr, dat) = ops[0]
        assert name == "select"
        assert addr == [0, 1]
        assert dat == [2]
        assert len(data_table) == 4

    def test_data_table_correctness(self):
        """Verify the dense Bool[][] table encodes the sparse dict correctly."""
        # Address (1,0) → data (1,0), address (0,1) → data (0,1)
        table = {(1, 0): (1, 0), (0, 1): (0, 1)}
        ops = _lookup_select(table, [0, 1], [2, 3])
        _, (data_table, _, _) = ops[0]
        # addr_int for (1,0): bit0=1, bit1=0 → addr_int=1
        assert data_table[1] == [True, False]
        # addr_int for (0,1): bit0=0, bit1=1 → addr_int=2
        assert data_table[2] == [False, True]
        # Other entries should be all-false
        assert data_table[0] == [False, False]
        assert data_table[3] == [False, False]

    def test_select_and_mode(self):
        """use_measurement_and=True emits select_and instead of select."""
        table = {(1,): (1,)}
        ops = _lookup_select(table, [0], [1], use_measurement_and=True)
        assert ops[0][0] == "select_and"
