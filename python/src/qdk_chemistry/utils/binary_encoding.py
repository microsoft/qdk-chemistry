"""Binary encoding circuit synthesiser for RREF matrices.

This module implements the "binary encoding" step of the GF2+X pipeline:
given a reduced binary matrix (in RREF or upper-staircase diagonal form),
it synthesises a circuit that compresses the sparse rows into a dense
binary-counter register using batched Toffoli gates and Partial Unary
Iteration (PUI) lookup blocks.

Public API
----------
:class:`BinaryEncodingSynthesizer`
    Main facade.  Use :meth:`BinaryEncodingSynthesizer.from_matrix` to
    create a solved instance, then call
    :meth:`~BinaryEncodingSynthesizer.to_gf2x_operations` to export
    the resulting gate sequence.
:class:`RrefTableau`
    Mutable binary tableau with RREF validation and gate-level updates.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from qdk_chemistry.utils import Logger

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "BinaryEncodingSynthesizer",
    "MatrixCompressionType",
    "NotRrefError",
    "RrefTableau",
]


def _dense_qubits_size(num_cols: int) -> int:
    """Return the dense-register width required to index ``num_cols`` columns."""
    return 1 if num_cols < 2 else math.ceil(math.log2(num_cols))


def _int_to_bits(val: int, nbits: int) -> list[bool]:
    """Convert an integer to a fixed-width MSB-first bit sequence."""
    return [bool((val >> i) & 1) for i in range(nbits - 1, -1, -1)]


def _bits_to_int(bits: Iterable[int | bool]) -> int:
    """Convert an MSB-first bit sequence to integer."""
    return sum(int(b) << i for i, b in enumerate(reversed(list(bits))))


class NotRrefError(ValueError):
    """Raised when a matrix is not in reduced row echelon form (RREF)."""


class MatrixCompressionType(Enum):
    """Operations types recorded during matrix compression."""

    CX = "cx"
    SWAP = "swap"
    TOFFOLI = "toffoli"
    PUI_BLOCK = "pui_block"
    X = "x"


def _is_diagonal_reduction_shape(data: np.ndarray) -> bool:
    """Return True when the pivot sub-matrix is already upper-staircase.

    The filled-REF pattern produced by ``_ref_to_staircase`` has pivot
    columns forming ``np.triu(ones(r, r))`` — upper triangular with 1s on
    and above the diagonal.  When this is detected, binary encoding can
    skip its own CX cascade in ``_apply_unary_staircase`` (only X on the
    first row is needed).
    """
    row_norms = np.any(data, axis=1)
    effective_rows = int(row_norms.sum()) if row_norms.any() else 0
    if effective_rows == 0:
        return False

    # Identify pivot columns (leftmost 1 in each non-zero row)
    pivot_cols: list[int] = []
    for r in range(effective_rows):
        nz = np.flatnonzero(data[r])
        if nz.size == 0:
            return False
        pivot_cols.append(int(nz[0]))

    # Check that pivot sub-matrix is upper-triangular with all 1s
    pivot_submatrix = data[np.ix_(range(effective_rows), pivot_cols)]
    expected = np.triu(np.ones((effective_rows, effective_rows), dtype=np.int8))
    return bool(np.array_equal(pivot_submatrix, expected))


def _check_rref(data: np.ndarray) -> None:
    """Validate that a binary matrix is in reduced row echelon form."""
    num_rows, _ = data.shape
    prev_pivot = -1
    found_zero_row = False
    for row in range(num_rows):
        nz = np.flatnonzero(data[row])
        if nz.size == 0:
            found_zero_row = True
            continue
        if found_zero_row:
            raise NotRrefError(f"Non-zero row {row} appears after an all-zero row")

        pivot_col = int(nz[0])
        if pivot_col <= prev_pivot:
            raise NotRrefError(
                f"Pivot at row {row}, col {pivot_col} is not strictly to the right of previous pivot col {prev_pivot}"
            )
        col_sum = int(data[:, pivot_col].sum())
        if col_sum != 1:
            raise NotRrefError(f"Pivot column {pivot_col} has {col_sum} non-zero entries (expected 1)")
        prev_pivot = pivot_col


class RrefTableau:
    """Binary tableau for the batched sparse-isometry algorithm.

    The input matrix must be in reduced row echelon form (RREF) or
    upper-staircase diagonal-reduction shape.

    The tableau supports in-place updates via the compression operations.
    """

    def __init__(self, data: np.ndarray):
        """Create a tableau from a binary matrix and validate its shape.

        Args:
            data: Binary matrix with rows as qubits and columns as determinant
                basis states. Values are coerced to ``np.int8``.

        Raises:
            NotRrefError: If ``data`` is neither in RREF nor in the accepted
                upper-staircase diagonal-reduction form.
            AssertionError: If the matrix rank/size assumptions required by
                the algorithm are violated.

        """
        self.data = np.asarray(data, dtype=np.int8)
        assert self.data.ndim == 2

        self.is_diagonal_reduced = _is_diagonal_reduction_shape(self.data)
        if not self.is_diagonal_reduced:
            _check_rref(self.data)

        self.num_rows, self.num_cols = self.data.shape
        self.dense_size = _dense_qubits_size(self.num_cols)
        assert self.dense_size < self.num_rows

        self._tmp_row = np.zeros(self.num_cols, dtype=np.int8)
        self.pivots = self.identify_rref_pivots()

        Logger.debug(f"Tableau shape: {self.data.shape}, dense size: {self.dense_size}, pivots: {self.pivots}")

    def get(self, row: int, col: int) -> bool:
        """Return the value at ``(row, col)``."""
        return bool(self.data[row, col])

    def get_col(self, col: int) -> np.ndarray:
        """Return column *col* as a 1-D array."""
        return self.data[:, col]

    def row_is_zero(self, row: int) -> bool:
        """Return True if *row* is all zeros."""
        return not np.any(self.data[row])

    def identify_rref_pivots(self) -> list[tuple[int, int]]:
        """Find pivot positions using vectorized operations."""
        row_indices, col_indices = np.nonzero(self.data)
        _, first_occurrences = np.unique(row_indices, return_index=True)
        return list(
            zip(
                row_indices[first_occurrences].tolist(),
                col_indices[first_occurrences].tolist(),
                strict=True,
            )
        )

    def cx(self, control: int, target: int):
        """Apply CX: ``target ^= control``."""
        self.data[target] ^= self.data[control]

    def swap(self, a: int, b: int):
        """Swap rows *a* and *b*."""
        self.data[[a, b]] = self.data[[b, a]]

    def x(self, row: int):
        """Apply bit-flip (X) to every entry in *row*."""
        self.data[row] ^= 1

    def permute_columns(self, col_order: list[int]):
        """Reorder tableau columns and refresh derived metadata.

        Args:
            col_order: New-to-old index mapping used to permute columns.

        Notes:
            This recomputes ``num_cols``, resets the temporary PUI mask buffer,
            and refreshes cached pivot positions.

        """
        self.data = self.data[:, col_order].copy()
        self.num_cols = self.data.shape[1]
        self._tmp_row = np.zeros(self.num_cols, dtype=np.int8)
        self.pivots = self.identify_rref_pivots()

    def toffoli(self, target: int, ctrl0: tuple[int, bool], ctrl1: tuple[int, bool]):
        """Apply a two-control conditional XOR into ``target``.

        Args:
            target: Target-row index.
            ctrl0: Pair ``(row, value)`` for first control; when ``value`` is
                ``False``, the negated control is used.
            ctrl1: Pair ``(row, value)`` for second control; when ``value`` is
                ``False``, the negated control is used.

        """
        c0, v0 = ctrl0
        c1, v1 = ctrl1
        m0 = self.data[c0] if v0 else (1 - self.data[c0])
        m1 = self.data[c1] if v1 else (1 - self.data[c1])
        self.data[target] ^= m0 & m1

    def _and_controls_into_mask(self, mask: np.ndarray, controls: list[tuple[int, bool]]) -> None:
        """Constrain a mask by conjunction over signed controls.

        Args:
            mask: Mutable bit-mask updated in place.
            controls: Control predicates ``(row, required_value)``.

        """
        for row, val in controls:
            mask &= self.data[row] if val else (1 - self.data[row])

    def toffoli_pui_fixed(self, controls: list[tuple[int, bool]]):
        """Initialize shared PUI mask from fixed controls.

        Args:
            controls: Controls common to every target in one PUI block.

        """
        self._tmp_row[:] = 1
        self._and_controls_into_mask(self._tmp_row, controls)

    def toffoli_pui_rest(self, target_row_offset: int, controls: list[tuple[int, bool]]):
        """Apply one branch of a prepared PUI update.

        Args:
            target_row_offset: Offset from dense/sparse boundary to the target
                sparse row.
            controls: Additional branch-specific controls.

        """
        target = self.dense_size + target_row_offset
        mask = self._tmp_row.copy()
        self._and_controls_into_mask(mask, controls)
        self.data[target] ^= mask


@dataclass
class _BatchElement:
    """Internal tracking structure for one element within a synthesis batch."""

    col: int | None
    dense_content: int


class BinaryEncodingSynthesizer:
    """Synthesise a circuit from a binary RREF tableau using batched sparse isometry.

    The synthesiser executes a two-stage algorithm:

    * **Stage 1 — diagonal encoding**: converts the identity pivot block into
      a compact binary-counter register via a unary-to-binary ladder.
    * **Stage 2 — non-pivot processing**: encodes remaining columns using
      batched Toffoli gates and Partial Unary Iteration (PUI) lookup blocks.

    Typical usage::

        synth = BinaryEncodingSynthesizer.from_matrix(rref_matrix)
        ops, n_anc = synth.to_gf2x_operations(num_local_qubits=n)

    Or, for more control::

        synth = BinaryEncodingSynthesizer(RrefTableau(matrix))
        synth.run()
        ops, n_anc = synth.to_gf2x_operations(num_local_qubits=n)
    """

    def __init__(
        self,
        tableau: RrefTableau,
        *,
        include_negative_controls: bool = True,
        measurement_based_uncompute: bool = False,
    ):
        """Construct solver state for a validated tableau.

        Args:
            tableau: Mutable tableau to transform during synthesis.
            include_negative_controls: If True, include both positive and
                negative (0-valued) fixed controls in PUI blocks.  If False,
                only positive (1-valued) controls are emitted.
            measurement_based_uncompute: If True, emit ``select_and`` ops
                that use measurement-based AND uncomputation (requires
                Adaptive_RI target profile or higher).

        """
        self.tableau = tableau
        self.include_negative_controls = include_negative_controls
        self.measurement_based_uncompute = measurement_based_uncompute

        self.batch: list[_BatchElement] = []
        self.batch_index: int = 0

        self.circuit: list[tuple[MatrixCompressionType, tuple[Any, ...]]] = []
        self.bijection: list[tuple[int, int]] = []
        self.bad_element_count: int = 0

    @property
    def dense_size(self) -> int:
        """Return the number of dense-register rows."""
        return self.tableau.dense_size

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        *,
        include_negative_controls: bool = True,
        measurement_based_uncompute: bool = False,
    ) -> BinaryEncodingSynthesizer:
        """Create a synthesiser, run both stages, and return the solved instance.

        This is the primary entry point.  It validates the input RREF matrix,
        executes the full two-stage synthesis, and returns the ready-to-export
        synthesiser.

        Args:
            matrix: Binary (0/1) matrix in RREF or upper-staircase diagonal
                form, shaped ``(num_qubits, num_determinants)``.
            include_negative_controls: If True, include both positive and
                negative (0-valued) fixed controls in PUI blocks.  If False,
                only positive (1-valued) controls are emitted.
            measurement_based_uncompute: If True, emit ``select_and`` ops
                that use measurement-based AND uncomputation (requires
                Adaptive_RI target profile or higher).

        Returns:
            A solved :class:`BinaryEncodingSynthesizer`.

        Raises:
            NotRrefError: If *matrix* is not in valid RREF form.

        """
        synth = cls(
            RrefTableau(matrix),
            include_negative_controls=include_negative_controls,
            measurement_based_uncompute=measurement_based_uncompute,
        )
        synth.run()
        return synth

    def max_batch_size(self) -> int:
        """Return the maximum batch size supported by the current tableau shape.

        The batch size is the largest power of 2 that fits within the number
        of sparse rows (``num_rows - dense_size``).

        Each batch element occupies a dedicated sparse
        row as a one-hot indicator (element *i* has a 1 at sparse row *i*).
        Therefore the batch cannot exceed the number of available sparse rows.
        """
        sparse_size = self.tableau.num_rows - self.dense_size
        assert sparse_size > 0
        if sparse_size & (sparse_size - 1) == 0:
            return sparse_size
        return 1 << (sparse_size.bit_length() - 1)

    def _record(self, op: tuple[MatrixCompressionType, tuple[Any, ...]]):
        """Append an operation and update the tableau."""
        self.circuit.append(op)
        kind, payload = op

        if kind is MatrixCompressionType.CX:
            self.tableau.cx(*payload)
        elif kind is MatrixCompressionType.SWAP:
            self.tableau.swap(*payload)
        elif kind is MatrixCompressionType.TOFFOLI:
            tgt, ctrl_pos, ctrl_row, ctrl_val = payload
            self.tableau.toffoli(tgt, (ctrl_pos, True), (ctrl_row, ctrl_val))
        elif kind is MatrixCompressionType.PUI_BLOCK:
            fixed_controls, rest_entries = payload
            self.tableau.toffoli_pui_fixed(fixed_controls)
            for off, ctrls in rest_entries:
                self.tableau.toffoli_pui_rest(off, ctrls)
        elif kind is MatrixCompressionType.X:
            self.tableau.x(*payload)

    def run(self):
        """Execute full synthesis and restore original column order."""
        rank, col_perm = self._permute_columns_pivots_first()

        self._run_stage1_diagonal_encoding(rank)

        if self.tableau.num_cols - rank > 0:
            stage_two_start = self._choose_stage_two_start_index(rank)
            self._run_stage2_non_pivot_col_processing(stage_two_start)

        self._complete_bijection()
        self._validate()

        # Remap bijection and tableau back to original column order
        self.bijection = [(dv, col_perm[c]) for dv, c in self.bijection]
        inv_perm = [0] * len(col_perm)
        for new_idx, old_idx in enumerate(col_perm):
            inv_perm[old_idx] = new_idx
        self.tableau.permute_columns(inv_perm)

    def _validate(self):
        """Assert final solver invariants."""
        for row in range(self.dense_size, self.tableau.num_rows):
            assert self.tableau.row_is_zero(row), f"Row {row} not zeroed"
        assert len(self.bijection) == self.tableau.num_cols, "Bijection incomplete"

    def _permute_columns_pivots_first(self) -> tuple[int, list[int]]:
        """Move pivot columns to the front to form a diagonal block.

        Returns:
            Tuple ``(rank, col_perm)`` where ``rank`` is the number of pivot
            columns and ``col_perm`` is the applied forward permutation.

        """
        pivot_cols = [p[1] for p in self.tableau.pivots]
        rank = len(pivot_cols)
        pivot_set = set(pivot_cols)
        non_pivot_cols = [c for c in range(self.tableau.num_cols) if c not in pivot_set]

        col_perm = pivot_cols + non_pivot_cols
        self.tableau.permute_columns(col_perm)
        return rank, col_perm

    # --- Stage 1: Diagonal Encoding ---

    def _run_stage1_diagonal_encoding(self, rank: int):
        """Stage 1: Diagonal encoding.

        Processes only the rank*rank identity pivot block, assigning contiguous
        integer labels (0, 1, 2, …) to the leading pivot columns.

        Two steps:
        1. Unary encoding: CX ladder + X + SWAP converts the identity block
           into an upper-staircase (unary) matrix where column ``c`` has 1s in
           rows 0 through c-1.
        2. Binary compression: A divide-and-conquer loop folds the unary rows
           into binary-counter dense rows using one Toffoli per erased unary
           bit, with no ancilla waste.
        """
        if rank == 0:
            return

        logical_rows = self._apply_unary_staircase(rank)
        self._convert_unary_to_binary(rank, logical_rows)

        # Record bijection for the contiguous pivot columns
        dense_size = self.dense_size
        for c in range(rank):
            dense_val = _bits_to_int(self.tableau.data[:dense_size, c])
            self.bijection.append((dense_val, c))

    def _apply_unary_staircase(self, rank: int) -> list[int]:
        """Convert the identity block into an upper-staircase matrix."""
        logical_rows = list(range(rank))
        if not self.tableau.is_diagonal_reduced:
            for i in range(rank - 2, -1, -1):
                self._record((MatrixCompressionType.CX, (logical_rows[i + 1], logical_rows[i])))
        self._record((MatrixCompressionType.X, (logical_rows[0],)))
        return logical_rows

    def _convert_unary_to_binary(self, limit: int, logical_rows: list[int]):
        """Fold unary rows into binary-counter dense rows."""
        logical_rows = [*logical_rows[1:], logical_rows[0]]

        if limit > 1:
            active_unary = logical_rows[: limit - 1]
            leftover_zero = logical_rows[limit - 1]
            dense_rows, zero_rows = [], []

            while len(active_unary) > 1:
                accumulator = active_unary[0]
                dense_rows.append(accumulator)
                unary_bits = active_unary[1:]
                next_active_unary = []

                for p in range(len(unary_bits) // 2):
                    x, y = unary_bits[2 * p], unary_bits[2 * p + 1]
                    self._record((MatrixCompressionType.CX, (x, accumulator)))
                    self._record((MatrixCompressionType.CX, (y, accumulator)))
                    self._record((MatrixCompressionType.TOFFOLI, (y, accumulator, x, True)))
                    next_active_unary.append(x)
                    zero_rows.append(y)

                if len(unary_bits) % 2 == 1:
                    x = unary_bits[-1]
                    self._record((MatrixCompressionType.CX, (x, accumulator)))
                    next_active_unary.append(x)

                active_unary = next_active_unary

            dense_rows.append(active_unary[0])
            dense_rows = dense_rows[::-1]  # Reverse to MSB-first

            all_zero_rows = [*zero_rows, leftover_zero]
            num_msb_padding = min(self.dense_size - len(dense_rows), len(all_zero_rows))
            final_physical_rows = all_zero_rows[:num_msb_padding] + dense_rows + all_zero_rows[num_msb_padding:]

            # Cycle sort to align physical permutations
            current_pos = {i: i for i in range(limit)}
            row_at = {i: i for i in range(limit)}
            for i in range(limit):
                target_row = final_physical_rows[i]
                if row_at[i] != target_row:
                    curr_idx = current_pos[target_row]
                    self._record((MatrixCompressionType.SWAP, (i, curr_idx)))
                    swapped_row = row_at[i]
                    row_at[curr_idx], current_pos[swapped_row] = swapped_row, curr_idx
                    row_at[i], current_pos[target_row] = target_row, i

    # --- Stage 2: Non-Pivot Processing ---

    def _choose_stage_two_start_index(self, rank: int) -> int:
        """Choose Stage Two start label to reduce first-batch PUI cost.

        Prefer starting at the next ``max_batch_size`` boundary so the first
        Stage Two batch is already alignment-friendly. This is only safe when
        enough dense-label capacity remains to encode all non-pivot columns.

        If capacity is insufficient, return ``rank`` and allow Stage Two to
        flush an early partial batch to reach the next aligned boundary.
        """
        mbs = self.max_batch_size()
        if mbs <= 1:
            return rank

        next_aligned = ((rank + mbs - 1) // mbs) * mbs
        if next_aligned == rank:
            return rank

        non_pivot_cols = self.tableau.num_cols - rank
        return next_aligned if (next_aligned + non_pivot_cols) <= (1 << self.dense_size) else rank

    def _run_stage2_non_pivot_col_processing(self, k_start: int):
        """Stage 2: Non-pivot column processing.

        For each unmapped non-pivot column, locates the next actionable element,
        synthesises the target dense row pattern via CX adjustments, and
        normalises the sparse indicator bit into a one-hot batch row.

        Batches are flushed mid-loop (emitting a PUI block) whenever they reach
        ``max_batch_size`` or would cross an alignment boundary, because sparse
        indicator rows are reused across batches.  The final (partial) batch is
        flushed at the end of the loop, and any remaining edge case is resolved.
        """
        mbs = self.max_batch_size()
        self.batch_index = k_start
        mapped_cols: set[int] = {col for _, col in self.bijection}

        while True:
            if self.batch:
                new_len = len(self.batch) + 1
                block_shift = _dense_qubits_size(new_len)
                crosses = (self.batch[0].dense_content >> block_shift) != (self.batch_index >> block_shift)

                if new_len > mbs or crosses:
                    self._clear_sparse_bits()
                    self.batch.clear()

            target_row = self.dense_size + len(self.batch)
            element = self._find_next_non_zero_element(target_row, mapped_cols)

            if element is not None:
                target_col = self._create_target_row(target_row, element)
                self._permute_col_and_add_to_batch(target_col, target_row)
                mapped_cols.add(target_col)
            else:
                if self.batch:
                    self._clear_sparse_bits()
                    self.batch.clear()
                    continue
                break

    def _find_next_non_zero_element(self, target_row: int, mapped_cols: set[int]) -> tuple[bool, int, int] | None:
        """Find next actionable non-zero element using fast numpy slicing.

        Args:
            target_row: First sparse row to scan for direct one-hot markers.
            mapped_cols: Column indices already assigned in the bijection;
                passed by the caller to avoid repeated reconstruction.

        Returns:
            Triple ``(is_direct, col, row)`` for the best candidate, or
            ``None`` when every unmapped column is fully zeroed in the
            accessible rows.

        """
        unmapped_cols = [c for c in range(self.tableau.num_cols) if c not in mapped_cols]
        if not unmapped_cols:
            return None

        # 1. Check direct sparse rows
        sub_data = self.tableau.data[target_row:, unmapped_cols]
        rows, cols = np.nonzero(sub_data)
        if rows.size > 0:
            return (True, unmapped_cols[cols[0]], target_row + rows[0])

        # 2. Check current batch indicators
        for i, be in enumerate(self.batch):
            brow = self.dense_size + i
            for col in unmapped_cols:
                if be.col is not None and col == be.col:
                    continue
                if self.tableau.data[brow, col]:
                    return (False, col, brow)
        return None

    def _create_target_row(self, target_row: int, element: tuple[bool, int, int]) -> int:
        """Create/normalize the next target-row element and return its column.

        Args:
            target_row: Sparse row that will host the one-hot batch marker.
            element: Triple ``(is_direct, col, row)`` from
                :meth:`_find_next_non_zero_element`.

        Returns:
            Column index selected for insertion into the current batch.

        """
        is_direct, col, row = element
        if is_direct:
            if row != target_row:
                self._record((MatrixCompressionType.SWAP, (target_row, row)))
            return col

        self._synthesize_target_row(target_row, col, row)
        return col

    def _synthesize_target_row(self, target_row: int, col: int, row: int):
        """Synthesize a target row element utilizing vectorized array masking.

        When the only non-zero entry for an unmapped column lives in an
        already-batched row, a Toffoli is emitted to create a fresh indicator
        at ``target_row`` by exploiting a difference between the expected and
        actual column contents.

        Args:
            target_row: Destination sparse row for the new indicator.
            col: Unmapped column to process.
            row: Existing batch row where the non-zero entry was found.

        """
        self.bad_element_count += 1
        batch_idx = row - self.dense_size
        batch_element_bits = _int_to_bits(self.batch[batch_idx].dense_content, self.dense_size)

        is_batch_index = [i == batch_idx for i in range(self.tableau.num_rows - self.dense_size)]
        combined_idx = np.array(batch_element_bits + is_batch_index, dtype=bool)

        col_data = self.tableau.get_col(col).astype(bool)

        # Find differing row, ignoring the current 'row'
        diffs = combined_idx != col_data
        diffs[row] = False

        diff_row = int(np.flatnonzero(diffs)[0])
        diff_val = bool(col_data[diff_row])

        self._record((MatrixCompressionType.TOFFOLI, (target_row, row, diff_row, diff_val)))

    def _permute_col_and_add_to_batch(self, current_col: int, ctrl_row: int):
        """Normalize a column's dense/sparse bits and append it to the batch.

        Emits CX gates controlled by ``ctrl_row`` to align the dense register
        to ``batch_index`` and the sparse register to a one-hot marker, then
        records the new :class:`_BatchElement` and bijection entry.

        Args:
            current_col: Column being processed.
            ctrl_row: Sparse row whose 1-entry controls the CX corrections.

        """
        dense_size = self.dense_size
        k_bits = np.array(_int_to_bits(self.batch_index, dense_size), dtype=bool)

        # Align dense qubits
        dense_col_data = self.tableau.data[:dense_size, current_col].astype(bool)
        for d_qubit in np.flatnonzero(dense_col_data != k_bits):
            self._record((MatrixCompressionType.CX, (ctrl_row, int(d_qubit))))

        # Align sparse qubits to isolate the one-hot marker
        sparse_col_data = self.tableau.data[dense_size:, current_col].astype(bool)
        target_bits = np.zeros(self.tableau.num_rows - dense_size, dtype=bool)
        target_bits[len(self.batch)] = True

        for s_qubit in np.flatnonzero(sparse_col_data != target_bits):
            self._record((MatrixCompressionType.CX, (ctrl_row, dense_size + int(s_qubit))))

        self.batch.append(_BatchElement(current_col, self.batch_index))
        self.bijection.append((self.batch_index, current_col))
        self.batch_index += 1

    def _complete_bijection(self):
        """Fill missing bijection entries from current dense column contents."""
        mapped = {col for _, col in self.bijection}
        self.bijection.extend(
            (_bits_to_int(self.tableau.data[: self.dense_size, c]), c)
            for c in range(self.tableau.num_cols)
            if c not in mapped
        )

    # --- PUI Lowering & Exporting ---

    def _clear_sparse_bits(self):
        """Emit a PUI block that zeroes all sparse indicator rows for the current batch."""
        assert self.batch
        dense_size = self.dense_size
        num_changing = _dense_qubits_size(len(self.batch))
        num_fixed = dense_size - num_changing
        k0 = self.batch[0].dense_content

        fixed_controls = [
            (r, bool((k0 >> (dense_size - 1 - r)) & 1))
            for r in range(num_fixed)
            if self.include_negative_controls or ((k0 >> (dense_size - 1 - r)) & 1)
        ]

        rest_entries = []
        for i, be in enumerate(self.batch):
            changing_controls = [
                (
                    num_fixed + off,
                    bool((be.dense_content >> (dense_size - 1 - num_fixed - off)) & 1),
                )
                for off in range(num_changing)
            ]
            rest_entries.append((i, changing_controls))

        self._record((MatrixCompressionType.PUI_BLOCK, (fixed_controls, rest_entries)))

    def to_gf2x_operations(
        self,
        num_local_qubits: int,
        active_qubit_indices: list[int] | None = None,
        ancilla_start: int | None = None,
    ) -> tuple[list[tuple[str, Any]], int]:
        """Translate recorded circuit operations into GF2+X instruction tuples.

        Args:
            num_local_qubits: Index where ancilla allocation can start.
            active_qubit_indices: Optional mapping from local qubit index (0..num_local_qubits-1)
                to global qubit index. If provided, operations are translated to global indices.
            ancilla_start: Optional global starting index for ancillas. Used if
                active_qubit_indices is provided.

        Returns:
            Pair ``(ops, num_ancilla)`` with generated operations (optionally translated)
            and the total ancilla count consumed by lookup blocks.

        """
        ops: list[tuple[str, Any]] = []
        start_ancilla_idx = num_local_qubits
        max_ancilla_idx = start_ancilla_idx - 1

        for kind, payload in self.circuit:
            if kind is MatrixCompressionType.CX:
                ops.append(("cx", payload))
            elif kind is MatrixCompressionType.SWAP:
                ops.append(("swap", payload))
            elif kind is MatrixCompressionType.TOFFOLI:
                target, ctrl_pos, ctrl_row, ctrl_val = payload
                if not ctrl_val:
                    ops.append(("x", ctrl_row))
                ops.append(("ccx", (target, ctrl_pos, ctrl_row)))
                if not ctrl_val:
                    ops.append(("x", ctrl_row))
            elif kind is MatrixCompressionType.X:
                ops.append(("x", payload[0]))
            elif kind is MatrixCompressionType.PUI_BLOCK:
                fixed_controls, rest_entries = payload
                new_max = self._flush_pui_lookup_block(
                    ops,
                    self.dense_size,
                    fixed_controls,
                    rest_entries,
                    start_ancilla_idx,
                )
                max_ancilla_idx = max(max_ancilla_idx, new_max)

        num_ancilla = max(0, max_ancilla_idx - start_ancilla_idx + 1)

        if active_qubit_indices is not None and ancilla_start is not None:
            ops = self._translate_ops(ops, num_local_qubits, active_qubit_indices, ancilla_start)

        return ops, num_ancilla

    @staticmethod
    def _translate_ops(
        ops: list[tuple[str, Any]],
        num_local_qubits: int,
        active_qubit_indices: list[int],
        ancilla_start: int,
    ) -> list[tuple[str, Any]]:
        """Remap local qubit indices to global topological indices.

        Indices below ``num_local_qubits`` are mapped through
        ``active_qubit_indices``; higher indices are treated as ancillae
        starting at ``ancilla_start``.

        Args:
            ops: Operation list with local indices.
            num_local_qubits: Boundary between active and ancilla indices.
            active_qubit_indices: Local-to-global mapping for active qubits.
            ancilla_start: Global start index for ancilla qubits.

        Returns:
            New operation list with all qubit indices remapped.

        """

        def map_idx(idx: int) -> int:
            return (
                int(active_qubit_indices[idx]) if idx < num_local_qubits else ancilla_start + (idx - num_local_qubits)
            )

        translated: list[tuple[str, Any]] = []
        for op_name, op_args in ops:
            if op_name == "x":
                translated.append(("x", map_idx(int(op_args))))
            elif op_name in {"cx", "swap"}:
                translated.append((op_name, (map_idx(int(op_args[0])), map_idx(int(op_args[1])))))
            elif op_name in {"ccx"}:
                translated.append((op_name, tuple(map_idx(int(a)) for a in op_args)))
            elif op_name == "mcx":
                controls, ctrl_state, target = op_args
                translated.append(
                    (
                        "mcx",
                        (
                            [map_idx(int(q)) for q in controls],
                            list(ctrl_state),
                            map_idx(int(target)),
                        ),
                    )
                )
            elif op_name in ("select", "select_and"):
                data_table, addr_qubits, dat_qubits = op_args
                translated.append(
                    (
                        op_name,
                        (
                            data_table,
                            [map_idx(int(q)) for q in addr_qubits],
                            [map_idx(int(q)) for q in dat_qubits],
                        ),
                    )
                )
            else:
                translated.append((op_name, op_args))
        return translated

    def _flush_pui_lookup_block(
        self,
        ops: list[tuple[str, Any]],
        sbs: int,
        fixed_controls: list[tuple[int, bool]],
        rest_entries: list[tuple[int, list[tuple[int, bool]]]],
        start_ancilla_idx: int,
    ) -> int:
        """Convert one recorded PUI block into lookup-based GF2+X operations.

        Args:
            ops: Destination operation list to append into.
            sbs: Dense-register width.
            fixed_controls: Shared controls for all block entries.
            rest_entries: Per-target offsets and changing controls.
            start_ancilla_idx: First ancilla index available for lookup synth.

        Returns:
            Maximum ancilla index used by this block, or
            ``start_ancilla_idx - 1`` when no ancilla is required.

        """
        if not rest_entries:
            return start_ancilla_idx - 1

        mono_ops, mono_max, mono_and = self._synthesize_single_pui_lookup_block(
            sbs,
            fixed_controls,
            rest_entries,
            start_ancilla_idx,
        )

        chunked = self._split_rest_entries_into_power_of_two_chunks(rest_entries)
        if len(chunked) <= 1:
            ops.extend(mono_ops)
            return mono_max

        chunked_ops, chunked_max, chunked_and = [], start_ancilla_idx - 1, 0
        for chunk in chunked:
            sub_ops, sub_max, sub_and = self._synthesize_single_pui_lookup_block(
                sbs,
                fixed_controls,
                chunk,
                start_ancilla_idx,
            )
            chunked_ops.extend(sub_ops)
            chunked_max = max(chunked_max, sub_max)
            chunked_and += sub_and

        if chunked_and <= mono_and:
            ops.extend(chunked_ops)
            return chunked_max

        ops.extend(mono_ops)
        return mono_max

    def _synthesize_single_pui_lookup_block(
        self,
        sbs: int,
        fixed_controls: list[tuple[int, bool]],
        rest_entries: list[tuple[int, list[tuple[int, bool]]]],
        start_ancilla_idx: int,
    ) -> tuple[list[tuple[str, Any]], int, int]:
        """Lower one PUI sub-block into lookup ops.

        Returns:
            ``(ops, max_ancilla_idx, and_count)`` where ``and_count`` is the
            number of emitted ``and`` operations, used as a Toffoli-cost proxy.

        """
        if not rest_entries:
            return [], start_ancilla_idx - 1, 0

        fixed_controls, rest_entries = self._canonicalize_pui_controls(fixed_controls, rest_entries)
        address_qubits = self._collect_pui_address_qubits(fixed_controls, rest_entries)
        data_qubits = [sbs + offset for offset, _ in rest_entries]

        filtered_table = self._build_pui_lookup_table(fixed_controls, rest_entries, address_qubits)
        if not filtered_table:
            return [], start_ancilla_idx - 1, 0

        lookup_ops, num_ancilla_used = _lookup_select(
            filtered_table,
            address_qubits=address_qubits,
            data_qubits=data_qubits,
            use_measurement_and=self.measurement_based_uncompute,
        )

        typed_lookup_ops = cast("list[tuple[str, Any]]", lookup_ops)
        gf2x_ops = list(reversed(typed_lookup_ops))
        and_count = sum(1 for name, _ in typed_lookup_ops if name == "and")

        max_ancilla = start_ancilla_idx + num_ancilla_used - 1 if num_ancilla_used > 0 else start_ancilla_idx - 1
        return gf2x_ops, max_ancilla, and_count

    def _canonicalize_pui_controls(
        self,
        fixed_controls: list[tuple[int, bool]],
        rest_entries: list[tuple[int, list[tuple[int, bool]]]],
    ) -> tuple[list[tuple[int, bool]], list[tuple[int, list[tuple[int, bool]]]]]:
        """Promote chunk-local constant controls from changing to fixed.

        For a given block, rows that appear in every entry with the same value
        do not need to remain in per-entry changing controls.
        """
        if not rest_entries:
            return fixed_controls, rest_entries

        n_entries = len(rest_entries)
        fixed_map = dict(fixed_controls)

        # Single pass: count occurrences and collect unique values per row
        row_info: dict[int, tuple[int, set[bool]]] = {}
        for _, changing_controls in rest_entries:
            for row, val in changing_controls:
                count, vals = row_info.get(row, (0, set()))
                vals.add(bool(val))
                row_info[row] = (count + 1, vals)

        # Promote rows that appear in all entries with a single value
        for row, (count, values) in row_info.items():
            if count == n_entries and len(values) == 1:
                promoted_val = next(iter(values))
                if row not in fixed_map or fixed_map[row] == promoted_val:
                    fixed_map[row] = promoted_val

        fixed_rows = set(fixed_map)
        simplified_rest = [(off, [(r, v) for r, v in ctrls if r not in fixed_rows]) for off, ctrls in rest_entries]
        return sorted(fixed_map.items()), simplified_rest

    def _split_rest_entries_into_power_of_two_chunks(
        self, rest_entries: list[tuple[int, list[tuple[int, bool]]]]
    ) -> list[list[tuple[int, list[tuple[int, bool]]]]]:
        """Split entries into contiguous power-of-two chunks.

        This keeps control patterns local while converting expensive
        non-power-of-two lookup tables into cheaper composable pieces.
        """
        n = len(rest_entries)
        if n <= 2:
            return [rest_entries]

        chunks, i, remaining = [], 0, n
        while remaining > 0:
            chunk_size = 1 << (remaining.bit_length() - 1)
            chunks.append(rest_entries[i : i + chunk_size])
            i += chunk_size
            remaining -= chunk_size
        return chunks

    def _collect_pui_address_qubits(
        self,
        fixed_controls: list[tuple[int, bool]],
        rest_entries: list[tuple[int, list[tuple[int, bool]]]],
    ) -> list[int]:
        """Collect and sort all control rows that address a PUI lookup table."""
        all_ctrl_rows = {row for row, _ in fixed_controls}
        for _, changing_controls in rest_entries:
            all_ctrl_rows.update(row for row, _ in changing_controls)
        return sorted(all_ctrl_rows)

    def _build_pui_lookup_table(
        self,
        fixed_controls: list[tuple[int, bool]],
        rest_entries: list[tuple[int, list[tuple[int, bool]]]],
        address_qubits: list[int],
    ) -> dict[tuple[int, ...], tuple[int, ...]]:
        """Build sparse truth table for one PUI lookup block.

        Returns:
            Mapping from address bit tuples to one-hot output tuples, with
            all-zero outputs omitted.

        """
        n_outputs = len(rest_entries)
        table: dict[tuple[int, ...], tuple[int, ...]] = {}
        for i, (_, changing_controls) in enumerate(rest_entries):
            ctrl_map = {**dict(fixed_controls), **dict(changing_controls)}
            address = tuple(int(ctrl_map[row]) for row in address_qubits)
            data = tuple(1 if j == i else 0 for j in range(n_outputs))
            table[address] = data

        return table


def _lookup_select(
    table_dict: dict[tuple[int, ...], tuple[int, ...]],
    address_qubits: list[int],
    data_qubits: list[int],
    *,
    use_measurement_and: bool = False,
) -> tuple[list[tuple[str, Any]], int]:
    """Synthesize a lookup-based select or select_and operation for a given truth table.

    Args:
        table_dict: Mapping from address bit tuples to output bit tuples, with all-zero outputs omitted.
        address_qubits: Qubit indices corresponding to the address bits.
        data_qubits: Qubit indices corresponding to the data bits.
        use_measurement_and: If True, emit ``select_and`` ops that use measurement-based AND uncomputation
            (requires Adaptive_RI target profile or higher).
            If False, emit standard ``select`` ops with internal ancilla management.
            The choice affects the number of ancillas used and the structure of the emitted operations.

    Returns:
        Tuple ``(ops, num_ancilla_used)`` where ``ops`` is a list of GF2+X operations implementing the lookup,
        and ``num_ancilla_used`` is the number of ancilla qubits consumed.

    """
    if not table_dict:
        return [], 0

    operations: list[tuple[str, Any]] = []

    n_address = len(address_qubits)
    n_data = len(data_qubits)
    n_entries = 1 << n_address

    # Build dense Bool[][] table from sparse dict.
    # addr_tuple uses little-endian bit ordering: addr_tuple[0] = LSB.
    data_table: list[list[bool]] = [[False] * n_data for _ in range(n_entries)]
    for addr_tuple, data_tuple in table_dict.items():
        addr_int = sum(int(bit) << i for i, bit in enumerate(addr_tuple))
        data_table[addr_int] = [bool(b) for b in data_tuple]

    # Our table index also uses little-endian (addr_tuple[0] = LSB), so
    # pass address_qubits directly without reversing.
    op_name = "select_and" if use_measurement_and else "select"
    operations.append((op_name, (data_table, list(address_qubits), list(data_qubits))))

    # Select uses no ancilla qubits (managed internally by Q#).
    num_ancilla_used = 0
    return operations, num_ancilla_used
