"""Binary encoding circuit synthesiser for REF matrices."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.utils import CaseInsensitiveStrEnum, Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "BinaryEncodingSynthesizer",
    "MatrixCompressionOp",
    "MatrixCompressionType",
    "NotRefError",
    "RefTableau",
]


def _dense_qubits_size(num_cols: int) -> int:
    """Return the dense-register width required to index ``num_cols`` columns.

    Args:
        num_cols: Number of columns to index.

    Returns:
        Number of qubits needed in the dense register to uniquely index all columns.

    """
    return 1 if num_cols < 2 else math.ceil(math.log2(num_cols))


def _int_to_bits(val: int, nbits: int) -> list[bool]:
    """Convert an integer to a fixed-width MSB-first bit sequence.

    Args:
        val: Integer value to convert.
        nbits: Number of bits in the output sequence.

    Returns:
        List of booleans representing the bits of *val*, with the most significant bit first.

    """
    return [bool((val >> i) & 1) for i in range(nbits - 1, -1, -1)]


def _bits_to_int(bits: Iterable[int | bool]) -> int:
    """Convert an MSB-first bit sequence to integer.

    Args:
        bits: Iterable of bits (as integers or booleans), with the most significant bit first.

    Returns:
        Integer value represented by the bit sequence.

    """
    return sum(int(b) << i for i, b in enumerate(reversed(list(bits))))


class NotRefError(ValueError):
    """Raised when a matrix is not in row echelon form (REF)."""


class MatrixCompressionType(CaseInsensitiveStrEnum):
    """Supported operation types for matrix compression."""

    X = "X"
    CX = "CX"
    SWAP = "SWAP"
    CCX = "CCX"
    MCX = "MCX"
    SELECT = "SELECT"
    SELECT_AND = "SELECT_AND"


@dataclass
class MatrixCompressionOp:
    """Gate representation for matrix compression operations."""

    name: MatrixCompressionType
    """Gate type, one of the MatrixCompressionType values."""
    qubits: list[int]
    """Qubit indices involved in the operation."""
    control_state: int = 0
    """Integer encoding of the control state for multi-controlled gates.
    For ``SELECT``/``SELECT_AND``, this stores the number of address qubits."""
    lookup_data: list[list[bool]] = field(default_factory=list)
    """Boolean lookup table for ``SELECT`` operations; empty list for all
    other gate types."""

    def __post_init__(self):
        """Validate the MatrixCompressionOp parameters."""
        if self.name in {MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND} and not self.lookup_data:
            raise ValueError(f"lookup_data must be provided for {self.name} operations")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a camelCase dict matching the Q# ``MatrixCompressionOp`` struct."""
        return {
            "name": self.name,
            "qubits": self.qubits,
            "controlState": self.control_state,
            "lookupData": self.lookup_data,
        }

    def to_qsharp_parameter(self):
        """Convert to a Q# ``MatrixCompressionOp`` struct."""
        return QSHARP_UTILS.BinaryEncoding.MatrixCompressionOp(
            name=self.name,
            qubits=self.qubits,
            controlState=self.control_state,
            lookupData=self.lookup_data,
        )


def _check_ref(data: np.ndarray) -> None:
    """Validate that a binary matrix is in row echelon form (REF).

    REF requires non-zero rows to appear before any all-zero rows and each
    row's leading 1 (pivot) to be strictly to the right of the pivot above.

    Args:
        data: Binary matrix to validate.

    Raises:
        NotRefError: If *data* is not in REF.

    """
    num_rows, _ = data.shape
    prev_pivot = -1
    found_zero_row = False
    for row in range(num_rows):
        nz = np.flatnonzero(data[row])
        if nz.size == 0:
            found_zero_row = True
            continue
        if found_zero_row:
            raise NotRefError(f"Non-zero row {row} appears after an all-zero row")

        pivot_col = int(nz[0])
        if pivot_col <= prev_pivot:
            raise NotRefError(
                f"Pivot at row {row}, col {pivot_col} is not strictly to the right of previous pivot col {prev_pivot}"
            )
        prev_pivot = pivot_col


class RefTableau:
    """Binary tableau for the batched sparse-isometry algorithm.

    The input matrix must be in row echelon form (REF).
    The tableau supports in-place updates via the compression operations.

    """

    def __init__(self, data: np.ndarray):
        """Create a tableau from a binary matrix and validate its shape.

        Args:
            data: Binary matrix with rows as qubits and columns as determinant
                basis states. Values are coerced to ``np.int8``.

        Raises:
            NotRefError: If ``data`` is not in REF.
            AssertionError: If the matrix rank/size assumptions required by
                the algorithm are violated.

        """
        self.data = np.asarray(data, dtype=np.int8)
        if self.data.ndim != 2:
            raise ValueError("Input data must be a 2-dimensional array")

        _check_ref(self.data)

        self.num_rows, self.num_cols = self.data.shape
        self.dense_size = _dense_qubits_size(self.num_cols)
        if self.dense_size >= self.num_rows:
            raise ValueError(
                f"Dense size ({self.dense_size}) must be strictly less than number of rows ({self.num_rows})"
            )

        self._tmp_row = np.zeros(self.num_cols, dtype=np.int8)
        self.pivots = self.identify_pivots()

        Logger.debug(f"Tableau shape: {self.data.shape}, dense size: {self.dense_size}, pivots: {self.pivots}")

    def get(self, row: int, col: int) -> bool:
        """Return the value at ``(row, col)``.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            Boolean value at the specified position in the tableau.

        """
        return bool(self.data[row, col])

    def get_col(self, col: int) -> np.ndarray:
        """Return column *col* as a 1-D array.

        Args:
            col: Column index.

        Returns:
            1-D array representing the specified column.

        """
        return self.data[:, col]

    def row_is_zero(self, row: int) -> bool:
        """Return True if *row* is all zeros.

        Args:
            row: Row index.

        Returns:
            True if the specified row is all zeros, False otherwise.

        """
        return not np.any(self.data[row])

    def identify_pivots(self) -> list[tuple[int, int]]:
        """Find pivot positions using vectorized operations.

        Returns:
            List of pivot positions as (row, col) tuples.

        """
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
        """Apply CX: ``target ^= control``.

        Args:
            control: Control-row index.
            target: Target-row index.

        """
        self.data[target] ^= self.data[control]

    def swap(self, a: int, b: int):
        """Swap rows *a* and *b*.

        Args:
            a: First row index.
            b: Second row index.

        """
        self.data[[a, b]] = self.data[[b, a]]

    def x(self, row: int):
        """Apply bit-flip (X) to every entry in *row*.

        Args:
            row: Row index.

        """
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
        self.pivots = self.identify_pivots()

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

    def select(self, data_table: list[list[bool]], addr_qubits: list[int], dat_qubits: list[int]):
        """Apply a SELECT lookup operation to the tableau.

        For each column, compute an address from ``addr_qubits`` rows
        (little-endian), look up the corresponding ``data_table`` entry,
        and XOR the data bits into the ``dat_qubits`` rows.

        Args:
            data_table: Dense Boolean lookup table indexed by address integer.
            addr_qubits: Row indices used as address bits (index 0 = LSB).
            dat_qubits: Row indices to XOR data into.

        """
        addr_vals = np.zeros(self.num_cols, dtype=int)
        for i, q in enumerate(addr_qubits):
            addr_vals += self.data[q].astype(int) << i
        for j, dq in enumerate(dat_qubits):
            mask = np.array([data_table[a][j] for a in addr_vals], dtype=np.int8)
            self.data[dq] ^= mask


@dataclass
class _BatchElement:
    """Internal tracking structure for one element within a synthesis batch."""

    col: int | None
    """Column index of the element's non-zero entry, or None if the batch row is currently zero."""
    dense_content: int
    """Dense-register content of the batch element."""


class BinaryEncodingSynthesizer:
    """Synthesise a circuit from a binary REF tableau using batched sparse isometry.

    The synthesiser executes a two-stage algorithm:

    * **Stage 1 — diagonal encoding**: converts the identity pivot block into
      a compact binary-counter register via a unary-to-binary ladder.
    * **Stage 2 — non-pivot processing**: encodes remaining columns using
      batched Toffoli gates and Partial Unary Iteration (PUI) lookup blocks.

    """

    def __init__(
        self,
        tableau: RefTableau,
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

        self.circuit: list[tuple[str, Any]] = []
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

        This is the primary entry point.  It validates the input matrix,
        executes the full two-stage synthesis, and returns the ready-to-export
        synthesiser.

        Args:
            matrix: Binary (0/1) matrix in REF form, shaped ``(num_qubits, num_determinants)``.
            include_negative_controls: If True, include both positive and
                negative (0-valued) fixed controls in PUI blocks.  If False,
                only positive (1-valued) controls are emitted.
            measurement_based_uncompute: If True, emit ``select_and`` ops
                that use measurement-based AND uncomputation (requires
                Adaptive_RI target profile or higher).

        Returns:
            A solved :class:`BinaryEncodingSynthesizer`.

        Raises:
            NotRefError: If *matrix* is not in valid REF form.

        """
        synth = cls(
            RefTableau(matrix),
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

    def _record(self, op: tuple[str, Any]):
        """Append an operation and update the tableau.

        Args:
            op: Operation to record, as a tuple of (MatrixCompressionType, qubit_args).

        """
        self.circuit.append(op)
        compress_type, qubit_args = op

        if compress_type is MatrixCompressionType.CX:
            self.tableau.cx(*qubit_args)
        elif compress_type is MatrixCompressionType.SWAP:
            self.tableau.swap(*qubit_args)
        elif compress_type is MatrixCompressionType.CCX:
            self.tableau.toffoli(qubit_args[0], (qubit_args[1], True), (qubit_args[2], True))
        elif compress_type in {MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND}:
            data_table, addr_qubits, dat_qubits = qubit_args
            self.tableau.select(data_table, addr_qubits, dat_qubits)
        elif compress_type is MatrixCompressionType.X:
            self.tableau.x(*qubit_args)

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

        Args:
            rank: Number of pivot columns (size of the identity block).

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
        """Convert the pivot block into an upper-staircase matrix.

        Inspects each above-diagonal entry in the pivot block (columns
        0 … rank-1 after pivot permutation) and emits a CX to fill any
        missing 1

        Processing columns left-to-right ensures side effects on later
        columns are absorbed when they are reached.

        Args:
            rank: Number of pivot columns (size of the pivot block).

        Returns:
            List of logical row indices corresponding to the original pivot rows.

        """
        logical_rows = list(range(rank))
        # Fill above-diagonal 0s in the pivot block to reach upper-staircase
        for j in range(1, rank):
            for i in range(j):
                if not self.tableau.data[logical_rows[i], j]:
                    self._record((MatrixCompressionType.CX, (logical_rows[j], logical_rows[i])))
        self._record((MatrixCompressionType.X, (logical_rows[0],)))
        return logical_rows

    def _convert_unary_to_binary(self, limit: int, logical_rows: list[int]):
        """Fold unary rows into binary-counter dense rows.

        This is a recursive divide-and-conquer process that iteratively folds
        unary rows into binary-counter dense rows.

        Args:
            limit: Number of unary rows to process (initially the rank).
            logical_rows: Current mapping of logical row indices to physical rows.

        """
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
                    self._record((MatrixCompressionType.CCX, (y, accumulator, x)))
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

        Args:
            rank: Number of pivot columns (size of the identity block).

        Returns:
            The chosen start index for Stage Two processing.

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

        Args:
            k_start: Starting index for non-pivot columns to process,
                typically chosen to optimize the first batch's PUI cost.

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

        if not diff_val:
            self._record((MatrixCompressionType.X, (diff_row,)))
        self._record((MatrixCompressionType.CCX, (target_row, row, diff_row)))
        if not diff_val:
            self._record((MatrixCompressionType.X, (diff_row,)))

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

        select_ops: list[tuple[str, Any]] = []
        self._flush_pui_lookup_block(select_ops, dense_size, fixed_controls, rest_entries)
        for op in select_ops:
            self._record(op)

    def to_operations(
        self,
        num_local_qubits: int,
        active_qubit_indices: list[int] | None = None,
        ancilla_start: int | None = None,
        *,
        reverse: bool = False,
    ) -> list[MatrixCompressionOp]:
        """Translate recorded circuit operations into MatrixCompressionOp instances.

        Args:
            num_local_qubits: Number of local (active) qubits.
            active_qubit_indices: Optional mapping from local qubit index (0..num_local_qubits-1)
                to global qubit index. If provided, operations are translated to global indices.
            ancilla_start: Optional global starting index for ancillas. Used if
                active_qubit_indices is provided.
            reverse: If True, reverse the operation order before returning.

        Returns:
            List of MatrixCompressionOp.

        """
        raw_ops: list[tuple[str, Any]] = []

        for compress_type, qubit_args in self.circuit:
            op_type = MatrixCompressionType(compress_type)
            if op_type is MatrixCompressionType.X:
                raw_ops.append((op_type, qubit_args[0]))
            else:
                raw_ops.append((op_type, qubit_args))

        if active_qubit_indices is not None and ancilla_start is not None:
            raw_ops = self._translate_ops(raw_ops, num_local_qubits, active_qubit_indices, ancilla_start)

        ops = [self._to_compression_op(op_type, op_args) for op_type, op_args in raw_ops]
        if reverse:
            ops.reverse()
        return ops

    @staticmethod
    def _to_compression_op(op_type: str, op_args: Any) -> MatrixCompressionOp:
        """Convert a raw circuit tuple into a MatrixCompressionOp.

        Args:
            op_type: The gate type.
            op_args: Gate arguments (qubit indices and optional data).

        Returns:
            A MatrixCompressionOp instance.

        """
        op_type = MatrixCompressionType(op_type)
        if op_type is MatrixCompressionType.X:
            return MatrixCompressionOp(op_type, [int(op_args)])
        if op_type in {MatrixCompressionType.CX, MatrixCompressionType.SWAP}:
            return MatrixCompressionOp(op_type, [int(op_args[0]), int(op_args[1])])
        if op_type is MatrixCompressionType.CCX:
            target, ctrl1, ctrl2 = op_args
            return MatrixCompressionOp(op_type, [int(ctrl1), int(ctrl2), int(target)])
        if op_type in {MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND}:
            data_table, addr_qubits, dat_qubits = op_args
            qubits = [int(q) for q in addr_qubits] + [int(q) for q in dat_qubits]
            return MatrixCompressionOp(op_type, qubits, control_state=len(addr_qubits), lookup_data=data_table)
        if op_type is MatrixCompressionType.MCX:
            controls, control_state, target_qubit = op_args
            qubits = [int(q) for q in controls] + [int(target_qubit)]
            return MatrixCompressionOp(op_type, qubits, control_state=control_state)
        raise ValueError(f"Unknown op type: {op_type}")

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
        for compress_type, op_args in ops:
            op_type = MatrixCompressionType(compress_type)
            if op_type is MatrixCompressionType.X:
                translated.append((MatrixCompressionType.X, map_idx(int(op_args))))
            elif op_type in {MatrixCompressionType.CX, MatrixCompressionType.SWAP}:
                translated.append((op_type, (map_idx(int(op_args[0])), map_idx(int(op_args[1])))))
            elif op_type is MatrixCompressionType.CCX:
                translated.append((op_type, tuple(map_idx(int(a)) for a in op_args)))
            elif op_type is MatrixCompressionType.MCX:
                controls, ctrl_state, target = op_args
                translated.append(
                    (
                        MatrixCompressionType.MCX,
                        (
                            [map_idx(int(q)) for q in controls],
                            ctrl_state,
                            map_idx(int(target)),
                        ),
                    )
                )
            elif op_type in {MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND}:
                data_table, addr_qubits, dat_qubits = op_args
                translated.append(
                    (
                        op_type,
                        (
                            data_table,
                            [map_idx(int(q)) for q in addr_qubits],
                            [map_idx(int(q)) for q in dat_qubits],
                        ),
                    )
                )
            else:
                translated.append((op_type, op_args))
        return translated

    def _flush_pui_lookup_block(
        self,
        ops: list[tuple[str, Any]],
        sbs: int,
        fixed_controls: list[tuple[int, bool]],
        rest_entries: list[tuple[int, list[tuple[int, bool]]]],
    ) -> None:
        """Convert one recorded PUI block into lookup-based GF2+X operations.

        Args:
            ops: Destination operation list to append into.
            sbs: Dense-register width.
            fixed_controls: Shared controls for all block entries.
            rest_entries: Per-target offsets and changing controls.

        """
        if not rest_entries:
            return

        mono_ops, mono_count = self._synthesize_single_pui_lookup_block(
            sbs,
            fixed_controls,
            rest_entries,
        )

        chunked = self._split_rest_entries_into_power_of_two_chunks(rest_entries)
        if len(chunked) <= 1:
            ops.extend(mono_ops)
            return

        chunked_ops, chunked_count = [], 0
        for chunk in chunked:
            sub_ops, sub_count = self._synthesize_single_pui_lookup_block(
                sbs,
                fixed_controls,
                chunk,
            )
            chunked_ops.extend(sub_ops)
            chunked_count += sub_count

        if chunked_count <= mono_count:
            ops.extend(chunked_ops)
            return

        ops.extend(mono_ops)

    def _synthesize_single_pui_lookup_block(
        self,
        sbs: int,
        fixed_controls: list[tuple[int, bool]],
        rest_entries: list[tuple[int, list[tuple[int, bool]]]],
    ) -> tuple[list[tuple[str, Any]], int]:
        """Lower one PUI sub-block into lookup ops.

        Args:
            sbs: Dense-register width.
            fixed_controls: Shared controls for all block entries.
            rest_entries: Per-target offsets and changing controls.

        Returns:
            ``(ops, select_count)`` where ``select_count`` is the number of
            emitted SELECT/SELECT_AND operations, used as a cost proxy.

        """
        if not rest_entries:
            return [], 0

        fixed_controls, rest_entries = self._canonicalize_pui_controls(fixed_controls, rest_entries)
        address_qubits = self._collect_pui_address_qubits(fixed_controls, rest_entries)
        data_qubits = [sbs + offset for offset, _ in rest_entries]

        filtered_table = self._build_pui_lookup_table(fixed_controls, rest_entries, address_qubits)
        if not filtered_table:
            return [], 0

        lookup_ops = _lookup_select(
            filtered_table,
            address_qubits=address_qubits,
            data_qubits=data_qubits,
            use_measurement_and=self.measurement_based_uncompute,
        )

        gf2x_ops = list(reversed(lookup_ops))
        select_count = sum(
            1 for name, _ in lookup_ops if name in (MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND)
        )

        return gf2x_ops, select_count

    def _canonicalize_pui_controls(
        self,
        fixed_controls: list[tuple[int, bool]],
        rest_entries: list[tuple[int, list[tuple[int, bool]]]],
    ) -> tuple[list[tuple[int, bool]], list[tuple[int, list[tuple[int, bool]]]]]:
        """Promote chunk-local constant controls from changing to fixed.

        For a given block, rows that appear in every entry with the same value
        do not need to remain in per-entry changing controls.

        Args:
            fixed_controls: Initial list of fixed controls, as (row, value) pairs.
            rest_entries: List of (offset, changing_controls) where changing_controls is
                a list of (row, value) pairs that may differ between entries.

        Returns:
            Tuple of (new_fixed_controls, new_rest_entries)
                where new_fixed_controls is the updated list of fixed controls and new_rest_entries
                is the updated list of entries with promoted controls removed from changing_controls.

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

        Args:
            rest_entries: List of (offset, changing_controls) where changing_controls is a list of
                (row, value) pairs that may differ between entries.

        Returns:
            List of chunks, where each chunk is a contiguous sublist of rest_entries with length that is a power of two.
                The original order of entries is preserved.

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
        """Collect and sort all control rows that address a PUI lookup table.

        Args:
            fixed_controls: List of fixed controls, as (row, value) pairs.
            rest_entries: List of (offset, changing_controls) where changing_controls is a list of
                (row, value) pairs that may differ between entries.

        Returns:
            Sorted list of all control rows that address the PUI lookup table.

        """
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

        Args:
            fixed_controls: List of fixed controls, as (row, value) pairs.
            rest_entries: List of (offset, changing_controls) where changing_controls is a list of
                (row, value) pairs that may differ between entries.
            address_qubits: List of control rows that will serve as address bits for the lookup.

        Returns:
            Mapping from address bit tuples to one-hot output tuples, with all-zero outputs omitted.

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
) -> list[tuple[str, Any]]:
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
        List of GF2+X operations implementing the lookup.

    """
    if not table_dict:
        return []

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
    op_type = MatrixCompressionType.SELECT_AND if use_measurement_and else MatrixCompressionType.SELECT
    operations.append((op_type, (data_table, list(address_qubits), list(data_qubits))))

    return operations
