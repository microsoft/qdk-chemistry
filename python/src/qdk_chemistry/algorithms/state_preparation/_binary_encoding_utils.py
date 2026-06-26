"""Binary encoding circuit synthesiser for REF matrices."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qdk_chemistry.utils import CaseInsensitiveStrEnum, Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__ = [
    "MatrixCompressionOp",
    "MatrixCompressionType",
]


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
    qubits: list[int]
    control_state: int = 0
    lookup_data: list[list[bool]] = field(default_factory=list)

    def __post_init__(self):
        """Validate that SELECT/SELECT_AND operations have lookup_data.

        Raises:
            ValueError: If name is SELECT or SELECT_AND but lookup_data is empty.

        """
        if self.name in {MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND} and not self.lookup_data:
            raise ValueError(f"lookup_data must be provided for {self.name} operations")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a camelCase dict matching the Q# ``MatrixCompressionOp`` struct.

        Returns:
            dict[str, Any]: Dictionary with keys 'name', 'qubits', 'controlState', 'lookupData'.

        """
        return {
            "name": self.name,
            "qubits": self.qubits,
            "controlState": self.control_state,
            "lookupData": self.lookup_data,
        }

    def to_qsharp_parameter(self):
        """Convert to a Q# ``MatrixCompressionOp`` struct.

        Returns:
            A Q# MatrixCompressionOp struct instance for use in Q# interop.

        """
        return QSHARP_UTILS.BinaryEncoding.MatrixCompressionOp(
            name=self.name,
            qubits=self.qubits,
            controlState=self.control_state,
            lookupData=self.lookup_data,
        )


class RefTableau:
    """Binary REF tableau with in-place gate operations for synthesis simulation."""

    def __init__(self, data: np.ndarray):
        """Initialize a RefTableau from a binary matrix in row echelon form.

        Args:
            data: A 2-D binary numpy array in row echelon form (REF).

        Raises:
            ValueError: If data is not 2-dimensional.
            NotRefError: If data is not in valid row echelon form.

        """
        self.data = np.asarray(data, dtype=np.int8)
        if self.data.ndim != 2:
            raise ValueError("Input data must be a 2-dimensional array")
        self._validate_ref(self.data)
        self.num_rows, self.num_cols = self.data.shape
        self.dense_size = 1 if self.num_cols < 2 else math.ceil(math.log2(self.num_cols))
        self.pivots = self.identify_pivots()
        Logger.debug(f"Tableau shape: {self.data.shape}, dense size: {self.dense_size}, pivots: {self.pivots}")

    @staticmethod
    def _validate_ref(data: np.ndarray) -> None:
        """Validate that a binary matrix is in row echelon form (REF).

        Args:
            data: A 2-D numpy array to validate.

        Raises:
            ValueError: If the matrix violates REF constraints.

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
                raise ValueError(f"Non-zero row {row} appears after an all-zero row")
            pivot_col = int(nz[0])
            if pivot_col <= prev_pivot:
                raise ValueError(
                    f"Pivot at row {row}, col {pivot_col} is not strictly "
                    f"to the right of previous pivot col {prev_pivot}"
                )
            prev_pivot = pivot_col

    def identify_pivots(self) -> list[tuple[int, int]]:
        """Find pivot positions using vectorized operations.

        Returns:
            list[tuple[int, int]]: List of (row, col) pairs for each pivot.

        """
        row_indices, col_indices = np.nonzero(self.data)
        _, first_occurrences = np.unique(row_indices, return_index=True)
        return list(zip(row_indices[first_occurrences].tolist(), col_indices[first_occurrences].tolist(), strict=True))

    def cx(self, control: int, target: int):
        """Apply CX: ``target ^= control``.

        Args:
            control: Row index of the control qubit.
            target: Row index of the target qubit.

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
            row: Row index to flip.

        """
        self.data[row] ^= 1

    def permute_columns(self, col_order: list[int]):
        """Reorder tableau columns and refresh derived metadata.

        Args:
            col_order: New column ordering as a list of original column indices.

        """
        self.data = self.data[:, col_order].copy()
        self.num_cols = self.data.shape[1]
        self.pivots = self.identify_pivots()

    def toffoli(self, target: int, ctrl0: tuple[int, bool], ctrl1: tuple[int, bool]):
        """Apply a two-control conditional XOR into ``target``.

        Args:
            target: Row index of the target qubit.
            ctrl0: Tuple of (row_index, polarity) for the first control.
            ctrl1: Tuple of (row_index, polarity) for the second control.

        """
        c0, v0 = ctrl0
        c1, v1 = ctrl1
        m0 = self.data[c0] if v0 else (1 - self.data[c0])
        m1 = self.data[c1] if v1 else (1 - self.data[c1])
        self.data[target] ^= m0 & m1

    def select(self, data_table: list[list[bool]], addr_qubits: list[int], dat_qubits: list[int]):
        """Apply a SELECT lookup operation to the tableau.

        Args:
            data_table: 2-D boolean table indexed by address value.
            addr_qubits: Row indices forming the address register.
            dat_qubits: Row indices forming the data register.

        """
        addr_vals = np.zeros(self.num_cols, dtype=int)
        for i, q in enumerate(addr_qubits):
            addr_vals += self.data[q].astype(int) << i
        for j, dq in enumerate(dat_qubits):
            mask = np.array([data_table[a][j] for a in addr_vals], dtype=np.int8)
            self.data[dq] ^= mask


class _BinaryEncodingSynthesizer:
    """Internal: synthesise a circuit from a binary REF tableau using batched sparse isometry.

    Two-stage algorithm:
    * Stage 1 — diagonal encoding: unary-to-binary compression of pivot columns.
    * Stage 2 — non-pivot processing: batched Toffoli + PUI lookup blocks.
    """

    def __init__(
        self, tableau: RefTableau, *, include_negative_controls: bool = True, measurement_based_uncompute: bool = False
    ):
        """Initialize the synthesizer with a REF tableau and configuration.

        Args:
            tableau: A validated RefTableau instance to synthesize from.
            include_negative_controls: Whether to include negative (anti-) controls in PUI blocks.
            measurement_based_uncompute: Whether to use measurement-based uncomputation (SELECT_AND).

        Raises:
            ValueError: If the tableau is already dense and binary encoding is not applicable.

        """
        self.tableau = tableau
        if tableau.dense_size >= tableau.num_rows:
            raise ValueError(
                f"Binary encoding is not applicable: state is already dense "
                f"({tableau.num_cols} determinant(s) require a {tableau.dense_size}-qubit dense register, "
                f"leaving no spare rows in a {tableau.num_rows}-row matrix)."
            )
        self.include_negative_controls = include_negative_controls
        self.measurement_based_uncompute = measurement_based_uncompute
        self.batch: list[tuple[int, int]] = []
        self.batch_index: int = 0
        self.circuit: list[tuple[str, Any]] = []
        self.bijection: list[tuple[int, int]] = []

    def synthesize(
        self, *, num_local_qubits: int, active_qubit_indices: list[int], ancilla_start: int
    ) -> tuple[list[MatrixCompressionOp], list[tuple[int, int]], int]:
        """Run full synthesis pipeline and return circuit operations.

        Args:
            num_local_qubits: Number of system (local) qubits in the circuit.
            active_qubit_indices: Mapping from tableau row indices to global qubit indices.
            ancilla_start: Global qubit index where ancilla qubits begin.

        Returns:
            tuple[list[MatrixCompressionOp], list[tuple[int, int]], int]: A 3-tuple of (ops, bijection, dense_size).

        """
        rank, col_perm = self._permute_columns_pivots_first()
        self._run_stage1_diagonal_encoding(rank)

        if self.tableau.num_cols - rank > 0:
            stage_two_start = self._choose_stage_two_start_index(rank)
            self._run_stage2_non_pivot_col_processing(stage_two_start)

        # Complete bijection for any remaining unmapped columns
        mapped = {col for _, col in self.bijection}
        self.bijection.extend(
            (sum(int(b) << i for i, b in enumerate(reversed(list(self.tableau.data[: self.tableau.dense_size, c])))), c)
            for c in range(self.tableau.num_cols)
            if c not in mapped
        )

        # Validate final invariants
        for row in range(self.tableau.dense_size, self.tableau.num_rows):
            assert not np.any(self.tableau.data[row]), f"Row {row} not zeroed"
        assert len(self.bijection) == self.tableau.num_cols, "Bijection incomplete"

        # Remap bijection and tableau back to original column order
        self.bijection = [(dv, col_perm[c]) for dv, c in self.bijection]
        inv_perm = [0] * len(col_perm)
        for new_idx, old_idx in enumerate(col_perm):
            inv_perm[old_idx] = new_idx
        self.tableau.permute_columns(inv_perm)

        ops = self._to_operations(
            num_local_qubits=num_local_qubits,
            active_qubit_indices=active_qubit_indices,
            ancilla_start=ancilla_start,
        )
        return ops, self.bijection, self.tableau.dense_size

    def max_batch_size(self) -> int:
        """Largest power-of-2 batch that fits in the sparse rows.

        Returns:
            int: Maximum batch size as the largest power-of-2 not exceeding the sparse row count.

        """
        sparse_size = self.tableau.num_rows - self.tableau.dense_size
        assert sparse_size > 0
        if sparse_size & (sparse_size - 1) == 0:
            return sparse_size
        return 1 << (sparse_size.bit_length() - 1)

    def _record(self, op: tuple[str, Any]):
        """Append an operation and apply it to the tableau.

        Args:
            op: A tuple of (MatrixCompressionType, qubit_args) representing the gate to record.

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

    def _permute_columns_pivots_first(self) -> tuple[int, list[int]]:
        """Move pivot columns to the front; return rank and column permutation.

        Returns:
            tuple[int, list[int]]: A pair of (rank, col_perm) where rank is the number of pivots
                and col_perm is the applied column ordering.

        """
        pivot_cols = [p[1] for p in self.tableau.pivots]
        rank = len(pivot_cols)
        pivot_set = set(pivot_cols)
        non_pivot_cols = [c for c in range(self.tableau.num_cols) if c not in pivot_set]
        col_perm = pivot_cols + non_pivot_cols
        self.tableau.permute_columns(col_perm)
        return rank, col_perm

    def _run_stage1_diagonal_encoding(self, rank: int):
        """Convert identity pivot block to unary staircase then to binary counter.

        Args:
            rank: Number of pivot columns (rows with pivots).

        """
        if rank == 0:
            return
        logical_rows = self._apply_unary_staircase(rank)
        self._convert_unary_to_binary(rank, logical_rows)

        # Record bijection for the contiguous pivot columns
        for c in range(rank):
            dense_val = sum(
                int(b) << i for i, b in enumerate(reversed(list(self.tableau.data[: self.tableau.dense_size, c])))
            )
            self.bijection.append((dense_val, c))

    def _apply_unary_staircase(self, rank: int) -> list[int]:
        """Convert the pivot block into an upper-staircase pattern.

        Args:
            rank: Number of pivot rows to process.

        Returns:
            list[int]: Logical row indices after staircase transformation.

        """
        logical_rows = list(range(rank))
        for j in range(1, rank):
            for i in range(j):
                if not self.tableau.data[logical_rows[i], j]:
                    self._record((MatrixCompressionType.CX, (logical_rows[j], logical_rows[i])))
        self._record((MatrixCompressionType.X, (logical_rows[0],)))
        return logical_rows

    def _convert_unary_to_binary(self, limit: int, logical_rows: list[int]):
        """Fold unary rows into binary-counter dense rows via divide-and-conquer.

        Args:
            limit: Number of unary rows to convert.
            logical_rows: Current logical-to-physical row mapping.

        """
        logical_rows = [*logical_rows[1:], logical_rows[0]]
        if limit <= 1:
            return

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
        dense_rows = dense_rows[::-1]

        all_zero_rows = [*zero_rows, leftover_zero]
        num_msb_padding = min(self.tableau.dense_size - len(dense_rows), len(all_zero_rows))
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

    def _choose_stage_two_start_index(self, rank: int) -> int:
        """Choose start label aligned to max_batch_size boundary when possible.

        Args:
            rank: Number of pivot columns already encoded in stage 1.

        Returns:
            int: Starting dense-register label for stage 2 processing.

        """
        mbs = self.max_batch_size()
        if mbs <= 1:
            return rank
        next_aligned = ((rank + mbs - 1) // mbs) * mbs
        if next_aligned == rank:
            return rank
        non_pivot_cols = self.tableau.num_cols - rank
        return next_aligned if (next_aligned + non_pivot_cols) <= (1 << self.tableau.dense_size) else rank

    def _run_stage2_non_pivot_col_processing(self, k_start: int):
        """Process non-pivot columns in batches, flushing PUI blocks at boundaries.

        Args:
            k_start: Starting dense-register label for batch indexing.

        """
        mbs = self.max_batch_size()
        self.batch_index = k_start
        mapped_cols: set[int] = {col for _, col in self.bijection}

        while True:
            if self.batch:
                new_len = len(self.batch) + 1
                block_shift = 1 if new_len < 2 else math.ceil(math.log2(new_len))
                crosses = (self.batch[0][1] >> block_shift) != (self.batch_index >> block_shift)
                if new_len > mbs or crosses:
                    self._clear_sparse_bits()
                    self.batch.clear()

            target_row = self.tableau.dense_size + len(self.batch)
            element = self._find_next_non_zero_element(target_row, mapped_cols)

            if element is not None:
                is_direct, col, row = element
                if is_direct:
                    if row != target_row:
                        self._record((MatrixCompressionType.SWAP, (target_row, row)))
                else:
                    self._synthesize_target_row(target_row, col, row)
                self._permute_col_and_add_to_batch(col, target_row)
                mapped_cols.add(col)
            else:
                if self.batch:
                    self._clear_sparse_bits()
                    self.batch.clear()
                    continue
                break

    def _find_next_non_zero_element(self, target_row: int, mapped_cols: set[int]) -> tuple[bool, int, int] | None:
        """Find next actionable non-zero element in unmapped columns.

        Args:
            target_row: Row index to start searching from.
            mapped_cols: Set of column indices already processed.

        Returns:
            tuple[bool, int, int] | None: A tuple (is_direct, col, row) or None if no element found.

        """
        unmapped_cols = [c for c in range(self.tableau.num_cols) if c not in mapped_cols]
        if not unmapped_cols:
            return None

        # Direct sparse rows
        sub_data = self.tableau.data[target_row:, unmapped_cols]
        rows, cols = np.nonzero(sub_data)
        if rows.size > 0:
            return (True, unmapped_cols[cols[0]], target_row + rows[0])

        # Batch indicators
        for i, be in enumerate(self.batch):
            brow = self.tableau.dense_size + i
            for col in unmapped_cols:
                if col == be[0]:
                    continue
                if self.tableau.data[brow, col]:
                    return (False, col, brow)
        return None

    def _synthesize_target_row(self, target_row: int, col: int, row: int):
        """Emit Toffoli to create indicator at target_row from a batched row.

        Args:
            target_row: Row index where the indicator bit should be placed.
            col: Column index being processed.
            row: Source row index in the current batch that has the non-zero element.

        """
        batch_idx = row - self.tableau.dense_size
        dense_val = self.batch[batch_idx][1]
        ds = self.tableau.dense_size
        batch_element_bits = [bool((dense_val >> i) & 1) for i in range(ds - 1, -1, -1)]
        is_batch_index = [i == batch_idx for i in range(self.tableau.num_rows - self.tableau.dense_size)]
        combined_idx = np.array(batch_element_bits + is_batch_index, dtype=bool)

        col_data = self.tableau.data[:, col].astype(bool)
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
        """Normalize column's dense/sparse bits and append to batch.

        Args:
            current_col: Column index to normalize and add.
            ctrl_row: Row index of the control qubit for CX corrections.

        """
        dense_size = self.tableau.dense_size
        k_bits = np.array([bool((self.batch_index >> i) & 1) for i in range(dense_size - 1, -1, -1)], dtype=bool)

        dense_col_data = self.tableau.data[:dense_size, current_col].astype(bool)
        for d_qubit in np.flatnonzero(dense_col_data != k_bits):
            self._record((MatrixCompressionType.CX, (ctrl_row, int(d_qubit))))

        sparse_col_data = self.tableau.data[dense_size:, current_col].astype(bool)
        target_bits = np.zeros(self.tableau.num_rows - dense_size, dtype=bool)
        target_bits[len(self.batch)] = True
        for s_qubit in np.flatnonzero(sparse_col_data != target_bits):
            self._record((MatrixCompressionType.CX, (ctrl_row, dense_size + int(s_qubit))))

        self.batch.append((current_col, self.batch_index))
        self.bijection.append((self.batch_index, current_col))
        self.batch_index += 1

    def _clear_sparse_bits(self):
        """Emit PUI block to zero all sparse indicator rows for current batch."""
        assert self.batch
        dense_size = self.tableau.dense_size
        num_changing = 1 if len(self.batch) < 2 else math.ceil(math.log2(len(self.batch)))
        num_fixed = dense_size - num_changing
        k0 = self.batch[0][1]

        fixed_controls = [
            (r, bool((k0 >> (dense_size - 1 - r)) & 1))
            for r in range(num_fixed)
            if self.include_negative_controls or ((k0 >> (dense_size - 1 - r)) & 1)
        ]

        rest_entries = []
        for i, be in enumerate(self.batch):
            changing_controls = [
                (num_fixed + off, bool((be[1] >> (dense_size - 1 - num_fixed - off)) & 1))
                for off in range(num_changing)
            ]
            rest_entries.append((i, changing_controls))

        select_ops: list[tuple[str, Any]] = []
        self._flush_pui_lookup_block(select_ops, dense_size, fixed_controls, rest_entries)
        for op in select_ops:
            self._record(op)

    def _flush_pui_lookup_block(self, ops, sbs, fixed_controls, rest_entries):
        """Convert PUI block into lookup ops, choosing mono vs chunked by cost.

        Args:
            ops: Output list to append generated operations to.
            sbs: Sparse block start row index (equal to dense_size).
            fixed_controls: List of (row, polarity) pairs that are constant across the batch.
            rest_entries: List of (offset, changing_controls) pairs for each batch element.

        """
        if not rest_entries:
            return

        mono_ops, mono_count = self._synthesize_single_pui_lookup_block(sbs, fixed_controls, rest_entries)

        # Split into power-of-two chunks
        n = len(rest_entries)
        if n <= 2:
            ops.extend(mono_ops)
            return

        chunks, i, remaining = [], 0, n
        while remaining > 0:
            chunk_size = 1 << (remaining.bit_length() - 1)
            chunks.append(rest_entries[i : i + chunk_size])
            i += chunk_size
            remaining -= chunk_size

        if len(chunks) <= 1:
            ops.extend(mono_ops)
            return

        chunked_ops, chunked_count = [], 0
        for chunk in chunks:
            sub_ops, sub_count = self._synthesize_single_pui_lookup_block(sbs, fixed_controls, chunk)
            chunked_ops.extend(sub_ops)
            chunked_count += sub_count

        ops.extend(chunked_ops if chunked_count <= mono_count else mono_ops)

    def _synthesize_single_pui_lookup_block(self, sbs, fixed_controls, rest_entries):
        """Lower one PUI sub-block into lookup ops.

        Args:
            sbs: Sparse block start row index (equal to dense_size).
            fixed_controls: List of (row, polarity) pairs that are constant across the chunk.
            rest_entries: List of (offset, changing_controls) pairs for this chunk.

        Returns:
            tuple[list[tuple], int]: A pair of (ops_list, toffoli_cost).

        """
        if not rest_entries:
            return [], 0

        # Canonicalize: promote chunk-local constants to fixed
        n_entries = len(rest_entries)
        fixed_map = dict(fixed_controls)
        row_info: dict[int, tuple[int, set[bool]]] = {}
        for _, changing_controls in rest_entries:
            for row, val in changing_controls:
                count, vals = row_info.get(row, (0, set()))
                vals.add(bool(val))
                row_info[row] = (count + 1, vals)
        for row, (count, values) in row_info.items():
            if count == n_entries and len(values) == 1:
                promoted_val = next(iter(values))
                if row not in fixed_map or fixed_map[row] == promoted_val:
                    fixed_map[row] = promoted_val
        fixed_rows = set(fixed_map)
        fixed_controls = sorted(fixed_map.items())
        rest_entries = [(off, [(r, v) for r, v in ctrls if r not in fixed_rows]) for off, ctrls in rest_entries]

        # Collect address qubits
        all_ctrl_rows = {row for row, _ in fixed_controls}
        for _, changing_controls in rest_entries:
            all_ctrl_rows.update(row for row, _ in changing_controls)
        address_qubits = sorted(all_ctrl_rows)
        data_qubits = [sbs + offset for offset, _ in rest_entries]

        # Build lookup table
        n_outputs = len(rest_entries)
        table: dict[tuple[int, ...], tuple[int, ...]] = {}
        for i, (_, changing_controls) in enumerate(rest_entries):
            ctrl_map = {**dict(fixed_controls), **dict(changing_controls)}
            address = tuple(int(ctrl_map[row]) for row in address_qubits)
            table[address] = tuple(1 if j == i else 0 for j in range(n_outputs))

        if not table:
            return [], 0

        lookup_ops = self._lookup_select(
            table,
            address_qubits=address_qubits,
            data_qubits=data_qubits,
            use_measurement_and=self.measurement_based_uncompute,
        )
        gf2x_ops = list(reversed(lookup_ops))
        toffoli_cost = sum(
            self._scs_toffoli_cost(data_table, root=True)
            for name, (data_table, _, _) in lookup_ops
            if name in (MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND)
        )
        return gf2x_ops, toffoli_cost

    def _to_operations(
        self, num_local_qubits: int, active_qubit_indices: list[int], ancilla_start: int
    ) -> list[MatrixCompressionOp]:
        """Convert internal circuit to reversed MatrixCompressionOp list with global qubit indices.

        Args:
            num_local_qubits: Number of system (local) qubits.
            active_qubit_indices: Mapping from local row indices to global qubit indices.
            ancilla_start: Global qubit index where ancilla qubits begin.

        Returns:
            list[MatrixCompressionOp]: Circuit operations in reversed order with remapped qubit indices.

        """

        def map_idx(idx: int) -> int:
            return (
                int(active_qubit_indices[idx]) if idx < num_local_qubits else ancilla_start + (idx - num_local_qubits)
            )

        ops: list[MatrixCompressionOp] = []
        for compress_type, qubit_args in self.circuit:
            op_type = MatrixCompressionType(compress_type)
            if op_type is MatrixCompressionType.X:
                ops.append(MatrixCompressionOp(op_type, [map_idx(qubit_args[0])]))
            elif op_type in {MatrixCompressionType.CX, MatrixCompressionType.SWAP}:
                ops.append(MatrixCompressionOp(op_type, [map_idx(int(qubit_args[0])), map_idx(int(qubit_args[1]))]))
            elif op_type is MatrixCompressionType.CCX:
                target, ctrl1, ctrl2 = qubit_args
                ops.append(
                    MatrixCompressionOp(op_type, [map_idx(int(ctrl1)), map_idx(int(ctrl2)), map_idx(int(target))])
                )
            elif op_type in {MatrixCompressionType.SELECT, MatrixCompressionType.SELECT_AND}:
                data_table, addr_qubits, dat_qubits = qubit_args
                qubits = [map_idx(int(q)) for q in addr_qubits] + [map_idx(int(q)) for q in dat_qubits]
                ops.append(MatrixCompressionOp(op_type, qubits, control_state=len(addr_qubits), lookup_data=data_table))
            elif op_type is MatrixCompressionType.MCX:
                controls, control_state, target_qubit = qubit_args
                qubits = [map_idx(int(q)) for q in controls] + [map_idx(int(target_qubit))]
                ops.append(MatrixCompressionOp(op_type, qubits, control_state=control_state))
            else:
                raise ValueError(f"Unknown op type: {op_type}")
        ops.reverse()
        return ops

    @staticmethod
    def _scs_toffoli_cost(data: list[list[bool]], *, root: bool = False) -> int:
        """Toffoli cost of SparseOneHotSCS recursion.

        Args:
            data: 2-D boolean lookup table to estimate cost for.
            root: If True, the first binary split is free (no Toffoli needed).

        Returns:
            int: Estimated Toffoli gate count for the SELECT tree.

        """
        n = len(data)
        if n == 0 or all(not any(row) for row in data) or n == 1:
            return 0
        half = 2 ** (math.ceil(math.log2(n)) - 1)
        left, right = data[:half], data[half:]
        left_empty = all(not any(row) for row in left)
        right_empty = all(not any(row) for row in right)
        split_cost = 0 if root else 1
        if not left_empty and not right_empty:
            return (
                split_cost
                + _BinaryEncodingSynthesizer._scs_toffoli_cost(left)
                + _BinaryEncodingSynthesizer._scs_toffoli_cost(right)
            )
        if not right_empty:
            return split_cost + _BinaryEncodingSynthesizer._scs_toffoli_cost(right)
        if not left_empty:
            return split_cost + _BinaryEncodingSynthesizer._scs_toffoli_cost(left)
        return 0

    @staticmethod
    def _lookup_select(table_dict, address_qubits, data_qubits, *, use_measurement_and=False):
        """Build dense lookup table and emit a single SELECT op.

        Args:
            table_dict: Mapping from address tuples to data tuples.
            address_qubits: List of row indices forming the address register.
            data_qubits: List of row indices forming the data register.
            use_measurement_and: If True, emit SELECT_AND instead of SELECT.

        Returns:
            list[tuple]: List of (op_type, (data_table, address_qubits, data_qubits)) tuples.

        """
        if not table_dict:
            return []

        n_address = len(address_qubits)
        n_data = len(data_qubits)
        n_entries = 1 << n_address

        reversed_address = list(reversed(address_qubits))
        data_table: list[list[bool]] = [[False] * n_data for _ in range(n_entries)]
        for addr_tuple, data_tuple in table_dict.items():
            reversed_tuple = tuple(reversed(addr_tuple))
            addr_int = sum(int(bit) << i for i, bit in enumerate(reversed_tuple))
            data_table[addr_int] = [bool(b) for b in data_tuple]

        op_type = MatrixCompressionType.SELECT_AND if use_measurement_and else MatrixCompressionType.SELECT
        return [(op_type, (data_table, reversed_address, list(data_qubits)))]
