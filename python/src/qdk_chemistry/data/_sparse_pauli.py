"""Private storage and validation for packed non-identity Pauli factors."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import overload

import numpy as np

_PAULI_CODES = {"X": 1, "Y": 2, "Z": 3}
_PAULI_CHARS = "IXYZ"


def _validate_sparse_pauli_arrays(
    num_qubits: int,
    term_offsets: np.ndarray,
    qubit_indices: np.ndarray,
    pauli_codes: np.ndarray,
    num_terms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate packed terms and return owned, read-only uint64/uint32/uint8 arrays.

    Each term has sorted, unique qubit indices and codes in 1..3. Empty terms
    represent identity. Zero terms are allowed here, but the register width
    must be in 1..2**32. Raw sequences retain their types until validation so
    mixed booleans, fractions, and overflowing integers cannot be narrowed.
    """
    if (
        isinstance(num_qubits, bool | np.bool_)
        or not isinstance(num_qubits, int | np.integer)
        or not 1 <= num_qubits <= 2**32
    ):
        raise ValueError("num_qubits must be an integer in 1..2**32.")
    if (
        isinstance(num_terms, bool | np.bool_)
        or not isinstance(num_terms, int | np.integer)
        or not 0 <= num_terms < np.iinfo(np.intp).max
    ):
        raise ValueError("num_terms must be a non-negative integer representable as an array length.")

    arrays = []
    for name, raw_values, dtype in (
        ("term_offsets", term_offsets, np.uint64),
        ("qubit_indices", qubit_indices, np.uint32),
        ("pauli_codes", pauli_codes, np.uint8),
    ):
        # Object conversion of raw sequences preserves e.g. [0, True] from JSON.
        values = np.asarray(raw_values) if isinstance(raw_values, np.ndarray) else np.asarray(raw_values, dtype=object)
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        maximum = int(np.iinfo(dtype).max)
        if values.dtype.kind == "O":
            if any(isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer) for value in values):
                raise ValueError(f"{name} must contain integers, not booleans or fractions.")
            if any(value < 0 or value > maximum for value in values):
                raise ValueError(f"{name} contains values outside the {np.dtype(dtype).name} range.")
        elif values.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integers, not booleans or fractions.")
        elif values.size and (int(values.min()) < 0 or int(values.max()) > maximum):
            raise ValueError(f"{name} contains values outside the {np.dtype(dtype).name} range.")
        arrays.append(np.array(values, dtype=dtype, copy=True))

    offsets, indices, codes = arrays
    if len(offsets) != num_terms + 1 or len(indices) != len(codes):
        raise ValueError("Packed sparse Pauli array lengths do not match the number of terms and factors.")
    if (
        offsets[0] != 0
        or offsets[-1] != len(indices)
        or np.any(offsets > len(indices))
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("term_offsets must start at zero, be nondecreasing, and end at the number of factors.")
    if indices.size and int(indices.max()) >= int(num_qubits):
        raise ValueError("Sparse Pauli qubit indices must be in range.")
    if np.any((codes < 1) | (codes > 3)):
        raise ValueError("Sparse Pauli codes must be in 1..3 (X, Y, Z).")
    if len(indices) > 1:
        unordered = indices[1:] <= indices[:-1]
        boundaries = offsets[1:-1]
        boundaries = boundaries[(boundaries > 0) & (boundaries < len(indices))]
        unordered[boundaries - 1] = False
        if np.any(unordered):
            raise ValueError("Sparse Pauli qubit indices must be sorted and unique within each term.")

    for values in arrays:
        values.flags.writeable = False
    return offsets, indices, codes


class _SparsePauliStrings(Sequence[str]):
    """Compatibility view that materializes full-width labels only on access."""

    def __init__(
        self,
        num_qubits: int,
        term_offsets: np.ndarray,
        qubit_indices: np.ndarray,
        pauli_codes: np.ndarray,
    ) -> None:
        self._num_qubits = num_qubits
        self._term_offsets = term_offsets
        self._qubit_indices = qubit_indices
        self._pauli_codes = pauli_codes

    def __len__(self) -> int:
        return len(self._term_offsets) - 1

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Pauli term index out of range")
        label = ["I"] * self._num_qubits
        begin, end = int(self._term_offsets[index]), int(self._term_offsets[index + 1])
        for position in range(begin, end):
            qubit = int(self._qubit_indices[position])
            label[self._num_qubits - 1 - qubit] = _PAULI_CHARS[int(self._pauli_codes[position])]
        return "".join(label)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return NotImplemented
        return len(self) == len(other) and all(left == right for left, right in zip(self, other, strict=True))

    __hash__ = None  # type: ignore[assignment]  # Standard unhashable-sequence protocol.
