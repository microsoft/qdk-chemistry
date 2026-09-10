"""Sparse Pauli terms and product formulas with storage proportional to non-identity factors."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import operator
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, cast, overload

import numpy as np
from scipy.sparse import csr_matrix, vstack

from qdk_chemistry._core.data import sparse_pauli_word_to_label
from qdk_chemistry.data._hashing import _hash_array, _hash_float, _hash_int, _hash_str

from .pauli_product_formula import ExponentiatedPauliTerm, PauliProductFormulaContainer

__all__ = ["SparsePauliProductFormulaContainer", "SparsePauliTerms"]


@dataclass(frozen=True, eq=False)
class SparsePauliTerms(Sequence[str]):
    """Owned sparse factors with lazy, qubit-zero-rightmost labels.

    Arrays contain uint64 term boundaries, uint32 qubit indices, and uint8 codes
    1/2/3 for X/Y/Z. Indices are sorted and unique within each term; an empty row
    represents identity. Accessing a label expands only that row to register width.
    """

    num_qubits: int
    term_offsets: np.ndarray
    qubit_indices: np.ndarray
    pauli_codes: np.ndarray
    array_names: ClassVar = ("term_offsets", "qubit_indices", "pauli_codes")

    def __post_init__(self) -> None:
        """Validate before narrowing integer inputs, then copy and freeze the arrays."""
        if (
            isinstance(self.num_qubits, bool | np.bool_)
            or not isinstance(self.num_qubits, int | np.integer)
            or not 1 <= self.num_qubits <= 2**32
        ):
            raise ValueError("num_qubits must be an integer in 1..2**32.")
        object.__setattr__(self, "num_qubits", int(self.num_qubits))
        for name, dtype in zip(self.array_names, (np.uint64, np.uint32, np.uint8), strict=True):
            raw = getattr(self, name)
            values = np.asarray(raw) if isinstance(raw, np.ndarray) else np.asarray(raw, dtype=object)
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            maximum = int(np.iinfo(dtype).max)
            if values.dtype.kind == "O":
                if any(isinstance(v, bool | np.bool_) or not isinstance(v, int | np.integer) for v in values):
                    raise ValueError(f"{name} must contain integers, not booleans or fractions.")
            elif values.dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integers, not booleans or fractions.")
            if values.size and (int(values.min()) < 0 or int(values.max()) > maximum):
                raise ValueError(f"{name} contains values outside the {np.dtype(dtype).name} range.")
            owned = np.array(values, dtype=dtype, copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, name, owned)
        offsets, indices, codes = self.arrays()
        if not len(offsets) or len(indices) != len(codes):
            raise ValueError("Sparse Pauli array lengths are inconsistent.")
        if offsets[0] != 0 or offsets[-1] != len(indices) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("term_offsets must span the factors in nondecreasing order.")
        if indices.size and int(indices.max()) >= self.num_qubits:
            raise ValueError("Sparse Pauli qubit indices must be in range.")
        if np.any((codes < 1) | (codes > 3)):
            raise ValueError("Sparse Pauli codes must be in 1..3 (X, Y, Z).")
        if not self.to_csr().has_canonical_format:
            raise ValueError("Sparse Pauli qubit indices must be sorted and unique within each term.")

    @classmethod
    def from_terms(
        cls, num_qubits: int, terms: Iterable[Mapping[int, str] | Iterable[tuple[int, str]]]
    ) -> SparsePauliTerms:
        """Pack non-identity factors, sorting qubits while retaining term order and duplicates."""
        try:
            arrays = pack_pauli_terms(
                sorted(term.items() if isinstance(term, Mapping) else term, key=lambda factor: factor[0])
                for term in terms
            )
            return cls(num_qubits, *arrays)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid sparse Pauli factors.") from error

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return read-only offsets, indices, and codes without copying."""
        return self.term_offsets, self.qubit_indices, self.pauli_codes

    def to_csr(self) -> csr_matrix:
        """Expose factors as sparse rows for selection, never as an operator matrix."""
        return csr_matrix((self.pauli_codes, self.qubit_indices, self.term_offsets), shape=(len(self), self.num_qubits))

    def factors(self, index: int) -> tuple[tuple[int, str], ...]:
        """Return one term's non-identity factors without constructing a label."""
        index = range(len(self))[index]
        begin, end = self.term_offsets[index : index + 2]
        return tuple(
            (int(q), "IXYZ"[int(p)])
            for q, p in zip(self.qubit_indices[begin:end], self.pauli_codes[begin:end], strict=True)
        )

    def __len__(self) -> int:
        """Count terms, including identity rows."""
        return len(self.term_offsets) - 1

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        """Materialize only the requested labels."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = range(len(self))[index]
        begin, end = self.term_offsets[index : index + 2]
        return sparse_pauli_word_to_label(
            list(zip(self.qubit_indices[begin:end], self.pauli_codes[begin:end], strict=True)), self.num_qubits
        )

    def __eq__(self, other: object) -> bool:
        """Compare labels with another sequence."""
        return (
            isinstance(other, Sequence)
            and len(self) == len(other)
            and all(a == b for a, b in zip(self, other, strict=True))
        )

    __hash__ = None  # type: ignore[assignment]


def pack_pauli_terms(terms: Iterable[Iterable[tuple[int, str]]]) -> tuple[list[int], list[int], list[int]]:
    """Pack factor iterables; callers choose sorting, validation, and identity policy."""
    offsets, indices, codes = [0], [], []
    for term in terms:
        for qubit, axis in term:
            indices.append(qubit)
            codes.append(("I", "X", "Y", "Z").index(axis))
        offsets.append(len(indices))
    return offsets, indices, codes


def combine_sparse(
    left: PauliProductFormulaContainer, right: PauliProductFormulaContainer, atol: float
) -> SparsePauliProductFormulaContainer:
    """Fuse adjacent sparse rows after the common register/scale checks, without unpacking term objects."""
    values: list[Sequence[float]] = []
    tables = []
    for container in (left, right):
        if isinstance(container, SparsePauliProductFormulaContainer):
            tables.append(container.pauli_terms.to_csr())
            values.append(memoryview(container.sparse_term_arrays()[3]))
        else:
            offsets, indices, codes = pack_pauli_terms(sorted(term.pauli_term.items()) for term in container.step_terms)
            tables.append(
                csr_matrix(
                    (np.asarray(codes, dtype=np.uint8), np.asarray(indices, dtype=np.uint32), offsets),
                    shape=(len(container.step_terms), left.num_qubits),
                )
            )
            values.append([term.angle for term in container.step_terms])
    factors = vstack(tables, format="csr")
    boundaries, sites, axes = (memoryview(v) for v in (factors.indptr, factors.indices, factors.data))
    kept, angles = array("q"), array("d")
    offset = 0
    for container, source_angles in zip((left, right), values, strict=True):
        for _ in range(container.step_reps):
            for row, angle in enumerate(source_angles, offset):
                current = slice(boundaries[row], boundaries[row + 1])
                previous = slice(boundaries[kept[-1]], boundaries[kept[-1] + 1]) if kept else slice(0, 0)
                if kept and sites[current] == sites[previous] and axes[current] == axes[previous]:
                    angles[-1] += angle
                    if abs(angles[-1]) <= atol:
                        angles.pop()
                        kept.pop()
                else:
                    angles.append(angle)
                    kept.append(row)
        offset += len(source_angles)
    result = factors[kept]
    return SparsePauliProductFormulaContainer(
        result.indptr,
        result.indices,
        result.data,
        np.asarray(angles),
        step_reps=1,
        num_qubits=left.num_qubits,
        scale=left.scale,
    )


class SparsePauliProductFormulaContainer(PauliProductFormulaContainer):
    """An ordered sparse product formula with lazy compatibility access to exponentiated terms.

    Uses the existing product-formula wire type with compact version 0.3.0.
    Repetitions remain symbolic; only explicit term access creates dictionaries.
    """

    _serialization_version = "0.3.0"
    has_sparse_terms = True
    array_names = (*SparsePauliTerms.array_names, "angles")
    count = Sequence.count
    index = Sequence.index

    def __init__(
        self,
        term_offsets: np.ndarray,
        qubit_indices: np.ndarray,
        pauli_codes: np.ndarray,
        angles: np.ndarray,
        *,
        step_reps: int,
        num_qubits: int,
        scale: float = 1.0,
    ) -> None:
        """Own sparse factors and finite real angles; retain identity and empty formulas.

        Args:
            term_offsets: Row boundaries, one more than the number of angles; equal boundaries encode identity.
            qubit_indices: Sorted unique non-identity qubit indices within each row, in the uint32 range.
            pauli_codes: Axis codes 1/2/3 for X/Y/Z, aligned with qubit indices.
            angles: One finite real rotation angle per row; may be empty.
            step_reps: Positive integer repetition count, not expanded during construction.
            num_qubits: Register width in 1..2**32.
            scale: Evolution time used for eigenvalue-phase conversion.

        """
        values = np.asarray(angles)
        if values.ndim != 1:
            raise ValueError("angles must be a one-dimensional array.")
        if values.dtype.kind not in "iuf":
            raise TypeError("angles must contain real numbers.")
        self._angles = np.frombuffer(values.astype(np.float64, copy=False).tobytes(), dtype=np.float64)
        if not np.all(np.isfinite(self._angles)):
            raise ValueError("angles must be finite.")
        self.pauli_terms = SparsePauliTerms(num_qubits, term_offsets, qubit_indices, pauli_codes)
        if len(self.pauli_terms) != len(self._angles):
            raise ValueError("Sparse Pauli term_offsets must contain one boundary per angle plus one.")
        super().__init__(cast("Sequence[ExponentiatedPauliTerm]", self), step_reps, int(num_qubits), float(scale))

    def sparse_term_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return read-only factor arrays followed by the rotation angles."""
        return *self.pauli_terms.arrays(), self._angles

    def __len__(self) -> int:
        """Count exponentiated factors in one step."""
        return len(self._angles)

    @overload
    def __getitem__(self, index: int) -> ExponentiatedPauliTerm: ...

    @overload
    def __getitem__(self, index: slice) -> list[ExponentiatedPauliTerm]: ...

    def __getitem__(self, index: int | slice) -> ExponentiatedPauliTerm | list[ExponentiatedPauliTerm]:
        """Materialize only the requested term objects."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = range(len(self))[index]
        return ExponentiatedPauliTerm(dict(self.pauli_terms.factors(index)), float(self._angles[index]))

    def __iter__(self) -> Iterator[ExponentiatedPauliTerm]:
        """Iterate terms lazily for legacy consumers."""
        return (self[i] for i in range(len(self)))

    def _hash_update(self, h) -> None:
        _hash_str(h, "pauli_product_formula")
        for values in self.sparse_term_arrays():
            _hash_array(h, values)
        _hash_int(h, self.step_reps)
        _hash_int(h, self.num_qubits)
        _hash_float(h, self.scale)

    def reorder_terms(self, permutation: list[int]) -> SparsePauliProductFormulaContainer:
        """Select sparse rows in the requested order, retaining scale and repetitions."""
        self._validate_permutation(permutation)
        permutation = [operator.index(index) for index in permutation]
        terms = self.pauli_terms.to_csr()[permutation]
        return type(self)(
            terms.indptr,
            terms.indices,
            terms.data,
            self._angles[permutation],
            step_reps=self.step_reps,
            num_qubits=self.num_qubits,
            scale=self.scale,
        )

    def to_json(self) -> dict[str, Any]:
        """Serialize compact factors without expanding term dictionaries."""
        data = {name: values.tolist() for name, values in zip(self.array_names, self.sparse_term_arrays(), strict=True)}
        return self._add_json_version(
            dict(data, container_type=self.type, step_reps=self.step_reps, num_qubits=self.num_qubits, scale=self.scale)
        )

    def to_hdf5(self, group) -> None:
        """Write compact factors and scalar metadata to an HDF5 group."""
        group.attrs.update(
            container_type=self.type, step_reps=self.step_reps, num_qubits=self.num_qubits, scale=self.scale
        )
        self._add_hdf5_version(group)
        for name, values in zip(self.array_names, self.sparse_term_arrays(), strict=True):
            group.create_dataset(name, data=values)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SparsePauliProductFormulaContainer:
        """Restore compact arrays with validation before integer conversion."""
        cls._validate_json_version(cls._serialization_version, data)
        arrays = (data[name] for name in cls.array_names)
        return cls(*arrays, step_reps=data["step_reps"], num_qubits=data["num_qubits"], scale=data.get("scale", 1.0))

    @classmethod
    def from_hdf5(cls, group) -> SparsePauliProductFormulaContainer:
        """Restore compact HDF5 data through the validated constructor."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        arrays = (np.asarray(group[name]) for name in cls.array_names)
        return cls(
            *arrays,
            step_reps=group.attrs["step_reps"],
            num_qubits=group.attrs["num_qubits"],
            scale=group.attrs.get("scale", 1.0),
        )


Sequence.register(SparsePauliProductFormulaContainer)
