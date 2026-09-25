"""Sparse Pauli decomposition qubit operator container."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from qdk_chemistry._core.data import TaperingSpecification, label_to_sparse_pauli_word, sparse_pauli_word_to_label
from qdk_chemistry.data._hashing import _hash_arg, _hash_array, _hash_optional, _hash_str, _hash_uint
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.data.qubit_operator.containers.base import QubitOperatorContainer
from qdk_chemistry.data.qubit_operator.containers.pauli_decomposition import (
    PauliDecompositionContainer,
    _hash_tapering,
    _merge_term_partitions,
)
from qdk_chemistry.data.term_partition import TermPartition

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import h5py

__all__ = ["SparsePauliDecompositionContainer", "SparsePauliTerms"]


@dataclass(frozen=True, eq=False, init=False)
class SparsePauliTerms(Sequence[str]):
    """Immutable non-identity words with lazy, qubit-zero-rightmost labels.

    ``words`` retains ordered terms, duplicates, and empty identity words. Each
    word is a sorted tuple of ``(qubit, X/Y/Z)`` pairs; only label access expands
    to register width. Storage is proportional to the non-identity factors.
    """

    num_qubits: int
    words: tuple[tuple[tuple[int, str], ...], ...]

    def __init__(self, num_qubits: int, terms: Iterable[Mapping[int, str] | Iterable[tuple[int, str]]]) -> None:
        """Copy and validate sparse words without combining terms or constructing dense labels."""
        if (
            isinstance(num_qubits, bool | np.bool_)
            or not isinstance(num_qubits, int | np.integer)
            or not 1 <= num_qubits <= 2**32
        ):
            raise ValueError("num_qubits must be an integer in 1..2**32.")
        words = []
        for term in terms:
            word = []
            for qubit, axis in term.items() if isinstance(term, Mapping) else term:
                if (
                    isinstance(qubit, bool | np.bool_)
                    or not isinstance(qubit, int | np.integer)
                    or not 0 <= qubit < num_qubits
                    or axis not in ("X", "Y", "Z")
                ):
                    raise ValueError("Sparse Pauli factors require in-range integer indices and X/Y/Z axes.")
                word.append((int(qubit), axis))
            word.sort()
            if len({q for q, _ in word}) != len(word):
                raise ValueError("Sparse Pauli factors must have unique qubit indices.")
            words.append(tuple(word))
        object.__setattr__(self, "num_qubits", int(num_qubits))
        object.__setattr__(self, "words", tuple(words))

    def factors(self, index: int) -> tuple[tuple[int, str], ...]:
        """Access one word without materializing a label."""
        return self.words[index]

    def __len__(self) -> int:
        """Count terms, including identities and duplicates."""
        return len(self.words)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        """Materialize only the requested labels."""
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return sparse_pauli_word_to_label([(q, "IXYZ".index(axis)) for q, axis in self.words[index]], self.num_qubits)

    def __eq__(self, other: object) -> bool:
        """Compare sparse words directly, or labels with a legacy sequence."""
        if isinstance(other, SparsePauliTerms):
            return self.num_qubits == other.num_qubits and self.words == other.words
        return (
            isinstance(other, Sequence)
            and len(self) == len(other)
            and all(a == b for a, b in zip(self, other, strict=True))
        )

    __hash__ = None  # type: ignore[assignment]


class SparsePauliDecompositionContainer(PauliDecompositionContainer):
    """Pauli decomposition that stores each term as its non-identity factors.

    Storage, hashing, serialization and operations with this container on the left scale with
    the number of non-identity factors rather than the register width. :attr:`pauli_strings`
    is a lazy :class:`SparsePauliTerms` view: indexing or iterating it
    builds full-width labels, which :meth:`iter_sparse_terms` avoids. A dense left operand
    uses the dense implementation and returns dense storage. Coefficients are copied and
    read-only. Dense and sparse content hashes differ for equal operators; use :meth:`equiv`
    to compare Pauli expansions.
    """

    _data_type_name = "sparse_pauli_decomposition_container"

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for sparse Pauli decomposition containers.

        Returns:
            ``"sparse_pauli_decomposition_container"``.

        """
        return "sparse_pauli_decomposition_container"

    _serialization_version = "0.1.0"

    def __init__(
        self,
        pauli_strings: SparsePauliTerms,
        coefficients: np.ndarray,
        encoding: str | None = None,
        fermion_mode_order: FermionModeOrder | str | None = None,
        term_partition: TermPartition | None = None,
        tapering: TaperingSpecification | None = None,
    ) -> None:
        """Initialize from sparse terms and one coefficient per term.

        Args:
            pauli_strings: Ordered sparse terms; an empty word encodes identity.
            coefficients: One-dimensional numeric array with one coefficient per term.
            encoding: Fermion-to-qubit encoding (e.g., ``"jordan-wigner"``). Default ``None``.
            fermion_mode_order: Mode ordering (``"blocked"``/``"interleaved"``). Default ``None``.
            term_partition: Optional ``TermPartition`` carrying group/layer metadata.
            tapering: Applied tapering metadata, or None if untapered.

        Raises:
            TypeError: If *pauli_strings* is not :class:`SparsePauliTerms`.
            ValueError: If the coefficients do not match the terms or the partition does not cover them.

        """
        if not isinstance(pauli_strings, SparsePauliTerms):
            raise TypeError("SparsePauliDecompositionContainer requires SparsePauliTerms; use from_sparse_terms.")
        coefficients = np.array(coefficients, copy=True)
        if coefficients.ndim != 1 or coefficients.dtype.kind not in "biufc":
            raise ValueError("Sparse Pauli coefficients must be a one-dimensional numeric array.")
        if not len(coefficients):
            raise ValueError("Sparse Pauli terms cannot be empty.")
        if len(pauli_strings) != len(coefficients):
            raise ValueError("Mismatch between number of Pauli strings and coefficients.")
        if term_partition is not None and sorted(term_partition.all_indices()) != list(range(len(coefficients))):
            raise ValueError("term_partition must cover every term exactly once.")
        coefficients.flags.writeable = False
        self.pauli_strings = pauli_strings
        self.coefficients = coefficients
        self.term_partition: TermPartition | None = term_partition
        self.tapering: TaperingSpecification | None = tapering
        # Skip the dense initializer, whose label validation would build every full-width label.
        QubitOperatorContainer.__init__(self, encoding, fermion_mode_order)

    @classmethod
    def from_sparse_terms(
        cls,
        num_qubits: int,
        terms: Iterable[Mapping[int, str] | Iterable[tuple[int, str]]],
        coefficients: np.ndarray,
        *,
        encoding: str | None = None,
        fermion_mode_order: FermionModeOrder | str | None = None,
        term_partition: TermPartition | None = None,
        tapering: TaperingSpecification | None = None,
    ) -> SparsePauliDecompositionContainer:
        """Construct from mappings or iterables of ``(qubit, X/Y/Z)`` pairs.

        Empty terms encode identity. Factors are sorted; term order and duplicates are retained.

        Args:
            num_qubits: Register width.
            terms: One mapping or iterable of ``(qubit, axis)`` pairs per term.
            coefficients: One coefficient per term.
            encoding: Fermion-to-qubit encoding (e.g., ``"jordan-wigner"``). Default ``None``.
            fermion_mode_order: Mode ordering (``"blocked"``/``"interleaved"``). Default ``None``.
            term_partition: Optional ``TermPartition`` carrying group/layer metadata.
            tapering: Applied tapering metadata, or None if untapered.

        Returns:
            A new container owning copies of the terms and coefficients.

        """
        return cls(
            SparsePauliTerms(num_qubits, terms), coefficients, encoding, fermion_mode_order, term_partition, tapering
        )

    @property
    def type(self) -> str:
        """Return the container type."""
        return "sparse_pauli_decomposition"

    @property
    def num_qubits(self) -> int:
        """Return the register width."""
        return self.pauli_strings.num_qubits

    def iter_sparse_terms(self) -> Iterator[tuple[tuple[tuple[int, str], ...], complex]]:
        """Iterate over terms as sorted non-identity factors with their coefficients, without building labels.

        Yields:
            Pairs of ``((qubit_index, Pauli), ...)`` factors and complex coefficients; identity has no factors.

        """
        for word, coefficient in zip(self.pauli_strings.words, self.coefficients, strict=True):
            yield word, complex(coefficient)

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher, hashing the width and sparse words instead of labels."""
        _hash_str(h, "qubit_hamiltonian")
        _hash_str(h, "sparse_pauli")
        _hash_uint(h, self.num_qubits)
        _hash_arg(h, self.pauli_strings.words)
        _hash_array(h, self.coefficients)
        _hash_optional(h, self.encoding, _hash_str)
        _hash_optional(h, self.fermion_mode_order, lambda h, mode: _hash_str(h, str(mode)))
        _hash_optional(h, self.term_partition, lambda h, partition: _hash_str(h, partition.content_hash(0)))
        _hash_optional(h, self.tapering, _hash_tapering)

    def equiv(self, other: PauliDecompositionContainer, atol: float = 1e-12) -> bool:
        """Check equivalence by summing coefficients per non-identity word, without building labels.

        Args:
            other: The Pauli decomposition to compare against.
            atol: Absolute tolerance for coefficient comparison. Defaults to 1e-12.

        Returns:
            ``True`` if both operators have the same register width and Pauli expansion.

        """
        if not isinstance(other, PauliDecompositionContainer) or self.num_qubits != other.num_qubits:
            return False
        difference: dict[tuple[tuple[int, str], ...], complex] = {}
        for sign, operator in ((1, self), (-1, other)):
            for word, coefficient in _sparse_terms(operator):
                difference[word] = difference.get(word, 0) + sign * coefficient
        return all(abs(value) <= atol for value in difference.values())

    def __add__(self, other: PauliDecompositionContainer) -> SparsePauliDecompositionContainer:
        """Return the sum with sparse storage, under the dense container's metadata rules.

        Args:
            other: The Pauli decomposition to add.

        Returns:
            A new sparse container with concatenated terms, or ``NotImplemented`` for other representations.

        Raises:
            ValueError: If the two operators have different qubit counts, encodings, modes, or tapering.

        """
        if not isinstance(other, PauliDecompositionContainer):
            return NotImplemented
        if self.num_qubits != other.num_qubits:
            raise ValueError(f"Cannot add operators with {self.num_qubits} and {other.num_qubits} qubits.")
        if self.encoding != other.encoding:
            raise ValueError(f"Cannot add operators with different encodings: {self.encoding!r} vs {other.encoding!r}.")
        if self.fermion_mode_order != other.fermion_mode_order:
            raise ValueError(
                f"Cannot add operators with different fermion_mode_order: "
                f"{self.fermion_mode_order!r} vs {other.fermion_mode_order!r}."
            )
        # The bound TaperingSpecification.__eq__ rejects None, so compare presence first.
        if (self.tapering is None) != (other.tapering is None) or (
            self.tapering is not None and self.tapering != other.tapering
        ):
            raise ValueError(f"Cannot add operators with different tapering: {self.tapering!r} vs {other.tapering!r}.")
        partition = None
        if self.term_partition is not None and other.term_partition is not None:
            partition = _merge_term_partitions(self.term_partition, other.term_partition)
        return SparsePauliDecompositionContainer(
            SparsePauliTerms(
                self.num_qubits, (word for operator in (self, other) for word, _ in _sparse_terms(operator))
            ),
            np.concatenate([self.coefficients, other.coefficients]),
            self.encoding,
            self.fermion_mode_order,
            partition,
            self.tapering,
        )

    def __mul__(self, scalar) -> SparsePauliDecompositionContainer:
        """Return the operator with scaled coefficients, sharing its immutable sparse terms.

        Args:
            scalar: The scalar multiplier.

        Returns:
            A new sparse container, or ``NotImplemented`` for non-scalar operands.

        """
        if not isinstance(scalar, int | float | complex | np.number):
            return NotImplemented
        return SparsePauliDecompositionContainer(
            self.pauli_strings,
            self.coefficients * scalar,
            self.encoding,
            self.fermion_mode_order,
            self.term_partition,
            self.tapering,
        )

    def get_real_coefficients(
        self, tolerance: float = 1e-12, sort_by_magnitude: bool = False
    ) -> list[tuple[str, float]]:
        """Return ``(label, real_coeff)`` pairs, building labels only for the retained terms.

        Args:
            tolerance: Threshold for filtering small real coefficients. Defaults to 1e-12.
            sort_by_magnitude: If ``True``, sort by descending ``|coefficient|``. Defaults to ``False``.

        Returns:
            List of ``(pauli_label, coefficient)`` tuples.

        """
        terms = [
            (self.pauli_strings[index], real)
            for index, real in enumerate(complex(coefficient).real for coefficient in self.coefficients)
            if abs(real) > tolerance
        ]
        if sort_by_magnitude:
            terms.sort(key=lambda t: abs(t[1]), reverse=True)
        return terms

    def to_interleaved(self, n_spatial: int) -> SparsePauliDecompositionContainer:
        """Remap alpha qubit ``q`` to ``2q`` and beta qubit ``q`` to ``2(q - n_spatial) + 1``.

        Args:
            n_spatial (int): The number of spatial orbitals; the register must hold ``2 * n_spatial`` qubits.

        Returns:
            A new sparse container in interleaved order, without a term partition.

        Raises:
            ValueError: If num_qubits != 2 * n_spatial.

        """
        if self.num_qubits != 2 * n_spatial:
            raise ValueError(f"Number of qubits ({self.num_qubits}) must be 2 * n_spatial ({2 * n_spatial}).")
        terms = SparsePauliTerms(
            self.num_qubits,
            (
                ((2 * q if q < n_spatial else 2 * (q - n_spatial) + 1, axis) for q, axis in word)
                for word in self.pauli_strings.words
            ),
        )
        return SparsePauliDecompositionContainer(
            terms, self.coefficients.copy(), self.encoding, FermionModeOrder.INTERLEAVED, tapering=self.tapering
        )

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON, storing the register width and each term's ``[qubit, axis]`` factors.

        Returns:
            dict[str, Any]: Dictionary representation of the operator.

        """
        data = super().to_json()
        # The dense serializer only references the label view; replace it with the factors.
        del data["pauli_strings"]
        data["num_qubits"] = self.num_qubits
        data["pauli_terms"] = [[list(factor) for factor in word] for word in self.pauli_strings.words]
        return data

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save factors as compressed-row arrays: per-term offsets, qubit indices and X/Y/Z codes 1-3.

        Args:
            group (h5py.Group): HDF5 group or file to write the operator to.

        """
        words = self.pauli_strings.words
        offsets = np.zeros(len(words) + 1, dtype=np.int64)
        np.cumsum([len(word) for word in words], out=offsets[1:])
        self._add_hdf5_version(group)
        group.attrs["container_type"] = self.type
        group.attrs["num_qubits"] = self.num_qubits
        group.create_dataset("term_offsets", data=offsets)
        group.create_dataset(
            "qubit_indices",
            data=np.fromiter((q for word in words for q, _ in word), dtype=np.uint32, count=offsets[-1]),
        )
        group.create_dataset(
            "pauli_codes",
            data=np.fromiter(
                ("IXYZ".index(axis) for word in words for _, axis in word), dtype=np.uint8, count=offsets[-1]
            ),
        )
        group.create_dataset("coefficients", data=self.coefficients)
        if self.encoding is not None:
            group.attrs["encoding"] = self.encoding
        if self.fermion_mode_order is not None:
            group.attrs["fermion_mode_order"] = str(self.fermion_mode_order)
        if self.term_partition is not None:
            group.attrs["term_partition"] = json.dumps(self.term_partition.to_json())
        if self.tapering is not None:
            group.attrs["tapering"] = json.dumps(self.tapering.to_json())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> SparsePauliDecompositionContainer:
        """Create a sparse container from JSON written by :meth:`to_json`.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized data.

        Returns:
            SparsePauliDecompositionContainer: New instance reconstructed from JSON data.

        Raises:
            RuntimeError: If the version field is missing or incompatible.
            ValueError: If the real and imaginary coefficient arrays have different shapes.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        coefficients = json_data["coefficients"]
        if isinstance(coefficients, dict) and np.shape(coefficients["real"]) != np.shape(coefficients["imag"]):
            raise ValueError("Sparse Pauli real and imaginary coefficient arrays must have matching shapes.")
        terms = SparsePauliTerms(json_data["num_qubits"], json_data["pauli_terms"])
        return super().from_json({**json_data, "pauli_strings": terms})

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> SparsePauliDecompositionContainer:
        """Load a sparse container from HDF5 written by :meth:`to_hdf5`, rejecting malformed factor arrays.

        Args:
            group (h5py.Group): HDF5 group or file containing the data.

        Returns:
            SparsePauliDecompositionContainer: New instance reconstructed from HDF5 data.

        Raises:
            RuntimeError: If the version attribute is missing or incompatible.
            ValueError: If the factor arrays are malformed.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        offsets, qubits, codes = (np.asarray(group[name]) for name in ("term_offsets", "qubit_indices", "pauli_codes"))
        if (
            any(array.ndim != 1 or array.dtype.kind not in "iu" for array in (offsets, qubits, codes))
            or not offsets.size
            or offsets[0] != 0
            or offsets[-1] != len(qubits)
            or len(codes) != len(qubits)
            or np.any(offsets[1:] < offsets[:-1])
            or np.any((codes < 1) | (codes > 3))
        ):
            raise ValueError("Invalid sparse Pauli term arrays.")
        qubit_list = qubits.tolist()
        axes = ["IXYZ"[code] for code in codes.tolist()]
        terms = SparsePauliTerms(
            int(group.attrs["num_qubits"]),
            (zip(qubit_list[begin:end], axes[begin:end], strict=True) for begin, end in pairwise(offsets.tolist())),
        )
        partition, tapering = (_decoded(group.attrs.get(name)) for name in ("term_partition", "tapering"))
        return cls(
            terms,
            np.array(group["coefficients"]),
            _decoded(group.attrs.get("encoding")),
            _decoded(group.attrs.get("fermion_mode_order")),
            None if partition is None else TermPartition.from_json(json.loads(partition)),
            None if tapering is None else TaperingSpecification.from_json(json.loads(tapering)),
        )


def _sparse_terms(operator: PauliDecompositionContainer) -> Iterator[tuple[tuple[tuple[int, str], ...], complex]]:
    """Yield sorted non-identity factors and coefficients, converting dense labels when needed."""
    if isinstance(operator, SparsePauliDecompositionContainer):
        yield from operator.iter_sparse_terms()
        return
    for label, coefficient in zip(operator.pauli_strings, operator.coefficients, strict=True):
        yield tuple((q, "IXYZ"[code]) for q, code in label_to_sparse_pauli_word(label)), complex(coefficient)


def _decoded(value: Any) -> Any:
    """Decode an HDF5 string attribute that h5py may return as bytes."""
    return value.decode("utf-8") if isinstance(value, bytes) else value
