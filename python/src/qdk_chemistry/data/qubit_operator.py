"""QDK/Chemistry Qubit Operator module.

This module provides the ``QubitOperator`` dataclass: a general operator on qubits expressed
as a weighted sum of Pauli strings.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import re
import warnings
from array import array
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from qdk_chemistry.data._hashing import _hash_arg, _hash_array, _hash_optional, _hash_str, _hash_uint
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.term_partition import FlatPartition, LayeredPartition, TermPartition
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix, pauli_to_sparse_matrix

if TYPE_CHECKING:
    import h5py
    import scipy

from qdk_chemistry._core.data import TaperingSpecification
from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.utils import Logger

__all__: list[str] = []

_PAULI_CODES = {"X": 1, "Y": 2, "Z": 3}
_PAULI_CHARS = "IXYZ"


class _SparsePauliStrings(Sequence[str]):
    """Lazy full-width labels backed by packed non-identity factors."""

    def __init__(
        self,
        num_qubits: int,
        term_offsets: np.ndarray,
        qubit_indices: np.ndarray,
        pauli_codes: np.ndarray,
    ) -> None:
        self.num_qubits = num_qubits
        self.term_offsets = term_offsets
        self.qubit_indices = qubit_indices
        self.pauli_codes = pauli_codes

    def __len__(self) -> int:
        return len(self.term_offsets) - 1

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Pauli term index out of range")
        label = ["I"] * self.num_qubits
        begin = int(self.term_offsets[index])
        end = int(self.term_offsets[index + 1])
        for position in range(begin, end):
            qubit = int(self.qubit_indices[position])
            label[self.num_qubits - 1 - qubit] = _PAULI_CHARS[int(self.pauli_codes[position])]
        return "".join(label)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return False
        return len(self) == len(other) and all(left == right for left, right in zip(self, other, strict=True))

    __hash__ = None


def _merge_term_partitions(p0: TermPartition, p1: TermPartition) -> TermPartition:
    """Merge two partitions by concatenating groups and offsetting *p1* indices.

    The offset is derived from *p0*: all *p1* indices are shifted by
    ``len(p0.all_indices())``.  Both partitions must be the same concrete
    type; a mismatch raises ``TypeError``.

    Example: H0 has 3 terms with partition groups ``((0, 1), (2,))`` and
    H1 has 2 terms with groups ``((0,), (1,))``.  After concatenation
    H1's indices are shifted by 3 (len of H0), producing
    ``((0, 1), (2,), (3,), (4,))``.

    """
    offset = len(p0.all_indices())

    if isinstance(p0, FlatPartition) and isinstance(p1, FlatPartition):
        shifted = tuple(tuple(i + offset for i in group) for group in p1.groups)
        return FlatPartition(strategy=p0.strategy, groups=p0.groups + shifted)

    if isinstance(p0, LayeredPartition) and isinstance(p1, LayeredPartition):
        shifted = tuple(tuple(tuple(i + offset for i in layer) for layer in group) for group in p1.groups)
        return LayeredPartition(strategy=p0.strategy, groups=p0.groups + shifted)

    raise TypeError(f"Cannot merge partitions of different types: {type(p0).__name__} and {type(p1).__name__}.")


def _hash_tapering(h, tapering: TaperingSpecification) -> None:
    """Hash tapering metadata through its JSON-compatible representation."""
    _hash_arg(h, tapering.to_json())


class QubitOperator(DataClass):
    """Data class representing an operator as a weighted sum of Pauli strings.

    Attributes:
        pauli_strings (Sequence[str]): Eager standard labels or lazy sparse-term labels.
        coefficients (numpy.ndarray): Array of coefficients corresponding to each Pauli string.
        encoding (str | None): The fermion-to-qubit encoding used to create this operator
            (e.g., "jordan-wigner", "bravyi-kitaev", "parity"). If None, encoding is not specified.
        fermion_mode_order (FermionModeOrder | None): The fermion mode ordering convention used
            when mapping fermionic modes to qubits (``"blocked"`` or ``"interleaved"``). If None,
            the ordering is unspecified or not applicable.
        term_partition (TermPartition | None): Optional index-based partition of
            :attr:`pauli_strings` into algorithm-relevant groups (and, for layered
            partitions, into parallelisable layers within each group).  Set by
            geometry-aware constructors and by ``term_grouper`` algorithms; reset
            to ``None`` by transformations that change the term ordering.
        tapering (TaperingSpecification | None): If this operator was produced by a
            tapering-based encoding (e.g. SCBK), records the applied tapering
            for downstream consumers. ``None`` for untapered encodings.

    Supports arithmetic: ``H1 + H2`` concatenates terms and merges
    partitions; ``scalar * H`` scales coefficients and preserves the
    partition.

    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for qubit operators.

        Returns:
            ``"qubit_hamiltonian"``.

        """
        return "qubit_hamiltonian"

    # Serialization version for this class
    _serialization_version = "0.1.0"
    _compact_serialization_version = "0.2.0"

    def __init__(
        self,
        pauli_strings: list[str],
        coefficients: np.ndarray,
        encoding: str | None = None,
        fermion_mode_order: FermionModeOrder | str | None = None,
        term_partition: TermPartition | None = None,
        tapering: TaperingSpecification | None = None,
    ) -> None:
        """Initialize a QubitOperator.

        Args:
            pauli_strings (list[str]): List of Pauli strings representing the ``QubitOperator``.
            coefficients (numpy.ndarray): Array of coefficients corresponding to each Pauli string.
            encoding (str | None): Fermion-to-qubit encoding (e.g., ``"jordan-wigner"``). Default ``None``.
            fermion_mode_order (FermionModeOrder | str | None): Mode ordering (``"blocked"``/``"interleaved"``).
            term_partition (TermPartition | None): Optional ``TermPartition`` carrying group/layer metadata.
            tapering (TaperingSpecification | None): Applied tapering metadata, or None if untapered.

        Raises:
            ValueError: If the number of Pauli strings and coefficients don't match,
                or if the Pauli strings or coefficients are invalid.

        """
        Logger.trace_entering()
        if len(pauli_strings) != len(coefficients):
            raise ValueError("Mismatch between number of Pauli strings and coefficients.")

        self.pauli_strings: Sequence[str] = pauli_strings
        self.coefficients = coefficients
        self.encoding = encoding
        self.fermion_mode_order: FermionModeOrder | None = (
            FermionModeOrder(fermion_mode_order) if fermion_mode_order is not None else None
        )
        self.term_partition: TermPartition | None = term_partition
        self.tapering: TaperingSpecification | None = tapering

        # Validate Pauli strings
        _validate_pauli_strings(pauli_strings)

        self._validate_partition()

        # Make instance immutable after construction (handled by base class)
        super().__init__()

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
    ) -> QubitOperator:
        """Construct an operator from non-identity Pauli factors."""
        offsets = array("Q", [0])
        indices = array("I")
        codes = array("B")
        for term in terms:
            factors = term.items() if isinstance(term, Mapping) else term
            previous = -1
            for qubit, pauli in sorted(factors):
                if qubit <= previous or qubit < 0 or qubit >= num_qubits:
                    raise ValueError("Sparse Pauli qubit indices must be unique and in range.")
                if pauli not in _PAULI_CODES:
                    raise ValueError(f"Invalid sparse Pauli operator: {pauli!r}.")
                indices.append(qubit)
                codes.append(_PAULI_CODES[pauli])
                previous = qubit
            offsets.append(len(indices))

        coefficient_array = np.asarray(coefficients)
        if len(offsets) - 1 != len(coefficient_array):
            raise ValueError("Mismatch between number of sparse Pauli terms and coefficients.")
        return cls.from_sparse_arrays(
            num_qubits,
            np.frombuffer(offsets, dtype=np.uint64).copy(),
            np.frombuffer(indices, dtype=np.uint32).copy(),
            np.frombuffer(codes, dtype=np.uint8).copy(),
            coefficient_array,
            encoding=encoding,
            fermion_mode_order=fermion_mode_order,
            term_partition=term_partition,
            tapering=tapering,
        )

    @classmethod
    def from_sparse_arrays(
        cls,
        num_qubits: int,
        term_offsets: np.ndarray,
        qubit_indices: np.ndarray,
        pauli_codes: np.ndarray,
        coefficients: np.ndarray,
        *,
        encoding: str | None = None,
        fermion_mode_order: FermionModeOrder | str | None = None,
        term_partition: TermPartition | None = None,
        tapering: TaperingSpecification | None = None,
    ) -> QubitOperator:
        """Construct an operator from packed sparse-term arrays."""
        term_offsets = np.asarray(term_offsets, dtype=np.uint64)
        qubit_indices = np.asarray(qubit_indices, dtype=np.uint32)
        pauli_codes = np.asarray(pauli_codes, dtype=np.uint8)
        coefficients = np.asarray(coefficients)
        if any(array.ndim != 1 for array in (term_offsets, qubit_indices, pauli_codes, coefficients)):
            raise ValueError("Packed sparse Pauli arrays must be one-dimensional.")
        term_starts = np.zeros(len(qubit_indices), dtype=bool)
        if len(qubit_indices):
            valid_starts = term_offsets[:-1][term_offsets[:-1] < len(qubit_indices)]
            term_starts[valid_starts] = True
        duplicate_or_unsorted = len(qubit_indices) > 1 and np.any(
            (qubit_indices[1:] <= qubit_indices[:-1]) & ~term_starts[1:]
        )
        if (
            num_qubits < 1
            or len(term_offsets) != len(coefficients) + 1
            or len(coefficients) == 0
            or len(qubit_indices) != len(pauli_codes)
            or term_offsets[0] != 0
            or term_offsets[-1] != len(qubit_indices)
            or np.any(term_offsets > len(qubit_indices))
            or np.any(term_offsets[1:] < term_offsets[:-1])
            or np.any(qubit_indices >= num_qubits)
            or np.any((pauli_codes < 1) | (pauli_codes > 3))
            or duplicate_or_unsorted
        ):
            raise ValueError("Invalid packed sparse Pauli arrays.")
        object_ = cls.__new__(cls)
        object_.__dict__.update(
            _num_qubits=num_qubits,
            _term_offsets=term_offsets,
            _qubit_indices=qubit_indices,
            _pauli_codes=pauli_codes,
        )
        object_.pauli_strings = _SparsePauliStrings(num_qubits, term_offsets, qubit_indices, pauli_codes)
        object_.coefficients = coefficients
        object_.encoding = encoding
        object_.fermion_mode_order = FermionModeOrder(fermion_mode_order) if fermion_mode_order is not None else None
        object_.term_partition = term_partition
        object_.tapering = tapering
        QubitOperator._validate_partition(object_)
        DataClass.__init__(object_)
        return object_

    def _validate_partition(self) -> None:
        if self.term_partition is None:
            return
        indices = np.fromiter(self.term_partition.iter_indices(), dtype=np.int64)
        if len(indices) != self.num_terms or not np.array_equal(np.sort(indices), np.arange(self.num_terms)):
            raise ValueError("term_partition must cover every term exactly once.")

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "qubit_hamiltonian")
        if self.has_sparse_terms:
            _hash_uint(h, self.num_qubits)
            _hash_array(h, self._term_offsets)
            _hash_array(h, self._qubit_indices)
            _hash_array(h, self._pauli_codes)
        else:
            _hash_uint(h, len(self.pauli_strings))
            for ps in self.pauli_strings:
                _hash_str(h, ps)
        _hash_array(h, self.coefficients)
        _hash_optional(h, self.encoding, _hash_str)
        _hash_optional(h, self.fermion_mode_order, lambda h, mode: _hash_str(h, str(mode)))
        _hash_optional(h, self.term_partition, lambda h, partition: _hash_str(h, partition.content_hash(0)))
        _hash_optional(h, self.tapering, _hash_tapering)

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the operator.

        Returns:
            int: The number of qubits.

        """
        if hasattr(self, "_num_qubits"):
            return self._num_qubits
        return len(self.pauli_strings[0])

    @property
    def num_terms(self) -> int:
        """Return the number of Pauli terms."""
        return len(self.pauli_strings)

    @property
    def has_sparse_terms(self) -> bool:
        """Return whether the operator uses packed sparse-term storage."""
        return hasattr(self, "_term_offsets")

    def sparse_term_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return packed term offsets, qubit indices, and Pauli codes."""
        if not self.has_sparse_terms:
            raise RuntimeError("This QubitOperator does not use packed sparse-term storage.")
        return self._term_offsets, self._qubit_indices, self._pauli_codes

    def iter_sparse_terms(self) -> Iterable[tuple[tuple[tuple[int, str], ...], complex]]:
        """Iterate over compact non-identity factors and coefficients."""
        if hasattr(self, "_term_offsets"):
            for term_index, coefficient in enumerate(self.coefficients):
                begin = int(self._term_offsets[term_index])
                end = int(self._term_offsets[term_index + 1])
                factors = tuple(
                    (int(self._qubit_indices[i]), _PAULI_CHARS[int(self._pauli_codes[i])]) for i in range(begin, end)
                )
                yield factors, complex(coefficient)
            return
        for label, coefficient in zip(self.pauli_strings, self.coefficients, strict=True):
            yield (
                tuple(
                    sorted(
                        (self.num_qubits - 1 - position, pauli) for position, pauli in enumerate(label) if pauli != "I"
                    )
                ),
                complex(coefficient),
            )

    @property
    def schatten_norm(self) -> float:
        """Calculate the Schatten norm (L1 norm) of the operator.

        The Schatten norm is the sum of the absolute values of all coefficients
        in the operator. This quantity is commonly used in estimating parameters
        for quantum algorithms, most notably Quantum Phase Estimation (QPE).

        Returns:
            float: The Schatten norm (L1 norm) of the operator.

        """
        return float(np.sum(np.abs(self.coefficients)))

    def to_matrix(self, sparse: bool = False) -> np.ndarray | scipy.sparse.spmatrix:
        """Convert the qubit operator to its full matrix representation.

        Args:
            sparse: If True, return a csr matrix.
                Otherwise return a dense matrix. Defaults to False.

        Returns:
            The operator matrix (dense or sparse).

        """
        if sparse:
            return pauli_to_sparse_matrix(self.pauli_strings, self.coefficients)
        return np.asarray(pauli_to_dense_matrix(self.pauli_strings, self.coefficients))

    def equiv(self, other: QubitOperator, atol: float = 1e-12) -> bool:
        """Check mathematical equivalence with another QubitOperator.

        Two operators are equivalent if they contain the same Pauli
        terms with the same coefficients (within tolerance), regardless of
        term ordering.  Duplicate Pauli strings are summed before comparison.

        Args:
            other: The QubitOperator to compare against.
            atol: Absolute tolerance for coefficient comparison. Defaults to 1e-12.

        Returns:
            ``True`` if the two operators are mathematically equivalent.

        Examples:
            >>> qh1 = QubitOperator(["XI", "ZZ"], np.array([0.5, 0.3]))
            >>> qh2 = QubitOperator(["ZZ", "XI"], np.array([0.3, 0.5]))
            >>> qh1.equiv(qh2)
            True

        """
        if not isinstance(other, QubitOperator):
            return False

        def _sum_terms(qh: QubitOperator) -> dict[tuple[tuple[int, str], ...], complex]:
            d: dict[tuple[tuple[int, str], ...], complex] = {}
            for term, coefficient in qh.iter_sparse_terms():
                d[term] = d.get(term, 0) + coefficient
            return d

        self_dict = _sum_terms(self)
        other_dict = _sum_terms(other)

        all_keys = set(self_dict) | set(other_dict)
        return all(abs(self_dict.get(k, 0) - other_dict.get(k, 0)) <= atol for k in all_keys)

    def is_hermitian(self, tolerance: float = 1e-12) -> bool:
        """Check whether all coefficients are real within ``tolerance``.

        A qubit operator is Hermitian if and only if every coefficient in
        its Pauli expansion is real.

        Args:
            tolerance: Maximum allowed magnitude of the imaginary part of
                any coefficient.  Defaults to 1e-12.

        Returns:
            ``True`` if every coefficient has ``|imag| <= tolerance``.

        """
        return all(abs(complex(c).imag) <= tolerance for c in self.coefficients)

    def __add__(self, other: QubitOperator) -> QubitOperator:
        """Return the sum of two qubit operators.

        Pauli strings and coefficients are concatenated.  The ``encoding``,
        ``fermion_mode_order``, and ``tapering`` metadata must match between
        operands (or both be ``None``); a mismatch raises ``ValueError``.
        If both operands carry a :attr:`term_partition` of the same concrete
        type, the partitions are merged (with the right-hand operand's indices
        offset).  Otherwise the result has no partition.

        Args:
            other: The qubit operator to add.

        Returns:
            A new ``QubitOperator`` with concatenated terms.

        Raises:
            TypeError: If *other* is not a ``QubitOperator``.
            ValueError: If the two operators have different qubit counts, encodings, or modes.

        """
        if not isinstance(other, QubitOperator):
            raise TypeError(f"Cannot add QubitOperator with {type(other).__name__}.")
        if self.num_qubits != other.num_qubits:
            raise ValueError(f"Cannot add operators with {self.num_qubits} and {other.num_qubits} qubits.")
        if self.encoding != other.encoding:
            raise ValueError(f"Cannot add operators with different encodings: {self.encoding!r} vs {other.encoding!r}.")
        if self.fermion_mode_order != other.fermion_mode_order:
            raise ValueError(
                f"Cannot add operators with different fermion_mode_order: "
                f"{self.fermion_mode_order!r} vs {other.fermion_mode_order!r}."
            )
        if self.tapering != other.tapering:
            raise ValueError(f"Cannot add operators with different tapering: {self.tapering!r} vs {other.tapering!r}.")

        coefficients = np.concatenate([self.coefficients, other.coefficients])

        partition = None
        if self.term_partition is not None and other.term_partition is not None:
            partition = _merge_term_partitions(self.term_partition, other.term_partition)

        if self.has_sparse_terms and other.has_sparse_terms:
            self_offsets, self_indices, self_codes = self.sparse_term_arrays()
            other_offsets, other_indices, other_codes = other.sparse_term_arrays()
            offsets = np.concatenate([self_offsets, other_offsets[1:] + len(self_indices)])
            return QubitOperator.from_sparse_arrays(
                self.num_qubits,
                offsets,
                np.concatenate([self_indices, other_indices]),
                np.concatenate([self_codes, other_codes]),
                coefficients,
                encoding=self.encoding,
                fermion_mode_order=self.fermion_mode_order,
                term_partition=partition,
                tapering=self.tapering,
            )

        pauli_strings = list(self.pauli_strings) + list(other.pauli_strings)
        return QubitOperator(
            pauli_strings,
            coefficients,
            encoding=self.encoding,
            fermion_mode_order=self.fermion_mode_order,
            term_partition=partition,
            tapering=self.tapering,
        )

    def __mul__(self, scalar) -> QubitOperator:
        """Return the operator with all coefficients scaled by *scalar*.

        The :attr:`term_partition` is preserved since term indices are unchanged.

        Args:
            scalar: The scalar multiplier.

        Returns:
            A new ``QubitOperator`` with scaled coefficients.

        """
        if not isinstance(scalar, int | float | complex | np.number):
            return NotImplemented
        if hasattr(self, "_term_offsets"):
            return QubitOperator.from_sparse_arrays(
                self.num_qubits,
                self._term_offsets.copy(),
                self._qubit_indices.copy(),
                self._pauli_codes.copy(),
                self.coefficients * scalar,
                encoding=self.encoding,
                fermion_mode_order=self.fermion_mode_order,
                term_partition=self.term_partition,
                tapering=self.tapering,
            )
        return QubitOperator(
            list(self.pauli_strings),
            self.coefficients * scalar,
            encoding=self.encoding,
            fermion_mode_order=self.fermion_mode_order,
            term_partition=self.term_partition,
            tapering=self.tapering,
        )

    def __rmul__(self, scalar: float) -> QubitOperator:
        """Support ``scalar * operator``."""
        return self.__mul__(scalar)

    def get_real_coefficients(
        self, tolerance: float = 1e-12, sort_by_magnitude: bool = False
    ) -> list[tuple[str, float]]:
        """Return ``(label, real_coeff)`` pairs for non-negligible terms.

        Only terms whose real-part magnitude exceeds ``tolerance`` are
        included.  Callers should verify Hermiticity via
        :meth:`is_hermitian` before invoking this method; imaginary parts
        are silently discarded here.

        Args:
            tolerance: Threshold for filtering small real coefficients.
                Defaults to 1e-12.
            sort_by_magnitude: If ``True``, return terms sorted by
                descending ``|coefficient|``.  Defaults to ``False``.

        Returns:
            List of ``(pauli_label, coefficient)`` tuples.

        """
        terms: list[tuple[str, float]] = []
        for pauli_str, coeff in zip(self.pauli_strings, self.coefficients, strict=True):
            real = complex(coeff).real
            if abs(real) > tolerance:
                terms.append((pauli_str, real))
        if sort_by_magnitude:
            terms.sort(key=lambda t: abs(t[1]), reverse=True)
        return terms

    def to_interleaved(self, n_spatial: int) -> QubitOperator:
        """Convert from blocked to interleaved spin-orbital ordering.

        Converts a qubit operator from blocked ordering (alpha orbitals first,
        then beta orbitals) to interleaved ordering (alternating alpha/beta).
        Blocked ordering:    [α₀, α₁, ..., αₙ₋₁, β₀, β₁, ..., βₙ₋₁]
        Interleaved ordering: [α₀, β₀, α₁, β₁, ..., αₙ₋₁, βₙ₋₁]

        Args:
            n_spatial (int): The number of spatial orbitals. The total number of
                qubits should be 2 * n_spatial.

        Returns:
            QubitOperator: A new QubitOperator with interleaved ordering.

        Raises:
            ValueError: If num_qubits != 2 * n_spatial.

        Examples:
            >>> # H2 with 2 spatial orbitals (4 qubits)
            >>> # Blocked: [α₀, α₁, β₀, β₁] -> Interleaved: [α₀, β₀, α₁, β₁]
            >>> interleaved = blocked_operator.to_interleaved(n_spatial=2)

        """
        Logger.trace_entering()
        n_qubits = self.num_qubits

        if n_qubits != 2 * n_spatial:
            raise ValueError(f"Number of qubits ({n_qubits}) must be 2 * n_spatial ({2 * n_spatial}).")

        # Build permutation: blocked -> interleaved
        # Blocked ordering:      a0, a1, ..., a(n-1), b0, b1, ..., b(n-1)
        # Interleaved ordering:  a0, b0, a1, b1, ..., a(n-1), b(n-1)
        # Pauli strings are little-endian (rightmost char = qubit 0), so
        # string position j corresponds to qubit (n_qubits - 1 - j).
        # Qubit mapping: alpha (q < n_spatial) -> 2*q, beta -> 2*(q - n_spatial) + 1
        permutation = [0] * n_qubits
        for pos in range(n_qubits):
            q_old = n_qubits - 1 - pos
            q_new = 2 * q_old if q_old < n_spatial else 2 * (q_old - n_spatial) + 1
            permutation[pos] = n_qubits - 1 - q_new

        reordered_strings = []
        for pauli_str in self.pauli_strings:
            new_chars = ["I"] * n_qubits
            for old_pos, char in enumerate(pauli_str):
                new_chars[permutation[old_pos]] = char
            reordered_strings.append("".join(new_chars))

        return QubitOperator(
            pauli_strings=reordered_strings,
            coefficients=self.coefficients.copy(),
            encoding=self.encoding,
            fermion_mode_order=FermionModeOrder.INTERLEAVED,
            tapering=self.tapering,
        )

    # DataClass interface implementation
    def get_summary(self) -> str:
        """Get a human-readable summary of the qubit operator.

        Returns:
            str: Summary string describing the qubit operator.

        """
        summary = (
            f"Qubit Operator\n  Number of qubits: {self.num_qubits}\n  Number of terms: {len(self.pauli_strings)}\n"
        )
        if self.encoding is not None:
            summary += f"  Encoding: {self.encoding}\n"
        if self.fermion_mode_order is not None:
            summary += f"  Fermion mode order: {self.fermion_mode_order}\n"
        return summary

    def to_json(self) -> dict[str, Any]:
        """Convert the qubit operator to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the qubit operator.

        """
        # Serialize complex coefficients as {"real": [...], "imag": [...]}
        # This handles both real and complex coefficient arrays
        coeffs = self.coefficients
        data: dict[str, Any] = {
            "coefficients": {
                "real": coeffs.real.tolist(),
                "imag": coeffs.imag.tolist(),
            },
        }
        if hasattr(self, "_term_offsets"):
            data.update(
                {
                    "num_qubits": self.num_qubits,
                    "term_offsets": self._term_offsets.tolist(),
                    "qubit_indices": self._qubit_indices.tolist(),
                    "pauli_codes": self._pauli_codes.tolist(),
                }
            )
        else:
            data["pauli_strings"] = self.pauli_strings
        if self.encoding is not None:
            data["encoding"] = self.encoding
        if self.fermion_mode_order is not None:
            data["fermion_mode_order"] = str(self.fermion_mode_order)
        if self.term_partition is not None:
            data["term_partition"] = self.term_partition.to_json()
        if self.tapering is not None:
            data["tapering"] = self.tapering.to_json()
        result = self._add_json_version(data)
        if hasattr(self, "_term_offsets"):
            result["version"] = self._compact_serialization_version
        return result

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the qubit operator to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write the qubit operator to.

        """
        self._add_hdf5_version(group)
        if hasattr(self, "_term_offsets"):
            group.attrs["version"] = self._compact_serialization_version
            group.attrs["num_qubits"] = self.num_qubits
            group.create_dataset("term_offsets", data=self._term_offsets)
            group.create_dataset("qubit_indices", data=self._qubit_indices)
            group.create_dataset("pauli_codes", data=self._pauli_codes)
        else:
            group.create_dataset("pauli_strings", data=np.array(self.pauli_strings, dtype="S"))
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
    def from_json(cls, json_data: dict[str, Any]) -> QubitOperator:
        """Create a QubitOperator from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized data.

        Returns:
            QubitOperator: New instance reconstructed from JSON data.

        Raises:
            RuntimeError: If version field is missing or incompatible.

        """
        expected_version = (
            cls._compact_serialization_version if "term_offsets" in json_data else cls._serialization_version
        )
        cls._validate_json_version(expected_version, json_data)
        coeff_data = json_data["coefficients"]
        # Handle complex coefficients serialized as {"real": [...], "imag": [...]}
        if isinstance(coeff_data, dict) and "real" in coeff_data and "imag" in coeff_data:
            coefficients = np.array(coeff_data["real"]) + 1j * np.array(coeff_data["imag"])
        else:
            # Fallback for legacy format (simple list of real numbers)
            coefficients = np.array(coeff_data)
        partition_data = json_data.get("term_partition")
        term_partition = TermPartition.from_json(partition_data) if partition_data is not None else None
        tapering_data = json_data.get("tapering")
        tapering = TaperingSpecification.from_json(tapering_data) if tapering_data is not None else None
        metadata = {
            "encoding": json_data.get("encoding"),
            "fermion_mode_order": json_data.get("fermion_mode_order"),
            "term_partition": term_partition,
            "tapering": tapering,
        }
        if "term_offsets" in json_data:
            return cls.from_sparse_arrays(
                json_data["num_qubits"],
                np.asarray(json_data["term_offsets"], dtype=np.uint64),
                np.asarray(json_data["qubit_indices"], dtype=np.uint32),
                np.asarray(json_data["pauli_codes"], dtype=np.uint8),
                coefficients,
                **metadata,
            )
        return cls(pauli_strings=json_data["pauli_strings"], coefficients=coefficients, **metadata)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> QubitOperator:
        """Load a QubitOperator from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file containing the data.

        Returns:
            QubitOperator: New instance reconstructed from HDF5 data.

        Raises:
            RuntimeError: If version attribute is missing or incompatible.

        """
        expected_version = cls._compact_serialization_version if "term_offsets" in group else cls._serialization_version
        cls._validate_hdf5_version(expected_version, group)
        coefficients = np.array(group["coefficients"])
        encoding = group.attrs.get("encoding")
        # Decode encoding if it's stored as bytes (HDF5 behavior can vary)
        if encoding is not None and isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        fermion_mode_order = group.attrs.get("fermion_mode_order")
        if fermion_mode_order is not None and isinstance(fermion_mode_order, bytes):
            fermion_mode_order = fermion_mode_order.decode("utf-8")
        partition_attr = group.attrs.get("term_partition")
        if partition_attr is not None:
            if isinstance(partition_attr, bytes):
                partition_attr = partition_attr.decode("utf-8")
            term_partition = TermPartition.from_json(json.loads(partition_attr))
        else:
            term_partition = None
        tapering_attr = group.attrs.get("tapering")
        if tapering_attr is not None:
            if isinstance(tapering_attr, bytes):
                tapering_attr = tapering_attr.decode("utf-8")
            tapering = TaperingSpecification.from_json(json.loads(tapering_attr))
        else:
            tapering = None
        metadata = {
            "encoding": encoding,
            "fermion_mode_order": fermion_mode_order,
            "term_partition": term_partition,
            "tapering": tapering,
        }
        if "term_offsets" in group:
            return cls.from_sparse_arrays(
                int(group.attrs["num_qubits"]),
                np.array(group["term_offsets"], dtype=np.uint64),
                np.array(group["qubit_indices"], dtype=np.uint32),
                np.array(group["pauli_codes"], dtype=np.uint8),
                coefficients,
                **metadata,
            )
        pauli_strings = [s.decode() for s in group["pauli_strings"][:]]
        return cls(pauli_strings=pauli_strings, coefficients=coefficients, **metadata)


def _validate_pauli_strings(pauli_strings: list[str]) -> None:
    """Validate that all Pauli strings are well-formed.

    Checks that every string uses only the characters {I, X, Y, Z} and
    that all strings have the same length.

    Raises:
        ValueError: If any string is empty, has invalid characters, or if strings have inconsistent lengths.

    """
    if not pauli_strings:
        raise ValueError("Pauli strings list cannot be empty.")
    length = len(pauli_strings[0])
    valid_pauli_pattern = re.compile(r"^[IXYZ]+$")
    for i, ps in enumerate(pauli_strings):
        if not ps:
            raise ValueError(f"Pauli string at index {i} is empty.")
        if len(ps) != length:
            raise ValueError(f"Pauli string at index {i} has length {len(ps)}, expected {length}.")
        if not valid_pauli_pattern.fullmatch(ps):
            invalid = set(ps) - set("IXYZ")
            raise ValueError(f"Pauli string at index {i} contains invalid characters: {invalid}.")


class _DeprecatedQubitOperatorAliasMeta(type(QubitOperator)):  # type: ignore[misc]
    """Metaclass that makes the deprecated alias behave like :class:`QubitOperator` for type checks.

    ``isinstance`` and ``issubclass`` tests against the alias delegate to
    :class:`QubitOperator`, so existing checks keep working in both directions
    even though :class:`QubitHamiltonian` is a distinct subclass.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        """Report any :class:`QubitOperator` instance as an instance of the alias."""
        return isinstance(instance, QubitOperator)

    def __subclasscheck__(cls, subclass: type) -> bool:
        """Report any :class:`QubitOperator` subclass as a subclass of the alias."""
        return issubclass(subclass, QubitOperator)


class QubitHamiltonian(QubitOperator, metaclass=_DeprecatedQubitOperatorAliasMeta):
    """Deprecated alias for :class:`QubitOperator`.

    .. deprecated::
        ``QubitHamiltonian`` was renamed to :class:`QubitOperator`. This subclass
        is retained for backward compatibility and will be removed in a future
        release. Constructing it emits a :class:`DeprecationWarning`. Thanks to a
        custom metaclass, ``isinstance(obj, QubitHamiltonian)`` still matches any
        :class:`QubitOperator` instance (and vice versa), so existing type checks
        keep working.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Construct a :class:`QubitOperator`, warning that ``QubitHamiltonian`` is deprecated."""
        warnings.warn(
            "'QubitHamiltonian' has been renamed to 'QubitOperator' and is deprecated; it will be "
            "removed in a future release. Replace 'QubitHamiltonian' with 'QubitOperator' "
            "(from qdk_chemistry.data import QubitOperator).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
