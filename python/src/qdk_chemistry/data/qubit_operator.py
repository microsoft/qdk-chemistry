"""QDK/Chemistry Qubit Operator module.

This module provides the QubitOperator dataclass for electronic structure problems. It bridges fermionic Hamiltonians
and quantum circuit construction or measurement workflows.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import re
import warnings
from typing import TYPE_CHECKING, Any

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
    """Data class for representing chemical electronic Hamiltonians in qubits.

    Attributes:
        pauli_strings (list[str]): List of Pauli strings representing the ``QubitOperator``.
        coefficients (numpy.ndarray): Array of coefficients corresponding to each Pauli string.
        encoding (str | None): The fermion-to-qubit encoding used to create this Hamiltonian
            (e.g., "jordan-wigner", "bravyi-kitaev", "parity"). If None, encoding is not specified.
        fermion_mode_order (FermionModeOrder | None): The fermion mode ordering convention used
            when mapping fermionic modes to qubits (``"blocked"`` or ``"interleaved"``). If None,
            the ordering is unspecified or not applicable.
        term_partition (TermPartition | None): Optional index-based partition of
            :attr:`pauli_strings` into algorithm-relevant groups (and, for layered
            partitions, into parallelisable layers within each group).  Set by
            geometry-aware constructors and by ``term_grouper`` algorithms; reset
            to ``None`` by transformations that change the term ordering.
        tapering (TaperingSpecification | None): If this Hamiltonian was produced by a
            tapering-based encoding (e.g. SCBK), records the applied tapering
            for downstream consumers. ``None`` for untapered encodings.

    Supports arithmetic: ``H1 + H2`` concatenates terms and merges
    partitions; ``scalar * H`` scales coefficients and preserves the
    partition.

    """

    # Class attribute for filename validation
    _data_type_name = "qubit_hamiltonian"

    # Serialization version for this class
    _serialization_version = "0.1.0"

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

        self.pauli_strings = pauli_strings
        self.coefficients = coefficients
        self.encoding = encoding
        self.fermion_mode_order: FermionModeOrder | None = (
            FermionModeOrder(fermion_mode_order) if fermion_mode_order is not None else None
        )
        self.term_partition: TermPartition | None = term_partition
        self.tapering: TaperingSpecification | None = tapering

        # Validate Pauli strings
        _validate_pauli_strings(pauli_strings)

        # Validate partition coverage
        if term_partition is not None:
            indices = sorted(term_partition.all_indices())
            expected = list(range(len(pauli_strings)))
            if indices != expected:
                missing = set(expected) - set(indices)
                duped = {i for i in indices if indices.count(i) > 1}
                raise ValueError(
                    f"term_partition does not cover all {len(pauli_strings)} terms exactly once. "
                    f"Missing: {missing or 'none'}, duplicated: {duped or 'none'}."
                )

        # Make instance immutable after construction (handled by base class)
        super().__init__()

    def _hash_update(self, h) -> None:
        """Feed identifying data into the hasher."""
        _hash_str(h, "qubit_hamiltonian")
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
        """Get the number of qubits in the Hamiltonian.

        Returns:
            int: The number of qubits.

        """
        return len(self.pauli_strings[0])

    @property
    def schatten_norm(self) -> float:
        """Calculate the Schatten norm (L1 norm) of the Hamiltonian.

        The Schatten norm is the sum of the absolute values of all coefficients
        in the Hamiltonian. This quantity is commonly used in estimating parameters
        for quantum algorithms, most notably Quantum Phase Estimation (QPE).

        Returns:
            float: The Schatten norm (L1 norm) of the Hamiltonian.

        """
        return float(np.sum(np.abs(self.coefficients)))

    def to_matrix(self, sparse: bool = False) -> np.ndarray | scipy.sparse.spmatrix:
        """Convert the qubit Hamiltonian to its full matrix representation.

        Args:
            sparse: If True, return a csr matrix.
                Otherwise return a dense matrix. Defaults to False.

        Returns:
            The Hamiltonian matrix (dense or sparse).

        """
        if sparse:
            return pauli_to_sparse_matrix(self.pauli_strings, self.coefficients)
        return np.asarray(pauli_to_dense_matrix(self.pauli_strings, self.coefficients))

    def equiv(self, other: QubitOperator, atol: float = 1e-12) -> bool:
        """Check mathematical equivalence with another QubitOperator.

        Two QubitHamiltonians are equivalent if they contain the same Pauli
        terms with the same coefficients (within tolerance), regardless of
        term ordering.  Duplicate Pauli strings are summed before comparison.

        Args:
            other: The QubitOperator to compare against.
            atol: Absolute tolerance for coefficient comparison. Defaults to 1e-12.

        Returns:
            ``True`` if the two QubitHamiltonians are mathematically equivalent.

        Examples:
            >>> qh1 = QubitOperator(["XI", "ZZ"], np.array([0.5, 0.3]))
            >>> qh2 = QubitOperator(["ZZ", "XI"], np.array([0.3, 0.5]))
            >>> qh1.equiv(qh2)
            True

        """
        if not isinstance(other, QubitOperator):
            return False

        def _sum_terms(qh: QubitOperator) -> dict[str, complex]:
            d: dict[str, complex] = {}
            for ps, c in zip(qh.pauli_strings, qh.coefficients, strict=True):
                d[ps] = d.get(ps, 0) + c
            return d

        self_dict = _sum_terms(self)
        other_dict = _sum_terms(other)

        all_keys = set(self_dict) | set(other_dict)
        return all(abs(self_dict.get(k, 0) - other_dict.get(k, 0)) <= atol for k in all_keys)

    def is_hermitian(self, tolerance: float = 1e-12) -> bool:
        """Check whether all coefficients are real within ``tolerance``.

        A qubit Hamiltonian is Hermitian if and only if every coefficient in
        its Pauli expansion is real.

        Args:
            tolerance: Maximum allowed magnitude of the imaginary part of
                any coefficient.  Defaults to 1e-12.

        Returns:
            ``True`` if every coefficient has ``|imag| <= tolerance``.

        """
        return all(abs(complex(c).imag) <= tolerance for c in self.coefficients)

    def __add__(self, other: QubitOperator) -> QubitOperator:
        """Return the sum of two qubit Hamiltonians.

        Pauli strings and coefficients are concatenated.  The ``encoding``,
        ``fermion_mode_order``, and ``tapering`` metadata must match between
        operands (or both be ``None``); a mismatch raises ``ValueError``.
        If both operands carry a :attr:`term_partition` of the same concrete
        type, the partitions are merged (with the right-hand operand's indices
        offset).  Otherwise the result has no partition.

        Args:
            other: The qubit Hamiltonian to add.

        Returns:
            A new ``QubitOperator`` with concatenated terms.

        Raises:
            TypeError: If *other* is not a ``QubitOperator``.
            ValueError: If the two Hamiltonians have different qubit counts, encodings, or modes.

        """
        if not isinstance(other, QubitOperator):
            raise TypeError(f"Cannot add QubitOperator with {type(other).__name__}.")
        if self.num_qubits != other.num_qubits:
            raise ValueError(f"Cannot add Hamiltonians with {self.num_qubits} and {other.num_qubits} qubits.")
        if self.encoding != other.encoding:
            raise ValueError(
                f"Cannot add Hamiltonians with different encodings: {self.encoding!r} vs {other.encoding!r}."
            )
        if self.fermion_mode_order != other.fermion_mode_order:
            raise ValueError(
                f"Cannot add Hamiltonians with different fermion_mode_order: "
                f"{self.fermion_mode_order!r} vs {other.fermion_mode_order!r}."
            )
        if self.tapering != other.tapering:
            raise ValueError(
                f"Cannot add Hamiltonians with different tapering: {self.tapering!r} vs {other.tapering!r}."
            )

        pauli_strings = list(self.pauli_strings) + list(other.pauli_strings)
        coefficients = np.concatenate([self.coefficients, other.coefficients])

        partition = None
        if self.term_partition is not None and other.term_partition is not None:
            partition = _merge_term_partitions(self.term_partition, other.term_partition)

        return QubitOperator(
            pauli_strings,
            coefficients,
            encoding=self.encoding,
            fermion_mode_order=self.fermion_mode_order,
            term_partition=partition,
            tapering=self.tapering,
        )

    def __mul__(self, scalar) -> QubitOperator:
        """Return the Hamiltonian with all coefficients scaled by *scalar*.

        The :attr:`term_partition` is preserved since term indices are unchanged.

        Args:
            scalar: The scalar multiplier.

        Returns:
            A new ``QubitOperator`` with scaled coefficients.

        """
        if not isinstance(scalar, int | float | complex | np.number):
            return NotImplemented
        return QubitOperator(
            list(self.pauli_strings),
            self.coefficients * scalar,
            encoding=self.encoding,
            fermion_mode_order=self.fermion_mode_order,
            term_partition=self.term_partition,
            tapering=self.tapering,
        )

    def __rmul__(self, scalar: float) -> QubitOperator:
        """Support ``scalar * hamiltonian``."""
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

        Converts a qubit Hamiltonian from blocked ordering (alpha orbitals first,
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
            >>> interleaved = blocked_hamiltonian.to_interleaved(n_spatial=2)

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
        """Get a human-readable summary of the qubit Hamiltonian.

        Returns:
            str: Summary string describing the qubit Hamiltonian.

        """
        summary = (
            f"Qubit Hamiltonian\n  Number of qubits: {self.num_qubits}\n  Number of terms: {len(self.pauli_strings)}\n"
        )
        if self.encoding is not None:
            summary += f"  Encoding: {self.encoding}\n"
        if self.fermion_mode_order is not None:
            summary += f"  Fermion mode order: {self.fermion_mode_order}\n"
        return summary

    def to_json(self) -> dict[str, Any]:
        """Convert the qubit Hamiltonian to a dictionary for JSON serialization.

        Returns:
            dict[str, Any]: Dictionary representation of the qubit Hamiltonian.

        """
        # Serialize complex coefficients as {"real": [...], "imag": [...]}
        # This handles both real and complex coefficient arrays
        coeffs = self.coefficients
        data = {
            "pauli_strings": self.pauli_strings,
            "coefficients": {
                "real": coeffs.real.tolist(),
                "imag": coeffs.imag.tolist(),
            },
        }
        if self.encoding is not None:
            data["encoding"] = self.encoding
        if self.fermion_mode_order is not None:
            data["fermion_mode_order"] = str(self.fermion_mode_order)
        if self.term_partition is not None:
            data["term_partition"] = self.term_partition.to_json()
        if self.tapering is not None:
            data["tapering"] = self.tapering.to_json()
        return self._add_json_version(data)

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the qubit Hamiltonian to an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file to write the qubit Hamiltonian to.

        """
        self._add_hdf5_version(group)
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
        cls._validate_json_version(cls._serialization_version, json_data)
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
        return cls(
            pauli_strings=json_data["pauli_strings"],
            coefficients=coefficients,
            encoding=json_data.get("encoding"),
            fermion_mode_order=json_data.get("fermion_mode_order"),
            term_partition=term_partition,
            tapering=tapering,
        )

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
        cls._validate_hdf5_version(cls._serialization_version, group)
        pauli_strings = [s.decode() for s in group["pauli_strings"][:]]
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
        return cls(
            pauli_strings=pauli_strings,
            coefficients=coefficients,
            encoding=encoding,
            fermion_mode_order=fermion_mode_order,
            term_partition=term_partition,
            tapering=tapering,
        )


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
