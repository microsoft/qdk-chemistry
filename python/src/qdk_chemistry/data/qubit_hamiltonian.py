"""QDK/Chemistry Qubit Hamiltonian module.

This module provides the QubitHamiltonian dataclass for electronic structure problems. It bridges fermionic Hamiltonians
and quantum circuit construction or measurement workflows.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np

from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils.pauli_matrix import (
    pauli_to_dense_matrix,
    pauli_to_sparse_matrix,
)

if TYPE_CHECKING:
    import h5py
    import scipy

from qdk_chemistry.data.enums.fermion_mode_order import FermionModeOrder
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.pauli_commutation import do_pauli_labels_commute, do_pauli_labels_qw_commute

__all__: list[str] = []


class QubitHamiltonian(DataClass):
    """Data class for representing chemical electronic Hamiltonians in qubits.

    Attributes:
        pauli_strings (list[str]): List of Pauli strings representing the ``QubitHamiltonian``.
        coefficients (numpy.ndarray): Array of coefficients corresponding to each Pauli string.
        encoding (str | None): The fermion-to-qubit encoding used to create this Hamiltonian
            (e.g., "jordan-wigner", "bravyi-kitaev", "parity"). If None, encoding is not specified.
        fermion_mode_order (FermionModeOrder | None): The fermion mode ordering convention used
            when mapping fermionic modes to qubits (``"blocked"`` or ``"interleaved"``). If None,
            the ordering is unspecified or not applicable.

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
    ) -> None:
        """Initialize a QubitHamiltonian.

        Args:
            pauli_strings (list[str]): List of Pauli strings representing the ``QubitHamiltonian``.
            coefficients (numpy.ndarray): Array of coefficients corresponding to each Pauli string.
            encoding (str | None): Fermion-to-qubit encoding (e.g., ``"jordan-wigner"``). Default ``None``.
            fermion_mode_order (FermionModeOrder | str | None): Mode ordering (``"blocked"``/``"interleaved"``).

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

        # Validate Pauli strings
        _validate_pauli_strings(pauli_strings)

        # Make instance immutable after construction (handled by base class)
        super().__init__()

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

    def equiv(self, other: QubitHamiltonian, atol: float = 1e-12) -> bool:
        """Check mathematical equivalence with another QubitHamiltonian.

        Two QubitHamiltonians are equivalent if they contain the same Pauli
        terms with the same coefficients (within tolerance), regardless of
        term ordering.  Duplicate Pauli strings are summed before comparison.

        Args:
            other: The QubitHamiltonian to compare against.
            atol: Absolute tolerance for coefficient comparison. Defaults to 1e-12.

        Returns:
            ``True`` if the two QubitHamiltonians are mathematically equivalent.

        Examples:
            >>> qh1 = QubitHamiltonian(["XI", "ZZ"], np.array([0.5, 0.3]))
            >>> qh2 = QubitHamiltonian(["ZZ", "XI"], np.array([0.3, 0.5]))
            >>> qh1.equiv(qh2)
            True

        """
        if not isinstance(other, QubitHamiltonian):
            return False

        def _sum_terms(qh: QubitHamiltonian) -> dict[str, complex]:
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

    def to_interleaved(self, n_spatial: int) -> QubitHamiltonian:
        """Convert from blocked to interleaved spin-orbital ordering.

        Converts a qubit Hamiltonian from blocked ordering (alpha orbitals first,
        then beta orbitals) to interleaved ordering (alternating alpha/beta).

        Blocked ordering:    [α₀, α₁, ..., αₙ₋₁, β₀, β₁, ..., βₙ₋₁]
        Interleaved ordering: [α₀, β₀, α₁, β₁, ..., αₙ₋₁, βₙ₋₁]

        Args:
            n_spatial (int): The number of spatial orbitals. The total number of
                qubits should be 2 * n_spatial.

        Returns:
            QubitHamiltonian: A new QubitHamiltonian with interleaved ordering.

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

        return QubitHamiltonian(
            pauli_strings=reordered_strings,
            coefficients=self.coefficients.copy(),
            encoding=self.encoding,
            fermion_mode_order=FermionModeOrder.INTERLEAVED,
        )

    def group_commuting(self, qubit_wise: bool = True) -> list[QubitHamiltonian]:
        """Group the qubit Hamiltonian into commuting subsets.

        Args:
            qubit_wise (bool): Whether to use qubit-wise commuting grouping. Default is True.

        Returns:
            list[QubitHamiltonian]: A list of ``QubitHamiltonian`` representing the grouped Hamiltonian.

        """
        Logger.trace_entering()
        commutes = do_pauli_labels_qw_commute if qubit_wise else do_pauli_labels_commute

        # Each group is a list of (pauli_string, coefficient)
        groups: list[list[tuple[str, complex]]] = []

        for pauli_str, coeff in zip(self.pauli_strings, self.coefficients, strict=True):
            placed = False
            for group in groups:
                if all(commutes(pauli_str, existing_str) for existing_str, _ in group):
                    group.append((pauli_str, coeff))
                    placed = True
                    break
            if not placed:
                groups.append([(pauli_str, coeff)])

        return [
            QubitHamiltonian(
                pauli_strings=[p for p, _ in group],
                coefficients=np.array([c for _, c in group]),
                encoding=self.encoding,
                fermion_mode_order=self.fermion_mode_order,
            )
            for group in groups
        ]

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

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> QubitHamiltonian:
        """Create a QubitHamiltonian from a JSON dictionary.

        Args:
            json_data (dict[str, Any]): Dictionary containing the serialized data.

        Returns:
            QubitHamiltonian: New instance reconstructed from JSON data.

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
        return cls(
            pauli_strings=json_data["pauli_strings"],
            coefficients=coefficients,
            encoding=json_data.get("encoding"),
            fermion_mode_order=json_data.get("fermion_mode_order"),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> QubitHamiltonian:
        """Load a QubitHamiltonian from an HDF5 group.

        Args:
            group (h5py.Group): HDF5 group or file containing the data.

        Returns:
            QubitHamiltonian: New instance reconstructed from HDF5 data.

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
        return cls(
            pauli_strings=pauli_strings,
            coefficients=coefficients,
            encoding=encoding,
            fermion_mode_order=fermion_mode_order,
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
