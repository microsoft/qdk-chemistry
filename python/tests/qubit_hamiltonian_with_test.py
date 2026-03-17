"""QDK/Chemistry Qubit Hamiltonian module.

This module provides the QubitHamiltonian dataclass for electronic structure problems. It bridges fermionic Hamiltonians
and quantum circuit construction or measurement workflows.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

import h5py
import numpy as np

from qdk_chemistry._core.utils import (
    pauli_expectation,
    pauli_to_dense_matrix,
    pauli_to_sparse_matrix,
)
from qdk_chemistry.data import Wavefunction
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.pauli_commutation import do_pauli_labels_commute, do_pauli_labels_qw_commute

__all__ = ["filter_and_group_pauli_ops_from_wavefunction"]


class QubitHamiltonian(DataClass):
    """Data class for representing chemical electronic Hamiltonians in qubits.

    Attributes:
        pauli_strings (list[str]): List of Pauli strings representing the ``QubitHamiltonian``.
        coefficients (numpy.ndarray): Array of coefficients corresponding to each Pauli string.
        encoding (str | None): The fermion-to-qubit encoding used to create this Hamiltonian
            (e.g., "jordan-wigner", "bravyi-kitaev", "parity"). If None, encoding is not specified.

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
    ) -> None:
        """Initialize a QubitHamiltonian.

        Args:
            pauli_strings (list[str]): List of Pauli strings representing the ``QubitHamiltonian``.
            coefficients (numpy.ndarray): Array of coefficients corresponding to each Pauli string.
            encoding (str | None): The fermion-to-qubit encoding used to create this Hamiltonian.
                Valid values include "jordan-wigner", "bravyi-kitaev", "parity", or None.
                Defaults to None.

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

    def to_matrix(self, sparse: bool = False) -> np.ndarray:
        r"""Convert the Hamiltonian to its full matrix representation.

        Args:
            sparse: If ``True``, return a csr matrix.
                Otherwise return a dense matrix.  Defaults to ``False``.

        Returns:
            The Hamiltonian matrix (dense or sparse).

        """
        coeffs = self.coefficients.astype(np.complex128)

        if sparse:
            return pauli_to_sparse_matrix(self.pauli_strings, coeffs)
        return np.asarray(pauli_to_dense_matrix(self.pauli_strings, coeffs))

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

    def reorder_qubits(self, permutation: list[int]) -> "QubitHamiltonian":
        """Reorder qubits in all Pauli strings according to a permutation.

        Applies a qubit index permutation to all Pauli strings. The permutation
        specifies where each qubit should be mapped: permutation[old_index] = new_index.

        Args:
            permutation (list[int]): A permutation mapping old qubit indices to new indices.
                Must be a valid permutation of [0, 1, ..., num_qubits-1].

        Returns:
            QubitHamiltonian: A new QubitHamiltonian with reordered Pauli strings.

        Raises:
            ValueError: If the permutation is invalid (wrong length or not a valid permutation).

        Examples:
            >>> qh = QubitHamiltonian(["XIZI", "IYII"], np.array([0.5, 0.3]))
            >>> # Swap qubits 0 and 1: permutation[0]=1, permutation[1]=0, ...
            >>> reordered = qh.reorder_qubits([1, 0, 2, 3])
            >>> print(reordered.pauli_strings)
            ['IXZI', 'YIII']

        """
        Logger.trace_entering()
        n_qubits = self.num_qubits

        # Validate permutation
        if len(permutation) != n_qubits:
            raise ValueError(f"Permutation length ({len(permutation)}) must match number of qubits ({n_qubits}).")
        if sorted(permutation) != list(range(n_qubits)):
            raise ValueError(f"Invalid permutation: must be a permutation of [0, 1, ..., {n_qubits - 1}].")

        # Apply permutation to each Pauli string
        # Pauli strings use qiskit label convention: string[i] corresponds to qubit n-1-i
        reordered_strings = []
        for pauli_str in self.pauli_strings:
            # Create new string with reordered characters
            new_chars = ["I"] * n_qubits
            for old_idx, char in enumerate(pauli_str):
                new_idx = permutation[old_idx]
                new_chars[new_idx] = char
            reordered_strings.append("".join(new_chars))

        return QubitHamiltonian(
            pauli_strings=reordered_strings,
            coefficients=self.coefficients.copy(),
        )

    def to_interleaved(self, n_spatial: int) -> "QubitHamiltonian":
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
        # For blocked index i, alpha spin (i < n_spatial) maps to 2*i,
        # and beta spin (i >= n_spatial) maps to 2*(i - n_spatial) + 1
        permutation = []
        for i in range(n_qubits):
            if i < n_spatial:
                permutation.append(2 * i)
            else:
                permutation.append(2 * (i - n_spatial) + 1)

        return self.reorder_qubits(permutation)

    def group_commuting(self, qubit_wise: bool = True) -> list["QubitHamiltonian"]:
        """Group the qubit Hamiltonian into commuting subsets.

        Args:
            qubit_wise (bool): Whether to use qubit-wise commuting grouping.
                Qubit-wise commutation is stricter than general commutation.
                Default is True.

        Returns:
            list[QubitHamiltonian]: A list of ``QubitHamiltonian`` representing the grouped Hamiltonian.

        """
        Logger.trace_entering()
        commutes = do_pauli_labels_qw_commute if qubit_wise else do_pauli_labels_commute

        # Each group is a list of (pauli_string, coefficient)
        groups: list[list[tuple[str, complex]]] = []

        for pauli_str, coeff in zip(self.pauli_strings, self.coefficients, strict=False):
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

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "QubitHamiltonian":
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
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "QubitHamiltonian":
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
        return cls(pauli_strings=pauli_strings, coefficients=coefficients, encoding=encoding)


def _filter_and_group_pauli_ops_from_statevector(
    hamiltonian: QubitHamiltonian,
    statevector: np.ndarray,
    abelian_grouping: bool = True,
    trimming: bool = True,
    trimming_tolerance: float = 1e-8,
) -> tuple[list[QubitHamiltonian], list[float]]:
    """Filter and group the Pauli operators respect to a given quantum state.

    This function evaluates each Pauli term in the Hamiltonian with respect to the
    provided statevector:

    * Terms with zero expectation value are discarded.
    * Terms with expectation ±1 are treated as classical and their contribution is
        added to the energy at the end.
    * Remaining terms with fractional expectation values are retained and grouped by
        shared expectation value to reduce measurement redundancy
        (e.g., due to symmetry).
    * The rest of Hamiltonian is grouped into qubit wise commuting terms.

    Args:
        hamiltonian (QubitHamiltonian): QubitHamiltonian to be filtered and grouped.
        statevector (numpy.ndarray): Statevector used to compute expectation values.
        abelian_grouping (bool): Whether to group into qubit-wise commuting subsets.
        trimming (bool): If True, discard or reduce terms with ±1 or 0 expectation value.
        trimming_tolerance (float): Numerical tolerance for determining zero or ±1 expectation (Default: 1e-8).

    Returns:
        A tuple of ``(list[QubitHamiltonian], list[float])``
            * A list of grouped QubitHamiltonian.
            * A list of classical coefficients for terms that were reduced to classical contributions.

    """
    Logger.trace_entering()
    psi = np.asarray(statevector, dtype=complex)
    norm = np.linalg.norm(psi)
    if norm < np.finfo(np.float64).eps:
        raise ValueError("Statevector has zero norm.")
    psi /= norm

    retained_paulis: list[str] = []
    retained_coeffs: list[complex] = []
    expectations: list[float] = []
    classical: list[float] = []

    n = hamiltonian.num_qubits
    for pauli_str, coeff in zip(hamiltonian.pauli_strings, hamiltonian.coefficients, strict=True):
        expval = pauli_expectation(pauli_str, n, psi)

        if not trimming:
            retained_paulis.append(pauli_str)
            retained_coeffs.append(coeff)
            expectations.append(expval)
            continue

        if np.isclose(expval, 0.0, atol=trimming_tolerance):
            continue
        if np.isclose(expval, 1.0, atol=trimming_tolerance):
            classical.append(float(coeff.real))
        elif np.isclose(expval, -1.0, atol=trimming_tolerance):
            classical.append(float(-coeff.real))
        else:
            retained_paulis.append(pauli_str)
            retained_coeffs.append(coeff)
            expectations.append(expval)

    if not retained_paulis:
        return [], classical

    grouped: dict[int, list[tuple[str, complex, float]]] = {}
    key_counter = 0
    # Assign approximate groups based on tolerance
    for pauli, coeff, expval in zip(retained_paulis, retained_coeffs, expectations, strict=True):
        matched_key = None
        for k, terms in grouped.items():
            if np.isclose(expval, terms[0][2], atol=trimming_tolerance):
                matched_key = k
                break
        if matched_key is None:
            grouped[key_counter] = [(pauli, coeff, expval)]
            key_counter += 1
        else:
            grouped[matched_key].append((pauli, coeff, expval))

    reduced_pauli: list[str] = []
    reduced_coeffs: list[complex] = []

    for _, terms in grouped.items():
        coeff_sum = sum(c for _, c, _ in terms)
        # Choose Pauli with maximum # of I (most diagonal)
        best_pauli = sorted([p for p, _, _ in terms], key=lambda p: (-str(p).count("I"), str(p)))[0]
        reduced_pauli.append(best_pauli)
        reduced_coeffs.append(coeff_sum)

    reduced_hamiltonian = QubitHamiltonian(reduced_pauli, np.array(reduced_coeffs), encoding=hamiltonian.encoding)

    grouped_hamiltonians = (
        reduced_hamiltonian.group_commuting(qubit_wise=abelian_grouping) if abelian_grouping else [reduced_hamiltonian]
    )

    return grouped_hamiltonians, classical


def filter_and_group_pauli_ops_from_wavefunction(
    hamiltonian: QubitHamiltonian,
    wavefunction: Wavefunction,
    abelian_grouping: bool = True,
    trimming: bool = True,
    trimming_tolerance: float = 1e-8,
) -> tuple[list[QubitHamiltonian], list[float]]:
    """Filter and group the Pauli operators respect to a given quantum state.

    This function evaluates each Pauli term in the Hamiltonian with respect to the
    provided wavefunction:

    * Terms with zero expectation value are discarded.
    * Terms with expectation ±1 are treated as classical and their contribution is
        added to the energy at the end.
    * Remaining terms with fractional expectation values are retained and grouped by
        shared expectation value to reduce measurement redundancy
        (e.g., due to symmetry).
    * The rest of Hamiltonian is grouped into qubit wise commuting terms.

    Args:
        hamiltonian (QubitHamiltonian): QubitHamiltonian to be filtered and grouped.
        wavefunction (Wavefunction): Wavefunction used to compute expectation values.
        abelian_grouping (bool): Whether to group into qubit-wise commuting subsets.
        trimming (bool): If True, discard or reduce terms with ±1 or 0 expectation value.
        trimming_tolerance (float): Numerical tolerance for determining zero or ±1 expectation (Default: 1e-8).

    Returns:
        A tuple of ``(list[QubitHamiltonian], list[float])``
            * A list of grouped QubitHamiltonian.
            * A list of classical coefficients for terms that were reduced to classical contributions.

    """
    from qdk_chemistry.plugins.qiskit.conversion import create_statevector_from_wavefunction  # noqa: PLC0415

    Logger.trace_entering()
    psi = create_statevector_from_wavefunction(wavefunction, normalize=True)
    return _filter_and_group_pauli_ops_from_statevector(
        hamiltonian, psi, abelian_grouping, trimming, trimming_tolerance
    )


def _validate_pauli_strings(pauli_strings: list[str]) -> None:
    """Validate that all Pauli strings are well-formed.

    Checks that every string uses only the characters {I, X, Y, Z} and
    that all strings have the same length.

    Raises:
        ValueError: If any string contains invalid characters or lengths differ.

    """
    if not pauli_strings:
        return
    valid_chars = set("IXYZ")
    length = len(pauli_strings[0])
    for i, ps in enumerate(pauli_strings):
        if len(ps) != length:
            raise ValueError(f"Pauli string at index {i} has length {len(ps)}, expected {length}.")
        bad = set(ps) - valid_chars
        if bad:
            raise ValueError(f"Pauli string at index {i} contains invalid characters: {bad}.")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Quick smoke-test: verify qiskit label convention against explicit
    # Kronecker-product matrices.
    #
    # Qiskit label convention: label[0] = highest qubit (MSB),
    # label[n-1] = qubit 0 (LSB).  The matrix is
    #   P_{label[0]} ⊗ P_{label[1]} ⊗ … ⊗ P_{label[n-1]}
    # ------------------------------------------------------------------

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    _PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}

    def _kron_qiskit_order(label: str) -> np.ndarray:
        """Build the matrix for a Pauli label in qiskit convention."""
        # label[0] = highest qubit → standard left-to-right Kronecker product
        mat = _PAULI[label[0]]
        for ch in label[1:]:
            mat = np.kron(mat, _PAULI[ch])
        return mat

    test_labels = ["XI", "IX", "ZI", "IZ", "YI", "IY", "XY", "YX", "ZZ", "XX", "YY", "XYZ", "ZIX", "YZX", "XXYZ"]

    print("=== to_matrix Kronecker-product test ===")
    all_pass = True
    for label in test_labels:
        expected = _kron_qiskit_order(label)
        qh = QubitHamiltonian([label], np.array([1.0 + 0j]))
        got = qh.to_matrix()
        if np.allclose(expected, got):
            print(f"  {label:6s}  PASS")
        else:
            print(f"  {label:6s}  FAIL")
            print(f"    expected diag: {np.diag(expected)}")
            print(f"    got      diag: {np.diag(got)}")
            all_pass = False

    # Also test _pauli_expectation against matrix-based <psi|P|psi>
    print("\n=== _pauli_expectation test ===")
    rng = np.random.default_rng(42)
    for label in test_labels:
        n = len(label)
        psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
        psi /= np.linalg.norm(psi)
        mat = _kron_qiskit_order(label)
        expected_ev = float(np.real(psi.conj() @ mat @ psi))
        got_ev = pauli_expectation(label, psi)
        ok = np.isclose(expected_ev, got_ev)
        status = "PASS" if ok else "FAIL"
        print(f"  {label:6s}  {status}  (expected {expected_ev:+.6f}, got {got_ev:+.6f})")
        if not ok:
            all_pass = False

    # Multi-term Hamiltonian: H = 0.5*ZI + 0.3*IX + 0.2*XY
    print("\n=== Multi-term Hamiltonian test ===")
    labels = ["ZI", "IX", "XY"]
    coeffs = np.array([0.5, 0.3, 0.2])
    expected_H = sum(c * _kron_qiskit_order(l) for c, l in zip(coeffs, labels, strict=False))
    qh = QubitHamiltonian(labels, coeffs)
    got_H = qh.to_matrix()
    ok = np.allclose(expected_H, got_H)
    print(f"  H = 0.5*ZI + 0.3*IX + 0.2*XY  {'PASS' if ok else 'FAIL'}")
    if not ok:
        all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    # ------------------------------------------------------------------
    # Compare against qiskit SparsePauliOp to confirm consistency
    # ------------------------------------------------------------------
    try:
        from qiskit.quantum_info import SparsePauliOp

        print("\n=== Comparison with qiskit SparsePauliOp ===")

        # Single-term matrix comparison
        print("\n--- Single-term to_matrix vs SparsePauliOp.to_matrix ---")
        for label in test_labels:
            qiskit_mat = SparsePauliOp(label).to_matrix()
            our_mat = QubitHamiltonian([label], np.array([1.0 + 0j])).to_matrix()
            ok = np.allclose(qiskit_mat, our_mat)
            status = "PASS" if ok else "FAIL"
            print(f"  {label:6s}  {status}")
            if not ok:
                all_pass = False

        # Multi-term Hamiltonian
        print("\n--- Multi-term to_matrix vs SparsePauliOp.to_matrix ---")
        multi_labels = ["ZI", "IX", "XY", "YZ", "XX", "YY", "ZZ", "IZ"]
        multi_coeffs = [0.5, 0.3, 0.2, -0.1, 0.4, -0.25, 0.15, -0.35]
        qiskit_H = SparsePauliOp(multi_labels, multi_coeffs).to_matrix()
        our_H = QubitHamiltonian(multi_labels, np.array(multi_coeffs)).to_matrix()
        ok = np.allclose(qiskit_H, our_H)
        print(f"  8-term 2-qubit Hamiltonian  {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

        # 3-qubit multi-term
        labels_3q = ["XYZ", "ZIX", "YZX", "III", "ZZZ", "XXX"]
        coeffs_3q = [0.3, -0.2, 0.15, 1.0, -0.5, 0.25]
        qiskit_H3 = SparsePauliOp(labels_3q, coeffs_3q).to_matrix()
        our_H3 = QubitHamiltonian(labels_3q, np.array(coeffs_3q)).to_matrix()
        ok = np.allclose(qiskit_H3, our_H3)
        print(f"  6-term 3-qubit Hamiltonian  {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

        # Expectation value comparison
        print("\n--- _pauli_expectation vs qiskit ---")
        rng2 = np.random.default_rng(123)
        for label in test_labels:
            n = len(label)
            psi = rng2.standard_normal(2**n) + 1j * rng2.standard_normal(2**n)
            psi /= np.linalg.norm(psi)
            qiskit_ev = float(np.real(psi.conj() @ SparsePauliOp(label).to_matrix() @ psi))
            our_ev = pauli_expectation(label, psi)
            ok = np.isclose(qiskit_ev, our_ev)
            status = "PASS" if ok else "FAIL"
            print(f"  {label:6s}  {status}  (qiskit {qiskit_ev:+.6f}, ours {our_ev:+.6f})")
            if not ok:
                all_pass = False

        # group_commuting comparison
        print("\n--- group_commuting vs SparsePauliOp.group_commuting ---")
        gc_labels = ["ZI", "IZ", "ZZ", "XI", "IX", "XX", "YY"]
        gc_coeffs = [0.5, 0.3, 0.2, -0.1, 0.4, -0.25, 0.15]
        qiskit_groups = SparsePauliOp(gc_labels, gc_coeffs).group_commuting(qubit_wise=True)
        our_groups = QubitHamiltonian(gc_labels, np.array(gc_coeffs)).group_commuting(qubit_wise=True)
        # Reconstruct full matrix from groups — must equal original
        qiskit_reconstructed = sum(g.to_matrix() for g in qiskit_groups)
        our_reconstructed = sum(g.to_matrix() for g in our_groups)
        qiskit_full = SparsePauliOp(gc_labels, gc_coeffs).to_matrix()
        ok_qiskit = np.allclose(qiskit_reconstructed, qiskit_full)
        ok_ours = np.allclose(our_reconstructed, qiskit_full)
        print(f"  qiskit groups reconstruct original:  {'PASS' if ok_qiskit else 'FAIL'}")
        print(f"  our groups reconstruct original:     {'PASS' if ok_ours else 'FAIL'}")
        print(f"  number of groups: qiskit={len(qiskit_groups)}, ours={len(our_groups)}")
        if not ok_ours:
            all_pass = False

        print(f"\n{'ALL TESTS PASSED (incl. qiskit)' if all_pass else 'SOME TESTS FAILED'}")

        # ------------------------------------------------------------------
        # 16-qubit sparse benchmark: memory + timing vs SparsePauliOp
        # ------------------------------------------------------------------
        import random
        import threading
        import time
        import tracemalloc

        try:
            import psutil
        except ImportError:
            psutil = None

        print("\n=== 22-qubit sparse benchmark (ours vs SparsePauliOp) ===")
        n_qubits = 24
        n_terms = 5
        pauli_chars = "IXYZ"
        random.seed(2025)
        rng_bench = np.random.default_rng(2025)

        bench_labels = ["".join(random.choice(pauli_chars) for _ in range(n_qubits)) for _ in range(n_terms)]
        bench_coeffs = rng_bench.standard_normal(n_terms)

        qh_bench = QubitHamiltonian(bench_labels, bench_coeffs)
        spo_bench = SparsePauliOp(bench_labels, bench_coeffs)

        print(f"  {n_qubits} qubits, {n_terms} terms  (dim = {2**n_qubits})")

        def _measure_build(build_fn):
            """Return (matrix, elapsed_seconds, py_peak_bytes, rss_delta_peak_bytes|None)."""
            stop_evt = threading.Event()
            proc = psutil.Process() if psutil is not None else None
            rss_baseline = proc.memory_info().rss if proc is not None else None
            rss_peak = [rss_baseline if rss_baseline is not None else 0]

            def _sample_rss() -> None:
                if proc is None:
                    return
                while not stop_evt.is_set():
                    rss_now = proc.memory_info().rss
                    rss_peak[0] = max(rss_peak[0], rss_now)
                    time.sleep(0.001)

            sampler = threading.Thread(target=_sample_rss, daemon=True)
            sampler.start()

            tracemalloc.start()
            t0 = time.perf_counter()
            mat = build_fn()
            elapsed = time.perf_counter() - t0
            _, py_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            stop_evt.set()
            sampler.join(timeout=0.1)

            if proc is None:
                rss_delta_peak = None
            else:
                rss_peak[0] = max(rss_peak[0], proc.memory_info().rss)
                rss_delta_peak = max(0, rss_peak[0] - rss_baseline)

            return mat, elapsed, py_peak, rss_delta_peak

        our_sparse, t_ours, py_peak_ours, rss_peak_ours = _measure_build(lambda: qh_bench.to_matrix(sparse=True))
        qiskit_sparse, t_qiskit, py_peak_qiskit, rss_peak_qiskit = _measure_build(
            lambda: spo_bench.to_matrix(sparse=True)
        )

        diff = abs(our_sparse - qiskit_sparse).max()
        ok = diff < 1e-12
        print(f"\n  Matrices match (max diff {diff:.2e}): {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
        rss_ours_txt = f"{(rss_peak_ours / 1024**2):.1f} MiB delta" if rss_peak_ours is not None else "n/a"
        rss_qiskit_txt = f"{(rss_peak_qiskit / 1024**2):.1f} MiB delta" if rss_peak_qiskit is not None else "n/a"
        print(
            f"  Ours   sparse: {t_ours:.4f}s, "
            f"tracemalloc peak {py_peak_ours / 1024**2:.1f} MiB, "
            f"rss peak {rss_ours_txt}, "
            f"nnz {our_sparse.nnz}"
        )
        print(
            f"  Qiskit sparse: {t_qiskit:.4f}s, "
            f"tracemalloc peak {py_peak_qiskit / 1024**2:.1f} MiB, "
            f"rss peak {rss_qiskit_txt}, "
            f"nnz {qiskit_sparse.nnz}"
        )

        # ------------------------------------------------------------------
        # 6-qubit dense benchmark: small sanity check for dense path
        # ------------------------------------------------------------------
        print("\n=== 6-qubit dense benchmark (ours vs SparsePauliOp) ===")
        n_qubits_dense = 6
        n_terms_dense = 100

        dense_labels = [
            "".join(random.choice(pauli_chars) for _ in range(n_qubits_dense)) for _ in range(n_terms_dense)
        ]
        dense_coeffs = rng_bench.standard_normal(n_terms_dense)

        qh_dense = QubitHamiltonian(dense_labels, dense_coeffs)
        spo_dense = SparsePauliOp(dense_labels, dense_coeffs)

        our_dense, t_ours_dense, py_peak_ours_dense, rss_peak_ours_dense = _measure_build(
            lambda: qh_dense.to_matrix(sparse=False)
        )
        qiskit_dense, t_qiskit_dense, py_peak_qiskit_dense, rss_peak_qiskit_dense = _measure_build(
            lambda: spo_dense.to_matrix(sparse=False)
        )

        dense_diff = np.max(np.abs(our_dense - qiskit_dense))
        ok_dense = dense_diff < 1e-12
        print(f"\n  Matrices match (max diff {dense_diff:.2e}): {'PASS' if ok_dense else 'FAIL'}")
        if not ok_dense:
            all_pass = False

        rss_ours_dense_txt = (
            f"{(rss_peak_ours_dense / 1024**2):.2f} MiB delta" if rss_peak_ours_dense is not None else "n/a"
        )
        rss_qiskit_dense_txt = (
            f"{(rss_peak_qiskit_dense / 1024**2):.2f} MiB delta" if rss_peak_qiskit_dense is not None else "n/a"
        )
        print(
            f"  Ours   dense: {t_ours_dense:.4f}s, "
            f"tracemalloc peak {py_peak_ours_dense / 1024**2:.2f} MiB, "
            f"rss peak {rss_ours_dense_txt}, "
            f"shape {our_dense.shape}, bytes {our_dense.nbytes / 1024**2:.2f} MiB"
        )
        print(
            f"  Qiskit dense: {t_qiskit_dense:.4f}s, "
            f"tracemalloc peak {py_peak_qiskit_dense / 1024**2:.2f} MiB, "
            f"rss peak {rss_qiskit_dense_txt}, "
            f"shape {qiskit_dense.shape}, bytes {qiskit_dense.nbytes / 1024**2:.2f} MiB"
        )

        expected_dense_bytes = (2**n_qubits_dense) ** 2 * np.dtype(np.complex128).itemsize
        print(f"  Expected dense payload: {expected_dense_bytes / 1024**2:.2f} MiB")

        print(f"\n{'ALL BENCHMARKS + TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    except ImportError:
        print("\nqiskit not installed — skipping qiskit comparison tests.")
