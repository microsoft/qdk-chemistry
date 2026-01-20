"""QDK/Chemistry Qubit Hamiltonian module.

This module provides the QubitHamiltonian dataclass for electronic structure problems. It bridges fermionic Hamiltonians
and quantum circuit construction or measurement workflows.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from functools import cached_property
from typing import Any

import h5py
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from qdk_chemistry.data import Wavefunction
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.utils import Logger

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

        try:
            _ = self.pauli_ops  # Trigger cached property to validate Pauli strings
        except Exception as e:
            raise ValueError(f"Invalid Pauli strings or coefficients: {e}") from e

        # Make instance immutable after construction (handled by base class)
        super().__init__()

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the Hamiltonian.

        Returns:
            int: The number of qubits.

        """
        return self.pauli_ops.num_qubits

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

    @cached_property
    def pauli_ops(self) -> SparsePauliOp:
        """Get the qubit Hamiltonian as a ``SparsePauliOp``.

        Returns:
            qiskit.quantum_info.SparsePauliOp: The qubit Hamiltonian represented as a ``SparsePauliOp``.

        """
        return SparsePauliOp(self.pauli_strings, self.coefficients)

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
        # Pauli strings are in little-endian order: string[i] corresponds to qubit i
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
            qubit_wise (bool): Whether to use qubit-wise commuting grouping. Default is True.

        Returns:
            list[QubitHamiltonian]: A list of ``QubitHamiltonian`` representing the grouped Hamiltonian.

        """
        Logger.trace_entering()
        sparse_pauli_ops = self.pauli_ops.group_commuting(qubit_wise=qubit_wise)
        return [
            QubitHamiltonian(
                pauli_strings=group.paulis.to_labels(),
                coefficients=group.coeffs,
                encoding=self.encoding,
            )
            for group in sparse_pauli_ops
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

    for pauli, coeff in zip(hamiltonian.pauli_ops.paulis, hamiltonian.coefficients, strict=True):
        expval = float(np.vdot(psi, pauli.to_matrix(sparse=True) @ psi).real)

        if not trimming:
            retained_paulis.append(pauli.to_label())
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
            retained_paulis.append(pauli.to_label())
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
