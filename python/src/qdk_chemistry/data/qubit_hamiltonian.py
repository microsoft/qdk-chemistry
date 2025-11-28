"""QDK/Chemistry Qubit Hamiltonian Module.

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
from qdk_chemistry.utils.statevector import create_statevector_from_wavefunction

__all__ = ["filter_and_group_pauli_ops_from_wavefunction"]


class QubitHamiltonian(DataClass):
    """Data class for representing chemical electronic Hamiltonians in qubits.

    Attributes:
        pauli_strings: List of Pauli strings representing the ``QubitHamiltonian``.
        coefficients: Array of coefficients corresponding to each Pauli string.

    """

    # Class attribute for filename validation
    _data_type_name = "qubit_hamiltonian"

    def __init__(self, pauli_strings: list[str], coefficients: np.ndarray):
        """Initialize a QubitHamiltonian.

        Args:
            pauli_strings: List of Pauli strings.
            coefficients: Array of coefficients corresponding to each Pauli string.

        Raises:
            ValueError: If the number of Pauli strings and coefficients don't match,
                or if the Pauli strings or coefficients are invalid.

        """
        if len(pauli_strings) != len(coefficients):
            raise ValueError("Mismatch between number of Pauli strings and coefficients.")

        self.pauli_strings = pauli_strings
        self.coefficients = coefficients

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

    @cached_property
    def pauli_ops(self) -> SparsePauliOp:
        """Get the Hamiltonian as a ``SparsePauliOp``.

        Returns:
            The Hamiltonian represented as a ``SparsePauliOp``.

        """
        return SparsePauliOp(self.pauli_strings, self.coefficients)

    def group_commuting(self, qubit_wise: bool = True) -> list["QubitHamiltonian"]:
        """Group the Hamiltonian into commuting subsets.

        Args:
            qubit_wise: Whether to use qubit-wise commuting grouping. Default is True.

        Returns:
            A list of ``QubitHamiltonian`` representing the grouped Hamiltonian.

        """
        sparse_pauli_ops = self.pauli_ops.group_commuting(qubit_wise=qubit_wise)
        return [
            QubitHamiltonian(pauli_strings=group.paulis.to_labels(), coefficients=group.coeffs)
            for group in sparse_pauli_ops
        ]

    @property
    def exact_energy(self) -> float | None:
        """Compute the exact ground state energy via matrix diagonalization.

        Returns:
            float | None: The minimum eigenvalue if qubit count is small enough, else None.

        """
        return np.linalg.eigvalsh(self.pauli_ops.to_matrix()).min()

    # DataClass interface implementation
    def get_summary(self) -> str:
        """Get a human-readable summary of the Hamiltonian."""
        return (
            f"Qubit Hamiltonian\n"
            f"  Number of qubits: {self.num_qubits}\n"
            f"  Number of terms: {len(self.pauli_strings)}\n"
            f"  Exact energy: {self.exact_energy:.6f}"
        )

    def to_json(self) -> dict[str, Any]:
        """Convert the Hamiltonian to a dictionary for JSON serialization."""
        return {
            "pauli_strings": self.pauli_strings,
            "coefficients": self.coefficients.tolist(),
        }

    def to_hdf5(self, group: h5py.Group) -> None:
        """Save the Hamiltonian to an HDF5 group.

        Note:
            This method is used internally when saving to HDF5 files.
            Python users should call to_hdf5_file() directly.

        """
        group.create_dataset("pauli_strings", data=np.array(self.pauli_strings, dtype="S"))
        group.create_dataset("coefficients", data=self.coefficients)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "QubitHamiltonian":
        """Create a QubitHamiltonian from a JSON dictionary.

        Args:
            json_data: Dictionary containing the serialized data

        Returns:
            QubitHamiltonian: New instance reconstructed from JSON data

        """
        return cls(
            pauli_strings=json_data["pauli_strings"],
            coefficients=np.array(json_data["coefficients"]),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "QubitHamiltonian":
        """Load a QubitHamiltonian from an HDF5 group.

        Args:
            group: HDF5 group or file containing the data

        Returns:
            QubitHamiltonian: New instance reconstructed from HDF5 data

        """
        pauli_strings = [s.decode() for s in group["pauli_strings"][:]]
        coefficients = np.array(group["coefficients"])
        return cls(pauli_strings=pauli_strings, coefficients=coefficients)


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
        hamiltonian: QubitHamiltonian to be filtered and grouped.
        statevector: Statevector used to compute expectation values.
        abelian_grouping: Whether to group into qubit-wise commuting subsets.
        trimming: If True, discard or reduce terms with ±1 or 0 expectation value.
        trimming_tolerance: Numerical tolerance for determining zero or ±1 expectation (Default: 1e-8).

    Returns:
        A tuple of ``(list[QubitHamiltonian], list[float])``
            * A list of grouped QubitHamiltonian.
            * A list of classical coefficients for terms that were reduced to classical contributions.

    """
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

    reduced_hamiltonian = QubitHamiltonian(reduced_pauli, np.array(reduced_coeffs))

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
        hamiltonian: QubitHamiltonian to be filtered and grouped.
        wavefunction: Wavefunction used to compute expectation values.
        abelian_grouping: Whether to group into qubit-wise commuting subsets.
        trimming: If True, discard or reduce terms with ±1 or 0 expectation value.
        trimming_tolerance: Numerical tolerance for determining zero or ±1 expectation (Default: 1e-8).

    Returns:
        A tuple of ``(list[QubitHamiltonian], list[float])``
            * A list of grouped QubitHamiltonian.
            * A list of classical coefficients for terms that were reduced to classical contributions.

    """
    psi = create_statevector_from_wavefunction(wavefunction)
    return _filter_and_group_pauli_ops_from_statevector(
        hamiltonian, psi, abelian_grouping, trimming, trimming_tolerance
    )
