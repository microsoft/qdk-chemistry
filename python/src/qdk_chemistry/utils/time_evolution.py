"""Utility helpers for constructing controlled time-evolution circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from qiskit import QuantumCircuit
    from qiskit.circuit import Qubit
    from qiskit.quantum_info import SparsePauliOp

    from qdk_chemistry.data.qubit_hamiltonian import QubitHamiltonian

__all__ = [
    "PauliEvolutionTerm",
    "append_controlled_time_evolution",
    "controlled_pauli_rotation",
    "extract_terms_from_hamiltonian",
    "pauli_evolution_terms",
]


@dataclass(frozen=True)
class PauliEvolutionTerm:
    """Container for a Pauli string and its real coefficient."""

    pauli_map: dict[int, str]
    coefficient: float


def _pauli_label_to_map(label: str) -> dict[int, str]:
    """Translate a Qiskit Pauli label to a mapping ``qubit -> {X, Y, Z}``.

    Args:
        label: Pauli string label in Qiskit's little-endian ordering.

    Returns:
        Dictionary assigning each non-identity qubit index to its Pauli axis.

    """
    mapping: dict[int, str] = {}
    for index, char in enumerate(reversed(label)):  # reversed: right-most char -> qubit 0
        if char != "I":
            mapping[index] = char
    return mapping


def pauli_evolution_terms(pauli_op: SparsePauliOp, *, atol: float = 1e-12) -> list[PauliEvolutionTerm]:
    """Convert a :class:`~qiskit.quantum_info.SparsePauliOp` into rotation data for time evolution.

    Args:
        pauli_op (:class:`~qiskit.quantum_info.SparsePauliOp`): Operator to decompose.
        atol (float): Absolute tolerance used to discard negligible coefficients

            and to detect unwanted imaginary components.

    Returns:
        Ordered list of :class:`PauliEvolutionTerm` entries describing each non-identity component of ``pauli_op``.

    Raises:
        ValueError: If a coefficient has an imaginary part whose magnitude exceeds ``atol``.

    """
    terms: list[PauliEvolutionTerm] = []
    for pauli, coeff in zip(pauli_op.paulis, pauli_op.coeffs, strict=True):
        if abs(coeff) < atol:
            continue
        if abs(coeff.imag) > atol:
            raise ValueError(
                "Iterative phase estimation currently supports only Hermitian Hamiltonians "
                f"with real coefficients. Encountered coefficient {coeff} for term {pauli.to_label()}."
            )
        mapping = _pauli_label_to_map(pauli.to_label())
        terms.append(PauliEvolutionTerm(pauli_map=mapping, coefficient=float(coeff.real)))
    return terms


def controlled_pauli_rotation(
    circuit: QuantumCircuit,
    control_qubit: Qubit | int,
    system_qubits: Sequence[Qubit | int],
    term: PauliEvolutionTerm,
    *,
    angle: float,
) -> QuantumCircuit:
    """Append a controlled ``exp(-i angle * P)`` to ``circuit``.

    Args:
        circuit: Quantum circuit receiving the controlled rotation.
        control_qubit: Index of the ancilla qubit providing the control.
        system_qubits: Ordered collection of system qubit indices.
        term: Pauli term describing the rotation axis.
        angle: Rotation angle before the factor of two applied by CRZ.

    Returns:
        The quantum circuit with the controlled rotation appended.

    """
    if not term.pauli_map:
        # Identity contribution results in a controlled phase on the ancilla.
        circuit.p(-angle, control_qubit)
        return circuit

    involved_indices = sorted(term.pauli_map.keys())
    involved_qubits = [system_qubits[i] for i in involved_indices]

    # Basis-change into Z
    for idx, qubit in zip(involved_indices, involved_qubits, strict=True):
        pauli = term.pauli_map[idx]
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)

    target = involved_qubits[-1]
    for qubit in involved_qubits[:-1]:
        circuit.cx(qubit, target)

    circuit.crz(2 * angle, control_qubit, target)

    for qubit in reversed(involved_qubits[:-1]):
        circuit.cx(qubit, target)

    for idx, qubit in reversed(list(zip(involved_indices, involved_qubits, strict=True))):
        pauli = term.pauli_map[idx]
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.h(qubit)
            circuit.s(qubit)

    return circuit


def append_controlled_time_evolution(
    circuit: QuantumCircuit,
    control_qubit: Qubit | int,
    system_qubits: Sequence[Qubit | int],
    terms: Iterable[PauliEvolutionTerm],
    *,
    time: float,
    power: int = 1,
) -> None:
    """Append the controlled unitary ``(exp(-i H time))**power``.

    Args:
        circuit: Circuit being extended.
        control_qubit: Index of the single ancilla control qubit.
        system_qubits: Ordered system qubits targeted by the evolution.
        terms: Iterable of Pauli decomposition entries for the Hamiltonian.
        time: Evolution time ``t`` for a single application of ``U``.
        power: Number of repeated applications (``U`` raised to ``power``).

    Raises:
        ValueError: If ``power`` is less than 1.

    """
    if power < 1:
        raise ValueError("power must be at least 1 for controlled time evolution.")

    for _ in range(power):
        for term in terms:
            rotation_angle = time * term.coefficient
            if np.isclose(rotation_angle, 0.0):
                continue
            controlled_pauli_rotation(
                circuit,
                control_qubit,
                system_qubits,
                term,
                angle=rotation_angle,
            )


def extract_terms_from_hamiltonian(hamiltonian: QubitHamiltonian) -> list[PauliEvolutionTerm]:
    """Compute the cached Pauli decomposition for a :class:`qdk_chemistry.data.QubitHamiltonian`.

    Args:
        hamiltonian: Hamiltonian whose Pauli terms are required.

    Returns:
        List of :class:`PauliEvolutionTerm` entries representing the Hamiltonian.

    """
    return pauli_evolution_terms(hamiltonian.pauli_ops)
