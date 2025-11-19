"""Tests for traditional phase estimation circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

from qdk_chemistry.algorithms import TraditionalPhaseEstimation, energy_from_phase
from qdk_chemistry.data import QubitHamiltonian

from .reference_tolerances import qpe_energy_tolerance, qpe_phase_fraction_tolerance

_SEED = 42


@dataclass(frozen=True)
class TraditionalProblem:
    """Bundle describing a benchmark for traditional phase estimation."""

    label: str
    hamiltonian: QubitHamiltonian
    state_prep: QuantumCircuit
    evolution_time: float
    num_bits: int
    shots: int
    expected_bitstring: str
    expected_phase: float
    expected_energy: float


@pytest.fixture
def two_qubit_phase_problem() -> TraditionalProblem:
    """Return the canonical two-qubit phase estimation setup."""
    sparse_pauli_op = SparsePauliOp.from_list([("XX", 0.25), ("ZZ", 0.5)])
    hamiltonian = QubitHamiltonian(
        pauli_strings=sparse_pauli_op.paulis.to_labels(), coefficients=sparse_pauli_op.coeffs
    )
    state_prep = QuantumCircuit(2, name="psi")
    state_prep.initialize([0.6, 0.0, 0.0, 0.8], [0, 1])

    return TraditionalProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=state_prep,
        evolution_time=float(np.pi / 2.0),
        num_bits=4,
        shots=3,
        expected_bitstring="1101",
        expected_phase=0.1875,
        expected_energy=0.75,
    )


@pytest.fixture
def four_qubit_phase_problem() -> TraditionalProblem:
    """Return the documented four-qubit benchmark."""
    sparse_pauli_op = SparsePauliOp.from_list([("XXXX", 0.25), ("ZZZZ", 4.5)])
    hamiltonian = QubitHamiltonian(
        pauli_strings=sparse_pauli_op.paulis.to_labels(), coefficients=sparse_pauli_op.coeffs
    )
    state_prep = QuantumCircuit(4, name="psi_4q")
    state_vector = np.zeros(2**4, dtype=complex)
    state_vector[int("1000", 2)] = 0.8
    state_vector[int("0111", 2)] = -0.6
    state_prep.initialize(state_vector, list(range(4)))

    return TraditionalProblem(
        label="four_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=state_prep,
        evolution_time=float(np.pi / 8.0),
        num_bits=6,
        shots=3,
        expected_bitstring="010011",
        expected_phase=45 / 64,
        expected_energy=-4.75,
    )


def _extract_traditional_results(problem: TraditionalProblem) -> tuple[str, float, float]:
    """Run traditional phase estimation and return the dominant measurement."""
    traditional = TraditionalPhaseEstimation(problem.hamiltonian, problem.evolution_time)
    simulator = AerSimulator(seed_simulator=_SEED)

    circuit = traditional.create_circuit(problem.state_prep, num_bits=problem.num_bits)
    compiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=problem.shots).result()
    counts = result.get_counts()

    dominant_bitstring = max(counts, key=counts.get)
    raw_phase = int(dominant_bitstring, 2) / (2**problem.num_bits)
    candidates = [raw_phase % 1.0, (1.0 - raw_phase) % 1.0]
    energies = [energy_from_phase(candidate, evolution_time=problem.evolution_time) for candidate in candidates]

    index = 0 if abs(energies[0] - problem.expected_energy) <= abs(energies[1] - problem.expected_energy) else 1
    return dominant_bitstring, candidates[index], energies[index]


def test_traditional_phase_estimation_extracts_phase_and_energy(two_qubit_phase_problem: TraditionalProblem) -> None:
    """Validate traditional QPE on the two-qubit benchmark."""
    dominant_bitstring, phase_fraction, energy = _extract_traditional_results(two_qubit_phase_problem)

    assert dominant_bitstring == two_qubit_phase_problem.expected_bitstring
    assert phase_fraction == pytest.approx(two_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance)
    assert energy == pytest.approx(two_qubit_phase_problem.expected_energy, abs=qpe_energy_tolerance)


def test_traditional_phase_estimation_four_qubit_problem(four_qubit_phase_problem: TraditionalProblem) -> None:
    """Validate traditional QPE on the documented four-qubit system."""
    dominant_bitstring, phase_fraction, energy = _extract_traditional_results(four_qubit_phase_problem)

    assert dominant_bitstring == four_qubit_phase_problem.expected_bitstring
    assert phase_fraction == pytest.approx(four_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance)
    assert energy == pytest.approx(four_qubit_phase_problem.expected_energy, abs=qpe_energy_tolerance)
