"""Tests for qiskit standard phase estimation circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit.library import StatePreparation as QiskitStatePreparation

from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Circuit, QpeResult, QubitHamiltonian
from qdk_chemistry.plugins.qiskit.circuit_executor import QiskitAerSimulator
from qdk_chemistry.plugins.qiskit.standard_phase_estimation import QiskitStandardPhaseEstimation
from qdk_chemistry.utils.phase import energy_from_phase

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
    qpe_phase_fraction_tolerance,
)

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
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    state_prep = QuantumCircuit(2, name="psi")
    state_prep.append(QiskitStatePreparation([0.6, 0.0, 0.0, 0.8]), list(range(2)))

    return TraditionalProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qasm=qasm3.dumps(state_prep)),
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
    hamiltonian = QubitHamiltonian(pauli_strings=["XXXX", "ZZZZ"], coefficients=[0.25, 4.5])
    state_prep = QuantumCircuit(4, name="psi_4q")
    state_vector = np.zeros(2**4, dtype=complex)
    state_vector[int("1000", 2)] = 0.8
    state_vector[int("0111", 2)] = -0.6
    state_prep.append(QiskitStatePreparation(state_vector), list(range(4)))

    return TraditionalProblem(
        label="four_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qasm=qasm3.dumps(state_prep)),
        evolution_time=float(np.pi / 8.0),
        num_bits=6,
        shots=3,
        expected_bitstring="010011",
        expected_phase=45 / 64,
        expected_energy=-4.75,
    )


def _extract_traditional_results(problem: TraditionalProblem) -> QpeResult:
    """Run traditional phase estimation and return the dominant measurement.

    Args:
        problem: The traditional phase estimation benchmark problem.

    Returns:
        QPE result including dominant bitstring, phase fraction, and energy.

    """
    qpe = QiskitStandardPhaseEstimation(num_bits=problem.num_bits, evolution_time=problem.evolution_time)
    simulator = QiskitAerSimulator(seed=_SEED)

    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    evolution_builder = create("time_evolution_builder", "trotter")

    return qpe.run(
        state_preparation=problem.state_prep,
        qubit_hamiltonian=problem.hamiltonian,
        evolution_builder=evolution_builder,
        circuit_mapper=circuit_mapper,
        circuit_executor=simulator,
    )


def test_traditional_phase_estimation_extracts_phase_and_energy(two_qubit_phase_problem: TraditionalProblem) -> None:
    """Validate traditional QPE on the two-qubit benchmark."""
    results = _extract_traditional_results(two_qubit_phase_problem)
    dominant_bitstring = results.bitstring_msb_first
    phase_fraction = results.phase_fraction

    # Resolve phase ambiguity
    phase_fraction_candidates = [phase_fraction % 1.0, (1.0 - phase_fraction) % 1.0]
    energies = [
        energy_from_phase(candidate, evolution_time=two_qubit_phase_problem.evolution_time)
        for candidate in phase_fraction_candidates
    ]

    index = (
        0
        if abs(energies[0] - two_qubit_phase_problem.expected_energy)
        <= abs(energies[1] - two_qubit_phase_problem.expected_energy)
        else 1
    )

    assert dominant_bitstring == two_qubit_phase_problem.expected_bitstring
    assert np.isclose(
        phase_fraction_candidates[index],
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        energies[index],
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_traditional_phase_estimation_four_qubit_problem(four_qubit_phase_problem: TraditionalProblem) -> None:
    """Validate traditional QPE on the documented four-qubit system."""
    results = _extract_traditional_results(four_qubit_phase_problem)
    dominant_bitstring = results.bitstring_msb_first
    phase_fraction = results.phase_fraction

    # Resolve phase ambiguity
    phase_fraction_candidates = [phase_fraction % 1.0, (1.0 - phase_fraction) % 1.0]
    energies = [
        energy_from_phase(candidate, evolution_time=four_qubit_phase_problem.evolution_time)
        for candidate in phase_fraction_candidates
    ]

    index = (
        0
        if abs(energies[0] - four_qubit_phase_problem.expected_energy)
        <= abs(energies[1] - four_qubit_phase_problem.expected_energy)
        else 1
    )

    assert dominant_bitstring == four_qubit_phase_problem.expected_bitstring
    assert np.isclose(
        phase_fraction_candidates[index],
        four_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        energies[index],
        four_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
