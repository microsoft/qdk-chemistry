"""Tests for iterative phase estimation algorithms."""

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

from qdk_chemistry.algorithms import (
    IterativePhaseEstimation,
    TraditionalPhaseEstimation,
)
from qdk_chemistry.data import QpeResult, QubitHamiltonian

from .reference_tolerances import qpe_energy_tolerance, qpe_phase_fraction_tolerance

_SEED = 42


@dataclass(frozen=True)
class PhaseEstimationProblem:
    """Container describing a reproducible phase estimation benchmark."""

    label: str
    hamiltonian: QubitHamiltonian
    state_prep: QuantumCircuit
    evolution_time: float
    num_bits: int
    expected_bits: list[int]
    expected_phase: float
    expected_energy: float
    expected_bitstring: str
    shots_iterative: int
    shots_traditional: int


@pytest.fixture
def two_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return the two-qubit phase estimation scenario used in documentation."""
    sparse_pauli_op = SparsePauliOp.from_list([("XX", 0.25), ("ZZ", 0.5)])
    hamiltonian = QubitHamiltonian(
        pauli_strings=sparse_pauli_op.paulis.to_labels(), coefficients=sparse_pauli_op.coeffs
    )
    state_prep = QuantumCircuit(2, name="psi")
    state_prep.initialize([0.6, 0.0, 0.0, 0.8], [0, 1])

    return PhaseEstimationProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=state_prep,
        evolution_time=float(np.pi / 2.0),
        num_bits=4,
        expected_bits=[1, 1, 0, 0],
        expected_phase=0.1875,
        expected_energy=0.75,
        expected_bitstring="1101",
        shots_iterative=3,
        shots_traditional=3,
    )


@pytest.fixture
def four_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return the four-qubit benchmark used in documentation."""
    sparse_pauli_op = SparsePauliOp.from_list([("XXXX", 0.25), ("ZZZZ", 4.5)])
    hamiltonian = QubitHamiltonian(
        pauli_strings=sparse_pauli_op.paulis.to_labels(), coefficients=sparse_pauli_op.coeffs
    )
    state_prep = QuantumCircuit(4, name="psi_4q")
    state_vector = np.zeros(2**4, dtype=complex)
    state_vector[int("1000", 2)] = 0.8
    state_vector[int("0111", 2)] = -0.6
    state_prep.initialize(state_vector, list(range(4)))

    return PhaseEstimationProblem(
        label="four_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=state_prep,
        evolution_time=float(np.pi / 8.0),
        num_bits=6,
        expected_bits=[1, 0, 1, 1, 0, 1],
        expected_phase=45 / 64,
        expected_energy=-4.75,
        expected_bitstring="010011",
        shots_iterative=3,
        shots_traditional=3,
    )


def _run_iterative(problem: PhaseEstimationProblem) -> QpeResult:
    """Execute iterative phase estimation and return structured results.

    Args:
        problem: Benchmark description supplying Hamiltonian, state prep, and expectations.

    Returns:
        :class:`QpeResult` instance summarizing the iterative run.

    """
    iqpe = IterativePhaseEstimation(problem.hamiltonian, problem.evolution_time)
    simulator = AerSimulator(seed_simulator=_SEED)

    phase_feedback = 0.0
    bits: list[int] = []

    for iteration in range(problem.num_bits):
        iteration_data = iqpe.create_iteration(
            problem.state_prep,
            iteration=iteration,
            total_iterations=problem.num_bits,
            phase_correction=phase_feedback,
        )
        compiled = transpile(iteration_data.circuit, simulator, optimization_level=0)
        result = simulator.run(compiled, shots=problem.shots_iterative).result()
        counts = result.get_counts()
        measured_bit = 0 if counts.get("0", 0) >= counts.get("1", 0) else 1

        bits.append(measured_bit)
        phase_feedback = iqpe.update_phase_feedback(phase_feedback, measured_bit)

    phase_fraction = iqpe.phase_fraction_from_feedback(phase_feedback)
    return QpeResult.from_phase_fraction(
        method=IterativePhaseEstimation.algorithm,
        phase_fraction=phase_fraction,
        evolution_time=problem.evolution_time,
        bits_msb_first=bits,
        reference_energy=problem.expected_energy,
        metadata={"label": problem.label},
    )


def _run_traditional(problem: PhaseEstimationProblem) -> QpeResult:
    """Execute traditional QPE and return structured results.

    Args:
        problem: Benchmark description supplying Hamiltonian, state prep, and expectations.

    Returns:
        :class:`QpeResult` instance summarizing the traditional run.

    """
    traditional = TraditionalPhaseEstimation(problem.hamiltonian, problem.evolution_time)
    simulator = AerSimulator(seed_simulator=_SEED)

    circuit = traditional.create_circuit(problem.state_prep, num_bits=problem.num_bits)
    compiled = transpile(circuit, simulator, optimization_level=0)
    result = simulator.run(compiled, shots=problem.shots_traditional).result()
    counts = result.get_counts()

    dominant_bitstring = max(counts, key=counts.get)
    phase_fraction = int(dominant_bitstring, 2) / (2**problem.num_bits)
    bits = [int(bit) for bit in dominant_bitstring]
    return QpeResult.from_phase_fraction(
        method=TraditionalPhaseEstimation.algorithm,
        phase_fraction=phase_fraction,
        evolution_time=problem.evolution_time,
        bits_msb_first=bits,
        bitstring_msb_first=dominant_bitstring,
        reference_energy=problem.expected_energy,
        metadata={"label": problem.label},
    )


def _run_iterative_with_parameters(
    pauli_terms: list[tuple[str, float]],
    state_vector: np.ndarray,
    *,
    evolution_time: float,
    num_bits: int,
    shots_per_iteration: int,
    seed: int,
    reference_energy: float | None = None,
) -> QpeResult:
    """Execute iterative phase estimation for a custom Hamiltonian/state pair.

    Args:
        pauli_terms: List of Pauli terms and coefficients defining the Hamiltonian.
        state_vector: Initial state amplitudes for the system register.
        evolution_time: Evolution time ``t`` used in ``U = exp(-i H t)``.
        num_bits: Number of iterative QPE rounds executed.
        shots_per_iteration: Number of simulator shots per iteration circuit.
        seed: PRNG seed for the simulator.
        reference_energy: Optional reference energy used to resolve alias branches.

    Returns:
        :class:`QpeResult` capturing the iterative estimation outcome.

    """
    sparse_pauli_op = SparsePauliOp.from_list(pauli_terms)
    hamiltonian = QubitHamiltonian(
        pauli_strings=sparse_pauli_op.paulis.to_labels(), coefficients=sparse_pauli_op.coeffs
    )
    num_qubits = int(np.log2(len(state_vector)))

    state_prep = QuantumCircuit(num_qubits, name="state")
    state_prep.initialize(state_vector, list(range(num_qubits)))

    iqpe = IterativePhaseEstimation(hamiltonian, evolution_time)
    simulator = AerSimulator(seed_simulator=seed)

    phase_feedback = 0.0
    bits: list[int] = []

    for iteration in range(num_bits):
        iteration_info = iqpe.create_iteration(
            state_prep,
            iteration=iteration,
            total_iterations=num_bits,
            phase_correction=phase_feedback,
        )
        compiled = transpile(iteration_info.circuit, simulator, optimization_level=0)
        result = simulator.run(compiled, shots=shots_per_iteration).result()
        counts = result.get_counts()
        measured_bit = 0 if counts.get("0", 0) >= counts.get("1", 0) else 1

        bits.append(measured_bit)
        phase_feedback = iqpe.update_phase_feedback(phase_feedback, measured_bit)

    phase_fraction = iqpe.phase_fraction_from_feedback(phase_feedback)
    return QpeResult.from_phase_fraction(
        method=IterativePhaseEstimation.algorithm,
        phase_fraction=phase_fraction,
        evolution_time=evolution_time,
        bits_msb_first=bits,
        reference_energy=reference_energy,
    )


def test_iterative_phase_estimation_extracts_phase_and_energy(two_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Verify the iterative algorithm recovers the expected phase and energy."""
    result = _run_iterative(two_qubit_phase_problem)

    assert list(result.bits_msb_first or []) == two_qubit_phase_problem.expected_bits
    assert result.phase_fraction == pytest.approx(
        two_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance
    )
    assert result.canonical_phase_fraction == pytest.approx(
        two_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance
    )
    assert result.raw_energy == pytest.approx(two_qubit_phase_problem.expected_energy, abs=qpe_energy_tolerance)
    assert result.resolved_energy == pytest.approx(two_qubit_phase_problem.expected_energy, abs=qpe_energy_tolerance)


def test_iterative_and_traditional_results_match(two_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Confirm iterative and traditional algorithms produce consistent estimates."""
    iterative_result = _run_iterative(two_qubit_phase_problem)
    traditional_result = _run_traditional(two_qubit_phase_problem)

    assert traditional_result.bitstring_msb_first == two_qubit_phase_problem.expected_bitstring
    assert iterative_result.phase_fraction == pytest.approx(
        two_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance
    )
    assert iterative_result.canonical_phase_fraction == pytest.approx(
        two_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance
    )
    assert iterative_result.resolved_energy == pytest.approx(
        two_qubit_phase_problem.expected_energy, abs=qpe_energy_tolerance
    )
    assert iterative_result.canonical_phase_fraction == pytest.approx(
        traditional_result.canonical_phase_fraction,
        abs=qpe_energy_tolerance,
    )
    assert iterative_result.resolved_energy == pytest.approx(
        traditional_result.resolved_energy, abs=qpe_energy_tolerance
    )


def test_iterative_phase_estimation_four_qubit_phase_and_energy(
    four_qubit_phase_problem: PhaseEstimationProblem,
) -> None:
    """Validate phase and energy estimates on the documented four-qubit case."""
    result = _run_iterative(four_qubit_phase_problem)

    assert list(result.bits_msb_first or []) == four_qubit_phase_problem.expected_bits
    assert result.phase_fraction == pytest.approx(
        four_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance
    )
    assert result.canonical_phase_fraction == pytest.approx(
        four_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance
    )
    assert result.raw_energy == pytest.approx(four_qubit_phase_problem.expected_energy, abs=qpe_energy_tolerance)
    assert result.resolved_energy == pytest.approx(four_qubit_phase_problem.expected_energy, abs=qpe_energy_tolerance)


def test_iterative_and_traditional_match_on_four_qubits(four_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Ensure iterative and traditional approaches agree for four qubits."""
    iterative_result = _run_iterative(four_qubit_phase_problem)
    traditional_result = _run_traditional(four_qubit_phase_problem)

    assert traditional_result.bitstring_msb_first == four_qubit_phase_problem.expected_bitstring
    assert iterative_result.phase_fraction == pytest.approx(
        four_qubit_phase_problem.expected_phase, abs=qpe_phase_fraction_tolerance
    )
    assert iterative_result.canonical_phase_fraction == pytest.approx(
        four_qubit_phase_problem.expected_phase,
        abs=qpe_phase_fraction_tolerance,
    )
    assert iterative_result.resolved_energy == pytest.approx(
        four_qubit_phase_problem.expected_energy, abs=qpe_energy_tolerance
    )
    assert iterative_result.canonical_phase_fraction == pytest.approx(
        traditional_result.canonical_phase_fraction,
        abs=qpe_energy_tolerance,
    )
    assert iterative_result.resolved_energy == pytest.approx(
        traditional_result.resolved_energy, abs=qpe_energy_tolerance
    )


def test_iterative_phase_estimation_non_commuting_xi_plus_zz() -> None:
    """Validate IQPE for H = 0.519 XI + ZZ with Hartree-Fock-like trial state."""
    pauli_terms = [("XI", 0.519), ("ZZ", 1.0)]
    state_vector = np.array([0.97, 0.0, np.sqrt(1 - 0.97**2), 0.0], dtype=complex)

    result = _run_iterative_with_parameters(
        pauli_terms,
        state_vector,
        evolution_time=np.pi / 4,
        num_bits=6,
        shots_per_iteration=3,
        seed=42,
        reference_energy=1.1266592208826944,
    )

    assert list(result.bits_msb_first or []) == [1, 0, 0, 1, 0, 0]
    assert result.phase_fraction == pytest.approx(0.140625, abs=qpe_phase_fraction_tolerance)
    assert result.raw_energy == pytest.approx(1.125, abs=qpe_energy_tolerance)
    assert result.resolved_energy == pytest.approx(1.125, abs=qpe_energy_tolerance)


def test_iterative_phase_estimation_second_non_commuting_example() -> None:
    """Validate IQPE for H = -0.0289(X1+X2) + 0.0541(Z1+Z2) + 0.0150 XX + 0.0590 ZZ."""
    pauli_terms = [
        ("XI", -0.0289),
        ("IX", -0.0289),
        ("ZI", 0.0541),
        ("IZ", 0.0541),
        ("XX", 0.0150),
        ("ZZ", 0.0590),
    ]

    state_vector = np.array([0.0, 0.47, 0.47, 0.75], dtype=complex)
    state_vector /= np.linalg.norm(state_vector)

    result = _run_iterative_with_parameters(
        pauli_terms,
        state_vector,
        evolution_time=np.pi / 4,
        num_bits=11,
        shots_per_iteration=3,
        seed=42,
        reference_energy=-0.0887787,
    )

    assert list(result.bits_msb_first or []) == [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    assert result.phase_fraction == pytest.approx(0.988770, abs=qpe_phase_fraction_tolerance)
    assert result.raw_energy == pytest.approx(-0.08984375, abs=qpe_energy_tolerance)
    assert result.resolved_energy == pytest.approx(-0.08984375, abs=qpe_energy_tolerance)
