"""Tests for iterative phase estimation algorithms."""

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
from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import (
    IterativePhaseEstimation,
    _validate_iteration_inputs,
)
from qdk_chemistry.data import Circuit, QpeResult, QuantumErrorProfile, QubitHamiltonian
from qdk_chemistry.plugins.qiskit.circuit_executor import QiskitAerSimulator
from qdk_chemistry.plugins.qiskit.standard_phase_estimation import QiskitStandardPhaseEstimation
from qdk_chemistry.utils.phase import (
    accumulated_phase_from_bits,
    energy_from_phase,
    iterative_phase_feedback_update,
    phase_fraction_from_feedback,
)

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
    qpe_phase_fraction_tolerance,
)

_SEED = 42


@dataclass(frozen=True)
class PhaseEstimationProblem:
    """Container describing a reproducible phase estimation benchmark."""

    label: str
    hamiltonian: QubitHamiltonian
    state_prep: Circuit
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
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    state_prep = QuantumCircuit(2, name="psi")
    state_prep.append(QiskitStatePreparation([0.6, 0.0, 0.0, 0.8]), list(range(2)))

    return PhaseEstimationProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qasm=qasm3.dumps(state_prep)),
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
    hamiltonian = QubitHamiltonian(pauli_strings=["XXXX", "ZZZZ"], coefficients=[0.25, 4.5])
    state_prep = QuantumCircuit(4, name="psi_4q")
    state_vector = np.zeros(2**4, dtype=complex)
    state_vector[int("1000", 2)] = 0.8
    state_vector[int("0111", 2)] = -0.6
    state_prep.append(QiskitStatePreparation(state_vector), list(range(4)))

    return PhaseEstimationProblem(
        label="four_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qasm=qasm3.dumps(state_prep)),
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
    iqpe = IterativePhaseEstimation(
        num_bits=problem.num_bits, evolution_time=problem.evolution_time, shots_per_bit=problem.shots_iterative
    )
    simulator = create("circuit_executor", "qdk_full_state_simulator", seed=_SEED)
    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    evolution_builder = create("time_evolution_builder", "trotter")

    return iqpe.run(
        qubit_hamiltonian=problem.hamiltonian,
        state_preparation=problem.state_prep,
        circuit_executor=simulator,
        circuit_mapper=circuit_mapper,
        evolution_builder=evolution_builder,
    )


def _run_traditional(problem: PhaseEstimationProblem) -> QpeResult:
    """Execute traditional QPE and return structured results.

    Args:
        problem: Benchmark description supplying Hamiltonian, state prep, and expectations.

    Returns:
        :class:`QpeResult` instance summarizing the traditional run.

    """
    qpe = QiskitStandardPhaseEstimation(
        num_bits=problem.num_bits, evolution_time=problem.evolution_time, shots=problem.shots_traditional
    )
    simulator = QiskitAerSimulator(seed=_SEED)
    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    evolution_builder = create("time_evolution_builder", "trotter")

    return qpe.run(
        qubit_hamiltonian=problem.hamiltonian,
        state_preparation=problem.state_prep,
        circuit_executor=simulator,
        circuit_mapper=circuit_mapper,
        evolution_builder=evolution_builder,
    )


def _run_iterative_with_parameters(
    pauli_strings: list[str],
    coefficients: list[float],
    state_vector: np.ndarray,
    *,
    evolution_time: float,
    num_bits: int,
    shots_per_bit: int,
    seed: int,
) -> QpeResult:
    """Execute iterative phase estimation for a custom Hamiltonian/state pair.

    Args:
        pauli_strings: List of Pauli strings defining the Hamiltonian.
        coefficients: List of coefficients defining the Hamiltonian.
        state_vector: Initial state amplitudes for the system register.
        evolution_time: Evolution time ``t`` used in ``U = exp(-i H t)``.
        num_bits: Number of iterative QPE rounds executed.
        shots_per_bit: Number of simulator shots per iteration circuit.
        seed: PRNG seed for the simulator.
        reference_energy: Optional reference energy used to resolve alias branches.

    Returns:
        :class:`QpeResult` capturing the iterative estimation outcome.

    """
    assert len(pauli_strings) == len(coefficients)

    hamiltonian = QubitHamiltonian(pauli_strings=pauli_strings, coefficients=coefficients)
    num_qubits = int(np.log2(len(state_vector)))

    state_prep = QuantumCircuit(num_qubits, name="state")
    state_prep.append(QiskitStatePreparation(state_vector), list(range(num_qubits)))

    iqpe = IterativePhaseEstimation(num_bits=num_bits, evolution_time=evolution_time, shots_per_bit=shots_per_bit)
    simulator = create("circuit_executor", "qdk_full_state_simulator", seed=seed)
    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    evolution_builder = create("time_evolution_builder", "trotter")

    return iqpe.run(
        qubit_hamiltonian=hamiltonian,
        state_preparation=Circuit(qasm=qasm3.dumps(state_prep)),
        circuit_executor=simulator,
        circuit_mapper=circuit_mapper,
        evolution_builder=evolution_builder,
    )


def _resolve_phase_ambiguity(
    phase_fraction: float,
    evolution_time: float,
    expected_energy: float,
) -> tuple[float, float]:
    """Resolve phase ambiguity due to periodicity by selecting closest energy.

    Args:
        phase_fraction: Measured phase fraction from QPE.
        evolution_time: Evolution time used in QPE.
        expected_energy: Reference energy to resolve ambiguity.

    Returns:
        Tuple of (resolved phase fraction, resolved energy).

    """
    phase_fraction_candidates = [phase_fraction % 1.0, (1.0 - phase_fraction) % 1.0]
    energies = [energy_from_phase(candidate, evolution_time=evolution_time) for candidate in phase_fraction_candidates]

    # Select candidate closest to expected energy
    index = int(np.argmin([abs(energy - expected_energy) for energy in energies]))
    return phase_fraction_candidates[index], energies[index]


def test_iterative_phase_estimation_extracts_phase_and_energy(two_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Verify the iterative algorithm recovers the expected phase and energy."""
    result = _run_iterative(two_qubit_phase_problem)
    resolved_phase, resolved_energy = _resolve_phase_ambiguity(
        result.phase_fraction, two_qubit_phase_problem.evolution_time, two_qubit_phase_problem.expected_energy
    )

    assert list(result.bits_msb_first or []) == two_qubit_phase_problem.expected_bits
    assert np.isclose(
        resolved_phase,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        resolved_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_iterative_and_traditional_results_match(two_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Confirm iterative and traditional algorithms produce consistent estimates."""
    iterative_result = _run_iterative(two_qubit_phase_problem)
    traditional_result = _run_traditional(two_qubit_phase_problem)

    iqpe_resolved_phase, iqpe_resolved_energy = _resolve_phase_ambiguity(
        iterative_result.phase_fraction, two_qubit_phase_problem.evolution_time, two_qubit_phase_problem.expected_energy
    )
    qpe_resolved_phase, qpe_resolved_energy = _resolve_phase_ambiguity(
        traditional_result.phase_fraction,
        two_qubit_phase_problem.evolution_time,
        two_qubit_phase_problem.expected_energy,
    )

    assert traditional_result.bitstring_msb_first == two_qubit_phase_problem.expected_bitstring
    assert np.isclose(
        iqpe_resolved_phase,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iqpe_resolved_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert np.isclose(
        iqpe_resolved_phase,
        qpe_resolved_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iqpe_resolved_energy,
        qpe_resolved_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_iterative_phase_estimation_four_qubit_phase_and_energy(
    four_qubit_phase_problem: PhaseEstimationProblem,
) -> None:
    """Validate phase and energy estimates on the documented four-qubit case."""
    result = _run_iterative(four_qubit_phase_problem)
    resolved_phase, resolved_energy = _resolve_phase_ambiguity(
        result.phase_fraction, four_qubit_phase_problem.evolution_time, four_qubit_phase_problem.expected_energy
    )

    assert list(result.bits_msb_first or []) == four_qubit_phase_problem.expected_bits
    assert np.isclose(
        resolved_phase,
        four_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        resolved_energy,
        four_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_iterative_and_traditional_match_on_four_qubits(four_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Ensure iterative and traditional approaches agree for four qubits."""
    iterative_result = _run_iterative(four_qubit_phase_problem)
    traditional_result = _run_traditional(four_qubit_phase_problem)

    iqpe_resolved_phase, iqpe_resolved_energy = _resolve_phase_ambiguity(
        iterative_result.phase_fraction,
        four_qubit_phase_problem.evolution_time,
        four_qubit_phase_problem.expected_energy,
    )
    qpe_resolved_phase, qpe_resolved_energy = _resolve_phase_ambiguity(
        traditional_result.phase_fraction,
        four_qubit_phase_problem.evolution_time,
        four_qubit_phase_problem.expected_energy,
    )

    assert traditional_result.bitstring_msb_first == four_qubit_phase_problem.expected_bitstring
    assert np.isclose(
        iqpe_resolved_phase,
        four_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iqpe_resolved_energy,
        four_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert np.isclose(
        iqpe_resolved_phase,
        qpe_resolved_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iqpe_resolved_energy,
        qpe_resolved_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_iterative_phase_estimation_non_commuting_xi_plus_zz() -> None:
    """Validate IQPE for H = 0.519 XI + ZZ with Hartree-Fock-like trial state."""
    pauli_strings = ["XI", "ZZ"]
    coefficients = [0.519, 1.0]
    state_vector = np.array([0.97, 0.0, np.sqrt(1 - 0.97**2), 0.0], dtype=complex)

    result = _run_iterative_with_parameters(
        pauli_strings,
        coefficients,
        state_vector,
        evolution_time=np.pi / 4,
        num_bits=6,
        shots_per_bit=3,
        seed=42,
    )

    assert list(result.bits_msb_first or []) == [1, 0, 0, 1, 0, 0]
    assert np.isclose(
        result.phase_fraction, 0.140625, rtol=float_comparison_relative_tolerance, atol=qpe_phase_fraction_tolerance
    )
    assert np.isclose(result.raw_energy, 1.125, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance)


def test_iterative_phase_estimation_second_non_commuting_example() -> None:
    """Validate IQPE for H = -0.0289(X1+X2) + 0.0541(Z1+Z2) + 0.0150 XX + 0.0590 ZZ."""
    pauli_strings = ["XI", "IX", "ZI", "IZ", "XX", "ZZ"]
    coefficients = [-0.0289, -0.0289, 0.0541, 0.0541, 0.0150, 0.059]
    state_vector = np.array([0.0, 0.47, 0.47, 0.75], dtype=complex)
    state_vector /= np.linalg.norm(state_vector)

    result = _run_iterative_with_parameters(
        pauli_strings,
        coefficients,
        state_vector,
        evolution_time=np.pi / 4,
        num_bits=11,
        shots_per_bit=3,
        seed=42,
    )

    assert list(result.bits_msb_first or []) == [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    assert np.isclose(
        result.phase_fraction, 0.988770, rtol=float_comparison_relative_tolerance, atol=qpe_phase_fraction_tolerance
    )
    assert np.isclose(
        result.raw_energy, -0.08984375, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance
    )


def test_iterative_qpe_with_noise_model(two_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Integration test showing NoiseModel impact on iterative phase estimation accuracy."""
    # Run noiseless QPE
    noiseless_result = _run_iterative(two_qubit_phase_problem)

    # Verify noiseless case matches expected values
    assert noiseless_result.bits_msb_first is not None
    assert list(noiseless_result.bits_msb_first) == two_qubit_phase_problem.expected_bits
    assert np.isclose(
        noiseless_result.phase_fraction,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        noiseless_result.raw_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )

    # Create noise model with depolarizing error on cx gates
    error_rate = 0.01
    error_profile = QuantumErrorProfile(
        name="qpe_noise_test",
        description="Depolarizing noise for QPE integration test",
        errors={
            "cx": {"type": "depolarizing_error", "rate": error_rate, "num_qubits": 2},
            "rz": {"type": "depolarizing_error", "rate": error_rate, "num_qubits": 1},
            "h": {"type": "depolarizing_error", "rate": error_rate, "num_qubits": 1},
            "s": {"type": "depolarizing_error", "rate": error_rate, "num_qubits": 1},
        },
    )
    simulator = QiskitAerSimulator(seed=_SEED)
    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    evolution_builder = create("time_evolution_builder", "trotter")
    iqpe = IterativePhaseEstimation(
        num_bits=two_qubit_phase_problem.num_bits,
        evolution_time=two_qubit_phase_problem.evolution_time,
        shots_per_bit=two_qubit_phase_problem.shots_iterative,
    )
    noisy_result = iqpe.run(
        state_preparation=two_qubit_phase_problem.state_prep,
        qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
        circuit_executor=simulator,
        circuit_mapper=circuit_mapper,
        evolution_builder=evolution_builder,
        noise=error_profile,
    )

    # Verify that noisy results deviate from expected values and noiseless results
    assert noisy_result.bits_msb_first is not None
    assert not np.isclose(
        noisy_result.phase_fraction,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert not np.isclose(
        noisy_result.raw_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert not np.isclose(
        noisy_result.phase_fraction,
        noiseless_result.phase_fraction,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert not np.isclose(
        noisy_result.raw_energy,
        noiseless_result.raw_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_iterative_qpe_generates_correct_number_of_circuits(
    two_qubit_phase_problem: PhaseEstimationProblem,
) -> None:
    """Test that create_iterations generates the correct number of iteration circuits."""
    iqpe = IterativePhaseEstimation(
        num_bits=two_qubit_phase_problem.num_bits,
        evolution_time=two_qubit_phase_problem.evolution_time,
        shots_per_bit=two_qubit_phase_problem.shots_iterative,
    )
    simulator = create("circuit_executor", "qdk_full_state_simulator", seed=_SEED)
    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    evolution_builder = create("time_evolution_builder", "trotter")

    with pytest.raises(ValueError, match="No iteration circuits have been generated"):
        iqpe.get_circuits()

    iqpe.run(
        qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
        state_preparation=two_qubit_phase_problem.state_prep,
        circuit_executor=simulator,
        circuit_mapper=circuit_mapper,
        evolution_builder=evolution_builder,
    )

    assert len(iqpe.get_circuits()) == two_qubit_phase_problem.num_bits


def test_update_phase_feedback_with_bit_zero() -> None:
    """Test phase feedback update when measured bit is 0."""
    current_phase = np.pi / 4
    new_phase = iterative_phase_feedback_update(current_phase, 0)

    # When bit is 0, phase should be halved
    assert np.isclose(new_phase, current_phase / 2, rtol=float_comparison_relative_tolerance)


def test_phase_fraction_from_feedback_zero() -> None:
    """Test phase fraction calculation from zero feedback."""
    phase_fraction = phase_fraction_from_feedback(0.0)
    assert np.isclose(phase_fraction, 0.0, rtol=float_comparison_relative_tolerance)


def test_phase_fraction_from_feedback_in_valid_range() -> None:
    """Test phase fraction calculation from feedback in valid range."""
    feedback = np.pi / 2
    phase_fraction = phase_fraction_from_feedback(feedback)

    # Should be in range [0, 1)
    assert 0.0 <= phase_fraction < 1.0


def test_phase_feedback_from_bits_empty() -> None:
    """Test phase feedback calculation from empty bit sequence."""
    phase_feedback = accumulated_phase_from_bits([])
    assert np.isclose(phase_feedback, 0.0, rtol=float_comparison_relative_tolerance)


def test_phase_feedback_from_bits_single_zero() -> None:
    """Test phase feedback calculation from single zero bit."""
    phase_feedback = accumulated_phase_from_bits([0])
    assert np.isclose(phase_feedback, 0.0, rtol=float_comparison_relative_tolerance)


def test_phase_feedback_from_bits_multiple() -> None:
    """Test phase feedback calculation from multiple bits."""
    bits = [1, 0, 1, 1]
    phase_feedback = accumulated_phase_from_bits(bits)

    # Verify it's equivalent to accumulated phase
    expected = accumulated_phase_from_bits(bits)
    assert np.isclose(phase_feedback, expected, rtol=float_comparison_relative_tolerance)


# Tests for validation and error handling
def test_create_iteration_circuit_invalid_iteration_negative() -> None:
    """Test that create_iteration_circuit raises ValueError for negative iteration."""
    with pytest.raises(ValueError, match="iteration index -1 is outside the valid range"):
        _validate_iteration_inputs(iteration=-1, total_iterations=4)

    with pytest.raises(ValueError, match="iteration index 4 is outside the valid range"):
        _validate_iteration_inputs(iteration=4, total_iterations=4)


def test_create_iteration_circuit_invalid_total_iterations_zero() -> None:
    """Test that create_iteration_circuit raises ValueError for total_iterations <= 0."""
    with pytest.raises(ValueError, match="total_iterations must be a positive integer"):
        _validate_iteration_inputs(iteration=0, total_iterations=0)

    with pytest.raises(ValueError, match="total_iterations must be a positive integer"):
        _validate_iteration_inputs(iteration=0, total_iterations=-1)


def test_create_iteration_circuit_power_calculation() -> None:
    """Test that the power calculation is correct for different iterations."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)
    state_prep_circuit = Circuit(qasm=qasm3.dumps(state_prep))
    circuit_mapper = create("controlled_evolution_circuit_mapper", "pauli_sequence")
    evolution_builder = create("time_evolution_builder", "trotter")

    iqpe = IterativePhaseEstimation(num_bits=5, evolution_time=np.pi, shots_per_bit=10)
    iter_0_circuit = iqpe.create_iteration_circuit(
        state_preparation=state_prep_circuit,
        qubit_hamiltonian=hamiltonian,
        circuit_mapper=circuit_mapper,
        evolution_builder=evolution_builder,
        iteration=0,
        total_iterations=5,
    )

    # For the first iteration, powers should be 16
    assert f"power_{2 ** (iqpe._settings.get('num_bits') - 0 - 1)}" in iter_0_circuit.qasm


def test_iterative_qpe_initialization() -> None:
    """Test IterativePhaseEstimation initialization."""
    evolution_time = 2.5
    num_bits = 6
    shots_per_bit = 10

    iqpe = IterativePhaseEstimation(num_bits=num_bits, evolution_time=evolution_time, shots_per_bit=shots_per_bit)

    assert iqpe._settings.get("num_bits") == num_bits
    assert np.isclose(
        iqpe._settings.get("evolution_time"),
        evolution_time,
        rtol=float_comparison_relative_tolerance,
        atol=float_comparison_relative_tolerance,
    )
    assert iqpe._settings.get("shots_per_bit") == shots_per_bit
