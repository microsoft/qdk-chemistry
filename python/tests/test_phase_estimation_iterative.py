"""Tests for iterative phase estimation algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qdk_chemistry.algorithms import (
    IterativePhaseEstimation,
    TraditionalPhaseEstimation,
)
from qdk_chemistry.data import QpeResult, QuantumErrorProfile, QubitHamiltonian
from qdk_chemistry.phase_estimation.base import PhaseEstimationAlgorithm
from qdk_chemistry.phase_estimation.iterative_qpe import IterativePhaseEstimationIteration
from qdk_chemistry.plugins.qiskit._interop.noise_model import get_noise_model_from_profile
from qdk_chemistry.utils.phase import accumulated_phase_from_bits

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
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
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
    hamiltonian = QubitHamiltonian(pauli_strings=["XXXX", "ZZZZ"], coefficients=[0.25, 4.5])
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
            circuit_folding=False,
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
    pauli_strings: list[str],
    coefficients: list[float],
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
        pauli_strings: List of Pauli strings defining the Hamiltonian.
        coefficients: List of coefficients defining the Hamiltonian.
        state_vector: Initial state amplitudes for the system register.
        evolution_time: Evolution time ``t`` used in ``U = exp(-i H t)``.
        num_bits: Number of iterative QPE rounds executed.
        shots_per_iteration: Number of simulator shots per iteration circuit.
        seed: PRNG seed for the simulator.
        reference_energy: Optional reference energy used to resolve alias branches.

    Returns:
        :class:`QpeResult` capturing the iterative estimation outcome.

    """
    assert len(pauli_strings) == len(coefficients)

    hamiltonian = QubitHamiltonian(pauli_strings=pauli_strings, coefficients=coefficients)
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
            circuit_folding=False,
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
    assert np.isclose(
        result.phase_fraction,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        result.canonical_phase_fraction,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        result.raw_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert np.isclose(
        result.resolved_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_iterative_and_traditional_results_match(two_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Confirm iterative and traditional algorithms produce consistent estimates."""
    iterative_result = _run_iterative(two_qubit_phase_problem)
    traditional_result = _run_traditional(two_qubit_phase_problem)

    assert traditional_result.bitstring_msb_first == two_qubit_phase_problem.expected_bitstring
    assert np.isclose(
        iterative_result.phase_fraction,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iterative_result.canonical_phase_fraction,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iterative_result.resolved_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert np.isclose(
        iterative_result.canonical_phase_fraction,
        traditional_result.canonical_phase_fraction,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iterative_result.resolved_energy,
        traditional_result.resolved_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_iterative_phase_estimation_four_qubit_phase_and_energy(
    four_qubit_phase_problem: PhaseEstimationProblem,
) -> None:
    """Validate phase and energy estimates on the documented four-qubit case."""
    result = _run_iterative(four_qubit_phase_problem)

    assert list(result.bits_msb_first or []) == four_qubit_phase_problem.expected_bits
    assert np.isclose(
        result.phase_fraction,
        four_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        result.canonical_phase_fraction,
        four_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        result.raw_energy,
        four_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert np.isclose(
        result.resolved_energy,
        four_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


def test_iterative_and_traditional_match_on_four_qubits(four_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Ensure iterative and traditional approaches agree for four qubits."""
    iterative_result = _run_iterative(four_qubit_phase_problem)
    traditional_result = _run_traditional(four_qubit_phase_problem)

    assert traditional_result.bitstring_msb_first == four_qubit_phase_problem.expected_bitstring
    assert np.isclose(
        iterative_result.phase_fraction,
        four_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iterative_result.canonical_phase_fraction,
        four_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iterative_result.resolved_energy,
        four_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )
    assert np.isclose(
        iterative_result.canonical_phase_fraction,
        traditional_result.canonical_phase_fraction,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        iterative_result.resolved_energy,
        traditional_result.resolved_energy,
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
        shots_per_iteration=3,
        seed=42,
        reference_energy=1.1266592208826944,
    )

    assert list(result.bits_msb_first or []) == [1, 0, 0, 1, 0, 0]
    assert np.isclose(
        result.phase_fraction, 0.140625, rtol=float_comparison_relative_tolerance, atol=qpe_phase_fraction_tolerance
    )
    assert np.isclose(result.raw_energy, 1.125, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance)
    assert np.isclose(
        result.resolved_energy, 1.125, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance
    )


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
        shots_per_iteration=3,
        seed=42,
        reference_energy=-0.0887787,
    )

    assert list(result.bits_msb_first or []) == [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    assert np.isclose(
        result.phase_fraction, 0.988770, rtol=float_comparison_relative_tolerance, atol=qpe_phase_fraction_tolerance
    )
    assert np.isclose(
        result.raw_energy, -0.08984375, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance
    )
    assert np.isclose(
        result.resolved_energy, -0.08984375, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance
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
        noiseless_result.resolved_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )

    # Create noise model with depolarizing error on cx gates
    error_rate = 0.1
    error_profile = QuantumErrorProfile(
        name="qpe_noise_test",
        description="Depolarizing noise for QPE integration test",
        errors={
            "cx": {"type": "depolarizing_error", "rate": error_rate, "num_qubits": 2},
        },
    )
    noise_model = get_noise_model_from_profile(error_profile)

    # Run noisy QPE with depolarizing noise on two-qubit gates
    iqpe = IterativePhaseEstimation(two_qubit_phase_problem.hamiltonian, two_qubit_phase_problem.evolution_time)
    simulator = AerSimulator(seed_simulator=_SEED, noise_model=noise_model)
    phase_feedback = 0.0
    bits: list[int] = []

    for iteration in range(two_qubit_phase_problem.num_bits):
        iteration_data = iqpe.create_iteration(
            two_qubit_phase_problem.state_prep,
            iteration=iteration,
            total_iterations=two_qubit_phase_problem.num_bits,
            phase_correction=phase_feedback,
            circuit_folding=False,
        )
        # Run noisy simulation with more shots to see noise impact despite statistics
        circuit = iteration_data.circuit.decompose(reps=4)
        transpiled_circuit = transpile(
            circuit, basis_gates=["cx", "rz", "h", "x", "s", "sdg", "crz"], optimization_level=1
        )
        result = simulator.run(transpiled_circuit, shots=100).result()
        counts = result.get_counts()
        measured_bit = 0 if counts.get("0", 0) >= counts.get("1", 0) else 1

        bits.append(measured_bit)
        phase_feedback = iqpe.update_phase_feedback(phase_feedback, measured_bit)

    phase_fraction = iqpe.phase_fraction_from_feedback(phase_feedback)
    noisy_result = QpeResult.from_phase_fraction(
        method=IterativePhaseEstimation.algorithm,
        phase_fraction=phase_fraction,
        evolution_time=two_qubit_phase_problem.evolution_time,
        bits_msb_first=bits,
        reference_energy=two_qubit_phase_problem.expected_energy,
        metadata={"label": two_qubit_phase_problem.label, "noise_model": "depolarizing", "error_rate": error_rate},
    )

    # Verify that noisy results deviate from expected values
    assert noisy_result.bits_msb_first is not None
    assert not np.isclose(
        noisy_result.phase_fraction,
        two_qubit_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert not np.isclose(
        noisy_result.resolved_energy,
        two_qubit_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


# Tests for create_iterations method
def test_create_iterations_generates_correct_number_of_circuits(
    two_qubit_phase_problem: PhaseEstimationProblem,
) -> None:
    """Test that create_iterations generates the correct number of iteration circuits."""
    iqpe = IterativePhaseEstimation(two_qubit_phase_problem.hamiltonian, two_qubit_phase_problem.evolution_time)

    iterations = iqpe.create_iterations(two_qubit_phase_problem.state_prep, num_bits=5, circuit_folding=False)

    assert len(iterations) == 5
    for idx, iteration in enumerate(iterations):
        assert iteration.iteration == idx
        assert iteration.total_iterations == 5


def test_create_iterations_with_phase_corrections() -> None:
    """Test create_iterations with custom phase corrections."""
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ"], coefficients=[1.0])
    state_prep = QuantumCircuit(2)
    state_prep.h([0, 1])

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    phase_corrections = [0.0, np.pi / 4, np.pi / 2, np.pi]
    iterations = iqpe.create_iterations(
        state_prep, num_bits=4, phase_corrections=phase_corrections, circuit_folding=False
    )

    assert len(iterations) == 4
    for idx, iteration in enumerate(iterations):
        assert iteration.phase_correction == phase_corrections[idx]


def test_create_iterations_with_custom_measurement_registers() -> None:
    """Test create_iterations with custom measurement registers."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XX"], coefficients=[0.5])
    state_prep = QuantumCircuit(2)
    state_prep.h([0, 1])

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi / 2)

    custom_registers = [ClassicalRegister(1, f"custom_{i}") for i in range(3)]
    iterations = iqpe.create_iterations(
        state_prep, num_bits=3, measurement_registers=custom_registers, circuit_folding=False
    )

    assert len(iterations) == 3
    for idx, iteration in enumerate(iterations):
        assert custom_registers[idx] in iteration.circuit.cregs


def test_create_iterations_with_iteration_names() -> None:
    """Test create_iterations with custom iteration names."""
    hamiltonian = QubitHamiltonian(pauli_strings=["ZI"], coefficients=[1.0])
    state_prep = QuantumCircuit(2)
    state_prep.h(0)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi / 4)

    names = ["iteration_0", "iteration_1", "iteration_2"]
    iterations = iqpe.create_iterations(state_prep, num_bits=3, iteration_names=names, circuit_folding=False)

    assert len(iterations) == 3
    for idx, iteration in enumerate(iterations):
        assert iteration.circuit.name == names[idx]


def test_create_iterations_invalid_num_bits() -> None:
    """Test that create_iterations raises ValueError for non-positive num_bits."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    with pytest.raises(ValueError, match="num_bits must be a positive integer"):
        iqpe.create_iterations(state_prep, num_bits=0)

    with pytest.raises(ValueError, match="num_bits must be a positive integer"):
        iqpe.create_iterations(state_prep, num_bits=-1)


def test_create_iterations_mismatched_phase_corrections_length() -> None:
    """Test that create_iterations raises ValueError when phase_corrections length doesn't match num_bits."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    with pytest.raises(ValueError, match="phase_corrections must have length equal to num_bits"):
        iqpe.create_iterations(state_prep, num_bits=3, phase_corrections=[0.0, 0.0])


def test_create_iterations_mismatched_measurement_registers_length() -> None:
    """Test that create_iterations raises ValueError when measurement_registers length doesn't match num_bits."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    with pytest.raises(ValueError, match="measurement_registers must have length equal to num_bits"):
        iqpe.create_iterations(
            state_prep, num_bits=3, measurement_registers=[ClassicalRegister(1, "c0"), ClassicalRegister(1, "c1")]
        )


def test_create_iterations_mismatched_iteration_names_length() -> None:
    """Test that create_iterations raises ValueError when iteration_names length doesn't match num_bits."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    with pytest.raises(ValueError, match="iteration_names must have length equal to num_bits"):
        iqpe.create_iterations(state_prep, num_bits=3, iteration_names=["name1", "name2"])


# Tests for static methods
def test_update_phase_feedback_with_bit_zero() -> None:
    """Test phase feedback update when measured bit is 0."""
    current_phase = np.pi / 4
    new_phase = IterativePhaseEstimation.update_phase_feedback(current_phase, 0)

    # When bit is 0, phase should be halved
    assert np.isclose(new_phase, current_phase / 2, rtol=float_comparison_relative_tolerance)


def test_phase_fraction_from_feedback_zero() -> None:
    """Test phase fraction calculation from zero feedback."""
    phase_fraction = IterativePhaseEstimation.phase_fraction_from_feedback(0.0)
    assert np.isclose(phase_fraction, 0.0, rtol=float_comparison_relative_tolerance)


def test_phase_fraction_from_feedback_in_valid_range() -> None:
    """Test phase fraction calculation from feedback in valid range."""
    feedback = np.pi / 2
    phase_fraction = IterativePhaseEstimation.phase_fraction_from_feedback(feedback)

    # Should be in range [0, 1)
    assert 0.0 <= phase_fraction < 1.0


def test_phase_feedback_from_bits_empty() -> None:
    """Test phase feedback calculation from empty bit sequence."""
    phase_feedback = IterativePhaseEstimation.phase_feedback_from_bits([])
    assert np.isclose(phase_feedback, 0.0, rtol=float_comparison_relative_tolerance)


def test_phase_feedback_from_bits_single_zero() -> None:
    """Test phase feedback calculation from single zero bit."""
    phase_feedback = IterativePhaseEstimation.phase_feedback_from_bits([0])
    assert np.isclose(phase_feedback, 0.0, rtol=float_comparison_relative_tolerance)


def test_phase_feedback_from_bits_multiple() -> None:
    """Test phase feedback calculation from multiple bits."""
    bits = [1, 0, 1, 1]
    phase_feedback = IterativePhaseEstimation.phase_feedback_from_bits(bits)

    # Verify it's equivalent to accumulated phase
    expected = accumulated_phase_from_bits(bits)
    assert np.isclose(phase_feedback, expected, rtol=float_comparison_relative_tolerance)


# Tests for IterativePhaseEstimationIteration dataclass
def test_iteration_dataclass_creation() -> None:
    """Test creation of IterativePhaseEstimationIteration dataclass."""
    circuit = QuantumCircuit(3, 1)
    iteration_obj = IterativePhaseEstimationIteration(
        circuit=circuit, iteration=0, total_iterations=4, power=8, phase_correction=0.0
    )

    assert iteration_obj.circuit == circuit
    assert iteration_obj.iteration == 0
    assert iteration_obj.total_iterations == 4
    assert iteration_obj.power == 8
    assert iteration_obj.phase_correction == 0.0


def test_iteration_dataclass_frozen() -> None:
    """Test that IterativePhaseEstimationIteration is frozen."""
    circuit = QuantumCircuit(2, 1)
    iteration_obj = IterativePhaseEstimationIteration(
        circuit=circuit, iteration=1, total_iterations=5, power=4, phase_correction=np.pi / 4
    )

    # Should not be able to modify frozen dataclass
    with pytest.raises(
        (AttributeError, TypeError)
    ):  # FrozenInstanceError in Python 3.10+, AttributeError in older versions
        iteration_obj.iteration = 2


def test_create_iteration_returns_correct_metadata() -> None:
    """Test that create_iteration returns correct metadata."""
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ"], coefficients=[1.0])
    state_prep = QuantumCircuit(2)
    state_prep.h([0, 1])

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi / 2)

    iteration_obj = iqpe.create_iteration(state_prep, iteration=2, total_iterations=6, phase_correction=np.pi / 8)

    assert iteration_obj.iteration == 2
    assert iteration_obj.total_iterations == 6
    assert iteration_obj.power == 2 ** (6 - 2 - 1)  # 2^3 = 8
    assert np.isclose(iteration_obj.phase_correction, np.pi / 8)
    assert iteration_obj.circuit is not None


# Tests for validation and error handling
def test_create_iteration_circuit_invalid_iteration_negative() -> None:
    """Test that create_iteration_circuit raises ValueError for negative iteration."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    with pytest.raises(ValueError, match="iteration index -1 is outside the valid range"):
        iqpe.create_iteration_circuit(state_prep, iteration=-1, total_iterations=4)


def test_create_iteration_circuit_invalid_iteration_too_large() -> None:
    """Test that create_iteration_circuit raises ValueError for iteration >= total_iterations."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    with pytest.raises(ValueError, match="iteration index 4 is outside the valid range"):
        iqpe.create_iteration_circuit(state_prep, iteration=4, total_iterations=4)


def test_create_iteration_circuit_invalid_total_iterations_zero() -> None:
    """Test that create_iteration_circuit raises ValueError for total_iterations <= 0."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    with pytest.raises(ValueError, match="total_iterations must be a positive integer"):
        iqpe.create_iteration_circuit(state_prep, iteration=0, total_iterations=0)


def test_create_iteration_circuit_invalid_total_iterations_negative() -> None:
    """Test that create_iteration_circuit raises ValueError for negative total_iterations."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    with pytest.raises(ValueError, match="total_iterations must be a positive integer"):
        iqpe.create_iteration_circuit(state_prep, iteration=0, total_iterations=-1)


def test_create_iteration_circuit_power_calculation() -> None:
    """Test that the power calculation is correct for different iterations."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    # For 5 iterations (num_bits=5), powers should be: 16, 8, 4, 2, 1
    expected_powers = [16, 8, 4, 2, 1]
    for iteration, expected_power in enumerate(expected_powers):
        iteration_obj = iqpe.create_iteration(state_prep, iteration=iteration, total_iterations=5)
        assert iteration_obj.power == expected_power


def test_create_iteration_circuit_with_none_measurement_register() -> None:
    """Test create_iteration_circuit with None measurement register creates default."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    circuit = iqpe.create_iteration_circuit(state_prep, iteration=2, total_iterations=5, measurement_register=None)

    # Should have created a default classical register named "c2"
    assert circuit.num_clbits == 1
    assert any(creg.name == "c2" for creg in circuit.cregs)


def test_iterative_qpe_initialization() -> None:
    """Test IterativePhaseEstimation initialization."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.5, 1.0])
    evolution_time = 2.5

    iqpe = IterativePhaseEstimation(hamiltonian, evolution_time)

    assert iqpe.hamiltonian == hamiltonian
    assert iqpe.evolution_time == evolution_time
    assert iqpe.algorithm == PhaseEstimationAlgorithm.ITERATIVE


def test_create_folding_circuits() -> None:
    """Test circuit folding in create_iteration_circuit."""
    hamiltonian = QubitHamiltonian(pauli_strings=["Z"], coefficients=[1.0])
    state_prep = QuantumCircuit(1)
    state_prep.h(0)

    iqpe = IterativePhaseEstimation(hamiltonian, np.pi)

    # Create circuit without folding
    circuit_no_folding = iqpe.create_iteration_circuit(
        state_prep, iteration=1, total_iterations=3, circuit_folding=False
    )

    # Create circuit with folding
    circuit_with_folding = iqpe.create_iteration_circuit(
        state_prep, iteration=1, total_iterations=3, circuit_folding=True
    )

    # The folded circuit should have state_prep operations included
    assert "state_prep" not in circuit_no_folding.count_ops()
    assert "state_prep" in circuit_with_folding.count_ops()
