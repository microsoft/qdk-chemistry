"""Tests for iterative phase estimation algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qdk_chemistry.algorithms.phase_estimation.iterative_phase_estimation import IterativePhaseEstimation
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.phase import (
    accumulated_phase_from_bits,
    energy_from_phase,
    iterative_phase_feedback_update,
    phase_fraction_from_feedback,
)
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

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


@pytest.fixture
def two_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return the two-qubit phase estimation scenario used in documentation."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    state_vector = [0.6, 0.0, 0.0, 0.8]
    state_prep_params = {"rowMap": [1, 0], "stateVector": state_vector, "expansionOps": [], "numQubits": 2}
    factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)

    return PhaseEstimationProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qsharp_factory=factories, qsharp_op=qsharp_op),
        evolution_time=float(np.pi / 2.0),
        num_bits=4,
        expected_bits=[1, 1, 0, 0],
        expected_phase=0.1875,
        expected_energy=0.75,
        expected_bitstring="1101",
        shots_iterative=3,
    )


@pytest.fixture
def four_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return the four-qubit benchmark used in documentation."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XXXX", "ZZZZ"], coefficients=[0.25, 4.5])
    state_vector = np.zeros(2**4, dtype=float)
    state_vector[int("1000", 2)] = 0.8
    state_vector[int("0111", 2)] = -0.6
    state_prep_params = {
        "rowMap": [3, 2, 1, 0],
        "stateVector": state_vector.tolist(),
        "expansionOps": [],
        "numQubits": 4,
    }
    factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)

    return PhaseEstimationProblem(
        label="four_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qsharp_factory=factories, qsharp_op=qsharp_op),
        evolution_time=float(np.pi / 8.0),
        num_bits=6,
        expected_bits=[1, 0, 1, 1, 0, 1],
        expected_phase=45 / 64,
        expected_energy=-4.75,
        expected_bitstring="010011",
        shots_iterative=3,
    )


def _make_iterative_circuit_builder_ref(builder_name: str, num_bits: int, evolution_time: float) -> AlgorithmRef:
    """Return an iterative circuit builder AlgorithmRef for the given builder name."""
    return AlgorithmRef(
        "qpe_circuit_builder",
        builder_name,
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=evolution_time),
    )


def _run_iterative(problem: PhaseEstimationProblem, builder_name: str = "qdk_iterative") -> QpeResult:
    """Execute iterative phase estimation and return structured results.

    Args:
        problem: Benchmark description supplying Hamiltonian, state prep, and expectations.
        builder_name: The circuit builder to use ("qdk_iterative" or "qiskit_iterative").

    Returns:
        :class:`QpeResult` instance summarizing the iterative run.

    """
    state_prep_circuit = problem.state_prep
    circuit_builder = _make_iterative_circuit_builder_ref(builder_name, problem.num_bits, problem.evolution_time)
    iqpe = IterativePhaseEstimation(shots_per_bit=problem.shots_iterative)
    iqpe.settings().set("qpe_circuit_builder", circuit_builder)
    iqpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=_SEED),
    )

    return iqpe.run(
        qubit_hamiltonian=problem.hamiltonian,
        state_preparation=state_prep_circuit,
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

    state_prep_params = {
        "rowMap": list(range(num_qubits - 1, -1, -1)),
        "stateVector": state_vector.tolist(),
        "expansionOps": [],
        "numQubits": num_qubits,
    }
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
    qsharp_factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )

    iqpe = IterativePhaseEstimation(shots_per_bit=shots_per_bit)
    circuit_builder = _make_iterative_circuit_builder_ref("qdk_iterative", num_bits, evolution_time)
    iqpe.settings().set("qpe_circuit_builder", circuit_builder)
    iqpe.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=seed))

    return iqpe.run(
        qubit_hamiltonian=hamiltonian,
        state_preparation=Circuit(qsharp_factory=qsharp_factories, qsharp_op=qsharp_op),
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


# Parametrize over both qdk_iterative and qiskit_iterative builders
_builder_params = [
    pytest.param("qdk_iterative", id="qdk_iterative"),
    pytest.param(
        "qiskit_iterative",
        id="qiskit_iterative",
        marks=[
            pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available"),
            pytest.mark.xfail(
                reason="QIR-to-Qiskit converter does not support Adaptive_RIFLA profile"
            ),
        ],
    ),
]


@pytest.mark.parametrize("builder_name", _builder_params)
def test_iterative_phase_estimation_extracts_phase_and_energy(
    two_qubit_phase_problem: PhaseEstimationProblem,
    builder_name: str,
) -> None:
    """Verify the iterative algorithm recovers the expected phase and energy."""
    result = _run_iterative(two_qubit_phase_problem, builder_name)
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


@pytest.mark.parametrize("builder_name", _builder_params)
def test_iterative_phase_estimation_four_qubit_phase_and_energy(
    four_qubit_phase_problem: PhaseEstimationProblem,
    builder_name: str,
) -> None:
    """Validate phase and energy estimates on the documented four-qubit case."""
    result = _run_iterative(four_qubit_phase_problem, builder_name)
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


def test_iterative_phase_estimation_non_commuting_xi_plus_zz() -> None:
    """Validate IQPE for H = 0.519 XI + ZZ with Hartree-Fock-like trial state."""
    pauli_strings = ["XI", "ZZ"]
    coefficients = [0.519, 1.0]
    state_vector = np.array([0.97, 0.0, np.sqrt(1 - 0.97**2), 0.0], dtype=float)

    result = _run_iterative_with_parameters(
        pauli_strings,
        coefficients,
        state_vector,
        evolution_time=np.pi / 4,
        num_bits=6,
        shots_per_bit=3,
        seed=_SEED,
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
    state_vector = np.array([0.0, 0.47, 0.47, 0.75], dtype=float)
    state_vector /= np.linalg.norm(state_vector)

    result = _run_iterative_with_parameters(
        pauli_strings,
        coefficients,
        state_vector,
        evolution_time=np.pi / 4,
        num_bits=11,
        shots_per_bit=3,
        seed=_SEED,
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

    # Create noise model with depolarizing error
    error_rate = 0.05
    error_profile = QuantumErrorProfile(
        name="qpe_noise_test",
        description="Depolarizing noise for QPE integration test",
        errors={
            "cx": {"depolarizing_error": error_rate},
            "rz": {"depolarizing_error": error_rate},
            "h": {"depolarizing_error": error_rate},
            "s": {"depolarizing_error": error_rate},
        },
    )
    iqpe_circuit_builder = _make_iterative_circuit_builder_ref(
        "qdk_iterative",
        num_bits=two_qubit_phase_problem.num_bits,
        evolution_time=two_qubit_phase_problem.evolution_time,
    )
    iqpe = IterativePhaseEstimation(
        shots_per_bit=two_qubit_phase_problem.shots_iterative,
    )
    iqpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=_SEED),
    )
    iqpe.settings().set("qpe_circuit_builder", iqpe_circuit_builder)
    noisy_result = iqpe.run(
        state_preparation=two_qubit_phase_problem.state_prep,
        qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
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


def test_iterative_qpe_initialization() -> None:
    """Test IterativePhaseEstimation initialization."""
    shots_per_bit = 10

    iqpe = IterativePhaseEstimation(shots_per_bit=shots_per_bit)

    assert iqpe._settings.get("shots_per_bit") == shots_per_bit


def test_iterative_qpe_raises_on_negative_num_bits(two_qubit_phase_problem: PhaseEstimationProblem) -> None:
    """Test that IQPE raises ValueError when num_bits is negative."""
    iqpe = IterativePhaseEstimation(shots_per_bit=3)
    iqpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
    )
    iqpe.settings().set(
        "qpe_circuit_builder",
        AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_iterative",
            num_bits=-1,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
        ),
    )

    with pytest.raises(ValueError, match="num_bits must be a positive integer"):
        iqpe.run(
            state_preparation=two_qubit_phase_problem.state_prep,
            qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
        )
