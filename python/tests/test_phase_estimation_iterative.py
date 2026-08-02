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
    QubitOperator,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
from qdk_chemistry.utils.pauli_matrix import pauli_to_dense_matrix
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
    hamiltonian: QubitOperator
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
    """Return a canonical two-qubit phase estimation benchmark.

    Uses ``H = 0.25*XX + 0.5*ZZ``.
    The prepared state is the exact eigenstate ``(|00> + |11>)/sqrt(2)``
    with eigenvalue ``E = +0.75``.

    Theory (``U = e^{-iHt}`` with ``t = pi/2``, textbook convention
    ``phi = (-E t / 2pi) mod 1``):
        phi = (-0.75 * (pi/2) / 2pi) mod 1 = 13/16 = 0.8125
        MSB-first bitstring = "1101"
        E = -angle(phi)/t = +0.75
    """
    hamiltonian = QubitOperator(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    state_prep_params = {
        "rowMap": [1, 0],
        "stateVector": [1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)],
        "expansionOps": [],
        "numQubits": 2,
    }
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
        expected_bits=[1, 1, 0, 1],
        expected_phase=13 / 16,
        expected_energy=0.75,
        expected_bitstring="1101",
        shots_iterative=3,
    )


@pytest.fixture
def four_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return a canonical four-qubit phase estimation benchmark.

    Uses ``H = 0.25*XXXX + 4.5*ZZZZ``.
    The prepared state is the exact eigenstate
    ``(|1000> - |0111>)/sqrt(2)`` with eigenvalue ``E = -4.75``.

    Theory (``U = e^{-iHt}`` with ``t = pi/8``, textbook convention
    ``phi = (-E t / 2pi) mod 1``):
        phi = (4.75 * (pi/8) / 2pi) mod 1 = 19/64 = 0.296875
        MSB-first bitstring = "010011"
        E = -angle(phi)/t = -4.75  (angle folded into (-pi, pi])
    """
    hamiltonian = QubitOperator(pauli_strings=["XXXX", "ZZZZ"], coefficients=[0.25, 4.5])
    state_vector = np.zeros(2**4, dtype=float)
    state_vector[int("1000", 2)] = 1.0 / np.sqrt(2.0)
    state_vector[int("0111", 2)] = -1.0 / np.sqrt(2.0)
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
        expected_bits=[0, 1, 0, 0, 1, 1],
        expected_phase=19 / 64,
        expected_energy=-4.75,
        expected_bitstring="010011",
        shots_iterative=3,
    )


def _make_iterative_circuit_builder_ref(
    builder_name: str,
    num_bits: int,
    evolution_time: float,
    unitary_builder_name: str = "trotter",
) -> AlgorithmRef:
    """Return an iterative circuit builder AlgorithmRef for the given builder name."""
    return AlgorithmRef(
        "qpe_circuit_builder",
        builder_name,
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", unitary_builder_name, time=evolution_time),
    )


def _run_iterative(
    problem: PhaseEstimationProblem,
    builder_name: str = "qdk_iterative",
    unitary_builder_name: str = "trotter",
) -> QpeResult:
    """Execute iterative phase estimation and return structured results.

    Args:
        problem: Benchmark description supplying Hamiltonian, state prep, and expectations.
        builder_name: The circuit builder to use ("qdk_iterative" or "qiskit_iterative").
        unitary_builder_name: Name of the unitary builder to use.

    Returns:
        :class:`QpeResult` instance summarizing the iterative run.

    """
    state_prep_circuit = problem.state_prep
    circuit_builder = _make_iterative_circuit_builder_ref(
        builder_name, problem.num_bits, problem.evolution_time, unitary_builder_name
    )
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

    hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients)
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


# Parametrize over both qdk_iterative and qiskit_iterative builders
_builder_params = [
    pytest.param("qdk_iterative", id="qdk_iterative"),
    pytest.param(
        "qiskit_iterative",
        id="qiskit_iterative",
        marks=pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available"),
    ),
]

# Parametrize over time evolution unitary builders
_unitary_builder_params = [
    pytest.param("trotter", id="trotter"),
    pytest.param("zassenhaus", id="zassenhaus"),
]


@pytest.mark.parametrize("builder_name", _builder_params)
@pytest.mark.parametrize("unitary_builder_name", _unitary_builder_params)
def test_iterative_phase_estimation_extracts_phase_and_energy(
    two_qubit_phase_problem: PhaseEstimationProblem,
    builder_name: str,
    unitary_builder_name: str,
) -> None:
    """Verify the iterative algorithm recovers the expected phase and energy."""
    result = _run_iterative(two_qubit_phase_problem, builder_name, unitary_builder_name)

    assert list(result.bits_msb_first or []) == two_qubit_phase_problem.expected_bits
    assert np.isclose(
        result.phase_fraction,
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


@pytest.mark.parametrize("builder_name", _builder_params)
@pytest.mark.parametrize("unitary_builder_name", _unitary_builder_params)
def test_iterative_phase_estimation_four_qubit_phase_and_energy(
    four_qubit_phase_problem: PhaseEstimationProblem,
    builder_name: str,
    unitary_builder_name: str,
) -> None:
    """Validate phase and energy estimates on the documented four-qubit case."""
    result = _run_iterative(four_qubit_phase_problem, builder_name, unitary_builder_name)

    assert list(result.bits_msb_first or []) == four_qubit_phase_problem.expected_bits
    assert np.isclose(
        result.phase_fraction,
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


def test_iterative_phase_estimation_non_commuting_xi_plus_zz() -> None:
    """Validate IQPE for H = 0.519 XI + ZZ with Hartree-Fock-like trial state.

    H = 0.519 X_0 + Z_0 Z_1 block-diagonalizes on the pairs {|00>, |10>} and
    {|01>, |11>}. Each block has the form [[+/-1, 0.519], [0.519, -/+1]] with
    eigenvalues +/- sqrt(1 + 0.519^2) = +/- 1.126659. The trial state
    0.97|00> + sqrt(1 - 0.97^2)|10> lies entirely in the {|00>, |10>} block and
    overlaps the E = +1.126659 eigenstate with probability 0.99996.

    Theory (U = e^{-iHt}, t = pi/4, textbook convention phi = (-E t / 2pi) mod 1):
        phi = (-1.126659 * (pi/4) / 2pi) mod 1 = -1.126659/8 mod 1 = 0.859168
        6-bit rounding: round(0.859168 * 64) = 55 -> phi = 55/64 = 0.859375
        MSB-first bits = 110111 = [1, 1, 0, 1, 1, 1]
        E = -angle(phi)/t with phi folded to (-1/2, 1/2] (55/64 -> -9/64):
            E = -(2pi * (-9/64)) / (pi/4) = 9/8 = 1.125
        (nearest point on the 1/8-spaced energy grid to the exact 1.126659)
    """
    pauli_strings = ["XI", "ZZ"]
    coefficients = [0.519, 1.0]
    state_vector = np.array([0.97, 0.0, np.sqrt(1 - 0.97**2), 0.0], dtype=float)
    evolution_time = np.pi / 4
    num_bits = 6

    # Canonical expectation, computed inline via exact diagonalization of this H.
    hamiltonian_matrix = pauli_to_dense_matrix(pauli_strings, np.asarray(coefficients)).real
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    trial = state_vector / np.linalg.norm(state_vector)
    dominant = int(np.argmax((eigenvectors.T @ trial) ** 2))
    dominant_energy = float(eigenvalues[dominant])  # +sqrt(1 + 0.519**2) = 1.126659

    # phi = (-E t / 2pi) mod 1, rounded to num_bits of precision.
    phase_true = (-dominant_energy * evolution_time / (2 * np.pi)) % 1.0
    index = round(phase_true * 2**num_bits) % 2**num_bits
    expected_phase = index / 2**num_bits  # 55/64
    expected_bits = [(index >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]  # [1, 1, 0, 1, 1, 1]

    # Recover energy from the rounded phase, folding the angle into (-pi, pi].
    angle = expected_phase * 2 * np.pi
    if angle > np.pi:
        angle -= 2 * np.pi
    expected_energy = -angle / evolution_time  # 1.125

    result = _run_iterative_with_parameters(
        pauli_strings,
        coefficients,
        state_vector,
        evolution_time=evolution_time,
        num_bits=num_bits,
        shots_per_bit=3,
        seed=_SEED,
    )

    assert list(result.bits_msb_first or []) == expected_bits
    assert np.isclose(
        result.phase_fraction,
        expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        result.raw_energy, expected_energy, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance
    )


def test_iterative_phase_estimation_second_non_commuting_example() -> None:
    """Validate IQPE for H = -0.0289(X1+X2) + 0.0541(Z1+Z2) + 0.0150 XX + 0.0590 ZZ.

    H is symmetric under swapping the two qubits, so it block-diagonalizes into a
    1-D antisymmetric sector ((|01> - |10>)/sqrt(2), eigenvalue -0.0740) and a 3-D
    symmetric sector. Exact diagonalization gives eigenvalues
    {-0.088779, -0.074000, -0.014303, 0.177082}. The trial state
    (0, 0.47, 0.47, 0.75) (normalized) lies in the symmetric sector and overlaps
    the ground state E0 = -0.088779 with probability 0.99089.

    Theory (U = e^{-iHt}, t = pi/4, textbook convention phi = (-E t / 2pi) mod 1):
        phi = (0.088779 / 8) mod 1 = 0.011097
        11-bit rounding: round(0.011097 * 2048) = 23 -> phi = 23/2048 = 0.011230
        MSB-first bits = 00000010111 = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
        E = -angle(phi)/t = -(2pi * 23/2048) / (pi/4) = -184/2048 = -0.08984375
    """
    pauli_strings = ["XI", "IX", "ZI", "IZ", "XX", "ZZ"]
    coefficients = [-0.0289, -0.0289, 0.0541, 0.0541, 0.0150, 0.059]
    state_vector = np.array([0.0, 0.47, 0.47, 0.75], dtype=float)
    state_vector /= np.linalg.norm(state_vector)
    evolution_time = np.pi / 4
    num_bits = 11

    hamiltonian_matrix = pauli_to_dense_matrix(pauli_strings, np.asarray(coefficients)).real
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
    trial = state_vector / np.linalg.norm(state_vector)
    dominant = int(np.argmax((eigenvectors.T @ trial) ** 2))
    dominant_energy = float(eigenvalues[dominant])  # ground state = -0.088779

    # phi = (-E t / 2pi) mod 1, rounded to num_bits of precision.
    phase_true = (-dominant_energy * evolution_time / (2 * np.pi)) % 1.0
    index = round(phase_true * 2**num_bits) % 2**num_bits
    expected_phase = index / 2**num_bits  # 23/2048
    expected_bits = [(index >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]  # [0,0,0,0,0,0,1,0,1,1,1]

    # Recover energy from the rounded phase, folding the angle into (-pi, pi].
    angle = expected_phase * 2 * np.pi
    if angle > np.pi:
        angle -= 2 * np.pi
    expected_energy = -angle / evolution_time  # -0.08984375

    result = _run_iterative_with_parameters(
        pauli_strings,
        coefficients,
        state_vector,
        evolution_time=evolution_time,
        num_bits=num_bits,
        shots_per_bit=3,
        seed=_SEED,
    )

    assert list(result.bits_msb_first or []) == expected_bits
    assert np.isclose(
        result.phase_fraction,
        expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        result.raw_energy, expected_energy, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance
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
