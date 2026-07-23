"""Tests for standard phase estimation algorithms."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qdk_chemistry.algorithms.phase_estimation.standard_phase_estimation import StandardPhaseEstimation
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QubitOperator,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT
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
    shots: int


@pytest.fixture
def two_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return the two-qubit phase estimation scenario used in documentation."""
    hamiltonian = QubitOperator(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    state_prep_params = {
        "rowMap": [1, 0],
        "stateVector": [0.6, 0.0, 0.0, 0.8],
        "expansionOps": [],
        "numQubits": 2,
    }
    factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
    state_prep = Circuit(qsharp_factory=factories, qsharp_op=qsharp_op)

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
        shots=3,
    )


@pytest.fixture
def four_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return the four-qubit benchmark used in documentation."""
    hamiltonian = QubitOperator(pauli_strings=["XXXX", "ZZZZ"], coefficients=[0.25, 4.5])
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
        shots=3,
    )


@pytest.fixture
def number_conserving_phase_problem() -> PhaseEstimationProblem:
    """Return a number-conserving Hamiltonian.

    The Hamiltonian ``0.5 (XX + YY)`` is the Jordan-Wigner image of a two-site
    fermionic hopping term ``a0^dag a1 + a1^dag a0``. Its vacuum ``|00>`` is an
    eigenstate with eigenvalue ``0``, so the CSWAP-sandwich controlled circuit
    mapper introduces no vacuum-reference phase and agrees exactly with the direct
    controlled-unitary mapper. The prepared state ``(|01> + |10>) / sqrt(2)`` is the
    single-excitation eigenstate with energy ``+1``.
    """
    hamiltonian = QubitOperator(pauli_strings=["XX", "YY"], coefficients=[0.5, 0.5])
    inv_sqrt2 = float(1.0 / np.sqrt(2.0))
    state_prep_params = {
        "rowMap": [1, 0],
        "stateVector": [0.0, inv_sqrt2, inv_sqrt2, 0.0],
        "expansionOps": [],
        "numQubits": 2,
    }
    factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)
    state_prep = Circuit(qsharp_factory=factories, qsharp_op=qsharp_op)

    return PhaseEstimationProblem(
        label="chemistry_hopping",
        hamiltonian=hamiltonian,
        state_prep=state_prep,
        evolution_time=float(np.pi / 2.0),
        num_bits=4,
        expected_bits=[1, 1, 0, 0],
        expected_phase=0.25,
        expected_energy=1.0,
        expected_bitstring="1100",
        shots=3,
    )


def _make_circuit_builder_ref(
    builder_name: str,
    problem: PhaseEstimationProblem,
    controlled_circuit_mapper_name: str = "pauli_sequence",
) -> AlgorithmRef:
    """Create an AlgorithmRef for the given circuit builder name and problem.

    Args:
        builder_name: The registered name of the circuit builder (e.g. "qdk_standard", "qiskit_standard").
        problem: The phase estimation problem providing num_bits and evolution_time.
        controlled_circuit_mapper_name: Name of the controlled circuit mapper to use.

    Returns:
        An AlgorithmRef configured for the given builder.

    """
    kwargs: dict = {
        "num_bits": problem.num_bits,
        "controlled_circuit_mapper": AlgorithmRef("controlled_circuit_mapper", controlled_circuit_mapper_name),
        "unitary_builder": AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=problem.evolution_time),
    }
    if builder_name == "qiskit_standard":
        kwargs["qft_do_swaps"] = True
    return AlgorithmRef("qpe_circuit_builder", builder_name, **kwargs)


def _run_standard(
    problem: PhaseEstimationProblem,
    builder_name: str = "qdk_standard",
    controlled_circuit_mapper_name: str = "pauli_sequence",
) -> QpeResult:
    """Execute standard QPE and return structured results.

    Args:
        problem: Benchmark description supplying Hamiltonian, state prep, and expectations.
        builder_name: The circuit builder to use ("qdk_standard" or "qiskit_standard").
        controlled_circuit_mapper_name: Name of the controlled circuit mapper to use.

    Returns:
        :class:`QpeResult` instance summarizing the standard run.

    """
    qpe_circuit_builder = _make_circuit_builder_ref(builder_name, problem, controlled_circuit_mapper_name)
    qpe = StandardPhaseEstimation(shots=problem.shots)
    qpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=_SEED),
    )
    qpe.settings().set(
        "qpe_circuit_builder",
        qpe_circuit_builder,
    )

    return qpe.run(
        qubit_hamiltonian=problem.hamiltonian,
        state_preparation=problem.state_prep,
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
    energies = []
    for candidate in phase_fraction_candidates:
        angle = (candidate % 1.0) * (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        energies.append(angle / evolution_time)

    # Select candidate closest to expected energy
    index = int(np.argmin([abs(energy - expected_energy) for energy in energies]))
    return phase_fraction_candidates[index], energies[index]


# Parametrize over both qdk_standard and qiskit_standard builders
_builder_params = [
    pytest.param("qdk_standard", id="qdk_standard"),
    pytest.param(
        "qiskit_standard",
        id="qiskit_standard",
        marks=pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available"),
    ),
]

# Parametrize over controlled circuit mapper variants
_controlled_mapper_params = [
    pytest.param("pauli_sequence", id="pauli_sequence"),
    pytest.param("cswap_pauli_sequence", id="cswap_pauli_sequence"),
]


@pytest.mark.parametrize("builder_name", _builder_params)
def test_standard_phase_estimation_extracts_phase_and_energy(
    two_qubit_phase_problem: PhaseEstimationProblem,
    builder_name: str,
) -> None:
    """Verify standard phase estimation recovers expected phase and energy."""
    result = _run_standard(two_qubit_phase_problem, builder_name)
    resolved_phase, resolved_energy = _resolve_phase_ambiguity(
        result.phase_fraction, two_qubit_phase_problem.evolution_time, two_qubit_phase_problem.expected_energy
    )

    assert result.bitstring_msb_first == two_qubit_phase_problem.expected_bitstring
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
@pytest.mark.parametrize("controlled_circuit_mapper_name", _controlled_mapper_params)
def test_standard_phase_estimation_controlled_mapper_variants(
    number_conserving_phase_problem: PhaseEstimationProblem,
    builder_name: str,
    controlled_circuit_mapper_name: str,
) -> None:
    """Compare controlled circuit mappers on a number-conserving chemistry Hamiltonian.

    The vacuum ``|00>`` is an eigenstate of ``0.5 (XX + YY)``, so the CSWAP-sandwich
    mapper carries no vacuum-reference phase and must recover the same phase and
    energy as the direct controlled-unitary mapper.
    """
    result = _run_standard(
        number_conserving_phase_problem,
        builder_name=builder_name,
        controlled_circuit_mapper_name=controlled_circuit_mapper_name,
    )
    resolved_phase, resolved_energy = _resolve_phase_ambiguity(
        result.phase_fraction,
        number_conserving_phase_problem.evolution_time,
        number_conserving_phase_problem.expected_energy,
    )

    assert result.bitstring_msb_first == number_conserving_phase_problem.expected_bitstring
    assert np.isclose(
        resolved_phase,
        number_conserving_phase_problem.expected_phase,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_phase_fraction_tolerance,
    )
    assert np.isclose(
        resolved_energy,
        number_conserving_phase_problem.expected_energy,
        rtol=float_comparison_relative_tolerance,
        atol=qpe_energy_tolerance,
    )


@pytest.mark.parametrize("builder_name", _builder_params)
def test_standard_phase_estimation_four_qubit(
    four_qubit_phase_problem: PhaseEstimationProblem,
    builder_name: str,
) -> None:
    """Validate standard phase estimation on the four-qubit benchmark."""
    result = _run_standard(four_qubit_phase_problem, builder_name)
    resolved_phase, resolved_energy = _resolve_phase_ambiguity(
        result.phase_fraction, four_qubit_phase_problem.evolution_time, four_qubit_phase_problem.expected_energy
    )

    assert result.bitstring_msb_first == four_qubit_phase_problem.expected_bitstring
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


@pytest.mark.parametrize(
    "builder_name",
    [
        pytest.param("qdk_standard", id="qdk_standard"),
        pytest.param(
            "qiskit_standard",
            id="qiskit_standard",
            marks=pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available"),
        ),
    ],
)
def test_standard_qpe_initialization(builder_name: str) -> None:
    """Test StandardPhaseEstimation initialization with circuit builder."""
    shots = 100
    num_bits = 10
    qpe = StandardPhaseEstimation(shots=shots)

    # Verify basic settings
    assert qpe._settings.get("shots") == shots

    # Configure with circuit builder
    kwargs: dict = {"num_bits": num_bits}
    if builder_name == "qiskit_standard":
        kwargs["qft_do_swaps"] = True
    qpe.settings().set(
        "qpe_circuit_builder",
        AlgorithmRef("qpe_circuit_builder", builder_name, **kwargs),
    )

    # Verify circuit builder is set
    circuit_builder_ref = qpe.settings().get("qpe_circuit_builder")
    assert circuit_builder_ref is not None
    assert isinstance(circuit_builder_ref, AlgorithmRef)


def test_standard_qpe_rejects_iterative_circuit_builder(
    two_qubit_phase_problem: PhaseEstimationProblem,
) -> None:
    """Verify standard phase estimation raises TypeError when configured with iterative circuit builder."""
    qpe_circuit_builder = AlgorithmRef(
        "qpe_circuit_builder",
        "qdk_iterative",  # Iterative circuit builder - should fail with Standard QPE
        num_bits=two_qubit_phase_problem.num_bits,
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        unitary_builder=AlgorithmRef(
            "hamiltonian_unitary_builder", "trotter", time=two_qubit_phase_problem.evolution_time
        ),
    )
    qpe = StandardPhaseEstimation(shots=two_qubit_phase_problem.shots)
    qpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=_SEED),
    )
    qpe.settings().set("qpe_circuit_builder", qpe_circuit_builder)

    # Should raise TypeError when trying to run with iterative circuit builder
    with pytest.raises(
        TypeError,
        match="Expected qpe_circuit_builder to be an instance of StandardQpeCircuitBuilder",
    ):
        qpe.run(
            qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
            state_preparation=two_qubit_phase_problem.state_prep,
        )
