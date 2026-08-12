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
    expected_phase: float
    expected_energy: float
    expected_bitstring: str
    shots: int


@pytest.fixture
def two_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return a canonical two-qubit phase estimation benchmark.

    Uses ``H = 0.25*XX + 0.5*ZZ`` (commuting terms, so the Trotter unitary is
    exact). The prepared state is the exact eigenstate ``(|00> + |11>)/sqrt(2)``
    with eigenvalue ``E = +0.75``.

    Theory (``U = e^{-iHt}`` with ``t = pi/2``):
        phi = (-E t / 2pi) mod 1 = (-0.75 * (pi/2) / 2pi) mod 1 = 13/16 = 0.8125
        4-bit MSB-first bitstring = "1101"
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
    state_prep = Circuit(qsharp_factory=factories, qsharp_op=qsharp_op)

    return PhaseEstimationProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=state_prep,
        evolution_time=float(np.pi / 2.0),
        num_bits=4,
        expected_phase=13 / 16,
        expected_energy=0.75,
        expected_bitstring="1101",
        shots=3,
    )


@pytest.fixture
def four_qubit_phase_problem() -> PhaseEstimationProblem:
    """Return a canonical four-qubit phase estimation benchmark.

    Uses ``H = 0.25*XXXX + 4.5*ZZZZ`` (commuting terms, so the Trotter unitary is
    exact). The prepared state is the exact eigenstate
    ``(|1000> - |0111>)/sqrt(2)`` with eigenvalue ``E = -4.75``.

    Theory (``U = e^{-iHt}`` with ``t = pi/8``):
        phi = (-E t / 2pi) mod 1 = (4.75 * (pi/8) / 2pi) mod 1 = 19/64 = 0.296875
        6-bit MSB-first bitstring = "010011"
        E = -angle(phi)/t = -4.75
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
        expected_phase=19 / 64,
        expected_energy=-4.75,
        expected_bitstring="010011",
        shots=3,
    )


def _make_circuit_builder_ref(builder_name: str, problem: PhaseEstimationProblem) -> AlgorithmRef:
    """Create an AlgorithmRef for the given circuit builder name and problem.

    Args:
        builder_name: The registered name of the circuit builder (e.g. "qdk_standard", "qiskit_standard").
        problem: The phase estimation problem providing num_bits and evolution_time.

    Returns:
        An AlgorithmRef configured for the given builder.

    """
    kwargs: dict = {
        "num_bits": problem.num_bits,
        "controlled_circuit_mapper": AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
        "unitary_builder": AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=problem.evolution_time),
    }
    if builder_name == "qiskit_standard":
        kwargs["qft_do_swaps"] = True
    return AlgorithmRef("qpe_circuit_builder", builder_name, **kwargs)


def _run_standard(problem: PhaseEstimationProblem, builder_name: str = "qdk_standard") -> QpeResult:
    """Execute standard QPE and return structured results.

    Args:
        problem: Benchmark description supplying Hamiltonian, state prep, and expectations.
        builder_name: The circuit builder to use ("qdk_standard" or "qiskit_standard").

    Returns:
        :class:`QpeResult` instance summarizing the standard run.

    """
    qpe_circuit_builder = _make_circuit_builder_ref(builder_name, problem)
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


# Parametrize over both qdk_standard and qiskit_standard builders
_builder_params = [
    pytest.param("qdk_standard", id="qdk_standard"),
    pytest.param(
        "qiskit_standard",
        id="qiskit_standard",
        marks=pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available"),
    ),
]


@pytest.mark.parametrize("builder_name", _builder_params)
def test_standard_phase_estimation_extracts_phase_and_energy(
    two_qubit_phase_problem: PhaseEstimationProblem,
    builder_name: str,
) -> None:
    """Verify standard phase estimation recovers expected phase and energy."""
    result = _run_standard(two_qubit_phase_problem, builder_name)

    assert result.bitstring_msb_first == two_qubit_phase_problem.expected_bitstring
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
def test_standard_phase_estimation_four_qubit(
    four_qubit_phase_problem: PhaseEstimationProblem,
    builder_name: str,
) -> None:
    """Validate standard phase estimation on the four-qubit benchmark."""
    result = _run_standard(four_qubit_phase_problem, builder_name)

    assert result.bitstring_msb_first == four_qubit_phase_problem.expected_bitstring
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
