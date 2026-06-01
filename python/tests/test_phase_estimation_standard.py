"""Tests for Q# standard phase estimation circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from dataclasses import dataclass

import numpy as np
import pytest

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import HamiltonianUnitaryBuilder
from qdk_chemistry.algorithms.phase_estimation.standard_phase_estimation import StandardPhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit, QpeResult, QubitHamiltonian
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.phase import energy_from_phase
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
    qpe_phase_fraction_tolerance,
)

_SEED = 42


@dataclass(frozen=True)
class StandardQPEProblem:
    """Bundle describing a benchmark for standard phase estimation via Q#."""

    label: str
    hamiltonian: QubitHamiltonian
    state_prep: Circuit
    evolution_time: float
    num_bits: int
    shots: int
    expected_bitstring: str
    expected_phase: float
    expected_energy: float


@pytest.fixture
def two_qubit_phase_problem() -> StandardQPEProblem:
    """Return the canonical two-qubit phase estimation setup."""
    hamiltonian = QubitHamiltonian(pauli_strings=["XX", "ZZ"], coefficients=[0.25, 0.5])
    state_vector = [0.6, 0.0, 0.0, 0.8]
    state_prep_params = {"rowMap": [1, 0], "stateVector": state_vector, "expansionOps": [], "numQubits": 2}
    factories = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=state_prep_params
    )
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(state_prep_params)

    return StandardQPEProblem(
        label="two_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qsharp_factory=factories, qsharp_op=qsharp_op),
        evolution_time=float(np.pi / 2.0),
        num_bits=4,
        shots=3,
        expected_bitstring="1101",
        expected_phase=0.1875,
        expected_energy=0.75,
    )


@pytest.fixture
def four_qubit_phase_problem() -> StandardQPEProblem:
    """Return the documented four-qubit benchmark."""
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

    return StandardQPEProblem(
        label="four_qubit_reference",
        hamiltonian=hamiltonian,
        state_prep=Circuit(qsharp_factory=factories, qsharp_op=qsharp_op),
        evolution_time=float(np.pi / 8.0),
        num_bits=6,
        shots=3,
        expected_bitstring="010011",
        expected_phase=45 / 64,
        expected_energy=-4.75,
    )


def _extract_standard_results(problem: StandardQPEProblem) -> QpeResult:
    """Run Q# standard phase estimation and return the result.

    Args:
        problem: The standard phase estimation benchmark problem.

    Returns:
        QPE result including dominant bitstring, phase fraction, and energy.

    """
    qpe = StandardPhaseEstimation(num_bits=problem.num_bits, shots=problem.shots)

    qpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=_SEED),
    )
    qpe.settings().set(
        "circuit_mapper",
        AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
    )
    qpe.settings().set(
        "unitary_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=problem.evolution_time),
    )

    return qpe.run(
        state_preparation=problem.state_prep,
        qubit_hamiltonian=problem.hamiltonian,
    )


def test_standard_qpe_extracts_phase_and_energy(two_qubit_phase_problem: StandardQPEProblem) -> None:
    """Validate Q# standard QPE on the two-qubit benchmark."""
    results = _extract_standard_results(two_qubit_phase_problem)
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


def test_standard_qpe_four_qubit_problem(four_qubit_phase_problem: StandardQPEProblem) -> None:
    """Validate Q# standard QPE on the documented four-qubit system."""
    results = _extract_standard_results(four_qubit_phase_problem)
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


def test_raises_not_implemented_for_non_time_evolution_builder(
    two_qubit_phase_problem: StandardQPEProblem,
) -> None:
    """Standard QPE raises NotImplementedError when unitary_builder is not a TimeEvolutionBuilder."""

    class _MockBuilder(HamiltonianUnitaryBuilder):
        """A non-TimeEvolutionBuilder for testing the unsupported path."""

        def _run_impl(self, qubit_hamiltonian: QubitHamiltonian):  # noqa: ARG002
            return None

        def name(self):
            return "mock"

        def type_name(self):
            return "mock_unitary_builder"

    qpe = StandardPhaseEstimation(num_bits=two_qubit_phase_problem.num_bits, shots=two_qubit_phase_problem.shots)
    qpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=_SEED),
    )
    qpe.settings().set(
        "circuit_mapper",
        AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
    )
    qpe.settings().set(
        "unitary_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_phase_problem.evolution_time),
    )

    # Override cached_property with a non-TimeEvolutionBuilder instance
    qpe.__dict__["unitary_builder"] = _MockBuilder()

    with pytest.raises(NotImplementedError, match="only supports post-processing from time evolution"):
        qpe.run(
            state_preparation=two_qubit_phase_problem.state_prep,
            qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
        )


def test_standard_qpe_circuit_has_correct_qubit_count(two_qubit_phase_problem: StandardQPEProblem) -> None:
    """Validate that the QPE circuit allocates the expected number of qubits (ancilla + system)."""
    num_bits = two_qubit_phase_problem.num_bits
    num_system_qubits = two_qubit_phase_problem.hamiltonian.num_qubits

    qpe = StandardPhaseEstimation(num_bits=num_bits, shots=two_qubit_phase_problem.shots)
    qpe.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=_SEED),
    )
    qpe.settings().set(
        "circuit_mapper",
        AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
    )
    qpe.settings().set(
        "unitary_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_phase_problem.evolution_time),
    )

    circuit = qpe.create_circuit(
        state_preparation=two_qubit_phase_problem.state_prep,
        qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
    )

    qsc = circuit.get_qsharp_circuit()
    circuit_data = json.loads(qsc.json())
    total_qubits = len(circuit_data["qubits"])

    expected_total_qubits = num_bits + num_system_qubits
    assert total_qubits == expected_total_qubits, (
        f"Expected {expected_total_qubits} qubits ({num_bits} ancilla + {num_system_qubits} system), got {total_qubits}"
    )
