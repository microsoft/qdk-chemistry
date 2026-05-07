"""Tests for qiskit standard phase estimation circuits."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import dataclass

import numpy as np
import pytest

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import HamiltonianUnitaryBuilder
from qdk_chemistry.data import AlgorithmRef, Circuit, QpeResult, QubitHamiltonian
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
    qpe_phase_fraction_tolerance,
)

if QDK_CHEMISTRY_HAS_QISKIT:
    from qiskit import QuantumCircuit, qasm3
    from qiskit.circuit.library import StatePreparation as QiskitStatePreparation

    from qdk_chemistry.plugins.qiskit.standard_phase_estimation import (
        QiskitStandardPhaseEstimation,
        QiskitStandardQpeCircuitBuilder,
    )
    from qdk_chemistry.utils.phase import energy_from_phase

else:
    # Define placeholders for type checking when Qiskit is not available
    QuantumCircuit = object
    qasm3 = object
    QiskitStatePreparation = object
    QiskitStandardPhaseEstimation = object
    QiskitStandardQpeCircuitBuilder = object


pytestmark = pytest.mark.skipif(not QDK_CHEMISTRY_HAS_QISKIT, reason="Qiskit not available")

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
    qpe = QiskitStandardPhaseEstimation(num_bits=problem.num_bits)

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


def test_raises_not_implemented_for_non_time_evolution_builder(
    two_qubit_phase_problem: TraditionalProblem,
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

    qpe = QiskitStandardPhaseEstimation(num_bits=two_qubit_phase_problem.num_bits)
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

    # Patch _create_nested to return the mock builder for "unitary_builder"
    mock_builder = _MockBuilder()
    original_create_nested = qpe._create_nested

    def _patched_create_nested(key, **kwargs):
        if key == "unitary_builder":
            return mock_builder
        return original_create_nested(key, **kwargs)

    qpe._create_nested = _patched_create_nested

    with pytest.raises(NotImplementedError, match="only supports post-processing from time evolution"):
        qpe.run(
            state_preparation=two_qubit_phase_problem.state_prep,
            qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
        )


def test_builder_run_returns_circuits(two_qubit_phase_problem: TraditionalProblem) -> None:
    """Validate that QiskitStandardQpeCircuitBuilder.run produces a single QPE circuit."""
    builder = QiskitStandardQpeCircuitBuilder(num_bits=two_qubit_phase_problem.num_bits)
    builder.settings().set(
        "circuit_mapper",
        AlgorithmRef("controlled_circuit_mapper", "pauli_sequence"),
    )
    builder.settings().set(
        "unitary_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=two_qubit_phase_problem.evolution_time),
    )

    circuits = builder.run(
        state_preparation=two_qubit_phase_problem.state_prep,
        qubit_hamiltonian=two_qubit_phase_problem.hamiltonian,
    )

    assert isinstance(circuits, list)
    assert len(circuits) == 1
    circuit = circuits[0]
    assert isinstance(circuit, Circuit)
    result = circuit.estimate()
    assert result is not None
    assert hasattr(result, "logical_counts")
