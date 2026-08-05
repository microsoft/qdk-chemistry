"""Tests for amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.algorithms.amplitude_amplification import AmplitudeAmplification, phase_marking_oracle
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context


@pytest.fixture(scope="module")
def qsharp_context():
    """Load the chemistry Q# utilities exactly once."""
    if importlib.util.find_spec("qdk.qsharp") is None:
        pytest.skip("qdk.qsharp is not installed")
    return get_qsharp_context()


def _amplified_expression(theta: float, rounds: int) -> str:
    """Amplify one qubit prepared as cos(theta)|0> + sin(theta)|1>, marking |1>."""
    amplification = "QDKChemistry.Utils.AmplitudeAmplification"
    return (
        f"{amplification}.MakeAmplifiedCircuit("
        f"Std.StatePreparation.PreparePureStateD([{math.cos(theta)}, {math.sin(theta)}], _), "
        f"{amplification}.MarkTargetStateOp(1, [], 1, 2), {rounds}, 1, [0])"
    )


def _acceptance_frequency(qsharp_context, expression: str) -> float:
    """Return the fraction of shots that land in the good subspace."""
    shots = 4000
    outcomes = qsharp_context.run(expression, shots=shots)
    return sum(1 for outcome in outcomes if str(outcome[0]) == "One") / shots


@pytest.mark.parametrize(("overlap", "rounds"), [(0.05, 0), (0.05, 2), (0.05, 3), (0.1, 1), (0.1, 2), (0.25, 1)])
def test_plain_amplification_matches_the_closed_form(qsharp_context, overlap: float, rounds: int):
    theta = AmplitudeAmplification._rotation_angle(overlap)
    observed = _acceptance_frequency(qsharp_context, _amplified_expression(theta, rounds))
    assert observed == pytest.approx(AmplitudeAmplification.success_probability(overlap, rounds), abs=0.04)


def test_plain_amplification_overshoots_when_the_overlap_is_underestimated(qsharp_context):
    """Three rounds are optimal for a = 0.02 but wrap past the maximum for a = 0.25."""
    overlap = 0.25
    rounds = 3
    theta = AmplitudeAmplification._rotation_angle(overlap)
    observed = _acceptance_frequency(qsharp_context, _amplified_expression(theta, rounds))
    assert observed == pytest.approx(AmplitudeAmplification.success_probability(overlap, rounds), abs=0.04)
    assert observed < 0.5

    # A single round is what a = 0.25 actually wants.
    better = _acceptance_frequency(qsharp_context, _amplified_expression(theta, 1))
    assert better > observed


def _diagonal_hamiltonian() -> QubitOperator:
    """Return H = (pi/4) ZI + (pi/4) IZ, whose spectrum is {pi/2, 0, 0, -pi/2}."""
    coefficient = math.pi / 4.0
    return QubitOperator(pauli_strings=["ZI", "IZ"], coefficients=np.array([coefficient, coefficient]))


def _guiding_state(amplitude: float, index: int, num_qubits: int = 2) -> Circuit:
    """Prepare a state with the given amplitude on one computational basis state."""
    vector = [0.0] * (1 << num_qubits)
    vector[index] = amplitude
    remainder = math.sqrt(max(0.0, 1.0 - amplitude**2))
    vector[(index + 1) % (1 << num_qubits)] = remainder

    parameters = {
        "rowMap": list(reversed(range(num_qubits))),
        "stateVector": vector,
        "expansionOps": [],
        "numQubits": num_qubits,
    }
    return Circuit(
        qsharp_factory=QsharpFactoryData(
            program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit,
            parameter=parameters,
        ),
        qsharp_op=QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(parameters),
    )


def _qpe_preparation(
    qubit_hamiltonian: QubitOperator,
    state_preparation: Circuit,
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
) -> tuple[Circuit, int, list[int]]:
    """Build a measurement-free QPE circuit plus its register layout."""
    unitary = unitary or AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True)
    builder = create("qpe_circuit_builder", "qdk_standard")
    builder.settings().update("num_bits", num_bits)
    builder.settings().update("controlled_circuit_mapper", AlgorithmRef("controlled_circuit_mapper", mapper))
    builder.settings().update("unitary_builder", unitary)
    builder.settings().update("measure_phase", False)
    preparation = builder.run(state_preparation=state_preparation, qubit_hamiltonian=qubit_hamiltonian)[0]

    unitary_algorithm = create(unitary.algorithm_type, unitary.algorithm_name, **unitary.settings)
    num_system_qubits = qubit_hamiltonian.num_qubits
    num_ancilla_qubits = unitary_algorithm.run(qubit_hamiltonian).get_num_qubits() - num_system_qubits
    num_qubits = num_bits + num_system_qubits + num_ancilla_qubits
    signal_ancilla_indices = list(range(num_system_qubits, num_system_qubits + num_ancilla_qubits))
    return preparation, num_qubits, signal_ancilla_indices


def _amplified_qpe_circuit(
    qubit_hamiltonian: QubitOperator,
    state_preparation: Circuit,
    accepted_range: tuple[int, int],
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
    **settings,
) -> Circuit:
    """Compose a coherent QPE preparation with amplitude amplification."""
    state_prep_oracle, num_qubits, signal_ancilla_indices = _qpe_preparation(
        qubit_hamiltonian,
        state_preparation,
        num_bits=num_bits,
        mapper=mapper,
        unitary=unitary,
    )
    good_state_oracle = phase_marking_oracle(num_bits, accepted_range, signal_ancilla_indices)
    # The executor reverses the Q# results, so emitting the ancillas reversed and
    # ahead of the phase indices makes the key read phase register MSB first.
    ancilla_indices = list(range(num_qubits - len(signal_ancilla_indices), num_qubits))
    measured_indices = list(reversed(ancilla_indices)) + list(range(num_bits))

    algorithm = create("amplitude_amplification")
    for key, value in settings.items():
        algorithm.settings().update(key, value)
    return algorithm.run(
        state_prep_oracle, good_state_oracle, num_qubits=num_qubits, measured_indices=measured_indices
    )


def _accepted_phase_counts(
    circuit: Circuit, num_bits: int, accepted_range: tuple[int, int], shots: int
) -> dict[str, int]:
    """Execute a circuit and count, per phase bitstring, the shots in the good subspace."""
    lower_bound, upper_bound = accepted_range
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    counts: dict[str, int] = {}
    for bitstring, count in executor.run(circuit, shots=shots).bitstring_counts.items():
        phase_bits, ancilla_bits = bitstring[:num_bits], bitstring[num_bits:]
        if any(bit != "0" for bit in ancilla_bits) or not lower_bound <= int(phase_bits, 2) < upper_bound:
            continue
        counts[phase_bits] = counts.get(phase_bits, 0) + count
    return counts


def _dominant_accepted_phase(circuit: Circuit, num_bits: int, accepted_range: tuple[int, int], shots: int = 400) -> str:
    """Execute a circuit and return the most common bitstring from the good subspace."""
    counts = _accepted_phase_counts(circuit, num_bits, accepted_range, shots)
    assert counts, f"No shot landed in the accepted window {accepted_range}."
    return max(counts, key=lambda phase: counts[phase])


def test_amplitude_amplification_is_registered():
    assert available("amplitude_amplification") == ["qdk"]
    default = create("amplitude_amplification")
    assert default.name() == "qdk"
    assert default.type_name() == "amplitude_amplification"
    assert isinstance(default, AmplitudeAmplification)


def test_rounds_setting_defaults_to_one():
    algorithm = create("amplitude_amplification")
    assert algorithm.settings().get("rounds") == 1


def test_amplified_qpe_circuit():
    """Check that amplitude amplification can be applied to a QPE circuit."""
    # The |11> eigenvector has energy -lambda, which the qubitization walk maps
    # to the phase bin 1/2 -- index 8 of 16, big-endian 0b1000.
    accepted = (8, 9)
    circuit = _amplified_qpe_circuit(
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        accepted,
        rounds=2,
    )
    assert _dominant_accepted_phase(circuit, 4, accepted) == "1000"


def test_amplified_qpe_circuit_with_trotter():
    """Check that amplitude amplification can be applied to a QPE circuit with the pauli-sequence mapper."""
    # The pauli-sequence mapper has no block-encoding ancillas, so the accepted
    # window is defined purely on the phase register.
    accepted = (4, 5)
    circuit = _amplified_qpe_circuit(
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        accepted,
        mapper="pauli_sequence",
        unitary=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
        rounds=1,
    )
    # e^{-iHt} with t = 1 maps the eigenvalue -pi/2 to the phase 1/4, bin 4 of 16.
    assert _dominant_accepted_phase(circuit, 4, accepted, shots=200) == "0100"


def test_amplified_qpe_acceptance_follows_the_round_count():
    """Each round rotates the QPE state by the same angle toward the accepted window."""
    accepted = (8, 9)
    shots = 2000
    observed = {
        rounds: sum(
            _accepted_phase_counts(
                _amplified_qpe_circuit(
                    _diagonal_hamiltonian(),
                    _guiding_state(0.3, 3),
                    accepted,
                    rounds=rounds,
                ),
                4,
                accepted,
                shots,
            ).values()
        )
        / shots
        for rounds in (0, 1, 2)
    }

    overlap = observed[0]
    assert observed[1] > overlap
    for rounds in (1, 2):
        expected = AmplitudeAmplification.success_probability(overlap, rounds)
        assert observed[rounds] == pytest.approx(expected, abs=0.1)
