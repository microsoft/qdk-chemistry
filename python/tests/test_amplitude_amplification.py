"""Tests for amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math

import numpy as np
from qdk import qsharp

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.algorithms.amplitude_amplification import AmplitudeAmplification, phase_marking_oracle
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS


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

    # MakeStandardQPECircuit allocates numBits + Length(systems) + numAncillaQubits qubits.
    parameters = preparation._qsharp_factory.parameter
    num_system_qubits = len(parameters["systems"])
    num_ancilla_qubits = parameters["numAncillaQubits"]
    num_qubits = parameters["numBits"] + num_system_qubits + num_ancilla_qubits
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
) -> tuple[Circuit, int, int]:
    """Compose a coherent QPE preparation with amplitude amplification."""
    state_prep_oracle, num_qubits, signal_ancilla_indices = _qpe_preparation(
        qubit_hamiltonian,
        state_preparation,
        num_bits=num_bits,
        mapper=mapper,
        unitary=unitary,
    )
    good_state_oracle = phase_marking_oracle(num_bits, accepted_range, signal_ancilla_indices)

    algorithm = create("amplitude_amplification")
    for key, value in settings.items():
        algorithm.settings().update(key, value)
    circuit = algorithm.run(state_prep_oracle, good_state_oracle, num_qubits=num_qubits)
    return circuit, num_qubits, len(signal_ancilla_indices)


def _accepted_phase_counts(
    circuit: Circuit, num_bits: int, num_ancilla: int, accepted_range: tuple[int, int], shots: int
) -> dict[str, int]:
    """Execute a circuit and count, per phase bitstring, the shots in the good subspace.

    The whole register is measured. The executor reverses the Q# results, so the little-endian
    phase register lands MSB first at the end of the key and the trailing ancillas land at the front.
    """
    lower_bound, upper_bound = accepted_range
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    counts: dict[str, int] = {}
    for bitstring, count in executor.run(circuit, shots=shots).bitstring_counts.items():
        phase_bits, ancilla_bits = bitstring[-num_bits:], bitstring[:num_ancilla]
        if any(bit != "0" for bit in ancilla_bits) or not lower_bound <= int(phase_bits, 2) < upper_bound:
            continue
        counts[phase_bits] = counts.get(phase_bits, 0) + count
    return counts


def _dominant_accepted_phase(
    circuit: Circuit, num_bits: int, num_ancilla: int, accepted_range: tuple[int, int], shots: int = 400
) -> str:
    """Execute a circuit and return the most common bitstring from the good subspace."""
    counts = _accepted_phase_counts(circuit, num_bits, num_ancilla, accepted_range, shots)
    assert counts, f"No shot landed in the accepted window {accepted_range}."
    return max(counts, key=lambda phase: counts[phase])


def test_amplitude_amplification_is_registered():
    assert available("amplitude_amplification") == ["base"]
    default = create("amplitude_amplification")
    assert default.name() == "base"
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
    circuit, _, num_ancilla = _amplified_qpe_circuit(
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        accepted,
        rounds=2,
    )
    assert _dominant_accepted_phase(circuit, 4, num_ancilla, accepted) == "1000"


def test_amplified_qpe_circuit_with_trotter():
    """Check that amplitude amplification can be applied to a QPE circuit with the pauli-sequence mapper."""
    # The pauli-sequence mapper has no block-encoding ancillas, so the accepted
    # window is defined purely on the phase register.
    accepted = (4, 5)
    circuit, _, num_ancilla = _amplified_qpe_circuit(
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        accepted,
        mapper="pauli_sequence",
        unitary=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
        rounds=1,
    )
    # e^{-iHt} with t = 1 maps the eigenvalue -pi/2 to the phase 1/4, bin 4 of 16.
    assert _dominant_accepted_phase(circuit, 4, num_ancilla, accepted, shots=200) == "0100"


def test_amplified_qpe_acceptance_follows_the_round_count():
    """More rounds drive more shots into the accepted window at this overlap."""
    accepted = (8, 9)
    shots = 2000
    observed = {}
    for rounds in (0, 1, 2):
        circuit, _, num_ancilla = _amplified_qpe_circuit(
            _diagonal_hamiltonian(),
            _guiding_state(0.3, 3),
            accepted,
            rounds=rounds,
        )
        counts = _accepted_phase_counts(circuit, 4, num_ancilla, accepted, shots)
        observed[rounds] = sum(counts.values()) / shots

    assert observed[1] > observed[0]
    assert observed[2] > observed[1]


def test_marking_oracle_circuit_is_executable():
    """The oracle circuit runs on its own: it marks the all-zeros register when bin 0 is accepted."""
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    assert executor.run(phase_marking_oracle(3, (0, 1)), shots=20).bitstring_counts == {"1": 20}
    assert executor.run(phase_marking_oracle(3, (1, 8)), shots=20).bitstring_counts == {"0": 20}


def test_amplified_circuit_exposes_a_measurement_free_operation():
    """The result carries an unmeasured qsharp_op, so a caller can append its own measurement."""
    circuit, num_qubits, _ = _amplified_qpe_circuit(
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        (8, 9),
        rounds=1,
    )
    assert circuit._qsharp_op is not None

    measured = Circuit(
        qsharp_factory=QsharpFactoryData(
            program=QSHARP_UTILS.MeasurementBasis.MakeMeasurementCircuit,
            parameter={
                "baseCircuit": circuit._qsharp_op,
                "bases": [qsharp.Pauli.Z] * num_qubits,
                "numQubits": num_qubits,
            },
        )
    )
    counts = create("circuit_executor", "qdk_sparse_state_simulator").run(measured, shots=50).bitstring_counts
    assert sum(counts.values()) == 50
    assert all(len(bitstring) == num_qubits for bitstring in counts)
