"""Tests for amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math

import numpy as np
import pytest
from qdk import qsharp

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.algorithms.amplitude_amplification import (
    AmplitudeAmplification,
    _phase_bins_from_energy_range,
    phase_marking_oracle,
)
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context


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


def _all_ones_marking_oracle() -> Circuit:
    """Return an oracle that flips the flag on the all-ones state."""
    context = get_qsharp_context()
    context.eval(
        "operation AmplitudeAmplificationTestMarkAllOnes(register : Qubit[], flag : Qubit) : "
        "Unit is Adj + Ctl { Controlled X(register, flag); }"
    )
    operation = context.eval("AmplitudeAmplificationTestMarkAllOnes")
    return Circuit(
        qsharp_op=operation,
        qsharp_factory=QsharpFactoryData(program=operation, parameter={}),
    )


def _qpe_preparation(
    qubit_hamiltonian: QubitOperator,
    state_preparation: Circuit,
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
) -> tuple[Circuit, int, int]:
    """Build a measurement-free QPE circuit plus its register layout."""
    unitary = unitary or AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True)
    builder = create(
        "qpe_circuit_builder",
        "qdk_standard",
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", mapper),
        unitary_builder=unitary,
        measure_phase=False,
    )
    preparation = builder.run(state_preparation=state_preparation, qubit_hamiltonian=qubit_hamiltonian)[0]

    # MakeStandardQPECircuit allocates numBits + Length(systems) + numAncillaQubits qubits.
    parameters = preparation._qsharp_factory.parameter
    num_ancilla_qubits = parameters["numAncillaQubits"]
    num_qubits = parameters["numBits"] + len(parameters["systems"]) + num_ancilla_qubits
    return preparation, num_qubits, num_ancilla_qubits


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
    state_prep_oracle, num_qubits, num_ancilla_qubits = _qpe_preparation(
        qubit_hamiltonian,
        state_preparation,
        num_bits=num_bits,
        mapper=mapper,
        unitary=unitary,
    )
    good_state_oracle = phase_marking_oracle(state_prep_oracle, accepted_range)

    algorithm = create("amplitude_amplification", **settings)
    circuit = algorithm.run(state_prep_oracle, good_state_oracle)
    return circuit, num_qubits, num_ancilla_qubits


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
    assert available("amplitude_amplification") == ["qdk_base"]
    default = create("amplitude_amplification")
    assert default.name() == "qdk_base"
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
    """Acceptance on the QPE window obeys the same closed form as a plain preparation."""
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

    angle = math.asin(math.sqrt(observed[0]))
    for rounds in (1, 2):
        expected = math.sin((2 * rounds + 1) * angle) ** 2
        assert abs(observed[rounds] - expected) < 0.1, f"rounds={rounds}: {observed[rounds]} != {expected}"


def test_amplification_matches_the_closed_form_and_overshoots():
    r"""P(good) tracks :math:`\sin^2((2k+1)\vartheta)` for k = 0..5, decline included."""
    amplitude = 0.3
    angle = math.asin(amplitude)
    shots = 4000
    state_prep_oracle = _guiding_state(amplitude, 3)
    good_state_oracle = _all_ones_marking_oracle()
    executor = create("circuit_executor", "qdk_sparse_state_simulator")

    observed = []
    for rounds in range(6):
        circuit = create("amplitude_amplification", rounds=rounds).run(state_prep_oracle, good_state_oracle)
        counts = executor.run(circuit, shots=shots).bitstring_counts
        probability = counts.get("11", 0) / shots
        expected = math.sin((2 * rounds + 1) * angle) ** 2
        assert abs(probability - expected) < 0.05, f"rounds={rounds}: {probability} != {expected}"
        observed.append(probability)

    # The peak sits at k = 2 for a = 0.09, so more rounds must do worse, not better.
    assert observed[2] == max(observed)
    assert observed[3] < observed[2]
    assert observed[4] < observed[3]


def test_unestimable_state_prep_reports_a_runtime_error():
    """The register width comes from a resource estimate, so a preparation that cannot be costed fails cleanly."""
    # The marking oracle takes (Qubit[], Qubit), so costing it as a standalone entry point fails.
    unestimable = _all_ones_marking_oracle()
    with pytest.raises(RuntimeError, match="register width"):
        create("amplitude_amplification").run(unestimable, _all_ones_marking_oracle())


def test_marking_oracle_circuit_is_executable():
    """The oracle circuit runs on its own: it marks the all-zeros register when bin 0 is accepted."""
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    qpe_circuit, _, _ = _qpe_preparation(_diagonal_hamiltonian(), _guiding_state(0.3, 3), num_bits=3)
    assert executor.run(phase_marking_oracle(qpe_circuit, (0, 1)), shots=20).bitstring_counts == {"1": 20}
    assert executor.run(phase_marking_oracle(qpe_circuit, (1, 8)), shots=20).bitstring_counts == {"0": 20}


def test_energy_window_selects_the_same_bins_as_the_hand_computed_window():
    """An energy window around -lambda reproduces the hand-computed bin 8 of 16."""
    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    oracle = phase_marking_oracle(
        qpe_circuit,
        target_energy_range=(-math.inf, -0.99 * hamiltonian.schatten_norm),
        qubit_hamiltonian=hamiltonian,
    )
    parameters = oracle._qsharp_factory.parameter
    assert (parameters["lowerBounds"], parameters["upperBounds"]) == ([8], [9])


def test_energy_window_marks_both_walk_branches():
    """A walk has eigenvalues exp(+-i arccos(E/lambda)), so one energy needs two mirrored bins."""
    coefficients = np.array([0.3, 0.5])
    hamiltonian = QubitOperator(pauli_strings=["ZI", "IZ"], coefficients=coefficients)
    # |01> has energy 0.3 - 0.5 = -0.2, away from +-lambda, so the two branches are distinct bins.
    energy = float(coefficients[0] - coefficients[1])
    qpe_circuit, _, _ = _qpe_preparation(hamiltonian, _guiding_state(1.0, 1), num_bits=4)
    oracle = phase_marking_oracle(
        qpe_circuit,
        target_energy_range=(energy - 0.02, energy + 0.02),
        qubit_hamiltonian=hamiltonian,
    )
    parameters = oracle._qsharp_factory.parameter
    marked = {
        phase_bin
        for lower, upper in zip(parameters["lowerBounds"], parameters["upperBounds"], strict=True)
        for phase_bin in range(lower, upper)
    }
    # arccos(-0.25) / 2pi = 0.2902 -> bin 4.64, and the mirror at 1 - 0.2902 -> bin 11.36.
    assert len(parameters["lowerBounds"]) == 2
    assert {5, 11} <= marked
    assert 8 not in marked


def test_energy_window_rejects_an_ambiguous_or_incomplete_request():
    """The two ways of naming the target are mutually exclusive, and energy needs a Hamiltonian."""
    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    with pytest.raises(ValueError, match="exactly one"):
        phase_marking_oracle(qpe_circuit)
    with pytest.raises(ValueError, match="exactly one"):
        phase_marking_oracle(qpe_circuit, (8, 9), target_energy_range=(-2.0, -1.0), qubit_hamiltonian=hamiltonian)
    with pytest.raises(ValueError, match="qubit_hamiltonian"):
        phase_marking_oracle(qpe_circuit, target_energy_range=(-2.0, -1.0))
    with pytest.raises(ValueError, match="low < high"):
        phase_marking_oracle(qpe_circuit, target_energy_range=(-1.0, -2.0), qubit_hamiltonian=hamiltonian)


@pytest.mark.parametrize(
    ("target_energy_range", "expected_bins"),
    [
        # arccos is decreasing, so E = +lambda is phase 0 and E = -lambda is phase 0.5.
        # Both are their own mirror, so each edge of the band collapses to a single bin.
        pytest.param((0.99, math.inf), [(0, 1)], id="top-of-band"),
        pytest.param((1.0, math.inf), [(0, 1)], id="top-of-band-exact"),
        pytest.param((-math.inf, -0.99), [(8, 9)], id="bottom-of-band"),
        pytest.param((-math.inf, -1.0), [(8, 9)], id="bottom-of-band-exact"),
        pytest.param((-math.inf, math.inf), [(0, 16)], id="whole-band"),
        # Bounds outside [-lambda, lambda] clamp onto the edge rather than marking nothing.
        pytest.param((2.0, 3.0), [(0, 1)], id="entirely-above-band"),
        pytest.param((-3.0, -2.0), [(8, 9)], id="entirely-below-band"),
        # A window straddling zero keeps the two mirrored branches apart.
        pytest.param((-0.1, 0.1), [(4, 5), (12, 13)], id="straddling-zero"),
    ],
)
def test_energy_window_edges_map_onto_representable_bins(target_energy_range, expected_bins):
    """Energy windows at and beyond the band edges stay inside the phase register."""
    bins = _phase_bins_from_energy_range(target_energy_range, normalization=1.0, num_phase_qubits=4)
    assert bins == expected_bins
    assert all(0 <= start < stop <= 16 for start, stop in bins)


@pytest.mark.parametrize(
    "target_energy_range",
    [
        pytest.param((math.nan, 1.0), id="nan-low"),
        pytest.param((1.0, math.nan), id="nan-high"),
        pytest.param((1.0, 1.0), id="empty-window"),
        pytest.param((1.0, 0.0), id="reversed"),
        pytest.param((math.inf, math.inf), id="both-infinite"),
        pytest.param((math.inf, -math.inf), id="reversed-infinite"),
    ],
)
def test_energy_window_rejects_degenerate_bounds(target_energy_range):
    """A window that does not name a nonempty interval is rejected instead of clamped."""
    with pytest.raises(ValueError, match="low < high"):
        _phase_bins_from_energy_range(target_energy_range, normalization=1.0, num_phase_qubits=4)


def test_energy_window_at_the_top_of_the_band_builds_a_runnable_oracle():
    """Mirroring bin 0 names bin 2^n, which the phase register cannot hold, so it must be dropped."""
    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    oracle = phase_marking_oracle(
        qpe_circuit,
        target_energy_range=(0.99 * hamiltonian.schatten_norm, math.inf),
        qubit_hamiltonian=hamiltonian,
    )
    parameters = oracle._qsharp_factory.parameter
    assert (parameters["lowerBounds"], parameters["upperBounds"]) == ([0], [1])

    # An out-of-range bound reaches Q# as ApplyXorInPlace(2^n, Qubit[n]) and fails to run.
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    assert executor.run(oracle, shots=20).bitstring_counts == {"1": 20}


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
