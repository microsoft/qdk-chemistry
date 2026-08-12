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
from qdk_chemistry.algorithms.amplitude_amplification.amplitude_amplification import AmplitudeAmplification
from qdk_chemistry.algorithms.amplitude_amplification.qpe_subspace import (
    QPESubspaceMarking,
    _phase_bins_from_energy,
)
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context


def _walk_eigenvalue_from_phase(normalization: float = 1.0):
    r"""Return the qubitization-walk law :math:`E = \lambda\cos(2\pi\varphi)`."""
    return lambda phase_fraction: normalization * math.cos(2 * math.pi * (phase_fraction % 1.0))


def _evolution_eigenvalue_from_phase(time: float = 1.0):
    r"""Return the time-evolution law :math:`E = -\arg(e^{2\pi i\varphi})/t`."""

    def eigenvalue_from_phase(phase_fraction: float) -> float:
        angle = (phase_fraction % 1.0) * 2 * math.pi
        if angle > math.pi:
            angle -= 2 * math.pi
        return -angle / time

    return eigenvalue_from_phase


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
):
    """Build a measurement-free QPE circuit, its unitary representation, and its register layout."""
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
    # The oracle reads the phase-to-energy law off the same unitary the builder estimates.
    unitary_representation = builder._create_nested("unitary_builder").run(qubit_hamiltonian)

    # MakeStandardQPECircuit allocates numBits + Length(systems) + numAncillaQubits qubits.
    parameters = preparation._qsharp_factory.parameter
    num_ancilla_qubits = parameters["numAncillaQubits"]
    num_qubits = parameters["numBits"] + len(parameters["systems"]) + num_ancilla_qubits
    return preparation, unitary_representation, num_qubits, num_ancilla_qubits


def _amplified_qpe_circuit(
    qubit_hamiltonian: QubitOperator,
    state_preparation: Circuit,
    target_energy: float,
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
    **settings,
):
    """Compose a coherent QPE preparation with amplitude amplification."""
    state_prep_oracle, unitary_representation, num_qubits, num_ancilla_qubits = _qpe_preparation(
        qubit_hamiltonian,
        state_preparation,
        num_bits=num_bits,
        mapper=mapper,
        unitary=unitary,
    )
    good_state_oracle = create("subspace_oracle", target_energy=target_energy).run(
        state_prep_oracle, unitary_representation
    )
    marked = _marked_bin_ranges(good_state_oracle)

    algorithm = create("amplitude_amplification", **settings)
    circuit = algorithm.run(state_prep_oracle, good_state_oracle)
    return circuit, num_qubits, num_ancilla_qubits, marked


def _marked_bin_ranges(oracle: Circuit) -> list[tuple[int, int]]:
    """Read back the half-open phase-bin ranges an oracle circuit marks."""
    parameters = oracle._qsharp_factory.parameter
    return list(zip(parameters["lowerBounds"], parameters["upperBounds"], strict=True))


def _accepted_phase_counts(
    circuit: Circuit,
    num_bits: int,
    num_ancilla: int,
    accepted_ranges: list[tuple[int, int]],
    shots: int,
) -> dict[str, int]:
    """Execute a circuit and count, per phase bitstring, the shots in the good subspace.

    The whole register is measured. The executor reverses the Q# results, so the little-endian
    phase register lands MSB first at the end of the key and the trailing ancillas land at the front.
    """
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    counts: dict[str, int] = {}
    for bitstring, count in executor.run(circuit, shots=shots).bitstring_counts.items():
        phase_bits, ancilla_bits = bitstring[-num_bits:], bitstring[:num_ancilla]
        phase_bin = int(phase_bits, 2)
        if any(bit != "0" for bit in ancilla_bits):
            continue
        if not any(lower <= phase_bin < upper for lower, upper in accepted_ranges):
            continue
        counts[phase_bits] = counts.get(phase_bits, 0) + count
    return counts


def _dominant_accepted_phase(
    circuit: Circuit,
    num_bits: int,
    num_ancilla: int,
    accepted_ranges: list[tuple[int, int]],
    shots: int = 400,
) -> str:
    """Execute a circuit and return the most common bitstring from the good subspace."""
    counts = _accepted_phase_counts(circuit, num_bits, num_ancilla, accepted_ranges, shots)
    assert counts, f"No shot landed in the accepted window {accepted_ranges}."
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


def test_subspace_oracle_is_registered():
    assert available("subspace_oracle") == ["qdk_qpe_subspace"]
    default = create("subspace_oracle")
    assert default.name() == "qdk_qpe_subspace"
    assert default.type_name() == "subspace_oracle"
    assert isinstance(default, QPESubspaceMarking)


def test_subspace_oracle_target_energy_defaults_to_unset():
    """There is no meaningful default energy, so the setting starts as NaN and run refuses it."""
    algorithm = create("subspace_oracle")
    assert math.isnan(algorithm.settings().get("target_energy"))

    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, unitary_representation, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    with pytest.raises(ValueError, match="target_energy"):
        algorithm.run(qpe_circuit, unitary_representation)


def test_subspace_oracle_rejects_a_circuit_that_is_not_qpe():
    """The oracle reads its register layout off the QPE circuit, so it refuses anything else."""
    hamiltonian = _diagonal_hamiltonian()
    _, unitary_representation, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    with pytest.raises(ValueError, match="standard QPE circuit"):
        create("subspace_oracle", target_energy=-1.0).run(_all_ones_marking_oracle(), unitary_representation)


def test_subspace_oracle_requires_a_unitary_representation():
    """The phase-to-energy law comes from the unitary, so nothing else can stand in for it."""
    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, _, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    with pytest.raises(TypeError, match="UnitaryRepresentation"):
        create("subspace_oracle", target_energy=-1.0).run(qpe_circuit, hamiltonian)


def test_subspace_oracle_rejects_a_unitary_on_a_different_register():
    """A unitary of the wrong width would place the block-encoding ancillas at the wrong indices."""
    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, _, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    # A Trotter step on the same Hamiltonian carries no block-encoding ancillas.
    mismatched = create("hamiltonian_unitary_builder", "trotter", time=1.0).run(hamiltonian)
    with pytest.raises(ValueError, match="does not"):
        create("subspace_oracle", target_energy=-1.0).run(qpe_circuit, mismatched)


def test_amplified_qpe_circuit():
    """Check that amplitude amplification can be applied to a QPE circuit."""
    # The |11> eigenvector has energy -lambda, which the qubitization walk maps
    # to the phase bin 1/2 -- index 8 of 16, big-endian 0b1000.
    hamiltonian = _diagonal_hamiltonian()
    circuit, _, num_ancilla, marked = _amplified_qpe_circuit(
        hamiltonian,
        _guiding_state(0.3, 3),
        -hamiltonian.schatten_norm,
        rounds=2,
    )
    assert marked == [(8, 9)]
    assert _dominant_accepted_phase(circuit, 4, num_ancilla, marked) == "1000"


def test_amplified_qpe_circuit_with_trotter():
    """Check that amplitude amplification can be applied to a QPE circuit with the pauli-sequence mapper."""
    # e^{-iHt} with t = 1 maps the eigenvalue -pi/2 to the phase 1/4, bin 4 of 16. The
    # pauli-sequence mapper has no block-encoding ancillas, so nothing is post-selected.
    circuit, _, num_ancilla, marked = _amplified_qpe_circuit(
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        -math.pi / 2,
        mapper="pauli_sequence",
        unitary=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
        rounds=1,
    )
    assert marked == [(4, 5)]
    assert _dominant_accepted_phase(circuit, 4, num_ancilla, marked, shots=200) == "0100"


def test_amplified_qpe_acceptance_follows_the_round_count():
    """Acceptance on the QPE window obeys the same closed form as a plain preparation."""
    hamiltonian = _diagonal_hamiltonian()
    shots = 2000
    observed = {}
    for rounds in (0, 1, 2):
        circuit, _, num_ancilla, marked = _amplified_qpe_circuit(
            hamiltonian,
            _guiding_state(0.3, 3),
            -hamiltonian.schatten_norm,
            rounds=rounds,
        )
        counts = _accepted_phase_counts(circuit, 4, num_ancilla, marked, shots)
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
    """The oracle circuit runs on its own: the top of the band is bin 0, which the all-zeros register holds."""
    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, unitary_representation, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=3)
    marks_bin_zero = create("subspace_oracle", target_energy=hamiltonian.schatten_norm).run(
        qpe_circuit, unitary_representation
    )
    marks_bin_four = create("subspace_oracle", target_energy=-hamiltonian.schatten_norm).run(
        qpe_circuit, unitary_representation
    )
    assert _marked_bin_ranges(marks_bin_zero) == [(0, 1)]
    assert _marked_bin_ranges(marks_bin_four) == [(4, 5)]
    assert executor.run(marks_bin_zero, shots=20).bitstring_counts == {"1": 20}
    assert executor.run(marks_bin_four, shots=20).bitstring_counts == {"0": 20}


def test_target_energy_selects_the_hand_computed_bin():
    """The energy -lambda reproduces the hand-computed bin 8 of 16."""
    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, unitary_representation, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    oracle = create("subspace_oracle", target_energy=-hamiltonian.schatten_norm).run(
        qpe_circuit, unitary_representation
    )
    parameters = oracle._qsharp_factory.parameter
    assert (parameters["lowerBounds"], parameters["upperBounds"]) == ([8], [9])


def test_target_energy_marks_both_walk_branches():
    """A walk has eigenvalues exp(+-i arccos(E/lambda)), so one energy needs two mirrored bins."""
    coefficients = np.array([0.3, 0.5])
    hamiltonian = QubitOperator(pauli_strings=["ZI", "IZ"], coefficients=coefficients)
    # |01> has energy 0.3 - 0.5 = -0.2, away from +-lambda, so the two branches are distinct bins.
    energy = float(coefficients[0] - coefficients[1])
    qpe_circuit, unitary_representation, _, _ = _qpe_preparation(hamiltonian, _guiding_state(1.0, 1), num_bits=4)
    oracle = create("subspace_oracle", target_energy=energy).run(qpe_circuit, unitary_representation)
    # arccos(-0.25) / 2pi = 0.2902 -> bin 4.64, and the mirror at 1 - 0.2902 -> bin 11.36.
    assert _marked_bin_ranges(oracle) == [(5, 6), (11, 12)]


def test_marked_bins_stay_inside_the_phase_register():
    """Mirroring bin 0 names bin 2^n, which the phase register cannot hold, so it must be dropped."""
    hamiltonian = _diagonal_hamiltonian()
    qpe_circuit, unitary_representation, _, _ = _qpe_preparation(hamiltonian, _guiding_state(0.3, 3), num_bits=4)
    oracle = create("subspace_oracle", target_energy=hamiltonian.schatten_norm).run(qpe_circuit, unitary_representation)
    parameters = oracle._qsharp_factory.parameter
    assert (parameters["lowerBounds"], parameters["upperBounds"]) == ([0], [1])

    executor = create("circuit_executor", "qdk_sparse_state_simulator")
    assert executor.run(oracle, shots=20).bitstring_counts == {"1": 20}


@pytest.mark.parametrize(
    ("target_energy", "expected_bins"),
    [
        # E = +lambda is phase 0 and E = -lambda is phase 0.5; both are their own mirror.
        pytest.param(1.0, [(0, 1)], id="top-of-band"),
        pytest.param(-1.0, [(8, 9)], id="bottom-of-band"),
        # Energies outside [-lambda, lambda] clamp onto the edge.
        pytest.param(2.0, [(0, 1)], id="above-band"),
        pytest.param(-3.0, [(8, 9)], id="below-band"),
        pytest.param(0.0, [(4, 5), (12, 13)], id="middle-of-band"),
        pytest.param(-0.25, [(5, 6), (11, 12)], id="off-centre"),
        # A bin near an end still mirrors, even though the end itself does not.
        pytest.param(0.95, [(1, 2), (15, 16)], id="near-top-of-band"),
    ],
)
def test_walk_energies_map_onto_representable_bins(target_energy, expected_bins):
    """The block-encoding law is inverted onto both branches and stays inside the phase register."""
    bins = _phase_bins_from_energy(target_energy, _walk_eigenvalue_from_phase(), num_phase_qubits=4)
    assert bins == expected_bins
    assert all(0 <= start < stop <= 16 for start, stop in bins)


@pytest.mark.parametrize(
    ("target_energy", "expected_bins"),
    [
        pytest.param(0.0, [(0, 1)], id="zero"),
        pytest.param(-math.pi / 2, [(4, 5)], id="negative"),
        pytest.param(math.pi / 2, [(12, 13)], id="positive"),
        # No bin sits between bin 0 at E = 0 and bin 15 at E = 2 pi / 16, so an energy in
        # that gap takes the nearer of the two rather than falling off the band.
        pytest.param(0.05, [(0, 1)], id="just-above-zero"),
        pytest.param(0.35, [(15, 16)], id="just-below-the-last-bin"),
    ],
)
def test_time_evolution_energies_use_their_own_law(target_energy, expected_bins):
    """A time evolution follows E = -arg/t, and the same inversion handles it without special casing."""
    bins = _phase_bins_from_energy(target_energy, _evolution_eigenvalue_from_phase(time=1.0), num_phase_qubits=4)
    assert bins == expected_bins


@pytest.mark.parametrize(
    "target_energy",
    [
        pytest.param(math.nan, id="nan"),
        pytest.param(math.inf, id="infinite"),
        pytest.param(-math.inf, id="negative-infinite"),
    ],
)
def test_non_finite_energies_are_rejected(target_energy):
    """An energy that does not name a point on the band is rejected."""
    with pytest.raises(ValueError, match="finite"):
        _phase_bins_from_energy(target_energy, _walk_eigenvalue_from_phase(), num_phase_qubits=4)


def test_amplified_circuit_exposes_a_measurement_free_operation():
    """The result carries an unmeasured qsharp_op, so a caller can append its own measurement."""
    hamiltonian = _diagonal_hamiltonian()
    circuit, num_qubits, _, _ = _amplified_qpe_circuit(
        hamiltonian,
        _guiding_state(0.3, 3),
        -hamiltonian.schatten_norm,
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
