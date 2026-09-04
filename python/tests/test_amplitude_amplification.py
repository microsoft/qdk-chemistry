"""Tests for amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from qdk import qsharp

from qdk_chemistry.algorithms import available, create
from qdk_chemistry.algorithms.amplitude_amplification.amplitude_amplification import AmplitudeAmplification
from qdk_chemistry.algorithms.amplitude_amplification.qpe_subspace import QPESubspaceMarking
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    Configuration,
    ModelOrbitals,
    QubitOperator,
    StateVectorContainer,
    Wavefunction,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import PauliProductFormulaContainer
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer, QuantumWalkContainer
from qdk_chemistry.utils.qsharp import QSHARP_UTILS, get_qsharp_context


def _walk_container(normalization: float = 1.0) -> LCUWalkContainer:
    r"""Return a walk container carrying the law :math:`E = \lambda\cos(2\pi\varphi)`."""
    return LCUWalkContainer(block_encoding=None, power=1, scale=normalization)


def _trotter_container(evolution_time: float = 1.0) -> PauliProductFormulaContainer:
    r"""Return a container carrying the law :math:`E = -\theta/t`, which wraps at :math:`\varphi = 1/2`."""
    return PauliProductFormulaContainer(step_terms=[], step_reps=1, num_qubits=1, scale=evolution_time)


def _diagonal_hamiltonian() -> QubitOperator:
    """Return H = -(pi/4) ZI - (pi/4) IZ, whose spectrum is {pi/2, 0, 0, -pi/2} with |11> on top."""
    coefficient = -math.pi / 4.0
    return QubitOperator(pauli_strings=["ZI", "IZ"], coefficients=np.array([coefficient, coefficient]))


def _interior_hamiltonian(marked: str) -> QubitOperator:
    r"""Return H whose ``marked`` eigenvector sits at :math:`+\lambda\cos(\pi/4)`.

    That is phase :math:`1/8`, an exact bin of any register at least three bits wide, yet
    strictly inside the band rather than at an edge where the two walk branches merge and the
    block-encoding ancilla decouples. Its partner sits at phase :math:`3/8`, which no window
    holding the marked one reaches.

    Args:
        marked: Which eigenvector carries the positive energy, ``"00"`` or ``"11"``.

    """
    # (x - y) / (x + y) = cos(pi/4) fixes x / y = 3 + 2*sqrt(2).
    ratio = 3.0 + 2.0 * math.sqrt(2.0)
    sign = -1.0 if marked == "11" else 1.0
    return QubitOperator(
        pauli_strings=["ZI", "IZ"],
        coefficients=np.array([sign * ratio, -sign]),
    )


def _guiding_state(amplitude: float, index: int, num_qubits: int = 2) -> Circuit:
    """Prepare a state with the given amplitude on one computational basis state."""
    remainder = math.sqrt(max(0.0, 1.0 - amplitude**2))
    other = (index + 1) % (1 << num_qubits)
    # from_bitstring reads bits little-endian, so a basis index is its reversed binary form.
    configurations = [Configuration.from_bitstring(format(basis, f"0{num_qubits}b")[::-1]) for basis in (index, other)]
    guiding_state = Wavefunction(
        StateVectorContainer(np.array([amplitude, remainder]), configurations, ModelOrbitals(num_qubits))
    )
    return create("state_prep", "dense_pure_state").run(guiding_state)


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


def _subspace_oracle(
    qubit_hamiltonian: QubitOperator,
    energy_lower_bound: float,
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
) -> Circuit:
    """Build the oracle marking the QPE window above ``energy_lower_bound``."""
    return create(
        "amplitude_amplification_oracle",
        "qdk_qpe_subspace",
        energy_lower_bound=energy_lower_bound,
        qpe_circuit_builder=AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_standard",
            num_bits=num_bits,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", mapper),
            unitary_builder=unitary or AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        ),
    ).run(qubit_hamiltonian)


def _amplified_qpe_circuit(
    qubit_hamiltonian: QubitOperator,
    state_preparation: Circuit,
    energy_lower_bound: float,
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
    **settings,
):
    """Amplify a state preparation against the QPE window above ``energy_lower_bound``."""
    good_state_oracle = _subspace_oracle(
        qubit_hamiltonian,
        energy_lower_bound,
        num_bits=num_bits,
        mapper=mapper,
        unitary=unitary,
    )
    algorithm = create("amplitude_amplification", **settings)
    circuit = algorithm.run(state_preparation, good_state_oracle)
    return circuit, _marked_bin_ranges(good_state_oracle)


def _marked_bin_ranges(oracle: Circuit) -> list[tuple[int, int]]:
    """Read back the half-open phase-bin ranges an oracle circuit marks."""
    parameters = oracle._qsharp_factory.parameter
    return list(zip(parameters["lowerBounds"], parameters["upperBounds"], strict=True))


def _measure(circuit: Circuit, shots: int = 400) -> dict[str, int]:
    """Execute a circuit and return its bitstring counts."""
    return create("circuit_executor", "qdk_sparse_state_simulator").run(circuit, shots=shots).bitstring_counts


def test_amplitude_amplification_is_registered():
    default = create("amplitude_amplification")
    assert default.name() == "qdk_base"
    assert default.type_name() == "amplitude_amplification"
    assert isinstance(default, AmplitudeAmplification)


def test_subspace_oracle_is_registered():
    assert "qdk_qpe_subspace" in available("amplitude_amplification_oracle")
    oracle = create("amplitude_amplification_oracle")
    assert oracle.name() == "qdk_qpe_subspace"
    assert oracle.type_name() == "amplitude_amplification_oracle"
    assert isinstance(oracle, QPESubspaceMarking)


def test_subspace_oracle_is_configured_by_an_energy_and_a_nested_builder():
    """The oracle holds the energy bound to mark and a reference to the QPE that measures it."""
    settings = create("amplitude_amplification_oracle").settings()
    assert math.isnan(settings.get("energy_lower_bound"))
    assert "qpe_circuit_builder" in settings
    for key in ("num_bits", "unitary_builder", "controlled_circuit_mapper"):
        assert key not in settings


def test_subspace_oracle_run_takes_the_hamiltonian_alone():
    """Its run names the marked subspace by a Hamiltonian and returns the one oracle circuit."""
    hamiltonian = _diagonal_hamiltonian()
    circuit = _subspace_oracle(hamiltonian, hamiltonian.schatten_norm / 2, num_bits=3)
    assert isinstance(circuit, Circuit)


def test_subspace_oracle_is_not_a_phase_estimation_builder():
    """It builds an oracle, not a phase estimation, and the QPE it nests has to be a standard one."""
    assert "qdk_qpe_subspace" not in available("qpe_circuit_builder")

    hamiltonian = _diagonal_hamiltonian()
    oracle = create(
        "amplitude_amplification_oracle",
        "qdk_qpe_subspace",
        energy_lower_bound=hamiltonian.schatten_norm / 2,
        qpe_circuit_builder=AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_iterative",
            num_bits=3,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        ),
    )
    # An iterative builder measures and feeds back, so its circuit cannot be undone coherently.
    with pytest.raises(TypeError, match="standard"):
        oracle.run(hamiltonian)


def test_subspace_oracle_rejects_a_nested_builder_without_phase_bits():
    """A register no bits wide holds no phase to mark."""
    hamiltonian = _diagonal_hamiltonian()
    with pytest.raises(ValueError, match="num_bits"):
        _subspace_oracle(hamiltonian, hamiltonian.schatten_norm / 2, num_bits=0)


def test_amplified_qpe_circuit():
    """Check that amplitude amplification can be applied to a QPE circuit."""
    # The spectrum is {+lambda, 0, 0, -lambda}, so a bound of +lambda/2 keeps only the |11>
    # eigenvector. The walk law puts that band at phi in [0, 1/6] and its mirror [5/6, 1),
    # which the register splits at either end.
    hamiltonian = _diagonal_hamiltonian()
    circuit, marked = _amplified_qpe_circuit(
        hamiltonian,
        _guiding_state(0.3, 3),
        hamiltonian.schatten_norm / 2,
        rounds=2,
    )
    assert marked == [(0, 3), (14, 16)]
    counts = _measure(circuit)
    assert max(counts, key=lambda bitstring: counts[bitstring]) == "11"


def test_amplified_qpe_circuit_with_trotter():
    """Check that amplitude amplification can be applied to a QPE circuit with the pauli-sequence mapper."""
    # e^{-iHt} with t = 1 maps the eigenvalue +pi/2 to the phase 3/4, bin 12 of 16. A bound of
    # 1.0 sits between it and the next eigenvalue at 0. The pauli-sequence mapper has no
    # block-encoding ancillas, so nothing is post-selected.
    circuit, marked = _amplified_qpe_circuit(
        _diagonal_hamiltonian(),
        _guiding_state(0.3, 3),
        1.0,
        mapper="pauli_sequence",
        unitary=AlgorithmRef("hamiltonian_unitary_builder", "trotter", time=1.0),
        rounds=1,
    )
    assert marked == [(9, 14)]
    counts = _measure(circuit, shots=200)
    assert max(counts, key=lambda bitstring: counts[bitstring]) == "11"


def test_amplified_qpe_acceptance_follows_the_round_count():
    """Acceptance on the QPE window obeys the same closed form as a plain preparation."""
    hamiltonian = _diagonal_hamiltonian()
    shots = 2000
    observed = {}
    for rounds in (0, 1, 2):
        circuit, _ = _amplified_qpe_circuit(
            hamiltonian,
            _guiding_state(0.3, 3),
            hamiltonian.schatten_norm / 2,
            rounds=rounds,
        )
        observed[rounds] = _measure(circuit, shots=shots).get("11", 0) / shots

    assert observed[1] > observed[0]
    assert observed[2] > observed[1]

    angle = math.asin(math.sqrt(observed[0]))
    for rounds in (1, 2):
        expected = math.sin((2 * rounds + 1) * angle) ** 2
        assert abs(observed[rounds] - expected) < 0.1, f"rounds={rounds}: {observed[rounds]} != {expected}"


def test_subspace_oracle_flags_an_interior_eigenstate():
    r"""The marking tests the phase register alone, never the block-encoding ancilla.

    |00> sits strictly inside the band, so the walk splits it across both settings of the
    signal ancilla; the other oracle tests mark a band-edge eigenvector, where it decouples.
    """
    oracle = _subspace_oracle(_interior_hamiltonian(marked="00"), 1.0, num_bits=4)
    # Phase 1/8 is bin 2, strictly inside the accepted [0, 4) rather than at its edge.
    assert _marked_bin_ranges(oracle) == [(0, 4), (13, 16)]
    assert _measure(oracle, shots=40) == {"1": 40}


def test_amplified_qpe_acceptance_at_an_interior_eigenvalue():
    r"""P(good) tracks :math:`\sin^2((2k+1)\vartheta)` when the marked eigenvector is interior.

    Pins the closed form for an eigenvector the walk really does split, not just for the
    band-edge ones the other tests use.
    """
    hamiltonian = _interior_hamiltonian(marked="11")
    amplitude = 0.3
    angle = math.asin(amplitude)
    shots = 2000
    observed = []
    for rounds in range(4):
        circuit, marked = _amplified_qpe_circuit(
            hamiltonian,
            # |11> is the marked eigenvector and |00> holds the rest of the guiding state at
            # the negated energy, so the window reaching one cannot reach the other.
            _guiding_state(amplitude, 3),
            1.0,
            rounds=rounds,
        )
        assert marked == [(0, 4), (13, 16)]
        observed.append(_measure(circuit, shots=shots).get("11", 0) / shots)

    for rounds, probability in enumerate(observed):
        expected = math.sin((2 * rounds + 1) * angle) ** 2
        assert abs(probability - expected) < 0.05, f"rounds={rounds}: {probability} != {expected}"


def test_a_declared_register_width_is_used_without_a_resource_estimate():
    """A state prep circuit carrying ``num_qubits`` reports that width instead of estimating one."""
    prepared = _guiding_state(0.3, 3)
    state_prep_oracle = Circuit(
        qsharp_op=prepared._qsharp_op,
        qsharp_factory=prepared._qsharp_factory,
        num_qubits=3,
    )
    good_state_oracle = _all_ones_marking_oracle()

    circuit = create("amplitude_amplification", rounds=1).run(state_prep_oracle, good_state_oracle)
    assert circuit._qsharp_factory.parameter["numQubits"] == 3


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


def test_marking_oracle_circuit_is_executable():
    """The oracle circuit runs on its own, over an all-zeros register, and the flag tracks the energy.

    Both Hamiltonians give the same accepted bins, so the only thing that differs is where
    |00> sits.
    """
    # |00> sits at +lambda*cos(pi/4), phase 1/8, which is bin 1 of 8 and inside the window.
    flag_fires = _subspace_oracle(_interior_hamiltonian(marked="00"), 1.0, num_bits=3)
    # |00> sits at -lambda, phase 1/2, which is bin 4 of 8 and outside it.
    hamiltonian = _diagonal_hamiltonian()
    flag_stays_down = _subspace_oracle(hamiltonian, hamiltonian.schatten_norm / 2, num_bits=3)
    assert _marked_bin_ranges(flag_fires) == [(0, 2), (7, 8)]
    assert _marked_bin_ranges(flag_stays_down) == [(0, 2), (7, 8)]
    assert _measure(flag_fires, shots=20) == {"1": 20}
    assert _measure(flag_stays_down, shots=20) == {"0": 20}


@pytest.mark.parametrize(
    ("energy_lower_bound", "expected_bins"),
    [
        # E = +lambda is phase 0, the only bin at the top of the band.
        pytest.param(1.0, [(0, 1)], id="top-of-band"),
        pytest.param(0.9, [(0, 2), (15, 16)], id="near-top-of-band"),
        pytest.param(0.5, [(0, 3), (14, 16)], id="upper-half-of-band"),
        # E = -lambda is phase 1/2, the bottom bin, so anything above it still leaves that
        # bin out and keeps the marking a proper subspace.
        pytest.param(math.nextafter(-1.0, 0.0), [(0, 8), (9, 16)], id="just-above-the-bottom"),
    ],
)
def test_walk_energies_select_a_band_around_phase_zero(energy_lower_bound, expected_bins):
    """The walk law is symmetric about phi = 1/2, so the accepted band splits at either end."""
    marked = QPESubspaceMarking._marked_phase_bins(energy_lower_bound, _walk_container(), num_phase_qubits=4)
    assert marked == expected_bins


def test_walk_bins_stay_symmetric_at_an_exact_energy_boundary():
    """Mirrored walk phases use one floating-point value at an inclusive boundary."""
    container = _walk_container()
    energy_lower_bound = container.eigenvalue_from_phase(1 / 8)
    assert QPESubspaceMarking._marked_phase_bins(energy_lower_bound, container, num_phase_qubits=3) == [
        (0, 2),
        (7, 8),
    ]


@pytest.mark.parametrize(
    ("energy_lower_bound", "expected_bins"),
    [
        # The law falls from 0 to -pi over the first half and wraps to +pi across it, so a
        # positive bound accepts a band below phase 1, and a negative one adds it to that half.
        pytest.param(1.0, [(9, 14)], id="upper-half-of-range"),
        pytest.param(0.0, [(0, 1), (9, 16)], id="non-negative-energies"),
        pytest.param(-1.0, [(0, 3), (9, 16)], id="above-minus-one"),
        # E = -pi/t is phase 1/2, the one bin the wrap leaves at the bottom of the range.
        pytest.param(math.nextafter(-math.pi, 0.0), [(0, 8), (9, 16)], id="just-above-the-bottom"),
    ],
)
def test_trotter_energies_select_a_band_across_the_branch_cut(energy_lower_bound, expected_bins):
    """The time-evolution law wraps at phi = 1/2, and the marked bins follow it across."""
    marked = QPESubspaceMarking._marked_phase_bins(energy_lower_bound, _trotter_container(), num_phase_qubits=4)
    assert marked == expected_bins


@pytest.mark.parametrize(
    ("energy_lower_bound", "message"),
    [
        pytest.param(math.nan, "must be set", id="unset"),
        pytest.param(math.inf, "must be a finite energy", id="infinite"),
        pytest.param(-math.inf, "must be a finite energy", id="negative-infinite"),
    ],
)
def test_run_rejects_an_unusable_energy_lower_bound(energy_lower_bound, message):
    """Both the unset and the non-finite bound are refused before any circuit is built."""
    with pytest.raises(ValueError, match=message):
        _subspace_oracle(_diagonal_hamiltonian(), energy_lower_bound, num_bits=3)


@pytest.mark.parametrize(
    ("container", "energy_lower_bound"),
    [
        pytest.param(_walk_container(), 2.0, id="walk-above-band"),
        pytest.param(_trotter_container(), 4.0, id="trotter-above-range"),
    ],
)
def test_energy_above_the_encoded_range_is_rejected(container, energy_lower_bound):
    """No phase carries an energy over the encoded range, so the bound is refused, not clamped."""
    with pytest.raises(ValueError, match="No phase bin"):
        QPESubspaceMarking._marked_phase_bins(energy_lower_bound, container, num_phase_qubits=4)


@pytest.mark.parametrize(
    ("container", "energy_lower_bound"),
    [
        # The bottom bin sits exactly at the foot of each encoded range, so a bound there
        # already clears every bin; below it is the same answer, only more so.
        pytest.param(_walk_container(), -1.0, id="walk-band-bottom"),
        pytest.param(_walk_container(), -2.0, id="walk-below-band"),
        pytest.param(_trotter_container(), -math.pi, id="trotter-range-bottom"),
        pytest.param(_trotter_container(), -4.0, id="trotter-below-range"),
    ],
)
def test_energy_every_bin_clears_is_rejected(container, energy_lower_bound):
    """A bound the whole register clears marks everything, which is no subspace at all."""
    with pytest.raises(ValueError, match="Every phase bin"):
        QPESubspaceMarking._marked_phase_bins(energy_lower_bound, container, num_phase_qubits=4)


@pytest.mark.parametrize(
    ("scale", "message"),
    [
        pytest.param(2.0, "No phase bin", id="above-the-band"),
        pytest.param(-2.0, "Every phase bin", id="below-the-band"),
    ],
)
def test_a_degenerate_bound_is_refused_when_building_the_oracle(scale, message):
    """Neither end reaches the circuit: a flag that never fires and one that always does are both dead."""
    hamiltonian = _diagonal_hamiltonian()
    with pytest.raises(ValueError, match=message):
        _subspace_oracle(hamiltonian, scale * hamiltonian.schatten_norm, num_bits=3)


class _QuarterCutLaw:
    """A phase law shaped like the time-evolution one, but cut at phi = 1/4.

    Guards against a phi = 1/2 split being baked back into the search.
    """

    def eigenvalue_from_phase(self, phase_fraction: float) -> float:
        angle = (phase_fraction % 1.0) * 2.0 * math.pi
        if angle > math.pi / 2.0:
            angle -= 2.0 * math.pi
        return -angle


def _assert_marks_exactly_the_bins_above(container, energy_lower_bound, num_phase_qubits):
    """Assert the marking holds every bin clearing the bound and no other, or refuses it."""
    phase_bin_count = 1 << num_phase_qubits
    context = f"E={energy_lower_bound} on a {phase_bin_count}-bin register"
    # The walk law is even about phi = 1/2, so bins k and phase_bin_count - k hold one energy.
    # cos rounds the pair to either side of a bound that falls exactly between them, so read
    # both off the lower bin the way the marking does.
    mirrored = isinstance(container, QuantumWalkContainer)

    def energy_of(phase_bin: int) -> float:
        canonical = min(phase_bin, phase_bin_count - phase_bin) if mirrored else phase_bin
        return container.eigenvalue_from_phase(canonical / phase_bin_count)

    expected = {phase_bin for phase_bin in range(phase_bin_count) if energy_of(phase_bin) >= energy_lower_bound}
    if len(expected) in (0, phase_bin_count):
        with pytest.raises(ValueError, match="phase bin"):
            QPESubspaceMarking._marked_phase_bins(energy_lower_bound, container, num_phase_qubits)
        return
    bins = QPESubspaceMarking._marked_phase_bins(energy_lower_bound, container, num_phase_qubits)
    marked = {phase_bin for start, stop in bins for phase_bin in range(start, stop)}
    assert marked == expected, context
    # A walk eigenspace reaches the phase register as a mirrored pair, so marking one branch
    # without the other would leave the oracle unable to restore the ancillas it releases.
    if mirrored:
        assert {(phase_bin_count - phase_bin) % phase_bin_count for phase_bin in marked} == marked, context
    # Sorted, non-empty and separated by at least one bin, so the ranges cannot double-flip
    # the flag: MarkAcceptedPhase applies MarkPhaseRange to each one independently.
    assert all(start < stop for start, stop in bins), context
    assert all(previous[1] < following[0] for previous, following in itertools.pairwise(bins)), context


@pytest.mark.parametrize("num_phase_qubits", [3, 4, 5, 6])
@pytest.mark.parametrize("energy_lower_bound", [-6.0, -3.0, -1.0, 0.0, 1.0, 1.5])
def test_marked_bins_follow_a_law_that_turns_away_from_phase_half(energy_lower_bound, num_phase_qubits):
    """A container that turns at phi = 1/4 still gets exactly the bins its law says it should."""
    _assert_marks_exactly_the_bins_above(_QuarterCutLaw(), energy_lower_bound, num_phase_qubits)


def test_a_target_just_inside_the_band_edge_marks_only_the_top_bin():
    """One ulp under the top of the band leaves phase 0 the only bin above the bound.

    A crossing solved through arccos would land nowhere near a bin here; comparing energies
    bin by bin never inverts anything.
    """
    marked = QPESubspaceMarking._marked_phase_bins(math.nextafter(1.0, 0.0), _walk_container(1.0), num_phase_qubits=12)
    assert marked == [(0, 1)]


@pytest.mark.parametrize("num_phase_qubits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize(
    ("container", "energies"),
    [
        pytest.param(
            _walk_container(2.5),
            (-2.5, -1.4, 0.0, 1.25, 2.4999, 2.5),
            id="walk",
        ),
        pytest.param(
            _trotter_container(0.5),
            (-2 * math.pi, -3.3, -1.0, 0.0, 2.7, 6.2),
            id="trotter",
        ),
    ],
)
def test_marked_bins_hold_exactly_the_energies_above_the_target(container, energies, num_phase_qubits):
    """The ranges cover every bin clearing the bound and no other, register width and all."""
    for energy_lower_bound in energies:
        _assert_marks_exactly_the_bins_above(container, energy_lower_bound, num_phase_qubits)


def test_amplified_circuit_exposes_a_measurement_free_operation():
    """The result carries an unmeasured qsharp_op, so a caller can append its own measurement."""
    hamiltonian = _diagonal_hamiltonian()
    num_qubits = 2  # the amplified register is the system register alone
    circuit, _ = _amplified_qpe_circuit(
        hamiltonian,
        _guiding_state(0.3, 3),
        hamiltonian.schatten_norm / 2,
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
