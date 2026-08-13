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
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import (
    IterativeQpeCircuitBuilder,
    QpeCircuitBuilder,
    StandardQpeCircuitBuilder,
)
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
from qdk_chemistry.data.unitary_representation.containers.quantum_walk import LCUWalkContainer
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
    target_energy: float,
    *,
    num_bits: int = 4,
    mapper: str = "prepare_select_prepare",
    unitary: AlgorithmRef | None = None,
) -> Circuit:
    """Build the oracle marking the QPE window that holds ``target_energy``."""
    oracle = create(
        "qpe_circuit_builder",
        "qdk_qpe_subspace",
        num_bits=num_bits,
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", mapper),
        unitary_builder=unitary or AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        target_energy=target_energy,
    )
    # The oracle ignores the state preparation; it takes the register it is applied to as it finds it.
    # It follows the qpe_circuit_builder contract, so it returns a list holding the one oracle.
    return oracle.run(_guiding_state(1.0, 0), qubit_hamiltonian)[0]


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
    """Amplify a state preparation against the QPE window holding ``target_energy``."""
    good_state_oracle = _subspace_oracle(
        qubit_hamiltonian,
        target_energy,
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
    assert "qdk_qpe_subspace" in available("qpe_circuit_builder")
    oracle = create("qpe_circuit_builder", "qdk_qpe_subspace")
    assert oracle.name() == "qdk_qpe_subspace"
    assert oracle.type_name() == "qpe_circuit_builder"
    assert isinstance(oracle, QPESubspaceMarking)


def test_subspace_oracle_shares_the_qpe_circuit_builder_settings():
    """The oracle is configured like the QPE builder it wraps, plus the energy to mark."""
    algorithm = create("qpe_circuit_builder", "qdk_qpe_subspace")
    assert math.isnan(algorithm.settings().get("target_energy"))
    for key in ("num_bits", "unitary_builder", "controlled_circuit_mapper"):
        assert key in algorithm.settings()


def test_subspace_oracle_conforms_to_the_builder_contract():
    """Its run returns a list of circuits, as every qpe_circuit_builder does."""
    hamiltonian = _diagonal_hamiltonian()
    oracle = create(
        "qpe_circuit_builder",
        "qdk_qpe_subspace",
        num_bits=3,
        controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
        target_energy=hamiltonian.schatten_norm / 2,
    )
    circuits = oracle.run(_guiding_state(1.0, 0), hamiltonian)
    assert isinstance(circuits, list)
    assert len(circuits) == 1
    assert all(isinstance(circuit, Circuit) for circuit in circuits)


def test_subspace_oracle_cannot_stand_in_for_a_phase_estimation_builder():
    """It builds an oracle, not a phase estimation, so a PhaseEstimation must refuse to select it."""
    oracle = create("qpe_circuit_builder", "qdk_qpe_subspace")
    # It implements the builder interface, but derives directly from QpeCircuitBuilder rather
    # than from either base phase estimation dispatches on, so it is rejected up front instead
    # of being run as a phase estimation.
    assert isinstance(oracle, QpeCircuitBuilder)
    assert not isinstance(oracle, StandardQpeCircuitBuilder)
    assert not isinstance(oracle, IterativeQpeCircuitBuilder)

    hamiltonian = _diagonal_hamiltonian()
    qpe = create("phase_estimation", "qdk_standard")
    qpe.settings().set(
        "qpe_circuit_builder",
        AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_qpe_subspace",
            num_bits=3,
            controlled_circuit_mapper=AlgorithmRef("controlled_circuit_mapper", "prepare_select_prepare"),
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=True),
            target_energy=hamiltonian.schatten_norm / 2,
        ),
    )
    with pytest.raises(TypeError, match="Expected qpe_circuit_builder to be an instance of StandardQpeCircuitBuilder"):
        qpe.run(state_preparation=_guiding_state(1.0, 0), qubit_hamiltonian=hamiltonian)


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
    """The oracle circuit runs on its own, over an all-zeros register.

    |00> is the eigenvector at the bottom of the band, which the walk maps to phase bin 4 of 8.
    """
    hamiltonian = _diagonal_hamiltonian()
    # The walk normalization is the Schatten norm, so its negation is exactly the bottom of
    # the band: the lowest bound the encoding can represent, and one every bin clears.
    marks_every_bin = _subspace_oracle(hamiltonian, -hamiltonian.schatten_norm, num_bits=3)
    marks_the_top = _subspace_oracle(hamiltonian, hamiltonian.schatten_norm / 2, num_bits=3)
    assert _marked_bin_ranges(marks_every_bin) == [(0, 8)]
    assert _marked_bin_ranges(marks_the_top) == [(0, 2), (7, 8)]
    assert _measure(marks_every_bin, shots=20) == {"1": 20}
    assert _measure(marks_the_top, shots=20) == {"0": 20}


@pytest.mark.parametrize(
    ("target_energy", "expected_bins"),
    [
        # E = +lambda is phase 0, the only bin at the top of the band.
        pytest.param(1.0, [(0, 1)], id="top-of-band"),
        pytest.param(0.9, [(0, 2), (15, 16)], id="near-top-of-band"),
        pytest.param(0.5, [(0, 3), (14, 16)], id="upper-half-of-band"),
        # E = -lambda is phase 1/2, so a bound there accepts the whole register.
        pytest.param(-1.0, [(0, 16)], id="bottom-of-band"),
    ],
)
def test_walk_energies_select_a_band_around_phase_zero(target_energy, expected_bins):
    """The walk law is symmetric about phi = 1/2, so the accepted band splits at either end."""
    assert QPESubspaceMarking._marked_phase_bins(target_energy, _walk_container(), num_phase_qubits=4) == expected_bins


@pytest.mark.parametrize(
    ("target_energy", "expected_bins"),
    [
        # The law falls from 0 to -pi over the first half and wraps to +pi across it, so a
        # positive bound accepts a band below phase 1, and a negative one adds it to that half.
        pytest.param(1.0, [(9, 14)], id="upper-half-of-range"),
        pytest.param(0.0, [(0, 1), (9, 16)], id="non-negative-energies"),
        pytest.param(-1.0, [(0, 3), (9, 16)], id="above-minus-one"),
        # E = -pi/t is phase 1/2, the one bin the wrap leaves at the bottom of the range.
        pytest.param(-math.pi, [(0, 16)], id="bottom-of-range"),
    ],
)
def test_trotter_energies_select_a_band_across_the_branch_cut(target_energy, expected_bins):
    """The time-evolution law wraps at phi = 1/2, and the marked bins follow it across."""
    marked = QPESubspaceMarking._marked_phase_bins(target_energy, _trotter_container(), num_phase_qubits=4)
    assert marked == expected_bins


@pytest.mark.parametrize(
    ("target_energy", "message"),
    [
        pytest.param(math.nan, "must be set", id="unset"),
        pytest.param(math.inf, "must be a finite energy", id="infinite"),
        pytest.param(-math.inf, "must be a finite energy", id="negative-infinite"),
    ],
)
def test_run_rejects_an_unusable_target_energy(target_energy, message):
    """Both the unset and the non-finite energy are refused before any circuit is built."""
    with pytest.raises(ValueError, match=message):
        _subspace_oracle(_diagonal_hamiltonian(), target_energy, num_bits=3)


@pytest.mark.parametrize(
    ("container", "target_energy", "message"),
    [
        pytest.param(_walk_container(), 2.0, "outside the band", id="walk-above-band"),
        pytest.param(_walk_container(), -2.0, "outside the band", id="walk-below-band"),
        pytest.param(_trotter_container(), 4.0, "aliases", id="trotter-above-range"),
        pytest.param(_trotter_container(), -4.0, "aliases", id="trotter-below-range"),
    ],
)
def test_energy_the_encoding_cannot_represent_is_rejected(container, target_energy, message):
    """No phase carries an energy off the encoded range, so the bound is refused, not clamped."""
    with pytest.raises(ValueError, match=message):
        QPESubspaceMarking._marked_phase_bins(target_energy, container, num_phase_qubits=4)


def test_energy_above_the_band_is_rejected():
    """A bound over every bin would leave the flag dead, so building the oracle refuses it."""
    hamiltonian = _diagonal_hamiltonian()
    with pytest.raises(ValueError, match="outside the band"):
        _subspace_oracle(hamiltonian, 2 * hamiltonian.schatten_norm, num_bits=3)


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
    """The ranges cover every bin clearing the target and no other, register width and all."""
    phase_bin_count = 1 << num_phase_qubits
    for target_energy in energies:
        context = f"E={target_energy} on a {phase_bin_count}-bin register"
        expected = {
            phase_bin
            for phase_bin in range(phase_bin_count)
            if container.eigenvalue_from_phase(phase_bin / phase_bin_count) >= target_energy
        }
        if not expected:
            # A narrow register can fall short of an energy the encoding still represents.
            with pytest.raises(ValueError, match="No phase bin"):
                QPESubspaceMarking._marked_phase_bins(target_energy, container, num_phase_qubits)
            continue
        bins = QPESubspaceMarking._marked_phase_bins(target_energy, container, num_phase_qubits)
        assert {phase_bin for start, stop in bins for phase_bin in range(start, stop)} == expected, context
        # Sorted, non-empty and separated by at least one bin, so the ranges cannot double-flip
        # the flag: MarkPhaseRanges applies each one independently.
        assert all(start < stop for start, stop in bins), context
        assert all(previous[1] < following[0] for previous, following in itertools.pairwise(bins)), context


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
