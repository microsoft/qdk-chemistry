"""Simulation tests for the Q# amplitude-amplification module.

The Q# operations are executed on the full-state simulator and compared against
the closed forms in
:mod:`qdk_chemistry.algorithms.amplitude_amplification.schedule`, so the Q# and
Python halves are checked against each other rather than against themselves.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import math

import pytest
from qdk import qsharp as qdk_qsharp

from qdk_chemistry.algorithms.amplitude_amplification import schedule
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

_HAS_QSHARP = importlib.util.find_spec("qdk.qsharp") is not None

pytestmark = pytest.mark.skipif(not _HAS_QSHARP, reason="qdk.qsharp is not installed")

_SHOTS = 4000
_TOLERANCE = 0.04

_HARNESS = """
namespace QDKChemistryAmplitudeAmplificationTests {
    open QDKChemistry.Utils.AmplitudeAmplification;
    import Std.Arithmetic.*;
    import Std.Arrays.*;
    import Std.Convert.*;
    import Std.Math.*;
    import Std.Measurement.*;

    operation PrepareOverlap(theta : Double, register : Qubit[]) : Unit is Adj + Ctl {
        Ry(2.0 * theta, register[0]);
    }

    operation MarkFirstQubit(register : Qubit[], target : Qubit) : Unit is Adj + Ctl {
        CNOT(register[0], target);
    }

    operation RunPlainAmplification(theta : Double, rounds : Int) : Result {
        use register = Qubit[1];
        ApplyAmplitudeAmplification(PrepareOverlap(theta, _), MarkFirstQubit, rounds, register);
        return MResetZ(register[0]);
    }

    operation RunFixedPointAmplification(
        theta : Double,
        markPhases : Double[],
        statePhases : Double[],
    ) : Result {
        use register = Qubit[1];
        ApplyFixedPointAmplitudeAmplification(
            PrepareOverlap(theta, _),
            MarkFirstQubit,
            markPhases,
            statePhases,
            register,
        );
        return MResetZ(register[0]);
    }

    operation MarkPhaseIndex(numBits : Int, value : Int, accepted : Int[]) : Result {
        use register = Qubit[numBits];
        use flag = Qubit();
        ApplyXorInPlace(value, register);
        ApplyAcceptedPhaseMark(register, accepted, flag);
        let outcome = MResetZ(flag);
        ResetAll(register);
        return outcome;
    }

    operation MarkQpeAcceptance(
        numPhaseQubits : Int,
        phaseValue : Int,
        numAncillas : Int,
        ancillaValue : Int,
        signalAncillaIndices : Int[],
        accepted : Int[],
    ) : Result {
        use register = Qubit[numPhaseQubits + numAncillas];
        use flag = Qubit();
        // The QPE phase register is stored most-significant-bit first.
        ApplyXorInPlace(phaseValue, Reversed(register[0..numPhaseQubits - 1]));
        ApplyXorInPlace(ancillaValue, register[numPhaseQubits...]);
        ApplyQpeAcceptanceMark(numPhaseQubits, signalAncillaIndices, accepted, register, flag);
        let outcome = MResetZ(flag);
        ResetAll(register);
        return outcome;
    }

    operation PrepareGuidingState(theta : Double, systems : Qubit[]) : Unit is Adj + Ctl {
        Ry(2.0 * theta, systems[0]);
    }

    operation PrepareUniformPhaseRegister(register : Qubit[]) : Unit is Adj + Ctl {
        ApplyToEachCA(H, register);
    }

    /// Controlled U^power for the single-qubit unitary U|1> = exp(2 pi i / 4)|1>.
    operation ApplyControlledPower(power : Int, control : Qubit, targets : Qubit[]) : Unit is Adj + Ctl {
        Controlled R1([control], (2.0 * PI() * IntAsDouble(power) / 4.0, targets[0]));
    }

    operation RunAmplifiedQpe(theta : Double, rounds : Int) : (Result[], Bool) {
        let unitaries = [
            ApplyControlledPower(4, _, _),
            ApplyControlledPower(2, _, _),
            ApplyControlledPower(1, _, _),
        ];
        return RunAmplifiedStandardQPE(
            PrepareGuidingState(theta, _),
            unitaries,
            PrepareUniformPhaseRegister,
            3,
            1,
            0,
            [],
            [2],
            rounds
        );
    }
}
"""

_NAMESPACE = "QDKChemistryAmplitudeAmplificationTests"


@pytest.fixture(scope="module")
def qsharp_module():
    """Load the chemistry Q# utilities plus the test harness exactly once."""
    # Touch the proxy so the chemistry utilities are evaluated first.
    _ = QSHARP_UTILS.AmplitudeAmplification
    qdk_qsharp.eval(_HARNESS)
    return qdk_qsharp


def _acceptance_frequency(qsharp_module, expression: str) -> float:
    """Return the fraction of shots that land in the good subspace."""
    outcomes = qsharp_module.run(expression, shots=_SHOTS)
    return sum(1 for outcome in outcomes if str(outcome) == "One") / _SHOTS


@pytest.mark.parametrize(("overlap", "rounds"), [(0.05, 0), (0.05, 2), (0.05, 3), (0.1, 1), (0.1, 2), (0.25, 1)])
def test_plain_amplification_matches_the_closed_form(qsharp_module, overlap: float, rounds: int):
    theta = schedule.rotation_angle(overlap)
    observed = _acceptance_frequency(qsharp_module, f"{_NAMESPACE}.RunPlainAmplification({theta}, {rounds})")
    assert observed == pytest.approx(schedule.success_probability(overlap, rounds), abs=_TOLERANCE)


def test_plain_amplification_overshoots_when_the_overlap_is_underestimated(qsharp_module):
    # Three rounds are optimal for a = 0.02 but wrap past the maximum for a = 0.25.
    overlap = 0.25
    rounds = schedule.optimal_rounds(0.02)
    theta = schedule.rotation_angle(overlap)
    observed = _acceptance_frequency(qsharp_module, f"{_NAMESPACE}.RunPlainAmplification({theta}, {rounds})")
    assert observed == pytest.approx(schedule.success_probability(overlap, rounds), abs=_TOLERANCE)
    assert observed < 0.5

    safe = schedule.safe_rounds(overlap)
    safe_observed = _acceptance_frequency(qsharp_module, f"{_NAMESPACE}.RunPlainAmplification({theta}, {safe})")
    assert safe_observed > observed


@pytest.mark.parametrize(("rounds", "tolerance"), [(2, 0.3), (4, 0.2), (6, 0.1)])
@pytest.mark.parametrize("overlap", [0.05, 0.5, 0.9])
def test_fixed_point_amplification_matches_the_chebyshev_closed_form(
    qsharp_module, rounds: int, tolerance: float, overlap: float
):
    theta = schedule.rotation_angle(overlap)
    mark_phases, state_phases = schedule.fixed_point_phases(rounds, tolerance)
    expression = f"{_NAMESPACE}.RunFixedPointAmplification({theta}, {list(mark_phases)}, {list(state_phases)})".replace(
        "'", ""
    )
    observed = _acceptance_frequency(qsharp_module, expression)
    predicted = schedule.fixed_point_success_probability(overlap, rounds, tolerance)
    assert observed == pytest.approx(predicted, abs=_TOLERANCE)


@pytest.mark.parametrize(
    "accepted",
    [[0, 1, 6, 7], [0, 1, 2], [5, 6, 7], [2, 3, 5], [4], list(range(8)), []],
)
def test_accepted_phase_mark_is_exact(qsharp_module, accepted: list[int]):
    num_bits = 3
    for value in range(1 << num_bits):
        expression = f"{_NAMESPACE}.MarkPhaseIndex({num_bits}, {value}, {accepted})"
        outcome = str(qsharp_module.run(expression, shots=1)[0])
        assert (outcome == "One") == (value in accepted)


def test_qpe_acceptance_requires_the_window_and_clean_signal_ancillas(qsharp_module):
    accepted = [0, 1, 6, 7]
    for phase_value in range(8):
        for ancilla_value in range(4):
            expression = f"{_NAMESPACE}.MarkQpeAcceptance(3, {phase_value}, 2, {ancilla_value}, [0, 1], {accepted})"
            outcome = str(qsharp_module.run(expression, shots=1)[0])
            assert (outcome == "One") == (phase_value in accepted and ancilla_value == 0)


def test_accepted_phase_interval_lengths_detect_wrapped_windows():
    utils = QSHARP_UTILS.AmplitudeAmplification
    assert tuple(utils.AcceptedPhaseIntervalLengths(3, [0, 1, 6, 7])) == (True, 2, 2)
    assert tuple(utils.AcceptedPhaseIntervalLengths(3, [0, 1, 2])) == (True, 3, 0)
    assert tuple(utils.AcceptedPhaseIntervalLengths(3, [5, 6, 7])) == (True, 0, 3)
    is_wrapped, _, _ = tuple(utils.AcceptedPhaseIntervalLengths(3, [2, 3, 5]))
    assert is_wrapped is False


def test_rotation_angle_agrees_with_the_qsharp_convention():
    # The Q# harness prepares Ry(2 * theta), whose |1> amplitude is sin(theta).
    for overlap in (0.05, 0.25, 0.81):
        assert math.sin(schedule.rotation_angle(overlap)) ** 2 == pytest.approx(overlap)


@pytest.mark.parametrize("rounds", [0, 1, 2, 3])
def test_amplified_qpe_boosts_acceptance_without_changing_the_answer(qsharp_module, rounds: int):
    # A single system qubit whose |1> eigenstate has phase 1/4, prepared with a
    # deliberately poor 5% overlap. The accepted window is the single bin 2/8.
    overlap = 0.05
    theta = schedule.rotation_angle(overlap)
    outcomes = qsharp_module.run(f"{_NAMESPACE}.RunAmplifiedQpe({theta}, {rounds})", shots=_SHOTS)

    frequency = sum(1 for _, accepted in outcomes if accepted) / _SHOTS
    assert frequency == pytest.approx(schedule.success_probability(overlap, rounds), abs=_TOLERANCE)

    # Amplification changes how often the window is accepted, never what it accepts:
    # every accepted shot must decode to the phase index 2, big-endian 0b010.
    wrong = [bits for bits, accepted in outcomes if accepted and [str(bit) for bit in bits] != ["Zero", "One", "Zero"]]
    assert wrong == []
