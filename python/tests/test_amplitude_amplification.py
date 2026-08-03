"""Tests for the amplitude-amplification round schedule.

Every closed form in :mod:`qdk_chemistry.algorithms.amplitude_amplification.schedule`
is checked against an independent simulation of the two-dimensional invariant
subspace, so the tests validate the mathematics rather than restating it.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import cmath
import math

import pytest

from qdk_chemistry.algorithms.amplitude_amplification import schedule

OVERLAPS = [1e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]


def _simulate(overlap: float, mark_phases: list[float], state_phases: list[float]) -> float:
    """Return the acceptance probability from an explicit 2D subspace simulation.

    The state is represented in the ``(|G>, |B>)`` basis. Each iterate applies a
    phase to the good component and then the partial reflection about the
    prepared state, exactly as the Q# ``GeneralizedAmplitudeAmplificationStep``
    does.
    """
    angle = math.asin(math.sqrt(overlap))
    prepared = (math.sin(angle), math.cos(angle))
    good, bad = complex(prepared[0]), complex(prepared[1])

    for mark_phase, state_phase in zip(mark_phases, state_phases, strict=True):
        good *= cmath.exp(1j * mark_phase)
        projection = prepared[0] * good + prepared[1] * bad
        factor = (1.0 - cmath.exp(1j * state_phase)) * projection
        good -= factor * prepared[0]
        bad -= factor * prepared[1]

    return abs(good) ** 2


def _standard_phases(rounds: int) -> tuple[list[float], list[float]]:
    """Return the phase sequence of plain Grover amplification."""
    return [math.pi] * rounds, [math.pi] * rounds


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("rounds", [0, 1, 2, 3, 7, 15])
def test_success_probability_matches_simulation(overlap: float, rounds: int):
    mark_phases, state_phases = _standard_phases(rounds)
    simulated = _simulate(overlap, mark_phases, state_phases)
    assert schedule.success_probability(overlap, rounds) == pytest.approx(simulated, abs=1e-12)


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_optimal_rounds_is_a_local_maximum(overlap: float):
    best = schedule.optimal_rounds(overlap)
    probability = schedule.success_probability(overlap, best)
    assert probability >= schedule.success_probability(overlap, best + 1)
    if best > 0:
        assert probability >= schedule.success_probability(overlap, best - 1)


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_optimal_rounds_lands_close_to_certainty(overlap: float):
    # (2k+1) theta is within theta of pi/2, so the shortfall is at most sin^2(theta).
    best = schedule.optimal_rounds(overlap)
    shortfall = 1.0 - schedule.success_probability(overlap, best)
    assert shortfall <= overlap + 1e-12


@pytest.mark.parametrize("rounds", [1, 2, 3, 5, 9, 20])
def test_overshoot_overlap_annihilates_acceptance(rounds: int):
    overlap = schedule.overshoot_overlap(rounds)
    assert schedule.success_probability(overlap, rounds) == pytest.approx(0.0, abs=1e-20)


def test_overshoot_is_far_worse_than_undershoot():
    # Underestimating the overlap by 4x wipes out acceptance; overestimating by
    # the same factor merely loses amplification.
    truth = 0.04
    pessimistic = schedule.success_probability_with_assumed_overlap(truth / 4.0, truth)
    optimistic = schedule.success_probability_with_assumed_overlap(truth * 4.0, truth)
    assert pessimistic < 0.05
    assert optimistic > 0.3
    assert optimistic > 5.0 * pessimistic


@pytest.mark.parametrize("max_overlap", OVERLAPS)
def test_safe_rounds_never_overshoots(max_overlap: float):
    rounds = schedule.safe_rounds(max_overlap)
    factor = 2 * rounds + 1
    assert factor * schedule.rotation_angle(max_overlap) <= math.pi / 2.0 + 1e-9

    # Acceptance is monotonically increasing in the true overlap, so a larger
    # than expected overlap can only help.
    probabilities = [
        schedule.success_probability(max_overlap * fraction, rounds) for fraction in (0.05, 0.2, 0.5, 0.8, 1.0)
    ]
    assert probabilities == sorted(probabilities)


def test_safe_rounds_is_the_largest_non_overshooting_choice():
    for max_overlap in OVERLAPS:
        rounds = schedule.safe_rounds(max_overlap)
        if rounds > 0:
            assert (2 * rounds + 3) * schedule.rotation_angle(max_overlap) > math.pi / 2.0


@pytest.mark.parametrize(
    ("min_overlap", "max_overlap"),
    [(1e-4, 1e-3), (1e-3, 0.01), (0.01, 0.1), (0.02, 0.05), (0.1, 0.3), (0.05, 0.9), (0.2, 0.2)],
)
def test_worst_case_matches_dense_sampling(min_overlap: float, max_overlap: float):
    for rounds in range(40):
        samples = [
            schedule.success_probability(min_overlap + (max_overlap - min_overlap) * index / 4000.0, rounds)
            for index in range(4001)
        ]
        predicted = schedule.worst_case_success_probability(rounds, min_overlap, max_overlap)
        assert predicted <= min(samples) + 1e-6


@pytest.mark.parametrize(
    ("min_overlap", "max_overlap"),
    [(1e-4, 1e-3), (1e-3, 0.01), (0.01, 0.1), (0.02, 0.05), (0.1, 0.3), (0.05, 0.9), (0.2, 0.2)],
)
def test_robust_rounds_beats_safe_rounds(min_overlap: float, max_overlap: float):
    robust = schedule.robust_rounds(min_overlap, max_overlap)
    safe = schedule.safe_rounds(max_overlap)
    robust_worst = schedule.worst_case_success_probability(robust, min_overlap, max_overlap)
    safe_worst = schedule.worst_case_success_probability(safe, min_overlap, max_overlap)
    assert robust_worst >= safe_worst - 1e-12

    # And it really is the minimax choice.
    for candidate in range(schedule.optimal_rounds(min_overlap) + 1):
        assert robust_worst >= schedule.worst_case_success_probability(candidate, min_overlap, max_overlap) - 1e-12


def test_exponential_schedule_shape():
    assert schedule.exponential_schedule(0) == []
    assert schedule.exponential_schedule(5) == [1, 2, 2, 2, 3]
    assert schedule.exponential_schedule(4, growth=1.25) == [1, 2, 2, 2]


@pytest.mark.parametrize("overlap", [1e-4, 1e-3, 0.01, 0.05, 0.2])
def test_mean_success_probability_closed_form(overlap: float):
    for sampling_bound in (1, 2, 3, 5, 11, 30):
        direct = sum(schedule.success_probability(overlap, k) for k in range(sampling_bound)) / sampling_bound
        assert schedule._mean_success_probability(overlap, sampling_bound) == pytest.approx(direct, abs=1e-12)


@pytest.mark.parametrize("overlap", [1e-4, 1e-3, 0.01, 0.05, 0.2])
def test_exponential_search_keeps_the_quadratic_speedup(overlap: float):
    expected = schedule.expected_rounds_exponential(overlap)
    # Quadratic in the amplitude, not the probability, with a modest constant.
    assert expected <= 6.0 / math.sqrt(overlap)
    assert expected >= 0.0


@pytest.mark.parametrize("min_overlap", [0.01, 0.05, 0.1, 0.25])
@pytest.mark.parametrize("tolerance", [0.5, 0.2, 0.05])
def test_fixed_point_meets_its_tolerance_everywhere_above_threshold(min_overlap: float, tolerance: float):
    rounds = schedule.fixed_point_rounds(min_overlap, tolerance)
    mark_phases, state_phases = schedule.fixed_point_phases(rounds, tolerance)
    assert len(mark_phases) == rounds
    assert len(state_phases) == rounds

    for index in range(41):
        overlap = min_overlap + (1.0 - min_overlap) * index / 40.0
        probability = _simulate(overlap, mark_phases, state_phases)
        assert probability >= 1.0 - tolerance**2 - 1e-9


@pytest.mark.parametrize("min_overlap", [0.02, 0.1])
def test_fixed_point_removes_the_overshoot_cliff(min_overlap: float):
    tolerance = 0.1
    rounds = schedule.fixed_point_rounds(min_overlap, tolerance)
    mark_phases, state_phases = schedule.fixed_point_phases(rounds, tolerance)

    fixed_point_worst = min(
        _simulate(min_overlap + (1.0 - min_overlap) * index / 200.0, mark_phases, state_phases) for index in range(201)
    )
    plain_worst = min(
        schedule.success_probability(min_overlap + (1.0 - min_overlap) * index / 200.0, rounds) for index in range(201)
    )
    assert fixed_point_worst >= 1.0 - tolerance**2 - 1e-9
    assert plain_worst < 1e-2


def test_fixed_point_phase_symmetry():
    rounds = 5
    mark_phases, state_phases = schedule.fixed_point_phases(rounds, 0.1)
    for index in range(rounds):
        assert mark_phases[index] == pytest.approx(state_phases[rounds - 1 - index])


@pytest.mark.parametrize("rounds", [1, 2, 3, 5, 8, 12])
@pytest.mark.parametrize("tolerance", [0.5, 0.1, 0.01])
def test_fixed_point_phases_realize_the_chebyshev_closed_form(rounds: int, tolerance: float):
    mark_phases, state_phases = schedule.fixed_point_phases(rounds, tolerance)
    for index in range(101):
        overlap = 0.001 + 0.998 * index / 100.0
        simulated = _simulate(overlap, mark_phases, state_phases)
        predicted = schedule.fixed_point_success_probability(overlap, rounds, tolerance)
        assert simulated == pytest.approx(predicted, abs=1e-9)


@pytest.mark.parametrize("rounds", [1, 3, 6, 10])
@pytest.mark.parametrize("tolerance", [0.5, 0.1, 0.01])
def test_fixed_point_success_probability_has_no_cliff(rounds: int, tolerance: float):
    probabilities = [
        schedule.fixed_point_success_probability(0.001 + 0.998 * index / 300.0, rounds, tolerance)
        for index in range(301)
    ]
    assert max(probabilities) <= 1.0 + 1e-12
    assert min(probabilities) >= 0.0

    # Below the plateau the schedule climbs monotonically; above it the acceptance
    # probability only ripples between 1 - tolerance ** 2 and 1, never collapsing.
    queries = 2 * rounds + 1
    scale = math.cosh(math.acosh(1.0 / tolerance) / queries)
    plateau_overlap = 1.0 - 1.0 / scale**2
    ramp = [
        probability
        for index, probability in enumerate(probabilities)
        if 0.001 + 0.998 * index / 300.0 <= plateau_overlap
    ]
    assert ramp == sorted(ramp)
    plateau = probabilities[len(ramp) :]
    assert all(probability >= 1.0 - tolerance**2 - 1e-12 for probability in plateau)


@pytest.mark.parametrize("overlap", [0.0, -0.1, 1.5, math.nan, math.inf])
def test_invalid_overlap_is_rejected(overlap: float):
    with pytest.raises(ValueError, match="overlap"):
        schedule.rotation_angle(overlap)


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="rounds"):
        schedule.success_probability(0.1, -1)
    with pytest.raises(ValueError, match="min_overlap"):
        schedule.robust_rounds(0.5, 0.1)
    with pytest.raises(ValueError, match="growth"):
        schedule.exponential_schedule(3, growth=1.5)
    with pytest.raises(ValueError, match="stages"):
        schedule.exponential_schedule(-1)
    with pytest.raises(ValueError, match="tolerance"):
        schedule.fixed_point_phases(3, 1.0)
    with pytest.raises(ValueError, match="rounds"):
        schedule.fixed_point_phases(0, 0.1)
    with pytest.raises(ValueError, match="tolerance"):
        schedule.fixed_point_success_probability(0.1, 3, 0.0)
