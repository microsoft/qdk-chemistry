"""Tests for the robust phase estimation post-processing math (utils/rpe.py)."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest

from qdk_chemistry.utils.rpe import (
    angular_distance,
    energy_from_rpe_angle,
    expectation_from_counts,
    num_rounds,
    qdrift_phase_to_energy,
    qdrift_schedule,
    rpe_angle_update,
)


# --------------------------------------------------------------------------------------------
# expectation_from_counts
# --------------------------------------------------------------------------------------------
def test_expectation_from_counts_all_zero_is_plus_one() -> None:
    assert expectation_from_counts({"0": 100}) == 1.0


def test_expectation_from_counts_all_one_is_minus_one() -> None:
    assert expectation_from_counts({"1": 100}) == -1.0


def test_expectation_from_counts_balanced_is_zero() -> None:
    assert expectation_from_counts({"0": 50, "1": 50}) == 0.0


def test_expectation_from_counts_known_ratio() -> None:
    assert expectation_from_counts({"0": 75, "1": 25}) == pytest.approx(0.5)


def test_expectation_from_counts_empty_is_guarded() -> None:
    assert expectation_from_counts({}) == 0.0


# --------------------------------------------------------------------------------------------
# num_rounds
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("lambda_norm", "epsilon", "expected"),
    [
        (1.0, 1.0, 0),  # lambda == epsilon -> single (base) round
        (0.5, 1.0, 0),  # lambda < epsilon
        (8.0, 1.0, 3),  # log2(8) = 3
        (10.0, 1.0, 4),  # ceil(log2(10)) = 4
    ],
)
def test_num_rounds_values(lambda_norm: float, epsilon: float, expected: int) -> None:
    assert num_rounds(lambda_norm, epsilon) == expected


def test_num_rounds_rejects_nonpositive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        num_rounds(1.0, 0.0)


# --------------------------------------------------------------------------------------------
# qdrift_schedule
# --------------------------------------------------------------------------------------------
def test_qdrift_schedule_formula() -> None:
    shots, samples = qdrift_schedule(total_rounds=3, round_index=0)
    assert shots == int(np.ceil(np.e * (11 + 4 * 3)))
    assert samples == 2  # 2 ** (2*0 + 1)


def test_qdrift_schedule_shots_shrink_samples_grow() -> None:
    total = 5
    shots = [qdrift_schedule(total, m)[0] for m in range(total + 1)]
    samples = [qdrift_schedule(total, m)[1] for m in range(total + 1)]
    assert shots == sorted(shots, reverse=True)  # non-increasing
    assert samples == sorted(samples)  # increasing
    assert all(samples[m] == 2 ** (2 * m + 1) for m in range(total + 1))


# --------------------------------------------------------------------------------------------
# angular_distance
# --------------------------------------------------------------------------------------------
def test_angular_distance_is_symmetric_and_wraps() -> None:
    assert angular_distance(0.1, 0.4) == pytest.approx(0.3)
    assert angular_distance(0.4, 0.1) == pytest.approx(0.3)
    # 0.1 and (2*pi - 0.1) are 0.2 apart across the wrap, not 2*pi - 0.2
    assert angular_distance(0.1, 2 * np.pi - 0.1) == pytest.approx(0.2)


def test_angular_distance_max_is_pi() -> None:
    assert angular_distance(0.0, np.pi) == pytest.approx(np.pi)


# --------------------------------------------------------------------------------------------
# rpe_angle_update
# --------------------------------------------------------------------------------------------
def test_rpe_angle_update_round_zero_returns_measured_phase() -> None:
    assert rpe_angle_update(0.0, 0.3, round_index=0) == pytest.approx(0.3)


def test_rpe_angle_update_picks_alias_closest_to_previous() -> None:
    # Round 1: candidates are phi/2 and (phi + 2*pi)/2. With phi = 0.4 the
    # candidates are 0.2 and ~3.34; a previous estimate near 3.3 must select
    # the second branch, not the numerically smaller one.
    phi = 0.4
    near_high = rpe_angle_update(3.3, phi, round_index=1)
    assert near_high == pytest.approx((phi + 2 * np.pi) / 2)
    near_low = rpe_angle_update(0.0, phi, round_index=1)
    assert near_low == pytest.approx(phi / 2)


# --------------------------------------------------------------------------------------------
# energy_from_rpe_angle
# --------------------------------------------------------------------------------------------
def test_energy_from_rpe_angle_inverts_sign_and_time() -> None:
    # angle = -E * tau  =>  E = -angle / tau
    assert energy_from_rpe_angle(-0.75 * (np.pi / 2), np.pi / 2) == pytest.approx(0.75)


def test_energy_from_rpe_angle_rejects_nonpositive_time() -> None:
    with pytest.raises(ValueError, match="base_time"):
        energy_from_rpe_angle(0.1, 0.0)


# --------------------------------------------------------------------------------------------
# qdrift_phase_to_energy
# --------------------------------------------------------------------------------------------
def _qdrift_forward_phase(energy: float, lambda_norm: float, evolution_time: float, num_samples: int) -> float:
    """Forward model: expected qDRIFT signal phase for an eigenstate of given energy."""
    a = lambda_norm * evolution_time / num_samples
    return -num_samples * np.arctan((energy / lambda_norm) * np.tan(a))


@pytest.mark.parametrize("energy", [0.5, -0.5, 0.123])
def test_qdrift_phase_to_energy_round_trips(energy: float) -> None:
    lambda_norm, tau, n = 1.0, 0.3, 8
    phi = _qdrift_forward_phase(energy, lambda_norm, tau, n)
    recovered = qdrift_phase_to_energy(phi, tau, lambda_norm, n)
    assert recovered == pytest.approx(energy, abs=1e-9)


def test_qdrift_correction_beats_linear_and_bias_shrinks() -> None:
    energy, lambda_norm, tau = 0.6, 1.0, 0.5
    errors_linear: list[float] = []
    for n in (2, 8, 32, 128):
        phi = _qdrift_forward_phase(energy, lambda_norm, tau, n)
        linear = -phi / tau
        corrected = qdrift_phase_to_energy(phi, tau, lambda_norm, n)
        # The tangent correction removes the bias exactly at every N.
        assert abs(corrected - energy) < abs(linear - energy)
        errors_linear.append(abs(linear - energy))
    # The naive linear estimate converges to the truth as samples grow.
    assert errors_linear == sorted(errors_linear, reverse=True)


# --------------------------------------------------------------------------------------------
# Full synthetic RPE recovery: the algorithm math returns the correct energy
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("energy", [0.75, 0.25, -0.6, 1.1])
def test_rpe_recovers_energy_from_ideal_signal(energy: float) -> None:
    """Feeding ideal phases arg(e^{-iE t_m}) must reconstruct E (exact-evolution limit)."""
    lambda_norm = 1.5  # >= |energy|
    base_time = np.pi / (2 * lambda_norm)  # ensures |energy * base_time| < pi/2
    total = num_rounds(lambda_norm, epsilon=1e-3)

    theta = 0.0
    for m in range(total + 1):
        t_m = (2**m) * base_time
        measured_phase = float(np.angle(np.exp(-1j * energy * t_m)))
        theta = rpe_angle_update(theta, measured_phase, m)

    recovered = energy_from_rpe_angle(theta, base_time)
    assert recovered == pytest.approx(energy, abs=1e-3)
