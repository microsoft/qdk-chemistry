"""Unit tests for phase utility helpers."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math

import numpy as np
import pytest

from qdk_chemistry.utils.phase import (
    accumulated_phase_from_bits,
    energy_alias_candidates,
    energy_from_phase,
    iterative_phase_feedback_update,
    phase_fraction_from_feedback,
    resolve_energy_aliases,
)


def test_energy_from_phase_wraps_into_branch() -> None:
    """Energy calculation should unwrap angles greater than π."""
    energy = energy_from_phase(0.75, evolution_time=0.5)
    expected_angle = -0.5 * np.pi  # 1.5π wraps to -0.5π
    expected_energy = expected_angle / 0.5
    assert energy == pytest.approx(expected_energy)


def test_energy_alias_candidates_default_window() -> None:
    """Default shift range should produce the expected alias values."""
    candidates = energy_alias_candidates(raw_energy=1.0, evolution_time=0.5)
    period = 2 * np.pi / 0.5
    expected = sorted({1.0 + period * k for k in range(-2, 3)} | {-1.0 + period * k for k in range(-2, 3)})
    assert candidates == pytest.approx(expected)


def test_resolve_energy_aliases_selects_closest_branch() -> None:
    """Alias resolution must pick the candidate nearest to the reference."""
    raw_energy = 1.0
    evolution_time = 0.5
    period = 2 * np.pi / evolution_time
    reference = raw_energy + 1.2 * period
    resolved = resolve_energy_aliases(raw_energy, evolution_time=evolution_time, reference_energy=reference)
    assert resolved == pytest.approx(raw_energy + period)


def test_iterative_phase_feedback_update_rejects_invalid_bit() -> None:
    """Updating the phase with an invalid measurement should fail."""
    with pytest.raises(ValueError, match="must be 0 or 1"):
        iterative_phase_feedback_update(0.0, measured_bit=2)


def test_phase_fraction_from_feedback_matches_manual_integral() -> None:
    """Final feedback phase converts back into the expected fraction."""
    feedback_phase = math.pi / 3
    fraction = phase_fraction_from_feedback(feedback_phase)
    assert fraction == pytest.approx(1 / 3)


def test_accumulated_phase_from_bits_matches_recursive_update() -> None:
    """Accumulated phase helper should align with iterative update logic."""
    bits = [1, 0, 1]
    phase = accumulated_phase_from_bits(bits)

    recursive_phase = 0.0
    for bit in reversed(bits):
        recursive_phase = iterative_phase_feedback_update(recursive_phase, bit)

    assert phase == pytest.approx(recursive_phase)
