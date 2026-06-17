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
    qpe_evolution_time_from_hamiltonian,
    resolve_energy_aliases,
)

from .reference_tolerances import (
    float_comparison_relative_tolerance,
    qpe_energy_tolerance,
    qpe_phase_fraction_tolerance,
)


class _HamiltonianWithNorm:
    def __init__(self, norm: float) -> None:
        self.schatten_norm = norm


def test_qpe_evolution_time_from_hamiltonian_uses_norm_bound() -> None:
    """Evolution time should be derived from the Hamiltonian norm."""
    hamiltonian = _HamiltonianWithNorm(4.0)

    assert qpe_evolution_time_from_hamiltonian(hamiltonian) == pytest.approx(np.pi / 4.0)
    assert qpe_evolution_time_from_hamiltonian(hamiltonian, phase_bound=np.pi / 2.0) == pytest.approx(
        np.pi / 8.0
    )


@pytest.mark.parametrize("norm", [0.0, -1.0, np.inf, np.nan])
def test_qpe_evolution_time_from_hamiltonian_rejects_invalid_norm(norm: float) -> None:
    """Hamiltonian norm must be positive and finite."""
    with pytest.raises(ValueError, match="Hamiltonian norm"):
        qpe_evolution_time_from_hamiltonian(_HamiltonianWithNorm(norm))


@pytest.mark.parametrize("phase_bound", [0.0, -1.0, np.pi + 1e-12, np.inf, np.nan])
def test_qpe_evolution_time_from_hamiltonian_rejects_invalid_phase_bound(phase_bound: float) -> None:
    """The phase bound must keep the selected time inside the principal branch."""
    with pytest.raises(ValueError, match="phase_bound"):
        qpe_evolution_time_from_hamiltonian(_HamiltonianWithNorm(1.0), phase_bound=phase_bound)


def test_energy_from_phase_wraps_into_branch() -> None:
    """Energy calculation should unwrap angles greater than π."""
    energy = energy_from_phase(0.75, evolution_time=0.5)
    expected_angle = -0.5 * np.pi  # 1.5π wraps to -0.5π
    expected_energy = expected_angle / 0.5
    assert np.allclose(energy, expected_energy, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance)


def test_energy_alias_candidates_default_window() -> None:
    """Default shift range should produce the expected alias values."""
    candidates = energy_alias_candidates(raw_energy=1.0, evolution_time=0.5)
    period = 2.0 * np.pi / 0.5
    expected = sorted({1.0 + period * k for k in range(-2, 3)} | {-1.0 + period * k for k in range(-2, 3)})
    assert np.allclose(candidates, expected, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance)


def test_resolve_energy_aliases_selects_closest_branch() -> None:
    """Alias resolution must pick the candidate nearest to the reference."""
    raw_energy = 1.0
    evolution_time = 0.5
    period = 2 * np.pi / evolution_time
    reference = raw_energy + 1.2 * period
    resolved = resolve_energy_aliases(raw_energy, evolution_time=evolution_time, reference_energy=reference)
    assert np.allclose(
        resolved, raw_energy + period, rtol=float_comparison_relative_tolerance, atol=qpe_energy_tolerance
    )


def test_iterative_phase_feedback_update_rejects_invalid_bit() -> None:
    """Updating the phase with an invalid measurement should fail."""
    with pytest.raises(ValueError, match="must be 0 or 1"):
        iterative_phase_feedback_update(0.0, measured_bit=2)


def test_phase_fraction_from_feedback_matches_manual_integral() -> None:
    """Final feedback phase converts back into the expected fraction."""
    feedback_phase = math.pi / 3.0
    fraction = phase_fraction_from_feedback(feedback_phase)
    assert np.isclose(fraction, 1.0 / 3.0, rtol=float_comparison_relative_tolerance, atol=qpe_phase_fraction_tolerance)


def test_accumulated_phase_from_bits_matches_recursive_update() -> None:
    """Accumulated phase helper should align with iterative update logic."""
    bits = [1, 0, 1]
    phase = accumulated_phase_from_bits(bits)

    recursive_phase = 0.0
    for bit in reversed(bits):
        recursive_phase = iterative_phase_feedback_update(recursive_phase, bit)

    assert np.allclose(
        phase, recursive_phase, rtol=float_comparison_relative_tolerance, atol=qpe_phase_fraction_tolerance
    )
