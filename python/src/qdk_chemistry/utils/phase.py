"""Utility functions for manipulating phases produced by QPE."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Iterable, Sequence

import numpy as np

from qdk_chemistry.utils import Logger

__all__ = [
    "accumulated_phase_from_bits",
    "energy_alias_candidates",
    "energy_from_phase",
    "iterative_phase_feedback_update",
    "phase_fraction_from_feedback",
    "resolve_energy_aliases",
]


def energy_from_phase(phase_fraction: float, *, evolution_time: float) -> float:
    """Convert a measured phase fraction to energy using ``E = angle / t``.

    Args:
        phase_fraction: Fractional phase obtained from the phase register.
        evolution_time: Evolution time ``t`` used in ``U = exp(-i H t)``.

    Returns:
        Energy estimate corresponding to ``phase_fraction``.

    """
    Logger.trace_entering()
    angle = (phase_fraction % 1.0) * (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return float(angle / evolution_time)


def energy_alias_candidates(
    raw_energy: float,
    *,
    evolution_time: float,
    shift_range: Iterable[int] = range(-2, 3),
) -> list[float]:
    """Enumerate alias energies compatible with ``raw_energy``.

    Args:
        raw_energy: Energy derived from the measured phase.
        evolution_time: Evolution time ``t`` used by the unitary.
        shift_range: Integer shifts (in multiples of ``2π / t``) to explore.

            The default ``range(-2, 3)`` checks ``k = -2, -1, 0, 1, 2``—a pragmatic
            window that typically covers chemical-energy estimates because they
            rarely deviate by more than one or two ``2π / t`` periods from the
            raw measurement.

    Returns:
        Sorted list of alias energy values covering positive and negative branches.

    """
    Logger.trace_entering()
    period = 2 * np.pi / evolution_time
    # Materialize the range in case a generator is provided and guarantee the zero shift.
    shifts = tuple(shift_range)
    if 0 not in shifts:
        shifts = (*shifts, 0)

    candidate_set: set[float] = set()
    for shift in shifts:
        candidate_set.add(float(raw_energy + period * shift))
        candidate_set.add(float(-raw_energy + period * shift))

    return sorted(candidate_set)


def resolve_energy_aliases(
    raw_energy: float,
    *,
    evolution_time: float,
    reference_energy: float,
    shift_range: Iterable[int] = range(-2, 3),
) -> float:
    """Select the alias energy closest to a known reference value.

    Args:
        raw_energy (float): Energy derived from the measured phase.
        evolution_time (float): Evolution time ``t`` used by the unitary.
        reference_energy (float): External reference guiding alias selection.
        shift_range (Iterable[int]): Integer shifts (in multiples of ``2π / t``) to explore.

            Use a wider window when the true value may sit multiple
            periods away from the raw estimate.

    Returns:
        Alias energy closest to ``reference_energy``.

    """
    Logger.trace_entering()
    candidates = energy_alias_candidates(raw_energy, evolution_time=evolution_time, shift_range=shift_range)
    return min(candidates, key=lambda energy: abs(energy - reference_energy))


def iterative_phase_feedback_update(current_phase: float, measured_bit: int) -> float:
    """Update the feedback phase according to Kitaev's recursion :cite:`Kitaev1995`.

    Args:
        current_phase: Phase ``Φ(k+1)`` from the previous iteration.
        measured_bit: Measured classical bit ``j_k`` (0 or 1).

    Returns:
        Updated feedback phase ``Φ(k)`` for the next iteration.

    Raises:
        ValueError: If ``measured_bit`` is not 0 or 1.

    """
    Logger.trace_entering()
    if measured_bit not in (0, 1):
        raise ValueError(f"measured_bit must be 0 or 1, received {measured_bit}.")
    return current_phase / 2.0 + np.pi * measured_bit / 2.0


def phase_fraction_from_feedback(phase_feedback: float) -> float:
    """Convert the final feedback phase ``Φ(1)`` into the phase fraction ``φ``.

    Args:
        phase_feedback: Final feedback phase angle returned by IQPE.

    Returns:
        Phase fraction corresponding to ``phase_feedback``.

    """
    Logger.trace_entering()
    return float(phase_feedback / np.pi)


def accumulated_phase_from_bits(bits_msb_first: Sequence[int]) -> float:
    """Compute ``Φ(1)`` directly from a measured bit string (MSB to LSB).

    Args:
        bits_msb_first: Sequence of measured bits ordered from MSB to LSB.

    Returns:
        Accumulated phase angle ``Φ(1)``.

    """
    Logger.trace_entering()
    phase = 0.0
    for bit in reversed(bits_msb_first):
        phase = iterative_phase_feedback_update(phase, bit)
    return phase
