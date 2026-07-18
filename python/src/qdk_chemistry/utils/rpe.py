r"""Classical post-processing for robust (randomized) phase estimation.

These are pure helper functions for the qDRIFT / partially-randomized phase
estimation path: turning Hadamard-test counts into a control-qubit expectation
value, deriving the per-round shot/sample schedule, performing the robust
angle-consistency update across rounds, and mapping a recovered phase to an
energy (including the qDRIFT tangent de-biasing).

All functions assume the time-evolution convention :math:`U(t) = e^{-iHt}`, so
that for an eigenstate of energy :math:`E` the Hadamard-test signal is
:math:`\langle\psi|U(t)|\psi\rangle = e^{-iEt}` and its argument is
:math:`-Et`.

References:
    Günther, J., Witteveen, F., et al. (2025). Phase estimation with partially
    randomized time evolution. PRX Quantum 7, 020332. arXiv:2503.05647.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.utils import Logger

__all__ = [
    "energy_from_rpe_angle",
    "expectation_from_counts",
    "num_rounds",
    "qdrift_phase_to_energy",
    "qdrift_schedule",
    "rpe_angle_update",
    "wrap_to_principal",
]


def wrap_to_principal(angle: float) -> float:
    """Wrap an angle into the principal interval ``(-pi, pi]``.

    Args:
        angle: Angle in radians.

    Returns:
        The equivalent angle in ``(-pi, pi]``.

    """
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def expectation_from_counts(counts: dict[str, int]) -> float:
    r"""Reduce control-qubit measurement counts to a :math:`\langle Z\rangle` expectation.

    For the Hadamard test the control qubit is measured in the X (real part) or
    Y (imaginary part) basis, yielding outcomes labelled ``"0"`` (eigenvalue
    ``+1``) and ``"1"`` (eigenvalue ``-1``). The estimator is
    ``(n0 - n1) / (n0 + n1)``.

    Args:
        counts: Mapping of bit label (``"0"`` / ``"1"``) to occurrence count.

    Returns:
        The estimated expectation value in ``[-1, 1]``. Returns ``0.0`` when no
        shots are present.

    """
    Logger.trace_entering()
    n0 = int(counts.get("0", 0))
    n1 = int(counts.get("1", 0))
    total = n0 + n1
    if total == 0:
        return 0.0
    return (n0 - n1) / total


def num_rounds(lambda_norm: float, epsilon: float) -> int:
    """Compute the number of doubling rounds for a target accuracy.

    The robust phase estimation ladder doubles the evolution time each round.
    Reaching accuracy ``epsilon`` on a spectrum bounded by the Hamiltonian
    1-norm ``lambda_norm`` requires ``M = ceil(log2(lambda_norm / epsilon))``
    additional rounds beyond the base round.

    Args:
        lambda_norm: 1-norm of the Hamiltonian coefficients, ``sum |h_j|``.
        epsilon: Target absolute accuracy on the energy.

    Returns:
        Non-negative round count ``M`` (the loop runs rounds ``0..M`` inclusive).

    Raises:
        ValueError: If ``epsilon`` is not positive or ``lambda_norm`` is negative.

    """
    Logger.trace_entering()
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, received {epsilon}.")
    if lambda_norm < 0.0:
        raise ValueError(f"lambda_norm must be non-negative, received {lambda_norm}.")
    if lambda_norm <= epsilon:
        return 0
    return int(np.ceil(np.log2(lambda_norm / epsilon)))


def qdrift_schedule(total_rounds: int, round_index: int) -> tuple[int, int]:
    """Return the (shots, qDRIFT samples) budget for a given round.

    Early rounds set the most-significant bits and are allocated more shots;
    later rounds need more qDRIFT samples because they probe longer evolution
    times where the randomized channel must approximate :math:`e^{-iHt}` more
    faithfully.

    Args:
        total_rounds: Total round count ``M`` from :func:`num_rounds`.
        round_index: Current round ``m`` in ``0..M``.

    Returns:
        Tuple ``(shots, samples)`` with ``shots = ceil(e * (11 + 4 * (M - m)))``
        and ``samples = 2 ** (2 * m + 1)``.

    """
    Logger.trace_entering()
    shots = int(np.ceil(np.e * (11 + 4 * (total_rounds - round_index))))
    samples = 2 ** (2 * round_index + 1)
    return shots, samples


def angular_distance(angle_a: float, angle_b: float) -> float:
    """Return the smallest unsigned distance between two angles on the circle.

    Args:
        angle_a: First angle in radians.
        angle_b: Second angle in radians.

    Returns:
        Distance in ``[0, pi]`` accounting for the ``2*pi`` wrap-around.

    """
    diff = (angle_a - angle_b + np.pi) % (2 * np.pi) - np.pi
    return abs(float(diff))


def rpe_angle_update(previous_angle: float, measured_phase: float, round_index: int) -> float:
    """Refine the phase estimate with one robust angle-consistency step.

    At round ``m`` the measured phase ``measured_phase`` only constrains the true
    per-base-time phase ``theta`` modulo ``2*pi / 2**m``. The candidate set is
    ``{ (measured_phase + 2*pi*k) / 2**m : k = 0 .. 2**m - 1 }`` and the
    candidate closest (in angular distance) to the previous round's estimate is
    selected (Günther et al., Algorithm 1).

    Args:
        previous_angle: Estimate ``theta`` from the previous round (``0.0`` to start).
        measured_phase: Phase ``arg(Z)`` measured this round, in ``(-pi, pi]``.
        round_index: Current round ``m`` (``>= 0``).

    Returns:
        The updated per-base-time phase estimate ``theta``.

    Raises:
        ValueError: If ``round_index`` is negative.

    """
    Logger.trace_entering()
    if round_index < 0:
        raise ValueError(f"round_index must be non-negative, received {round_index}.")
    scale = 2**round_index
    best_candidate = measured_phase / scale
    best_distance = angular_distance(best_candidate, previous_angle)
    for k in range(1, scale):
        candidate = (measured_phase + 2 * np.pi * k) / scale
        distance = angular_distance(candidate, previous_angle)
        if distance < best_distance:
            best_distance = distance
            best_candidate = candidate
    return float(best_candidate)


def energy_from_rpe_angle(angle: float, base_time: float) -> float:
    """Map a recovered per-base-time phase to an energy (exact evolution).

    Under ``U(t) = e^{-iHt}`` the per-base-time phase of an eigenstate is
    ``angle = -E * base_time``, so ``E = -angle / base_time``. The angle is first
    wrapped into ``(-pi, pi]`` so that the correct energy branch is selected (the
    robust update may return any ``2*pi`` representative of the true phase).

    Args:
        angle: Recovered per-base-time phase ``theta`` from :func:`rpe_angle_update`.
        base_time: Base evolution time ``tau`` (round-0 evolution time).

    Returns:
        The energy estimate.

    Raises:
        ValueError: If ``base_time`` is not positive.

    """
    Logger.trace_entering()
    if base_time <= 0.0:
        raise ValueError(f"base_time must be positive, received {base_time}.")
    return -wrap_to_principal(angle) / base_time


def qdrift_phase_to_energy(
    phase: float,
    evolution_time: float,
    lambda_norm: float,
    num_samples: int,
) -> float:
    """Invert the qDRIFT signal bias to recover an energy from a measured phase.

    A qDRIFT channel of ``num_samples`` steps over total time ``evolution_time``
    does not reproduce ``e^{-iHt}`` exactly: on an eigenstate the expected signal
    is ``(cos(a) - i (E/lambda) sin(a)) ** num_samples`` with
    ``a = lambda_norm * evolution_time / num_samples``. Its argument is
    ``phase = -num_samples * arctan((E/lambda) tan(a))``, which inverts to

    ``E = -lambda_norm * tan(phase / num_samples) / tan(a)``.

    As ``num_samples -> infinity`` (``a -> 0``) this reduces to the linear map
    ``E = -phase / evolution_time``.

    Args:
        phase: Unwrapped phase accumulated over ``evolution_time``.
        evolution_time: Total evolution time the phase was accumulated over.
        lambda_norm: Hamiltonian 1-norm ``sum |h_j|``.
        num_samples: Number of qDRIFT samples used for this evolution.

    Returns:
        The de-biased energy estimate.

    Raises:
        ValueError: If ``evolution_time`` is not positive or ``num_samples < 1``.

    """
    Logger.trace_entering()
    if evolution_time <= 0.0:
        raise ValueError(f"evolution_time must be positive, received {evolution_time}.")
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, received {num_samples}.")
    step_angle = lambda_norm * evolution_time / num_samples
    denominator = np.tan(step_angle)
    # Near a -> 0 the channel is effectively exact; fall back to the linear map
    # to avoid an ill-conditioned ratio.
    if lambda_norm == 0.0 or abs(denominator) < 1e-12:
        return -phase / evolution_time
    return float(-lambda_norm * np.tan(phase / num_samples) / denominator)
