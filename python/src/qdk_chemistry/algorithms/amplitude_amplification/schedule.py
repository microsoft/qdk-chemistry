r"""Round scheduling for amplitude amplification.

Amplitude amplification rotates the prepared state
:math:`|\psi\rangle = \sin\vartheta\,|G\rangle + \cos\vartheta\,|B\rangle`
by :math:`2\vartheta` per round inside a two-dimensional invariant subspace, so
after ``k`` rounds the probability of landing in the good subspace is

.. math::

    p_k = \sin^2\!\big((2k+1)\vartheta\big),
    \qquad
    \vartheta = \arcsin\sqrt{a},

with ``a`` the initial overlap :math:`|\langle G|\psi\rangle|^2`.

The rotation is periodic, so the round count cannot simply be made large: past
the first maximum the acceptance probability falls again, and at
:math:`(2k+1)\vartheta = \pi` it vanishes entirely. When ``a`` is only known
approximately -- the usual situation for a guiding state fed into quantum phase
estimation -- the round count must therefore be chosen defensively.

The central and slightly counter-intuitive fact is that **overshoot is
controlled by an upper bound on the overlap, not a lower bound**. Underestimating
``a`` makes :math:`\vartheta` too small, which makes the round count too large,
which overshoots. :func:`safe_rounds` takes the upper bound and guarantees
:math:`(2k+1)\vartheta \le \pi/2` for every admissible overlap, so the
acceptance probability stays on the monotone flank of the sine.

Three policies are provided, in increasing order of robustness and cost:

``optimal_rounds``
    Exact first maximum for a known overlap. Use only when the overlap is
    genuinely known, for example after amplitude estimation.
``safe_rounds`` / ``robust_rounds``
    Interval-valued overlap. ``safe_rounds`` never overshoots;
    ``robust_rounds`` maximizes the guaranteed acceptance probability over the
    interval and is never worse.
``fixed_point_phases``
    Only a lower bound on the overlap is known. The Yoder-Low-Chuang phase
    sequence replaces the sine with a Chebyshev plateau -- overshoot becomes
    impossible -- at a constant-factor cost in queries.

References:
    Lin, L. *Lecture Notes on Quantum Algorithms for Scientific Computation*,
    arXiv:2201.08309, Chapter 2.

    Brassard, G., Hoyer, P., Mosca, M., and Tapp, A. *Quantum Amplitude
    Amplification and Estimation*, arXiv:quant-ph/0005055.

    Yoder, T. J., Low, G. H., and Chuang, I. L. *Fixed-point quantum search with
    an optimal number of queries*, Phys. Rev. Lett. 113, 210501 (2014),
    arXiv:1409.3305.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math

__all__: list[str] = [
    "DEFAULT_GROWTH_RATE",
    "expected_rounds_exponential",
    "exponential_schedule",
    "fixed_point_phases",
    "fixed_point_rounds",
    "fixed_point_success_probability",
    "optimal_rounds",
    "overshoot_overlap",
    "robust_rounds",
    "rotation_angle",
    "safe_rounds",
    "success_probability",
    "success_probability_with_assumed_overlap",
    "worst_case_success_probability",
]

#: Growth rate of the Brassard-Hoyer-Mosca-Tapp exponential search schedule.
DEFAULT_GROWTH_RATE = 6.0 / 5.0


def _validate_overlap(overlap: float, name: str = "overlap") -> None:
    """Raise if ``overlap`` is not a usable squared overlap.

    Args:
        overlap: The value to check.
        name: Name used in the error message.

    Raises:
        ValueError: If ``overlap`` is not finite or does not lie in ``(0, 1]``.

    """
    if not math.isfinite(overlap) or not 0.0 < overlap <= 1.0:
        raise ValueError(f"{name} must be finite and lie in (0, 1]. Got {overlap}.")


def _validate_interval(min_overlap: float, max_overlap: float) -> None:
    """Raise if ``[min_overlap, max_overlap]`` is not a usable overlap interval.

    Args:
        min_overlap: Lower bound on the squared overlap.
        max_overlap: Upper bound on the squared overlap.

    Raises:
        ValueError: If either bound is invalid or the interval is empty.

    """
    _validate_overlap(min_overlap, "min_overlap")
    _validate_overlap(max_overlap, "max_overlap")
    if min_overlap > max_overlap:
        raise ValueError(f"min_overlap must not exceed max_overlap. Got {min_overlap} > {max_overlap}.")


def _validate_rounds(rounds: int) -> None:
    """Raise if ``rounds`` is not a valid round count.

    Args:
        rounds: The number of amplification rounds.

    Raises:
        ValueError: If ``rounds`` is negative.

    """
    if rounds < 0:
        raise ValueError(f"rounds must be nonnegative. Got {rounds}.")


def rotation_angle(overlap: float) -> float:
    r"""Return the half-rotation angle :math:`\vartheta = \arcsin\sqrt{a}`.

    Args:
        overlap: The squared overlap ``a`` of the prepared state with the good
            subspace, in ``(0, 1]``.

    Returns:
        The angle in radians, in ``(0, pi/2]``.

    """
    _validate_overlap(overlap)
    return math.asin(math.sqrt(overlap))


def success_probability(overlap: float, rounds: int) -> float:
    r"""Return the acceptance probability :math:`\sin^2((2k+1)\vartheta)`.

    Args:
        overlap: The squared overlap of the prepared state with the good subspace.
        rounds: The number of amplification rounds ``k``.

    Returns:
        The probability of measuring the good subspace after ``rounds`` rounds.

    """
    _validate_rounds(rounds)
    angle = rotation_angle(overlap)
    return math.sin((2 * rounds + 1) * angle) ** 2


def optimal_rounds(overlap: float) -> int:
    r"""Return the round count closest to the first maximum, for a *known* overlap.

    This is :math:`\mathrm{round}\big(\pi/(4\vartheta) - 1/2\big)`, the integer
    nearest the exact optimum :math:`(2k+1)\vartheta = \pi/2`.

    Using this with a merely *estimated* overlap is the classic way to overshoot:
    if the true overlap is larger than assumed the rotation runs past the maximum.
    Prefer :func:`safe_rounds` or :func:`robust_rounds` when the overlap is
    uncertain.

    Args:
        overlap: The squared overlap of the prepared state with the good subspace.

    Returns:
        The nonnegative round count nearest the first acceptance maximum.

    """
    angle = rotation_angle(overlap)
    return max(0, round(math.pi / (4.0 * angle) - 0.5))


def safe_rounds(max_overlap: float) -> int:
    r"""Return the largest round count that cannot overshoot.

    Returns :math:`\lfloor \pi/(4\vartheta_{\max}) - 1/2 \rfloor`, where
    :math:`\vartheta_{\max} = \arcsin\sqrt{a_{\max}}`. For this ``k`` every
    overlap ``a <= max_overlap`` satisfies :math:`(2k+1)\vartheta \le \pi/2`, so
    the acceptance probability is a monotonically increasing function of the true
    overlap: being luckier than expected can only help.

    Only the *upper* bound matters here. A lower bound tells you how well the
    schedule will do, not whether it is safe.

    Args:
        max_overlap: Upper bound on the squared overlap.

    Returns:
        The largest nonnegative round count with no overshoot risk.

    """
    _validate_overlap(max_overlap, "max_overlap")
    angle = rotation_angle(max_overlap)
    # The floor is taken with a small tolerance so that an overlap sitting exactly
    # on a boundary (a_max = 1/4, say) is not pushed down a round by rounding.
    return max(0, math.floor(math.pi / (4.0 * angle) - 0.5 + 1e-12))


def overshoot_overlap(rounds: int) -> float:
    r"""Return the smallest overlap at which ``rounds`` rounds accept with probability zero.

    The acceptance probability first returns to zero when
    :math:`(2k+1)\vartheta = \pi`, that is at
    :math:`a = \sin^2\!\big(\pi/(2k+1)\big)`. A true overlap at or above this
    value is the worst case for the given round count.

    Args:
        rounds: The number of amplification rounds ``k``.

    Returns:
        The overlap at which the schedule fails completely, or ``1.0`` when no
        overlap in ``(0, 1]`` can overshoot that far.

    """
    _validate_rounds(rounds)
    if rounds == 0:
        return 1.0
    return math.sin(math.pi / (2 * rounds + 1)) ** 2


def success_probability_with_assumed_overlap(assumed_overlap: float, actual_overlap: float) -> float:
    """Return the acceptance probability when the round count is chosen from the wrong overlap.

    Quantifies the asymmetry between under- and over-estimating the overlap:
    overestimating costs a mild loss of amplification, while underestimating can
    drive the acceptance probability to zero.

    Args:
        assumed_overlap: The overlap used to pick the round count.
        actual_overlap: The true overlap of the prepared state.

    Returns:
        The acceptance probability actually obtained.

    """
    return success_probability(actual_overlap, optimal_rounds(assumed_overlap))


def worst_case_success_probability(rounds: int, min_overlap: float, max_overlap: float) -> float:
    r"""Return the guaranteed acceptance probability over an overlap interval.

    The acceptance probability :math:`\sin^2((2k+1)\vartheta)` has interior
    minima only where :math:`(2k+1)\vartheta` is an integer multiple of
    :math:`\pi`. The worst case over the interval is therefore exactly zero when
    the rotation sweeps through such a multiple, and otherwise attained at one of
    the two endpoints. No search is needed.

    Args:
        rounds: The number of amplification rounds ``k``.
        min_overlap: Lower bound on the squared overlap.
        max_overlap: Upper bound on the squared overlap.

    Returns:
        The smallest acceptance probability consistent with the interval.

    """
    _validate_rounds(rounds)
    _validate_interval(min_overlap, max_overlap)

    factor = 2 * rounds + 1
    low = factor * rotation_angle(min_overlap)
    high = factor * rotation_angle(max_overlap)

    # A multiple of pi inside the swept arc drives acceptance to zero.
    if math.floor(high / math.pi) > math.floor(low / math.pi) or low % math.pi == 0.0:
        return 0.0
    return min(math.sin(low) ** 2, math.sin(high) ** 2)


def robust_rounds(min_overlap: float, max_overlap: float) -> int:
    """Return the round count maximizing the guaranteed acceptance probability.

    This is the minimax choice over the overlap interval. It is never worse than
    :func:`safe_rounds` and is usually equal to it; it can be larger when the
    interval is narrow enough that a mild, bounded overshoot at the top of the
    interval buys more at the bottom than it costs at the top.

    Ties are broken toward the smallest round count, since rounds are queries.

    Args:
        min_overlap: Lower bound on the squared overlap.
        max_overlap: Upper bound on the squared overlap.

    Returns:
        The nonnegative round count with the best worst-case acceptance
        probability.

    """
    _validate_interval(min_overlap, max_overlap)

    # Beyond the optimum for the smallest admissible overlap, every candidate has
    # already swept past the first maximum for the whole interval.
    upper = optimal_rounds(min_overlap)
    best_rounds = 0
    best_probability = worst_case_success_probability(0, min_overlap, max_overlap)
    for candidate in range(1, upper + 1):
        probability = worst_case_success_probability(candidate, min_overlap, max_overlap)
        if probability > best_probability:
            best_rounds = candidate
            best_probability = probability
    return best_rounds


def exponential_schedule(stages: int, growth: float = DEFAULT_GROWTH_RATE) -> list[int]:
    r"""Return the Brassard-Hoyer-Mosca-Tapp exponential search schedule.

    When nothing at all is known about the overlap, the standard strategy is to
    draw the round count uniformly from ``{0, ..., m-1}`` with an exponentially
    growing ``m = ceil(growth ** stage)``, retrying on failure. Randomizing the
    round count averages the acceptance probability over the rotation and so
    removes the overshoot cliff, and the expected total number of rounds is
    :math:`O(1/\sqrt{a})` -- the same quadratic speedup as the exact schedule, up
    to a constant.

    Args:
        stages: The number of stages to generate.
        growth: The growth rate ``c``. Any ``1 < c < 4/3`` preserves the
            expected-cost bound; the reference value is ``6/5``.

    Returns:
        The per-stage sampling bounds ``m``.

    Raises:
        ValueError: If ``stages`` is negative or ``growth`` is not in ``(1, 4/3)``.

    """
    if stages < 0:
        raise ValueError(f"stages must be nonnegative. Got {stages}.")
    if not math.isfinite(growth) or not 1.0 < growth < 4.0 / 3.0:
        raise ValueError(f"growth must lie in (1, 4/3). Got {growth}.")
    return [math.ceil(growth**stage) for stage in range(stages)]


def _mean_success_probability(overlap: float, sampling_bound: int) -> float:
    r"""Return the acceptance probability averaged over ``k`` uniform in ``[0, m)``.

    Uses the closed form
    :math:`\tfrac12 - \sin(4m\vartheta)/(4m\sin 2\vartheta)`, falling back to the
    direct average when :math:`\sin 2\vartheta` underflows.

    Args:
        overlap: The squared overlap of the prepared state with the good subspace.
        sampling_bound: The exclusive upper bound ``m`` on the sampled round count.

    Returns:
        The mean acceptance probability of one stage.

    """
    angle = rotation_angle(overlap)
    denominator = 4.0 * sampling_bound * math.sin(2.0 * angle)
    if abs(denominator) < 1e-15:
        return sum(success_probability(overlap, k) for k in range(sampling_bound)) / sampling_bound
    return 0.5 - math.sin(4.0 * sampling_bound * angle) / denominator


def expected_rounds_exponential(
    overlap: float,
    stages: int = 64,
    growth: float = DEFAULT_GROWTH_RATE,
) -> float:
    """Return the expected number of amplification rounds used by exponential search.

    Computed exactly from the per-stage success probabilities, truncated after
    ``stages`` stages. Use it to compare the price of not knowing the overlap
    against the exact schedule, whose cost is :func:`optimal_rounds`.

    Args:
        overlap: The true squared overlap of the prepared state.
        stages: The number of stages to include before truncating.
        growth: The growth rate of the schedule.

    Returns:
        The expected total number of Grover iterates.

    """
    _validate_overlap(overlap)
    total = 0.0
    survival = 1.0
    for sampling_bound in exponential_schedule(stages, growth):
        total += survival * (sampling_bound - 1) / 2.0
        survival *= 1.0 - _mean_success_probability(overlap, sampling_bound)
    return total


def fixed_point_rounds(min_overlap: float, tolerance: float) -> int:
    r"""Return the iterate count for fixed-point amplification.

    The Yoder-Low-Chuang schedule reaches acceptance probability at least
    :math:`1 - \delta^2` for *every* overlap at or above ``min_overlap`` once the
    number of queries satisfies
    :math:`L \ge \log(2/\delta)/\sqrt{a_{\min}}`. This returns the smallest
    ``l`` with ``L = 2l + 1`` meeting that bound.

    Args:
        min_overlap: Lower bound on the squared overlap.
        tolerance: The failure amplitude ``delta`` in ``(0, 1)``; the acceptance
            probability is at least ``1 - delta ** 2``.

    Returns:
        The number of iterates ``l``, so that ``2 * l + 1`` queries are used.

    Raises:
        ValueError: If ``tolerance`` is not in ``(0, 1)``.

    """
    _validate_overlap(min_overlap, "min_overlap")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must lie in (0, 1). Got {tolerance}.")
    queries = math.log(2.0 / tolerance) / math.sqrt(min_overlap)
    return max(1, math.ceil((math.ceil(queries) - 1) / 2.0))


def fixed_point_phases(rounds: int, tolerance: float) -> tuple[list[float], list[float]]:
    r"""Return the Yoder-Low-Chuang phase sequence for fixed-point amplification.

    With ``L = 2 * rounds + 1`` queries and

    .. math::

        \gamma^{-1} = T_{1/L}(1/\delta)
                    = \cosh\!\big(L^{-1}\operatorname{arccosh}(1/\delta)\big),

    the state-reflection phases are

    .. math::

        \beta_j = 2\operatorname{arccot}\!\big(\tan(2\pi j/L)\sqrt{1-\gamma^2}\big),
        \qquad j = 1,\dots,l ,

    and the mark phases are the same list reversed,
    :math:`\alpha_j = \beta_{l+1-j}`. Both reflections are taken in the
    ``I - (1 - e^{i\varphi}) P`` convention used by the Q# implementation, which
    is why no sign flip appears between the two sequences.

    Feeding these to
    ``QDKChemistry.Utils.AmplitudeAmplification.ApplyFixedPointAmplitudeAmplification``
    reproduces :func:`fixed_point_success_probability` exactly: the acceptance
    probability climbs monotonically up to
    :math:`a = 1 - T_{1/L}(1/\delta)^{-2}` and from there stays inside
    :math:`[1-\delta^2, 1]` for every larger overlap. There is no first maximum
    to run past, so overshoot is impossible above the design threshold.

    Args:
        rounds: The number of iterates ``l``; ``2 * l + 1`` queries are used.
        tolerance: The failure amplitude ``delta`` in ``(0, 1)``.

    Returns:
        The mark phases ``alpha`` and the state phases ``beta``, both of length
        ``rounds`` and ordered by iterate.

    Raises:
        ValueError: If ``rounds`` is not positive or ``tolerance`` is not in
            ``(0, 1)``.

    """
    if rounds < 1:
        raise ValueError(f"rounds must be positive. Got {rounds}.")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must lie in (0, 1). Got {tolerance}.")

    queries = 2 * rounds + 1
    gamma = 1.0 / math.cosh(math.acosh(1.0 / tolerance) / queries)
    scale = math.sqrt(max(0.0, 1.0 - gamma * gamma))

    # arccot with range (0, pi), so that the phases are continuous in j.
    state_phases = [2.0 * math.atan2(1.0, math.tan(2.0 * math.pi * j / queries) * scale) for j in range(1, rounds + 1)]
    mark_phases = list(reversed(state_phases))
    return mark_phases, state_phases


def _chebyshev(degree: float, argument: float) -> float:
    """Return the Chebyshev polynomial of the first kind, valid outside ``[-1, 1]``.

    Args:
        degree: The (possibly fractional) degree.
        argument: The evaluation point.

    Returns:
        ``T_degree(argument)``.

    """
    if argument >= 1.0:
        return math.cosh(degree * math.acosh(argument))
    if argument <= -1.0:
        magnitude = math.cosh(degree * math.acosh(-argument))
        return magnitude if int(degree) % 2 == 0 else -magnitude
    return math.cos(degree * math.acos(argument))


def fixed_point_success_probability(overlap: float, rounds: int, tolerance: float) -> float:
    r"""Return the acceptance probability of the fixed-point schedule.

    .. math::

        p = 1 - \delta^2\,
            T_L\!\big(T_{1/L}(1/\delta)\sqrt{1-a}\big)^2 ,
        \qquad L = 2l+1 .

    The right-hand side increases monotonically up to
    :math:`a = 1 - T_{1/L}(1/\delta)^{-2}` and then stays within
    :math:`[1-\delta^2, 1]`, rippling but never collapsing. That is precisely the
    property the plain schedule lacks: there is no first maximum to run past.

    Args:
        overlap: The squared overlap of the prepared state with the good subspace.
        rounds: The number of iterates ``l`` used by :func:`fixed_point_phases`.
        tolerance: The failure amplitude ``delta`` in ``(0, 1)``.

    Returns:
        The acceptance probability after ``2 * rounds + 1`` queries.

    Raises:
        ValueError: If ``rounds`` is negative or ``tolerance`` is not in ``(0, 1)``.

    """
    _validate_overlap(overlap)
    _validate_rounds(rounds)
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError(f"tolerance must lie in (0, 1). Got {tolerance}.")

    queries = 2 * rounds + 1
    scale = math.cosh(math.acosh(1.0 / tolerance) / queries)
    return 1.0 - tolerance**2 * _chebyshev(queries, scale * math.sqrt(1.0 - overlap)) ** 2
