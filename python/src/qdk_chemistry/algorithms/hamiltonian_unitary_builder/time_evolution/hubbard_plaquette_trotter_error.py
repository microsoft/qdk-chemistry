"""Commutator bound on the Trotter error of a plaquette Fermi-Hubbard step, from Theorem 1 of :cite:`Schubert2023`."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import functools
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

from qdk_chemistry.utils.qsharp import QSHARP_UTILS

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__: list[str] = ["hubbard_plaquette_commutator_bound"]

_PINK, _INTERACTION, _GOLD = 0, 1, 2

_Monomial = tuple[int, ...]


def hubbard_plaquette_commutator_bound(
    width: int, height: int, order: int, hopping: float, interaction: float
) -> float:
    r"""Return :math:`W` with :math:`\lVert S_p(s) - e^{-isH} \rVert \le W s^{p+1}` for one step of ``RepPlaquetteExp``.

    Args:
        width: Number of lattice columns.
        height: Number of lattice rows.
        order: Trotter order of the step, a positive even integer.
        hopping: Hopping amplitude of every plaquette bond.
        interaction: On-site interaction :math:`U` of the particle-hole symmetric form.

    Returns:
        The error constant :math:`W`.

    """
    return sum(
        coefficient * abs(hopping) ** hopping_degree * abs(interaction) ** interaction_degree
        for hopping_degree, interaction_degree, coefficient in _bound_coefficients(width, height, order)
    )


@functools.cache
def _bound_coefficients(width: int, height: int, order: int) -> tuple[tuple[int, int, float], ...]:
    """Return the coefficient of each power of the hopping amplitude and interaction in the bound."""
    sites = width * height
    cell_columns, cell_rows = width // 2, height // 2
    # Each Majorana moved by each supercell translation, and the translation into the first supercell.
    moved: list[list[int]] = []
    for shift_row in range(cell_rows):
        for shift_column in range(cell_columns):
            translation = []
            for majorana in range(4 * sites):
                spin, site = divmod(majorana // 2, sites)
                row, column = divmod(site, width)
                site = (row + 2 * shift_row) % height * width + (column + 2 * shift_column) % width
                translation.append(2 * (spin * sites + site) + majorana % 2)
            moved.append(translation)
    home: list[int] = []
    for majorana in range(4 * sites):
        row, column = divmod((majorana // 2) % sites, width)
        home.append((-(row // 2)) % cell_rows * cell_columns + (-(column // 2)) % cell_columns)

    def canonical(monomial: _Monomial) -> tuple[int, _Monomial]:
        """Return the sign and representative of the translation class of ``monomial``."""
        return min(
            (_sort([moved[home[majorana]][other] for other in monomial]) for majorana in monomial),
            key=lambda candidate: candidate[1],
        )

    # The unit-amplitude terms of each fragment, and for each Majorana the terms holding it.
    terms: list[list[tuple[_Monomial, complex]]] = [[], [], []]
    for fragment, pink in ((_PINK, True), (_GOLD, False)):
        for cycle in QSHARP_UTILS.HubbardPlaquette.PlaquetteSection(width, height, pink):
            for corner in range(4):
                a, b = cycle[corner], cycle[(corner + 1) % 4]
                # -(c_a^+ c_b + c_b^+ c_a) = -(i/2) (g_2a g_2b+1 - g_2a+1 g_2b)
                for majoranas, value in (((2 * a, 2 * b + 1), -0.5j), ((2 * a + 1, 2 * b), 0.5j)):
                    sign, monomial = _sort(majoranas)
                    terms[fragment].append((monomial, sign * value))
    for up in range(sites):
        down = up + sites
        # (1/4) Z_up Z_down, with Z_m = -i g_2m g_2m+1
        terms[_INTERACTION].append(((2 * up, 2 * up + 1, 2 * down, 2 * down + 1), -0.25))
    containing: list[defaultdict[int, list[int]]] = []
    for fragment_terms in terms:
        holding: defaultdict[int, list[int]] = defaultdict(list)
        for index, (monomial, _) in enumerate(fragment_terms):
            for majorana in monomial:
                holding[majorana].append(index)
        containing.append(holding)

    @functools.cache
    def nested(letters: tuple[int, ...], fragment: int) -> dict[_Monomial, complex]:
        """Return the nested commutator of ``letters`` (outermost first) applied to the fragment ``fragment``."""
        if not letters:
            return {monomial: value for monomial, value in terms[fragment] if canonical(monomial)[1] == monomial}
        letter = letters[0]
        result: defaultdict[_Monomial, complex] = defaultdict(complex)
        for monomial, value in nested(letters[1:], fragment).items():
            support = set(monomial)
            for index in {index for majorana in monomial for index in containing[letter].get(majorana, ())}:
                term, term_value = terms[letter][index]
                # Even monomials commute unless they share an odd number of Majoranas, when [a, b] = 2ab.
                if len(support.intersection(term)) % 2:
                    swaps = sum(len(term) - bisect_right(term, majorana) for majorana in monomial)
                    sign, representative = canonical(tuple(sorted(support.symmetric_difference(term))))
                    result[representative] += 2 * (-1) ** swaps * sign * term_value * value
        return {monomial: value for monomial, value in result.items() if value}

    # The step's exponentials in order of application as (fragment, share of the step), merged as in Q#.
    fractions = QSHARP_UTILS.HubbardPlaquette.SuzukiStageFractions(order)
    sequence = [(_PINK, 0.5 * fractions[0])]
    for stage, fraction in enumerate(fractions):
        following = fractions[stage + 1] if stage + 1 < len(fractions) else 0.0
        sequence += [
            (_INTERACTION, 0.5 * fraction),
            (_GOLD, fraction),
            (_INTERACTION, 0.5 * fraction),
            (_PINK, 0.5 * (fraction + following)),
        ]

    # Theorem 1, split at the middle exponential. Each representative stands for ``cells`` translates of unitary
    # monomials, so ``cells`` times its absolute coefficient sum bounds a norm.
    cells = cell_columns * cell_rows
    split = math.ceil(len(sequence) / 2)
    coefficients: defaultdict[tuple[int, int], float] = defaultdict(float)
    for j in range(2, len(sequence) + 1):
        accumulated = [0.0, 0.0, 0.0]
        for fragment, share in sequence[: j - 1]:
            accumulated[fragment] += share
        positions = range(split, j - 1, -1) if j <= split else range(split + 1, j + 1)
        # The ad powers at ``positions`` sum to the order, with a positive power at j.
        for rest in combinations_with_replacement(positions, order - 1):
            chosen = (*rest, j)
            letters = tuple(sequence[position - 1][0] for position in chosen)
            prefactor = math.prod(abs(sequence[position - 1][1]) for position in chosen) / (
                (order + 1) * math.prod(map(math.factorial, Counter(chosen).values()))
            )
            # Bounding the hopping and interaction parts separately keeps W a polynomial in the two amplitudes.
            hopping_part: defaultdict[_Monomial, complex] = defaultdict(complex)
            for fragment in (_PINK, _GOLD):
                for monomial, value in nested(letters, fragment).items():
                    hopping_part[monomial] += accumulated[fragment] * value
            hopping_norm = sum(map(abs, hopping_part.values()))
            interaction_norm = abs(accumulated[_INTERACTION]) * sum(map(abs, nested(letters, _INTERACTION).values()))
            interactions = letters.count(_INTERACTION)
            coefficients[order + 1 - interactions, interactions] += cells * prefactor * hopping_norm
            coefficients[order - interactions, interactions + 1] += cells * prefactor * interaction_norm
    return tuple((degrees[0], degrees[1], value) for degrees, value in sorted(coefficients.items()) if value)


def _sort(majoranas: Sequence[int]) -> tuple[int, _Monomial]:
    """Return the sign of sorting the anticommuting ``majoranas`` and their sorted monomial."""
    inversions = sum(1 for index, first in enumerate(majoranas) for second in majoranas[index + 1 :] if first > second)
    return (-1) ** inversions, tuple(sorted(majoranas))
