r"""Graded Pauli-polynomial algebra and the Zassenhaus factor construction.

This module implements the order-by-order Zassenhaus expansion used by the
:class:`~qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus.Zassenhaus`
builder.  It is kept separate from the builder so the (purely algebraic) factor
construction can be unit-tested without constructing
:class:`~qdk_chemistry.data.UnitaryRepresentation` objects.

The key observation is that for a Pauli Hamiltonian every nested commutator of the
summands is again a scalar multiple of a *single* Pauli string, so the whole
construction stays inside the Pauli algebra -- no matrix logarithms are required.

Operators are represented as **graded Pauli polynomials**: a mapping

    ``word -> {degree: complex_coefficient}``

where ``word`` is a canonical sparse Pauli word (a sorted tuple of
``(qubit_index, pauli_axis)`` pairs, ``pauli_axis`` in ``{1: X, 2: Y, 3: Z}``) and
``degree`` tracks the power of the evolution time :math:`t`.  Pauli-string products
(with their phases) are delegated to
:meth:`~qdk_chemistry.data.PauliTermAccumulator.multiply_uncached`, the same tested
routine used by :func:`~qdk_chemistry.utils.pauli_commutation.commutator`.

References:
    R. M. Wilcox, *Exponential operators and parameter differentiation in quantum
    physics*, J. Math. Phys. **8**, 962 (1967).

    F. Casas, A. Murua, and M. Nadinic, *Efficient computation of the Zassenhaus
    formula*, Comput. Phys. Commun. **183**, 2386 (2012). arXiv:1204.0389.

    A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, and S. Zhu, *Theory of Trotter error
    with commutator scaling*, Phys. Rev. X **11**, 011020 (2021). arXiv:1912.08854.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

from qdk_chemistry.data import PauliTermAccumulator

__all__: list[str] = ["ZassenhausFactor", "zassenhaus_factors"]

# A canonical sparse Pauli word: sorted tuple of (qubit_index, pauli_axis),
# pauli_axis in {1: X, 2: Y, 3: Z}.  The empty tuple is the identity.
PauliWord = tuple[tuple[int, int], ...]

# A graded Pauli polynomial: word -> {degree: coefficient}.
GradedPoly = dict[PauliWord, dict[int, complex]]

_AXIS_TO_CHAR = {1: "X", 2: "Y", 3: "Z"}


@dataclass(frozen=True)
class ZassenhausFactor:
    r"""A single exponential factor :math:`\exp(-i\,\theta\,t^{d}\,P)` of the Zassenhaus product.

    The rotation angle for a concrete per-step time :math:`t` is
    ``angle_coefficient * t**degree``; keeping ``degree`` explicit lets the builder
    evaluate the same symbolic factor list at any time step.
    """

    pauli_term: dict[int, str]
    """Mapping ``qubit_index -> {'X', 'Y', 'Z'}`` describing the Pauli string :math:`P`."""

    angle_coefficient: float
    """Real prefactor of the rotation angle; the angle is ``angle_coefficient * t**degree``."""

    degree: int
    """Power of the evolution time :math:`t` carried by this factor."""


# ----------------------------------------------------------------------------------
# Graded Pauli-polynomial algebra
# ----------------------------------------------------------------------------------


def _word_to_term_map(word: PauliWord) -> dict[int, str]:
    """Convert a canonical sparse Pauli word to a ``{qubit: axis_char}`` mapping."""
    return {qubit: _AXIS_TO_CHAR[axis] for qubit, axis in word}


def _canonical_word(word: list[tuple[int, int]]) -> PauliWord:
    """Return a sorted tuple form of a sparse Pauli word, dropping identity entries."""
    return tuple(sorted((q, a) for q, a in word if a != 0))


def _poly_add_inplace(target: GradedPoly, other: GradedPoly, scale: complex = 1.0) -> GradedPoly:
    """Accumulate ``scale * other`` into ``target`` (modified in place) and return it."""
    for word, degree_map in other.items():
        bucket = target.setdefault(word, {})
        for degree, coeff in degree_map.items():
            bucket[degree] = bucket.get(degree, 0j) + scale * coeff
    return target


def _poly_clean(poly: GradedPoly, tol: float = 1e-14) -> GradedPoly:
    """Drop coefficients (and emptied words) whose magnitude is below ``tol``."""
    cleaned: GradedPoly = {}
    for word, degree_map in poly.items():
        kept = {degree: coeff for degree, coeff in degree_map.items() if abs(coeff) > tol}
        if kept:
            cleaned[word] = kept
    return cleaned


def _poly_multiply(left: GradedPoly, right: GradedPoly, max_degree: int) -> GradedPoly:
    """Multiply two graded Pauli polynomials, truncating contributions above ``max_degree``."""
    out: GradedPoly = {}
    for word_a, deg_a in left.items():
        list_a = list(word_a)
        for word_b, deg_b in right.items():
            phase, product = PauliTermAccumulator.multiply_uncached(list_a, list(word_b))
            result_word = _canonical_word(product)
            bucket = out.setdefault(result_word, {})
            for da, ca in deg_a.items():
                for db, cb in deg_b.items():
                    degree = da + db
                    if degree > max_degree:
                        continue
                    bucket[degree] = bucket.get(degree, 0j) + phase * ca * cb
    return out


def _poly_identity() -> GradedPoly:
    """Return the identity element of the graded Pauli algebra."""
    return {(): {0: 1 + 0j}}


def _poly_exp(generator: GradedPoly, max_degree: int) -> GradedPoly:
    r"""Return the truncated series :math:`\exp(generator)`.

    ``generator`` must have minimum degree :math:`\ge 1` so the Taylor series
    truncates: the :math:`k`-th power has minimum degree :math:`k`.
    """
    result = _poly_identity()
    term = _poly_identity()
    for k in range(1, max_degree + 1):
        term = _poly_multiply(term, generator, max_degree)
        if not term:
            break
        term = {word: {degree: coeff / k for degree, coeff in degree_map.items()} for word, degree_map in term.items()}
        _poly_add_inplace(result, term)
    return _poly_clean(result)


def _poly_log(unitary: GradedPoly, max_degree: int) -> GradedPoly:
    r"""Return the truncated series :math:`\log(unitary)` for ``unitary = I + M``.

    Uses :math:`\log(I + M) = \sum_{k \ge 1} (-1)^{k+1} M^{k}/k`, where
    :math:`M` has minimum degree :math:`\ge 1`.
    """
    residual = _poly_add_inplace({word: dict(dm) for word, dm in unitary.items()}, {(): {0: -1 + 0j}})
    residual = _poly_clean(residual)

    result: GradedPoly = {}
    power = _poly_identity()
    for k in range(1, max_degree + 1):
        power = _poly_multiply(power, residual, max_degree)
        if not power:
            break
        sign = 1.0 if k % 2 == 1 else -1.0
        scaled = {
            word: {degree: sign * coeff / k for degree, coeff in degree_map.items()}
            for word, degree_map in power.items()
        }
        _poly_add_inplace(result, scaled)
    return _poly_clean(result)


# ----------------------------------------------------------------------------------
# Zassenhaus factor construction
# ----------------------------------------------------------------------------------


def zassenhaus_factors(
    terms: list[tuple[str, float]],
    order: int,
    *,
    tol: float = 1e-14,
) -> list[ZassenhausFactor]:
    r"""Construct the Zassenhaus product factorisation of :math:`\exp(-iHt)`.

    Given the real Pauli expansion :math:`H = \sum_k \alpha_k P_k` (passed as
    ``terms`` of ``(pauli_label, alpha_k)`` pairs), this returns an ordered list of
    single-Pauli exponential factors whose product approximates :math:`\exp(-iHt)`
    with operator-norm error :math:`O(t^{\,\text{order}+1})`.

    The factors are built order by order.  Starting from the bare first-order
    Lie-Trotter product :math:`\prod_k \exp(-i\alpha_k t P_k)`, the routine repeatedly

    1. forms the truncated unitary series :math:`U = \prod_m \exp(f_m)`,
    2. computes its logarithm :math:`L = \log U` as a graded Pauli polynomial,
    3. extracts the degree-:math:`n` residual :math:`R_n` of :math:`L - (-iHt)`, and
    4. appends single-Pauli factors realising :math:`-R_n`.

    Appending degree-:math:`n` factors cancels the order-:math:`n` term of
    :math:`\log U` exactly (their mutual cross terms are of degree :math:`\ge n+1`),
    and recomputing :math:`L` at every step folds all splitting errors into later
    corrections.  After processing :math:`n = 2, \dots, \text{order}` the logarithm
    equals :math:`-iHt + O(t^{\,\text{order}+1})`.

    Args:
        terms: ``(pauli_label, coefficient)`` pairs for the Hermitian Hamiltonian.
            Labels use the little-endian convention (rightmost character is qubit 0).
        order: The Zassenhaus expansion order :math:`p \ge 1`.  ``order = 1`` reproduces
            the first-order Trotter product.
        tol: Magnitude below which graded coefficients are discarded.

    Returns:
        Ordered list of :class:`ZassenhausFactor`.  The rotation angle of factor
        ``f`` at per-step time ``t`` is ``f.angle_coefficient * t**f.degree``.

    Raises:
        ValueError: If ``order < 1``.

    """
    if order < 1:
        raise ValueError(f"Zassenhaus expansion order must be >= 1, got {order}.")

    # Seed: a_k = -i * alpha_k * P_k, carrying one power of t (degree 1).
    factors: list[tuple[PauliWord, int, complex]] = []
    target: GradedPoly = {}  # A = -i t H, the exact generator we are matching.
    for label, alpha in terms:
        word = _canonical_word(_label_to_word(label))
        coeff = -1j * alpha
        factors.append((word, 1, coeff))
        bucket = target.setdefault(word, {})
        bucket[1] = bucket.get(1, 0j) + coeff
    target = _poly_clean(target, tol)

    for degree in range(2, order + 1):
        unitary = _poly_identity()
        for word, factor_degree, coeff in factors:
            unitary = _poly_multiply(unitary, _poly_exp({word: {factor_degree: coeff}}, order), order)
        log_unitary = _poly_log(_poly_clean(unitary, tol), order)

        difference = _poly_add_inplace({w: dict(dm) for w, dm in log_unitary.items()}, target, scale=-1.0)
        residual = _poly_clean(difference, tol)

        # Append -R_n: the degree-`degree` part of the residual, in canonical order.
        for word in sorted(residual.keys()):
            coeff = residual[word].get(degree, 0j)
            if abs(coeff) > tol:
                factors.append((word, degree, -coeff))

    return [_to_factor(word, degree, coeff) for word, degree, coeff in factors]


def _to_factor(word: PauliWord, degree: int, coeff: complex) -> ZassenhausFactor:
    r"""Convert an internal ``(word, degree, antihermitian coeff)`` triple to a :class:`ZassenhausFactor`.

    The internal coefficient ``coeff`` is the antihermitian prefactor of
    :math:`\exp(\text{coeff}\, t^{d} P)`.  For a Hermitian Hamiltonian it is purely
    imaginary, so the rotation angle ``theta`` with :math:`\exp(-i\,\theta\,t^{d}P)`
    is ``theta = i * coeff`` (real).
    """
    angle_coefficient = (1j * coeff).real
    return ZassenhausFactor(pauli_term=_word_to_term_map(word), angle_coefficient=angle_coefficient, degree=degree)


def _label_to_word(label: str) -> list[tuple[int, int]]:
    """Convert a little-endian Pauli label to a sparse word (qubit-0-first, axis ints)."""
    axis = {"X": 1, "Y": 2, "Z": 3}
    word: list[tuple[int, int]] = []
    for index, char in enumerate(reversed(label)):
        if char != "I":
            word.append((index, axis[char]))
    return word
