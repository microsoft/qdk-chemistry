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


# ----------------------------------------------------------------------------------
# Commutator-based log of a product (avoids materialising the dense product)
# ----------------------------------------------------------------------------------

# Dexpinv coefficients c_k = (-1)**k * B_k / k!  (B_k = Bernoulli numbers), from
#   log(e^Z e^f) = Z + integral_0^1  sum_k c_k ad_{Omega(s)}^k (f)  ds.
# Odd k > 1 vanish.  Tabulated through k = 8 (covers expansion orders well past use).
_DEXPINV_COEFFS: dict[int, float] = {
    0: 1.0,
    1: 0.5,
    2: 1.0 / 12.0,
    4: -1.0 / 720.0,
    6: 1.0 / 30240.0,
    8: -1.0 / 1209600.0,
}

# An operator carrying an explicit power of the BCH integration variable ``s``:
# s_power -> graded Pauli polynomial.
SPoly = dict[int, GradedPoly]


def _words_commute(word_a: PauliWord, word_b: PauliWord) -> bool:
    """Whether two Pauli words commute.

    Two Pauli strings commute iff the number of qubits where both act with
    *different* (non-identity) single-qubit operators is even.  This mirrors the
    label-based check in :func:`qdk_chemistry.utils.pauli_commutation.commutator`,
    on sparse words instead of full strings.
    """
    axes_b = dict(word_b)
    anticommuting = sum(1 for qubit, axis_a in word_a if axes_b.get(qubit, axis_a) != axis_a)
    return anticommuting % 2 == 0


def _word_commutator(word_a: PauliWord, word_b: PauliWord) -> tuple[complex, PauliWord] | None:
    r"""Commutator of two Pauli words: ``(coeff, word)`` for :math:`[P_a, P_b]`, or ``None`` if they commute.

    When the two strings anticommute, :math:`P_b P_a = -P_a P_b`, so
    :math:`[P_a, P_b] = P_a P_b - P_b P_a = 2\,P_a P_b`.  We therefore short-circuit on
    the cheap commutativity pre-check (skipping commuting pairs entirely) and evaluate a
    single product :math:`P_a P_b` -- the :math:`P_b P_a` phase is fixed by the
    anticommutation relation and never computed.  Every commutator is a single Pauli
    word, which is what keeps the corrections sparse.
    """
    if _words_commute(word_a, word_b):
        return None
    phase_ab, product = PauliTermAccumulator.multiply_uncached(list(word_a), list(word_b))
    return 2.0 * phase_ab, _canonical_word(product)


def _poly_commutator(left: GradedPoly, right: GradedPoly, max_degree: int, tol: float = 1e-14) -> GradedPoly:
    """Graded commutator ``[left, right]``, computed term by term.

    Only anticommuting word pairs contribute; commuting pairs are skipped, so the
    result stays as sparse as the generated Lie algebra allows (no dense intermediate).
    """
    out: GradedPoly = {}
    for word_a, deg_a in left.items():
        for word_b, deg_b in right.items():
            commuted = _word_commutator(word_a, word_b)
            if commuted is None:
                continue
            comm_coeff, word = commuted
            bucket = out.setdefault(word, {})
            for da, ca in deg_a.items():
                for db, cb in deg_b.items():
                    degree = da + db
                    if degree > max_degree:
                        continue
                    bucket[degree] = bucket.get(degree, 0j) + comm_coeff * ca * cb
    return _poly_clean(out, tol)


def _sop_add_inplace(target: SPoly, other: SPoly, scale: complex = 1.0) -> SPoly:
    """Accumulate ``scale * other`` into an s-power-indexed operator ``target`` (in place)."""
    for s_power, poly in other.items():
        _poly_add_inplace(target.setdefault(s_power, {}), poly, scale)
    return target


def _sop_to_poly(sop: SPoly) -> GradedPoly:
    """Evaluate an s-power operator at ``s = 1`` by summing over its s-powers."""
    total: GradedPoly = {}
    for poly in sop.values():
        _poly_add_inplace(total, poly)
    return total


def _sop_ad(omega: SPoly, term: SPoly, max_degree: int) -> SPoly:
    """Apply the adjoint action ``[Omega, .]`` to an s-power operator (s-powers add)."""
    out: SPoly = {}
    for i, op_i in omega.items():
        for j, op_j in term.items():
            comm = _poly_commutator(op_i, op_j, max_degree)
            if comm:
                _poly_add_inplace(out.setdefault(i + j, {}), comm)
    return out


def _polys_close(a: GradedPoly, b: GradedPoly, tol: float) -> bool:
    """Whether two graded polynomials agree to within ``tol`` on every coefficient."""
    for word in set(a) | set(b):
        da, db = a.get(word, {}), b.get(word, {})
        for degree in set(da) | set(db):
            if abs(da.get(degree, 0j) - db.get(degree, 0j)) > tol:
                return False
    return True


def _bch_merge(z_poly: GradedPoly, f_poly: GradedPoly, max_degree: int, tol: float = 1e-14) -> GradedPoly:
    r"""Return ``log(e^Z e^f)`` truncated to t-degree ``max_degree`` using commutators only.

    Solves the dexpinv ODE :math:`\Omega'(s) = \sum_k c_k\,\mathrm{ad}_{\Omega(s)}^k(f)`
    with :math:`\Omega(0)=Z` for :math:`\Omega(1)`, by Picard iteration that tracks the
    integration variable ``s`` as an explicit polynomial degree.  ``e^Z`` (the dense
    product) is never formed; the only non-sparse object is the log itself.
    """
    seed = {w: dict(dm) for w, dm in z_poly.items()}
    omega: SPoly = {0: {w: dict(dm) for w, dm in seed.items()}}
    previous = _poly_clean(_sop_to_poly(omega), tol)
    converged = False
    for _ in range(max_degree + 3):
        # integrand(s) = sum_k c_k ad_{Omega(s)}^k (f)
        integrand: SPoly = {}
        term: SPoly = {0: {w: dict(dm) for w, dm in f_poly.items()}}
        _sop_add_inplace(integrand, term, _DEXPINV_COEFFS[0])
        for k in range(1, max_degree + 1):
            term = _sop_ad(omega, term, max_degree)
            if not term:
                break
            ck = _DEXPINV_COEFFS.get(k, 0.0)
            if ck != 0.0:
                _sop_add_inplace(integrand, term, ck)
        # Omega(s) = Z + integral_0^s integrand d(sigma):  s_power j -> j+1, divide by j+1
        new_omega: SPoly = {0: {w: dict(dm) for w, dm in seed.items()}}
        for s_power, poly in integrand.items():
            integrated = {word: {deg: c / (s_power + 1) for deg, c in dm.items()} for word, dm in poly.items()}
            _poly_add_inplace(new_omega.setdefault(s_power + 1, {}), integrated)
        omega = {sp: cleaned for sp, p in new_omega.items() if (cleaned := _poly_clean(p, tol))}
        current = _poly_clean(_sop_to_poly(omega), tol)
        converged = _polys_close(current, previous, tol)
        previous = current
        if converged:
            break
    if not converged:
        raise RuntimeError(
            f"BCH merge (dexpinv Picard iteration) did not converge within {max_degree + 3} iterations "
            f"at max_degree={max_degree}."
        )
    return previous


def _log_of_product(
    factors: list[tuple[PauliWord, int, complex]], max_degree: int, tol: float = 1e-14
) -> GradedPoly:
    r"""Compute ``log(prod_m exp(f_m))`` to t-degree ``max_degree`` via incremental BCH merges.

    Each ``f_m`` is a single Pauli term ``(word, degree, coeff)``.  The factors are
    folded left to right, merging each into the running log with :func:`_bch_merge`,
    so the dense product is never materialised and commuting pairs are skipped.
    """
    running: GradedPoly = {}
    for word, degree, coeff in factors:
        f_poly: GradedPoly = {word: {degree: coeff}}
        running = f_poly if not running else _bch_merge(running, f_poly, max_degree, tol)
    return _poly_clean(running, tol)


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

    1. computes :math:`L = \log\bigl(\prod_m \exp(f_m)\bigr)` as a graded Pauli polynomial,
    2. extracts the degree-:math:`n` residual :math:`R_n` of :math:`L - (-iHt)`, and
    3. appends single-Pauli factors realising :math:`-R_n`.

    Appending degree-:math:`n` factors cancels the order-:math:`n` term of
    :math:`\log U` exactly (their mutual cross terms are of degree :math:`\ge n+1`),
    and recomputing :math:`L` at every step folds all splitting errors into later
    corrections.  After processing :math:`n = 2, \dots, \text{order}` the logarithm
    equals :math:`-iHt + O(t^{\,\text{order}+1})`.

    The logarithm :math:`L = \log\prod_m \exp(f_m)` is computed directly via nested
    commutators (the BCH/dexpinv recursion in :func:`_log_of_product`) -- each commutator
    of Pauli words is a single word and commuting pairs are skipped, so the dense
    :math:`4^n` product is never materialised.

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

    # Maintain the running log incrementally: start from the seed product, then fold in
    # each batch of corrections as they are appended -- rather than recomputing the full
    # product log every order (the corrections from earlier orders are already baked in).
    running_log = _log_of_product(factors, order, tol)

    for degree in range(2, order + 1):
        difference = _poly_add_inplace({w: dict(dm) for w, dm in running_log.items()}, target, scale=-1.0)
        residual = _poly_clean(difference, tol)

        # Append -R_n: the degree-`degree` part of the residual, in canonical order.
        new_factors: list[tuple[PauliWord, int, complex]] = []
        for word in sorted(residual.keys()):
            coeff = residual[word].get(degree, 0j)
            if abs(coeff) > tol:
                new_factors.append((word, degree, -coeff))
        factors.extend(new_factors)

        # Fold the newly appended factors into the running log for the next order
        # (skipped on the final order: the log is not needed again).
        if degree < order:
            for word, factor_degree, coeff in new_factors:
                running_log = _bch_merge(running_log, {word: {factor_degree: coeff}}, order, tol)

    return [_to_factor(word, degree, coeff) for word, degree, coeff in factors]


def _to_factor(word: PauliWord, degree: int, coeff: complex, hermiticity_tol: float = 1e-9) -> ZassenhausFactor:
    r"""Convert an internal ``(word, degree, antihermitian coeff)`` triple to a :class:`ZassenhausFactor`.

    The internal coefficient ``coeff`` is the antihermitian prefactor of
    :math:`\exp(\text{coeff}\, t^{d} P)`.  For a Hermitian Hamiltonian it is purely
    imaginary, so the rotation angle ``theta`` with :math:`\exp(-i\,\theta\,t^{d}P)`
    is ``theta = i * coeff`` (real); the rotation angle keeps ``Im(coeff)`` and the
    real part is discarded.

    A non-negligible real part signals non-Hermitian drift (an upstream non-Hermitian
    input or an arithmetic bug), so it is rejected rather than silently dropped.

    Raises:
        ValueError: If ``abs(coeff.real) > hermiticity_tol``.
    """
    if abs(coeff.real) > hermiticity_tol:
        axes = {1: "X", 2: "Y", 3: "Z"}
        label = " ".join(f"{axes[a]}{q}" for q, a in word) or "I"
        raise ValueError(
            f"Non-Hermitian drift: factor on [{label}] (degree {degree}) has "
            f"|Re(coeff)| = {abs(coeff.real):.2e} > {hermiticity_tol:.1e}."
        )
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
