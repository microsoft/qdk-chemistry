r"""Symbolic generator for Zassenhaus exponents and evaluation plans.

This module provides utilities for generating symbolic Zassenhaus exponents
and building structured commutator evaluation plans. The generated plans
identify shared nested commutator sub-expressions, enabling efficient
caching and linear-time evaluation of the commutator DAG.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import factorial

Term = Hashable | tuple["Term", "Term"]
Expr = dict[Term, Fraction]
Series = dict[int, Expr]


@dataclass(frozen=True)
class CommutatorNode:
    """Reference to a nested commutator in a generated evaluation plan."""

    index: int


PlanTerm = Hashable | CommutatorNode
PlanExpr = dict[PlanTerm, Fraction]
CommutatorPlan = dict[CommutatorNode, tuple[PlanTerm, PlanTerm]]

__all__ = [
    "CommutatorNode",
    "CommutatorPlan",
    "PlanExpr",
    "PlanTerm",
    "zassenhaus_commutator_plan",
    "zassenhaus_exponents",
]


def _term(x: Term, coeff=Fraction(1)) -> Expr:
    coeff = Fraction(coeff)
    if coeff == 0:
        return {}
    return {x: coeff}


def _clean(expr: Expr) -> Expr:
    return {x: c for x, c in expr.items() if c != 0}


def _add(*exprs: Expr) -> Expr:
    out: defaultdict[Term, Fraction] = defaultdict(Fraction)

    for expr in exprs:
        for x, coeff in expr.items():
            out[x] += coeff

    return _clean(dict(out))


def _scale(expr: Expr, factor) -> Expr:
    factor = Fraction(factor)

    if factor == 0:
        return {}

    return _clean({x: factor * coeff for x, coeff in expr.items()})


def _term_key(x: Term) -> str:
    return repr(x)


def _commutator_term(x: Term, y: Term) -> tuple[Term | None, Fraction]:
    if x == y:
        return None, Fraction(0)

    # Canonicalize [y, x] = -[x, y].
    if _term_key(x) > _term_key(y):
        z, sign = _commutator_term(y, x)
        return z, -sign

    return (x, y), Fraction(1)


def _commutator(expr_a: Expr, expr_b: Expr) -> Expr:
    out: defaultdict[Term, Fraction] = defaultdict(Fraction)

    for x, coeff_x in expr_a.items():
        for y, coeff_y in expr_b.items():
            z, sign = _commutator_term(x, y)

            if z is not None:
                out[z] += coeff_x * coeff_y * sign

    return _clean(dict(out))


def _series_add_to(
    series: Series,
    degree: int,
    expr: Expr,
    coeff=Fraction(1),
    max_degree: int | None = None,
) -> None:
    if max_degree is not None and degree > max_degree:
        return

    coeff = Fraction(coeff)

    if coeff == 0 or not expr:
        return

    series[degree] = _add(series.get(degree, {}), _scale(expr, coeff))


def _apply_exp_ad_to_series(
    series: Series,
    generator: Expr,
    q: int,
    max_degree: int,
) -> Series:
    out: Series = {}

    for degree, value in series.items():
        current = value
        ell = 0

        while degree + q * ell <= max_degree:
            _series_add_to(
                out,
                degree + q * ell,
                current,
                Fraction(1, factorial(ell)),
                max_degree=max_degree,
            )

            ell += 1
            current = _commutator(generator, current)

            if not current:
                break

    return out


def _normalize_leaves(
    leaves: Sequence[Hashable] = ("A", "B"),
) -> tuple[Hashable, ...]:
    out = tuple(leaves)

    if len(out) < 2:
        raise ValueError("Zassenhaus expansions require at least two leaves.")

    if len(set(out)) != len(out):
        raise ValueError("Zassenhaus leaf labels must be unique.")

    return out


def _compute_exponent(
    leaves: Sequence[Hashable],
    order: int,
    previous_exponents: dict[int, Expr],
) -> Expr:
    if order < 2:
        raise ValueError("Zassenhaus exponents start at C_2.")

    leaf_terms = [_term(leaf) for leaf in leaves]

    # First residual: [t^(order-1)] of each X_i transformed by all earlier exp(t ad_X_j).
    residual: Expr = {}
    for i in range(1, len(leaf_terms)):
        transformed: Series = {0: leaf_terms[i]}

        for generator in reversed(leaf_terms[:i]):
            transformed = _apply_exp_ad_to_series(
                transformed,
                generator,
                q=1,
                max_degree=order - 1,
            )

        residual = _add(residual, transformed.get(order - 1, {}))

    # Add contributions from previous C_k by extracting [t^(order-k)] from A_{k-1}(t) C_k.
    for k in range(2, order):
        if k not in previous_exponents:
            raise ValueError(f"Missing previous exponent C_{k}.")

        max_degree = order - k
        transformed_k: Series = {0: previous_exponents[k]}
        generators: list[tuple[Expr, int]] = [(leaf_expr, 1) for leaf_expr in leaf_terms]
        generators += [(previous_exponents[j], j) for j in range(2, k)]

        for generator, q in reversed(generators):
            transformed_k = _apply_exp_ad_to_series(
                transformed_k,
                generator,
                q,
                max_degree=max_degree,
            )

        residual = _add(residual, _scale(transformed_k.get(max_degree, {}), k))

    return _scale(residual, Fraction(-1, order))


def zassenhaus_exponents(
    leaves: Sequence[Hashable] = ("A", "B"),
    max_order: int = 5,
) -> dict[int, Expr]:
    """Compute C_2, ..., C_max_order for an ordered product of primitive leaves."""
    leaf_tuple = _normalize_leaves(leaves)
    exponents: dict[int, Expr] = {}

    for order in range(2, max_order + 1):
        exponents[order] = _compute_exponent(leaf_tuple, order, exponents)

    return exponents


def _add_term_dependencies(
    x: Term,
    plan: CommutatorPlan,
    primitive_leaves: set[Hashable],
    term_nodes: dict[Term, CommutatorNode],
) -> PlanTerm:
    if x in primitive_leaves or not isinstance(x, tuple) or len(x) != 2:
        return x

    if x in term_nodes:
        return term_nodes[x]

    left, right = x
    left_ref = _add_term_dependencies(left, plan, primitive_leaves, term_nodes)
    right_ref = _add_term_dependencies(right, plan, primitive_leaves, term_nodes)

    node = CommutatorNode(len(term_nodes))
    term_nodes[x] = node
    plan[node] = (left_ref, right_ref)
    return node


def _plan_expr(
    expr: Expr,
    plan: CommutatorPlan,
    primitive_leaves: set[Hashable],
    term_nodes: dict[Term, CommutatorNode],
) -> PlanExpr:
    out: defaultdict[PlanTerm, Fraction] = defaultdict(Fraction)

    for x, coeff in expr.items():
        ref = _add_term_dependencies(x, plan, primitive_leaves, term_nodes)
        out[ref] += coeff

    return _clean(dict(out))


# Cache the symbolic plan. Since the commutator plan depends purely on the number of leaves
# and the target order (not on coefficients or time), caching avoids redundant DAG builds.
@functools.lru_cache(maxsize=128)
def _cached_zassenhaus_commutator_plan(
    leaves: tuple[Hashable, ...],
    max_order: int,
) -> tuple[dict[int, PlanExpr], CommutatorPlan]:
    exponents = zassenhaus_exponents(leaves, max_order=max_order)
    plan: CommutatorPlan = {}
    planned_exponents: dict[int, PlanExpr] = {}
    primitive_leaves = set(leaves)
    term_nodes: dict[Term, CommutatorNode] = {}

    for order, expr in exponents.items():
        planned_exponents[order] = _plan_expr(expr, plan, primitive_leaves, term_nodes)

    return planned_exponents, plan


def zassenhaus_commutator_plan(
    leaves: Sequence[Hashable],
    max_order: int = 5,
) -> tuple[dict[int, PlanExpr], CommutatorPlan]:
    """Compute Zassenhaus exponents and an explicit nested-commutator DAG.

    The returned exponents reference either primitive leaves or
    :class:`CommutatorNode` instances. The returned plan maps each
    ``CommutatorNode`` to its left and right child references. Nodes are
    inserted after their children, so iteration over ``plan.items()`` gives a
    valid evaluation order.
    """
    leaf_tuple = _normalize_leaves(leaves)
    cached_exponents, cached_plan = _cached_zassenhaus_commutator_plan(leaf_tuple, max_order)

    # Return fresh copies to prevent caller mutation from corrupting the cached objects
    exponents_copy = {k: dict(v) for k, v in cached_exponents.items()}
    plan_copy = dict(cached_plan)
    return exponents_copy, plan_copy
