"""Tests for symbolic Zassenhaus generation utilities."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from collections.abc import Hashable
from fractions import Fraction

from qdk_chemistry.utils.zassenhaus_generation import CommutatorPlan, PlanExpr, PlanTerm, zassenhaus_commutator_plan

ExpandedTerm = Hashable | tuple["ExpandedTerm", "ExpandedTerm"]


def _expand_plan_term(ref: PlanTerm, plan: CommutatorPlan) -> ExpandedTerm:
    if ref not in plan:
        return ref

    left, right = plan[ref]
    return (_expand_plan_term(left, plan), _expand_plan_term(right, plan))


def _expand_plan_expr(expr: PlanExpr, plan: CommutatorPlan) -> dict[ExpandedTerm, Fraction]:
    return {_expand_plan_term(ref, plan): coeff for ref, coeff in expr.items()}


def _assert_plan_is_dependency_ordered(plan: CommutatorPlan, leaves: tuple[Hashable, ...]) -> None:
    available: set[PlanTerm] = set(leaves)

    for node, (left, right) in plan.items():
        assert left in available
        assert right in available
        available.add(node)


def test_zassenhaus_generation_two_operator_formula_plan():
    planned_exponents, plan = zassenhaus_commutator_plan(("A", "B"), max_order=5)

    _assert_plan_is_dependency_ordered(plan, ("A", "B"))

    # verify second order terms:
    assert _expand_plan_expr(planned_exponents[2], plan) == {
        ("A", "B"): Fraction(-1, 2),
    }

    # verify third order terms are correct:
    assert _expand_plan_expr(planned_exponents[3], plan) == {
        ("A", ("A", "B")): Fraction(1, 6),
        ("B", ("A", "B")): Fraction(1, 3),
    }

    # verify fourth order terms are correct:
    assert _expand_plan_expr(planned_exponents[4], plan) == {
        ("A", ("A", ("A", "B"))): Fraction(-1, 24),
        ("B", ("A", ("A", "B"))): Fraction(-1, 8),
        ("B", ("B", ("A", "B"))): Fraction(-1, 8),
    }

    # verify fifth order terms are correct:
    assert _expand_plan_expr(planned_exponents[5], plan) == {
        ("A", ("A", ("A", ("A", "B")))): Fraction(1, 120),
        ("B", ("A", ("A", ("A", "B")))): Fraction(1, 30),
        ("B", ("B", ("A", ("A", "B")))): Fraction(1, 20),
        ("B", ("B", ("B", ("A", "B")))): Fraction(1, 30),
        (("A", "B"), ("A", ("A", "B"))): Fraction(1, 20),
        (("A", "B"), ("B", ("A", "B"))): Fraction(1, 10),
    }


def test_zassenhaus_generation_three_operator_formula_plan():
    planned_exponents, plan = zassenhaus_commutator_plan(("A", "B", "C"), max_order=2)

    _assert_plan_is_dependency_ordered(plan, ("A", "B", "C"))
    assert _expand_plan_expr(planned_exponents[2], plan) == {
        ("A", "B"): Fraction(-1, 2),
        ("A", "C"): Fraction(-1, 2),
        ("B", "C"): Fraction(-1, 2),
    }
