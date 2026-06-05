r"""Zassenhaus error bound estimation for accuracy-aware parameterization.

This module provides functions to compute the number of Zassenhaus steps
required to achieve a target accuracy for product-formula decompositions.

The naive triangle-inequality bound is implemented first. A tighter
commutator-aware bound will be added separately once the generated
Zassenhaus commutator terms are evaluated for error estimation.

References:
    Childs, A. M., et al. "Theory of Trotter Error with Commutator
    Scaling." *Physical Review X* 11.1 (2021): 011020.
    https://arxiv.org/abs/1912.08854

"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qdk_chemistry.utils.zassenhaus_generation import zassenhaus_commutator_plan

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qdk_chemistry.data import QubitHamiltonian
    from qdk_chemistry.utils.zassenhaus_generation import PlanExpr

__all__: list[str] = [
    "zassenhaus_steps_commutator",
    "zassenhaus_steps_naive",
]


def _zassenhaus_coefficient_sum(
    *,
    order: int,
    num_terms: int,
    commutator_exponents: Mapping[int, PlanExpr] | None = None,
) -> float:
    r"""Return the absolute coefficient sum for the first omitted exponent.

    An order-``p`` Zassenhaus approximation keeps correction exponents through
    ``C_p``.  The leading omitted local remainder is therefore ``C_{p+1}``.
    For the naive bound we combine like symbolic commutators and use
    ``sum(abs(c_w))`` over the terms in ``C_{p+1}``.
    """
    if num_terms < 2:
        return 0.0

    omitted_order = order + 1
    if commutator_exponents is None:
        commutator_exponents, _ = zassenhaus_commutator_plan(
            tuple(range(num_terms)),
            max_order=omitted_order,
        )

    if omitted_order not in commutator_exponents:
        raise ValueError(
            f"commutator_exponents must include Zassenhaus exponent C_{omitted_order} "
            f"for order {order}."
        )

    return float(sum(abs(coeff) for coeff in commutator_exponents[omitted_order].values()))


def zassenhaus_steps_naive(
    hamiltonian: QubitHamiltonian,
    time: float,
    target_accuracy: float,
    *,
    order: int = 1,
    weight_threshold: float = 1e-12,
    commutator_exponents: Mapping[int, PlanExpr] | None = None,
) -> int:
    r"""Compute the number of Zassenhaus steps using a naive bound.

    This triangle-inequality estimate uses the Hamiltonian coefficient
    1-norm and treats an order-``p`` Zassenhaus formula as having local
    remainder scaling like ``O(t ** (p + 1))``.  The leading local remainder
    is the first omitted Zassenhaus exponent ``C_{p+1}``, whose combined
    symbolic commutator coefficients are bounded by
    :math:`\kappa_{p+1} = \sum_w |c_w|`.  Applying
    :math:`\|[A, B]\| \le 2 \|A\| \|B\|` repeatedly gives:

    .. math::

        \epsilon \le
        \frac{\kappa_{p+1} 2^p \|H\|_1^{p+1} |t|^{p+1}}{N^p}.

    After splitting the total simulation time into ``N`` steps, the minimum
    naive step count is:

    .. math::

        N = \left\lceil
            \frac{(\kappa_{p+1} 2^p \|H\|_1^{p+1})^{1/p}
            |t|^{1+1/p}}{\epsilon^{1/p}}
        \right\rceil .

    Args:
        hamiltonian: The qubit Hamiltonian to simulate.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.
        order: The order of the Zassenhaus formula.
        weight_threshold: Absolute threshold below which coefficients are discarded.
        commutator_exponents: Optional precomputed symbolic Zassenhaus exponents
            returned as the first element of :func:`zassenhaus_commutator_plan`.
            If provided, the mapping must include ``order + 1`` so the bound can
            use the first omitted exponent without regenerating the plan.

    Returns:
        The minimum number of Zassenhaus steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive or ``order`` is not positive.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    if order < 1:
        raise ValueError(f"Zassenhaus order must be positive, got {order}.")

    real_terms = hamiltonian.get_real_coefficients(tolerance=weight_threshold)
    one_norm = sum(abs(coeff) for _, coeff in real_terms)
    coefficient_sum = _zassenhaus_coefficient_sum(
        order=order,
        num_terms=len(real_terms),
        commutator_exponents=commutator_exponents,
    )

    if coefficient_sum == 0.0 or one_norm == 0.0 or time == 0.0:
        return 1

    return max(
        1,
        math.ceil(
            (coefficient_sum * 2**order * one_norm ** (order + 1)) ** (1 / order)
            * abs(time) ** (1 + 1 / order)
            / (target_accuracy ** (1 / order))
        ),
    )


def zassenhaus_steps_commutator(
    hamiltonian: QubitHamiltonian,
    time: float,
    target_accuracy: float,
    *,
    order: int = 1,
    weight_threshold: float = 1e-12,
) -> int:
    """Compute Zassenhaus steps using a commutator bound.

    This tighter bound is intentionally left for the commutator-aware
    implementation. Use :func:`zassenhaus_steps_naive` for now.
    """
    raise NotImplementedError("Zassenhaus commutator step estimation is not implemented yet.")
