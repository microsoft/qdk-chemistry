r"""Zassenhaus error bound estimation for accuracy-aware parameterization.

This module provides functions to compute the number of steps
required to achieve a target accuracy for Zassenhaus product-formula
decompositions.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.utils.pauli_commutation import commutator
from qdk_chemistry.utils.zassenhaus_generation import zassenhaus_commutator_plan

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qdk_chemistry.data import QubitHamiltonian
    from qdk_chemistry.utils.zassenhaus_generation import PlanExpr, PlanTerm

__all__: list[str] = [
    "zassenhaus_coefficient_sum",
    "zassenhaus_omitted_commutator_norm",
    "zassenhaus_steps_commutator",
    "zassenhaus_steps_naive",
]


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
            returned as the first element of :func:`~qdk_chemistry.utils.zassenhaus_generation.zassenhaus_commutator_plan`.
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
    num_real_terms = len(real_terms)
    coefficient_sum = zassenhaus_coefficient_sum(
        order=order,
        num_terms=num_real_terms,
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
    r"""Compute Zassenhaus steps using a commutator-aware bound.

    An order-``p`` Zassenhaus formula keeps exponents through ``C_p``.  The
    leading local remainder is the omitted exponent ``C_{p+1}``.  This bound
    evaluates that omitted exponent as an actual nested-commutator expression
    over the Hamiltonian's Pauli terms, rather than replacing every commutator
    by a worst-case triangle-inequality estimate.  If
    ``beta = ||C_{p+1}(-iH_1, ..., -iH_m)||`` is bounded by the Pauli
    coefficient 1-norm, then ``N`` time slices give

    .. math::

        \epsilon \le \frac{\beta |t|^{p+1}}{N^p}.

    Args:
        hamiltonian: The qubit Hamiltonian to simulate.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.
        order: The order of the Zassenhaus formula.
        weight_threshold: Absolute threshold below which coefficients are discarded.

    Returns:
        The minimum number of Zassenhaus steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive or ``order`` is not positive.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    if order < 1:
        raise ValueError(f"Zassenhaus order must be positive, got {order}.")

    omitted_norm = zassenhaus_omitted_commutator_norm(
        hamiltonian,
        order=order,
        weight_threshold=weight_threshold,
    )

    if omitted_norm == 0.0 or time == 0.0:
        return 1

    return max(
        1,
        math.ceil(omitted_norm ** (1 / order) * abs(time) ** (1 + 1 / order) / (target_accuracy ** (1 / order))),
    )


def zassenhaus_coefficient_sum(
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
        raise ValueError(f"commutator_exponents must include Zassenhaus exponent C_{omitted_order} for order {order}.")

    return float(sum(abs(coeff) for coeff in commutator_exponents[omitted_order].values()))


def _combine_hamiltonian_terms(
    terms: Mapping[str, complex],
    *,
    num_qubits: int,
    weight_threshold: float,
) -> QubitHamiltonian:
    from qdk_chemistry.data import QubitHamiltonian  # noqa: PLC0415

    filtered = [(label, coeff) for label, coeff in terms.items() if abs(coeff) > weight_threshold]
    if not filtered:
        return QubitHamiltonian(["I" * num_qubits], np.array([0.0], dtype=complex))

    labels, coefficients = zip(*filtered, strict=True)
    return QubitHamiltonian(list(labels), np.asarray(coefficients, dtype=complex))


def zassenhaus_omitted_commutator_norm(
    hamiltonian: QubitHamiltonian,
    *,
    order: int,
    weight_threshold: float,
) -> float:
    r"""Return a Pauli 1-norm bound for the first omitted Zassenhaus exponent.

    The returned value bounds ``||C_{order + 1}(-i H_1, ..., -i H_m)||``.
    Multiplication by powers of ``-i`` does not change the norm, so the
    Hamiltonian commutator DAG can be evaluated directly and scaled by the
    symbolic Zassenhaus coefficients.
    """
    real_terms = hamiltonian.get_real_coefficients(tolerance=weight_threshold)
    if len(real_terms) < 2:
        return 0.0

    omitted_order = order + 1
    leaves = tuple(range(len(real_terms)))
    commutator_exponents, plan = zassenhaus_commutator_plan(leaves, max_order=omitted_order)

    exponent = commutator_exponents[omitted_order]
    if not exponent:
        return 0.0

    from qdk_chemistry.data import QubitHamiltonian  # noqa: PLC0415

    leaf_hamiltonians: dict[int, QubitHamiltonian] = {
        idx: QubitHamiltonian([label], np.asarray([coeff], dtype=complex))
        for idx, (label, coeff) in enumerate(real_terms)
    }
    cache: dict[PlanTerm, QubitHamiltonian] = {}

    def evaluate(ref: PlanTerm) -> QubitHamiltonian:
        if isinstance(ref, int):
            return leaf_hamiltonians[ref]

        if ref not in plan:
            raise TypeError(f"Unexpected Zassenhaus plan reference: {ref!r}.")

        if ref not in cache:
            left, right = plan[ref]
            cache[ref] = commutator(evaluate(left), evaluate(right))
        return cache[ref]

    combined_terms: dict[str, complex] = defaultdict(complex)
    for ref, symbolic_coeff in exponent.items():
        contribution = evaluate(ref)
        for label, coeff in zip(contribution.pauli_strings, contribution.coefficients, strict=True):
            combined_terms[label] += complex(symbolic_coeff) * complex(coeff)

    omitted_hamiltonian = _combine_hamiltonian_terms(
        combined_terms,
        num_qubits=hamiltonian.num_qubits,
        weight_threshold=weight_threshold,
    )
    return omitted_hamiltonian.schatten_norm
