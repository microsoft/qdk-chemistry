r"""Trotter error bound estimation for accuracy-aware parameterization.

This module provides functions to compute the number of Trotter steps
required to achieve a target accuracy for product-formula decompositions.

Two bounds are offered:

* **naive** - uses the triangle-inequality bound
  :math:`N = \\lceil (\\sum_j |\\alpha_j|)^2 t^2 / \\epsilon \\rceil`.

* **commutator** (tighter) - uses the commutator-based bound from
  Childs *et al.* (2021):
  :math:`N = \\lceil \\frac{t^2}{2\\epsilon}
  \\sum_{j<k} \\lVert [\\alpha_j P_j, \\alpha_k P_k] \\rVert \\rceil`.

References:
    Childs, A. M., et al. "Theory of Trotter Error with Commutator
    Scaling." *Physical Review X* 11.1 (2021): 011020.
    https://arxiv.org/abs/1912.08854

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qdk_chemistry.utils.pauli_commutation import (
    commutator_bound_first_order,
    commutator_bound_higher_order,
    commutator_bound_second_order,
)

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitHamiltonian

__all__: list[str] = [
    "trotter_steps_commutator",
    "trotter_steps_naive",
]


def trotter_steps_naive(
    hamiltonian: QubitHamiltonian,
    time: float,
    target_accuracy: float,
    *,
    order: int = 1,
    weight_threshold: float = 1e-12,
) -> int:
    r"""Compute the number of Trotter steps using the naive bound.

    The naive (triangle-inequality) bound is

    .. math::

        N = \left\lceil \frac{2(\sum_j |\alpha_j|)^{1 + 1/p} \, t^{1+1/p} \, C_{\text{max}}^{1/p}}
        {\epsilon^{1/p}} \right\rceil

    where :math:`\sum_j |\alpha_j|` is the 1-norm of the Hamiltonian
    coefficients, :math:`p` is the order of the Trotter-Suzuki product formula,
    and :math:`C_{\text{max}}` is the largest coefficient in the Taylor expansion of
    the Trotter error, :math:`(exp(H_1+...+H_L) - S_order)`, to :math:`t^(order+1)`
    when all coefficients are 1 and :math:`t = 1`.

    Args:
        hamiltonian: The qubit Hamiltonian to simulate.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.
        order: The order of the Trotter-Suzuki product formula.
        weight_threshold: Absolute threshold below which coefficients are discarded.

    Returns:
        The minimum number of Trotter steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive.
        NotImplementedError: If *order* is not supported.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    if order not in {1, 2} and not (order > 2 and order % 2 == 0):
        raise NotImplementedError(
            f"Trotter step estimation for order {order} is not yet implemented. "
            "Non-positive and higher odd orders are not supported."
        )
    real_terms = hamiltonian.get_real_coefficients(tolerance=weight_threshold)
    one_norm = sum(abs(coeff) for _, coeff in real_terms)
    if order == 1:
        return max(1, math.ceil(((2 * one_norm**2) * (1 / 2.0) * time**2) / target_accuracy))
    if order == 2:
        return max(
            1,
            math.ceil(
                ((2**2 * one_norm**3) ** (1 / 2) * (1 / 12.0 ** (1 / 2)) * abs(time) ** (1 + 1 / 2))
                / target_accuracy ** (1 / 2)
            ),
        )
    return max(
        1,
        math.ceil(
            (2**order * one_norm ** (order + 1)) ** (1 / order)
            * abs(time) ** (1 + 1 / order)
            / (target_accuracy ** (1 / order))
        ),
    )


def trotter_steps_commutator(
    hamiltonian: QubitHamiltonian,
    time: float,
    target_accuracy: float,
    *,
    order: int = 1,
    weight_threshold: float = 1e-12,
) -> int:
    r"""Compute the number of Trotter steps using the commutator bound.

    The tighter commutator-based bound from Childs *et al.* (2021) is

    .. math::

        N_1 = \left\lceil \frac{\alpha_1 t^2}{2\epsilon}\right\rceil

        \alpha_1 =
            \sum_{j<k} \lVert [\alpha_j P_j,\, \alpha_k P_k] \rVert

        N_2 = \left\lceil \frac{t^{3/2}\alpha_2^{1/2}}{(12\epsilon)^{1/2}} \right\rceil

        \alpha_2 = \sum_{k > j,l > j} \lVert [\alpha_l P_l,\, [\alpha_k P_k,\, \alpha_j P_j] \rVert +
            \frac{1}{2} \sum_{k > j} \lVert [\alpha_j P_j,\, [\alpha_j P_j,\, \alpha_k P_k] \rVert

        N_p = \left\lceil \frac{C_{\text{max}}^{1/p} t^{1+1/p}\alpha_p^{1/p}}{\epsilon^{1/p}} \right\rceil

        \alpha_p = \sum_{j_1,\ldots,j_{p+1}} \lVert [\alpha_{j_1} P_{j_1},\, [\ldots [\alpha_{j_p} P_{j_p},
        \alpha_{j_{p+1}}P_{j_{p+1}}]\ldots]\rVert

    For Pauli strings the commutator norm is :math:`2|\alpha_j||\alpha_k|`
    when the pair anticommutes and 0 when it commutes.  This bound is never
    looser than the naive bound and can be substantially tighter when many
    terms commute.

    Args:
        hamiltonian: The qubit Hamiltonian to simulate.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.
        order: The order of the Trotter-Suzuki product formula.
        weight_threshold: Absolute threshold below which coefficients are discarded.

    Returns:
        The minimum number of Trotter steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive.
        NotImplementedError: If *order* is not supported.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    if order not in {1, 2} and not (order > 2 and order % 2 == 0):
        raise NotImplementedError(
            f"Trotter step estimation for order {order} is not yet implemented. "
            "Non-positive and higher odd orders are not supported."
        )
    if order == 1:
        comm_bound = commutator_bound_first_order(hamiltonian, weight_threshold=weight_threshold)
        return max(1, math.ceil(comm_bound * 1 / 2.0 * time**2 / (target_accuracy)))
    if order == 2:
        comm_bound = commutator_bound_second_order(hamiltonian, weight_threshold=weight_threshold)

        return max(
            1,
            math.ceil(
                comm_bound ** (1 / 2) * (1 / 12.0) ** (1 / 2) * abs(time) ** (1 + 1 / 2) / (target_accuracy) ** (1 / 2)
            ),
        )

    comm_bound = commutator_bound_higher_order(hamiltonian, order=order, weight_threshold=weight_threshold)
    return max(
        1,
        math.ceil(
            (comm_bound / (order + 1)) ** (1 / order) * abs(time) ** (1 + 1 / order) / (target_accuracy) ** (1 / order)
        ),
    )
