"""Trotter error bound estimation for accuracy-aware parameterization.

This module provides functions to compute the number of Trotter steps
required to achieve a target accuracy for the first-order (Lie–Trotter)
product formula.

Two bounds are offered:

* **naive** – uses the triangle-inequality bound
  :math:`N = \\lceil (\\sum_j |\\alpha_j|)^2 t^2 / \\epsilon \\rceil`.

* **commutator** (tighter) – uses the commutator-based bound from
  Childs *et al.* (2021) :cite:`Childs2021`:
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

import math

from qdk_chemistry.algorithms.time_evolution.builder.pauli_commutation import (
    commutator_bound_first_order,
)

__all__: list[str] = [
    "trotter_steps_naive",
    "trotter_steps_commutator",
]


def trotter_steps_naive(
    one_norm: float,
    time: float,
    target_accuracy: float,
) -> int:
    r"""Compute the number of Trotter steps using the naive bound.

    The naive (triangle-inequality) bound is

    .. math::

        N = \left\lceil \frac{(\sum_j |\alpha_j|)^2 \, t^2}{\epsilon} \right\rceil

    where :math:`\sum_j |\alpha_j|` is the 1-norm (Schatten norm) of the
    Hamiltonian coefficients.

    Args:
        one_norm: The 1-norm :math:`\sum_j |\alpha_j|` of the Hamiltonian.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.

    Returns:
        The minimum number of Trotter steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    return max(1, math.ceil(one_norm**2 * time**2 / target_accuracy))


def trotter_steps_commutator(
    pauli_labels: list[str],
    coefficients: list[float],
    time: float,
    target_accuracy: float,
) -> int:
    r"""Compute the number of Trotter steps using the commutator bound.

    The tighter commutator-based bound from Childs *et al.* (2021) is

    .. math::

        N = \left\lceil \frac{t^2}{2\epsilon}
            \sum_{j<k} \lVert [\alpha_j P_j,\, \alpha_k P_k] \rVert
        \right\rceil

    For Pauli strings the commutator norm is :math:`2|\alpha_j||\alpha_k|`
    when the pair anticommutes and 0 when it commutes.  This bound is never
    looser than the naive bound and can be substantially tighter when many
    terms commute.

    Args:
        pauli_labels: List of Pauli string labels for each Hamiltonian term.
        coefficients: Corresponding real coefficients.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.

    Returns:
        The minimum number of Trotter steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    comm_bound = commutator_bound_first_order(pauli_labels, coefficients)
    return max(1, math.ceil(comm_bound * time**2 / (2.0 * target_accuracy)))
