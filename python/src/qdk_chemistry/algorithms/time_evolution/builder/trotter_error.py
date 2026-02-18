r"""Trotter error bound estimation for accuracy-aware parameterization.

This module provides functions to compute the number of Trotter steps
required to achieve a target accuracy for product-formula decompositions.

Currently implemented for first-order (Lie–Trotter).  The framework is
designed to be extended to higher-order formulas — see the stub
:func:`trotter_steps` for the intended entry point.

Two bounds are offered for first order:

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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qdk_chemistry.utils.pauli_commutation import commutator_bound_first_order

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitHamiltonian

__all__: list[str] = [
    "trotter_steps",
    "trotter_steps_commutator",
    "trotter_steps_naive",
]


# =====================================================================
# Public entry point (order-aware dispatcher)
# =====================================================================


def trotter_steps(
    hamiltonian: QubitHamiltonian,
    time: float,
    target_accuracy: float,
    *,
    order: int = 1,
    error_bound: str = "commutator",
    weight_threshold: float = 1e-12,
) -> int:
    r"""Compute the minimum number of Trotter steps for a target accuracy.

    This is the primary entry point for accuracy-aware step-count
    estimation.  It dispatches to the appropriate order-specific
    implementation.

    Args:
        hamiltonian: The qubit Hamiltonian to simulate.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.
        order: The order of the Trotter–Suzuki product formula.
            Currently only ``1`` (Lie–Trotter) is implemented.
        error_bound: Strategy for computing the error bound.
            ``"commutator"`` (default, tighter) or ``"naive"``.
        weight_threshold: Absolute threshold below which Hamiltonian
            coefficients are treated as zero.  Defaults to 1e-12.

    Returns:
        The minimum number of Trotter steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive.
        NotImplementedError: If *order* is not yet supported.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    if order != 1:
        raise NotImplementedError(
            f"Trotter step estimation for order {order} is not yet implemented. "
            "Only first-order (order=1) is currently supported."
        )
    if error_bound == "commutator":
        return trotter_steps_commutator(hamiltonian, time, target_accuracy, weight_threshold=weight_threshold)
    if error_bound == "naive":
        return trotter_steps_naive(hamiltonian, time, target_accuracy, weight_threshold=weight_threshold)
    raise ValueError(f"error_bound must be 'commutator' or 'naive', got {error_bound!r}.")


# =====================================================================
# First-order bounds
# =====================================================================


def trotter_steps_naive(
    hamiltonian: QubitHamiltonian,
    time: float,
    target_accuracy: float,
    *,
    weight_threshold: float = 1e-12,
) -> int:
    r"""Compute the number of first-order Trotter steps using the naive bound.

    The naive (triangle-inequality) bound is

    .. math::

        N = \left\lceil \frac{(\sum_j |\alpha_j|)^2 \, t^2}{\epsilon} \right\rceil

    where :math:`\sum_j |\alpha_j|` is the 1-norm of the Hamiltonian
    coefficients.

    Args:
        hamiltonian: The qubit Hamiltonian to simulate.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.
        weight_threshold: Absolute threshold below which coefficients
            are treated as zero.  Defaults to 1e-12.

    Returns:
        The minimum number of Trotter steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    real_terms = hamiltonian.get_real_coefficients(tolerance=weight_threshold)
    one_norm = sum(abs(coeff) for _, coeff in real_terms)
    return max(1, math.ceil(one_norm**2 * time**2 / target_accuracy))


def trotter_steps_commutator(
    hamiltonian: QubitHamiltonian,
    time: float,
    target_accuracy: float,
    *,
    weight_threshold: float = 1e-12,
) -> int:
    r"""Compute the number of first-order Trotter steps using the commutator bound.

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
        hamiltonian: The qubit Hamiltonian to simulate.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.
        weight_threshold: Absolute threshold below which coefficients
            are treated as zero.  Defaults to 1e-12.

    Returns:
        The minimum number of Trotter steps (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
    comm_bound = commutator_bound_first_order(hamiltonian, weight_threshold=weight_threshold)
    return max(1, math.ceil(comm_bound * time**2 / (2.0 * target_accuracy)))
