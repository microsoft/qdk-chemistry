r"""qDRIFT sample-count estimation for accuracy-aware parameterization.

Provides the Campbell (2019) bound for the minimum number of qDRIFT
samples required to achieve a target accuracy:

.. math::

    N \geq \left\lceil \frac{2 \lambda^2 t^2}{\epsilon} \right\rceil,
    \quad \lambda = \sum_j |\alpha_j|.

References:
    Campbell, E. (2019). Random Compiler for Fast Hamiltonian Simulation.
    Physical Review Letters, 123(7), 070503.
    https://arxiv.org/abs/1811.08017

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdk_chemistry.data import QubitHamiltonian

__all__: list[str] = ["qdrift_samples_campbell"]


def qdrift_samples_campbell(
    hamiltonian: QubitHamiltonian,
    time: float,
    target_accuracy: float,
    *,
    weight_threshold: float = 1e-12,
) -> int:
    r"""Compute the number of qDRIFT samples using the Campbell (2019) bound.

    .. math::

        N = \left\lceil \frac{2 \lambda^2 t^2}{\epsilon} \right\rceil,
        \quad \lambda = \sum_j |\alpha_j|

    where the sum runs over the real coefficients of *hamiltonian* whose
    magnitudes exceed *weight_threshold*.

    Args:
        hamiltonian: The qubit Hamiltonian to simulate.
        time: The total evolution time *t*.
        target_accuracy: The target accuracy :math:`\epsilon > 0`.
        weight_threshold: Absolute threshold below which coefficients are discarded.

    Returns:
        The minimum number of qDRIFT samples (at least 1).

    Raises:
        ValueError: If ``target_accuracy`` is not positive.

    """
    if target_accuracy <= 0:
        raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")

    real_terms = hamiltonian.get_real_coefficients(tolerance=weight_threshold)
    lambda_norm = sum(abs(coeff) for _, coeff in real_terms)

    if lambda_norm == 0.0 or time == 0.0:
        return 1

    return max(1, math.ceil(2.0 * (lambda_norm * time) ** 2 / target_accuracy))
