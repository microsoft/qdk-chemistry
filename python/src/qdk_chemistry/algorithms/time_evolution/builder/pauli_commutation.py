"""Pauli string commutation utilities for Trotter error bounds.

This module provides functions to check whether Pauli strings commute
and to compute commutator norms used in tighter Trotter error bounds.

References:
    Childs, A. M., et al. "Toward the first quantum simulation with
    quantum speedup." *Proceedings of the National Academy of Sciences*
    115.38 (2018): 9456-9461.

    Childs, A. M., et al. "Theory of Trotter Error with Commutator
    Scaling." *Physical Review X* 11.1 (2021): 011020.
    https://arxiv.org/abs/1912.08854

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

__all__: list[str] = [
    "do_pauli_strings_commute",
    "commutator_bound_first_order",
]


def do_pauli_strings_commute(label_a: str, label_b: str) -> bool:
    r"""Check whether two Pauli strings commute.

    Two multi-qubit Pauli strings :math:`P_a` and :math:`P_b` commute if and
    only if the number of qubit positions where the two single-qubit Pauli
    operators are *both* non-identity and *different* from each other is even.

    The labels use the Qiskit / ``SparsePauliOp`` convention: the rightmost
    character corresponds to qubit 0 (little-endian bit ordering).

    Args:
        label_a: Pauli string label (e.g. ``"XIZI"``).
        label_b: Pauli string label of the same length (e.g. ``"YZXI"``).

    Returns:
        ``True`` if the two Pauli strings commute, ``False`` otherwise.

    Raises:
        ValueError: If the labels have different lengths.

    Examples:
        >>> do_pauli_strings_commute("XI", "IX")
        True
        >>> do_pauli_strings_commute("XX", "YY")
        True
        >>> do_pauli_strings_commute("XY", "YX")
        True
        >>> do_pauli_strings_commute("XI", "YI")
        False

    """
    if len(label_a) != len(label_b):
        raise ValueError(
            f"Pauli labels must have the same length, got {len(label_a)} and {len(label_b)}."
        )
    anticommuting_count = 0
    for char_a, char_b in zip(label_a, label_b):
        if char_a != "I" and char_b != "I" and char_a != char_b:
            anticommuting_count += 1
    return anticommuting_count % 2 == 0


def commutator_bound_first_order(
    pauli_labels: list[str],
    coefficients: list[float],
) -> float:
    r"""Compute the first-order Trotter commutator bound.

    For a Hamiltonian :math:`H = \sum_j \alpha_j P_j` the first-order
    (Lie–Trotter) product formula has error bounded by

    .. math::

        \lVert U(t) - S_1(t) \rVert \le
            \frac{t^2}{2} \sum_{j < k}
            \lVert [\alpha_j P_j,\, \alpha_k P_k] \rVert

    For Pauli strings the spectral norm of the commutator is

    * 0  if :math:`P_j` and :math:`P_k` commute, or
    * :math:`2 |\alpha_j| |\alpha_k|`  if they anticommute.

    This function returns
    :math:`\sum_{j < k} \lVert [\alpha_j P_j, \alpha_k P_k] \rVert`,
    so the user can multiply by :math:`t^{2} / (2N)` to get the per-step
    error.

    Args:
        pauli_labels: List of Pauli string labels for each term.
        coefficients: Corresponding real coefficients :math:`\alpha_j`.

    Returns:
        The sum of commutator norms over all unique pairs.

    Raises:
        ValueError: If the number of labels and coefficients differ.

    """
    if len(pauli_labels) != len(coefficients):
        raise ValueError(
            f"Number of Pauli labels ({len(pauli_labels)}) and "
            f"coefficients ({len(coefficients)}) must match."
        )
    total = 0.0
    n = len(pauli_labels)
    for j in range(n):
        for k in range(j + 1, n):
            if not do_pauli_strings_commute(pauli_labels[j], pauli_labels[k]):
                total += 2.0 * abs(coefficients[j]) * abs(coefficients[k])
    return total
