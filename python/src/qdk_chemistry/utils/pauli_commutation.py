"""Pauli string commutation utilities.

This module provides reusable functions for checking commutativity of
Pauli operators and computing commutator norms.

Label-based functions (``do_pauli_labels_*``) operate on Pauli string
labels such as ``"XIZI"``.  Map-based functions (``do_pauli_maps_*``)
operate on sparse ``dict[int, str]`` qubit-index-to-Pauli-axis mappings.

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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.data import QubitHamiltonian

__all__: list[str] = [
    "commutator_bound_first_order",
    "do_pauli_labels_commute",
    "do_pauli_labels_qw_commute",
    "do_pauli_maps_commute",
    "do_pauli_maps_qw_commute",
    "does_nested_commutator_vanish",
    "get_commutation_checker",
]


def do_pauli_labels_commute(label_a: str, label_b: str) -> bool:
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
        >>> do_pauli_labels_commute("XI", "IX")
        True
        >>> do_pauli_labels_commute("XX", "YY")
        True
        >>> do_pauli_labels_commute("XY", "YX")
        True
        >>> do_pauli_labels_commute("XI", "YI")
        False

    """
    if len(label_a) != len(label_b):
        raise ValueError(f"Pauli labels must have the same length, got {len(label_a)} and {len(label_b)}.")
    anticommuting_count = 0
    for char_a, char_b in zip(label_a, label_b, strict=False):
        if char_a != "I" and char_b not in ("I", char_a):
            anticommuting_count += 1
    return anticommuting_count % 2 == 0


def do_pauli_labels_qw_commute(label_a: str, label_b: str) -> bool:
    r"""Check whether two Pauli strings qubit-wise commute.

    Two multi-qubit Pauli operators qubit-wise commute when every
    corresponding single-qubit pair commutes individually.  This is
    strictly stronger than general commutativity: qubit-wise commuting
    operators always commute, but the converse is not true
    (e.g. ``"XY"`` and ``"YX"`` commute globally but do **not**
    qubit-wise commute).

    The operators qubit-wise commute if and only if there are **no**
    qubit positions where both are non-identity and different.

    Args:
        label_a: Pauli string label (e.g. ``"XIZI"``).
        label_b: Pauli string label of the same length.

    Returns:
        ``True`` if the terms qubit-wise commute.

    Raises:
        ValueError: If the labels have different lengths.

    Examples:
        >>> do_pauli_labels_qw_commute("XI", "IX")
        True
        >>> do_pauli_labels_qw_commute("XI", "YI")
        False
        >>> do_pauli_labels_qw_commute("XY", "YX")   # commute, but NOT qw-commute
        False

    """
    if len(label_a) != len(label_b):
        raise ValueError(f"Pauli labels must have the same length, got {len(label_a)} and {len(label_b)}.")
    return not any(ca != "I" and cb not in ("I", ca) for ca, cb in zip(label_a, label_b, strict=False))


def do_pauli_maps_commute(a: dict[int, str], b: dict[int, str]) -> bool:
    """Check whether two Pauli terms commute (general/standard commutation).

    Two multi-qubit Pauli operators commute if and only if the number
    of qubit positions where both are non-identity *and* different is
    even.  This is weaker than qubit-wise commutation and allows
    larger merge groups.

    Args:
        a: First Pauli term mapping (qubit index → Pauli axis).
        b: Second Pauli term mapping (qubit index → Pauli axis).

    Returns:
        ``True`` if the terms commute.

    """
    anti_commuting = sum(1 for q in a if q in b and a[q] != b[q])
    return anti_commuting % 2 == 0


def do_pauli_maps_qw_commute(a: dict[int, str], b: dict[int, str]) -> bool:
    """Check whether two Pauli terms qubit-wise commute.

    Two multi-qubit Pauli operators qubit-wise commute when every
    corresponding single-qubit pair commutes individually.  This is
    strictly stronger than general commutativity: qubit-wise
    commuting operators always commute, but the converse is not true
    (e.g. ``{0: 'X', 1: 'Y'}`` and ``{0: 'Y', 1: 'X'}`` commute
    globally but do not qubit-wise commute).

    The operators qubit-wise commute if and only if there are **no**
    qubit positions where both are non-identity and different.

    Args:
        a: First Pauli term mapping (qubit index → Pauli axis).
        b: Second Pauli term mapping (qubit index → Pauli axis).

    Returns:
        ``True`` if the terms qubit-wise commute.

    """
    return not any(a[q] != b[q] for q in a if q in b)


def get_commutation_checker(
    commutation_type: str,
) -> Callable[[dict[int, str], dict[int, str]], bool]:
    """Return the commutation checker function for the given type.

    Args:
        commutation_type: ``"qubit_wise"`` or ``"general"``.

    Returns:
        A callable ``(a, b) -> bool`` that checks commutation of two
        Pauli term mappings.

    Raises:
        ValueError: If *commutation_type* is not recognised.

    """
    if commutation_type == "general":
        return do_pauli_maps_commute
    if commutation_type == "qubit_wise":
        return do_pauli_maps_qw_commute
    raise ValueError(f"Unknown commutation_type {commutation_type!r}; expected 'general' or 'qubit_wise'.")


def pauli_product_label(label_a: str, label_b: str) -> str:
    r"""Compute the Pauli label of the product :math:`P_a P_b` (up to phase).

    For each qubit position the single-qubit product rule is:

    * :math:`I \cdot Q = Q`
    * :math:`Q \cdot I = Q`
    * :math:`Q \cdot Q = I`
    * Otherwise the result is the third Pauli axis
      (e.g. :math:`X Y \to Z`, :math:`Y Z \to X`, :math:`Z X \to Y`).

    The global phase (:math:`\pm 1` or :math:`\pm i`) is discarded because
    it does not affect commutativity checks.
    """
    return "".join(
        b if a == "I" else a if b == "I" else "I" if a == b else ({"X", "Y", "Z"} - {a, b}).pop()
        for a, b in zip(label_a, label_b, strict=True)
    )


def does_nested_commutator_vanish(
    label_x: str,
    label_y: str,
    label_z: str,
) -> bool:
    r"""Determine whether :math:`[P_x,\,[P_y,\,P_z]]` vanishes.

    The nested commutator of three Pauli strings is zero when *either*
    of the following holds:

    1. The inner commutator :math:`[P_y, P_z]` vanishes
       (i.e. :math:`P_y` and :math:`P_z` commute), **or**
    2. The outer commutator :math:`[P_x, P_y P_z]` vanishes
       (i.e. :math:`P_x` commutes with the Pauli label of
       :math:`P_y P_z`).

    Note that the Pauli label of :math:`[P_y, P_z]` is proportional
    to :math:`P_y P_z` (up to a scalar), so commutation of :math:`P_x`
    with :math:`P_y P_z` is equivalent to commutation with the full
    commutator operator.

    Args:
        label_x: Pauli string label for :math:`P_x`.
        label_y: Pauli string label for :math:`P_y`.
        label_z: Pauli string label for :math:`P_z`.

    Returns:
        ``True`` if :math:`[P_x, [P_y, P_z]] = 0`, ``False`` otherwise.

    Raises:
        ValueError: If the labels have different lengths.

    Examples:
        >>> does_nested_commutator_vanish("XI", "IX", "II")
        True
        >>> does_nested_commutator_vanish("XI", "YI", "ZI")
        False
        >>> does_nested_commutator_vanish("IX", "XI", "YI")
        True

    """
    # Inner commutator [P_y, P_z] vanishes => whole thing vanishes
    if do_pauli_labels_commute(label_y, label_z):
        return True
    # Compute Pauli label of P_y P_z (∝ [P_y, P_z])
    inner_label = pauli_product_label(label_y, label_z)

    # Outer commutator [P_x, inner] vanishes?
    return do_pauli_labels_commute(label_x, inner_label)


def commutator_bound_first_order(
    hamiltonian: QubitHamiltonian,
    weight_threshold: float = 1e-12,
) -> float:
    r"""Compute the first-order Trotter commutator bound.

    For a Hamiltonian :math:`H = \sum_j \alpha_j P_j` the first-order
    (Lie-Trotter) product formula has error bounded by

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
        hamiltonian: The qubit Hamiltonian whose terms to analyse.
        weight_threshold: Absolute threshold below which coefficients are discarded.

    Returns:
        The sum of commutator norms over all unique pairs.

    """
    real_terms = hamiltonian.get_real_coefficients(tolerance=weight_threshold)
    pauli_labels = [label for label, _ in real_terms]
    coefficients = [coeff for _, coeff in real_terms]

    total = 0.0
    n = len(pauli_labels)
    for j in range(n):
        for k in range(j + 1, n):
            if not do_pauli_labels_commute(pauli_labels[j], pauli_labels[k]):
                total += 2.0 * abs(coefficients[j]) * abs(coefficients[k])
    return total


def commutator_bound_second_order(
    hamiltonian: QubitHamiltonian,
    weight_threshold: float = 1e-12,
) -> float:
    r"""Compute the second-order Trotter commutator bound.

    For a Hamiltonian :math:`H = \sum_j \alpha_j P_j` the second-order
    (Lie-Trotter) product formula has error bounded by

    .. math::

        \lVert U(t) - S_2(t) \rVert \le
            \frac{t^3}{3!} \sum_{j < k < l}
            \lVert \lvert [\alpha_j P_i,\, [\alpha_j P_j,\, \alpha_k P_k] \rVert ] \rVert

    For Pauli strings the spectral norm of the commutator is

    * 0  if :math:`P_j` and :math:`P_k` commute, or
    * :math:`2 |\alpha_j| |\alpha_k|`  if they anticommute.

    This function returns
    :math:`\sum_{j < k < l}
            \lVert \lvert [\alpha_j P_i,\, [\alpha_j P_j,\, \alpha_k P_k] \rVert ] \rVert`,
    so the user can multiply by :math:`t^{3} / (3! * N)` to get the per-step
    error.

    Args:
        hamiltonian: The qubit Hamiltonian whose terms to analyse.
        weight_threshold: Absolute threshold below which coefficients are discarded.

    Returns:
        The sum of commutator norms over all triples without redundancies.

    """
    real_terms = hamiltonian.get_real_coefficients(tolerance=weight_threshold)
    pauli_labels = [label for label, _ in real_terms]
    coefficients = [coeff for _, coeff in real_terms]

    total = 0.0
    n = len(pauli_labels)
    for i in range(n):
        for j in range(n):
            for k in range(j + 1, n):
                if not does_nested_commutator_vanish(pauli_labels[i], pauli_labels[j], pauli_labels[k]):
                    total += 2.0**2 * abs(coefficients[i]) * abs(coefficients[j]) * abs(coefficients[k])
    return total
