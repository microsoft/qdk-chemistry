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

from qdk_chemistry.data import PauliTermAccumulator

if TYPE_CHECKING:
    from collections.abc import Callable

    from qdk_chemistry.data import QubitHamiltonian

__all__: list[str] = [
    "commutator_bound_first_order",
    "commutator_bound_higher_order",
    "commutator_bound_second_order",
    "do_pauli_labels_commute",
    "do_pauli_labels_qw_commute",
    "do_pauli_maps_commute",
    "do_pauli_maps_qw_commute",
    "does_nested_commutator_vanish",
    "get_commutation_checker",
]


def _label_to_sparse_word(label: str) -> list[tuple[int, int]]:
    """Convert a Pauli string label to a ``SparsePauliWord``."""
    word = []
    for i, c in enumerate(label):
        if c in {"X", "Y", "Z"}:
            word.append((i, 1 if c == "X" else 2 if c == "Y" else 3))
        elif c != "I":
            raise ValueError(f"Invalid character {c!r} in Pauli label; expected 'I', 'X', 'Y', or 'Z'.")
    return word


def _sparse_word_to_label(word: list[tuple[int, int]], n_qubits: int) -> str:
    """Convert a ``SparsePauliWord`` back to a Pauli string label."""
    chars = ["I"] * n_qubits
    for q, p in word:
        chars[q] = "X" if p == 1 else "Y" if p == 2 else "Z"
    return "".join(chars)


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


def does_nested_commutator_vanish(*labels: str) -> bool:
    """Check whether a nested commutator of Pauli labels vanishes.

    Args:
        labels: A sequence of Pauli labels representing the nested commutator.

    Returns:
        ``True`` if the nested commutator vanishes, ``False`` otherwise.

    Raises:
        ValueError: If fewer than two Pauli labels are provided.

    """
    pauli_string_length = len(labels[0])
    if any(len(lbl) != pauli_string_length for lbl in labels):
        raise ValueError("All Pauli labels must have the same length.")

    if len(labels) < 2:
        raise ValueError("At least two Pauli labels are required for a commutator.")

    # Base case: [P_a, P_b] vanishes iff the two strings commute.
    if len(labels) == 2:
        return do_pauli_labels_commute(labels[0], labels[1])

    # Recursive case: [P_1, [P_2, ..., P_n]]
    # 1. Inner nested commutator vanishes => whole expression vanishes.
    if does_nested_commutator_vanish(*labels[1:]):
        return True

    # 2. Compute the product P_2 P_3 … P_n via sparse-word multiplication
    #    (proportional to the inner commutator when it is non-zero) and
    #    check whether P_1 commutes with it.
    word = _label_to_sparse_word(labels[1])
    for lbl in labels[2:]:
        _, word = PauliTermAccumulator.multiply_uncached(word, _label_to_sparse_word(lbl))
    inner_product = _sparse_word_to_label(word, pauli_string_length)
    return do_pauli_labels_commute(labels[0], inner_product)


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
            \frac{t^3}{12} \alpha_2
    Where :math:`\alpha_2` involves a sum of nested commutator norms (see Childs *et al.* (2021) for details).
    For Pauli strings the spectral norm of the commutator is

    * 0  if :math:`P_j` and :math:`P_k` commute, or
    * :math:`2 |\alpha_j| |\alpha_k|`  if they anticommute.

    This function returns
    :math:`\alpha_2`, so the user can multiply by :math:`t^{3} / 12` to get the per-step
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

    total_term1 = 0.0
    n = len(pauli_labels)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(i + 1, n):
                if not does_nested_commutator_vanish(pauli_labels[k], pauli_labels[j], pauli_labels[i]):
                    total_term1 += 2.0**2 * abs(coefficients[i]) * abs(coefficients[j]) * abs(coefficients[k])

    total_term2 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if not does_nested_commutator_vanish(pauli_labels[i], pauli_labels[i], pauli_labels[j]):
                total_term2 += 2.0**2 * abs(coefficients[i]) ** 2 * abs(coefficients[j])

    return total_term1 + 0.5 * total_term2


def commutator_bound_higher_order(
    hamiltonian: QubitHamiltonian,
    order: int,
    weight_threshold: float = 1e-12,
) -> float:
    r"""Compute the commutator bound :math:`\alpha` for arbitrary-order Trotter errors (bottom-up).

    This is functionally equivalent to :func:`commutator_bound_higher_order`
    but uses a bottom-up dynamic-programming strategy.  Instead of
    iterating over all :math:`n^{p+1}` index tuples independently, the
    algorithm builds the nested commutator level-by-level starting from
    the innermost pair.  At each level the intermediate Pauli-string
    products are cached and entries that share the same product label
    are grouped together, summing their coefficient contributions.
    This avoids redundant Pauli-string multiplications and enables
    early pruning of vanishing commutators.

    Args:
        hamiltonian: The qubit Hamiltonian for which to compute the bound.
        order: The order :math:`p` of the Trotter decomposition.
        weight_threshold: Absolute threshold for filtering small
            Hamiltonian coefficients.

    Returns:
        The commutator bound term :math:`\alpha` multiplying
        :math:`t^{p+1}` in Theorem 6 of Childs et al. (2021).

    """
    real_terms = hamiltonian.get_real_coefficients(tolerance=weight_threshold)
    pauli_labels = [label for label, _ in real_terms]
    coefficients = [coeff for _, coeff in real_terms]
    abs_coeffs = [abs(c) for c in coefficients]

    n = len(pauli_labels)
    if n == 0:
        return 0.0
    n_qubits = len(pauli_labels[0])

    # Pre-compute sparse-word representations for every term.
    sparse_words = [_label_to_sparse_word(label) for label in pauli_labels]

    # ---- Bottom-up construction ------------------------------------------------
    #
    # At each level the dictionary ``current`` maps a product Pauli-string
    # label to (accumulated_coefficient_sum, product_sparse_word).  Only
    # entries whose nested commutator is *non-vanishing* are kept.
    #
    # Level 0: innermost pairs [P_a, P_b]:
    #   For every ordered pair (a, b) where P_a and P_b anticommute,
    #   record the product P_a·P_b and accumulate |c_a|·|c_b|.
    #
    # Level k (k = 1 … order-1): prepend one more index:
    #   For each surviving product Q with coefficient sum S, and every
    #   index i where P_i anticommutes with Q, record the new product
    #   P_i·Q and accumulate |c_i|·S.
    #
    # After (order-1) prepend steps we have depth = order brackets and
    # (order+1) indices, matching the original brute-force iteration.
    # ---- Level 0: pairs --------------------------------------------------------
    current: dict[str, tuple[float, list[tuple[int, int]]]] = {}
    for a in range(n):
        for b in range(n):
            if not do_pauli_labels_commute(pauli_labels[a], pauli_labels[b]):
                _, product_word = PauliTermAccumulator.multiply_uncached(
                    sparse_words[a],
                    sparse_words[b],
                )
                product_label = _sparse_word_to_label(product_word, n_qubits)
                coeff_contrib = abs_coeffs[a] * abs_coeffs[b]
                if product_label in current:
                    prev_coeff, word = current[product_label]
                    current[product_label] = (prev_coeff + coeff_contrib, word)
                else:
                    current[product_label] = (coeff_contrib, product_word)

    # ---- Levels 1 … order-1: prepend one index each time -----------------------
    for _ in range(order - 1):
        next_level: dict[str, tuple[float, list[tuple[int, int]]]] = {}
        for product_label, (coeff_sum, product_word) in current.items():
            for i in range(n):
                if not do_pauli_labels_commute(pauli_labels[i], product_label):
                    _, new_word = PauliTermAccumulator.multiply_uncached(
                        sparse_words[i],
                        product_word,
                    )
                    new_label = _sparse_word_to_label(new_word, n_qubits)
                    new_coeff = abs_coeffs[i] * coeff_sum
                    if new_label in next_level:
                        prev_c, w = next_level[new_label]
                        next_level[new_label] = (prev_c + new_coeff, w)
                    else:
                        next_level[new_label] = (new_coeff, new_word)
        current = next_level

    # ---- Final summation -------------------------------------------------------
    total = sum(coeff_sum for coeff_sum, _ in current.values())
    return (2.0**order) * total
