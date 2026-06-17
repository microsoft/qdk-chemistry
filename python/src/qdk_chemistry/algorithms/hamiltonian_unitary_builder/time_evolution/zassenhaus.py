r"""QDK/Chemistry implementation of the Zassenhaus product-formula Builder.

The Zassenhaus formula is the dual of Baker-Campbell-Hausdorff: it expands a
single exponential of a sum into an ordered product of exponentials in which
low-order nested-commutator corrections appear as *explicit* factors, rather
than being controlled implicitly by a step count (as in Trotter-Suzuki):

    exp(X + Y) = exp(X) exp(Y) exp(C2) exp(C3) exp(C4) ...

with, in terms of X = -i t A and Y = -i t B (A, B Hermitian Hamiltonian groups),

    C2 = -1/2 [X, Y]
    C3 =  1/3 [Y, [X, Y]] + 1/6 [X, [X, Y]]
    C4 = -1/24 [X, [X, [X, Y]]] - 1/8 [Y, [X, [X, Y]]] - 1/8 [Y, [Y, [X, Y]]]

Truncating after C_p yields an operator-norm error of O(t^{p+1}).

The Hamiltonian is split into K internally-commuting groups and the builder uses
the *multi-operator* Zassenhaus formula directly --
``exp(sum_i X_i) = exp(X_0) ... exp(X_{K-1}) exp(C_2) ... exp(C_p)`` with one
genuine K-operator exponent C_n per degree. The C_n are generated symbolically for
any order (rather than hard-coded), so the two-operator C2/C3/C4 above are just the
familiar low-order instances; see :func:`_zassenhaus_word_exponents`.

References:
    Wilcox, R. M. "Exponential operators and parameter differentiation in
    quantum physics." J. Math. Phys. 8, 962 (1967).

    Casas, F., Murua, A., Nadinic, M. "Efficient computation of the Zassenhaus
    formula." Comput. Phys. Commun. 183, 2386 (2012). arXiv:1204.0389.

    Childs, A. M., et al. "Theory of Trotter Error with Commutator Scaling."
    Phys. Rev. X 11, 011020 (2021). arXiv:1912.08854.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from fractions import Fraction
from functools import cache

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder, TimeEvolutionSettings
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter
from qdk_chemistry.data import (
    FlatPartition,
    LayeredPartition,
    QubitHamiltonian,
    TermPartition,
    UnitaryRepresentation,
)
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.pauli_commutation import (
    commutator,
    commutator_bound_higher_order,
    do_pauli_labels_commute,
)

__all__: list[str] = ["Zassenhaus", "ZassenhausSettings"]

# The symbolic exponent generator builds ~num_generators^order words, so its cost
# grows exponentially in the order. Warn before attempting an intractable size --
# reached only for Hamiltonians with many commuting groups (large molecules) at high
# order. No hardcoded maximum order is needed: corrections are generated and flattened
# for any order (see Zassenhaus._correction_trotter_order), and this guardrail flags
# the rare case where that becomes too expensive.
_WORD_SERIES_WARN_THRESHOLD = 100_000


# ----------------------------------------------------------------------------------- #
# Universal Zassenhaus exponents (symbolic, operator-independent, cached)
#
# The multi-operator Zassenhaus formula factorises the exponential of a sum of K
# operators X_0, ..., X_{K-1} as
#
#     exp(X_0 + ... + X_{K-1}) = exp(X_0) ... exp(X_{K-1}) exp(C_2) exp(C_3) ...
#
# where each C_n is a homogeneous degree-n element of the free Lie algebra on the
# generators. C_n is computed once per (K, order) as a truncated noncommutative
# word series with exact rational coefficients, then turned into nested commutators
# of the actual operators via the Dynkin map (see Zassenhaus._evaluate_exponent).
# This generalises the recursive two-operator scheme to a single correction block
# per degree (textbook form: fewer, non-redundant factors) at any order.
# ----------------------------------------------------------------------------------- #


def _series_mul(a: dict, b: dict, max_degree: int) -> dict:
    """Multiply two truncated word series (words = tuples of generator indices)."""
    out: dict = {}
    for w1, c1 in a.items():
        for w2, c2 in b.items():
            if len(w1) + len(w2) > max_degree:
                continue
            w = w1 + w2
            out[w] = out.get(w, Fraction(0)) + c1 * c2
    return {w: c for w, c in out.items() if c}


def _series_exp(generator: dict, max_degree: int) -> dict:
    """Exponential of a word series with no degree-0 term, truncated to ``max_degree``."""
    result = {(): Fraction(1)}
    term = {(): Fraction(1)}
    for k in range(1, max_degree + 1):
        term = {w: c * Fraction(1, k) for w, c in _series_mul(term, generator, max_degree).items()}
        if not term:
            break
        for w, c in term.items():
            result[w] = result.get(w, Fraction(0)) + c
    return {w: c for w, c in result.items() if c}


def _series_log(series: dict, max_degree: int) -> dict:
    """Logarithm of ``1 + r`` (r the non-constant part), truncated to ``max_degree``."""
    remainder = {w: c for w, c in series.items() if w != ()}
    result: dict = {}
    power = {(): Fraction(1)}
    for k in range(1, max_degree + 1):
        power = _series_mul(power, remainder, max_degree)
        if not power:
            break
        sign = Fraction((-1) ** (k + 1), k)
        for w, c in power.items():
            result[w] = result.get(w, Fraction(0)) + sign * c
    return {w: c for w, c in result.items() if c}


def _warn_if_series_large(num_generators: int, order: int) -> None:
    """Warn when the word series (~``num_generators^order`` words) is large enough to be slow."""
    estimated_words = num_generators**order
    if estimated_words > _WORD_SERIES_WARN_THRESHOLD:
        Logger.warn(
            f"Zassenhaus exponent generation for {num_generators} commuting groups at order "
            f"{order} builds ~{estimated_words:.1e} terms and may be slow or memory-intensive; "
            f"consider a lower order, coarser grouping, or disabling automatic accuracy estimation."
        )


@cache
def _zassenhaus_word_exponents(num_generators: int, order: int) -> tuple:
    """Return the Zassenhaus exponents C_2..C_order over ``num_generators`` symbols.

    The result is a tuple whose entry ``n - 2`` is the word expansion of C_n as a
    tuple of ``(word, Fraction)`` pairs, where ``word`` is a tuple of generator
    indices. Depends only on ``(num_generators, order)`` so it is cached.

    Peeling: with X_i the i-th generator and S their sum,
    ``R = exp(-X_{K-1}) ... exp(-X_0) exp(S)`` has lowest degree 2, and
    ``C_n`` is the degree-n part of ``log(exp(-C_{n-1}) ... exp(-C_2) R)``.
    """
    _warn_if_series_large(num_generators, order)
    generators = [{(i,): Fraction(1)} for i in range(num_generators)]
    total: dict = {}
    for gen in generators:
        for w, c in gen.items():
            total[w] = total.get(w, Fraction(0)) + c

    remainder = _series_exp(total, order)
    for gen in generators:
        neg = {w: -c for w, c in gen.items()}
        remainder = _series_mul(_series_exp(neg, order), remainder, order)

    exponents: list = []
    for n in range(2, order + 1):
        log_remainder = _series_log(remainder, order)
        c_n = {w: c for w, c in log_remainder.items() if len(w) == n}
        exponents.append(tuple(sorted(c_n.items())))
        neg_c_n = {w: -c for w, c in c_n.items()}
        remainder = _series_mul(_series_exp(neg_c_n, order), remainder, order)
    return tuple(exponents)


class ZassenhausSettings(TimeEvolutionSettings):
    """Settings for the Zassenhaus product-formula builder."""

    def __init__(self):
        """Initialize ZassenhausSettings with default values.

        Attributes:
            order: Expansion order p (>= 2); corrections C_2..C_p are included so the
                truncation error scales as O(t^{p+1}). Order 1 (no corrections) falls
                back to the first-order Trotter builder.
            num_divisions: Number of time divisions N. The full Zassenhaus
                product approximates exp(-iH t/N) and is repeated N times.
            target_accuracy: Target operator-norm accuracy for automatic N
                (0.0 disables it). When set, N is raised so the leading
                truncation error ``||C_{p+1}|| t^{p+1} / N^p`` is below it.
            weight_threshold: Absolute threshold for filtering small coefficients.

        """
        super().__init__()
        self._set_default("order", "int", 2, "The Zassenhaus expansion order p (>= 2; 1 falls back to Trotter).")
        self._set_default("num_divisions", "int", 1, "Number of time divisions N (>= 1).")
        self._set_default(
            "target_accuracy",
            "double",
            0.0,
            "Target accuracy for automatic division-count estimation (0.0 means disabled).",
        )
        self._set_default(
            "weight_threshold", "float", 1e-12, "The absolute threshold for filtering small coefficients."
        )


class Zassenhaus(TimeEvolutionBuilder):
    """Zassenhaus product-formula time-evolution builder.

    Note:
        ``num_divisions`` N may be set explicitly (default 1) or estimated
        automatically from ``target_accuracy`` (Zassenhaus error scales as
        ``||C_{p+1}|| t^{p+1} / N^p``); when both are given the larger N wins.
        The expansion ``order`` is always user-provided (1 falls back to Trotter).

        Remaining gate-count optimisations are left as future work: exploiting
        disjoint-support (layer) parallelism instead of flattening layers, and
        merging redundant terms at division / Suzuki-copy boundaries.

    """

    def __init__(
        self,
        order: int = 2,
        *,
        time: float = 0.0,
        num_divisions: int = 1,
        target_accuracy: float = 0.0,
        weight_threshold: float = 1e-12,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        r"""Initialize the Zassenhaus builder.

        Args:
            order: Expansion order p >= 2; order 1 falls back to first-order Trotter. Defaults to 2.
            time: The evolution time. Defaults to 0.0.
            num_divisions: Number of time divisions N (>= 1). Defaults to 1.
            target_accuracy: Target accuracy for automatic N estimation. Use 0.0 (default) to disable.
            weight_threshold: Threshold for filtering small coefficients. Defaults to 1e-12.
            power: The power to raise the unitary to. Defaults to 1.
            power_strategy: Strategy for U^power: ``"rescale"`` or ``"repeat"`` (default).

        Raises:
            ValueError: If ``num_divisions`` is less than 1.

        """
        super().__init__()
        if num_divisions < 1:
            raise ValueError(f"num_divisions must be >= 1, got {num_divisions}.")
        self._settings = ZassenhausSettings()
        self._settings.set("time", time)
        self._settings.set("power", power)
        self._settings.set("power_strategy", power_strategy)
        self._settings.set("order", order)
        self._settings.set("num_divisions", num_divisions)
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("weight_threshold", weight_threshold)

    # ------------------------------------------------------------------ #
    # Entry point (mirrors Trotter._run_impl -> _trotter)
    # ------------------------------------------------------------------ #
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct the unitary representation using the Zassenhaus expansion."""
        order = self._settings.get("order")
        if order < 1:
            raise ValueError(f"Zassenhaus order must be >= 1 (1 falls back to Trotter); got {order}.")
        if order == 1:
            # Order 1 carries no commutator corrections (e^X e^Y), which is
            # exactly the first-order Trotter product -- delegate to it.
            return self._trotter_fallback(qubit_hamiltonian)

        effective_time, power_repetitions = self._resolve_power()
        weight_threshold = self._settings.get("weight_threshold")

        if not qubit_hamiltonian.is_hermitian(tolerance=weight_threshold):
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, effective_time, order)
        delta = effective_time / num_divisions

        # One Zassenhaus product approximating exp(-i H delta).
        step_terms = self._decompose_zassenhaus_step(qubit_hamiltonian, time=delta, order=order, atol=weight_threshold)

        container = PauliProductFormulaContainer(
            step_terms=step_terms,
            step_reps=num_divisions * power_repetitions,
            num_qubits=qubit_hamiltonian.num_qubits,
        )
        return UnitaryRepresentation(container=container)

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitHamiltonian, time: float, order: int) -> int:
        """Determine the number of Zassenhaus divisions N.

        Uses ``max(num_divisions, auto)`` when ``target_accuracy`` is set, where
        ``auto`` is :meth:`_estimate_divisions`; otherwise the explicit
        ``num_divisions`` (>= 1, validated at construction).
        """
        manual = self._settings.get("num_divisions")
        target_accuracy = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0:
            return manual
        estimated = self._estimate_divisions(qubit_hamiltonian, time, order, target_accuracy)
        return max(manual, estimated)

    def _estimate_divisions(
        self, qubit_hamiltonian: QubitHamiltonian, time: float, order: int, target_accuracy: float
    ) -> int:
        r"""Estimate N so the leading Zassenhaus truncation error meets the target.

        An order-p Zassenhaus step has leading error ``||C_{p+1}|| t^{p+1} / N^p``,
        the same scaling as an order-p Trotter step. We reuse the shared
        commutator bound :func:`~qdk_chemistry.utils.pauli_commutation.commutator_bound_higher_order`
        (the alpha multiplying ``t^{p+1}`` in Theorem 6 of Childs et al. 2021, also
        used by the Trotter step-count estimators) and solve for N with the same
        closed form:

            N = ceil( alpha^{1/p} * t^{1 + 1/p} / target_accuracy^{1/p} ).

        The bound is shared with -- not re-derived from -- the Trotter machinery;
        unlike ``trotter_steps_commutator`` it is applied at every order p (the
        Trotter wrapper only supports orders 1, 2 and even, whereas Zassenhaus also
        allows odd orders). Like all such bounds it is a conservative estimate, not
        an exact error guarantee.
        """
        if time == 0.0:
            return 1
        weight_threshold = self._settings.get("weight_threshold")
        alpha = commutator_bound_higher_order(qubit_hamiltonian, order=order, weight_threshold=weight_threshold)
        if alpha <= 0.0:
            # No inter-term commutators: the step is already exact.
            return 1

        n_float = alpha ** (1.0 / order) * abs(time) ** (1.0 + 1.0 / order) / target_accuracy ** (1.0 / order)
        return max(1, int(np.ceil(n_float)))

    def _trotter_fallback(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Delegate order 1 to the first-order Trotter builder.

        The Zassenhaus product at order 1 is just ``e^X e^Y`` with no commutator
        corrections, identical to the first-order Trotter formula. The current
        settings (time, divisions, power, threshold) are forwarded.
        """
        trotter = Trotter(
            order=1,
            time=self._settings.get("time"),
            num_divisions=self._settings.get("num_divisions"),
            weight_threshold=self._settings.get("weight_threshold"),
            power=self._settings.get("power"),
            power_strategy=self._settings.get("power_strategy"),
        )
        return trotter.run(qubit_hamiltonian)

    # ------------------------------------------------------------------ #
    # Core approximation logic (this is the Trotter-analogue swap point)
    # ------------------------------------------------------------------ #
    def _decompose_zassenhaus_step(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time: float,
        order: int,
        *,
        atol: float = 1e-12,
    ) -> list[ExponentiatedPauliTerm]:
        r"""Decompose a single Zassenhaus step into exponentiated Pauli terms.

        Partitions H into internally-commuting groups G_0, ..., G_{K-1} and builds
        the multi-operator Zassenhaus product (see :meth:`_multi_operator_factors`)

            exp(-i t H) = exp(X_0) ... exp(X_{K-1}) exp(C_2) ... exp(C_order),

        then flattens each anti-Hermitian factor into ``ExponentiatedPauliTerm``s.
        """
        groups = self._commuting_groups(qubit_hamiltonian)
        if not groups:
            Logger.warn("No Pauli terms above the tolerance; returning empty term list.")
            return []

        # Ordered (leftmost-first) list of (factor, degree) pairs; degree 0 marks a
        # group (leaf) factor, degree n >= 2 marks the correction C_n.
        factors = self._multi_operator_factors(groups, time, order)

        # Each factor exp(F) is realized by a Trotter-Suzuki product of its Pauli
        # terms. A degree-n factor has ||F|| ~ t^n, so a Suzuki order-s product
        # has error O(t^{n(s+1)}); we pick the smallest s in {1, 2, 4, 6, ...}
        # with n(s+1) >= order + 1 so the flattening never dominates the O(t^{order+1})
        # truncation (see :meth:`_correction_trotter_order`). Group (leaf) factors
        # commute internally, so order 1 is already exact for them.
        terms: list[ExponentiatedPauliTerm] = []
        for factor, degree in factors:
            suzuki_order = self._correction_trotter_order(degree, order)
            terms.extend(self._exponentiate_factor_suzuki(factor, suzuki_order, atol=atol))

        # The mapper applies step_terms[0] first (rightmost operator), so reverse
        # to realize the product above: the leftmost factor is applied last.
        terms.reverse()
        return terms

    def _multi_operator_factors(
        self, groups: list[QubitHamiltonian], time: float, order: int
    ) -> list[tuple[QubitHamiltonian, int]]:
        r"""Multi-operator (textbook) Zassenhaus product over all K groups at once.

        With X_i = -i t G_i for the internally-commuting groups G_0, ..., G_{K-1},

            exp(sum_i X_i) = exp(X_0) ... exp(X_{K-1}) exp(C_2) ... exp(C_order),

        where each C_n is the genuine K-operator Zassenhaus exponent of degree n --
        a single correction block per degree, with no recursive re-expansion and no
        redundant per-level corrections. Returned as an ordered (leftmost-first)
        list of ``(factor, degree)`` pairs: degree 0 for the exact group factors,
        degree n for the correction C_n.
        """
        ops = [(-1j * time) * group for group in groups]
        factors: list[tuple[QubitHamiltonian, int]] = [(op, 0) for op in ops]
        if len(ops) < 2:
            # A single commuting group is exponentiated exactly; no corrections.
            return factors

        word_exponents = _zassenhaus_word_exponents(len(ops), order)
        for degree in range(2, order + 1):
            c_n = self._evaluate_exponent(word_exponents[degree - 2], degree, ops)
            factors.append((c_n, degree))
        return factors

    def _evaluate_exponent(self, word_terms: tuple, degree: int, ops: list[QubitHamiltonian]) -> QubitHamiltonian:
        r"""Evaluate a symbolic Zassenhaus exponent on the actual operators.

        ``word_terms`` is the cached word expansion of C_n from
        :func:`_zassenhaus_word_exponents`. By the Dynkin-Specht-Wever theorem a
        homogeneous degree-n Lie element equals ``1/n`` times the right-nested
        bracketing of its words, so

            C_n = (1 / n) sum_w c_w [g_{w_0}, [g_{w_1}, [ ..., g_{w_{n-1}}]]],

        evaluated with the QDK-native ``commutator`` (Qiskit-free). Nested
        commutators of (anti-Hermitian) Pauli operators are again Pauli operators,
        so the result is an anti-Hermitian ``QubitHamiltonian``.
        """
        inverse_degree = 1.0 / degree
        total: QubitHamiltonian | None = None
        for word, coeff in word_terms:
            nested = ops[word[-1]]
            for index in reversed(word[:-1]):
                nested = commutator(ops[index], nested)
            contribution = (float(coeff) * inverse_degree) * nested
            total = contribution if total is None else total + contribution
        return total

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _commuting_groups(self, qubit_hamiltonian: QubitHamiltonian) -> list[QubitHamiltonian]:
        r"""Partition H into internally-commuting Hermitian groups.

        A correct Zassenhaus expansion requires each ``exp(-i t G_i)`` to be
        exact, i.e. every group must be a set of mutually-commuting Pauli terms.

        Strategy: consume ``qubit_hamiltonian.term_partition`` when it is present
        *and* every group is internally commuting (exploits user-provided sparse
        structure, e.g. even/odd lattice bonds). Otherwise group greedily with
        :class:`FullCommutingTermGrouper`.
        """
        partition = qubit_hamiltonian.term_partition
        if partition is not None:
            groups = self._materialize_groups(qubit_hamiltonian, partition)
            if all(self._is_internally_commuting(g) for g in groups):
                return groups
            Logger.debug("Zassenhaus: provided term_partition is not internally commuting; re-grouping.")

        # Lazy import avoids a module-load cycle (registry imports this builder).
        from qdk_chemistry.algorithms.term_grouper.commuting import FullCommutingTermGrouper  # noqa: PLC0415

        grouped = FullCommutingTermGrouper().run(qubit_hamiltonian)
        return self._materialize_groups(grouped, grouped.term_partition)

    def _materialize_groups(
        self, qubit_hamiltonian: QubitHamiltonian, partition: TermPartition
    ) -> list[QubitHamiltonian]:
        """Materialise an index-based ``TermPartition`` into sub-Hamiltonians.

        ``FlatPartition`` groups are used directly; ``LayeredPartition`` groups
        are flattened across layers into a single commuting set per group.
        """
        labels = qubit_hamiltonian.pauli_strings
        coeffs = np.asarray(qubit_hamiltonian.coefficients)
        encoding = qubit_hamiltonian.encoding
        fmo = qubit_hamiltonian.fermion_mode_order

        if isinstance(partition, LayeredPartition):
            index_groups = [tuple(i for layer in group for i in layer) for group in partition.groups]
        elif isinstance(partition, FlatPartition):
            index_groups = [tuple(group) for group in partition.groups]
        else:
            raise TypeError(
                f"Unsupported TermPartition subtype: {type(partition).__name__}. "
                "Expected FlatPartition or LayeredPartition."
            )

        groups: list[QubitHamiltonian] = []
        for indices in index_groups:
            if not indices:
                continue
            groups.append(
                QubitHamiltonian(
                    pauli_strings=[labels[i] for i in indices],
                    coefficients=coeffs[list(indices)],
                    encoding=encoding,
                    fermion_mode_order=fmo,
                )
            )
        return groups

    @staticmethod
    def _is_internally_commuting(group: QubitHamiltonian) -> bool:
        """Return ``True`` if every pair of Pauli terms in *group* commutes."""
        labels = group.pauli_strings
        return all(
            do_pauli_labels_commute(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))
        )

    def _exponentiate_factor(self, factor: QubitHamiltonian, *, atol: float = 1e-12) -> list[ExponentiatedPauliTerm]:
        r"""Flatten an anti-Hermitian factor F = sum f_j P_j into exp terms.

        Since F is anti-Hermitian, f_j is purely imaginary, and
        exp(F) = prod_j exp(f_j P_j) = prod_j exp(-i (-Im f_j) P_j),
        i.e. the ``ExponentiatedPauliTerm`` angle is ``-Im(f_j)``.

        Identity terms are kept as global-phase rotations (empty Pauli map),
        matching the Trotter builder: dropping them corrupts controlled-U
        phases in PhaseEstimation (a leading ``I...I`` term, as in H2/STO-3G,
        carries a real physical phase).

        This is the first-order (plain) Trotter product. A factor's terms
        generally do NOT commute, so it carries an O(||F||^2) = O(t^{2n}) error
        for a degree-n factor; the caller (:meth:`_decompose_zassenhaus_step`)
        escalates to a higher Suzuki order when that is not high enough. Leaf
        (group) factors commute internally, so this product is exact for them.

        Like Pauli terms are summed before emission. ``QubitHamiltonian``
        addition concatenates rather than merging, and the commutator sums in
        :meth:`_correction_factors` can produce the same Pauli several times;
        since ``exp(a P) exp(b P) = exp((a + b) P)`` exactly (P commutes with
        itself), collecting them removes redundant rotations with no reordering
        and no loss of accuracy.
        """
        merged: dict[str, complex] = {}
        for label, coeff in zip(factor.pauli_strings, factor.coefficients, strict=True):
            merged[label] = merged.get(label, 0j) + complex(coeff)

        terms: list[ExponentiatedPauliTerm] = []
        for label, c in merged.items():
            if abs(c) < atol:
                continue
            if abs(c.real) > atol:
                # Anti-Hermitian factor should have ~zero real part. Surface
                # violations instead of silently dropping them (Copilot review
                # flagged this exact failure mode on PR #508).
                raise ValueError(f"Expected anti-Hermitian factor; term {label!r} has real coeff {c.real}.")
            angle = -c.imag
            mapping = self._pauli_label_to_map(label)  # empty for identity -> global phase
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))
        return terms

    def _exponentiate_factor_symmetric(
        self, factor: QubitHamiltonian, *, atol: float = 1e-12
    ) -> list[ExponentiatedPauliTerm]:
        r"""Symmetric (Strang) flattening of an anti-Hermitian factor.

        Emits half-angle terms forward then in reverse:
        ``exp(f_1/2) ... exp(f_m/2) exp(f_m/2) ... exp(f_1/2)``, whose error
        versus ``exp(F)`` is O(||F||^3). For a degree-n factor (||F|| ~ t^n)
        this is O(t^{3n}), one symmetric order tighter than the plain product.
        The emitted block is a palindrome, so the outer ``terms.reverse()`` in
        :meth:`_decompose_zassenhaus_step` leaves its internal order intact.

        The two identical centre terms ``exp(f_m/2) exp(f_m/2) = exp(f_m)`` are
        merged into a single full-angle rotation, saving one gate per factor
        (and collapsing a single-term factor to one rotation).
        """
        half = self._exponentiate_factor((0.5) * factor, atol=atol)
        if not half:
            return []
        *body, last = half
        centre = ExponentiatedPauliTerm(pauli_term=last.pauli_term, angle=2.0 * last.angle)
        return [*body, centre, *body[::-1]]

    def _exponentiate_factor_suzuki(
        self, factor: QubitHamiltonian, suzuki_order: int, *, atol: float = 1e-12
    ) -> list[ExponentiatedPauliTerm]:
        r"""Flatten ``exp(F)`` with a Trotter-Suzuki product formula of order ``suzuki_order``.

        Order 1 is the plain product (:meth:`_exponentiate_factor`), order 2 the
        symmetric Strang product (:meth:`_exponentiate_factor_symmetric`), and even
        orders 2k >= 4 use the standard Suzuki recursion (Suzuki 1992)

            S_{2k}(F) = S_{2k-2}(u_k F)^2 S_{2k-2}((1 - 4 u_k) F) S_{2k-2}(u_k F)^2,
            u_k = 1 / (4 - 4^{1/(2k-1)}),

        whose error versus ``exp(F)`` is O(||F||^{2k+1}). Each S_{2k} is time
        symmetric, so the emitted block is a palindrome and the outer
        ``terms.reverse()`` in :meth:`_decompose_zassenhaus_step` leaves it intact.
        """
        if suzuki_order <= 1:
            return self._exponentiate_factor(factor, atol=atol)
        if suzuki_order == 2:
            return self._exponentiate_factor_symmetric(factor, atol=atol)

        k = suzuki_order // 2
        u_k = 1.0 / (4.0 - 4.0 ** (1.0 / (2 * k - 1)))
        outer = self._exponentiate_factor_suzuki(u_k * factor, suzuki_order - 2, atol=atol)
        middle = self._exponentiate_factor_suzuki((1.0 - 4.0 * u_k) * factor, suzuki_order - 2, atol=atol)
        return [*outer, *outer, *middle, *outer, *outer]

    @staticmethod
    def _correction_trotter_order(degree: int, order: int) -> int:
        r"""Smallest Suzuki order s in {1, 2, 4, 6, ...} so the flattening is high enough.

        A degree-n factor has ``||F|| ~ t^n``; a Suzuki order-s product formula
        flattens it with error ``O(t^{n(s+1)})``. The truncation target is
        ``O(t^{order+1})``, so we need ``n(s+1) >= order + 1``. Group (leaf) factors
        (degree 0) commute internally and are exact at order 1.
        """
        if degree < 2:
            return 1
        needed = order + 1
        if degree * 2 >= needed:  # plain product (s = 1) already suffices
            return 1
        suzuki_order = 2
        while degree * (suzuki_order + 1) < needed:
            suzuki_order += 2
        return suzuki_order

    def name(self) -> str:
        """Return the name of the unitary builder."""
        return "zassenhaus"

    def type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"
