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
from qdk_chemistry.utils.pauli_commutation import commutator, do_pauli_labels_commute

__all__: list[str] = ["Zassenhaus", "ZassenhausSettings"]

# Supported expansion orders (matches acceptance criteria p in {2, 3, 4}).
_MIN_ORDER = 2
_MAX_ORDER = 4


class ZassenhausSettings(TimeEvolutionSettings):
    """Settings for the Zassenhaus product-formula builder."""

    def __init__(self):
        """Initialize ZassenhausSettings with default values.

        Attributes:
            order: Expansion order p; corrections C_2..C_p are included so the
                truncation error scales as O(t^{p+1}). Supported: 2, 3, 4.
                Order 1 (no corrections) falls back to the first-order Trotter builder.
            num_divisions: Number of time divisions N. The full Zassenhaus
                product approximates exp(-iH t/N) and is repeated N times.
            weight_threshold: Absolute threshold for filtering small coefficients.

        """
        super().__init__()
        self._set_default("order", "int", 2, "The Zassenhaus expansion order p (2, 3, or 4; 1 falls back to Trotter).")
        self._set_default("num_divisions", "int", 1, "Number of time divisions N (>= 1).")
        self._set_default(
            "weight_threshold", "float", 1e-12, "The absolute threshold for filtering small coefficients."
        )


class Zassenhaus(TimeEvolutionBuilder):
    """Zassenhaus product-formula time-evolution builder.

    Note:
        Unlike the Trotter builder, automatic step-count estimation from a
        ``target_accuracy`` (Zassenhaus error scales as
        ``||C_{p+1}|| t^{p+1} / N^p``) is not yet provided; ``num_divisions``
        is set explicitly (default 1). Such an estimator is left as future work.

        Remaining gate-count optimisations are also left as future work:
        ordering the recursion groups, exploiting disjoint-support (layer)
        parallelism instead of flattening layers, and merging redundant terms
        at division / Suzuki-copy boundaries.

    """

    def __init__(
        self,
        order: int = 2,
        *,
        time: float = 0.0,
        num_divisions: int = 1,
        weight_threshold: float = 1e-12,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        r"""Initialize the Zassenhaus builder.

        Args:
            order: Expansion order p in {2, 3, 4}; order 1 falls back to first-order Trotter. Defaults to 2.
            time: The evolution time. Defaults to 0.0.
            num_divisions: Number of time divisions N (>= 1). Defaults to 1.
            weight_threshold: Threshold for filtering small coefficients. Defaults to 1e-12.
            power: The power to raise the unitary to. Defaults to 1.
            power_strategy: Strategy for U^power: ``"rescale"`` or ``"repeat"`` (default).

        """
        super().__init__()
        self._settings = ZassenhausSettings()
        self._settings.set("time", time)
        self._settings.set("power", power)
        self._settings.set("power_strategy", power_strategy)
        self._settings.set("order", order)
        self._settings.set("num_divisions", num_divisions)
        self._settings.set("weight_threshold", weight_threshold)

    # ------------------------------------------------------------------ #
    # Entry point (mirrors Trotter._run_impl -> _trotter)
    # ------------------------------------------------------------------ #
    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct the unitary representation using the Zassenhaus expansion."""
        order = self._settings.get("order")
        if order == 1:
            # Order 1 carries no commutator corrections (e^X e^Y), which is
            # exactly the first-order Trotter product -- delegate to it.
            return self._trotter_fallback(qubit_hamiltonian)
        if not (_MIN_ORDER <= order <= _MAX_ORDER):
            raise NotImplementedError(
                f"Zassenhaus order must be 1 (Trotter fallback) or in [{_MIN_ORDER}, {_MAX_ORDER}], got {order}."
            )

        effective_time, power_repetitions = self._resolve_power()
        weight_threshold = self._settings.get("weight_threshold")

        if not qubit_hamiltonian.is_hermitian(tolerance=weight_threshold):
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        num_divisions = max(1, self._settings.get("num_divisions"))
        delta = effective_time / num_divisions

        # One Zassenhaus product approximating exp(-i H delta).
        step_terms = self._decompose_zassenhaus_step(qubit_hamiltonian, time=delta, order=order, atol=weight_threshold)

        container = PauliProductFormulaContainer(
            step_terms=step_terms,
            step_reps=num_divisions * power_repetitions,
            num_qubits=qubit_hamiltonian.num_qubits,
        )
        return UnitaryRepresentation(container=container)

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

        Partitions H into internally-commuting groups G_1, ..., G_K and builds
        the recursive K-group Zassenhaus product (see :meth:`_zassenhaus_factors`),
        then flattens each anti-Hermitian factor into ``ExponentiatedPauliTerm``s.
        """
        groups = self._commuting_groups(qubit_hamiltonian)
        if not groups:
            Logger.warn("No Pauli terms above the tolerance; returning empty term list.")
            return []

        # Ordered (leftmost-first) list of (factor, is_correction) pairs.
        factors = self._zassenhaus_factors(groups, time, order)

        # Flatten each factor in operator order exp(G_1) ... exp(C_n). Group
        # factors are exact (internally commuting), so a plain product suffices.
        # A degree-n correction flattens with error O(t^{2n}) (plain) or O(t^{3n})
        # (symmetric/Strang). The lowest correction C_2 (n=2) gives O(t^4) plain,
        # which is < O(t^{p+1}) only for p >= 4 -- so symmetric flattening is needed
        # for order 4 but redundant (extra gates) for orders 2 and 3.
        flatten_correction = self._exponentiate_factor_symmetric if order > 3 else self._exponentiate_factor
        terms: list[ExponentiatedPauliTerm] = []
        for factor, is_correction in factors:
            flatten = flatten_correction if is_correction else self._exponentiate_factor
            terms.extend(flatten(factor, atol=atol))

        # The mapper applies step_terms[0] first (rightmost operator), so reverse
        # to realize the product above: the leftmost factor is applied last.
        terms.reverse()
        return terms

    def _zassenhaus_factors(
        self, groups: list[QubitHamiltonian], time: float, order: int
    ) -> list[tuple[QubitHamiltonian, bool]]:
        r"""Recursive K-group Zassenhaus product.

        For groups (G_1, ..., G_K), each internally commuting, with
        X_1 = -i t G_1 and X_R = -i t (G_2 + ... + G_K):

            Z_p(G_1, ..., G_K) = e^{X_1} . Z_p(G_2, ..., G_K) . e^{C_2} ... e^{C_p}

        where the corrections C_n = C_n(X_1, X_R) use the *exact* rest block X_R.
        Returned as an ordered (leftmost-first) list of ``(factor, is_correction)``
        pairs; each e^{X_i} group factor is exact because G_i is internally
        commuting.
        """
        x_first = (-1j * time) * groups[0]
        if len(groups) == 1:
            return [(x_first, False)]

        rest = groups[1]
        for group in groups[2:]:
            rest = rest + group
        x_rest = (-1j * time) * rest

        inner = self._zassenhaus_factors(groups[1:], time, order)
        corrections = [(c, True) for c in self._correction_factors(x_first, x_rest, order=order)]
        return [(x_first, False), *inner, *corrections]

    def _correction_factors(self, x: QubitHamiltonian, y: QubitHamiltonian, order: int) -> list[QubitHamiltonian]:
        r"""Return the commutator-correction factors C_2 .. C_order for blocks X, Y.

        Uses the native ``commutator`` (QDK-native, Qiskit-free) recursively.
        Each returned factor is an anti-Hermitian ``QubitHamiltonian``.
        """
        k = commutator(x, y)  # [X, Y] -- shared building block

        factors: list[QubitHamiltonian] = []

        # Order 2: -1/2 [X, Y]
        factors.append((-0.5) * k)
        if order == 2:
            return factors

        # Order 3: 1/3 [Y, [X, Y]] + 1/6 [X, [X, Y]]
        c3 = (1.0 / 3.0) * commutator(y, k) + (1.0 / 6.0) * commutator(x, k)
        factors.append(c3)
        if order == 3:
            return factors

        # Order 4: -1/24 [X,[X,[X,Y]]] - 1/8 [Y,[X,[X,Y]]] - 1/8 [Y,[Y,[X,Y]]].
        # Coefficients verified two ways: (i) roots-of-unity extraction of the
        # degree-4 term, and (ii) the operator-norm slope test gives p+1 = 5.
        xk = commutator(x, k)  # [X, [X, Y]]
        yk = commutator(y, k)  # [Y, [X, Y]]
        c4 = (-1.0 / 24.0) * commutator(x, xk) + (-1.0 / 8.0) * commutator(y, xk) + (-1.0 / 8.0) * commutator(y, yk)
        factors.append(c4)
        return factors

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
                    coefficients=np.asarray([coeffs[i] for i in indices]),
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

        Note: a factor's terms generally do NOT commute, so this first-order
        flattening introduces an O(t^{2n}) error for an O(t^n) factor -- which
        is >= O(t^{p+1}) for n >= 2, p <= 4, hence harmless to the target order.
        """
        terms: list[ExponentiatedPauliTerm] = []
        for label, coeff in zip(factor.pauli_strings, factor.coefficients, strict=True):
            c = complex(coeff)
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

    def name(self) -> str:
        """Return the name of the unitary builder."""
        return "zassenhaus"

    def type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"
