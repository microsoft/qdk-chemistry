"""QDK/Chemistry Zassenhaus time-evolution builder.

References:
    Wilcox, R. M. "Exponential operators and parameter differentiation in quantum physics."
    *Journal of Mathematical Physics* 8.4 (1967): 962-982.

    Casas, F., A. Murua, and M. Nadinic. "Efficient computation of the Zassenhaus formula."
    *Computer Physics Communications* 183.11 (2012): 2386-2391.

    Childs, A. M., et al. "Theory of Trotter Error with Commutator
    Scaling." *Physical Review X* 11.1 (2021): 011020.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from functools import cache

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import (
    TimeEvolutionBuilder,
    TimeEvolutionSettings,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter_error import (
    trotter_steps_commutator,
    trotter_steps_naive,
)
from qdk_chemistry.algorithms.term_grouper import FullCommutingTermGrouper
from qdk_chemistry.data import PauliTermAccumulator, QubitHamiltonian, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.pauli_commutation import does_nested_commutator_vanish

__all__: list[str] = ["Zassenhaus", "ZassenhausSettings"]

Word = tuple[int, ...]
WordPolynomial = dict[Word, Fraction]
CachedWordPolynomial = tuple[tuple[Word, Fraction], ...]
PauliWord = tuple[tuple[int, int], ...]
PauliSummand = tuple[str, PauliWord, complex]


class ZassenhausSettings(TimeEvolutionSettings):
    """Settings for the Zassenhaus time-evolution builder."""

    def __init__(self):
        """Initialize ZassenhausSettings with default values."""
        super().__init__()
        self._set_default(
            "order",
            "int",
            2,
            "Expansion order. The local error scales as O(time^(order + 1)).",
        )
        self._set_default(
            "target_accuracy",
            "double",
            0.0,
            "Target accuracy for automatic division-count estimation (0.0 means disabled).",
        )
        self._set_default(
            "num_divisions",
            "int",
            1,
            "Explicit number of time divisions.",
        )
        self._set_default(
            "error_bound",
            "string",
            "commutator",
            "Strategy for computing the Trotter error bound ('commutator' or 'naive').",
            ["commutator", "naive"],
        )
        self._set_default(
            "weight_threshold",
            "float",
            1e-12,
            "The absolute threshold for filtering small coefficients.",
        )


class Zassenhaus(TimeEvolutionBuilder):
    r"""Zassenhaus product-formula Hamiltonian simulation builder.

    The builder constructs a product approximation to
    :math:`\exp(-iHt)` by first writing one time slice as
    :math:`\exp(\sum_j X_j)`, with :math:`X_j=-i\Delta t\,H_j`, and then
    factoring it as

    .. math::

        e^{\sum_j X_j} =
        \left(\prod_j e^{X_j}\right) e^{C_2} e^{C_3}\cdots e^{C_p}
        + O(\Delta t^{p+1}).

    The homogeneous correction exponents :math:`C_k` are generated as a
    formal non-commutative word series and evaluated using the package's
    existing Pauli multiplication utilities.
    """

    def __init__(
        self,
        order: int = 2,
        *,
        time: float = 0.0,
        target_accuracy: float = 0.0,
        num_divisions: int = 1,
        error_bound: str = "commutator",
        weight_threshold: float = 1e-12,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        """Initialize a Zassenhaus time-evolution builder.

        Args:
            order: Expansion order. Order 1 delegates to the first-order
                Trotter builder. Defaults to 2.
            time: Evolution time. Defaults to 0.0.
            target_accuracy: Optional target accuracy used to choose a
                conservative number of time divisions. Defaults to 0.0.
            num_divisions: Manual lower bound on the number of time divisions.
                Defaults to 1.
            error_bound: Error bound strategy for automatic division estimation:
                ``"commutator"`` (default) or ``"naive"``.
            weight_threshold: Absolute threshold for filtering small
                coefficients. Defaults to 1e-12.
            power: Power to raise the unitary to. Defaults to 1.
            power_strategy: ``"rescale"`` scales time; ``"repeat"`` repeats
                the resulting product. Defaults to ``"repeat"``.

        """
        super().__init__()
        self._settings = ZassenhausSettings()
        self._settings.set("time", time)
        self._settings.set("power", power)
        self._settings.set("power_strategy", power_strategy)
        self._settings.set("order", order)
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("num_divisions", num_divisions)
        self._settings.set("error_bound", error_bound)
        self._settings.set("weight_threshold", weight_threshold)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct a Zassenhaus product-formula unitary representation."""
        order = self._settings.get("order")
        if order < 1:
            raise ValueError(f"Zassenhaus order must be positive, got {order}.")

        if order == 1:
            return self._run_trotter_order_one(qubit_hamiltonian)

        weight_threshold = self._settings.get("weight_threshold")
        if not qubit_hamiltonian.is_hermitian(tolerance=weight_threshold):
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        effective_time, power_repetitions = self._resolve_power()
        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, effective_time, order)
        delta = effective_time / num_divisions

        blocks = self._hamiltonian_blocks(qubit_hamiltonian, weight_threshold)
        step_terms = self._decompose_zassenhaus_step(
            blocks,
            time=delta,
            order=order,
            num_qubits=qubit_hamiltonian.num_qubits,
            atol=weight_threshold,
        )

        return UnitaryRepresentation(
            container=PauliProductFormulaContainer(
                step_terms=step_terms,
                step_reps=num_divisions * power_repetitions,
                num_qubits=qubit_hamiltonian.num_qubits,
            )
        )

    def _run_trotter_order_one(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Delegate the first-order Zassenhaus case to the Trotter builder."""
        return Trotter(
            order=1,
            time=self._settings.get("time"),
            target_accuracy=self._settings.get("target_accuracy"),
            num_divisions=self._settings.get("num_divisions"),
            error_bound=self._settings.get("error_bound"),
            weight_threshold=self._settings.get("weight_threshold"),
            power=self._settings.get("power"),
            power_strategy=self._settings.get("power_strategy"),
        ).run(qubit_hamiltonian)

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitHamiltonian, time: float, order: int) -> int:
        """Resolve manual and accuracy-driven time divisions.

        Automatic division estimates reuse the same Trotter error-bound helpers
        used by :class:`Trotter`.  Odd Zassenhaus orders use the highest
        supported Trotter error order that does not exceed the expansion order.
        """
        num_divisions = self._settings.get("num_divisions")
        if num_divisions <= 0:
            raise ValueError(f"num_divisions must be a positive integer, got {num_divisions}.")
        manual = num_divisions

        target_accuracy = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0:
            return manual

        error_order = _trotter_error_order(order)
        weight_threshold = self._settings.get("weight_threshold")
        error_bound = self._settings.get("error_bound")
        if error_bound == "commutator":
            auto = trotter_steps_commutator(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=target_accuracy,
                order=error_order,
                weight_threshold=weight_threshold,
            )
        else:
            auto = trotter_steps_naive(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=target_accuracy,
                order=error_order,
                weight_threshold=weight_threshold,
            )

        return max(manual, auto)

    def _hamiltonian_blocks(self, qubit_hamiltonian: QubitHamiltonian, atol: float) -> list[QubitHamiltonian]:
        """Split the Hamiltonian into exactly exponentiable blocks."""
        if qubit_hamiltonian.term_partition is None and len(qubit_hamiltonian.pauli_strings) > 1:
            qubit_hamiltonian = FullCommutingTermGrouper().run(qubit_hamiltonian)

        grouped = Trotter()._group_terms(qubit_hamiltonian)  # noqa: SLF001
        blocks: list[QubitHamiltonian] = []
        for group in grouped:
            for block in group:
                if any(abs(coeff) > atol for _, coeff in block.get_real_coefficients(tolerance=atol)):
                    blocks.append(block)

        if not blocks:
            Logger.warn("No coefficients above the tolerance; returning empty Zassenhaus term list.")
        return blocks

    def _decompose_zassenhaus_step(
        self,
        blocks: list[QubitHamiltonian],
        *,
        time: float,
        order: int,
        num_qubits: int,
        atol: float,
    ) -> list[ExponentiatedPauliTerm]:
        """Decompose one Zassenhaus time slice into exponentiated Pauli terms."""
        if not blocks:
            return []

        factor_terms: list[list[ExponentiatedPauliTerm]] = []
        trotter = Trotter(order=1)

        for block in blocks:
            factor_terms.append(trotter._exponentiate_commuting(block, time=time, atol=atol))  # noqa: SLF001

        correction_factors = _zassenhaus_word_factors(len(blocks), order)
        pauli_blocks = [_hamiltonian_to_pauli_terms(block, atol) for block in blocks]

        for degree, factor in enumerate(correction_factors, start=2):
            antihermitian_terms = _evaluate_word_polynomial(
                factor,
                pauli_blocks,
                time=time,
                num_qubits=num_qubits,
                atol=atol,
            )
            correction_hamiltonian = _antihermitian_terms_to_hamiltonian(
                antihermitian_terms,
                atol=atol,
            )
            if correction_hamiltonian is None:
                continue

            formula_order = _correction_formula_order(degree, order)
            factor_terms.append(
                Trotter(order=formula_order)._decompose_trotter_step(  # noqa: SLF001
                    correction_hamiltonian,
                    time=1.0,
                    atol=atol,
                )
            )

        # PauliProductFormulaContainer materializes a term list as
        # E(last) ... E(first).  The formal Zassenhaus factors above are in
        # algebraic left-to-right order, so write factor containers in reverse.
        terms: list[ExponentiatedPauliTerm] = []
        for sequence in reversed(factor_terms):
            terms.extend(sequence)
        return _merge_adjacent_terms(terms, atol=atol)

    def name(self) -> str:
        """Return the name of the unitary builder."""
        return "zassenhaus"

    def type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"


@cache
def _zassenhaus_word_factors(num_symbols: int, order: int) -> tuple[CachedWordPolynomial, ...]:
    """Generate homogeneous Zassenhaus correction factors as word polynomials."""
    if num_symbols <= 1 or order < 2:
        return ()

    target_generator: WordPolynomial = {(i,): Fraction(1) for i in range(num_symbols)}
    target = _poly_exp(target_generator, order)

    current: WordPolynomial = {(): Fraction(1)}
    for i in range(num_symbols):
        current = _poly_mul(current, _poly_exp({(i,): Fraction(1)}, order), order)

    factors: list[WordPolynomial] = []
    for degree in range(2, order + 1):
        correction = _poly_homogeneous(_poly_add(target, current, scale=Fraction(-1)), degree)
        factors.append(correction)
        if correction and degree < order:
            current = _poly_mul(current, _poly_exp(correction, order), order)

    return tuple(tuple(factor.items()) for factor in factors)


def _poly_add(a: WordPolynomial, b: WordPolynomial, *, scale: Fraction = Fraction(1)) -> WordPolynomial:
    result = dict(a)
    for word, coeff in b.items():
        value = result.get(word, Fraction(0)) + scale * coeff
        if value:
            result[word] = value
        else:
            result.pop(word, None)
    return result


def _poly_mul(a: WordPolynomial, b: WordPolynomial, max_degree: int) -> WordPolynomial:
    result: WordPolynomial = {}
    for word_a, coeff_a in a.items():
        for word_b, coeff_b in b.items():
            word = word_a + word_b
            if len(word) > max_degree:
                continue
            result[word] = result.get(word, Fraction(0)) + coeff_a * coeff_b
    return {word: coeff for word, coeff in result.items() if coeff}


def _poly_exp(poly: WordPolynomial, max_degree: int) -> WordPolynomial:
    result: WordPolynomial = {(): Fraction(1)}
    power: WordPolynomial = {(): Fraction(1)}
    factorial = 1
    for k in range(1, max_degree + 1):
        power = _poly_mul(power, poly, max_degree)
        factorial *= k
        result = _poly_add(result, {word: coeff / factorial for word, coeff in power.items()})
        if not power:
            break
    return result


def _poly_homogeneous(poly: WordPolynomial, degree: int) -> WordPolynomial:
    return {word: coeff for word, coeff in poly.items() if len(word) == degree and coeff}


def _hamiltonian_to_pauli_terms(hamiltonian: QubitHamiltonian, atol: float) -> list[PauliSummand]:
    terms: dict[PauliWord, complex] = {}
    for label, coeff in hamiltonian.get_real_coefficients(tolerance=atol):
        word = _label_to_sparse_word(label)
        terms[word] = terms.get(word, 0.0) + complex(coeff)
    return [
        (_sparse_word_to_label(word, hamiltonian.num_qubits), word, coeff)
        for word, coeff in terms.items()
        if abs(coeff) > atol
    ]


def _evaluate_word_polynomial(
    polynomial: CachedWordPolynomial,
    pauli_blocks: list[list[PauliSummand]],
    *,
    time: float,
    num_qubits: int,
    atol: float,
) -> dict[str, complex]:
    """Evaluate a homogeneous Lie word polynomial as nested Pauli commutators.

    The generated Zassenhaus corrections are homogeneous Lie polynomials.  The
    Dynkin-Specht-Wever identity rewrites a degree-n correction as 1/n times
    the sum of its word coefficients applied to right-nested commutators, which
    lets vanishing Pauli commutators short-circuit before sparse multiplication.
    """
    accumulator: dict[PauliWord, complex] = {}
    for word, rational_coeff in polynomial:
        degree = len(word)
        if degree == 0 or any(not pauli_blocks[symbol] for symbol in word):
            continue

        scale = complex(rational_coeff / degree) * ((-1j * time) ** degree)
        if degree == 1:
            for _, pauli_word, term_coeff in pauli_blocks[word[0]]:
                accumulator[pauli_word] = accumulator.get(pauli_word, 0.0) + scale * term_coeff
            continue

        for terms in itertools.product(*(pauli_blocks[symbol] for symbol in word)):
            nested = _evaluate_pauli_nested_commutator(terms)
            if nested is None:
                continue

            pauli_word, commutator_coeff = nested
            coeff = scale * commutator_coeff
            for _, _, term_coeff in terms:
                coeff *= term_coeff
            accumulator[pauli_word] = accumulator.get(pauli_word, 0.0) + coeff

    labels: dict[str, complex] = {}
    for pauli_word, coeff in accumulator.items():
        if abs(coeff) <= atol:
            continue
        label = _sparse_word_to_label(pauli_word, num_qubits)
        labels[label] = labels.get(label, 0.0) + coeff
    return {label: coeff for label, coeff in labels.items() if abs(coeff) > atol}


def _evaluate_pauli_nested_commutator(terms: tuple[PauliSummand, ...]) -> tuple[PauliWord, complex] | None:
    """Evaluate ``[P_1, [P_2, ... P_n]]`` for Pauli summands, or return ``None`` if it vanishes."""
    labels = tuple(label for label, _, _ in terms)
    if does_nested_commutator_vanish(*labels):
        return None

    product_word = terms[0][1]
    phase = 1.0 + 0.0j
    for _, word, _ in terms[1:]:
        step_phase, multiplied_word = PauliTermAccumulator.multiply_uncached(list(product_word), list(word))
        phase *= complex(step_phase)
        product_word = tuple((int(q), int(p)) for q, p in multiplied_word)

    return product_word, (2 ** (len(terms) - 1)) * phase


def _antihermitian_terms_to_hamiltonian(
    antihermitian_terms: dict[str, complex],
    *,
    atol: float,
) -> QubitHamiltonian | None:
    labels: list[str] = []
    coefficients: list[float] = []
    for label, coeff in antihermitian_terms.items():
        if abs(coeff) <= atol:
            continue
        real_tolerance = max(10 * atol, 10 * atol * abs(coeff.imag))
        if abs(coeff.real) > real_tolerance:
            raise ValueError(f"Zassenhaus correction produced a non-anti-Hermitian coefficient for {label}: {coeff}.")
        angle = (1j * coeff).real
        if abs(angle) > atol:
            labels.append(label)
            coefficients.append(angle)

    if not labels:
        return None
    return QubitHamiltonian(pauli_strings=labels, coefficients=np.asarray(coefficients), encoding=None)


def _correction_formula_order(correction_degree: int, target_order: int) -> int:
    required = max(math.ceil((target_order + 1) / correction_degree) - 1, 1)
    if required <= 1:
        return 1
    if required <= 2:
        return 2
    return required if required % 2 == 0 else required + 1


def _trotter_error_order(zassenhaus_order: int) -> int:
    """Map a Zassenhaus order to an order supported by Trotter error bounds."""
    if zassenhaus_order <= 2 or zassenhaus_order % 2 == 0:
        return zassenhaus_order
    return zassenhaus_order - 1


def _label_to_sparse_word(label: str) -> PauliWord:
    word: list[tuple[int, int]] = []
    for index, char in enumerate(reversed(label)):
        if char == "I":
            continue
        if char == "X":
            word.append((index, 1))
        elif char == "Y":
            word.append((index, 2))
        elif char == "Z":
            word.append((index, 3))
        else:
            raise ValueError(f"Invalid character {char!r} in Pauli label; expected 'I', 'X', 'Y', or 'Z'.")
    return tuple(word)


def _sparse_word_to_label(word: PauliWord, num_qubits: int) -> str:
    chars = ["I"] * num_qubits
    for qubit, pauli in word:
        if pauli == 1:
            chars[qubit] = "X"
        elif pauli == 2:
            chars[qubit] = "Y"
        elif pauli == 3:
            chars[qubit] = "Z"
        else:
            raise ValueError(f"Invalid Pauli code {pauli}; expected 1, 2, or 3.")
    return "".join(reversed(chars))


def _merge_adjacent_terms(
    terms: list[ExponentiatedPauliTerm],
    *,
    atol: float,
) -> list[ExponentiatedPauliTerm]:
    merged: list[ExponentiatedPauliTerm] = []
    for term in terms:
        if abs(term.angle) <= atol:
            continue
        if merged and merged[-1].pauli_term == term.pauli_term:
            angle = merged[-1].angle + term.angle
            if abs(angle) > atol:
                merged[-1] = ExponentiatedPauliTerm(pauli_term=term.pauli_term, angle=angle)
            else:
                merged.pop()
        else:
            merged.append(term)
    return merged
