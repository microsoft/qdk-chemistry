r"""Zassenhaus product-formula decomposition builder.

This module implements the Zassenhaus time evolution builder. The Zassenhaus
formula decomposes the unitary time evolution operator exp(-i H t) into a
sequence of exponentiated Pauli terms, structured with higher-order commutator
corrections.

It supports:
* Custom evolution order (supporting orders 2, 3, and 4).
* Automatic step count computation under naive or commutator-aware bounds.
* Dynamic selection of the optimal Zassenhaus order to minimize total gate count.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math
from itertools import product
import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder, TimeEvolutionSettings
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus_error import (
    zassenhaus_steps_commutator,
    zassenhaus_steps_naive,
)
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
from qdk_chemistry.utils.pauli_commutation import commutator, do_pauli_maps_commute
from qdk_chemistry.utils.zassenhaus_generation import PlanTerm, zassenhaus_commutator_plan

__all__: list[str] = ["Zassenhaus", "ZassenhausSettings"]


class ZassenhausSettings(TimeEvolutionSettings):
    """Settings for the Zassenhaus decomposition builder."""

    def __init__(self):
        """Initialize ZassenhausSettings with default values.

        Attributes:
            order: The order of the Zassenhaus decomposition (0 means auto).
            target_accuracy: Target accuracy for automatic step computation (0.0 means disabled).
            num_divisions: Explicit number of repeated Zassenhaus time slices (0 means automatic).
            error_bound: Strategy for computing the Zassenhaus error bound ("commutator" or "naive").
            weight_threshold: The absolute threshold for filtering small coefficients.

        """
        super().__init__()
        self._set_default("order", "int", 0, "The order of the Zassenhaus decomposition (0 means auto).")
        self._set_default(
            "target_accuracy",
            "double",
            0.0,
            "Target accuracy for automatic step computation (0.0 means disabled).",
        )
        self._set_default(
            "num_divisions",
            "int",
            0,
            "Explicit number of repeated Zassenhaus time slices (0 means automatic).",
        )
        self._set_default(
            "error_bound",
            "string",
            "commutator",
            "Strategy for computing the Zassenhaus error bound ('commutator' or 'naive').",
            ["commutator", "naive"],
        )
        self._set_default(
            "weight_threshold", "float", 1e-12, "The absolute threshold for filtering small coefficients."
        )


class Zassenhaus(TimeEvolutionBuilder):
    """Zassenhaus decomposition builder."""

    def __init__(
        self,
        order: int = 0,
        *,
        time: float = 0.0,
        target_accuracy: float = 0.0,
        num_divisions: int = 0,
        error_bound: str = "commutator",
        weight_threshold: float = 1e-12,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        """Initialize Zassenhaus builder with specified Zassenhaus decomposition settings.

        Args:
            order: Zassenhaus decomposition order (0 for auto, or 1, 2, 3, 4). Defaults to 0.
            time: The evolution time. Defaults to 0.0.
            target_accuracy: Target accuracy for auto step computation. Use 0.0 (default) to disable.
            num_divisions: Repeated Zassenhaus time slices. Max of this and auto value is used. Defaults to 0.
            error_bound: Error bound strategy: ``"commutator"`` (default) or ``"naive"``.
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
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("num_divisions", num_divisions)
        self._settings.set("error_bound", error_bound)
        self._settings.set("weight_threshold", weight_threshold)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct the unitary representation using Zassenhaus decomposition.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.

        Returns:
            UnitaryRepresentation: The unitary representation built by the Zassenhaus decomposition.

        """
        effective_time, power_repetitions = self._resolve_power()
        order = self._settings.get("order")

        if order == 0:
            resolved_order, num_divisions = self._resolve_optimal_order_and_divisions(
                qubit_hamiltonian, effective_time
            )
        elif order >= 1:
            resolved_order = order
            num_divisions = self._resolve_num_divisions(qubit_hamiltonian, effective_time, order=order)
        else:
            raise NotImplementedError("Zassenhaus orders must be non-negative (0 for auto, or >= 1).")

        delta = effective_time / num_divisions
        terms = self._decompose_zassenhaus_step(
            qubit_hamiltonian, time=delta, atol=self._settings.get("weight_threshold"), order=resolved_order
        )

        num_qubits = qubit_hamiltonian.num_qubits

        container = PauliProductFormulaContainer(
            step_terms=terms,
            step_reps=num_divisions * power_repetitions,
            num_qubits=num_qubits,
        )

        return UnitaryRepresentation(container=container)

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitHamiltonian, time: float, *, order: int = None) -> int:
        """Determine the number of Zassenhaus divisions to use.

        When both *num_divisions* and *target_accuracy* are provided, the
        larger value wins.  When neither is provided, the default is 1.

        """
        num_divisions = self._settings.get("num_divisions")
        manual = num_divisions if num_divisions > 0 else 1

        target_accuracy = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0:
            return manual

        if order is None:
            order = self._settings.get("order")
            # If still 0 (auto), default to 2 as base order for auto-resolution fallback
            if order == 0:
                order = 2

        weight_threshold = self._settings.get("weight_threshold")
        error_bound = self._settings.get("error_bound")

        if error_bound == "commutator":
            auto = zassenhaus_steps_commutator(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=target_accuracy,
                order=order,
                weight_threshold=weight_threshold,
            )
        else:
            auto = zassenhaus_steps_naive(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=target_accuracy,
                order=order,
                weight_threshold=weight_threshold,
            )
        return max(manual, auto)

    def _resolve_optimal_order_and_divisions(
        self, qubit_hamiltonian: QubitHamiltonian, time: float
    ) -> tuple[int, int]:
        """Determine the optimal Zassenhaus order and the corresponding number of divisions.

        Sweeps orders 2, 3, and 4, computes the required divisions for each order
        to achieve target_accuracy, and selects the order that minimizes the total
        number of Pauli rotations (step_reps * step_terms).
        """
        target_accuracy = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0:
            # Fallback to order 2 if target accuracy is not specified
            return 2, self._resolve_num_divisions(qubit_hamiltonian, time, order=2)

        best_order = 2
        best_divisions = 1
        min_total_terms = float("inf")

        for p in (2, 3, 4):
            divisions = self._resolve_num_divisions(qubit_hamiltonian, time, order=p)
            try:
                delta = time / divisions
                step_terms = self._decompose_zassenhaus_step(
                    qubit_hamiltonian,
                    time=delta,
                    atol=self._settings.get("weight_threshold"),
                    order=p,
                )
                total_terms = len(step_terms) * divisions
                if total_terms < min_total_terms:
                    min_total_terms = total_terms
                    best_order = p
                    best_divisions = divisions
            except Exception:
                pass

        return best_order, best_divisions

    def _decompose_zassenhaus_step(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time: float,
        *,
        atol: float = 1e-12,
        order: int = None,
    ) -> list[ExponentiatedPauliTerm]:
        """Decompose a single Zassenhaus step into exponentiated Pauli terms.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be decomposed.
            time: The evolution time for the single step.
            atol: Absolute tolerance for filtering small coefficients.
            order: Zassenhaus expansion order (if None, taken from settings).

        Returns:
            A list of ``ExponentiatedPauliTerm`` representing the decomposed terms.

        """
        terms: list[ExponentiatedPauliTerm] = []

        if not qubit_hamiltonian.is_hermitian(tolerance=atol):
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        # If all coefficients are below the tolerance, there is nothing to decompose.
        if not any(abs(complex(c).real) > atol for c in qubit_hamiltonian.coefficients):
            Logger.warn("No coefficients above the tolerance; returning empty term list.")
            return terms

        if order is None:
            order = self._settings.get("order")
            if order == 0:
                order = 2

        grouped_hamiltonians = self._group_terms(qubit_hamiltonian)

        if not grouped_hamiltonians:
            Logger.warn("Term partition produced no groups; returning empty term list.")
            return terms

        for group in grouped_hamiltonians:
            for subgroup in group:
                terms.extend(
                    self._exponentiate_commuting(
                        subgroup,
                        time=time,
                        atol=atol,
                    )
                )

        if order == 1:
            return terms

        terms.extend(
            self._decompose_zassenhaus_corrections(
                grouped_hamiltonians,
                order=order,
                time=time,
                atol=atol,
            )
        )

        return list(reversed(terms))

    def _decompose_zassenhaus_corrections(
        self,
        grouped_hamiltonians: list[list[QubitHamiltonian]],
        *,
        order: int,
        time: float,
        atol: float = 1e-12,
    ) -> list[ExponentiatedPauliTerm]:
        """Decompose higher-order Zassenhaus correction exponents."""
        terms: list[ExponentiatedPauliTerm] = []
        leaves = tuple(range(len(grouped_hamiltonians)))
        planned_exponents, plan = zassenhaus_commutator_plan(leaves, max_order=order)

        leaf_sequences: dict[PlanTerm, tuple[int, ...]] = {}
        commutator_cache: dict[tuple[PlanTerm, tuple[int, ...]], QubitHamiltonian] = {}

        def leaf_sequence(ref: PlanTerm) -> tuple[int, ...]:
            if ref in leaf_sequences:
                return leaf_sequences[ref]

            if ref in plan:
                left, right = plan[ref]
                sequence = leaf_sequence(left) + leaf_sequence(right)
            elif isinstance(ref, int):
                sequence = (ref,)
            else:
                raise TypeError(f"Unexpected Zassenhaus plan reference: {ref!r}.")

            leaf_sequences[ref] = sequence
            return sequence

        def evaluate(ref: PlanTerm, choices: tuple[int, ...]) -> QubitHamiltonian:
            if ref in plan:
                key = (ref, choices)
                if key not in commutator_cache:
                    left, right = plan[ref]
                    left_size = len(leaf_sequence(left))
                    commutator_cache[key] = commutator(
                        evaluate(left, choices[:left_size]),
                        evaluate(right, choices[left_size:]),
                    )
                return commutator_cache[key]

            if not isinstance(ref, int):
                raise TypeError(f"Unexpected Zassenhaus leaf reference: {ref!r}.")

            if len(choices) != 1:
                raise ValueError(f"Leaf {ref!r} expected one subgroup choice, got {len(choices)}.")

            return grouped_hamiltonians[ref][choices[0]]

        for n in range(2, order + 1):
            phase = 1j * ((-1j) ** n)
            correction_terms: list[ExponentiatedPauliTerm] = []

            for ref, coeff in planned_exponents[n].items():
                sequence = leaf_sequence(ref)
                subgroup_ranges = (range(len(grouped_hamiltonians[label])) for label in sequence)

                for choices in product(*subgroup_ranges):
                    contribution = evaluate(ref, choices) * (phase * complex(coeff))
                    correction_terms.extend(
                        self._exponentiate_commuting(
                            contribution,
                            time=time**n,
                            atol=atol,
                        )
                    )

            terms.extend(self._decompose_correction_exponent(correction_terms, correction_degree=n, order=order))

        return terms

    def _decompose_correction_exponent(
        self,
        correction_terms: list[ExponentiatedPauliTerm],
        *,
        correction_degree: int,
        order: int,
    ) -> list[ExponentiatedPauliTerm]:
        """Decompose one Zassenhaus correction exponent into Pauli rotations."""
        if not correction_terms:
            return []

        if self._terms_commute(correction_terms):
            return correction_terms

        internal_order = math.ceil((order + 1) / correction_degree) - 1
        if internal_order <= 1:
            return correction_terms

        correction_hamiltonian = self._terms_to_hamiltonian(correction_terms)
        internal_builder = Zassenhaus(order=internal_order, time=1.0)
        internal_execution_terms = internal_builder._decompose_zassenhaus_step(
            correction_hamiltonian, time=1.0, atol=0.0
        )
        return list(reversed(internal_execution_terms))

    def _terms_commute(self, terms: list[ExponentiatedPauliTerm]) -> bool:
        """Return whether all Pauli rotations in *terms* mutually commute."""
        for i, left in enumerate(terms):
            for right in terms[i + 1 :]:
                if not do_pauli_maps_commute(left.pauli_term, right.pauli_term):
                    return False
        return True

    def _terms_to_hamiltonian(self, terms: list[ExponentiatedPauliTerm]) -> QubitHamiltonian:
        """Convert exponentiated Pauli terms to a Hamiltonian with matching angles."""
        num_qubits = 1 + max((qubit for term in terms for qubit in term.pauli_term), default=0)
        labels = []
        coefficients = []
        for term in terms:
            labels.append(self._pauli_map_to_label(term.pauli_term, num_qubits))
            coefficients.append(term.angle)
        return QubitHamiltonian(pauli_strings=labels, coefficients=np.asarray(coefficients, dtype=complex))

    def _pauli_map_to_label(self, pauli_term: dict[int, str], num_qubits: int) -> str:
        """Convert a qubit-indexed Pauli map to a QubitHamiltonian label."""
        chars = ["I"] * num_qubits
        for qubit, pauli in pauli_term.items():
            chars[num_qubits - qubit - 1] = pauli
        return "".join(chars)

    def _group_terms(
        self,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[list[QubitHamiltonian]]:
        """Group Hamiltonian terms for Zassenhaus decomposition."""
        partition = qubit_hamiltonian.term_partition
        if partition is not None:
            Logger.debug(
                f"Zassenhaus: consuming QubitHamiltonian.term_partition "
                f"(strategy={partition.strategy!r}, num_groups={partition.num_groups})."
            )
            return self._groups_from_partition(qubit_hamiltonian, partition)

        Logger.debug("Zassenhaus: no term_partition present; treating each Pauli term as its own group.")
        return [
            [
                QubitHamiltonian(
                    pauli_strings=[label],
                    coefficients=[coeff],
                    encoding=qubit_hamiltonian.encoding,
                    fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
                )
            ]
            for label, coeff in zip(qubit_hamiltonian.pauli_strings, qubit_hamiltonian.coefficients, strict=True)
        ]

    def _groups_from_partition(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        partition: TermPartition,
    ) -> list[list[QubitHamiltonian]]:
        """Materialise a :class:`TermPartition` into Zassenhaus sub-groups."""
        labels = qubit_hamiltonian.pauli_strings
        coeffs = qubit_hamiltonian.coefficients
        encoding = qubit_hamiltonian.encoding
        fmo = qubit_hamiltonian.fermion_mode_order

        def _make(indices: tuple[int, ...]) -> QubitHamiltonian:
            return QubitHamiltonian(
                pauli_strings=[labels[i] for i in indices],
                coefficients=np.asarray([coeffs[i] for i in indices]),
                encoding=encoding,
                fermion_mode_order=fmo,
            )

        if isinstance(partition, LayeredPartition):
            layered_groups = partition.groups
        elif isinstance(partition, FlatPartition):
            layered_groups = tuple((g,) for g in partition.groups)
        else:
            raise TypeError(
                f"Unsupported TermPartition subtype: {type(partition).__name__}. "
                "Expected FlatPartition or LayeredPartition."
            )

        groups: list[list[QubitHamiltonian]] = [
            [_make(layer) for layer in group_layers if layer] for group_layers in layered_groups
        ]

        groups = [g for g in groups if g]
        groups.sort(key=len)
        return groups

    def _exponentiate_commuting(
        self,
        group: QubitHamiltonian,
        time: float,
        *,
        atol: float = 1e-12,
    ) -> list[ExponentiatedPauliTerm]:
        r"""Exponentiate a group of commuting Pauli terms."""
        terms: list[ExponentiatedPauliTerm] = []
        for label, coeff in group.get_real_coefficients(tolerance=atol):
            mapping = self._pauli_label_to_map(label)
            angle = coeff * time
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))
        return terms

    def name(self) -> str:
        """Return the name of the unitary builder."""
        return "zassenhaus"

    def type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"
