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
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter_error import (
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
            order: The order of the Zassenhaus decomposition.
            target_accuracy: Target accuracy for automatic step computation (0.0 means disabled).
            num_divisions: Explicit number of repeated Zassenhaus time slices (0 means automatic).
            error_bound: Strategy for computing the Zassenhaus error bound ("commutator" or "naive").
            weight_threshold: The absolute threshold for filtering small coefficients.

        """
        super().__init__()
        self._set_default("order", "int", 2, "The order of the Zassenhaus decomposition.")
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
        order: int = 2,
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
            order: Zassenhaus decomposition order (1, 2, 3, 4). Defaults to 2.
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

        if order == 1:
            # Delegate order=1 to the Trotter builder to avoid code duplication.
            trotter_builder = Trotter(
                order=1,
                time=self._settings.get("time"),
                target_accuracy=self._settings.get("target_accuracy"),
                num_divisions=self._settings.get("num_divisions"),
                error_bound=self._settings.get("error_bound"),
                weight_threshold=self._settings.get("weight_threshold"),
                power=self._settings.get("power"),
                power_strategy=self._settings.get("power_strategy"),
            )
            return trotter_builder.run(qubit_hamiltonian)

        if order >= 2:
            resolved_order = order
            num_divisions = self._resolve_num_divisions(qubit_hamiltonian, effective_time, order=order)
        else:
            raise NotImplementedError("Zassenhaus orders must be greater than or equal to 1.")

        delta = effective_time / num_divisions
        terms = self._decompose_zassenhaus_step(
            qubit_hamiltonian, time=delta, atol=self._settings.get("weight_threshold"), order=resolved_order
        )
        terms = self._optimize_pauli_sequence(terms, atol=1e-15)

        num_qubits = qubit_hamiltonian.num_qubits

        container = PauliProductFormulaContainer(
            step_terms=terms,
            step_reps=num_divisions * power_repetitions,
            num_qubits=num_qubits,
        )

        return UnitaryRepresentation(container=container)

    def _resolve_num_divisions(
        self, qubit_hamiltonian: QubitHamiltonian, time: float, *, order: int | None = None
    ) -> int:
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

    def _decompose_zassenhaus_step(
        self,
        qubit_hamiltonian: QubitHamiltonian,
        time: float,
        *,
        atol: float = 1e-12,
        order: int | None = None,
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
                    h_left = evaluate(left, choices[:left_size])
                    h_right = evaluate(right, choices[left_size:])

                    # Convert to maps and check standard commutation
                    left_maps = [self._pauli_label_to_map(s) for s in h_left.pauli_strings]
                    right_maps = [self._pauli_label_to_map(s) for s in h_right.pauli_strings]

                    all_commute = True
                    for map_l in left_maps:
                        for map_r in right_maps:
                            if not do_pauli_maps_commute(map_l, map_r):
                                all_commute = False
                                break
                        if not all_commute:
                            break

                    if all_commute:
                        num_qubits = max(h_left.num_qubits, h_right.num_qubits)
                        # Avoid the expensive commutator calculation if all terms commute
                        commutator_cache[key] = QubitHamiltonian(
                            pauli_strings=["I" * num_qubits],
                            coefficients=np.array([0.0], dtype=complex),
                        )
                    else:
                        commutator_cache[key] = commutator(h_left, h_right)
                return commutator_cache[key]

            if not isinstance(ref, int):
                raise TypeError(f"Unexpected Zassenhaus leaf reference: {ref!r}.")

            if len(choices) != 1:
                raise ValueError(f"Leaf {ref!r} expected one subgroup choice, got {len(choices)}.")

            return grouped_hamiltonians[ref][choices[0]]

        for n in range(2, order + 1):
            # For a degree-n correction exponent in the Zassenhaus expansion of e^{-i H t},
            # the symbolic Lie polynomial C_n is homogeneous of degree n in the generators {-i H_j}.
            # Thus, C_n has a factor of (-i)^n. Since we represent the exponentiated Pauli terms
            # in the form e^{-i C'_n t^n}, we factor out -i (or -1j), leaving a phase factor of
            # (-i)^n / (-i) = (-i)^{n-1} = i * (-i)^n.
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

    def _optimize_pauli_sequence(
        self,
        terms: list[ExponentiatedPauliTerm],
        *,
        atol: float = 1e-15,
    ) -> list[ExponentiatedPauliTerm]:
        """Optimize a sequence of exponentiated Pauli terms by merging adjacent and commuting terms."""
        if not terms:
            return []

        optimized: list[ExponentiatedPauliTerm] = []
        for term in terms:
            pauli_key = tuple(sorted(term.pauli_term.items()))
            merged = False

            # Bubble backwards to find a matching Pauli term to merge with
            for i in reversed(range(len(optimized))):
                prev_term = optimized[i]
                prev_key = tuple(sorted(prev_term.pauli_term.items()))

                if prev_key == pauli_key:
                    # Found matching term, merge the angles
                    new_angle = prev_term.angle + term.angle
                    optimized[i] = ExponentiatedPauliTerm(pauli_term=prev_term.pauli_term, angle=new_angle)
                    merged = True
                    break

                # If the terms do not commute, we cannot bubble past
                if not do_pauli_maps_commute(term.pauli_term, prev_term.pauli_term):
                    break

            if not merged:
                optimized.append(term)

        # Filter out near-zero terms
        final_terms = []
        for term in optimized:
            if abs(term.angle) >= atol:
                final_terms.append(term)

        return final_terms

    def name(self) -> str:
        """Return the name of the unitary builder."""
        return "zassenhaus"

    def type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"
