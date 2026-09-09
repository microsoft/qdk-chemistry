r"""QDK/Chemistry implementation of the Trotter decomposition Builder.

References:
    Childs, A. M., et al. "Theory of Trotter Error with Commutator
    Scaling." *Physical Review X* 11.1 (2021): 011020.

    Strang, G. "On the construction and comparison of difference
    schemes." SIAM Journal on Numerical Analysis 5.3 (1968): 506-517.

    Suzuki, M. "General theory of higher-order decomposition of
    exponential operators and symplectic integrators."
    Physics Letters A 165.5-6 (1992): 387-395.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from array import array

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder, TimeEvolutionSettings
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter_error import (
    trotter_steps_commutator,
    trotter_steps_naive,
)
from qdk_chemistry.data import (
    FlatPartition,
    LayeredPartition,
    QubitOperator,
    UnitaryRepresentation,
)
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils import Logger

__all__: list[str] = ["Trotter", "TrotterSettings"]


class TrotterSettings(TimeEvolutionSettings):
    """Settings for Trotter decomposition builder."""

    def __init__(self):
        """Initialize TrotterSettings with default values.

        Attributes:
            order: The order of the Trotter decomposition (currently only first order is supported).
            target_accuracy: Target accuracy for automatic step computation (0.0 means disabled).
            num_divisions: Explicit number of divisions within a Trotter step (0 means automatic).
            error_bound: Strategy for computing the Trotter error bound ("commutator" or "naive").
            weight_threshold: The absolute threshold for filtering small coefficients.

        """
        super().__init__()
        self._set_default("order", "int", 1, "The order of the Trotter decomposition.")
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
            "Explicit number of divisions within a Trotter step (0 means automatic).",
        )
        self._set_default(
            "error_bound",
            "string",
            "commutator",
            "Strategy for computing the Trotter error bound ('commutator' or 'naive').",
            ["commutator", "naive"],
        )
        self._set_default(
            "weight_threshold", "float", 1e-12, "The absolute threshold for filtering small coefficients."
        )


class Trotter(TimeEvolutionBuilder):
    """Trotter decomposition builder."""

    def __init__(
        self,
        order: int = 1,
        *,
        time: float = 0.0,
        target_accuracy: float = 0.0,
        num_divisions: int = 0,
        error_bound: str = "commutator",
        weight_threshold: float = 1e-12,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        r"""Initialize Trotter builder with specified Trotter decomposition settings.

        The Trotter decomposition approximates the time evolution operator :math:`e^{-iHt}`
        when the Hamiltonian :math:`H` can be expressed as a sum of terms :math:`H = \sum_j \alpha_j P_j`
        where :math:`P_j` are Pauli strings and :math:`\alpha_j` are scalar coefficients. Rather than
        exponentiating the full Hamiltonian at once, the Trotter method constructs an approximation by
        exponentiating each term separately and combining them in a product formula. For example,
        the first-order Trotter formula approximates the time evolution operator as

        :math:`e^{-iHt} \approx S_1^N(t) = \left[\prod_j e^{-i\alpha_j P_j t/N}\right]^N`, where :math:`N` is the
        number of divisions.

        The number of divisions *N* can be determined automatically from
        *target_accuracy*, fixed explicitly via *num_divisions*, or both
        (in which case the larger value is used).

        The error associated with the Trotter decomposition, :math:`S_k^N(t)`, can be expressted in terms of the
        spectral norm of the difference between the exact and approximate time evolution operators:

        :math:`\lVert e^{-iHt} - S_k^N(t) \rVert \leq \epsilon`

        However, the cost of computing this norm is equivalent to computing the exact exponential itself. For this
        reason, we provide two approximate error-bound strategies to determine the number of divisions required to
        achieve a target accuracy at a particular Trotter order (used only when *target_accuracy* is set):

        * ``"commutator"`` (default, tighter): uses the commutator-based bound
          from Childs *et al.* (2021).  :math:`N = \lceil \frac{t^{2}}{2\epsilon}
          \sum_{j<k}\lVert[\alpha_jP_j,\alpha_kP_k]\rVert \rceil`
        * ``"naive"``: uses the triangle-inequality bound.
          :math:`N = \lceil (\sum_j|\alpha_j|)^{2}t^{2}/\epsilon \rceil`

        When the input :class:`~qdk_chemistry.data.QubitOperator` carries a populated
        :attr:`~qdk_chemistry.data.QubitOperator.term_partition`, the builder consumes it
        directly for schedule-level grouping.  When no partition is present, each Pauli term
        is exponentiated as its own group.

        Args:
            order: Trotter decomposition order (1, 2, or any positive even integer). Defaults to 1.
            time: The evolution time. Defaults to 0.0.
            target_accuracy: Target accuracy for auto step computation. Use 0.0 (default) to disable.
            num_divisions: Divisions per Trotter step. Max of this and auto value is used. Defaults to 0.
            error_bound: Error bound strategy: ``"commutator"`` (default) or ``"naive"``.
            weight_threshold: Threshold for filtering small coefficients. Defaults to 1e-12.
            power: The power to raise the unitary to. Defaults to 1.
            power_strategy: Strategy for U^power: ``"rescale"`` or ``"repeat"`` (default).

        """
        super().__init__()
        self._settings = TrotterSettings()
        self._settings.set("time", time)
        self._settings.set("power", power)
        self._settings.set("power_strategy", power_strategy)
        self._settings.set("order", order)
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("num_divisions", num_divisions)
        self._settings.set("error_bound", error_bound)
        self._settings.set("weight_threshold", weight_threshold)

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        """Construct the unitary representation using Trotter decomposition.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.

        Returns:
            UnitaryRepresentation: The unitary representation built by the Trotter decomposition.

        """
        effective_time, power_repetitions = self._resolve_power()
        order = self._settings.get("order")
        if order in {1, 2} or (order > 2 and order % 2 == 0):
            return self._trotter(qubit_hamiltonian, effective_time, power_repetitions)
        raise NotImplementedError("Trotter orders must be positive and even for orders greater than 1")

    def _trotter(
        self, qubit_hamiltonian: QubitOperator, time: float, power_repetitions: int = 1
    ) -> UnitaryRepresentation:
        r"""Construct the unitary representation using the Trotter decomposition.

        The First Order Trotter method approximates the time evolution operator :math:`e^{-iHt}`
        by decomposing the Hamiltonian H into a sum of terms and using the product formula:
        :math:`e^{-iHt} \approx \left[\prod_i e^{-iH_i t/n}\right]^n`, where n is the number of divisions.

        The Second Order Trotter method approximates the time evolution operator :math:`e^{-iHt}`
        by decomposing the Hamiltonian H into a sum of terms and using the product formula:
        :math:`e^{-iHt} \approx \left[\prod_{i=1}^{L-1} e^{-iH_i t/2n}e^{-iH_L t/n}\prod_{i=L-1}^{1}
        e^{-iH_i t/2n}\right]^n`, where n is the number of divisions (See Strang (1968)).

        Higher order Trotter methods are constructed using the recursive Suzuki method, which builds order 2k formulas
        as: :math:`S_{2k}(t) = S_{2k-2}(u_k t)^2 S_{2k-2}((1-4u_k) t) S_{2k-2}(u_k t)^2`,
        where :math:`u_k = 1/(4-4^{1/(2k-1)})` (See Suzuki (1992)).

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.
            time: The total evolution time.
            power_repetitions: Number of times the full Trotter product is repeated
                (used by the "repeat" power strategy). Defaults to 1.

        Returns:
            UnitaryRepresentation: The unitary representation built by the Trotter decomposition.

        """
        weight_threshold = self._settings.get("weight_threshold")

        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, time)

        delta = time / num_divisions

        if qubit_hamiltonian.has_sparse_terms:
            container = self._decompose_packed_trotter_step(
                qubit_hamiltonian,
                time=delta,
                step_reps=num_divisions * power_repetitions,
                scale=time,
            )
            return UnitaryRepresentation(container=container)

        terms = self._decompose_trotter_step(qubit_hamiltonian, time=delta, atol=weight_threshold)

        num_qubits = qubit_hamiltonian.num_qubits

        container = PauliProductFormulaContainer(
            step_terms=terms,
            step_reps=num_divisions * power_repetitions,
            num_qubits=num_qubits,
            scale=time,
        )

        return UnitaryRepresentation(container=container)

    def _decompose_packed_trotter_step(
        self,
        hamiltonian: QubitOperator,
        *,
        time: float,
        step_reps: int,
        scale: float,
    ) -> PauliProductFormulaContainer:
        """Traverse partition indices without constructing sub-Hamiltonians, labels, or term objects."""
        threshold = self._settings.get("weight_threshold")
        if not hamiltonian.is_hermitian(tolerance=threshold):
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        # Match the legacy complex(c).real precision before applying the threshold.
        coefficients = np.asarray(hamiltonian.coefficients.real, dtype=np.float64)
        active = np.abs(coefficients) > threshold
        groups: list[tuple[tuple[int, ...], ...]] = []
        if np.any(active):
            partition = hamiltonian.term_partition
            if isinstance(partition, LayeredPartition):
                groups = [tuple(layer for layer in group if layer) for group in partition.groups]
                groups = [group for group in groups if group]
                # Match the dense path: sort by the number of nonempty layers, stably.
                groups.sort(key=len)
            elif isinstance(partition, FlatPartition):
                groups = [(group,) for group in partition.groups if group]
            elif partition is None:
                groups = [((index,),) for index in range(hamiltonian.num_terms)]
            else:
                raise TypeError(
                    f"Unsupported TermPartition subtype: {type(partition).__name__}. "
                    "Expected FlatPartition or LayeredPartition."
                )
        else:
            Logger.warn("No coefficients above the tolerance; returning empty term list.")

        source_offsets, source_indices, source_codes = hamiltonian.sparse_term_arrays()
        offsets = array("Q", [0])
        indices = array("I")
        codes = array("B")
        angles = array("d")
        for fraction, group_index in self._trotter_schedule(len(groups)):
            for layer in groups[group_index]:
                for term_index in layer:
                    if not active[term_index]:
                        continue
                    begin, end = int(source_offsets[term_index]), int(source_offsets[term_index + 1])
                    indices.frombytes(source_indices[begin:end].tobytes())
                    codes.frombytes(source_codes[begin:end].tobytes())
                    offsets.append(len(indices))
                    angles.append(float(coefficients[term_index]) * time * fraction)

        return PauliProductFormulaContainer.from_sparse_arrays(
            np.frombuffer(offsets, dtype=np.uint64),
            np.frombuffer(indices, dtype=np.uint32),
            np.frombuffer(codes, dtype=np.uint8),
            np.frombuffer(angles, dtype=np.float64),
            step_reps=step_reps,
            num_qubits=hamiltonian.num_qubits,
            scale=scale,
        )

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitOperator, time: float) -> int:
        """Determine the number of Trotter divisions to use.

        When both *num_divisions* and *target_accuracy* are provided, the
        larger value wins.  When neither is provided, the default is 1.

        """
        num_divisions = self._settings.get("num_divisions")
        manual = num_divisions if num_divisions > 0 else 1

        target_accuracy = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0:
            return manual

        order = self._settings.get("order")
        weight_threshold = self._settings.get("weight_threshold")

        error_bound = self._settings.get("error_bound")
        if error_bound == "commutator":
            auto = trotter_steps_commutator(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=target_accuracy,
                order=order,
                weight_threshold=weight_threshold,
            )

        else:
            auto = trotter_steps_naive(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=target_accuracy,
                order=order,
                weight_threshold=weight_threshold,
            )
        return max(manual, auto)

    def _decompose_trotter_step(
        self,
        qubit_hamiltonian: QubitOperator,
        time: float,
        *,
        atol: float = 1e-12,
    ) -> list[ExponentiatedPauliTerm]:
        """Decompose a single Trotter step into exponentiated Pauli terms.

        The order of the Trotter decomposition is taken from the settings associated
        with this builder.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be decomposed.
            time: The evolution time for the single step.
            atol: Absolute tolerance for filtering small coefficients.

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

        grouped_hamiltonians = self._group_terms(qubit_hamiltonian)

        if not grouped_hamiltonians:
            Logger.warn("Term partition produced no groups; returning empty term list.")
            return terms

        decomposed = [
            [self._commuting_pauli_maps(subgroup, atol=atol) for subgroup in group] for group in grouped_hamiltonians
        ]

        for fraction, group_index in self._trotter_schedule(len(decomposed)):
            for subgroup in decomposed[group_index]:
                terms.extend(
                    ExponentiatedPauliTerm(pauli_term=mapping, angle=coeff * time * fraction)
                    for mapping, coeff in subgroup
                )

        return terms

    def _trotter_schedule(self, num_groups: int) -> list[tuple[float, int]]:
        """Return shared Strang/Suzuki time fractions and group indices for one step."""
        if num_groups == 0:
            return []
        order = self._settings.get("order")
        if order == 1:
            return [(1.0, group_index) for group_index in range(num_groups)]

        # Strang splitting: half-time outer groups around a full-time central group.
        schedule = [(0.5, group_index) for group_index in range(num_groups - 1)]
        schedule.append((1.0, num_groups - 1))
        schedule.extend((0.5, group_index) for group_index in range(num_groups - 2, -1, -1))

        # S_{2k}(t) = S_{2k-2}(u_k t)^2 S_{2k-2}((1-4u_k)t) S_{2k-2}(u_k t)^2.
        for k in range(2, order // 2 + 1):
            u_k = 1 / (4 - 4 ** (1 / (2 * k - 1)))
            schedule = [
                (fraction * factor, group_index)
                for factor in (u_k, u_k, 1 - 4 * u_k, u_k, u_k)
                for fraction, group_index in schedule
            ]

        reduced: list[tuple[float, int]] = []
        for fraction, group_index in schedule:
            if reduced and reduced[-1][1] == group_index:
                reduced[-1] = (reduced[-1][0] + fraction, group_index)
            else:
                reduced.append((fraction, group_index))
        return reduced

    def name(self) -> str:
        """Return the name of the unitary builder."""
        return "trotter"

    def type_name(self) -> str:
        """Return unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"
