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

import numpy as np

from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.builder.trotter_error import (
    trotter_steps_commutator,
    trotter_steps_naive,
)
from qdk_chemistry.data import (
    FlatPartition,
    LayeredPartition,
    QubitHamiltonian,
    Settings,
    TermPartition,
    TimeEvolutionUnitary,
)
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils import Logger

__all__: list[str] = ["Trotter", "TrotterSettings"]


class TrotterSettings(Settings):
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
        target_accuracy: float = 0.0,
        num_divisions: int = 0,
        error_bound: str = "commutator",
        weight_threshold: float = 1e-12,
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

        When the input :class:`~qdk_chemistry.data.QubitHamiltonian` carries a populated
        :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition`, the builder consumes it
        directly for schedule-level grouping.  When no partition is present, each Pauli term
        is exponentiated as its own group.

        Args:
            order: Trotter decomposition order (1, 2, or any positive even integer). Defaults to 1.
            target_accuracy: Target accuracy for auto step computation. Use 0.0 (default) to disable.
            num_divisions: Divisions per Trotter step. Max of this and auto value is used. Defaults to 0.
            error_bound: Error bound strategy: ``"commutator"`` (default) or ``"naive"``.
            weight_threshold: Threshold for filtering small coefficients. Defaults to 1e-12.

        """
        super().__init__()
        self._settings = TrotterSettings()
        self._settings.set("order", order)
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("num_divisions", num_divisions)
        self._settings.set("error_bound", error_bound)
        self._settings.set("weight_threshold", weight_threshold)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        """Construct the time evolution unitary using Trotter decomposition.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.
            time: The total evolution time.

        Returns:
            TimeEvolutionUnitary: The time evolution unitary built by the Trotter decomposition.

        """
        order = self._settings.get("order")
        if order in {1, 2} or (order > 2 and order % 2 == 0):
            return self._trotter(qubit_hamiltonian, time)
        raise NotImplementedError("Trotter orders must be positive and even for orders greater than 1")

    def _trotter(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        r"""Construct the time evolution unitary using the Trotter decomposition.

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

        Returns:
            TimeEvolutionUnitary: The time evolution unitary built by the Trotter decomposition.

        """
        weight_threshold = self._settings.get("weight_threshold")

        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, time)

        delta = time / num_divisions

        terms = self._decompose_trotter_step(qubit_hamiltonian, time=delta, atol=weight_threshold)

        num_qubits = qubit_hamiltonian.num_qubits

        container = PauliProductFormulaContainer(
            step_terms=terms,
            step_reps=num_divisions,
            num_qubits=num_qubits,
        )

        return TimeEvolutionUnitary(container=container)

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> int:
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
        qubit_hamiltonian: QubitHamiltonian,
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

        order = self._settings.get("order")
        grouped_hamiltonians = self._group_terms(qubit_hamiltonian)

        if not grouped_hamiltonians:
            Logger.warn("Term partition produced no groups; returning empty term list.")
            return terms

        if order == 1:
            for group in grouped_hamiltonians:
                for subgroup in group:
                    terms.extend(
                        self._exponentiate_commuting(
                            subgroup,
                            time=time,
                            atol=atol,
                        )
                    )

        # order = 2 or order = 2k with k>1
        else:
            # Build an abstract schedule of (time_fraction, group_index) entries.
            # The Strang splitting puts group 0..L-2 at half-time on the outside
            # and group L-1 at full-time in the middle:
            #   S2(t) = [t/2 * G0, ..., t/2 * G_{L-2}, t * G_{L-1}, t/2 * G_{L-2}, ..., t/2 * G0]
            n_groups = len(grouped_hamiltonians)
            schedule: list[tuple[float, int]] = []
            for g in range(n_groups - 1):
                schedule.append((0.5, g))
            schedule.append((1.0, n_groups - 1))
            for g in range(n_groups - 2, -1, -1):
                schedule.append((0.5, g))

            # Apply Suzuki recursion at the schedule level for order > 2
            if order > 2 and order % 2 == 0:
                for k in range(2, int(order / 2) + 1):
                    u_k = 1 / (4 - 4 ** (1 / (2 * k - 1)))
                    new_schedule: list[tuple[float, int]] = []
                    # S_{2k}(t) = S_{2k-2}(u_k t)^2 S_{2k-2}((1-4u_k) t) S_{2k-2}(u_k t)^2
                    for _ in range(2):
                        for frac, g in schedule:
                            new_schedule.append((frac * u_k, g))
                    for frac, g in schedule:
                        new_schedule.append((frac * (1 - 4 * u_k), g))
                    for _ in range(2):
                        for frac, g in schedule:
                            new_schedule.append((frac * u_k, g))
                    schedule = new_schedule

            # Reduce the schedule: merge consecutive entries with the same group index
            reduced: list[tuple[float, int]] = []
            for frac, g in schedule:
                if reduced and reduced[-1][1] == g:
                    reduced[-1] = (reduced[-1][0] + frac, g)
                else:
                    reduced.append((frac, g))
            schedule = reduced

            # Expand the schedule into exponentiated Pauli terms
            for frac, g in schedule:
                for subgroup in grouped_hamiltonians[g]:
                    terms.extend(
                        self._exponentiate_commuting(
                            subgroup,
                            time=time * frac,
                            atol=atol,
                        )
                    )

        return terms

    def _group_terms(
        self,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[list[QubitHamiltonian]]:
        """Group Hamiltonian terms for Trotter decomposition.

        When the Hamiltonian carries a populated
        :attr:`~qdk_chemistry.data.QubitHamiltonian.term_partition`, it is
        consumed directly.  Both :class:`~qdk_chemistry.data.LayeredPartition`
        and :class:`~qdk_chemistry.data.FlatPartition` are accepted.

        When no partition is present, each Pauli term is treated as its own
        single-term group with no reordering.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to group.

        Returns:
            A list of groups, where each group is a list of
            ``QubitHamiltonian`` sub-groups (parallelisable layers).

        """
        partition = qubit_hamiltonian.term_partition
        if partition is not None:
            Logger.debug(
                f"Trotter: consuming QubitHamiltonian.term_partition "
                f"(strategy={partition.strategy!r}, num_groups={partition.num_groups})."
            )
            return self._groups_from_partition(qubit_hamiltonian, partition)

        Logger.debug("Trotter: no term_partition present; treating each Pauli term as its own group.")
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
        """Materialise a :class:`TermPartition` into Trotter sub-groups.

        Both :class:`~qdk_chemistry.data.LayeredPartition` and
        :class:`~qdk_chemistry.data.FlatPartition` are supported. A flat
        partition's groups are treated as single layers. Groups are sorted by
        ascending layer count so that the smallest groups sit on the outside
        of the Strang/Suzuki splitting and merge at boundaries.

        Args:
            qubit_hamiltonian: Source Hamiltonian whose Pauli terms the partition indexes into.
            partition: Index-based partition carried on ``qubit_hamiltonian``.

        Returns:
            List of groups; each group is a list of layer ``QubitHamiltonian`` objects.

        """
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

        # Normalise to (group → tuple of layers of indices)
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
        # Drop empty groups (no layers / all layers empty).
        groups = [g for g in groups if g]

        # Sort groups by ascending layer count so the smallest sits on the
        # outside of the Strang/Suzuki splitting (maximises boundary merging).
        groups.sort(key=len)
        return groups

    def _exponentiate_commuting(
        self,
        group: QubitHamiltonian,
        time: float,
        *,
        atol: float = 1e-12,
    ) -> list[ExponentiatedPauliTerm]:
        r"""Exponentiate a group of commuting Pauli terms.

        Each term :math:`P_j` with coefficient :math:`c_j` is converted to
        the rotation :math:`e^{-i\,c_j\,t\,P_j}`.  Because all terms in the
        group commute, the product of rotations equals the exponential of
        the sum regardless of ordering.

        Args:
            group: The group of commuting Hamiltonian terms to exponentiate.
            time: The evolution time used to compute rotation angles
                (:math:`\theta_j = c_j \cdot t`).
            atol: Absolute tolerance for filtering small coefficients.

        Returns:
            A flat list of :class:`ExponentiatedPauliTerm`.

        """
        terms: list[ExponentiatedPauliTerm] = []
        for label, coeff in group.get_real_coefficients(tolerance=atol):
            mapping = self._pauli_label_to_map(label)
            angle = coeff * time
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))
        return terms

    def name(self) -> str:
        """Return the name of the time evolution unitary builder."""
        return "trotter"

    def type_name(self) -> str:
        """Return time_evolution_builder as the algorithm type name."""
        return "time_evolution_builder"
