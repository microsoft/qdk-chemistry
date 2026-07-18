"""QDK/Chemistry implementation of partially randomized product formula builder.

This module implements a hybrid approach that combines deterministic Trotter decomposition
for large-weight terms with qDRIFT-style randomized sampling for small-weight terms.
This is particularly effective for quantum chemistry Hamiltonians where a few terms
dominate the weight while many small terms form a long tail.

The method is based on:
    Günther, J., Witteveen, F., et al. (2025). Phase estimation with partially
    randomized time evolution. https://doi.org/10.48550/arXiv.2503.05647

See Also:
    - Hagan, M., & Wiebe, N. (2023). Composite randomized benchmarking.
    - Ouyang, Y., et al. (2020). Compilation by stochastic Hamiltonian sparsification.
    - Jin, P., et al. (2023). Partially randomized Hamiltonian simulation.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math

import numpy as np

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionSettings
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.qdrift import QDrift
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.qdrift_error import (
    qdrift_samples_campbell,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter_error import (
    trotter_steps_commutator,
    trotter_steps_naive,
)
from qdk_chemistry.data import QubitOperator, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.definitions import ACCURACY_SPLIT_MAX, ACCURACY_SPLIT_MIN
from qdk_chemistry.utils.pauli_commutation import get_commutation_checker

__all__: list[str] = ["PartiallyRandomized", "PartiallyRandomizedSettings"]


class PartiallyRandomizedSettings(TimeEvolutionSettings):
    """Settings for partially randomized product formula builder.

    The partially randomized method splits the Hamiltonian into:
    - H_D: Large-weight terms treated deterministically (Trotter)
    - H_R: Small-weight terms treated randomly (qDRIFT-style)

    This hybrid approach benefits from:
    - Better ε-scaling from deterministic treatment of dominant terms
    - Reduced circuit depth from random sampling of the long tail
    """

    def __init__(self):
        r"""Initialize PartiallyRandomizedSettings with default values.

        Attributes:
            weight_threshold: Terms with \|h_j\| >= threshold are treated deterministically.
                Use -1.0 for automatic determination (top 10% of terms by weight).
            trotter_order: Order of Trotter formula for deterministic terms (1 or 2).
            num_random_samples: Number of random samples for the randomized part.
            target_accuracy: Target accuracy ε for automatic parameterization
                (0.0 disables it and preserves the legacy single-step behavior).
            accuracy_split: Fraction of the squared error budget assigned to the
                deterministic part (ε_D² = s·ε², ε_R² = (1 - s)·ε²).
            trotter_error_bound: Error bound used to size the outer Trotter step
                count ('commutator' or 'naive').
            seed: Random seed for reproducibility. Use -1 for non-deterministic.
            tolerance: Absolute tolerance for filtering small coefficients.

        """
        super().__init__()
        self._set_default(
            "weight_threshold",
            "float",
            -1.0,
            r"Terms with \|h_j\| >= threshold are treated deterministically. Use -1.0 for automatic.",
        )
        self._set_default(
            "trotter_order",
            "int",
            2,
            "Order of Trotter formula for deterministic terms (1 or 2).",
            (1, 2),
        )
        self._set_default(
            "num_random_samples",
            "int",
            100,
            "Number of random samples for the randomized part (H_R). "
            "Acts as a per-step floor when target_accuracy is set.",
        )
        self._set_default(
            "target_accuracy",
            "double",
            0.0,
            "Target accuracy ε for automatic parameterization (0.0 disables it). "
            "Splits into deterministic (ε_D) and random (ε_R) budgets in quadrature.",
        )
        self._set_default(
            "accuracy_split",
            "double",
            0.5,
            "Fraction s of the squared error budget given to the deterministic part: "
            "ε_D² = s·ε², ε_R² = (1 - s)·ε². Clamped to (0, 1).",
        )
        self._set_default(
            "trotter_error_bound",
            "string",
            "commutator",
            "Error bound for sizing the outer Trotter step count ('commutator' or 'naive').",
            ["commutator", "naive"],
        )
        self._set_default(
            "seed",
            "int",
            -1,
            "Random seed for reproducibility. Use -1 for non-deterministic.",
        )
        self._set_default(
            "tolerance",
            "float",
            1e-12,
            "Absolute tolerance for filtering small coefficients.",
        )
        self._set_default(
            "merge_duplicate_terms",
            "bool",
            True,
            "Fuse identical Pauli terms within consecutive commuting runs of the random block.",
        )
        self._set_default(
            "commutation_type",
            "string",
            "general",
            "Commutation check for merging: 'qubit_wise' (per-qubit) or 'general' (standard Pauli).",
            ["qubit_wise", "general"],
        )


class PartiallyRandomized(QDrift):
    r"""Partially randomized product formula builder.

    Implements a hybrid Hamiltonian simulation method that combines deterministic
    Trotter decomposition with randomized sampling. The Hamiltonian is split as:

    .. math::

        H = H_D + H_R = \sum_{l=1}^{L_D} H_l + \sum_{m=1}^{M} h_m P_m

    where:
    - :math:`H_D` contains the :math:`L_D` largest-weight terms, treated with Trotter
    - :math:`H_R` contains the remaining terms, treated with qDRIFT-style sampling

    The method is effective when:
    - :math:`\lambda_R = \sum_m |h_m| \ll \lambda` (random part has small total weight)
    - :math:`L_D \ll M` (few deterministic terms, many random terms)

    For a second-order Trotter formula, each step applies the deterministic
    part :math:`H_D` with half-angles in a symmetric (palindromic) sweep, with the
    randomized part :math:`H_R` sandwiched in between. The first-order variant
    applies :math:`H_D` at full angle followed by :math:`H_R`.

    When ``target_accuracy`` (ε) is set, the builder becomes accuracy-aware. The
    error budget is split in quadrature into a deterministic part ε_D and a
    random part ε_R (``ε_D² + ε_R² = ε²``, controlled by ``accuracy_split``).
    The evolution is divided into ``r`` outer Trotter steps sized from ε_D
    (reusing the Trotter error bounds), and the qDRIFT sample count is sized
    from ε_R (Campbell 2019 bound). Each of the ``r`` steps draws a *fresh*
    independent qDRIFT block, which is required for the randomized error to add
    up correctly across steps. With ``target_accuracy = 0.0`` (the default) the
    builder uses a single step (``r = 1``) with ``num_random_samples`` samples,
    preserving the legacy behavior.

    The total cost scales as
    :math:`O(\lambda_R^2 / \epsilon^2)` Pauli rotations for the randomized part,
    where :math:`\lambda_R = \sum_m |h_m|` is the 1-norm of :math:`H_R`
    (Theorem 2 of :cite:`Guenther2025`).

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> # Create a partially randomized builder
        >>> builder = create(
        ...     "hamiltonian_unitary_builder",
        ...     "partially_randomized",
        ...     weight_threshold=0.1,  # Terms with |h_j| >= 0.1 treated deterministically
        ...     num_random_samples=200,
        ...     trotter_order=2,
        ...     time=1.0,
        ... )
        >>> unitary_rep = builder.run(qubit_hamiltonian)
        >>> # Or let a target accuracy size the steps and samples automatically
        >>> builder = create(
        ...     "hamiltonian_unitary_builder",
        ...     "partially_randomized",
        ...     target_accuracy=1e-3,
        ...     trotter_order=2,
        ...     time=1.0,
        ... )
        >>> unitary_rep = builder.run(qubit_hamiltonian)

    References:
        Günther, J., Witteveen, F., et al. Phase estimation with partially
        randomized time evolution.

    """

    def __init__(
        self,
        *,
        time: float = 0.0,
        weight_threshold: float = -1.0,
        trotter_order: int = 2,
        num_random_samples: int = 100,
        target_accuracy: float = 0.0,
        accuracy_split: float = 0.5,
        trotter_error_bound: str = "commutator",
        seed: int = -1,
        tolerance: float = 1e-12,
        merge_duplicate_terms: bool = True,
        commutation_type: str = "general",
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        r"""Initialize partially randomized builder with specified settings.

        Args:
            time: The evolution time. Defaults to 0.0.
            weight_threshold: Terms with \|h_j\| >= threshold are treated
                deterministically with Trotter. Use -1.0 for automatic
                determination (top 10% of terms by weight, or a cost-optimal
                split when ``target_accuracy`` is set).
            trotter_order: Order of Trotter formula for deterministic part.
                1 = first order, 2 = second order (symmetric). Defaults to 2.
            num_random_samples: Number of random samples for the qDRIFT-style
                treatment of H_R. When ``target_accuracy`` is set, this acts as
                a per-step floor on the sample count. Defaults to 100.
            target_accuracy: Target accuracy ε for automatic parameterization of
                the outer Trotter step count and the qDRIFT sample count. Use
                ``0.0`` (default) to disable and preserve the legacy single-step
                behavior driven solely by ``num_random_samples``.
            accuracy_split: Fraction ``s`` of the squared error budget assigned
                to the deterministic part, with ``ε_D² = s·ε²`` and
                ``ε_R² = (1 - s)·ε²``. Larger values spend more budget on the
                deterministic Trotter part (fewer steps, more samples); smaller
                values do the reverse. Clamped to ``(0, 1)``. Defaults to 0.5.
            trotter_error_bound: Error bound used to size the outer Trotter step
                count when ``target_accuracy`` is set. ``"commutator"`` (default)
                is tighter; ``"naive"`` is cheaper to compute but looser.
            seed: Random seed for reproducibility. Use -1 for non-deterministic.
                Defaults to -1.
            tolerance: Threshold for filtering negligible coefficients.
                Defaults to 1e-12.
            merge_duplicate_terms: If ``True``, identical Pauli terms within
                consecutive mutually-commuting runs in the random block
                are fused to reduce circuit depth.  Distinct commuting
                terms are kept separate.  Defaults to ``True``.
            commutation_type: Commutation check used when merging duplicate
                terms.  ``"qubit_wise"`` requires every single-qubit
                pair to commute individually — stricter but always safe.
                ``"general"`` (default) uses standard Pauli commutation (even number of
                anti-commuting positions), which allows larger merge groups.
            power: The power to raise the unitary to. Defaults to 1.
            power_strategy: Strategy for U^power: ``"rescale"`` scales
                time, ``"repeat"`` repeats the circuit. Defaults to ``"repeat"``.

        """
        super().__init__()
        self._settings = PartiallyRandomizedSettings()
        self._settings.set("time", time)
        self._settings.set("power", power)
        self._settings.set("power_strategy", power_strategy)
        self._settings.set("weight_threshold", weight_threshold)
        self._settings.set("trotter_order", trotter_order)
        self._settings.set("num_random_samples", num_random_samples)
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("accuracy_split", accuracy_split)
        self._settings.set("trotter_error_bound", trotter_error_bound)
        self._settings.set("seed", seed)
        self._settings.set("tolerance", tolerance)
        self._settings.set("merge_duplicate_terms", merge_duplicate_terms)
        self._settings.set("commutation_type", commutation_type)

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> UnitaryRepresentation:
        r"""Construct the unitary representation using partially randomized product formula.

        The algorithm:

        1. Sort Hamiltonian terms by coefficient magnitude.
        2. Split into H_D (deterministic, large terms) and H_R (random, small terms).
        3. Size the outer Trotter step count ``r`` from the deterministic error
           budget ε_D and the per-step qDRIFT sample count from the random
           budget ε_R (only when ``target_accuracy`` is set).
        4. Build ``r`` independent sandwiches, each a single (first- or
           second-order) Trotter step with a freshly sampled qDRIFT block for
           H_R, and concatenate them.

        When ``target_accuracy`` is ``0.0`` (the default) this reduces to a
        single sandwich (``r = 1``) using ``num_random_samples`` samples,
        matching the legacy behavior.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.

        Returns:
            UnitaryRepresentation: The unitary representation built by the
                partially randomized method.

        """
        effective_time, power_repetitions = self._resolve_power()
        time: float = effective_time
        tolerance: float = self._settings.get("tolerance")
        trotter_order: int = self._settings.get("trotter_order")
        seed: int = self._settings.get("seed")
        rng = np.random.default_rng(seed if seed >= 0 else None)

        if not qubit_hamiltonian.is_hermitian(tolerance=tolerance):
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        if trotter_order not in (1, 2):
            raise NotImplementedError(
                f"Only first-order (1) and second-order (2) Trotter decompositions are supported, "
                f"but got trotter_order={trotter_order}."
            )

        # Get non-negligible real terms, sorted by descending weight
        real_terms = qubit_hamiltonian.get_real_coefficients(tolerance=tolerance, sort_by_magnitude=True)

        if len(real_terms) == 0:
            # Identity evolution
            return UnitaryRepresentation(
                container=PauliProductFormulaContainer(
                    step_terms=[],
                    step_reps=1,
                    num_qubits=qubit_hamiltonian.num_qubits,
                    scale=time,
                )
            )

        # Determine split between deterministic and random terms
        num_deterministic = self._determine_num_deterministic(qubit_hamiltonian, real_terms, time)

        # Split into H_D and H_R
        deterministic_terms = real_terms[:num_deterministic]
        random_terms = real_terms[num_deterministic:]

        # Size the outer Trotter step count r and the per-step sample count.
        # With no deterministic terms there is no Trotter bias, so a single
        # step suffices (the qDRIFT sample count alone controls accuracy).
        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, time) if deterministic_terms else 1
        num_block_samples = self._resolve_block_samples(random_terms, time, num_divisions)
        delta = time / num_divisions

        # Build r independent sandwiches with freshly sampled qDRIFT blocks.
        all_terms: list[ExponentiatedPauliTerm] = []
        for _ in range(num_divisions):
            all_terms.extend(
                self._build_step_terms(deterministic_terms, random_terms, delta, trotter_order, num_block_samples, rng)
            )

        return UnitaryRepresentation(
            container=PauliProductFormulaContainer(
                step_terms=all_terms,
                step_reps=power_repetitions,
                num_qubits=qubit_hamiltonian.num_qubits,
                scale=time,
            )
        )

    def _build_step_terms(
        self,
        deterministic_terms: list[tuple[str, float]],
        random_terms: list[tuple[str, float]],
        delta: float,
        trotter_order: int,
        num_block_samples: int,
        rng: np.random.Generator,
    ) -> list[ExponentiatedPauliTerm]:
        r"""Build one Trotter step (sandwich) for a sub-interval of length ``delta``.

        For a second-order step the deterministic terms are applied with
        half-angle ``coeff·δ/2`` in a forward sweep, a freshly sampled qDRIFT
        block for H_R (evolving for time ``δ``) is placed in the middle, and the
        deterministic terms are applied again in reverse order with the same
        half-angle. The first-order step applies the deterministic terms once at
        full angle ``coeff·δ`` followed by the qDRIFT block. The qDRIFT block is
        sampled independently on every call, which is required for the qDRIFT
        error to add up correctly across the ``r`` steps.

        Args:
            deterministic_terms: ``(label, coeff)`` pairs treated deterministically.
            random_terms: ``(label, coeff)`` pairs treated by qDRIFT sampling.
            delta: Evolution time for this single step (``time / r``).
            trotter_order: 1 or 2.
            num_block_samples: Number of qDRIFT samples in this step's random block.
            rng: Random number generator (advanced in place for fresh samples).

        Returns:
            The exponentiated Pauli terms for this single step.

        """
        step_terms: list[ExponentiatedPauliTerm] = []

        if trotter_order == 2:
            half_delta = delta / 2.0

            # Forward sweep of deterministic terms (half angle)
            for label, coeff in deterministic_terms:
                mapping = self._pauli_label_to_map(label)
                step_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=coeff * half_delta))

            # Freshly sampled qDRIFT block for H_R, evolving for time delta
            step_terms.extend(self._sample_random_block(random_terms, delta, num_block_samples, rng))

            # Backward sweep of deterministic terms (half angle, reversed order)
            for label, coeff in reversed(deterministic_terms):
                mapping = self._pauli_label_to_map(label)
                step_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=coeff * half_delta))
        else:
            # First-order: deterministic terms at full angle, then qDRIFT block
            for label, coeff in deterministic_terms:
                mapping = self._pauli_label_to_map(label)
                step_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=coeff * delta))

            step_terms.extend(self._sample_random_block(random_terms, delta, num_block_samples, rng))

        return step_terms

    def _sample_random_block(
        self,
        random_terms: list[tuple[str, float]],
        delta: float,
        num_block_samples: int,
        rng: np.random.Generator,
    ) -> list[ExponentiatedPauliTerm]:
        """Sample one qDRIFT block for H_R and optionally fuse duplicate terms."""
        block = self._sample_qdrift_terms(random_terms, delta, num_block_samples, rng)
        if self._settings.get("merge_duplicate_terms"):
            commute_fn = get_commutation_checker(self._settings.get("commutation_type"))
            block = self._merge_duplicate_terms(block, commute_fn=commute_fn)
        return block

    def _split_accuracy(self) -> tuple[float, float]:
        r"""Split the target accuracy ε into deterministic and random budgets.

        Uses a quadrature split (matching Eq. 46 of :cite:`Guenther2025`):

        .. math::

            \epsilon_D = \sqrt{s}\,\epsilon, \qquad
            \epsilon_R = \sqrt{1 - s}\,\epsilon, \qquad
            \epsilon_D^2 + \epsilon_R^2 = \epsilon^2

        where ``s`` is ``accuracy_split`` clamped to ``(0, 1)``.

        Returns:
            The pair ``(ε_D, ε_R)``. Both are ``0.0`` when ``target_accuracy``
            is disabled (``<= 0``).

        """
        target_accuracy: float = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0:
            return 0.0, 0.0
        split: float = self._settings.get("accuracy_split")
        split = min(max(split, ACCURACY_SPLIT_MIN), ACCURACY_SPLIT_MAX)
        eps_d = math.sqrt(split) * target_accuracy
        eps_r = math.sqrt(1.0 - split) * target_accuracy
        return eps_d, eps_r

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitOperator, time: float) -> int:
        """Determine the number of outer Trotter steps ``r``.

        Returns 1 when accuracy-aware sizing is disabled (``target_accuracy <= 0``)
        or the evolution time is zero. Otherwise the deterministic error budget
        ε_D drives the step count via the configured Trotter error bound. The
        bound is computed from the full Hamiltonian (treating it as fully
        deterministic), which is the heuristic recommended in
        :cite:`Guenther2025` (Sec. VI.2 / App. D.1).
        """
        eps_d, _ = self._split_accuracy()
        if eps_d <= 0.0 or time == 0.0:
            return 1

        order: int = self._settings.get("trotter_order")
        tolerance: float = self._settings.get("tolerance")
        error_bound: str = self._settings.get("trotter_error_bound")
        if error_bound == "commutator":
            return trotter_steps_commutator(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=eps_d,
                order=order,
                weight_threshold=tolerance,
            )
        return trotter_steps_naive(
            hamiltonian=qubit_hamiltonian,
            time=time,
            target_accuracy=eps_d,
            order=order,
            weight_threshold=tolerance,
        )

    def _resolve_block_samples(
        self,
        random_terms: list[tuple[str, float]],
        time: float,
        num_divisions: int,
    ) -> int:
        r"""Determine the qDRIFT sample count per step.

        The total qDRIFT sample budget over the whole evolution is sized from
        the random error budget ε_R via the Campbell (2019) bound
        ``N_total = ceil(2 λ_R² t² / ε_R)``, then distributed across the ``r``
        steps as ``ceil(N_total / r)``. ``num_random_samples`` is a per-step
        floor (mirroring the parent qDRIFT builder), so the larger value wins.

        Returns ``num_random_samples`` when accuracy-aware sizing is disabled,
        there are no random terms, or the evolution time is zero.
        """
        num_random_samples: int = self._settings.get("num_random_samples")
        if num_random_samples <= 0:
            raise ValueError(f"num_random_samples must be a positive integer, got {num_random_samples}.")

        _, eps_r = self._split_accuracy()
        if eps_r <= 0.0 or len(random_terms) == 0 or time == 0.0:
            return num_random_samples

        tolerance: float = self._settings.get("tolerance")
        h_random = QubitOperator(
            pauli_strings=[label for label, _ in random_terms],
            coefficients=np.array([coeff for _, coeff in random_terms]),
        )
        num_total = qdrift_samples_campbell(
            hamiltonian=h_random,
            time=time,
            target_accuracy=eps_r,
            weight_threshold=tolerance,
        )
        per_step = math.ceil(num_total / num_divisions)
        return max(num_random_samples, per_step)

    def _determine_num_deterministic(
        self,
        qubit_hamiltonian: QubitOperator,
        terms: list[tuple[str, float]],
        time: float,
    ) -> int:
        """Determine how many terms to treat deterministically.

        Precedence:

        1. An explicit ``weight_threshold >= 0`` always wins: count the terms
           whose magnitude meets the threshold.
        2. Otherwise, when ``target_accuracy`` is set, pick the cost-optimal
           split ``L_D`` that minimizes the total rotation count.
        3. Otherwise, fall back to the legacy heuristic (top 10% by weight).

        Args:
            qubit_hamiltonian: The full Hamiltonian (used to size ``r`` for the
                cost-optimal split).
            terms: List of ``(label, coeff)`` sorted by ``|coeff|`` descending.
            time: The evolution time.

        Returns:
            Number of terms to treat deterministically.

        """
        weight_threshold: float = self._settings.get("weight_threshold")
        if weight_threshold >= 0.0:
            # Count terms with |coeff| >= threshold; may be 0 (all-random)
            return sum(1 for _, c in terms if abs(c) >= weight_threshold)

        target_accuracy: float = self._settings.get("target_accuracy")
        if target_accuracy > 0.0 and time != 0.0:
            return self._determine_num_deterministic_cost_optimal(qubit_hamiltonian, terms, time)

        # Default: top 10% of terms (at least 1, at most all but 1 for random)
        num_deterministic = max(1, len(terms) // 10)
        # Ensure at least some terms remain for random treatment
        if num_deterministic >= len(terms):
            num_deterministic = max(1, len(terms) - 1)
        return num_deterministic

    def _determine_num_deterministic_cost_optimal(
        self,
        qubit_hamiltonian: QubitOperator,
        terms: list[tuple[str, float]],
        time: float,
    ) -> int:
        r"""Pick the split ``L_D`` minimizing a rotation-count cost proxy.

        With terms sorted by descending weight, assigning the ``L_D`` largest to
        H_D leaves ``λ_R(L_D) = Σ_{j > L_D} |h_j|`` for the random tail. The cost
        proxy combines the deterministic and random rotation counts
        (:cite:`Guenther2025`, Theorem V.1):

        .. math::

            G(L_D) = N_\text{stage}\,L_D\,r
                     + \frac{2\,\lambda_R(L_D)^2\,t^2}{\epsilon_R}

        where ``r`` (≈ constant in ``L_D``) is the outer Trotter step count and
        ``N_stage`` is 2 for second order and 1 for first order. The minimizer is
        returned; ties favor the smaller ``L_D``.

        Args:
            qubit_hamiltonian: The full Hamiltonian (used to size ``r``).
            terms: ``(label, coeff)`` pairs sorted by ``|coeff|`` descending.
            time: The evolution time.

        Returns:
            The cost-optimal number of deterministic terms (in ``[0, len(terms)]``).

        """
        eps_d, eps_r = self._split_accuracy()
        order: int = self._settings.get("trotter_order")
        n_stage = 2 if order == 2 else 1

        num_terms = len(terms)
        abs_coeffs = [abs(c) for _, c in terms]

        # Suffix sums: lambda_r[ld] = sum of |coeff| for terms[ld:]
        lambda_r = [0.0] * (num_terms + 1)
        for i in range(num_terms - 1, -1, -1):
            lambda_r[i] = lambda_r[i + 1] + abs_coeffs[i]

        # r is computed from the full Hamiltonian and is ~constant in L_D.
        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, time) if eps_d > 0.0 else 1

        best_ld = 0
        best_cost = math.inf
        for ld in range(num_terms + 1):
            lam_r = lambda_r[ld]
            g_random = 2.0 * (lam_r * time) ** 2 / eps_r if (eps_r > 0.0 and lam_r > 0.0) else 0.0
            g_det = n_stage * ld * num_divisions
            cost = g_det + g_random
            if cost < best_cost:
                best_cost = cost
                best_ld = ld
        return best_ld

    def name(self) -> str:
        """Return the name of the unitary builder."""
        return "partially_randomized"

    def type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"
