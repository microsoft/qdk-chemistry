"""QDK/Chemistry implementation of partially randomized product formula builder.

This module implements a hybrid approach that combines deterministic Trotter decomposition
for large-weight terms with qDRIFT-style randomized sampling for small-weight terms.
This is particularly effective for quantum chemistry Hamiltonians where a few terms
dominate the weight while many small terms form a long tail.

The method is based on:
    Günther, J., Witteveen, F., et al. (2024). Phase estimation with partially
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

import numpy as np

from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.builder.qdrift import QDrift
from qdk_chemistry.data import QubitHamiltonian, Settings, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

__all__: list[str] = ["PartiallyRandomized", "PartiallyRandomizedSettings"]


class PartiallyRandomizedSettings(Settings):
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
        )
        self._set_default(
            "num_random_samples",
            "int",
            100,
            "Number of random samples for the randomized part (H_R).",
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
            "merge_commuting",
            "bool",
            True,
            "Merge consecutive commuting random samples to reduce circuit depth.",
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

    The total cost scales as
    :math:`O(\lambda_R^2 / \epsilon^2)` Pauli rotations for the randomized part,
    where :math:`\lambda_R = \sum_m |h_m|` is the 1-norm of :math:`H_R`
    (Theorem 2 of :cite:`Guenther2025`).

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> # Create a partially randomized builder
        >>> builder = create(
        ...     "time_evolution_builder",
        ...     "partially_randomized",
        ...     weight_threshold=0.1,  # Terms with |h_j| >= 0.1 treated deterministically
        ...     num_random_samples=200,
        ...     trotter_order=2,
        ... )
        >>> time_evolution = builder.run(qubit_hamiltonian, time=1.0)

    References:
        Günther, J., Witteveen, F., et al. Phase estimation with partially
        randomized time evolution.

    """

    def __init__(
        self,
        weight_threshold: float = -1.0,
        trotter_order: int = 2,
        num_random_samples: int = 100,
        seed: int = -1,
        tolerance: float = 1e-12,
        merge_commuting: bool = True,
    ):
        r"""Initialize partially randomized builder with specified settings.

        Args:
            weight_threshold: Terms with \|h_j\| >= threshold are treated
                deterministically with Trotter. Use -1.0 for automatic
                determination (top 10% of terms by weight).
            trotter_order: Order of Trotter formula for deterministic part.
                1 = first order, 2 = second order (symmetric). Defaults to 2.
            num_random_samples: Number of random samples for the qDRIFT-style
                treatment of H_R. Defaults to 100.
            seed: Random seed for reproducibility. Use -1 for non-deterministic.
                Defaults to -1.
            tolerance: Threshold for filtering negligible coefficients.
                Defaults to 1e-12.
            merge_commuting: If ``True``, consecutive mutually-commuting
                sampled terms in the random block are fused to reduce
                circuit depth.  Defaults to ``True``.

        """
        # Bypass QDrift.__init__ which creates QDriftSettings; we have our
        # own settings class.
        TimeEvolutionBuilder.__init__(self)
        self._settings = PartiallyRandomizedSettings()
        self._settings.set("weight_threshold", weight_threshold)
        self._settings.set("trotter_order", trotter_order)
        self._settings.set("num_random_samples", num_random_samples)
        self._settings.set("seed", seed)
        self._settings.set("tolerance", tolerance)
        self._settings.set("merge_commuting", merge_commuting)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        r"""Construct the time evolution unitary using partially randomized product formula.

        The algorithm:
        1. Sort Hamiltonian terms by coefficient magnitude
        2. Split into H_D (deterministic, large terms) and H_R (random, small terms)
        3. Apply first- or second-order Trotter structure with qDRIFT for the H_R block

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.
            time: The total evolution time (δ in the formula).

        Returns:
            TimeEvolutionUnitary: The time evolution unitary built by the
                partially randomized method.

        """
        tolerance: float = self._settings.get("tolerance")
        trotter_order: int = self._settings.get("trotter_order")
        num_random_samples: int = self._settings.get("num_random_samples")
        seed: int = self._settings.get("seed")
        rng = np.random.default_rng(seed if seed >= 0 else None)

        if not qubit_hamiltonian.is_hermitian(tolerance=tolerance):
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        # Get non-negligible real terms, sorted by descending weight
        real_terms = qubit_hamiltonian.get_real_coefficients(tolerance=tolerance, sort_by_magnitude=True)

        if len(real_terms) == 0:
            # Identity evolution
            return TimeEvolutionUnitary(
                container=PauliProductFormulaContainer(
                    step_terms=[],
                    step_reps=1,
                    num_qubits=qubit_hamiltonian.num_qubits,
                )
            )

        # Determine split between deterministic and random terms
        num_deterministic = self._determine_num_deterministic(real_terms)

        # Split into H_D and H_R
        deterministic_terms = real_terms[:num_deterministic]
        random_terms = real_terms[num_deterministic:]

        # Build the product formula
        all_terms: list[ExponentiatedPauliTerm] = []

        if trotter_order == 2:
            # Second-order: e^{-iδH_1/2} ... e^{-iδH_L/2} e^{-iδH_R} e^{-iδH_L/2} ... e^{-iδH_1/2}
            half_time = time / 2.0

            # Forward sweep of deterministic terms (half angle)
            for label, coeff in deterministic_terms:
                mapping = self._pauli_label_to_map(label)
                angle = coeff * half_time
                all_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

            # Random part (full time)
            random_part_terms = self._sample_qdrift_terms(random_terms, time, num_random_samples, rng)
            if self._settings.get("merge_commuting"):
                random_part_terms = self._merge_commuting_runs(random_part_terms)
            all_terms.extend(random_part_terms)

            # Backward sweep of deterministic terms (half angle, reversed order)
            for label, coeff in reversed(deterministic_terms):
                mapping = self._pauli_label_to_map(label)
                angle = coeff * half_time
                all_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

        else:
            # First-order: e^{-iδH_1} ... e^{-iδH_L} e^{-iδH_R}
            # Deterministic terms
            for label, coeff in deterministic_terms:
                mapping = self._pauli_label_to_map(label)
                angle = coeff * time
                all_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

            # Random part
            random_part_terms = self._sample_qdrift_terms(random_terms, time, num_random_samples, rng)
            if self._settings.get("merge_commuting"):
                random_part_terms = self._merge_commuting_runs(random_part_terms)
            all_terms.extend(random_part_terms)

        return TimeEvolutionUnitary(
            container=PauliProductFormulaContainer(
                step_terms=all_terms,
                step_reps=1,
                num_qubits=qubit_hamiltonian.num_qubits,
            )
        )

    def _determine_num_deterministic(self, terms: list[tuple[str, float]]) -> int:
        """Determine how many terms to treat deterministically.

        Args:
            terms: List of (label, coeff) sorted by |coeff| descending.

        Returns:
            Number of terms to treat deterministically.

        """
        weight_threshold: float = self._settings.get("weight_threshold")
        if weight_threshold >= 0.0:
            # Count terms with |coeff| >= threshold
            count = sum(1 for _, c in terms if abs(c) >= weight_threshold)
            return max(1, count)  # At least 1 deterministic term

        # Default: top 10% of terms (at least 1, at most all but 1 for random)
        num_deterministic = max(1, len(terms) // 10)
        # Ensure at least some terms remain for random treatment
        if num_deterministic >= len(terms):
            num_deterministic = max(1, len(terms) - 1)
        return num_deterministic

    def name(self) -> str:
        """Return the name of the time evolution unitary builder."""
        return "partially_randomized"

    def type_name(self) -> str:
        """Return time_evolution_builder as the algorithm type name."""
        return "time_evolution_builder"
