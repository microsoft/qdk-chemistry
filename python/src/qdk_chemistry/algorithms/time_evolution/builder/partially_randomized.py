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
            num_deterministic_terms: Number of largest-weight terms to treat deterministically.
                Use -1 for automatic determination based on weight_threshold or default (top 10%).
            weight_threshold: Terms with \|h_j\| >= threshold are treated deterministically.
                Use -1.0 for automatic. Ignored if num_deterministic_terms >= 0.
            trotter_order: Order of Trotter formula for deterministic terms (1 or 2).
            num_random_samples: Number of random samples for the randomized part.
            seed: Random seed for reproducibility. Use -1 for non-deterministic.
            tolerance: Absolute tolerance for filtering small coefficients.

        """
        super().__init__()
        self._set_default(
            "num_deterministic_terms",
            "int",
            -1,
            "Number of largest-weight terms to treat deterministically. Use -1 for automatic.",
        )
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


class PartiallyRandomized(TimeEvolutionBuilder):
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

    For a second-order Trotter formula, each step has the structure:

    .. math::

        S_2(\delta) = e^{-\frac{i}{2}\delta H_1} \cdots e^{-\frac{i}{2}\delta H_{L_D}}
                      e^{-i\delta H_R}
                      e^{-\frac{i}{2}\delta H_{L_D}} \cdots e^{-\frac{i}{2}\delta H_1}

    where :math:`e^{-i\delta H_R}` is approximated using randomized sampling.

    The total cost scales as:
    - :math:`O(L_D \cdot C_{gs}^{1/p} \cdot \epsilon^{-1-1/p})` for deterministic evolutions
    - :math:`O(\lambda_R^2 \cdot \epsilon^{-2})` for Pauli rotations

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> # Create a partially randomized builder
        >>> builder = create(
        ...     "time_evolution_builder",
        ...     "partially_randomized",
        ...     num_deterministic_terms=10,  # Top 10 terms treated deterministically
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
        num_deterministic_terms: int = -1,
        weight_threshold: float = -1.0,
        trotter_order: int = 2,
        num_random_samples: int = 100,
        seed: int = -1,
        tolerance: float = 1e-12,
    ):
        r"""Initialize partially randomized builder with specified settings.

        Args:
            num_deterministic_terms: Number of largest-weight terms to treat
                deterministically with Trotter. Use -1 for automatic determination
                based on weight_threshold or default (top 10% of terms).
            weight_threshold: Terms with \|h_j\| >= threshold are treated
                deterministically. Use -1.0 for automatic. Only used if
                num_deterministic_terms is -1.
            trotter_order: Order of Trotter formula for deterministic part.
                1 = first order, 2 = second order (symmetric). Defaults to 2.
            num_random_samples: Number of random samples for the qDRIFT-style
                treatment of H_R. Defaults to 100.
            seed: Random seed for reproducibility. Use -1 for non-deterministic.
                Defaults to -1.
            tolerance: Threshold for filtering negligible coefficients.
                Defaults to 1e-12.

        """
        super().__init__()
        self._settings = PartiallyRandomizedSettings()
        self._settings.set("num_deterministic_terms", num_deterministic_terms)
        self._settings.set("weight_threshold", weight_threshold)
        self._settings.set("trotter_order", trotter_order)
        self._settings.set("num_random_samples", num_random_samples)
        self._settings.set("seed", seed)
        self._settings.set("tolerance", tolerance)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        r"""Construct the time evolution unitary using partially randomized product formula.

        The algorithm:
        1. Sort Hamiltonian terms by coefficient magnitude
        2. Split into H_D (deterministic, large terms) and H_R (random, small terms)
        3. Apply second-order Trotter structure with qDRIFT for the H_R block

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

        # Extract and validate Hamiltonian terms
        paulis = list(qubit_hamiltonian.pauli_ops.paulis)
        coeffs_raw = qubit_hamiltonian.pauli_ops.coeffs

        terms_data: list[tuple[int, str, float]] = []  # (original_idx, label, coeff)
        for idx, (pauli, coeff) in enumerate(zip(paulis, coeffs_raw, strict=True)):
            coeff_complex = complex(coeff)
            if abs(coeff_complex.imag) > tolerance:
                raise ValueError(
                    f"Non-Hermitian Hamiltonian: coefficient {coeff} for term "
                    f"{pauli.to_label()} has nonzero imaginary part."
                )
            real_coeff = coeff_complex.real
            if abs(real_coeff) > tolerance:
                terms_data.append((idx, pauli.to_label(), real_coeff))

        if len(terms_data) == 0:
            # Identity evolution
            return TimeEvolutionUnitary(
                container=PauliProductFormulaContainer(
                    step_terms=[],
                    step_reps=1,
                    num_qubits=qubit_hamiltonian.num_qubits,
                )
            )

        # Sort terms by absolute coefficient (descending)
        terms_data.sort(key=lambda x: abs(x[2]), reverse=True)

        # Determine split between deterministic and random terms
        num_det = self._determine_num_deterministic(terms_data)

        # Split into H_D and H_R
        deterministic_terms = terms_data[:num_det]
        random_terms = terms_data[num_det:]

        # Build the product formula
        all_terms: list[ExponentiatedPauliTerm] = []

        if trotter_order == 2:
            # Second-order: e^{-iδH_1/2} ... e^{-iδH_L/2} e^{-iδH_R} e^{-iδH_L/2} ... e^{-iδH_1/2}
            half_time = time / 2.0

            # Forward sweep of deterministic terms (half angle)
            for _, label, coeff in deterministic_terms:
                mapping = self._pauli_label_to_map(label)
                angle = coeff * half_time
                all_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

            # Random part (full time)
            random_part_terms = self._build_random_terms(random_terms, time, num_random_samples, rng)
            all_terms.extend(random_part_terms)

            # Backward sweep of deterministic terms (half angle, reversed order)
            for _, label, coeff in reversed(deterministic_terms):
                mapping = self._pauli_label_to_map(label)
                angle = coeff * half_time
                all_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

        else:
            # First-order: e^{-iδH_1} ... e^{-iδH_L} e^{-iδH_R}
            # Deterministic terms
            for _, label, coeff in deterministic_terms:
                mapping = self._pauli_label_to_map(label)
                angle = coeff * time
                all_terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

            # Random part
            random_part_terms = self._build_random_terms(random_terms, time, num_random_samples, rng)
            all_terms.extend(random_part_terms)

        return TimeEvolutionUnitary(
            container=PauliProductFormulaContainer(
                step_terms=all_terms,
                step_reps=1,
                num_qubits=qubit_hamiltonian.num_qubits,
            )
        )

    def _determine_num_deterministic(self, terms_data: list[tuple[int, str, float]]) -> int:
        """Determine how many terms to treat deterministically.

        Args:
            terms_data: List of (idx, label, coeff) sorted by |coeff| descending.

        Returns:
            Number of terms to treat deterministically.

        """
        num_det_setting: int = self._settings.get("num_deterministic_terms")
        if num_det_setting >= 0:
            return min(num_det_setting, len(terms_data))

        weight_threshold: float = self._settings.get("weight_threshold")
        if weight_threshold >= 0.0:
            # Count terms with |coeff| >= threshold
            count = sum(1 for _, _, c in terms_data if abs(c) >= weight_threshold)
            return max(1, count)  # At least 1 deterministic term

        # Default: top 10% of terms (at least 1, at most all but 1 for random)
        num_det = max(1, len(terms_data) // 10)
        # Ensure at least some terms remain for random treatment
        if num_det >= len(terms_data):
            num_det = max(1, len(terms_data) - 1)
        return num_det

    def _build_random_terms(
        self,
        random_terms: list[tuple[int, str, float]],
        time: float,
        num_samples: int,
        rng: np.random.Generator,
    ) -> list[ExponentiatedPauliTerm]:
        """Build qDRIFT-style random samples for H_R.

        Args:
            random_terms: List of (idx, label, coeff) for terms in H_R.
            time: Evolution time for this block.
            num_samples: Number of random samples.
            rng: Random number generator.

        Returns:
            List of ExponentiatedPauliTerm for the random part.

        """
        if len(random_terms) == 0:
            return []

        # Compute λ_R and probabilities
        coeffs = np.array([c for _, _, c in random_terms])
        abs_coeffs = np.abs(coeffs)
        lambda_r = float(np.sum(abs_coeffs))

        if lambda_r < 1e-14:
            return []

        probabilities = abs_coeffs / lambda_r

        # Sample term indices
        term_indices = rng.choice(len(random_terms), size=num_samples, p=probabilities)

        # Build sampled terms
        # Each sample: exp(-i * sign(h_m) * λ_R * t / r * P_m)
        angle_magnitude = lambda_r * time / num_samples

        terms: list[ExponentiatedPauliTerm] = []
        for idx in term_indices:
            _, label, coeff = random_terms[idx]
            sign = 1.0 if coeff >= 0 else -1.0
            mapping = self._pauli_label_to_map(label)
            angle = sign * angle_magnitude
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

        return terms

    @staticmethod
    def _pauli_label_to_map(label: str) -> dict[int, str]:
        """Translate a Pauli label to a mapping ``qubit -> {X, Y, Z}``.

        Args:
            label: Pauli string label in little-endian ordering.

        Returns:
            Dictionary assigning each non-identity qubit index to its Pauli axis.

        """
        mapping: dict[int, str] = {}
        for index, char in enumerate(reversed(label)):
            if char != "I":
                mapping[index] = char
        return mapping

    def name(self) -> str:
        """Return the name of the time evolution unitary builder."""
        return "partially_randomized"

    def type_name(self) -> str:
        """Return time_evolution_builder as the algorithm type name."""
        return "time_evolution_builder"
