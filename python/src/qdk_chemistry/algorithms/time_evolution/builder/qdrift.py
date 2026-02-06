"""QDK/Chemistry implementation of the qDRIFT randomized evolution builder.

This module implements the qDRIFT algorithm for Hamiltonian simulation, which provides
an alternative to deterministic Trotter decomposition by using randomized sampling.

References:
    Campbell, E. (2019). Random Compiler for Fast Hamiltonian Simulation.
    Physical Review Letters, 123(7), 070503. https://arxiv.org/abs/1811.08017

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.data import QubitHamiltonian, Settings, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

__all__: list[str] = ["QDrift", "QDriftSettings"]


class QDriftSettings(Settings):
    """Settings for qDRIFT randomized decomposition builder.

    The qDRIFT algorithm approximates the time evolution operator using randomized
    sampling of Hamiltonian terms. The error scales as O(λ²t²/N), where λ is the
    1-norm of the Hamiltonian coefficients, t is evolution time, and N is the
    number of samples.
    """

    def __init__(self):
        """Initialize QDriftSettings with default values.

        Attributes:
            num_samples: Number of random samples N. More samples = higher accuracy.
                         Error scales as O(λ²t²/N).
            seed: Random seed for reproducibility. None for non-deterministic behavior.
            tolerance: Absolute tolerance for filtering small coefficients.

        """
        super().__init__()
        self._set_default(
            "num_samples",
            "int",
            100,
            "Number of random samples N. Error scales as O(λ²t²/N).",
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


class QDrift(TimeEvolutionBuilder):
    r"""qDRIFT randomized product formula builder.

    Implements the qDRIFT algorithm from Campbell (2019), which approximates the
    time evolution operator :math:`U(t) = e^{-iHt}` using randomized sampling of
    Hamiltonian terms.

    Instead of applying all Hamiltonian terms in a fixed sequence (as in Trotter
    decomposition), qDRIFT randomly samples terms with probability proportional
    to their coefficient magnitudes. This can achieve better gate complexity for
    Hamiltonians with many terms.

    The algorithm works as follows:

    1. Compute :math:`\lambda = \sum_j |h_j|` (1-norm of coefficients)
    2. Build probability distribution :math:`p_j = |h_j| / \lambda`
    3. Sample N terms according to this distribution
    4. Each sample contributes :math:`e^{-i \cdot \text{sign}(h_j) \cdot \lambda t / N \cdot P_j}`

    The approximation error is bounded by :math:`\epsilon \leq 2\lambda^2 t^2 / N`.

    Attributes:
        num_samples: Number of random samples to draw.
        seed: Random seed for reproducibility.
        tolerance: Threshold for filtering negligible coefficients.

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> # Create a qDRIFT builder with 500 samples
        >>> qdrift = create("time_evolution_builder", "qdrift", num_samples=500, seed=42)
        >>> # Use it to build time evolution for a Hamiltonian
        >>> time_evolution = qdrift.run(qubit_hamiltonian, time=1.0)

    References:
        Campbell, E. (2019). Random Compiler for Fast Hamiltonian Simulation.
        https://arxiv.org/abs/1811.08017

    """

    def __init__(
        self,
        num_samples: int = 100,
        seed: int = -1,
        tolerance: float = 1e-12,
    ):
        """Initialize qDRIFT builder with specified settings.

        Args:
            num_samples: Number of random samples N. More samples increase accuracy
                but also increase circuit depth. Error scales as O(λ²t²/N).
                Defaults to 100.
            seed: Random seed for reproducibility. Use -1 for non-deterministic
                sampling. Defaults to -1.
            tolerance: Absolute threshold for filtering small Hamiltonian
                coefficients. Defaults to 1e-12.

        """
        super().__init__()
        self._settings = QDriftSettings()
        self._settings.set("num_samples", num_samples)
        self._settings.set("seed", seed)
        self._settings.set("tolerance", tolerance)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        r"""Construct the time evolution unitary using qDRIFT randomized sampling.

        The qDRIFT method approximates :math:`e^{-iHt}` by:

        1. Computing :math:`\lambda = \sum_j |h_j|`
        2. Sampling N term indices with probability :math:`p_j = |h_j|/\lambda`
        3. For each sampled term j, applying :math:`e^{-i \cdot \text{sign}(h_j) \cdot \lambda t / N \cdot P_j}`

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.
            time: The total evolution time.

        Returns:
            TimeEvolutionUnitary: The time evolution unitary built by qDRIFT sampling.

        """
        seed: int = self._settings.get("seed")
        rng = np.random.default_rng(seed if seed >= 0 else None)

        num_samples: int = self._settings.get("num_samples")
        tolerance: float = self._settings.get("tolerance")

        # Extract Hamiltonian terms and coefficients
        paulis = list(qubit_hamiltonian.pauli_ops.paulis)
        coeffs_raw = qubit_hamiltonian.pauli_ops.coeffs

        # Convert coefficients to real values and validate Hermiticity
        coeffs: list[float] = []
        for idx, coeff in enumerate(coeffs_raw):
            coeff_complex = complex(coeff)
            if abs(coeff_complex.imag) > tolerance:
                raise ValueError(
                    f"Non-Hermitian Hamiltonian: coefficient {coeff} for term "
                    f"{paulis[idx].to_label()} has nonzero imaginary part."
                )
            coeffs.append(coeff_complex.real)

        # Filter small terms
        filtered_paulis: list = []
        filtered_coeffs: list[float] = []
        for pauli, coeff in zip(paulis, coeffs, strict=True):
            if abs(coeff) > tolerance:
                filtered_paulis.append(pauli)
                filtered_coeffs.append(coeff)

        if len(filtered_coeffs) == 0:
            # Identity evolution (no significant terms)
            return TimeEvolutionUnitary(
                container=PauliProductFormulaContainer(
                    step_terms=[],
                    step_reps=1,
                    num_qubits=qubit_hamiltonian.num_qubits,
                )
            )

        # Compute λ (1-norm) and probabilities
        abs_coeffs = np.array([abs(c) for c in filtered_coeffs])
        lambda_norm = float(np.sum(abs_coeffs))
        probabilities = abs_coeffs / lambda_norm

        # Sample N term indices according to the probability distribution
        term_indices = rng.choice(len(filtered_coeffs), size=num_samples, p=probabilities)

        # Build the sequence of exponentiated Pauli terms
        # Each sample: exp(-i * sign(h_j) * λ * t / N * P_j)
        angle_magnitude = lambda_norm * time / num_samples

        terms: list[ExponentiatedPauliTerm] = []
        for idx in term_indices:
            pauli = filtered_paulis[idx]
            coeff = filtered_coeffs[idx]
            sign = 1.0 if coeff >= 0 else -1.0

            mapping = self._pauli_label_to_map(pauli.to_label())
            angle = sign * angle_magnitude

            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

        return TimeEvolutionUnitary(
            container=PauliProductFormulaContainer(
                step_terms=terms,
                step_reps=1,  # All samples are already in the terms list
                num_qubits=qubit_hamiltonian.num_qubits,
            )
        )

    @staticmethod
    def _pauli_label_to_map(label: str) -> dict[int, str]:
        """Translate a Pauli label to a mapping ``qubit -> {X, Y, Z}``.

        Args:
            label: Pauli string label in little-endian ordering.

        Returns:
            Dictionary assigning each non-identity qubit index to its Pauli axis.

        """
        mapping: dict[int, str] = {}
        for index, char in enumerate(reversed(label)):  # reversed: right-most char -> qubit 0
            if char != "I":
                mapping[index] = char
        return mapping

    def name(self) -> str:
        """Return the name of the time evolution unitary builder."""
        return "qdrift"

    def type_name(self) -> str:
        """Return time_evolution_builder as the algorithm type name."""
        return "time_evolution_builder"
