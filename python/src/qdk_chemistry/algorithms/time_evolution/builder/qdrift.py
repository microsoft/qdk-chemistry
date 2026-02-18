"""QDK/Chemistry implementation of the qDRIFT randomized evolution builder.

This module implements the qDRIFT algorithm for Hamiltonian simulation, which provides
an alternative to deterministic Trotter decomposition by using randomized sampling.

References:
    Campbell, E. (2019). Random Compiler for Fast Hamiltonian Simulation.
    Physical Review Letters, 123(7), 070503.
    https://arxiv.org/abs/1811.08017
    https://doi.org/10.1103/PhysRevLett.123.070503

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.utils.pauli_commutation import (
    do_pauli_terms_qw_commute,
    get_commutation_checker,
)
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
            seed: Random seed for reproducibility. Use -1 for non-deterministic behavior.
            merge_duplicate_terms: Whether to fuse identical Pauli terms that
                appear in consecutive mutually-commuting runs, reducing circuit
                depth.  Only equal operators are combined; distinct commuting
                terms are kept separate.  The merging is exact and preserves
                the error bound.

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
            "merge_duplicate_terms",
            "bool",
            True,
            "Fuse identical Pauli terms within consecutive commuting runs to reduce circuit depth.",
        )
        self._set_default(
            "commutation_type",
            "string",
            "general",
            "Commutation check for merging: 'qubit_wise' (per-qubit) or 'general' (standard Pauli).",
            ("qubit_wise", "general"),
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
        merge_duplicate_terms: Whether to fuse identical Pauli terms within
            consecutive commuting runs.

    Examples:
        >>> from qdk_chemistry.algorithms import create
        >>> # Create a qDRIFT builder with 500 samples
        >>> qdrift = create("time_evolution_builder", "qdrift", num_samples=500, seed=42)
        >>> # Use it to build time evolution for a Hamiltonian
        >>> time_evolution = qdrift.run(qubit_hamiltonian, time=1.0)

    References:
        Campbell, E. (2019). Random Compiler for Fast Hamiltonian Simulation.
        Physical Review Letters, 123(7), 070503.
        https://arxiv.org/abs/1811.08017
        https://doi.org/10.1103/PhysRevLett.123.070503

    """

    def __init__(
        self,
        num_samples: int = 100,
        seed: int = -1,
        merge_duplicate_terms: bool = True,
        commutation_type: str = "general",
    ):
        """Initialize qDRIFT builder with specified settings.

        Args:
            num_samples: Number of random samples N. More samples increase accuracy
                but also increase circuit depth. Error scales as O(λ²t²/N).
                Defaults to 100.
            seed: Random seed for reproducibility. Use -1 for non-deterministic
                sampling. Defaults to -1.
            merge_duplicate_terms: If ``True``, identical Pauli terms within
                consecutive mutually-commuting runs are fused to reduce
                circuit depth.  Distinct commuting terms are kept separate.
                The merging is exact and preserves the Campbell (2019)
                error bound.  Defaults to ``True``.
            commutation_type: Commutation check used when merging duplicate
                terms.  ``"qubit_wise"`` requires every single-qubit
                pair to commute individually — stricter but always safe.
                ``"general"`` (default) uses standard Pauli commutation (even number of
                anti-commuting positions), which allows larger merge groups.

        """
        super().__init__()
        self._settings = QDriftSettings()
        self._settings.set("num_samples", num_samples)
        self._settings.set("seed", seed)
        self._settings.set("merge_duplicate_terms", merge_duplicate_terms)
        self._settings.set("commutation_type", commutation_type)

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

        if not qubit_hamiltonian.is_hermitian():
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        # Build (label, real_coeff) pairs from the full Hamiltonian
        all_terms = list(
            zip(
                qubit_hamiltonian.pauli_strings,
                np.real(qubit_hamiltonian.coefficients).tolist(),
                strict=True,
            )
        )

        terms = self._sample_qdrift_terms(all_terms, time, num_samples, rng)

        # Optionally fuse identical Pauli operators that appear within
        # consecutive mutually-commuting runs.  Within such a run,
        # identical Pauli strings satisfy e^{-iaP} e^{-ibP} = e^{-i(a+b)P}
        # exactly, reducing circuit depth.  Distinct Pauli strings are
        # kept as separate rotations and non-commuting boundaries are
        # never crossed, preserving the Campbell (2019) error bound.
        if self._settings.get("merge_duplicate_terms"):
            commute_fn = get_commutation_checker(self._settings.get("commutation_type"))
            terms = self._merge_duplicate_terms(terms, commute_fn=commute_fn)

        return TimeEvolutionUnitary(
            container=PauliProductFormulaContainer(
                step_terms=terms,
                step_reps=1,  # All samples are already in the terms list
                num_qubits=qubit_hamiltonian.num_qubits,
            )
        )

    # ------------------------------------------------------------------
    # qDRIFT sampling and duplicate-term fusion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_qdrift_terms(
        terms: list[tuple[str, float]],
        time: float,
        num_samples: int,
        rng: np.random.Generator,
    ) -> list[ExponentiatedPauliTerm]:
        """Build qDRIFT-style random samples from a set of Pauli terms.

        Each term ``(label, coeff)`` is sampled with probability proportional
        to ``|coeff|``.  Every sample contributes a rotation
        ``exp(-i * sign(coeff) * λ * t / N * P)`` where ``λ`` is the 1-norm
        of the coefficients and ``N`` is the number of samples.

        Args:
            terms: List of ``(pauli_label, coefficient)`` pairs.
            time: Evolution time for this block.
            num_samples: Number of random samples (N).
            rng: Random number generator.

        Returns:
            List of :class:`ExponentiatedPauliTerm` for the sampled sequence.

        """
        if len(terms) == 0:
            return []

        if num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {num_samples}.")

        coeffs = np.array([c for _, c in terms])
        abs_coeffs = np.abs(coeffs)
        lambda_norm = float(abs_coeffs.sum())

        if lambda_norm < 1e-14:
            return []

        probabilities = abs_coeffs / lambda_norm
        term_indices = rng.choice(len(terms), size=num_samples, p=probabilities)

        angle_magnitude = lambda_norm * time / num_samples

        result: list[ExponentiatedPauliTerm] = []
        for idx in term_indices:
            label, coeff = terms[idx]
            sign = 1.0 if coeff >= 0 else -1.0
            mapping = TimeEvolutionBuilder._pauli_label_to_map(label)  # noqa: SLF001
            result.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=sign * angle_magnitude))

        return result

    @classmethod
    def _merge_duplicate_terms(
        cls,
        terms: list[ExponentiatedPauliTerm],
        commute_fn=None,
    ) -> list[ExponentiatedPauliTerm]:
        r"""Fuse identical Pauli operators within consecutive commuting runs.

        Walks through the term sequence and accumulates maximal runs of
        mutually commuting terms (using the supplied commutation checker).
        Within each run, *identical* Pauli operators (same string) have
        their rotation angles summed:

        .. math::

            e^{-i a P}\,e^{-i b P} = e^{-i(a+b)P}

        This is exact because scalar multiples of the same operator
        trivially commute.  Distinct Pauli strings are **not** combined;
        they remain as separate rotations.  Non-commuting boundaries are
        never crossed.

        Args:
            terms: Ordered list of exponentiated Pauli terms.
            commute_fn: A callable ``(a, b) -> bool`` that checks whether
                two Pauli term mappings commute.  Defaults to
                :func:`~.pauli_commutation.do_pauli_terms_qw_commute` if ``None``.

        Returns:
            A (potentially shorter) list producing the same unitary.

        """
        if not terms:
            return terms

        if commute_fn is None:
            commute_fn = do_pauli_terms_qw_commute

        result: list[ExponentiatedPauliTerm] = []
        group: list[ExponentiatedPauliTerm] = [terms[0]]

        for term in terms[1:]:
            if all(commute_fn(term.pauli_term, g.pauli_term) for g in group):
                group.append(term)
            else:
                result.extend(cls._flush_duplicate_terms(group))
                group = [term]

        result.extend(cls._flush_duplicate_terms(group))
        return result

    @staticmethod
    def _flush_duplicate_terms(
        group: list[ExponentiatedPauliTerm],
    ) -> list[ExponentiatedPauliTerm]:
        """Fuse *identical* Pauli operators within a mutually qubit-wise commuting group.

        Only rotations around the **same** Pauli string have their angles
        summed (e^{-iaP} e^{-ibP} = e^{-i(a+b)P}).  Distinct Pauli
        strings are emitted as separate terms.  Terms whose fused angle
        is exactly zero are dropped.

        """
        merged: dict[tuple[tuple[int, str], ...], float] = {}
        for term in group:
            key = tuple(sorted(term.pauli_term.items()))
            merged[key] = merged.get(key, 0.0) + term.angle

        return [
            ExponentiatedPauliTerm(pauli_term=dict(key), angle=angle) for key, angle in merged.items() if angle != 0.0
        ]

    def name(self) -> str:
        """Return the name of the time evolution unitary builder."""
        return "qdrift"

    def type_name(self) -> str:
        """Return time_evolution_builder as the algorithm type name."""
        return "time_evolution_builder"
