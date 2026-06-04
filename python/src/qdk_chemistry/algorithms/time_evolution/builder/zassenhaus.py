"""QDK/Chemistry implementation of the Zassenhaus expansion Builder."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from qiskit.quantum_info import SparsePauliOp

from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.data import QubitHamiltonian, Settings, TimeEvolutionUnitary
from qdk_chemistry.data.time_evolution.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)

__all__: list[str] = ["Zassenhaus", "ZassenhausSettings"]


class ZassenhausSettings(Settings):
    """Settings for Zassenhaus expansion builder."""

    def __init__(self):
        """Initialize ZassenhausSettings with default values.

        Attributes:
            order: Expansion order p (>=2); correction terms through O(t^p) are included.
            split_index: Index at which to split Hamiltonian terms into groups A (indices < split)
                and B (indices >= split). -1 means split at the midpoint.
            tolerance: Absolute tolerance for filtering small coefficients.

        """
        super().__init__()
        self._set_default("order", "int", 2, "Expansion order p (>=2).")
        self._set_default(
            "split_index",
            "int",
            -1,
            "Split index for Hamiltonian partitioning (-1 = midpoint).",
        )
        self._set_default("tolerance", "float", 1e-12, "Absolute tolerance for filtering small coefficients.")


class Zassenhaus(TimeEvolutionBuilder):
    """Zassenhaus expansion builder for Hamiltonian simulation.

    Approximates exp(-iHt) via the Zassenhaus product formula, splitting
    H = H_A + H_B and constructing explicit commutator-correction factors
    through the configured order p:

        exp(-iHt) ~ exp(-iH_A t) exp(-iH_B t)
                    x exp(t^2/2 [H_A, H_B])
                    x exp(i t^3/6 (F_A + 2 F_B))
                    x exp(-t^4/24 (G_AA + 3 G_BA + 3 G_BB))
                    ...

    where F_A = [H_A, [H_A, H_B]], F_B = [H_B, [H_A, H_B]], and
    G_AA = [H_A, F_A], G_BA = [H_B, F_A], G_BB = [H_B, F_B].

    The operator-norm error scales as O(t^(p+1)).
    """

    def __init__(self, order: int = 2, split_index: int = -1, tolerance: float = 1e-12):
        """Initialize Zassenhaus builder.

        Args:
            order: Expansion order p (>= 2, <= 4). Determines how many commutator-correction
                factors are included; the global error is O(t^(p+1)). Defaults to 2.
            split_index: Index at which to partition Hamiltonian terms into two groups.
                Group A receives indices < split_index, group B receives the rest.
                -1 (default) splits at the midpoint.
            tolerance: Absolute threshold for filtering small coefficients. Defaults to 1e-12.

        """
        super().__init__()
        self._settings = ZassenhausSettings()
        self._settings.set("order", order)
        self._settings.set("split_index", split_index)
        self._settings.set("tolerance", tolerance)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        """Construct the time evolution unitary using the Zassenhaus expansion.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.
            time: The total evolution time.

        Returns:
            TimeEvolutionUnitary: The time evolution unitary built by the Zassenhaus expansion.

        Raises:
            ValueError: If order < 2 or the Hamiltonian is non-Hermitian.
            NotImplementedError: If order > 4.

        """
        order = self._settings.get("order")
        if order < 2:
            raise ValueError(f"Zassenhaus expansion requires order >= 2, got {order}.")
        if order > 4:
            raise NotImplementedError("Zassenhaus expansion is implemented for orders 2, 3, and 4 only.")

        tol = self._settings.get("tolerance")
        A_spo, B_spo = self._split_hamiltonian(qubit_hamiltonian)

        terms: list[ExponentiatedPauliTerm] = []

        # Leading product: exp(-i H_A t) exp(-i H_B t)
        terms.extend(self._spo_to_first_order_terms(A_spo, time, tol))
        terms.extend(self._spo_to_first_order_terms(B_spo, time, tol))

        # D = [H_A, H_B] — anti-Hermitian (purely imaginary Pauli coefficients)
        D = (A_spo @ B_spo - B_spo @ A_spo).simplify(atol=tol)

        if order >= 2:
            # Correction: exp(t^2/2 * [H_A, H_B])
            terms.extend(self._anti_hermitian_to_terms(D, time**2 / 2, tol))

        F_A: SparsePauliOp | None = None
        F_B: SparsePauliOp | None = None

        if order >= 3:
            # F_A = [H_A, D] (Hermitian), F_B = [H_B, D] (Hermitian)
            F_A = (A_spo @ D - D @ A_spo).simplify(atol=tol)
            F_B = (B_spo @ D - D @ B_spo).simplify(atol=tol)
            # Correction: exp(i t^3/6 * (F_A + 2 F_B))
            G3 = (F_A + 2 * F_B).simplify(atol=tol)
            terms.extend(self._hermitian_to_terms(G3, time**3 / 6, tol))

        if order >= 4:
            # Ensure F_A, F_B are available (always the case when order >= 3)
            if F_A is None or F_B is None:
                F_A = (A_spo @ D - D @ A_spo).simplify(atol=tol)
                F_B = (B_spo @ D - D @ B_spo).simplify(atol=tol)
            # G_AA = [H_A, F_A], G_BA = [H_B, F_A], G_BB = [H_B, F_B] — anti-Hermitian
            G_AA = (A_spo @ F_A - F_A @ A_spo).simplify(atol=tol)
            G_BA = (B_spo @ F_A - F_A @ B_spo).simplify(atol=tol)
            G_BB = (B_spo @ F_B - F_B @ B_spo).simplify(atol=tol)
            G4 = (G_AA + 3 * G_BA + 3 * G_BB).simplify(atol=tol)
            # Correction: exp(-t^4/24 * G4)
            terms.extend(self._anti_hermitian_to_terms(G4, -(time**4 / 24), tol))

        container = PauliProductFormulaContainer(
            step_terms=terms,
            step_reps=1,
            num_qubits=qubit_hamiltonian.num_qubits,
        )
        return TimeEvolutionUnitary(container=container)

    def _split_hamiltonian(self, qubit_hamiltonian: QubitHamiltonian) -> tuple[SparsePauliOp, SparsePauliOp]:
        """Split the Hamiltonian into two SparsePauliOp groups A and B.

        Args:
            qubit_hamiltonian: The Hamiltonian to split.

        Returns:
            Tuple (A_spo, B_spo) of SparsePauliOps for the two groups.

        """
        spo = qubit_hamiltonian.pauli_ops
        n = len(spo)
        split = self._settings.get("split_index")
        if split < 0 or split >= n:
            split = max(1, n // 2)
        split = max(1, min(split, n - 1))
        A = SparsePauliOp(spo.paulis[:split], spo.coeffs[:split])
        B = SparsePauliOp(spo.paulis[split:], spo.coeffs[split:])
        return A, B

    @staticmethod
    def _spo_to_first_order_terms(
        spo: SparsePauliOp,
        time: float,
        atol: float,
    ) -> list[ExponentiatedPauliTerm]:
        """Convert Hermitian SparsePauliOp to first-order ExponentiatedPauliTerms.

        Produces exp(-i h_k t P_k) for each real coefficient h_k.

        Args:
            spo: Hermitian SparsePauliOp with real Pauli coefficients.
            time: Evolution time.
            atol: Absolute tolerance for filtering.

        Returns:
            List of ExponentiatedPauliTerms.

        """
        terms: list[ExponentiatedPauliTerm] = []
        for pauli, coeff in zip(spo.paulis, spo.coeffs, strict=True):
            c = complex(coeff)
            if abs(c) < atol:
                continue
            if abs(c.imag) > atol:
                raise ValueError(
                    f"Non-Hermitian Hamiltonian: coefficient {coeff} for term "
                    f"{pauli.to_label()} has nonzero imaginary part."
                )
            label = pauli.to_label()
            mapping = {i: ch for i, ch in enumerate(reversed(label)) if ch != "I"}
            if not mapping:
                continue
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=c.real * time))
        return terms

    @staticmethod
    def _anti_hermitian_to_terms(
        spo: SparsePauliOp,
        scale: float,
        atol: float,
    ) -> list[ExponentiatedPauliTerm]:
        """Convert exp(scale * spo) to ExponentiatedPauliTerms for anti-Hermitian spo.

        For each Pauli P_k with purely imaginary coefficient i*beta_k in spo:
            exp(scale * i*beta_k * P_k) = exp(-i * (-scale*beta_k) * P_k)
            → angle = -scale * Im(coeff_k)

        Args:
            spo: Anti-Hermitian SparsePauliOp (purely imaginary Pauli coefficients).
            scale: Scalar multiplier (incorporates the t^n/n! prefactor).
            atol: Absolute tolerance for filtering small angles.

        Returns:
            List of ExponentiatedPauliTerms.

        """
        terms: list[ExponentiatedPauliTerm] = []
        for pauli, coeff in zip(spo.paulis, spo.coeffs, strict=True):
            angle = -scale * complex(coeff).imag
            if abs(angle) < atol:
                continue
            label = pauli.to_label()
            mapping = {i: ch for i, ch in enumerate(reversed(label)) if ch != "I"}
            if not mapping:
                continue
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))
        return terms

    @staticmethod
    def _hermitian_to_terms(
        spo: SparsePauliOp,
        scale: float,
        atol: float,
    ) -> list[ExponentiatedPauliTerm]:
        """Convert exp(i * scale * spo) to ExponentiatedPauliTerms for Hermitian spo.

        For each Pauli P_k with real coefficient g_k in spo:
            exp(i * scale * g_k * P_k) = exp(-i * (-scale*g_k) * P_k)
            → angle = -scale * Re(coeff_k)

        Args:
            spo: Hermitian SparsePauliOp (real Pauli coefficients).
            scale: Scalar multiplier (incorporates the t^n/n! prefactor).
            atol: Absolute tolerance for filtering small angles.

        Returns:
            List of ExponentiatedPauliTerms.

        """
        terms: list[ExponentiatedPauliTerm] = []
        for pauli, coeff in zip(spo.paulis, spo.coeffs, strict=True):
            angle = -scale * complex(coeff).real
            if abs(angle) < atol:
                continue
            label = pauli.to_label()
            mapping = {i: ch for i, ch in enumerate(reversed(label)) if ch != "I"}
            if not mapping:
                continue
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))
        return terms

    def name(self) -> str:
        """Return the name of the time evolution unitary builder."""
        return "zassenhaus"

    def type_name(self) -> str:
        """Return time_evolution_builder as the algorithm type name."""
        return "time_evolution_builder"
