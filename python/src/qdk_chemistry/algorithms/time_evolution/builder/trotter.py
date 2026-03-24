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

from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.builder.trotter_error import (
    trotter_steps_commutator,
    trotter_steps_naive,
)
from qdk_chemistry.data import QubitHamiltonian, Settings, TimeEvolutionUnitary
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

        Args:
            order: The order of the Trotter decomposition (currently only
                first order is supported). Defaults to 1.
            target_accuracy: Target accuracy for automatic step computation.
                Must be positive to enable automatic computation.
                Use 0.0 (default) to disable.
            num_divisions: Explicit number of divisions within a Trotter
                step.  When both *num_divisions* and *target_accuracy*
                are given the larger value is used.  Use 0 (default) for
                automatic determination.
            error_bound: Strategy for computing the Trotter error bound
                when *target_accuracy* is set.  Either ``"commutator"``
                (default, tighter) or ``"naive"``.
            weight_threshold: Absolute threshold for filtering small
                Hamiltonian coefficients. Defaults to 1e-12.

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
        self, qubit_hamiltonian: QubitHamiltonian, time: float, *, atol: float = 1e-12
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

        order = self._settings.get("order")

        coeffs = list(qubit_hamiltonian.get_real_coefficients(tolerance=atol))
        # If there are no coefficients (e.g., empty Hamiltonian or all filtered by atol),
        # there is nothing to decompose; return the empty list of terms.
        if not coeffs:
            Logger.warn("No coefficients above the tolerance; returning empty term list.")
            return terms

        if order == 1:
            for label, coeff in coeffs:
                mapping = self._pauli_label_to_map(label)
                angle = coeff * time
                terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))
        # order = 2 or order = 2k with k>1
        else:
            # \prod_{i=1}^{L-1} e^{-iH_i t/(2n)}
            for label, coeff in coeffs[:-1]:
                mapping = self._pauli_label_to_map(label)
                angle = coeff * time / 2
                terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))
            # e^{-iH_L t/n}
            label, coeff = coeffs[-1]
            mapping = self._pauli_label_to_map(label)
            angle = coeff * time
            terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

            # \prod_{i=L-1}^1 e^{-iH_i t/(2n)}
            for label, coeff in reversed(coeffs[:-1]):
                mapping = self._pauli_label_to_map(label)
                angle = coeff * time / 2
                terms.append(ExponentiatedPauliTerm(pauli_term=mapping, angle=angle))

            # Construct order 2k formula bottom up dynamic-programming style
            if order > 2:
                step_terms = terms.copy()
                for k in range(2, int(order / 2) + 1):
                    u_k = 1 / (4 - 4 ** (1 / (2 * k - 1)))
                    new_terms = []

                    # S_{2k-2}(u_k t)^2 = S_{2k-2}(u_k t) S_{2k-2}(u_k t)
                    for _ in range(2):
                        for term in step_terms:
                            new_terms.append(
                                ExponentiatedPauliTerm(
                                    pauli_term=term.pauli_term,
                                    angle=term.angle * u_k,
                                )
                            )
                    # S_{2k-2}((1-4u_k) t)
                    for term in step_terms:
                        new_terms.append(
                            ExponentiatedPauliTerm(
                                pauli_term=term.pauli_term,
                                angle=term.angle * (1 - 4 * u_k),
                            )
                        )

                    # S_{2k-2}(u_k t)^2 = S_{2k-2}(u_k t) S_{2k-2}(u_k t)
                    for _ in range(2):
                        for term in step_terms:
                            new_terms.append(
                                ExponentiatedPauliTerm(
                                    pauli_term=term.pauli_term,
                                    angle=term.angle * u_k,
                                )
                            )

                    step_terms = new_terms
                terms = step_terms

            # Merge adjacent terms with the same pauli_term by summing angles.
            merged_terms: list[ExponentiatedPauliTerm] = []
            for term in terms:
                if merged_terms and merged_terms[-1].pauli_term == term.pauli_term:
                    last = merged_terms[-1]
                    merged_terms[-1] = ExponentiatedPauliTerm(
                        pauli_term=last.pauli_term,
                        angle=last.angle + term.angle,
                    )
                else:
                    merged_terms.append(term)
            terms = merged_terms
        return terms

    def name(self) -> str:
        """Return the name of the time evolution unitary builder."""
        return "trotter"

    def type_name(self) -> str:
        """Return time_evolution_builder as the algorithm type name."""
        return "time_evolution_builder"
