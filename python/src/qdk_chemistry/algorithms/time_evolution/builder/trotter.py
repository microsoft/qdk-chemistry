"""QDK/Chemistry implementation of the Trotter decomposition Builder."""

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

__all__: list[str] = ["Trotter", "TrotterSettings"]


class TrotterSettings(Settings):
    """Settings for Trotter decomposition builder."""

    def __init__(self):
        """Initialize TrotterSettings with default values.

        Attributes:
            order: The order of the Trotter decomposition (currently only first order is supported).
            weight_threshold: The absolute threshold for filtering small coefficients.

        """
        super().__init__()
        self._set_default("order", "int", 1, "The order of the Trotter decomposition.")
        self._set_default(
            "weight_threshold", "float", 1e-12, "The absolute threshold for filtering small coefficients."
        )


class Trotter(TimeEvolutionBuilder):
    """Trotter decomposition builder."""

    def __init__(
        self,
        order: int = 1,
        *,
        target_accuracy: float | None = None,
        num_divisions: int | None = None,
        error_bound: str = "commutator",
        weight_threshold: float = 1e-12,
    ):
        r"""Initialize Trotter builder with specified Trotter decomposition settings.

        The number of divisions *N* can be determined automatically from
        *target_accuracy*, fixed explicitly via *num_divisions*, or both
        (in which case the larger value is used).

        Two error-bound strategies are available (used only when
        *target_accuracy* is set):

        * ``"commutator"`` (default, tighter): uses the commutator-based bound
          from Childs *et al.* (2021).  :math:`N = \lceil \frac{t^{2}}{2\epsilon}
          \sum_{j<k}\lVert[\alpha_jP_j,\alpha_kP_k]\rVert \rceil`
        * ``"naive"``: uses the triangle-inequality bound.
          :math:`N = \lceil (\sum_j|\alpha_j|)^{2}t^{2}/\epsilon \rceil`

        Args:
            order: The order of the Trotter decomposition (currently only
                first order is supported). Defaults to 1.
            target_accuracy: If given, automatically compute the number of
                divisions needed to achieve this accuracy.  Must be
                positive.  Defaults to ``None`` (disabled).
            num_divisions: Explicit number of divisions within a Trotter
                step.  When both *num_divisions* and *target_accuracy*
                are given the larger value is used.  Defaults to ``None``.
            error_bound: Strategy for computing the Trotter error bound
                when *target_accuracy* is set.  Either ``"commutator"``
                (default, tighter) or ``"naive"``.
            weight_threshold: Absolute threshold for filtering small
                Hamiltonian coefficients. Defaults to 1e-12.

        """
        super().__init__()
        if target_accuracy is not None and target_accuracy <= 0:
            raise ValueError(f"target_accuracy must be positive, got {target_accuracy}.")
        if num_divisions is not None and num_divisions < 1:
            raise ValueError(f"num_divisions must be >= 1, got {num_divisions}.")
        if error_bound not in ("commutator", "naive"):
            raise ValueError(f"error_bound must be 'commutator' or 'naive', got {error_bound!r}.")
        self._settings = TrotterSettings()
        self._settings.set("order", order)
        self._settings.set("weight_threshold", weight_threshold)
        self._target_accuracy = target_accuracy
        self._num_divisions = num_divisions
        self._error_bound = error_bound

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        """Construct the time evolution unitary using Trotter decomposition.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.
            time: The total evolution time.

        Returns:
            TimeEvolutionUnitary: The time evolution unitary built by the Trotter decomposition.

        """
        if self._settings.get("order") == 1:
            return self._first_order_trotter(qubit_hamiltonian, time)
        raise NotImplementedError("Only first-order Trotter decomposition is currently supported.")

    def _first_order_trotter(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> TimeEvolutionUnitary:
        r"""Construct the time evolution unitary using first-order Trotter decomposition.

        The First Order Trotter method approximates the time evolution operator :math:`e^{-iHt}`
        by decomposing the Hamiltonian H into a sum of terms and using the product formula:
        :math:`e^{-iHt} \approx \left[\prod_i e^{-iH_i t/n}\right]^n`, where n is the number of divisions.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.
            time: The total evolution time.

        Returns:
            TimeEvolutionUnitary: The time evolution unitary built by the Trotter decomposition.

        """
        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, time)

        delta = time / num_divisions
        weight_threshold = self._settings.get("weight_threshold")

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
        manual = self._num_divisions if self._num_divisions is not None else 1

        if self._target_accuracy is None:
            return manual

        order = self._settings.get("order")
        weight_threshold = self._settings.get("weight_threshold")
        if self._error_bound == "commutator":
            auto = trotter_steps_commutator(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=self._target_accuracy,
                order=order,
                weight_threshold=weight_threshold,
            )
        else:
            auto = trotter_steps_naive(
                hamiltonian=qubit_hamiltonian,
                time=time,
                target_accuracy=self._target_accuracy,
                order=order,
                weight_threshold=weight_threshold,
            )
        return max(manual, auto)

    def _decompose_trotter_step(
        self, qubit_hamiltonian: QubitHamiltonian, time: float, *, atol: float = 1e-12
    ) -> list[ExponentiatedPauliTerm]:
        """Decompose a single Trotter step into exponentiated Pauli terms.

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

        for label, coeff in qubit_hamiltonian.get_real_coefficients(tolerance=atol):
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
