r"""QDK/Chemistry implementation of the Zassenhaus-expansion Builder.

The Zassenhaus formula is the dual of the Baker-Campbell-Hausdorff formula: it
writes :math:`\exp(-iHt)` as an *ordered product* of exponentials in which the
low-order commutator corrections appear as explicit factors, rather than being
absorbed implicitly into a Trotter step count.  For a configured expansion order
:math:`p` the operator-norm error scales as :math:`O(t^{\,p+1})` at fixed step
count, so Hamiltonians whose dominant commutators are sparse or analytically
tractable are simulated with fewer steps than Trotter-Suzuki would require.

References:
    Wilcox, R. M. "Exponential operators and parameter differentiation in quantum
    physics." *Journal of Mathematical Physics* 8.4 (1967): 962-982.

    Casas, F., Murua, A., and Nadinic, M. "Efficient computation of the Zassenhaus
    formula." *Computer Physics Communications* 183.11 (2012): 2386-2391.
    https://arxiv.org/abs/1204.0389

    Childs, A. M., et al. "Theory of Trotter Error with Commutator Scaling."
    *Physical Review X* 11.1 (2021): 011020. https://arxiv.org/abs/1912.08854

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import math

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder, TimeEvolutionSettings
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.zassenhaus_expansion import zassenhaus_factors
from qdk_chemistry.data import FlatPartition, LayeredPartition, QubitHamiltonian, UnitaryRepresentation
from qdk_chemistry.data.unitary_representation.containers.pauli_product_formula import (
    ExponentiatedPauliTerm,
    PauliProductFormulaContainer,
)
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.pauli_commutation import commutator_bound_higher_order

__all__: list[str] = ["Zassenhaus", "ZassenhausSettings"]


class ZassenhausSettings(TimeEvolutionSettings):
    """Settings for the Zassenhaus-expansion builder."""

    def __init__(self):
        """Initialize ZassenhausSettings with default values.

        Attributes:
            order: The Zassenhaus expansion order ``p``.  ``p = 1`` reproduces the
                first-order Trotter product; higher orders cancel successively
                higher commutator corrections, giving error ``O(t**(p+1))``.
            num_divisions: Number of time divisions ``N``.  Each division evolves
                for ``time / N`` and the step is repeated ``N`` times.
            target_accuracy: Target accuracy for automatic time-division count
                (0.0 means disabled).  When set, ``N`` is estimated from a
                commutator error bound and the larger of it and ``num_divisions`` is used.
            weight_threshold: The absolute threshold for filtering small coefficients.

        """
        super().__init__()
        self._set_default("order", "int", 2, "The order of the Zassenhaus expansion.")
        self._set_default(
            "num_divisions",
            "int",
            1,
            "Explicit number of time divisions (each evolves for time / num_divisions).",
        )
        self._set_default(
            "target_accuracy",
            "double",
            0.0,
            "Target accuracy for automatic time-division count (0.0 means disabled).",
        )
        self._set_default(
            "weight_threshold", "float", 1e-12, "The absolute threshold for filtering small coefficients."
        )


class Zassenhaus(TimeEvolutionBuilder):
    r"""Zassenhaus-expansion time-evolution builder.

    Approximates the time-evolution operator :math:`e^{-iHt}` for a Hermitian
    Hamiltonian :math:`H = \sum_j \alpha_j P_j` (Pauli strings :math:`P_j`) as an
    ordered product of single-Pauli exponentials.  Starting from the bare
    first-order Lie-Trotter product, the builder appends explicit commutator
    corrections order by order so that, for expansion order :math:`p`,

    .. math::

        \bigl\lVert e^{-iHt} - U_{\text{Zassenhaus}} \bigr\rVert = O\!\left(t^{\,p+1}\right).

    Because every nested commutator of Pauli strings is itself a scalar multiple
    of a single Pauli string, each correction factor is again an exponential
    :math:`e^{-i\theta P}` and the output is a
    :class:`~qdk_chemistry.data.unitary_representation.containers.pauli_product_formula.PauliProductFormulaContainer`,
    consumable by the same downstream algorithms (e.g. ``PhaseEstimation``) as the
    Trotter builder.

    The order-by-order commutator-cancellation construction is implemented by the
    ``zassenhaus_factors`` function in the ``zassenhaus_expansion`` module.

    Examples:
        >>> from qdk_chemistry.algorithms import registry
        >>> builder = registry.create(
        ...     "hamiltonian_unitary_builder", "zassenhaus", order=3, time=1.0
        ... )
        >>> unitary = builder.run(qubit_hamiltonian)

    """

    def __init__(
        self,
        order: int = 2,
        *,
        time: float = 0.0,
        num_divisions: int = 1,
        target_accuracy: float = 0.0,
        weight_threshold: float = 1e-12,
        power: int = 1,
        power_strategy: str = "repeat",
    ):
        """Initialize the Zassenhaus builder with the specified settings.

        Args:
            order: Zassenhaus expansion order ``p >= 1``. Defaults to 2.
            time: The evolution time. Defaults to 0.0.
            num_divisions: Number of time divisions ``N`` (each evolves for
                ``time / N`` and the step repeats ``N`` times). Defaults to 1.
            target_accuracy: Target accuracy for automatic division count. When > 0,
                ``N`` is estimated from a commutator error bound and the larger of it
                and ``num_divisions`` is used. Use 0.0 (default) to disable.
            weight_threshold: Threshold for filtering small coefficients. Defaults to 1e-12.
            power: The power to raise the unitary to. Defaults to 1.
            power_strategy: Strategy for U^power: ``"rescale"`` or ``"repeat"`` (default).

        """
        super().__init__()
        self._settings = ZassenhausSettings()
        self._settings.set("time", time)
        self._settings.set("power", power)
        self._settings.set("power_strategy", power_strategy)
        self._settings.set("order", order)
        self._settings.set("num_divisions", num_divisions)
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("weight_threshold", weight_threshold)

    def _run_impl(self, qubit_hamiltonian: QubitHamiltonian) -> UnitaryRepresentation:
        """Construct the unitary representation using the Zassenhaus expansion.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian to be used in the construction.

        Returns:
            UnitaryRepresentation: The unitary representation built by the Zassenhaus expansion.

        """
        effective_time, power_repetitions = self._resolve_power()
        order = self._settings.get("order")
        weight_threshold = self._settings.get("weight_threshold")

        if order < 1:
            raise ValueError(f"Zassenhaus expansion order must be >= 1, got {order}.")

        if self._settings.get("num_divisions") < 1:
            raise ValueError(f"num_divisions must be a positive integer, got {self._settings.get('num_divisions')}.")

        if not qubit_hamiltonian.is_hermitian(tolerance=weight_threshold):
            raise ValueError("Non-Hermitian Hamiltonian: coefficients have nonzero imaginary parts.")

        num_divisions = self._resolve_num_divisions(qubit_hamiltonian, effective_time)
        num_qubits = qubit_hamiltonian.num_qubits
        delta = effective_time / num_divisions

        terms = self._ordered_terms(qubit_hamiltonian, weight_threshold)
        step_terms = self._build_step_terms(terms, order=order, delta=delta, atol=weight_threshold)

        container = PauliProductFormulaContainer(
            step_terms=step_terms,
            step_reps=num_divisions * power_repetitions,
            num_qubits=num_qubits,
        )
        return UnitaryRepresentation(container=container)

    def _resolve_num_divisions(self, qubit_hamiltonian: QubitHamiltonian, time: float) -> int:
        r"""Determine the number of time divisions ``N``.

        Without *target_accuracy* this is just the ``num_divisions`` setting.  When
        *target_accuracy* :math:`\epsilon` is set, ``N`` is estimated from the
        commutator error bound: the order-:math:`p` remainder scales as
        :math:`\alpha\,t^{p+1}/N^{p}`, where :math:`\alpha` is the sum of degree-:math:`(p+1)`
        nested-commutator norms (:func:`~qdk_chemistry.utils.pauli_commutation.commutator_bound_higher_order`,
        the same infrastructure the Trotter builder uses).  Setting that
        :math:`\le \epsilon` gives :math:`N = \lceil (\alpha t^{p+1}/\epsilon)^{1/p} \rceil`.
        The larger of the manual and estimated value is returned.
        """
        manual = self._settings.get("num_divisions")
        target_accuracy = self._settings.get("target_accuracy")
        if target_accuracy <= 0.0 or time == 0.0:
            return manual

        order = self._settings.get("order")
        weight_threshold = self._settings.get("weight_threshold")
        alpha = commutator_bound_higher_order(qubit_hamiltonian, order, weight_threshold=weight_threshold)
        if alpha <= 0.0:
            return manual
        auto = math.ceil((alpha * abs(time) ** (order + 1) / target_accuracy) ** (1.0 / order))
        return max(manual, auto, 1)

    @staticmethod
    def _ordered_terms(qubit_hamiltonian: QubitHamiltonian, weight_threshold: float) -> list[tuple[str, float]]:
        """Return ``(pauli_label, real_coeff)`` pairs, ordered by ``term_partition`` if present.

        When the Hamiltonian carries a :class:`~qdk_chemistry.data.TermPartition`, the bare
        first-order product consumes it so that commuting terms within a group are adjacent
        (consistent with the Trotter builder); otherwise terms keep their natural order.
        Terms whose real coefficient is below *weight_threshold* are dropped.
        """
        partition = qubit_hamiltonian.term_partition
        labels = qubit_hamiltonian.pauli_strings
        coefficients = qubit_hamiltonian.coefficients

        if partition is None:
            indices: list[int] = list(range(len(labels)))
        else:
            indices = []
            for group in partition.groups:
                if isinstance(partition, LayeredPartition):
                    for layer in group:
                        indices.extend(layer)
                elif isinstance(partition, FlatPartition):
                    indices.extend(group)
                else:
                    raise TypeError(f"Unsupported TermPartition subtype: {type(partition).__name__}.")

        ordered: list[tuple[str, float]] = []
        for index in indices:
            real = complex(coefficients[index]).real
            if abs(real) > weight_threshold:
                ordered.append((labels[index], real))
        return ordered

    @staticmethod
    def _build_step_terms(
        terms: list[tuple[str, float]],
        *,
        order: int,
        delta: float,
        atol: float,
    ) -> list[ExponentiatedPauliTerm]:
        r"""Build the exponentiated-Pauli terms for a single time division of length ``delta``.

        The symbolic Zassenhaus factors (rotation-angle prefactor + power of ``t``)
        are evaluated at ``t = delta``: factor ``f`` contributes the rotation
        :math:`e^{-i\,\theta\,P}` with ``theta = f.angle_coefficient * delta**f.degree``.

        Args:
            terms: ``(pauli_label, coefficient)`` pairs of the Hermitian Hamiltonian.
            order: Zassenhaus expansion order.
            delta: Evolution time of a single division.
            atol: Absolute tolerance for dropping negligible rotations.

        Returns:
            Ordered list of :class:`ExponentiatedPauliTerm` for one division.

        """
        if not terms:
            Logger.warn("No coefficients above the tolerance; returning empty term list.")
            return []

        # ``zassenhaus_factors`` returns factors f_0, f_1, ... defining the product
        # U = exp(f_0) exp(f_1) ... with f_0 *leftmost*.  Downstream consumers
        # (``PauliSequenceMapper`` and the container's matrix reconstruction) apply
        # ``step_terms[0]`` *first* (i.e. rightmost in the matrix product), so the
        # factor list is reversed here to preserve the intended ordering.
        #
        # Structurally negligible factors are already pruned inside
        # ``zassenhaus_factors`` (via the ``t``-independent coefficient threshold);
        # the angle is *not* re-thresholded here, since ``angle = coeff * delta**degree``
        # shrinks with the time step and would otherwise drop high-order corrections at
        # small ``t``, spoiling the O(t**(order+1)) error scaling.
        factors = zassenhaus_factors(terms, order, tol=atol)

        return [
            ExponentiatedPauliTerm(
                pauli_term=dict(factor.pauli_term),
                angle=factor.angle_coefficient * (delta**factor.degree),
            )
            for factor in reversed(factors)
        ]

    def name(self) -> str:
        """Return the name of the unitary builder."""
        return "zassenhaus"

    def type_name(self) -> str:
        """Return hamiltonian_unitary_builder as the algorithm type name."""
        return "hamiltonian_unitary_builder"
