r"""QDK/Chemistry amplitude amplification.

Amplitude amplification boosts the probability that a prepared state is found in
a *good* subspace.  It is built from two independent halves:

* a **preparation** :math:`U_\psi` -- any measurement-free, adjointable circuit,
  and
* a **marking oracle** -- any adjointable predicate flipping a target qubit on
  the good subspace.

This algorithm knows nothing about either half beyond those two contracts, and
nothing about execution: it builds the amplified circuit and stops.  Running it
and deciding which shots landed in the good subspace belong to the caller.

The reflection about the prepared state is realised as
:math:`U_\psi S_0 U_\psi^\dagger`, since :math:`|0\cdots0\rangle` is the only
state the hardware recognises in a single gate.  Every round therefore costs one
:math:`U_\psi` and one :math:`U_\psi^\dagger`, so a ``k``-round circuit contains
:math:`2k+1` preparations -- the same :math:`2k+1` that appears in the acceptance
probability :math:`\sin^2((2k+1)\vartheta)`, and what turns an :math:`O(1/a)`
repeat-until-success loop into :math:`O(1/\sqrt{a})`.

References:
    L. Lin, *Lecture Notes on Quantum Algorithms for Scientific Computation*,
    arXiv:2201.08309, Chapter 2.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from collections.abc import Callable, Sequence
from typing import Any

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, Settings
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "AmplitudeAmplification",
    "AmplitudeAmplificationFactory",
    "AmplitudeAmplificationSettings",
]


class AmplitudeAmplificationSettings(Settings):
    r"""Settings for amplitude amplification.

    The round-count settings answer the central practical question: how many
    rounds can be run when the overlap of the guiding state is only known
    approximately?

    Plain amplitude amplification rotates by :math:`(2k+1)\vartheta` with
    :math:`\vartheta = \arcsin\sqrt{a}`, so its acceptance probability
    :math:`\sin^2((2k+1)\vartheta)` *falls back to zero* once the rotation runs
    past :math:`\pi/2`.  Guessing ``k`` from an uncertain overlap therefore
    risks overshooting, and an overshoot is indistinguishable from a small
    overlap in the measured counts -- it fails silently.

    The round count is consequently derived from the Yoder-Low-Chuang
    fixed-point schedule, which replaces that sinusoid by a Chebyshev plateau:
    acceptance climbs monotonically and then stays above
    :math:`1 - \text{tolerance}^2` for *every* overlap at or above
    ``min_overlap``.  Only a **lower** bound is required, which is the bound a
    classical overlap estimate actually provides, and no overshoot is possible.
    The guarantee costs roughly twice the queries of a perfectly-tuned plain
    schedule.

    Set ``rounds`` explicitly to bypass the schedule and run that many plain
    Grover iterates instead.

    """

    def __init__(self):
        """Initialize the settings for amplitude amplification."""
        super().__init__()
        self._set_default(
            "rounds",
            "int",
            -1,
            "Explicit number of plain Grover iterates. Negative derives a fixed-point schedule instead.",
        )
        self._set_default(
            "min_overlap",
            "double",
            0.0,
            "Lower bound on the probability that the prepared state lands in the good subspace.",
        )
        self._set_default(
            "tolerance",
            "double",
            0.1,
            "Fixed-point amplification tolerance; success is guaranteed to exceed 1 - tolerance^2.",
        )


class AmplitudeAmplification(Algorithm):
    r"""Build an amplitude-amplified circuit around a coherent preparation.

    ``run`` takes the preparation :math:`U_\psi` to reflect about and a marking
    oracle defining the good subspace, and returns the amplified circuit. It
    neither executes that circuit nor interprets its shots.

    Example:
        >>> from qdk_chemistry.algorithms import create  # doctest: +SKIP
        >>> from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # doctest: +SKIP
        >>> qpe = create("qpe_circuit_builder", "qdk_standard")  # doctest: +SKIP
        >>> qpe.settings().update("num_bits", 8)  # doctest: +SKIP
        >>> qpe.settings().update("measurement", "none")  # doctest: +SKIP
        >>> preparation = qpe.run(  # doctest: +SKIP
        ...     state_preparation=guiding_state, qubit_hamiltonian=hamiltonian
        ... )[0]
        >>> marker = QSHARP_UTILS.StandardPhaseEstimation.MakeAcceptanceMarkerOp(
        ...     8, [], [17, 18, 19]
        ... )  # doctest: +SKIP
        >>> aa = create("amplitude_amplification")  # doctest: +SKIP
        >>> aa.settings().update("min_overlap", 0.05)  # doctest: +SKIP
        >>> circuit = aa.run(preparation, marker, num_qubits=12)  # doctest: +SKIP

    """

    def __init__(self):
        """Initialize amplitude amplification."""
        Logger.trace_entering()
        super().__init__()
        self._settings = AmplitudeAmplificationSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification."""
        return "amplitude_amplification"

    def name(self) -> str:
        """Return the algorithm name as qdk_amplitude_amplification."""
        return "qdk_amplitude_amplification"

    def _run_impl(
        self,
        preparation: Circuit,
        marking_oracle: Callable[..., Any],
        num_qubits: int,
        measured_indices: Sequence[int] | None = None,
    ) -> Circuit:
        r"""Wrap a coherent preparation in the amplification loop.

        Args:
            preparation: A measurement-free circuit carrying an adjointable Q#
                operation; the loop reflects about the state it prepares.
            marking_oracle: An adjointable Q# operation of signature
                ``(Qubit[], Qubit) => Unit is Adj`` that flips its target on the
                good subspace and leaves the register otherwise unchanged.
            num_qubits: Size of the register both callables act on.
            measured_indices: Register indices to measure, in the order they
                should appear in each shot. Defaults to the whole register.

        Returns:
            The amplified circuit, containing :math:`2k+1` preparations.

        Raises:
            TypeError: If ``preparation`` carries no adjointable Q# operation.
            ValueError: If ``num_qubits`` or ``measured_indices`` is out of range.

        """
        Logger.trace_entering()
        operation = preparation._qsharp_op  # noqa: SLF001
        if operation is None:
            raise TypeError(
                "Amplitude amplification reflects about the prepared state, which requires applying "
                "the preparation's adjoint. Pass a measurement-free circuit carrying an adjointable "
                "Q# operation, such as a qdk_standard QPE circuit built with measurement='none'."
            )
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be positive. Got {num_qubits}.")

        indices = list(range(num_qubits)) if measured_indices is None else [int(index) for index in measured_indices]
        if any(index < 0 or index >= num_qubits for index in indices):
            raise ValueError(f"measured_indices must lie in [0, {num_qubits}). Got {indices}.")

        rounds = self.resolve_rounds()
        amplification = QSHARP_UTILS.AmplitudeAmplification
        parameters: dict[str, Any] = {
            "preparation": operation,
            "markingOracle": marking_oracle,
        }

        if int(self._settings.get("rounds")) < 0:
            mark_phases, state_phases = self.fixed_point_phases(rounds, float(self._settings.get("tolerance")))
            parameters["markPhases"] = mark_phases
            parameters["statePhases"] = state_phases
            program = amplification.MakeFixedPointAmplifiedCircuit
        else:
            parameters["rounds"] = rounds
            program = amplification.MakeAmplifiedCircuit

        parameters["numQubits"] = num_qubits
        parameters["measuredIndices"] = indices
        Logger.info(f"Amplified circuit uses {2 * rounds + 1} preparations.")
        return Circuit(qsharp_factory=QsharpFactoryData(program=program, parameter=parameters))

    def resolve_rounds(self) -> int:
        """Resolve the number of amplification rounds from the settings.

        An explicit non-negative ``rounds`` always wins and selects that many
        plain Grover iterates. Otherwise the count is derived from the
        Yoder-Low-Chuang fixed-point schedule, which needs only ``min_overlap``
        and ``tolerance`` and cannot overshoot.

        Returns:
            The number of amplification rounds to run.

        Raises:
            ValueError: If no explicit ``rounds`` was given and ``min_overlap``
                is not a usable lower bound.

        """
        rounds = int(self._settings.get("rounds"))
        if rounds >= 0:
            return rounds

        min_overlap = float(self._settings.get("min_overlap"))
        if min_overlap <= 0.0:
            raise ValueError(
                "Deriving a round count needs a positive 'min_overlap' lower bound on the overlap "
                "of the guiding state with the good subspace. Set 'min_overlap', or set 'rounds' "
                "explicitly to run a fixed number of plain Grover iterates."
            )
        return self.fixed_point_rounds(min_overlap, float(self._settings.get("tolerance")))

    @staticmethod
    def _validate_overlap(overlap: float, name: str = "overlap") -> None:
        """Raise if ``overlap`` is not a usable squared overlap.

        Args:
            overlap: The value to check.
            name: Name used in the error message.

        Raises:
            ValueError: If ``overlap`` is not finite or does not lie in ``(0, 1]``.

        """
        if not math.isfinite(overlap) or not 0.0 < overlap <= 1.0:
            raise ValueError(f"{name} must be finite and lie in (0, 1]. Got {overlap}.")

    @staticmethod
    def _validate_schedule_rounds(rounds: int) -> None:
        """Raise if ``rounds`` is not a valid round count.

        Args:
            rounds: The number of amplification rounds.

        Raises:
            ValueError: If ``rounds`` is negative.

        """
        if rounds < 0:
            raise ValueError(f"rounds must be nonnegative. Got {rounds}.")

    @classmethod
    def _rotation_angle(cls, overlap: float) -> float:
        r"""Return the half-rotation angle :math:`\vartheta = \arcsin\sqrt{a}`.

        Args:
            overlap: The squared overlap ``a`` of the prepared state with the good
                subspace, in ``(0, 1]``.

        Returns:
            The angle in radians, in ``(0, pi/2]``.

        """
        cls._validate_overlap(overlap)
        return math.asin(math.sqrt(overlap))

    @classmethod
    def success_probability(cls, overlap: float, rounds: int) -> float:
        r"""Return the acceptance probability :math:`\sin^2((2k+1)\vartheta)`.

        After ``k`` rounds the prepared state has rotated by :math:`(2k+1)\vartheta`
        inside the two-dimensional invariant subspace, so the probability of
        landing in the good subspace is :math:`\sin^2((2k+1)\vartheta)` with
        :math:`\vartheta = \arcsin\sqrt{a}`.

        Args:
            overlap: The squared overlap of the prepared state with the good subspace.
            rounds: The number of amplification rounds ``k``.

        Returns:
            The probability of measuring the good subspace after ``rounds`` rounds.

        """
        cls._validate_schedule_rounds(rounds)
        angle = cls._rotation_angle(overlap)
        return math.sin((2 * rounds + 1) * angle) ** 2

    @classmethod
    def fixed_point_rounds(cls, min_overlap: float, tolerance: float) -> int:
        r"""Return the iterate count for fixed-point amplification.

        The Yoder-Low-Chuang schedule reaches acceptance probability at least
        :math:`1 - \delta^2` for *every* overlap at or above ``min_overlap`` once
        the number of queries satisfies
        :math:`L \ge \log(2/\delta)/\sqrt{a_{\min}}`. This returns the smallest
        ``l`` with ``L = 2l + 1`` meeting that bound.

        Args:
            min_overlap: Lower bound on the squared overlap.
            tolerance: The failure amplitude ``delta`` in ``(0, 1)``; the acceptance
                probability is at least ``1 - delta ** 2``.

        Returns:
            The number of iterates ``l``, so that ``2 * l + 1`` queries are used.

        Raises:
            ValueError: If ``tolerance`` is not in ``(0, 1)``.

        """
        cls._validate_overlap(min_overlap, "min_overlap")
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError(f"tolerance must lie in (0, 1). Got {tolerance}.")
        queries = math.log(2.0 / tolerance) / math.sqrt(min_overlap)
        return max(1, math.ceil((math.ceil(queries) - 1) / 2.0))

    @staticmethod
    def fixed_point_phases(rounds: int, tolerance: float) -> tuple[list[float], list[float]]:
        r"""Return the Yoder-Low-Chuang phase sequence for fixed-point amplification.

        With ``L = 2 * rounds + 1`` queries and

        .. math::

            \gamma^{-1} = T_{1/L}(1/\delta)
                        = \cosh\!\big(L^{-1}\operatorname{arccosh}(1/\delta)\big),

        the state-reflection phases are

        .. math::

            \beta_j = 2\operatorname{arccot}\!\big(\tan(2\pi j/L)\sqrt{1-\gamma^2}\big),
            \qquad j = 1,\dots,l ,

        and the mark phases are the same list reversed,
        :math:`\alpha_j = \beta_{l+1-j}`. Both reflections are taken in the
        ``I - (1 - e^{i\varphi}) P`` convention used by the Q# implementation, with
        the mark applied before the state reflection, which is why no sign flip
        appears between the two sequences.

        The resulting acceptance probability climbs monotonically up to
        :math:`a = 1 - T_{1/L}(1/\delta)^{-2}` and from there stays inside
        :math:`[1-\delta^2, 1]` for every larger overlap -- a plateau, not a global
        maximum. There is no first maximum to run past, so overshoot is impossible
        above the design threshold.

        Args:
            rounds: The number of iterates ``l``; ``2 * l + 1`` queries are used.
            tolerance: The failure amplitude ``delta`` in ``(0, 1)``.

        Returns:
            The mark phases ``alpha`` and the state phases ``beta``, both of length
            ``rounds`` and ordered by iterate.

        Raises:
            ValueError: If ``rounds`` is not positive or ``tolerance`` is not in
                ``(0, 1)``.

        """
        if rounds < 1:
            raise ValueError(f"rounds must be positive. Got {rounds}.")
        if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError(f"tolerance must lie in (0, 1). Got {tolerance}.")

        queries = 2 * rounds + 1
        gamma = 1.0 / math.cosh(math.acosh(1.0 / tolerance) / queries)
        scale = math.sqrt(max(0.0, 1.0 - gamma * gamma))

        # arccot with range (0, pi), so that the phases are continuous in j.
        state_phases = [
            2.0 * math.atan2(1.0, math.tan(2.0 * math.pi * j / queries) * scale) for j in range(1, rounds + 1)
        ]
        mark_phases = list(reversed(state_phases))
        return mark_phases, state_phases


class AmplitudeAmplificationFactory(AlgorithmFactory):
    """Factory class for creating AmplitudeAmplification instances."""

    def __init__(self):
        """Initialize the AmplitudeAmplificationFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification."""
        return "amplitude_amplification"

    def default_algorithm_name(self) -> str:
        """Return qdk_amplitude_amplification as the default algorithm name."""
        return "qdk_amplitude_amplification"
