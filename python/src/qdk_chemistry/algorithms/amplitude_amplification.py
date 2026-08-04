r"""QDK/Chemistry amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from collections.abc import Sequence
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
    r"""Settings for amplitude amplification."""

    def __init__(self):
        """Initialize the settings for amplitude amplification."""
        super().__init__()
        self._set_default(
            "rounds",
            "int",
            -1,
            "Number of amplitude amplification rounds. -1 derives a fixed-point schedule instead.",
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
    r"""Build an amplitude-amplified circuit."""

    def __init__(self):
        """Initialize amplitude amplification."""
        Logger.trace_entering()
        super().__init__()
        self._settings = AmplitudeAmplificationSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification."""
        return "amplitude_amplification"

    def name(self) -> str:
        """Return the algorithm name as qdk."""
        return "qdk"

    def _run_impl(
        self,
        preparation: Circuit,
        marking_oracle: Circuit,
        num_qubits: int,
    ) -> Circuit:
        r"""Build an amplitude-amplified circuit.

        Args:
            preparation: Prepare the initial state.
            marking_oracle: Mark the good subspace with a phase flip.
            num_qubits: Size of the register both callables act on.
            measured_indices: Register indices to measure, in output order.
                Defaults to the whole register.

        Returns:
            The amplified circuit.

        Raises:
            TypeError: If either circuit carries no adjointable Q# operation.
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
        marking_operation = marking_oracle._qsharp_op  # noqa: SLF001
        if marking_operation is None:
            raise TypeError(
                "Amplitude amplification requires a marking oracle circuit carrying an adjointable "
                "Q# operation of type (Qubit[], Qubit) => Unit is Adj."
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
            "markingOracle": marking_operation,
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

        An explicit non-negative ``rounds`` wins; otherwise the count comes from
        the fixed-point schedule.

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

        Args:
            overlap: The squared overlap of the prepared state with the good subspace.
            rounds: The number of amplification rounds ``k``.

        Returns:
            The probability of measuring the good subspace after ``rounds`` rounds.

        """
        if rounds < 0:
            raise ValueError(f"rounds must be nonnegative. Got {rounds}.")
        angle = cls._rotation_angle(overlap)
        return math.sin((2 * rounds + 1) * angle) ** 2

    @classmethod
    def fixed_point_rounds(cls, min_overlap: float, tolerance: float) -> int:
        r"""Return the iterate count for fixed-point amplification.

        The schedule reaches acceptance :math:`\ge 1 - \delta^2` for every overlap
        at or above ``min_overlap`` once :math:`L \ge \log(2/\delta)/\sqrt{a_{\min}}`.
        Returns the smallest ``l`` with ``L = 2l + 1`` meeting that bound.

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

        and the mark phases are the same list reversed. Both reflections use the
        ``I - (1 - e^{i\varphi}) P`` convention of the Q# implementation, with the
        mark applied first, so no sign flip appears between the sequences.

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
        """Return qdk as the default algorithm name."""
        return "qdk"
