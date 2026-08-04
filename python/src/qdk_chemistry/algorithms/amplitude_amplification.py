r"""QDK/Chemistry amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import operator
from collections.abc import Sequence
from typing import Any, Literal

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, Settings
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "AmplitudeAmplification",
    "AmplitudeAmplificationFactory",
    "AmplitudeAmplificationSettings",
    "phase_marking_oracle",
]


def phase_marking_oracle(
    num_phase_qubits: int,
    *,
    target_indices: Sequence[int] | None = None,
    threshold: int | None = None,
    comparison: Literal["at_or_below", "at_or_above"] | None = None,
) -> Circuit:
    r"""Build a marking-oracle circuit for a little-endian QPE phase register.

    Args:
        num_phase_qubits: Number of phase qubits at the start of the oracle register.
        target_indices: Phase-bin values to mark.
        threshold: Inclusive phase-bin threshold to mark.
        comparison: Threshold comparison direction. Required with ``threshold``;
            must be ``"at_or_below"`` or ``"at_or_above"``.

    Returns:
        A circuit carrying an adjointable ``(Qubit[], Qubit) => Unit`` Q# operation.

    Raises:
        ValueError: If the register size or criterion is invalid.
        TypeError: If a register size, target index, or threshold is not an integer.

    """
    try:
        num_phase_qubits = operator.index(num_phase_qubits)
    except TypeError as error:
        raise TypeError("num_phase_qubits must be an integer.") from error
    if num_phase_qubits < 1:
        raise ValueError(f"num_phase_qubits must be positive. Got {num_phase_qubits}.")
    if (target_indices is None) == (threshold is None):
        raise ValueError("Set exactly one of target_indices or threshold.")

    max_index = (1 << num_phase_qubits) - 1
    amplification = QSHARP_UTILS.AmplitudeAmplification
    if target_indices is not None:
        if comparison is not None:
            raise ValueError("comparison is only valid with threshold.")
        normalized_indices: list[int] = []
        for target_index in target_indices:
            try:
                index = operator.index(target_index)
            except TypeError as error:
                raise TypeError("target_indices must contain only integers.") from error
            if not 0 <= index <= max_index:
                raise ValueError(f"target index must lie in [0, {max_index}]. Got {index}.")
            normalized_indices.append(index)
        if not normalized_indices:
            raise ValueError("target_indices must not be empty.")

        normalized_indices = sorted(set(normalized_indices))
        normalized_threshold = 0
        comparison_code = 0
    else:
        assert threshold is not None
        if comparison not in ("at_or_below", "at_or_above"):
            raise ValueError(
                'comparison must be "at_or_below" or "at_or_above" when threshold is set.'
            )
        try:
            normalized_threshold = operator.index(threshold)
        except TypeError as error:
            raise TypeError("threshold must be an integer.") from error
        if not 0 <= normalized_threshold <= max_index:
            raise ValueError(f"threshold must lie in [0, {max_index}]. Got {normalized_threshold}.")

        normalized_indices = []
        comparison_code = -1 if comparison == "at_or_below" else 1

    parameters = {
        "numPhaseQubits": num_phase_qubits,
        "targetIndices": normalized_indices,
        "threshold": normalized_threshold,
        "comparison": comparison_code,
    }
    make_oracle = amplification.MakePhaseMarkerOp
    operation = make_oracle(num_phase_qubits, normalized_indices, normalized_threshold, comparison_code)

    return Circuit(
        qsharp_factory=QsharpFactoryData(program=make_oracle, parameter=parameters),
        qsharp_op=operation,
    )


class AmplitudeAmplificationSettings(Settings):
    r"""Settings for amplitude amplification."""

    def __init__(self):
        """Initialize the settings for amplitude amplification."""
        super().__init__()
        self._set_default(
            "rounds",
            "int",
            1,
            "Number of Grover amplitude amplification rounds.",
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
        measured_indices: list[int] | None = None,
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

        rounds = int(self._settings.get("rounds"))
        if rounds < 0:
            raise ValueError(f"rounds must be nonnegative. Got {rounds}.")
        amplification = QSHARP_UTILS.AmplitudeAmplification
        parameters: dict[str, Any] = {
            "preparation": operation,
            "markingOracle": marking_operation,
            "rounds": rounds,
            "numQubits": num_qubits,
            "measuredIndices": indices,
        }
        Logger.info(f"Amplified circuit uses {2 * rounds + 1} preparations.")
        return Circuit(
            qsharp_factory=QsharpFactoryData(program=amplification.MakeAmplifiedCircuit, parameter=parameters)
        )

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
