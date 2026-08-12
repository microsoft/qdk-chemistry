r"""QDK/Chemistry amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

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
            1,
            "Number of Grover amplitude amplification rounds.",
        )


class AmplitudeAmplification(Algorithm):
    r"""Build an amplitude-amplified circuit.

    Amplitude amplification raises the probability of measuring a state in a chosen
    "good" subspace. Given a state preparation :math:`U` with
    :math:`|\psi\rangle = U|0\rangle` and an oracle that reflects about the good
    subspace, one round applies the Grover iterate
    :math:`Q = -(2|\psi\rangle\langle\psi| - I)(I - 2\Pi_G)`, a rotation by
    :math:`2\vartheta` in the plane spanned by the good and bad components. If the good
    subspace initially carries probability :math:`a = \sin^2\vartheta`, then after
    :math:`k` rounds it carries

    .. math::

        p_k = \sin^2\!\big((2k+1)\arcsin\sqrt{a}\big),

    so :math:`O(1/\sqrt{a})` rounds suffice where direct sampling would need
    :math:`O(1/a)` shots. More rounds are not always better: past the first maximum near
    :math:`k \approx \pi/(4\arcsin\sqrt{a})` the success probability falls again, so pick
    ``rounds`` from an estimate of :math:`a`.

    Reference: L. Lin, *Lecture Notes on Quantum Algorithms for Scientific Computation*,
    arXiv:2201.08309, Chapter 2.
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
        """Return the algorithm name as qdk_base."""
        return "qdk_base"

    def _run_impl(
        self,
        state_prep_oracle: Circuit,
        good_state_oracle: Circuit,
    ) -> Circuit:
        r"""Build an amplitude-amplified circuit.

        Args:
            state_prep_oracle: Prepares the initial state. Must carry an adjointable Q# operation.
            good_state_oracle: Flips a flag qubit on the good subspace. Must carry an adjointable Q# operation.

        Returns:
            The amplified circuit, measuring the whole register. Its ``qsharp_op`` is the same
            amplification without measurement, for callers that append their own.

        Raises:
            TypeError: If either circuit carries no adjointable Q# operation.
            ValueError: If the ``rounds`` setting is negative.
            RuntimeError: If the state preparation cannot be resource estimated for its width.

        """
        Logger.trace_entering()
        operation = state_prep_oracle._qsharp_op  # noqa: SLF001
        if operation is None:
            raise TypeError("Amplitude amplification requires a state prep oracle qsharp operation.")
        good_state_operation = good_state_oracle._qsharp_op  # noqa: SLF001
        if good_state_operation is None:
            raise TypeError("Amplitude amplification requires a good state oracle qsharp operation.")

        # A Q# callable carries no arity, so the register width is taken from a resource
        # estimate of the state preparation.
        try:
            num_qubits = int(state_prep_oracle.estimate().logical_counts["numQubits"])
        except Exception as error:
            raise RuntimeError(
                "Could not read the register width from a resource estimate of the state prep oracle."
            ) from error

        rounds = int(self._settings.get("rounds"))
        if rounds < 0:
            raise ValueError(f"rounds must be nonnegative. Got {rounds}.")
        amplification = QSHARP_UTILS.AmplitudeAmplification
        parameters: dict[str, Any] = {
            "statePrepOracle": operation,
            "goodStateOracle": good_state_operation,
            "rounds": rounds,
            "numQubits": num_qubits,
        }
        Logger.info(f"Amplified circuit uses {2 * rounds + 1} state preparations.")
        return Circuit(
            qsharp_factory=QsharpFactoryData(program=amplification.MakeAmplifiedCircuit, parameter=parameters),
            qsharp_op=amplification.MakeAmplifiedStateOp(operation, good_state_operation, rounds),
        )


class AmplitudeAmplificationFactory(AlgorithmFactory):
    """Factory class for creating AmplitudeAmplification instances."""

    def __init__(self):
        """Initialize the AmplitudeAmplificationFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification."""
        return "amplitude_amplification"

    def default_algorithm_name(self) -> str:
        """Return qdk_base as the default algorithm name."""
        return "qdk_base"
