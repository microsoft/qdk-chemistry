r"""QDK/Chemistry amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import operator
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
    "phase_marking_oracle",
]


def phase_marking_oracle(
    qpe_circuit: Circuit,
    target_phase_bins: tuple[int, int],
) -> Circuit:
    r"""Build a good state oracle marking a range of phase bins of a QPE circuit.

    A QPE circuit with :math:`n` phase qubits writes the phase :math:`\varphi` of the
    eigenvalue :math:`e^{2\pi i\varphi}` into the bin :math:`\lfloor 2^n\varphi\rceil`, so a
    target eigenvalue is selected by the bin its phase falls in. Bins are marked over the
    half-open interval ``(start, stop)``. 

    Args:
        qpe_circuit: The measurement-free QPE circuit whose phase register is marked.
        target_phase_bins: Half-open phase-bin interval ``(start, stop)`` to mark.

    Returns:
        A circuit for use as the ``good_state_oracle`` of :class:`AmplitudeAmplification`.

    Raises:
        ValueError: If the target range is invalid or the circuit is not a standard QPE circuit.
        TypeError: If the range endpoints are not integers.

    """
    factory = qpe_circuit._qsharp_factory  # noqa: SLF001
    parameters = factory.parameter if factory is not None else None
    if not isinstance(parameters, dict) or not {"numBits", "systems", "numAncillaQubits"} <= parameters.keys():
        raise ValueError("qpe_circuit must be a standard QPE circuit built by the qdk_standard builder.")

    num_phase_qubits = parameters["numBits"]
    num_system_qubits = len(parameters["systems"])
    num_ancilla_qubits = parameters["numAncillaQubits"]

    try:
        start, stop = target_phase_bins
    except (TypeError, ValueError) as error:
        raise TypeError("target_phase_bins must be a (start, stop) tuple.") from error
    try:
        lower_bound = operator.index(start)
        upper_bound = operator.index(stop)
    except TypeError as error:
        raise TypeError("target_phase_bins endpoints must be integers.") from error

    phase_bin_count = 1 << num_phase_qubits
    if not 0 <= lower_bound < upper_bound <= phase_bin_count:
        raise ValueError(
            f"target_phase_bins must satisfy 0 <= start < stop <= {phase_bin_count}. Got {target_phase_bins}."
        )

    ancilla_indices = list(range(num_system_qubits, num_system_qubits + num_ancilla_qubits))
    parameters = {
        "numPhaseQubits": num_phase_qubits,
        "signalAncillaIndices": ancilla_indices,
        "lowerBound": lower_bound,
        "upperBound": upper_bound,
        "numQubits": num_phase_qubits + num_system_qubits + num_ancilla_qubits,
    }
    amplification = QSHARP_UTILS.AmplitudeAmplification
    operation = amplification.MarkTargetStateOp(num_phase_qubits, ancilla_indices, lower_bound, upper_bound)

    return Circuit(
        qsharp_factory=QsharpFactoryData(program=amplification.MakeMarkedPhaseCircuit, parameter=parameters),
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
    r"""Build an amplitude-amplified circuit.

    Amplitude amplification raises the probability of measuring a state in a chosen
    "good" subspace. Given a state preparation :math:`U` with
    :math:`|\psi\rangle = U|0\rangle` and an oracle that flips a flag qubit on the good
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
        """Return the algorithm name as base."""
        return "base"

    def _run_impl(
        self,
        state_prep_oracle: Circuit,
        good_state_oracle: Circuit,
        num_qubits: int,
    ) -> Circuit:
        r"""Build an amplitude-amplified circuit.

        Args:
            state_prep_oracle: Prepares the initial state. Must carry an adjointable Q# operation.
            good_state_oracle: Flips a flag qubit on the good subspace. Must carry an adjointable Q# operation.
            num_qubits: Size of the register both oracles act on.

        Returns:
            The amplified circuit, measuring the whole register. Its ``qsharp_op`` is the same
            amplification without measurement, for callers that append their own.

        Raises:
            TypeError: If either circuit carries no adjointable Q# operation.
            ValueError: If ``num_qubits`` or the ``rounds`` setting is out of range.

        """
        Logger.trace_entering()
        operation = state_prep_oracle._qsharp_op  # noqa: SLF001
        if operation is None:
            raise TypeError("Amplitude amplification requires a state prep oracle qsharp operation.")
        good_state_operation = good_state_oracle._qsharp_op  # noqa: SLF001
        if good_state_operation is None:
            raise TypeError("Amplitude amplification requires a good state oracle qsharp operation.")
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be positive. Got {num_qubits}.")

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
        """Return base as the default algorithm name."""
        return "base"
