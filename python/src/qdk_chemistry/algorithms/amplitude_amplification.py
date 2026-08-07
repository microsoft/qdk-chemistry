r"""QDK/Chemistry amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import operator
from typing import Any

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, QubitOperator, Settings
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "AmplitudeAmplification",
    "AmplitudeAmplificationFactory",
    "AmplitudeAmplificationSettings",
    "phase_marking_oracle",
]


def _merge_bin_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge half-open bin ranges into a sorted, pairwise-disjoint list."""
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _phase_bins_from_energy_range(
    target_energy_range: tuple[float, float],
    normalization: float,
    num_phase_qubits: int,
) -> list[tuple[int, int]]:
    r"""Convert an energy window into the phase bins a qubitization walk maps it to.

    A qubitization walk on a block encoding of :math:`H/\lambda` has eigenvalues
    :math:`e^{\pm i\arccos(E/\lambda)}`, the inverse of
    :meth:`~qdk_chemistry.data.unitary_representation.containers.quantum_walk.QuantumWalkContainer.eigenvalue_from_phase`.
    Both signs occur, so one energy lands in two mirrored bins and both must be marked.
    """
    try:
        low_energy, high_energy = (float(bound) for bound in target_energy_range)
    except (TypeError, ValueError) as error:
        raise TypeError("target_energy_range must be a (low, high) tuple of floats.") from error
    if not low_energy < high_energy:
        raise ValueError(f"target_energy_range must satisfy low < high. Got {target_energy_range}.")
    if not normalization > 0.0:
        raise ValueError(f"normalization must be positive. Got {normalization}.")

    # arccos is decreasing, so the higher energy gives the lower phase.
    lower_phase = math.acos(min(max(high_energy / normalization, -1.0), 1.0)) / (2 * math.pi)
    upper_phase = math.acos(min(max(low_energy / normalization, -1.0), 1.0)) / (2 * math.pi)

    phase_bin_count = 1 << num_phase_qubits
    start = min(round(lower_phase * phase_bin_count), phase_bin_count - 1)
    stop = min(round(upper_phase * phase_bin_count) + 1, phase_bin_count)
    # The mirrored branch sits at bin -> phase_bin_count - bin.
    mirrored = (max(phase_bin_count - stop + 1, 0), min(phase_bin_count - start + 1, phase_bin_count))
    return _merge_bin_ranges([(start, stop), mirrored])


def phase_marking_oracle(
    qpe_circuit: Circuit,
    target_phase_bins: tuple[int, int] | None = None,
    *,
    target_energy_range: tuple[float, float] | None = None,
    qubit_hamiltonian: QubitOperator | None = None,
) -> Circuit:
    r"""Build a good state oracle marking a range of phase bins of a QPE circuit.

    A QPE circuit with :math:`n` phase qubits writes the phase :math:`\varphi` of the
    eigenvalue :math:`e^{2\pi i\varphi}` into the bin :math:`\lfloor 2^n\varphi\rceil`, so a
    target eigenvalue is selected by the bin its phase falls in. Bins are marked over the
    half-open interval ``(start, stop)``.

    The target can be given as an energy window instead, which only makes sense for a QPE
    circuit built on a qubitization walk: its eigenvalues are
    :math:`e^{\pm i\arccos(E/\lambda)}`, where :math:`\lambda` is the L1 norm of the
    Hamiltonian, so the window is converted with :math:`\varphi = \arccos(E/\lambda)/2\pi`.
    Both signs occur, so an energy is marked in two mirrored bins. Any other encoding, a
    Trotter step for instance, follows a different law and has to use ``target_phase_bins``.
    Energy bounds are clipped to the representable range :math:`[-\lambda, \lambda]`, so
    passing an infinite bound gives a one-sided threshold.

    Args:
        qpe_circuit: The measurement-free QPE circuit whose phase register is marked.
        target_phase_bins: Half-open phase-bin interval ``(start, stop)`` to mark.
        target_energy_range: Half-open energy window ``(low, high)``, an alternative to ``target_phase_bins``.
        qubit_hamiltonian: The Hamiltonian the QPE circuit estimates, supplying :math:`\lambda`.

    Returns:
        A circuit for use as the ``good_state_oracle`` of :class:`AmplitudeAmplification`.

    Raises:
        ValueError: If the target range is invalid or the circuit is not a standard QPE circuit.
        TypeError: If the range endpoints are not the expected type.

    """
    factory = qpe_circuit._qsharp_factory  # noqa: SLF001
    parameters = factory.parameter if factory is not None else None
    if not isinstance(parameters, dict) or not {"numBits", "systems", "numAncillaQubits"} <= parameters.keys():
        raise ValueError("qpe_circuit must be a standard QPE circuit built by the qdk_standard builder.")

    num_phase_qubits = parameters["numBits"]
    num_system_qubits = len(parameters["systems"])
    num_ancilla_qubits = parameters["numAncillaQubits"]
    phase_bin_count = 1 << num_phase_qubits

    if (target_phase_bins is None) == (target_energy_range is None):
        raise ValueError("Pass exactly one of target_phase_bins or target_energy_range.")

    if target_energy_range is not None:
        if qubit_hamiltonian is None:
            raise ValueError("target_energy_range requires qubit_hamiltonian to supply the L1 norm.")
        bin_ranges = _phase_bins_from_energy_range(
            target_energy_range, qubit_hamiltonian.schatten_norm, num_phase_qubits
        )
    else:
        try:
            start, stop = target_phase_bins  # type: ignore[misc]
        except (TypeError, ValueError) as error:
            raise TypeError("target_phase_bins must be a (start, stop) tuple.") from error
        try:
            lower_bound = operator.index(start)
            upper_bound = operator.index(stop)
        except TypeError as error:
            raise TypeError("target_phase_bins endpoints must be integers.") from error

        if not 0 <= lower_bound < upper_bound <= phase_bin_count:
            raise ValueError(
                f"target_phase_bins must satisfy 0 <= start < stop <= {phase_bin_count}. Got {target_phase_bins}."
            )
        bin_ranges = [(lower_bound, upper_bound)]

    ancilla_indices = list(range(num_system_qubits, num_system_qubits + num_ancilla_qubits))
    lower_bounds = [start for start, _ in bin_ranges]
    upper_bounds = [stop for _, stop in bin_ranges]
    parameters = {
        "numPhaseQubits": num_phase_qubits,
        "signalAncillaIndices": ancilla_indices,
        "lowerBounds": lower_bounds,
        "upperBounds": upper_bounds,
        "numQubits": num_phase_qubits + num_system_qubits + num_ancilla_qubits,
    }
    amplification = QSHARP_UTILS.AmplitudeAmplification
    operation = amplification.MarkTargetStateOp(num_phase_qubits, ancilla_indices, lower_bounds, upper_bounds)

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
        # `logical_counts` is used rather than indexing the result, because it also resolves
        # the batch shape the estimator returns for a frontier of parameter sets.
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
