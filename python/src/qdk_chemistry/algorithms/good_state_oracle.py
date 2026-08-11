r"""QDK/Chemistry good state oracles for amplitude amplification."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
import operator

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, QubitOperator, Settings
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "GoodStateOracleFactory",
    "PhaseMarkingOracle",
    "PhaseMarkingOracleSettings",
]


def _merge_bin_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge half-open bin ranges into a sorted, pairwise-disjoint list, dropping empty ones."""
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(bin_range for bin_range in ranges if bin_range[0] < bin_range[1]):
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

    Args:
        target_energy_range: Half-open energy window ``(low, high)``, clipped to the representable band.
        normalization: The L1 norm :math:`\lambda` of the Hamiltonian.
        num_phase_qubits: Width of the QPE phase register.

    Returns:
        Sorted, pairwise-disjoint half-open bin ranges covering both signs of the phase.

    Raises:
        ValueError: If the window is empty or the normalization is not positive.
        TypeError: If the bounds are not a pair of floats.

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


class PhaseMarkingOracleSettings(Settings):
    r"""Settings for the phase marking good state oracle."""

    def __init__(self):
        """Initialize the settings for the phase marking oracle."""
        super().__init__()
        self._set_default(
            "target_phase_bins",
            "vector<int>",
            [],
            "Half-open phase-bin interval [start, stop] to mark. Empty selects the energy window instead.",
        )
        self._set_default(
            "target_energy_range",
            "vector<double>",
            [],
            "Half-open energy window [low, high] to mark, an alternative to target_phase_bins.",
        )


class PhaseMarkingOracle(Algorithm):
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

    The circuit it returns is the ``good_state_oracle`` of
    :class:`~qdk_chemistry.algorithms.amplitude_amplification.AmplitudeAmplification`, and
    the QPE circuit it reads is that algorithm's ``state_prep_oracle``.
    """

    def __init__(self):
        """Initialize the phase marking oracle."""
        Logger.trace_entering()
        super().__init__()
        self._settings = PhaseMarkingOracleSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as good_state_oracle."""
        return "good_state_oracle"

    def name(self) -> str:
        """Return the algorithm name as qdk_phase_marking."""
        return "qdk_phase_marking"

    def _run_impl(
        self,
        qpe_circuit: Circuit,
        qubit_hamiltonian: QubitOperator | None = None,
    ) -> Circuit:
        r"""Build the good state oracle for a QPE circuit.

        Args:
            qpe_circuit: The measurement-free QPE circuit whose phase register is marked.
            qubit_hamiltonian: The Hamiltonian the QPE circuit estimates, supplying :math:`\lambda`.

        Returns:
            A circuit for use as the ``good_state_oracle`` of ``AmplitudeAmplification``.

        Raises:
            ValueError: If the target range is invalid or the circuit is not a standard QPE circuit.
            TypeError: If the range endpoints are not the expected type.

        """
        Logger.trace_entering()
        factory = qpe_circuit._qsharp_factory  # noqa: SLF001
        parameters = factory.parameter if factory is not None else None
        if not isinstance(parameters, dict) or not {"numBits", "systems", "numAncillaQubits"} <= parameters.keys():
            raise ValueError("qpe_circuit must be a standard QPE circuit built by the qdk_standard builder.")

        num_phase_qubits = parameters["numBits"]
        num_system_qubits = len(parameters["systems"])
        num_ancilla_qubits = parameters["numAncillaQubits"]
        phase_bin_count = 1 << num_phase_qubits

        target_phase_bins = list(self._settings.get("target_phase_bins"))
        target_energy_range = list(self._settings.get("target_energy_range"))
        if bool(target_phase_bins) == bool(target_energy_range):
            raise ValueError("Set exactly one of the target_phase_bins or target_energy_range settings.")

        if target_energy_range:
            if len(target_energy_range) != 2:
                raise ValueError(f"target_energy_range must hold exactly two bounds. Got {target_energy_range}.")
            if qubit_hamiltonian is None:
                raise ValueError("target_energy_range requires qubit_hamiltonian to supply the L1 norm.")
            bin_ranges = _phase_bins_from_energy_range(
                (target_energy_range[0], target_energy_range[1]),
                qubit_hamiltonian.schatten_norm,
                num_phase_qubits,
            )
        else:
            if len(target_phase_bins) != 2:
                raise ValueError(f"target_phase_bins must hold exactly two bounds. Got {target_phase_bins}.")
            try:
                lower_bound = operator.index(target_phase_bins[0])
                upper_bound = operator.index(target_phase_bins[1])
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
        oracle_parameters = {
            "numPhaseQubits": num_phase_qubits,
            "signalAncillaIndices": ancilla_indices,
            "lowerBounds": lower_bounds,
            "upperBounds": upper_bounds,
            "numQubits": num_phase_qubits + num_system_qubits + num_ancilla_qubits,
        }
        amplification = QSHARP_UTILS.AmplitudeAmplification
        operation = amplification.MarkTargetStateOp(num_phase_qubits, ancilla_indices, lower_bounds, upper_bounds)

        return Circuit(
            qsharp_factory=QsharpFactoryData(program=amplification.MakeMarkedPhaseCircuit, parameter=oracle_parameters),
            qsharp_op=operation,
        )


class GoodStateOracleFactory(AlgorithmFactory):
    """Factory class for creating good state oracle instances."""

    def __init__(self):
        """Initialize the GoodStateOracleFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as good_state_oracle."""
        return "good_state_oracle"

    def default_algorithm_name(self) -> str:
        """Return qdk_phase_marking as the default algorithm name."""
        return "qdk_phase_marking"
