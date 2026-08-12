r"""QDK/Chemistry subspace oracles marking a QPE phase register."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from collections.abc import Callable

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import Circuit, Settings
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.base import UnitaryRepresentation
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "QPESubspaceMarking",
    "QPESubspaceMarkingSettings",
    "SubspaceOracleFactory",
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


def _nearest_bin_on_branch(
    target_energy: float,
    eigenvalue_from_phase: Callable[[float], float],
    phase_bin_count: int,
    branch: tuple[int, int],
) -> int:
    """Invert a monotone branch of the phase-to-energy map by bisection on the bin index.

    Args:
        target_energy: The energy to locate.
        eigenvalue_from_phase: The phase-to-energy map of the unitary the QPE circuit estimates.
        phase_bin_count: Total number of phase bins, :math:`2^n`.
        branch: Inclusive ``(first, last)`` bin indices over which the map is monotone.

    Returns:
        The bin on this branch whose energy is closest to the target. A target outside the
        energy range the branch spans lands on whichever end of the branch is nearest it.

    """

    def energy_of(phase_bin: int) -> float:
        return eigenvalue_from_phase(phase_bin / phase_bin_count)

    first, last = branch
    first_energy, last_energy = energy_of(first), energy_of(last)
    increasing = last_energy >= first_energy
    # A branch spans a closed energy interval, so a target beyond an end is nearest that end.
    low, high = min(first_energy, last_energy), max(first_energy, last_energy)
    clamped_energy = min(max(target_energy, low), high)

    while last - first > 1:
        middle = (first + last) // 2
        if (energy_of(middle) <= clamped_energy) == increasing:
            first = middle
        else:
            last = middle
    return first if abs(energy_of(first) - target_energy) <= abs(energy_of(last) - target_energy) else last


def _phase_bins_from_energy(
    target_energy: float,
    eigenvalue_from_phase: Callable[[float], float],
    num_phase_qubits: int,
) -> list[tuple[int, int]]:
    r"""Convert a target energy into the phase bins a QPE circuit maps it to.

    QPE reports a phase, and each unitary encoding defines its own phase-to-energy law: the
    block-encoding post-processing equation :math:`E = \lambda\cos(2\pi\varphi)` for a
    qubitization walk, and :math:`E = -\arg(e^{2\pi i\varphi})/t` for a time evolution. That
    law is read off the unitary representation and inverted numerically here, so no encoding
    is hard-coded.

    The law is not injective, but it is monotone on each half of the phase circle, so each
    half is inverted separately and the halves whose best bin is closest to the target are
    kept. A walk is symmetric about :math:`\varphi = 1/2`, so an interior energy ties on both
    halves and both mirrored bins are marked; a time evolution has no such symmetry and
    marks one. Energies outside the representable band land on the bin at its edge.

    Args:
        target_energy: The energy whose eigenspace should be marked.
        eigenvalue_from_phase: The phase-to-energy map, taking a phase fraction in ``[0, 1)``.
        num_phase_qubits: Width of the QPE phase register.

    Returns:
        Sorted, pairwise-disjoint half-open bin ranges covering every branch the energy hits.

    Raises:
        ValueError: If the energy is not finite or the phase register is empty.

    """
    if not math.isfinite(target_energy):
        raise ValueError(f"target_energy must be a finite energy. Got {target_energy}.")
    if num_phase_qubits < 1:
        raise ValueError(f"num_phase_qubits must be a positive integer. Got {num_phase_qubits}.")

    phase_bin_count = 1 << num_phase_qubits
    half = phase_bin_count // 2
    # The map is monotone on each half of the phase circle. The upper half starts one bin past
    # the midpoint, because a time-evolution phase wraps discontinuously there.
    branches = [(first, last) for first, last in ((0, half), (half + 1, phase_bin_count - 1)) if first <= last]

    bins = [
        _nearest_bin_on_branch(target_energy, eigenvalue_from_phase, phase_bin_count, branch) for branch in branches
    ]
    errors = [abs(eigenvalue_from_phase(phase_bin / phase_bin_count) - target_energy) for phase_bin in bins]
    # Mirrored bins of a symmetric law tie only up to rounding, so compare within a tolerance.
    closest = min(errors)
    tolerance = closest + 1e-9 * max(1.0, abs(target_energy), closest)
    return _merge_bin_ranges(
        [(phase_bin, phase_bin + 1) for phase_bin, error in zip(bins, errors, strict=True) if error <= tolerance]
    )


class QPESubspaceMarkingSettings(Settings):
    r"""Settings for the QPE subspace marking oracle."""

    def __init__(self):
        """Initialize the settings for the QPE subspace marking oracle."""
        super().__init__()
        self._set_default(
            "target_energy",
            "double",
            math.nan,
            "Energy whose QPE phase bins are marked. Required; there is no meaningful default.",
        )


class QPESubspaceMarking(Algorithm):
    r"""Build a subspace oracle marking the QPE phase bins that hold a target energy.

    A QPE circuit with :math:`n` phase qubits writes the phase :math:`\varphi` of the
    eigenvalue :math:`e^{2\pi i\varphi}` into the bin :math:`\lfloor 2^n\varphi\rceil`, so an
    eigenspace is selected by the bin its phase falls in. The target is named as an energy,
    and the phase it corresponds to is recovered from the unitary representation the QPE
    circuit estimates: its
    :meth:`~qdk_chemistry.data.unitary_representation.containers.base.UnitaryContainer.eigenvalue_from_phase`
    is the post-processing equation QPE results are read with, and inverting it turns the
    energy back into a phase.

    For a qubitization walk that equation is the block-encoding relation
    :math:`E = \lambda\cos(2\pi\varphi)`, with :math:`\lambda` the L1 norm of the Hamiltonian,
    so both signs of the phase occur and one energy is marked in two mirrored bins. A time
    evolution follows :math:`E = -\arg/t` instead, and the same inversion handles it without
    special casing. An energy the register cannot resolve exactly takes the nearest bin, and
    one outside the representable band takes the bin at its edge.

    The circuit it returns is the ``good_state_oracle`` of
    :class:`~qdk_chemistry.algorithms.amplitude_amplification.amplitude_amplification.AmplitudeAmplification`,
    and the QPE circuit it reads is that algorithm's ``state_prep_oracle``.
    """

    def __init__(self):
        """Initialize the QPE subspace marking oracle."""
        Logger.trace_entering()
        super().__init__()
        self._settings = QPESubspaceMarkingSettings()

    def type_name(self) -> str:
        """Return the algorithm type name as subspace_oracle."""
        return "subspace_oracle"

    def name(self) -> str:
        """Return the algorithm name as qdk_qpe_subspace."""
        return "qdk_qpe_subspace"

    def _run_impl(
        self,
        qpe_circuit: Circuit,
        unitary_representation: UnitaryRepresentation,
    ) -> Circuit:
        r"""Build the subspace oracle for a QPE circuit.

        Args:
            qpe_circuit: The measurement-free QPE circuit whose phase register is marked.
            unitary_representation: The unitary the QPE circuit estimates. It supplies the width
                of the register the unitary acts on and the phase-to-energy map that is inverted
                to locate the target energy.

        Returns:
            A circuit for use as the ``good_state_oracle`` of ``AmplitudeAmplification``.

        Raises:
            ValueError: If ``target_energy`` is unset or invalid, if the circuit is not a
                standard QPE circuit, or if the unitary does not match the circuit's register.
            TypeError: If ``unitary_representation`` is not a ``UnitaryRepresentation``.

        """
        Logger.trace_entering()
        factory = qpe_circuit._qsharp_factory  # noqa: SLF001
        parameters = factory.parameter if factory is not None else None
        if not isinstance(parameters, dict) or not {"numBits", "systems", "numAncillaQubits"} <= parameters.keys():
            raise ValueError("qpe_circuit must be a standard QPE circuit built by the qdk_standard builder.")
        if not isinstance(unitary_representation, UnitaryRepresentation):
            raise TypeError(
                "The subspace oracle requires the UnitaryRepresentation the QPE circuit estimates. "
                f"Got {type(unitary_representation)}."
            )

        num_phase_qubits = parameters["numBits"]
        num_system_qubits = len(parameters["systems"])
        # The unitary acts on the system register plus whatever ancillas its encoding needs.
        num_ancilla_qubits = unitary_representation.get_num_qubits() - num_system_qubits
        if num_ancilla_qubits != parameters["numAncillaQubits"]:
            raise ValueError(
                f"unitary_representation acts on {unitary_representation.get_num_qubits()} qubits, which does not "
                f"match the {num_system_qubits + parameters['numAncillaQubits']} system and ancilla qubits of the "
                "QPE circuit."
            )

        target_energy = float(self._settings.get("target_energy"))
        if math.isnan(target_energy):
            raise ValueError("The target_energy setting must be set to the energy of the subspace to mark.")
        bin_ranges = _phase_bins_from_energy(
            target_energy,
            unitary_representation.get_container().eigenvalue_from_phase,
            num_phase_qubits,
        )

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


class SubspaceOracleFactory(AlgorithmFactory):
    """Factory class for creating subspace oracle instances."""

    def __init__(self):
        """Initialize the SubspaceOracleFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as subspace_oracle."""
        return "subspace_oracle"

    def default_algorithm_name(self) -> str:
        """Return qdk_qpe_subspace as the default algorithm name."""
        return "qdk_qpe_subspace"
