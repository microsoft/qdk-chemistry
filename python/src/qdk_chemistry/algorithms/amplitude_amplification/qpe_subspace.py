r"""QDK/Chemistry subspace oracles marking a QPE phase register."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math
from collections.abc import Callable

from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import (
    QpeCircuitBuilderSettings,
    StandardQpeCircuitBuilder,
)
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "QPESubspaceMarking",
    "QPESubspaceMarkingSettings",
]


def _marked_phase_bins(
    target_energy: float,
    eigenvalue_from_phase: Callable[[float], float],
    num_phase_qubits: int,
) -> list[tuple[int, int]]:
    r"""Return the half-open phase-bin ranges whose energy is at most ``target_energy``.

    ``eigenvalue_from_phase`` is the post-processing equation QPE results are read with, so
    reading it off every bin says which bins hold the marked subspace. Those bins are
    contiguous apart from the wrap at :math:`\varphi = 1`, hence a list of ranges rather than
    one: a qubitization walk follows :math:`E = \lambda\cos(2\pi\varphi)` and accepts a single
    band straddling :math:`\varphi = 1/2`, while a time evolution follows :math:`E = -\arg/t`
    and can accept a band that wraps.

    Args:
        target_energy: The highest energy the marked subspace may hold.
        eigenvalue_from_phase: The phase-to-energy map, taking a phase fraction in ``[0, 1)``.
        num_phase_qubits: Width of the QPE phase register.

    Returns:
        Sorted, pairwise-disjoint half-open bin ranges.

    Raises:
        ValueError: If the energy is not finite or lies below every bin of the register.

    """
    if not math.isfinite(target_energy):
        raise ValueError(f"target_energy must be a finite energy. Got {target_energy}.")

    phase_bin_count = 1 << num_phase_qubits
    ranges: list[tuple[int, int]] = []
    for phase_bin in range(phase_bin_count):
        if eigenvalue_from_phase(phase_bin / phase_bin_count) > target_energy:
            continue
        if ranges and ranges[-1][1] == phase_bin:
            ranges[-1] = (ranges[-1][0], phase_bin + 1)
        else:
            ranges.append((phase_bin, phase_bin + 1))

    if not ranges:
        raise ValueError(
            f"No phase bin of the {phase_bin_count}-bin register holds an energy at most {target_energy}."
        )
    return ranges


class QPESubspaceMarkingSettings(QpeCircuitBuilderSettings):
    r"""Settings for the QPE subspace marking oracle: a QPE circuit builder's, plus the target energy."""

    def __init__(self):
        """Initialize the settings for the QPE subspace marking oracle."""
        super().__init__()
        self._set_default(
            "target_energy",
            "double",
            math.nan,
            "Highest energy the marked subspace may hold. Required; there is no meaningful default.",
        )


class QPESubspaceMarking(StandardQpeCircuitBuilder):
    r"""Build a good state oracle that flags the eigenspace below a target energy.

    Configured like a standard QPE circuit builder, plus the energy to mark, but instead of a
    QPE circuit it returns an oracle: the QPE runs on the register handed to it, a flag qubit
    is flipped when the phase register lands in a bin whose energy is at most
    ``target_energy``, then the QPE is undone. The register comes back unchanged, so the
    oracle serves as the ``good_state_oracle`` of
    :class:`~qdk_chemistry.algorithms.amplitude_amplification.amplitude_amplification.AmplitudeAmplification`
    next to a plain state preparation as its ``state_prep_oracle``.

    Set ``target_energy`` between the eigenvalue to amplify and the next one up. The bins are
    found by reading
    :meth:`~qdk_chemistry.data.unitary_representation.containers.base.UnitaryContainer.eigenvalue_from_phase`,
    the post-processing equation QPE results are read with, off each bin of the register, so
    any unitary the builder can estimate is handled without special casing.
    """

    def __init__(
        self,
        num_bits: int = -1,
        unitary_builder: AlgorithmRef | None = None,
        controlled_circuit_mapper: AlgorithmRef | None = None,
        target_energy: float = math.nan,
    ):
        """Initialize the QPE subspace marking oracle.

        Args:
            num_bits: The number of phase bits the marked QPE estimates.
            unitary_builder: Optional algorithm reference for the unitary builder.
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.
            target_energy: Energy whose phase bins are marked.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = QPESubspaceMarkingSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("target_energy", target_energy)
        if unitary_builder is not None:
            self._settings.set("unitary_builder", unitary_builder)
        if controlled_circuit_mapper is not None:
            self._settings.set("controlled_circuit_mapper", controlled_circuit_mapper)

    def name(self) -> str:
        """Return the algorithm name as qdk_qpe_subspace."""
        return "qdk_qpe_subspace"

    def _run_impl(  # type: ignore[override]
        self,
        qubit_hamiltonian: QubitOperator,
    ) -> Circuit:
        r"""Build the good state oracle for the QPE of ``qubit_hamiltonian``.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian whose eigenspace is marked.

        Returns:
            A circuit for use as the ``good_state_oracle`` of ``AmplitudeAmplification``.

        Raises:
            ValueError: If ``num_bits`` or ``target_energy`` is unset or invalid.
            RuntimeError: If the controlled unitaries do not carry Q# operations.

        """
        Logger.trace_entering()
        num_bits = self._settings.get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")
        target_energy = float(self._settings.get("target_energy"))
        if math.isnan(target_energy):
            raise ValueError("The target_energy setting must be set to the energy of the subspace to mark.")

        num_system_qubits = qubit_hamiltonian.num_qubits
        controlled_unitaries = []
        num_ancilla_qubits = 0
        for bit in range(num_bits):
            power = 2 ** (num_bits - 1 - bit)
            circuit, num_ancilla_qubits = self._create_controlled_circuit(qubit_hamiltonian, power=power)
            controlled_unitaries.append(circuit._qsharp_op)  # noqa: SLF001

        if any(operation is None for operation in controlled_unitaries):
            raise RuntimeError("Failed to create the subspace oracle: Q# operations are not available.")

        # The register handed to the oracle already holds the state under test, so the QPE
        # inside it prepares nothing of its own.
        qpe_operation = QSHARP_UTILS.StandardPhaseEstimation.MakeStandardQPEOp(
            QSHARP_UTILS.StatePreparation.MakePrepareNothingOp(),
            controlled_unitaries,
            num_bits,
            list(range(num_bits)),
            [index + num_bits for index in range(num_system_qubits)],
            QSHARP_UTILS.StatePreparation.MakePrepareHadamardAllOp(),
            num_ancilla_qubits,
        )

        container = self._create_nested("unitary_builder").run(qubit_hamiltonian).get_container()
        bin_ranges = _marked_phase_bins(target_energy, container.eigenvalue_from_phase, num_bits)
        lower_bounds = [start for start, _ in bin_ranges]
        upper_bounds = [stop for _, stop in bin_ranges]
        Logger.info(f"Marking phase bins {bin_ranges} for energy {target_energy}.")

        amplification = QSHARP_UTILS.AmplitudeAmplification
        parameters = {
            "qpe": qpe_operation,
            "numPhaseQubits": num_bits,
            "numSignalAncillas": num_ancilla_qubits,
            "lowerBounds": lower_bounds,
            "upperBounds": upper_bounds,
            "numSystemQubits": num_system_qubits,
        }
        return Circuit(
            qsharp_factory=QsharpFactoryData(program=amplification.MakeMarkedPhaseCircuit, parameter=parameters),
            qsharp_op=amplification.MarkQPEPhaseOp(
                qpe_operation, num_bits, num_ancilla_qubits, lower_bounds, upper_bounds
            ),
        )
