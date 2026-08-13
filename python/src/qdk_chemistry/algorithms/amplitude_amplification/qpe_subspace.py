r"""QDK/Chemistry subspace oracles marking a QPE phase register."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import (
    QpeCircuitBuilder,
    QpeCircuitBuilderSettings,
)
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.base import UnitaryContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "QPESubspaceMarking",
    "QPESubspaceMarkingSettings",
]


class QPESubspaceMarkingSettings(QpeCircuitBuilderSettings):
    r"""Settings for the QPE subspace marking oracle: a QPE circuit builder's, plus the target energy."""

    def __init__(self):
        """Initialize the settings for the QPE subspace marking oracle."""
        super().__init__()
        self._set_default(
            "target_energy",
            "double",
            math.nan,
            "Lowest energy the marked subspace may hold. Required.",
        )


class QPESubspaceMarking(QpeCircuitBuilder):
    r"""Build a good state oracle that flags the eigenspace above a target energy.

    Configured like a standard QPE circuit builder, plus the energy to mark, but instead of a
    QPE circuit it returns an oracle: the QPE runs on the register handed to it, a flag qubit
    is flipped when the phase register lands in a bin whose energy is at least
    ``target_energy``, then the QPE is undone.

    The initial state preparation is not used, because the register already holds the state to
    amplify.
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

    @staticmethod
    def _marked_phase_bins(
        target_energy: float,
        container: UnitaryContainer,
        num_phase_qubits: int,
    ) -> list[tuple[int, int]]:
        r"""Return the half-open phase-bin ranges whose energy is at least ``target_energy``.

        Args:
            target_energy: The lowest energy the marked subspace may hold.
            container: The unitary encoding whose phases are being marked.
            num_phase_qubits: Width :math:`n` of the QPE phase register.

        Returns:
            Sorted half-open bin ranges, separated by at least one bin so that no
            bin is marked twice.  Never empty.

        Raises:
            ValueError: If the encoding cannot represent the target energy, or if it
                can but no bin of the register reaches it.

        """
        phase_bin_count = 1 << num_phase_qubits
        ranges: list[tuple[int, int]] = []
        for phase_bin in range(phase_bin_count):
            if container.eigenvalue_from_phase(phase_bin / phase_bin_count) < target_energy:
                continue
            if ranges and ranges[-1][1] == phase_bin:
                ranges[-1] = (ranges[-1][0], phase_bin + 1)
            else:
                ranges.append((phase_bin, phase_bin + 1))
        if not ranges:
            raise ValueError(
                f"No phase bin of the {phase_bin_count}-bin register holds an energy at least {target_energy}."
            )
        # Marking every bin is as useless as marking none: the oracle would flag the whole
        # register, its reflection would be the identity up to a phase, and the amplification
        # would have nothing to single out. A target this low names no subspace, so it is
        # refused rather than answered with an oracle that cannot amplify.
        if ranges == [(0, phase_bin_count)]:
            raise ValueError(
                f"Every phase bin of the {phase_bin_count}-bin register holds an energy at least "
                f"{target_energy}, so it marks no subspace to amplify. Give a bound inside the "
                f"range the encoding represents."
            )
        return ranges

    def _run_impl(
        self,
        state_preparation: Circuit,  # noqa: ARG002
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        r"""Build the good state oracle for the QPE of ``qubit_hamiltonian``.

        Args:
            state_preparation: Unused; the register handed to the oracle already holds the
                state under test, so the QPE inside it prepares nothing of its own.
            qubit_hamiltonian: The qubit Hamiltonian whose eigenspace is marked.

        Returns:
            A single-element list holding the circuit to use as the ``good_state_oracle`` of
            ``AmplitudeAmplification``. The list is what the ``qpe_circuit_builder`` contract
            requires; unlike a phase estimation builder, this one always returns exactly one.

        Raises:
            ValueError: If ``num_bits`` is not positive, if ``target_energy`` is unset or not
                finite, or if it names no proper subspace of the phase register, because
                either no bin reaches it or every bin does.
            RuntimeError: If the controlled unitaries do not carry Q# operations.

        """
        Logger.trace_entering()
        num_bits = self._settings.get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")
        target_energy = float(self._settings.get("target_energy"))
        if math.isnan(target_energy):
            raise ValueError("The target_energy setting must be set to the energy of the subspace to mark.")
        if math.isinf(target_energy):
            raise ValueError(f"The target_energy setting must be a finite energy. Got {target_energy}.")

        num_system_qubits = qubit_hamiltonian.num_qubits
        container = self._create_nested("unitary_builder").run(qubit_hamiltonian).get_container()
        bin_ranges = self._marked_phase_bins(target_energy, container, num_bits)
        lower_bounds = [start for start, _ in bin_ranges]
        upper_bounds = [stop for _, stop in bin_ranges]
        Logger.info(f"Marking phase bins {bin_ranges} for energy {target_energy}.")

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

        amplification = QSHARP_UTILS.AmplitudeAmplification
        parameters = {
            "qpe": qpe_operation,
            "numPhaseQubits": num_bits,
            "numSignalAncillas": num_ancilla_qubits,
            "lowerBounds": lower_bounds,
            "upperBounds": upper_bounds,
            "numSystemQubits": num_system_qubits,
        }
        return [
            Circuit(
                qsharp_factory=QsharpFactoryData(program=amplification.MakeMarkedPhaseCircuit, parameter=parameters),
                qsharp_op=amplification.MarkQPEPhaseOp(
                    qpe_operation, num_bits, num_ancilla_qubits, lower_bounds, upper_bounds
                ),
            )
        ]
