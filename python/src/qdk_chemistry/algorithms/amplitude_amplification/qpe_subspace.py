r"""QDK/Chemistry good state oracle marking a QPE phase register."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import math

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory, create_from_ref
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.base import StandardQpeCircuitBuilder
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator, Settings
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.data.unitary_representation.containers.base import UnitaryContainer
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

__all__: list[str] = [
    "AmplitudeAmplificationOracleFactory",
    "QPESubspaceMarking",
    "QPESubspaceMarkingSettings",
]


class QPESubspaceMarkingSettings(Settings):
    r"""Settings for the QPE subspace marking oracle."""

    def __init__(self):
        """Initialize the settings for the QPE subspace marking oracle."""
        super().__init__()
        self._set_default(
            "energy_lower_bound",
            "double",
            math.nan,
            "Lowest energy the marked subspace may hold. Required.",
        )
        self._set_default(
            "qpe_circuit_builder",
            "algorithm_ref",
            AlgorithmRef("qpe_circuit_builder", "qdk_standard"),
        )


class QPESubspaceMarking(Algorithm):
    r"""Build a good state oracle that flags the eigenspace above an energy bound.

    The nested ``qpe_circuit_builder`` estimates the phase of the register handed to the
    oracle, a flag is flipped when that phase lands in a bin whose energy is at least
    ``energy_lower_bound``, and the estimation is undone. No state preparation is used:
    the register already holds the state under test.

    This reflects about the marked eigenspaces only when the estimation is exact, that is
    when every eigenphase of that state is a multiple of :math:`2^{-n}` for the builder's
    ``num_bits`` :math:`n`. Off a bin the phase register comes back spread rather than to
    :math:`|0\rangle`, leaking part of the state into the ancillas the oracle releases.
    """

    def __init__(
        self,
        energy_lower_bound: float = math.nan,
        qpe_circuit_builder: AlgorithmRef | None = None,
    ):
        """Initialize the QPE subspace marking oracle.

        Args:
            energy_lower_bound: Lowest energy the marked subspace may hold.
            qpe_circuit_builder: Optional algorithm reference for the phase estimation to mark.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = QPESubspaceMarkingSettings()
        self._settings.set("energy_lower_bound", energy_lower_bound)
        if qpe_circuit_builder is not None:
            self._settings.set("qpe_circuit_builder", qpe_circuit_builder)

    def type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification_oracle."""
        return "amplitude_amplification_oracle"

    def name(self) -> str:
        """Return the algorithm name as qdk_qpe_subspace."""
        return "qdk_qpe_subspace"

    @staticmethod
    def _marked_phase_bins(
        energy_lower_bound: float,
        container: UnitaryContainer,
        num_phase_qubits: int,
    ) -> list[tuple[int, int]]:
        r"""Return the half-open phase-bin ranges whose energy is at least ``energy_lower_bound``.

        Every bin is tested against the encoding's phase-to-energy law, so the answer follows
        that law wherever it turns or wraps.

        Args:
            energy_lower_bound: The lowest energy the marked subspace may hold.
            container: The unitary encoding whose phases are being marked.
            num_phase_qubits: Width of the QPE phase register.

        Returns:
            Sorted disjoint half-open bin ranges.

        Raises:
            ValueError: If no bin reaches the bound, or if every bin does, because neither
                names a proper subspace to amplify.

        """
        phase_bin_count = 1 << num_phase_qubits
        ranges: list[tuple[int, int]] = []
        for phase_bin in range(phase_bin_count):
            if container.eigenvalue_from_phase(phase_bin / phase_bin_count) < energy_lower_bound:
                continue
            if ranges and ranges[-1][1] == phase_bin:
                ranges[-1] = (ranges[-1][0], phase_bin + 1)
            else:
                ranges.append((phase_bin, phase_bin + 1))
        if not ranges:
            raise ValueError(
                f"No phase bin of the {phase_bin_count}-bin register holds an energy at least {energy_lower_bound}."
            )
        if ranges == [(0, phase_bin_count)]:
            raise ValueError(
                f"Every phase bin of the {phase_bin_count}-bin register holds an energy at least "
                f"{energy_lower_bound}, so it marks no subspace to amplify. Give a bound inside "
                f"the range the encoding represents."
            )
        return ranges

    def _run_impl(self, qubit_hamiltonian: QubitOperator) -> Circuit:
        r"""Build the oracle flagging the eigenspace of ``qubit_hamiltonian`` above the bound.

        Args:
            qubit_hamiltonian: The qubit Hamiltonian whose eigenspace is marked.

        Returns:
            The circuit to use as the ``good_state_oracle`` of ``AmplitudeAmplification``.

        Raises:
            TypeError: If the nested ``qpe_circuit_builder`` is not a standard (QFT-based) one,
                the only kind whose circuit can be undone.
            ValueError: If its ``num_bits`` is not positive, if ``energy_lower_bound`` is unset
                or not finite, or if it names no proper subspace of the phase register.
            RuntimeError: If the phase estimation does not carry a Q# operation.

        """
        Logger.trace_entering()
        energy_lower_bound = float(self._settings.get("energy_lower_bound"))
        if math.isnan(energy_lower_bound):
            raise ValueError("The energy_lower_bound setting must be set to the lowest energy of the subspace to mark.")
        if math.isinf(energy_lower_bound):
            raise ValueError(f"The energy_lower_bound setting must be a finite energy. Got {energy_lower_bound}.")

        builder = self._create_nested("qpe_circuit_builder")
        if not isinstance(builder, StandardQpeCircuitBuilder):
            raise TypeError(f"qpe_circuit_builder must be a standard (QFT-based) builder. Got {builder.name()}.")
        num_bits = builder.settings().get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"The nested qpe_circuit_builder needs a positive num_bits. Got {num_bits}.")

        num_system_qubits = qubit_hamiltonian.num_qubits
        unitary = create_from_ref(builder.settings(), "unitary_builder").run(qubit_hamiltonian)
        num_signal_ancillas = unitary.get_num_qubits() - num_system_qubits
        bin_ranges = self._marked_phase_bins(energy_lower_bound, unitary.get_container(), num_bits)
        lower_bounds = [start for start, _ in bin_ranges]
        upper_bounds = [stop for _, stop in bin_ranges]
        Logger.info(f"Marking phase bins {bin_ranges} for energies at least {energy_lower_bound}.")

        state_prep = QSHARP_UTILS.StatePreparation
        prep_params = state_prep.SingleReferenceParams(bitStrings=[0] * num_system_qubits, numQubits=num_system_qubits)
        prepare_nothing = Circuit(
            qsharp_factory=QsharpFactoryData(
                program=state_prep.MakeSingleReferenceStateCircuit, parameter=vars(prep_params)
            ),
            qsharp_op=state_prep.MakePrepareSingleReferenceStateOp(prep_params),
        )
        qpe_circuit = builder.run(prepare_nothing, qubit_hamiltonian)[0]
        qpe_operation = qpe_circuit._qsharp_op  # noqa: SLF001
        if qpe_operation is None:
            raise RuntimeError("Failed to create the subspace oracle: the Q# phase estimation is not available.")

        amplification = QSHARP_UTILS.AmplitudeAmplification
        parameters = {
            "qpe": qpe_operation,
            "numPhaseQubits": num_bits,
            "numSignalAncillas": num_signal_ancillas,
            "lowerBounds": lower_bounds,
            "upperBounds": upper_bounds,
            "numSystemQubits": num_system_qubits,
        }
        return Circuit(
            qsharp_factory=QsharpFactoryData(program=amplification.MakeMarkedPhaseCircuit, parameter=parameters),
            qsharp_op=amplification.MarkQPEPhaseOp(
                qpe_operation, num_bits, num_signal_ancillas, lower_bounds, upper_bounds
            ),
        )


class AmplitudeAmplificationOracleFactory(AlgorithmFactory):
    """Factory class for creating amplitude amplification good state oracle instances."""

    def __init__(self):
        """Initialize the AmplitudeAmplificationOracleFactory."""
        super().__init__()

    def algorithm_type_name(self) -> str:
        """Return the algorithm type name as amplitude_amplification_oracle."""
        return "amplitude_amplification_oracle"

    def default_algorithm_name(self) -> str:
        """Return qdk_qpe_subspace as the default algorithm name."""
        return "qdk_qpe_subspace"
