"""Iterative phase estimation circuit builder.

This module implements the circuit-building component of the Kitaev-style iterative
quantum phase estimation (IQPE) algorithm. It constructs the iteration circuits
without executing them, enabling standalone resource estimation and circuit preview.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import IterativeQpeCircuitBuilder, QpeCircuitBuilderSettings

__all__: list[str] = [
    "QdkIterativeQpeCircuitBuilder",
    "QdkIterativeQpeCircuitBuilderSettings",
    "_validate_iteration_inputs",
]


def _require_adaptive_profile() -> None:
    """Ensure the active Q# target profile supports mid-circuit measurement and classical control.

    The single-circuit IQPE relies on in-circuit classical feedback, which does not compile
    under the Base profile. This raises a clear error instead of surfacing a cryptic Q# compile
    failure when the profile is Base.

    Raises:
        RuntimeError: If the active Q# target profile is Base.

    """
    try:
        from qdk._interpreter import get_config  # noqa: PLC0415
    except ImportError:
        from qsharp._qsharp import get_config  # noqa: PLC0415

    profile = get_config().get_target_profile().lower()
    if profile == "base":
        raise RuntimeError(
            "Single-circuit IQPE requires a Q# target profile that supports mid-circuit measurement "
            "and classical feedback (e.g. Adaptive_RI), but the active profile is 'Base'. "
            "Set an adaptive profile before importing qdk_chemistry, e.g. "
            "`import qsharp; qsharp.init(target_profile=qsharp.TargetProfile.Adaptive_RI)`, "
            "or use the default per-bit IQPE (single_circuit=False)."
        )


class QdkIterativeQpeCircuitBuilderSettings(QpeCircuitBuilderSettings):
    """Settings for the Iterative Phase Estimation Builder."""

    def __init__(self):
        """Initialize the settings for the Iterative Phase Estimation Builder."""
        super().__init__()
        self._set_default("phase_correction", "double", 0.0, "The accumulated phase feedback from prior iterations.")
        self._set_default(
            "num_iteration", "int", -1, "The specific iteration to build. Default to -1 to build all iterations."
        )
        self._set_default(
            "single_circuit",
            "bool",
            False,
            "Build the full IQPE as one circuit with in-circuit classical feedback (needs an Adaptive target).",
        )


class QdkIterativeQpeCircuitBuilder(IterativeQpeCircuitBuilder):
    """Iterative Phase Estimation circuit builder.

    Constructs the quantum circuits for each IQPE iteration without executing them.
    Can be used standalone for resource estimation or composed inside IterativePhaseEstimation.

    """

    def __init__(
        self,
        num_bits: int = -1,
        phase_correction: float = 0.0,
        num_iteration: int = -1,
        single_circuit: bool = False,
        unitary_builder: AlgorithmRef | None = None,
        controlled_circuit_mapper: AlgorithmRef | None = None,
    ):
        """Initialize the IterativeQpeCircuitBuilder.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            phase_correction: The accumulated phase feedback from prior iterations. Default to 0.0.
            num_iteration: The specific iteration to build. Default to -1 (build all iterations).
            single_circuit: Build the full IQPE as one circuit with in-circuit classical feedback. Default to False.
            unitary_builder: Optional algorithm reference for the unitary builder.
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = QdkIterativeQpeCircuitBuilderSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("phase_correction", phase_correction)
        self._settings.set("num_iteration", num_iteration)
        self._settings.set("single_circuit", single_circuit)
        if unitary_builder is not None:
            self._settings.set("unitary_builder", unitary_builder)
        if controlled_circuit_mapper is not None:
            self._settings.set("controlled_circuit_mapper", controlled_circuit_mapper)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        """Build IQPE iteration circuits.

        Uses settings ``phase_correction`` (default 0.0) and ``num_iteration``
        (default -1). When ``num_iteration`` is negative, all iteration circuits
        are returned. When positive, only the circuit for that single iteration
        (0-based) is returned. When ``single_circuit`` is True, a single circuit
        implementing the full IQPE with in-circuit classical feedback is returned
        (``phase_correction`` and ``num_iteration`` are ignored in that mode).

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build circuits.

        Returns:
            A list of quantum circuits, one per phase bit iteration (or a single-element
            list when ``num_iteration`` is set to a specific iteration index, or when
            ``single_circuit`` is enabled).

        Raises:
            ValueError: If ``num_iteration`` >= ``num_bits``.

        """
        num_bits = self.settings().get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")

        if self.settings().get("single_circuit"):
            circuit = self._create_full_circuit(
                state_preparation=state_preparation,
                qubit_hamiltonian=qubit_hamiltonian,
                num_bits=num_bits,
            )
            Logger.info("Built single full IQPE circuit with in-circuit classical feedback.")
            return [circuit]

        phase_correction = self.settings().get("phase_correction")
        num_iteration = self.settings().get("num_iteration")

        if num_iteration >= num_bits:
            raise ValueError(f"num_iteration ({num_iteration}) must be less than num_bits ({num_bits}).")

        iterations = [num_iteration] if num_iteration >= 0 else range(num_bits)
        circuits: list[Circuit] = []
        for iteration in iterations:
            circuit = self._create_iteration_circuit(
                state_preparation=state_preparation,
                qubit_hamiltonian=qubit_hamiltonian,
                iteration=iteration,
                total_iterations=num_bits,
                phase_correction=phase_correction,
            )
            circuits.append(circuit)

        Logger.info(f"Built {len(circuits)} iteration circuit(s) with phase_correction={phase_correction}.")
        return circuits

    def _create_iteration_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
        *,
        iteration: int,
        total_iterations: int,
        phase_correction: float = 0.0,
    ) -> Circuit:
        """Construct a single IQPE iteration circuit.

        Args:
            state_preparation: Trial-state preparation circuit that prepares the initial state on the system qubits.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            iteration: Current iteration index (0-based), where 0 corresponds to the most-significant bit.
            total_iterations: Total number of phase bits to measure across all iterations.
            phase_correction: Feedback phase angle to apply before controlled unitary, defaults to 0.0.

        Returns:
            A quantum circuit implementing one IQPE iteration.

        """
        _validate_iteration_inputs(iteration, total_iterations)
        num_system_qubits = qubit_hamiltonian.num_qubits
        power = 2 ** (total_iterations - iteration - 1)

        ctrl_unitary_circuit, num_ancilla_qubits = self._create_controlled_circuit(qubit_hamiltonian, power)

        if state_preparation._qsharp_op and ctrl_unitary_circuit._qsharp_op:  # noqa: SLF001
            return self._create_circuit_from_qsharp_op(
                state_preparation, ctrl_unitary_circuit, phase_correction, num_system_qubits, num_ancilla_qubits
            )

        raise RuntimeError(
            "Failed to create iteration circuit: Q# operations are not available. "
            "For Qiskit support, use QiskitIterativeQpeCircuitBuilder from the qiskit plugin."
        )

    def _create_circuit_from_qsharp_op(
        self,
        state_preparation: Circuit,
        controlled_unitary_circuit: Circuit,
        phase_correction: float,
        num_system_qubits: int,
        num_ancilla_qubits: int = 0,
    ) -> Circuit:
        """Create a Circuit object from a Q# operation.

        Args:
            state_preparation: Circuit object containing a Q# operation for state preparation.
            controlled_unitary_circuit: Circuit object containing a Q# operation for the controlled unitary.
            phase_correction: Feedback phase angle to apply before controlled unitary.
            num_system_qubits: Number of system qubits.
            num_ancilla_qubits: Number of ancilla qubits within the unitary (0 for Trotter).

        Returns:
            A Circuit object representing the IQPE iteration.

        """
        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        ctrl_unitary_op = controlled_unitary_circuit._qsharp_op  # noqa: SLF001
        iterative_parameters = {
            "statePrep": state_prep_op,
            "repControlledUnitary": ctrl_unitary_op,
            "accumulatePhase": phase_correction,
            "phaseQubit": 0,
            "systems": [i + 1 for i in range(num_system_qubits)],
            "numAncillaQubits": num_ancilla_qubits,
        }
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.IterativePhaseEstimation.MakeIQPECircuit,
                parameter=iterative_parameters,
            )
        )

    def _create_full_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
        *,
        num_bits: int,
    ) -> Circuit:
        """Construct a single circuit implementing the full IQPE with in-circuit feedback.

        Realizes ``controlled-U^(2^k)`` by repeating the power-1 controlled unitary,
        and uses mid-circuit measurement with classical feed-forward to apply the phase
        correction on device. The resulting circuit requires an Adaptive-profile target.

        Args:
            state_preparation: Trial-state preparation circuit that prepares the initial state on the system qubits.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            num_bits: Total number of phase bits to measure.

        Returns:
            A quantum circuit implementing the full IQPE run.

        Raises:
            RuntimeError: If the required Q# operations are not available, or if the active Q# target profile is Base.

        """
        num_system_qubits = qubit_hamiltonian.num_qubits
        ctrl_unitary_circuit, num_ancilla_qubits = self._create_controlled_circuit(qubit_hamiltonian, 1)

        if not (state_preparation._qsharp_op and ctrl_unitary_circuit._qsharp_op):  # noqa: SLF001
            raise RuntimeError(
                "Failed to create full IQPE circuit: Q# operations are not available. "
                "For Qiskit support, use QiskitIterativeQpeCircuitBuilder from the qiskit plugin."
            )

        _require_adaptive_profile()

        iterative_parameters = {
            "numBits": num_bits,
            "statePrep": state_preparation._qsharp_op,  # noqa: SLF001
            "repControlledUnitary": ctrl_unitary_circuit._qsharp_op,  # noqa: SLF001
            "phaseQubit": 0,
            "systems": [i + 1 for i in range(num_system_qubits)],
            "numAncillaQubits": num_ancilla_qubits,
        }
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.IterativePhaseEstimation.RunFullIQPE,
                parameter=iterative_parameters,
            )
        )

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "qdk_iterative"


def _validate_iteration_inputs(iteration: int, total_iterations: int) -> None:
    """Validate iteration parameters for IQPE circuit construction.

    Args:
        iteration: The current iteration index (0-based).
        total_iterations: The total number of iterations.

    """
    if total_iterations <= 0:
        raise ValueError("total_iterations must be a positive integer.")
    if iteration < 0 or iteration >= total_iterations:
        raise ValueError(
            f"iteration index {iteration} is outside the valid range [0, {total_iterations - 1}].",
        )
