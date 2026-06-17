"""Fast resource estimation circuit builder for standard QPE.

This module builds a circuit optimized for the Q# resource estimator using
``RepeatEstimates``. Instead of tracing through all 2^numBits - 1 controlled
Trotter steps individually, it wraps a single base step and tells the estimator
to multiply the cost. This is dramatically faster for large QPE circuits.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.data import AlgorithmRef, Circuit, QubitHamiltonian
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import QpeCircuitBuilderSettings, StandardQpeCircuitBuilder

__all__: list[str] = [
    "QdkEstimateQpeCircuitBuilder",
    "QdkEstimateQpeCircuitBuilderSettings",
]


class QdkEstimateQpeCircuitBuilderSettings(QpeCircuitBuilderSettings):
    """Settings for the Estimate QPE Circuit Builder."""

    def __init__(self):
        """Initialize the settings for the Estimate QPE Circuit Builder."""
        super().__init__()


class QdkEstimateQpeCircuitBuilder(StandardQpeCircuitBuilder):
    """Fast resource estimation circuit builder for standard QPE.

    Uses ``RepeatEstimates`` to tell the Q# resource estimator to analyze a single
    controlled Trotter step and multiply costs by the total number of queries
    (2^numBits - 1), rather than tracing through each step individually.

    This produces the same resource estimates as the full QPE circuit but runs
    orders of magnitude faster for large circuits.

    """

    def __init__(
        self,
        num_bits: int = -1,
        unitary_builder: AlgorithmRef | None = None,
        controlled_circuit_mapper: AlgorithmRef | None = None,
    ):
        """Initialize the QdkEstimateQpeCircuitBuilder.

        Args:
            num_bits: The number of phase bits (ancilla qubits) to estimate. Default to -1;
                        user needs to set a valid value.
            unitary_builder: Optional algorithm reference for the unitary builder.
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = QdkEstimateQpeCircuitBuilderSettings()
        self._settings.set("num_bits", num_bits)
        if unitary_builder is not None:
            self._settings.set("unitary_builder", unitary_builder)
        if controlled_circuit_mapper is not None:
            self._settings.set("controlled_circuit_mapper", controlled_circuit_mapper)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[Circuit]:
        """Build an estimation-optimized QPE circuit using RepeatEstimates.

        Only builds a single controlled-U (power=1) circuit, then wraps it with
        RepeatEstimates(2^numBits - 1) so the resource estimator multiplies costs.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build the circuit.

        Returns:
            A single-element list containing the estimation-optimized QPE circuit.

        Raises:
            ValueError: If ``num_bits`` is not a positive integer.

        """
        num_bits = self.settings().get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")

        num_system_qubits = qubit_hamiltonian.num_qubits

        # Build only the base controlled circuit (power=1)
        base_ctrl_circuit = self._create_controlled_circuit(qubit_hamiltonian, power=1)

        if state_preparation._qsharp_op and base_ctrl_circuit._qsharp_op:  # noqa: SLF001
            circuit = self._create_estimate_circuit(state_preparation, base_ctrl_circuit, num_bits, num_system_qubits)
            Logger.info(f"Built fast estimation QPE circuit with {num_bits} ancilla qubits.")
            return [circuit]

        raise RuntimeError("Failed to create estimate QPE circuit: Q# operations are not available.")

    def _create_estimate_circuit(
        self,
        state_preparation: Circuit,
        base_controlled_circuit: Circuit,
        num_bits: int,
        num_system_qubits: int,
    ) -> Circuit:
        """Create a Circuit using EstimateStandardQPE with RepeatEstimates.

        Args:
            state_preparation: Circuit object containing a Q# operation for state preparation.
            base_controlled_circuit: Circuit for controlled-U (power=1).
            num_bits: Number of ancilla qubits (phase bits).
            num_system_qubits: Number of system qubits.

        Returns:
            A Circuit object optimized for fast resource estimation.

        """
        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        single_ctrl_evo_op = base_controlled_circuit._qsharp_op  # noqa: SLF001

        estimate_parameters = {
            "singleControlledEvolution": single_ctrl_evo_op,
            "statePrep": state_prep_op,
            "numBits": num_bits,
            "numSystemQubits": num_system_qubits,
        }

        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StandardPhaseEstimation.EstimateStandardQPE,
                parameter=estimate_parameters,
            )
        )

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "qdk_standard_estimate"
