"""Standard (QFT-based) phase estimation circuit builder.

This module implements the circuit-building component of the standard quantum phase
estimation (QPE) algorithm. It constructs a single circuit that uses multiple ancilla
qubits and the inverse QFT, enabling standalone resource estimation and circuit preview.

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
    "QdkStandardQpeCircuitBuilder",
    "QdkStandardQpeCircuitBuilderSettings",
]


class QdkStandardQpeCircuitBuilderSettings(QpeCircuitBuilderSettings):
    """Settings for the Standard Phase Estimation Circuit Builder."""

    def __init__(self):
        """Initialize the settings for the Standard Phase Estimation Circuit Builder."""
        super().__init__()


class QdkStandardQpeCircuitBuilder(StandardQpeCircuitBuilder):
    """Standard (QFT-based) Phase Estimation circuit builder.

    Constructs a single quantum circuit that performs standard QPE using multiple
    ancilla qubits and the inverse QFT. Can be used standalone for resource estimation
    or composed inside StandardPhaseEstimation.

    """

    def __init__(
        self,
        num_bits: int = -1,
        unitary_builder: AlgorithmRef | None = None,
        controlled_circuit_mapper: AlgorithmRef | None = None,
    ):
        """Initialize the StandardQpeCircuitBuilder.

        Args:
            num_bits: The number of phase bits (ancilla qubits) to estimate. Default to -1;
                        user needs to set a valid value.
            unitary_builder: Optional algorithm reference for the unitary builder.
            controlled_circuit_mapper: Optional algorithm reference for the controlled circuit mapper.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = QdkStandardQpeCircuitBuilderSettings()
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
        """Build the standard QPE circuit.

        Constructs a single circuit with ``num_bits`` ancilla qubits, applying
        controlled-U^(2^k) for each ancilla and finishing with the inverse QFT.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build the circuit.

        Returns:
            A single-element list containing the standard QPE circuit.

        Raises:
            ValueError: If ``num_bits`` is not a positive integer.

        """
        num_bits = self.settings().get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")

        num_system_qubits = qubit_hamiltonian.num_qubits
        ctrl_unitary_circuit = self._create_controlled_circuit(qubit_hamiltonian, power=1)

        if state_preparation._qsharp_op and ctrl_unitary_circuit._qsharp_op:  # noqa: SLF001
            circuit = self._create_circuit_from_qsharp_op(
                state_preparation, ctrl_unitary_circuit, num_bits, num_system_qubits
            )
            Logger.info(f"Built standard QPE circuit with {num_bits} ancilla qubits.")
            return [circuit]

        raise RuntimeError(
            "Failed to create standard QPE circuit: Q# operations are not available. "
            "For Qiskit support, use QiskitStandardQpeCircuitBuilder from the qiskit plugin."
        )

    def _create_circuit_from_qsharp_op(
        self,
        state_preparation: Circuit,
        controlled_unitary_circuit: Circuit,
        num_bits: int,
        num_system_qubits: int,
    ) -> Circuit:
        """Create a Circuit object from a Q# operation using MakeStandardQPECircuit.

        Args:
            state_preparation: Circuit object containing a Q# operation for state preparation.
            controlled_unitary_circuit: Circuit object containing a Q# operation for the controlled unitary.
            num_bits: Number of ancilla qubits (phase bits).
            num_system_qubits: Number of system qubits.

        Returns:
            A Circuit object representing the standard QPE circuit.

        """
        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        ctrl_unitary_op = controlled_unitary_circuit._qsharp_op  # noqa: SLF001
        ancillas = list(range(num_bits))
        systems = [i + num_bits for i in range(num_system_qubits)]
        standard_parameters = {
            "statePrep": state_prep_op,
            "controlledEvolution": ctrl_unitary_op,
            "numBits": num_bits,
            "ancillas": ancillas,
            "systems": systems,
        }
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StandardPhaseEstimation.MakeStandardQPECircuit,
                parameter=standard_parameters,
            )
        )

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "qdk_standard"
