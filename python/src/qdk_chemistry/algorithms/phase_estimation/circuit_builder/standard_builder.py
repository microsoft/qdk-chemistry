"""Standard (QFT-based) phase estimation circuit builder.

This module implements the circuit-building component of the standard quantum phase
estimation (QPE) algorithm. It constructs a single circuit that uses multiple ancilla
qubits and the inverse QFT, enabling standalone resource estimation and circuit preview.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from typing import Any

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
        qubit_hamiltonian: QubitHamiltonian | Any = None,
        *,
        factorized_hamiltonian: Any = None,
    ) -> list[Circuit]:
        """Build the standard QPE circuit.

        Constructs a single circuit with ``num_bits`` ancilla qubits, applying
        controlled-U^(2^k) for each ancilla and finishing with the inverse QFT.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build the circuit.
            factorized_hamiltonian: A FactorizedHamiltonianContainer for SOSSA-based QPE.
                Mutually exclusive with qubit_hamiltonian.

        Returns:
            A single-element list containing the standard QPE circuit.

        Raises:
            ValueError: If ``num_bits`` is not a positive integer.

        """
        hamiltonian = factorized_hamiltonian if factorized_hamiltonian is not None else qubit_hamiltonian
        if hamiltonian is None:
            raise ValueError("Either qubit_hamiltonian or factorized_hamiltonian must be provided.")

        num_bits = self.settings().get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")

        # Determine system and block-encoding ancilla qubit counts.
        # - num_bare_system_qubits: qubits where state_prep is applied
        # - num_be_ancillas: extra ancillas needed by block encoding (0 for LCU/Trotter)
        if isinstance(hamiltonian, QubitHamiltonian):
            num_bare_system_qubits = hamiltonian.num_qubits
            num_be_ancillas = 0
        else:
            # FactorizedHamiltonianContainer (SOSSA): bare system = 2*N spin-orbitals
            num_bare_system_qubits = 2 * hamiltonian.get_num_orbitals()
            # Total walk operator qubits from unitary representation
            probe_builder = self._create_nested("unitary_builder")
            probe_builder.settings().update("power", 1)
            probe_rep = probe_builder.run(hamiltonian)
            num_be_ancillas = probe_rep.get_num_qubits() - num_bare_system_qubits

        # Build one controlled circuit per ancilla with power=2^k,
        # respecting the unitary builder's power_strategy (e.g. "rescale").
        # ancillas[0] = MSB controls U^(2^(n-1)), ancillas[n-1] = LSB controls U^1.
        ctrl_unitary_circuits = []
        for k in range(num_bits):
            power = 2 ** (num_bits - 1 - k)
            ctrl_unitary_circuits.append(self._create_controlled_circuit(hamiltonian, power=power))

        if state_preparation._qsharp_op and all(c._qsharp_op for c in ctrl_unitary_circuits):  # noqa: SLF001
            circuit = self._create_circuit_from_qsharp_op(
                state_preparation, ctrl_unitary_circuits, num_bits, num_bare_system_qubits, num_be_ancillas
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
        controlled_unitary_circuits: list[Circuit],
        num_bits: int,
        num_system_qubits: int,
        num_be_ancillas: int = 0,
    ) -> Circuit:
        """Create a Circuit object from a Q# operation using MakeStandardQPECircuit.

        Args:
            state_preparation: Circuit object containing a Q# operation for state preparation.
            controlled_unitary_circuits: List of Circuit objects (one per ancilla) containing
                Q# operations for controlled-U^(2^k).
            num_bits: Number of phase ancilla qubits (phase bits).
            num_system_qubits: Number of bare system qubits (state prep target).
            num_be_ancillas: Number of block-encoding ancilla qubits (0 for LCU/Trotter).

        Returns:
            A Circuit object representing the standard QPE circuit.

        """
        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        ctrl_unitary_ops = [c._qsharp_op for c in controlled_unitary_circuits]  # noqa: SLF001
        phase_qubit_prep_op = QSHARP_UTILS.StatePreparation.MakePrepareHadamardAllOp()
        ancillas = list(range(num_bits))
        systems = [i + num_bits for i in range(num_system_qubits)]
        standard_parameters = {
            "statePrep": state_prep_op,
            "controlledEvolutions": ctrl_unitary_ops,
            "numBits": num_bits,
            "ancillas": ancillas,
            "systems": systems,
            "phaseQubitPrep": phase_qubit_prep_op,
            "numBlockEncodingAncillas": num_be_ancillas,
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
