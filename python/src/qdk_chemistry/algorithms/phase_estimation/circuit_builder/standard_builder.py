"""Standard (QFT-based) phase estimation circuit builder.

This module implements the circuit-building component of the standard quantum phase
estimation (QPE) algorithm. It constructs a single circuit that uses multiple ancilla
qubits and the inverse QFT, enabling standalone resource estimation and circuit preview.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk import qsharp

from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator
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
        self._set_default(
            "measurement",
            "string",
            "phase",
            "Final measurement: 'phase' measures the phase register in the computational "
            "basis, 'eigenvector' also measures the system register in 'measurement_basis', "
            "and 'none' measures nothing and returns a coherent, adjointable circuit.",
        )
        self._set_default(
            "measurement_basis",
            "string",
            "Z",
            "Pauli basis for the system register when measurement is 'eigenvector'. A single "
            "letter is broadcast to every system qubit; otherwise one letter per system qubit. "
            "'I' resets a qubit without recording a bit.",
        )


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
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        """Build the standard QPE circuit.

        Constructs a single circuit with ``num_bits`` ancilla qubits, applying
        controlled-U^(2^k) for each ancilla and finishing with the inverse QFT.

        The circuit body is always the measurement-free, adjointable QPE
        operation, exposed as the returned circuit's Q# operation; ``measurement``
        only decides what is read out.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build the circuit.

        Returns:
            A single-element list containing the standard QPE circuit.

        Raises:
            ValueError: If ``num_bits`` is not a positive integer.
            RuntimeError: If the inputs do not carry Q# operations.

        """
        num_bits = self.settings().get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")

        num_system_qubits = qubit_hamiltonian.num_qubits

        # Build one controlled circuit per ancilla with power=2^k,
        # respecting the unitary builder's power_strategy (e.g. "rescale").
        # ancillas[0] = MSB controls U^(2^(n-1)), ancillas[n-1] = LSB controls U^1.
        ctrl_unitary_circuits = []
        num_ancilla_qubits = 0
        for k in range(num_bits):
            power = 2 ** (num_bits - 1 - k)
            circuit, num_ancilla_qubits = self._create_controlled_circuit(qubit_hamiltonian, power=power)
            ctrl_unitary_circuits.append(circuit)

        if state_preparation._qsharp_op and all(c._qsharp_op for c in ctrl_unitary_circuits):  # noqa: SLF001
            circuit = self._create_circuit_from_qsharp_op(
                state_preparation,
                ctrl_unitary_circuits,
                num_bits,
                num_system_qubits,
                num_ancilla_qubits,
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
        num_ancilla_qubits: int = 0,
    ) -> Circuit:
        """Create a Circuit object from a Q# operation using MakeStandardQPECircuit.

        The register is laid out as ``phase ++ system ++ unitary ancillas`` with the
        phase register least-significant-bit first, matching
        ``QDKChemistry.Utils.StandardPhaseEstimation.ApplyStandardQPE``.

        Args:
            state_preparation: Circuit object containing an adjointable Q# state preparation.
            controlled_unitary_circuits: List of Circuit objects (one per phase bit) containing
                adjointable Q# operations for controlled-U^(2^k).
            num_bits: Number of phase qubits.
            num_system_qubits: Number of system qubits.
            num_ancilla_qubits: Number of extra ancilla qubits within the unitary (0 for Trotter).

        Returns:
            A Circuit whose Q# operation applies QPE in place and whose factory
            applies the measurement selected by the ``measurement`` setting.

        """
        phase_estimation = QSHARP_UTILS.StandardPhaseEstimation
        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        ctrl_unitary_ops = [c._qsharp_op for c in controlled_unitary_circuits]  # noqa: SLF001
        phase_qubit_prep_op = QSHARP_UTILS.StatePreparation.MakePrepareHadamardAllOp()

        qpe_op = phase_estimation.MakeStandardQPEOp(
            state_prep_op,
            ctrl_unitary_ops,
            phase_qubit_prep_op,
            num_bits,
            num_system_qubits,
        )
        measured_indices, bases = self._measurement_plan(num_bits, num_system_qubits)
        parameters = {
            "statePrep": state_prep_op,
            "controlledUnitary": ctrl_unitary_ops,
            "phaseQubitPrep": phase_qubit_prep_op,
            "numPhaseQubits": num_bits,
            "numSystemQubits": num_system_qubits,
            "numAncillaQubits": num_ancilla_qubits,
            "measuredIndices": measured_indices,
            "bases": [getattr(qsharp.Pauli, letter) for letter in bases],
        }
        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=phase_estimation.MakeStandardQPECircuit,
                parameter=parameters,
            ),
            qsharp_op=qpe_op,
        )

    def _measurement_plan(self, num_bits: int, num_system_qubits: int) -> tuple[list[int], list[str]]:
        """Resolve the ``measurement`` setting into register indices and Pauli letters.

        The executor reverses the Q# ``Result[]``, so emitting the system indices
        reversed and ahead of the phase indices makes the key read phase register
        most-significant-bit first, followed by the system register in order.

        Args:
            num_bits: Number of phase qubits.
            num_system_qubits: Number of system qubits.

        Returns:
            A tuple of (register indices to measure, Pauli letter per index).

        Raises:
            ValueError: If ``measurement`` or ``measurement_basis`` is invalid.

        """
        policy = str(self._settings.get("measurement"))
        phase_indices = list(range(num_bits))
        if policy == "none":
            return [], []
        if policy == "phase":
            return phase_indices, ["Z"] * num_bits
        if policy != "eigenvector":
            raise ValueError(f"measurement must be one of ['phase', 'eigenvector', 'none']. Got '{policy}'.")

        basis = str(self._settings.get("measurement_basis")).upper()
        if len(basis) == 1:
            basis *= num_system_qubits
        if len(basis) != num_system_qubits:
            raise ValueError(
                f"measurement_basis must be a single Pauli letter or one letter per system "
                f"qubit ({num_system_qubits}). Got '{basis}'."
            )
        if any(letter not in "IXYZ" for letter in basis):
            raise ValueError(f"measurement_basis must only contain the letters I, X, Y or Z. Got '{basis}'.")

        system = list(range(num_bits, num_bits + num_system_qubits))
        indices = list(reversed(system)) + phase_indices
        bases = [basis[index] for index in reversed(range(num_system_qubits))] + ["Z"] * num_bits
        return indices, bases

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "qdk_standard"
