"""Standard (QFT-based) phase estimation implementation via Q#.

This module implements the standard quantum phase estimation algorithm using Q#,
which measures all phase bits in parallel using multiple ancilla qubits and an
inverse Quantum Fourier Transform.

References:
    Nielsen, M. A., & Chuang, I. L. (2010). :cite:`Nielsen-Chuang2010-QPE`

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder
from qdk_chemistry.data import (
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.qsharp import QSHARP_UTILS

from .base import PhaseEstimation, PhaseEstimationSettings

__all__: list[str] = ["StandardPhaseEstimation", "StandardPhaseEstimationSettings"]


class StandardPhaseEstimationSettings(PhaseEstimationSettings):
    """Settings for the Standard Phase Estimation algorithm (Q# implementation)."""

    def __init__(self):
        """Initialize the settings for Standard Phase Estimation.

        Includes shots setting for repeated circuit execution.

        """
        super().__init__()
        self._set_default(
            "shots",
            "int",
            3,
            "The number of shots to execute the QPE circuit.",
        )


class StandardPhaseEstimation(PhaseEstimation):
    """Standard QFT-based (non-iterative) phase estimation via Q#."""

    def __init__(self, num_bits: int = -1, shots: int = 3):
        """Initialize the standard phase estimation routine.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            shots: The number of shots to execute the circuit.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = StandardPhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("shots", shots)
        self._qpe_circuit: Circuit | None = None

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run the standard phase estimation algorithm with the given state preparation and qubit Hamiltonian.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the results of the phase estimation.

        """
        Logger.trace_entering()
        circuit_executor = self._create_nested("circuit_executor")
        circuit = self.create_circuit(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )
        self._qpe_circuit = circuit
        shots = self._settings.get("shots")
        execution_data = circuit_executor.run(circuit, shots=shots, noise=noise)
        counts = execution_data.bitstring_counts

        dominant_bitstring = max(counts, key=counts.get)
        num_bits = self._settings.get("num_bits")
        raw_phase = int(dominant_bitstring, 2) / (2**num_bits)

        if isinstance(self.unitary_builder, TimeEvolutionBuilder):
            evolution_time = self.unitary_builder.settings().get("time")
            return QpeResult.from_phase_fraction(
                method=self.name(),
                phase_fraction=raw_phase,
                evolution_time=evolution_time,
                bits_msb_first=dominant_bitstring,
            )
        raise NotImplementedError(
            "QPE result construction currently only supports post-processing from time evolution. "
            f"Got {type(self.unitary_builder)} instead."
        )

    def create_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> Circuit:
        """Build the standard QPE circuit using Q#.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.

        Returns:
            The constructed QPE quantum circuit.

        Raises:
            RuntimeError: If Q# operations are not available on the provided circuits.

        """
        Logger.trace_entering()
        num_bits = self._settings.get("num_bits")
        num_system_qubits = qubit_hamiltonian.num_qubits

        if not state_preparation._qsharp_op:  # noqa: SLF001
            raise RuntimeError(
                "Standard QPE (Q#) requires state preparation with a Q# operation. "
                "Ensure the state preparation circuit has a valid _qsharp_op."
            )

        # Build the controlled unitary using power=1 (the Q# circuit repeats it internally)
        ctrl_unitary_circuit = self._create_controlled_circuit(qubit_hamiltonian, power=1)
        if not ctrl_unitary_circuit._qsharp_op:  # noqa: SLF001
            raise RuntimeError(
                "Standard QPE (Q#) requires a controlled unitary with a Q# operation. "
                "Ensure the circuit mapper produces a circuit with a valid _qsharp_op."
            )

        state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
        ctrl_unitary_op = ctrl_unitary_circuit._qsharp_op  # noqa: SLF001

        # Qubit layout: ancillas [0..num_bits-1], systems [num_bits..num_bits+num_system-1]
        ancillas = list(range(num_bits))
        systems = [num_bits + i for i in range(num_system_qubits)]

        qpe_parameters = {
            "statePrep": state_prep_op,
            "controlledEvolution": ctrl_unitary_op,
            "numBits": num_bits,
            "ancillas": ancillas,
            "systems": systems,
        }

        Logger.debug(
            f"Creating standard QPE circuit (Q#) with {num_bits} ancilla qubits and {num_system_qubits} system qubits."
        )

        return Circuit(
            qsharp_factory=QsharpFactoryData(
                program=QSHARP_UTILS.StandardPhaseEstimation.MakeStandardQPECircuit,
                parameter=qpe_parameters,
            )
        )

    def get_circuit(self) -> Circuit:
        """Get the QPE circuit generated during algorithm execution.

        Returns:
            The quantum circuit used in the last execution.

        Raises:
            ValueError: If no QPE circuit is available.

        """
        if self._qpe_circuit is not None:
            return self._qpe_circuit
        raise ValueError("No QPE circuit has been generated. Please run the algorithm first.")

    def name(self) -> str:
        """Return the algorithm name as standard."""
        return "standard"
