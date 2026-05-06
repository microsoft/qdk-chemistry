"""Standard (QFT-based) phase estimation circuit builder.

This module implements the circuit-building component of the standard quantum phase
estimation algorithm, which constructs the full QPE circuit with inverse QFT without
executing it.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3
from qiskit.synthesis.qft.qft_decompose_full import synth_qft_full

from qdk_chemistry.algorithms.phase_estimation.builder.base import (
    PhaseEstimationBuilder,
    PhaseEstimationBuilderSettings,
)
from qdk_chemistry.data import (
    Circuit,
    QubitHamiltonian,
)
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitStandardPhaseEstimationBuilder", "QiskitStandardPhaseEstimationBuilderSettings"]


class QiskitStandardPhaseEstimationBuilderSettings(PhaseEstimationBuilderSettings):
    """Settings for the Standard Phase Estimation Builder."""

    def __init__(self):
        """Initialize the settings for the Standard Phase Estimation Builder."""
        super().__init__()
        self._set_default(
            "qft_do_swaps",
            "bool",
            True,
            "Whether to include the final swap layer in the inverse QFT.",
        )


class QiskitStandardPhaseEstimationBuilder(PhaseEstimationBuilder):
    """Standard QFT-based phase estimation circuit builder.

    Constructs the full QPE circuit (state prep, controlled unitaries, inverse QFT,
    measurements) without executing it. Can be used standalone for resource estimation
    or composed inside QiskitStandardPhaseEstimation.

    """

    def __init__(self, num_bits: int = -1, qft_do_swaps: bool = True):
        """Initialize the StandardPhaseEstimationBuilder.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            qft_do_swaps: Whether to include the final swap layer in the inverse QFT.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = QiskitStandardPhaseEstimationBuilderSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("qft_do_swaps", qft_do_swaps)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> list[Circuit]:
        """Build the standard QPE circuit.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to build the circuit.

        Returns:
            A single-element list containing the full QPE circuit.

        """
        circuit = self.build_circuit(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
        )
        return [circuit]

    def build_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
    ) -> Circuit:
        """Build the standard QPE circuit.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.

        Returns:
            The constructed QPE quantum circuit.

        """
        Logger.trace_entering()
        num_bits = self._settings.get("num_bits")
        ancilla = QuantumRegister(num_bits, "ancilla")
        system = QuantumRegister(qubit_hamiltonian.num_qubits, "system")
        classical = ClassicalRegister(num_bits, "c")
        qc = QuantumCircuit(ancilla, system, classical)

        Logger.debug(f"Creating traditional QPE circuit with {num_bits} ancilla qubits and measurements.")
        state_prep = state_preparation.get_qiskit_circuit()
        if state_prep.num_qubits != qubit_hamiltonian.num_qubits:
            raise ValueError(
                "state_preparation must prepare the same number of system qubits as the Hamiltonian "
                f"(expected {qubit_hamiltonian.num_qubits}, received {state_prep.num_qubits}).",
            )

        qc.compose(state_prep, qubits=system, inplace=True)

        for idx in range(num_bits):
            qc.h(ancilla[idx])

        for ancilla_idx in range(num_bits):
            power = 2**ancilla_idx
            self._append_controlled_unitary(
                circuit=qc,
                qubit_hamiltonian=qubit_hamiltonian,
                control_qubit=ancilla[ancilla_idx],
                target_qubits=system,
                power=power,
            )

        inverse_qft = synth_qft_full(
            num_bits, do_swaps=self._settings.get("qft_do_swaps"), inverse=True, name="Inverse QFT"
        )
        qc.compose(inverse_qft.to_gate(), qubits=ancilla, inplace=True)
        qc.measure(ancilla, classical)
        Logger.debug(f"Completed standard QPE circuit with {qc.num_qubits} qubits.")

        return Circuit(qasm3.dumps(qc))

    def _append_controlled_unitary(
        self,
        circuit: QuantumCircuit,
        qubit_hamiltonian: QubitHamiltonian,
        control_qubit: int,
        target_qubits: list,
        *,
        power: int,
    ) -> None:
        """Apply the controlled unitary to the circuit.

        Args:
            circuit: The quantum circuit to modify.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            control_qubit: The control qubit.
            target_qubits: List of target qubits.
            power: The power to which the controlled unitary is raised.

        """
        ctrl_unitary_circuit = self._create_controlled_circuit(qubit_hamiltonian=qubit_hamiltonian, power=power)
        cu_circuit = ctrl_unitary_circuit.get_qiskit_circuit()

        mapping = [control_qubit, *target_qubits]
        circuit.compose(cu_circuit, qubits=mapping, inplace=True)

    def name(self) -> str:
        """Return the name of the builder algorithm."""
        return "standard"
