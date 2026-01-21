"""Standard (QFT-based) phase estimation implementation via qiskit.

This module implements the standard quantum phase estimation algorithm using the qiskit synthesis
of the inverse Quantum Fourier Transform (QFT), which measures all phase bits in parallel using
multiple ancilla qubits.

References:
    Nielsen, M. A., & Chuang, I. L. (2010). :cite:`Nielsen-Chuang2010-QPE`

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3
from qiskit.synthesis.qft.qft_decompose_full import synth_qft_full

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.algorithms.phase_estimation.base import PhaseEstimation, PhaseEstimationSettings
from qdk_chemistry.algorithms.time_evolution.builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.time_evolution.controlled_circuit_mapper.base import ControlledEvolutionCircuitMapper
from qdk_chemistry.data import (
    Circuit,
    ControlledTimeEvolutionUnitary,
    QpeResult,
    QuantumErrorProfile,
    QubitHamiltonian,
)
from qdk_chemistry.utils import Logger

__all__: list[str] = ["QiskitStandardPhaseEstimation", "QiskitStandardPhaseEstimationSettings"]


class QiskitStandardPhaseEstimationSettings(PhaseEstimationSettings):
    """Settings for the Qiskit Standard Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the settings for Qiskit Standard Phase Estimation.

        Args:
            qft_do_swaps: Whether to include the final swap layer in the inverse QFT.
            shots: The number of shots to execute the circuit.

        """
        super().__init__()
        self._set_default(
            "qft_do_swaps",
            "bool",
            True,
            "Whether to include the final swap layer in the inverse QFT.",
        )
        self._set_default(
            "shots",
            "int",
            3,
            "The number of shots to execute the circuit.",
        )


class QiskitStandardPhaseEstimation(PhaseEstimation):
    """Standard QFT-based (non-iterative) phase estimation."""

    def __init__(self, num_bits: int = -1, evolution_time: float = 0.0, qft_do_swaps: bool = True, shots: int = 3):
        """Initialize the Qiskit standard phase estimation routine.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            evolution_time: Time parameter ``t`` used in the time-evolution unitary ``U = exp(-i H t)``,
                defaults to 0.0; user needs to set a valid value.
            qft_do_swaps: Whether to include the final swap layer in the inverse QFT.
                Defaults to ``True`` so that the measured bit string is
                ordered from most-significant to least-significant bit.
            shots: The number of shots to execute the circuit.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits, evolution_time=evolution_time)
        self._settings = QiskitStandardPhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("evolution_time", evolution_time)
        self._settings.set("qft_do_swaps", qft_do_swaps)
        self._settings.set("shots", shots)
        self._qpe_circuit: Circuit | None = None

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: ControlledEvolutionCircuitMapper,
        circuit_executor: CircuitExecutor,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run the standard phase estimation algorithm given the state preparation and qubit Hamiltonian.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate eigenvalues.
            evolution_builder: The time evolution builder to use.
            circuit_mapper: The controlled evolution circuit mapper to use.
            circuit_executor: The executor to run quantum circuits.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            A QpeResult object containing the results of the phase estimation.

        """
        Logger.trace_entering()
        circuit = self.create_circuit(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
            evolution_builder=evolution_builder,
            circuit_mapper=circuit_mapper,
        )
        self._qpe_circuit = circuit
        shots = self._settings.get("shots")
        execution_data = circuit_executor.run(circuit, shots=shots, noise=noise)
        counts = execution_data.bitstring_counts

        dominant_bitstring = max(counts, key=counts.get)
        raw_phase = int(dominant_bitstring, 2) / (2 ** self._settings.get("num_bits"))

        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=raw_phase,
            evolution_time=self.settings().get("evolution_time"),
            bits_msb_first=dominant_bitstring,
        )

    def create_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: ControlledEvolutionCircuitMapper,
    ) -> Circuit:
        """Build the standard QPE circuit.

        Args:
            state_preparation: The circuit that prepares the initial state.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            evolution_builder: The time evolution builder to use.
            circuit_mapper: The controlled evolution circuit mapper to use.

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
        state_prep = qasm3.loads(state_preparation.qasm)
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
            self._append_controlled_evolution(
                circuit=qc,
                qubit_hamiltonian=qubit_hamiltonian,
                time=self._settings.get("evolution_time"),
                control_qubit=ancilla[ancilla_idx],
                target_qubits=system,
                evolution_builder=evolution_builder,
                circuit_mapper=circuit_mapper,
                power=power,
            )

        inverse_qft = synth_qft_full(
            num_bits, do_swaps=self._settings.get("qft_do_swaps"), inverse=True, name="Inverse QFT"
        )
        qc.compose(inverse_qft.to_gate(), qubits=ancilla, inplace=True)
        qc.measure(ancilla, classical)
        Logger.debug(f"Completed standard QPE circuit with {qc.num_qubits} qubits.")

        return Circuit(qasm3.dumps(qc))

    def _append_controlled_evolution(
        self,
        circuit: QuantumCircuit,
        qubit_hamiltonian: QubitHamiltonian,
        control_qubit: int,
        target_qubits: list,
        *,
        time: float,
        power: int,
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: ControlledEvolutionCircuitMapper,
    ) -> None:
        """Apply the controlled time evolution unitary to the circuit.

        Args:
            circuit: The quantum circuit to modify.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            control_qubit: The control qubit.
            target_qubits: List of target qubits.
            time: The evolution time.
            power: The power to which the controlled evolution unitary is raised.
            evolution_builder: The time evolution builder to use.
            circuit_mapper: The controlled evolution circuit mapper to use.

        """
        time_evol_unitary = self._create_time_evolution(
            qubit_hamiltonian=qubit_hamiltonian,
            time=time,
            evolution_builder=evolution_builder,
        )
        ctrl_time_evol = ControlledTimeEvolutionUnitary(
            time_evolution_unitary=time_evol_unitary,
            control_indices=[0],
        )

        ctrl_time_evol_circuit = self._create_ctrl_time_evol_circuit(
            controlled_evolution=ctrl_time_evol, power=power, circuit_mapper=circuit_mapper
        )
        cu_circuit = qasm3.loads(ctrl_time_evol_circuit.qasm)

        mapping = [control_qubit, *target_qubits]
        circuit.compose(cu_circuit, qubits=mapping, inplace=True)

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
        """Return the algorithm name as qiskit_standard."""
        return "qiskit_standard"
