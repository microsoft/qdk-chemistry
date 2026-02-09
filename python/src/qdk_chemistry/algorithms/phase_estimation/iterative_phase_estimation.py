"""Iterative phase estimation implementation.

This module implements the Kitaev-style iterative quantum phase estimation (IQPE)
algorithm, which measures phase bits sequentially from most-significant to least-significant
using a single ancilla qubit and adaptive feedback corrections.

References:
    Kitaev, A. (1995). arXiv:quant-ph/9511026. :cite:`Kitaev1995`

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import qdk
from qdk import qsharp

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
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
from qdk_chemistry.utils.phase import iterative_phase_feedback_update, phase_fraction_from_feedback

from .base import PhaseEstimation, PhaseEstimationSettings

__all__: list[str] = ["IterativePhaseEstimation", "IterativePhaseEstimationSettings"]


class IterativePhaseEstimationSettings(PhaseEstimationSettings):
    """Settings for the Iterative Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the settings for Iterative Phase Estimation.

        Args:
            shots_per_bit: The number of shots to execute per measuring a bit in the iterative phase estimation.

        """
        super().__init__()
        self._set_default(
            "shots_per_bit",
            "int",
            3,
            "The number of shots to execute per measuring a bit in the iterative phase estimation.",
        )


class IterativePhaseEstimation(PhaseEstimation):
    """Iterative Phase Estimation algorithm implementation."""

    def __init__(
        self,
        num_bits: int = -1,
        evolution_time: float = 0.0,
        shots_per_bit: int = 3,
    ):
        """Initialize IterativePhaseEstimation with the given settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            evolution_time: Time parameter ``t`` used in the time-evolution unitary ``U = exp(-i H t)``,
                defaults to 0.0; user needs to set a valid value.
            shots_per_bit: The number of shots to execute per measuring a bit in the iterative phase estimation.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits, evolution_time=evolution_time)
        self._settings = IterativePhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("evolution_time", evolution_time)
        self._settings.set("shots_per_bit", shots_per_bit)
        self._iteration_circuits: list[Circuit] | None = None

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
        """Run the iterative phase estimation algorithm with the given state preparation circuit and qubit Hamiltonian.

        Args:
            state_preparation: The state preparation circuit.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            evolution_builder: The time evolution builder to use.
            circuit_mapper: The controlled evolution circuit mapper to use.
            circuit_executor: The executor to run quantum circuits.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            QpeResult: The result of the phase estimation.

        """
        # Initialize the parameters
        phase_feedback = 0.0
        bits: list[int] = []
        iter_circuits: list[Circuit] = []

        # Iterate over the number of phase bits
        for iteration in range(self.settings().get("num_bits")):
            # Create the iteration circuit
            iteration_circuit = self.create_iteration_circuit(
                state_preparation=state_preparation,
                qubit_hamiltonian=qubit_hamiltonian,
                evolution_builder=evolution_builder,
                circuit_mapper=circuit_mapper,
                iteration=iteration,
                total_iterations=self.settings().get("num_bits"),
                phase_correction=phase_feedback,
            )
            iter_circuits.append(iteration_circuit)
            Logger.info(f"Iteration {iteration + 1} / {self.settings().get('num_bits')}: circuit generated.")
            # Run the iteration circuit on the simulator
            executor_data = circuit_executor.run(
                iteration_circuit, shots=self.settings().get("shots_per_bit"), noise=noise
            )
            bitstring_result = executor_data.bitstring_counts
            Logger.info(
                f"Iteration {iteration + 1} / {self.settings().get('num_bits')}: "
                f"Measurement results: {bitstring_result}"
            )
            # Phase bit through majority vote
            measured_bit = 0 if bitstring_result.get("0", 0) >= bitstring_result.get("1", 0) else 1
            Logger.debug(f"Majority measured bit: {measured_bit}")
            # Store the measured bit
            bits.append(measured_bit)

            # Update the phase feedback for next iteration
            phase_feedback = iterative_phase_feedback_update(phase_feedback, measured_bit)

        # Compute the final phase fraction
        phase_fraction = phase_fraction_from_feedback(phase_feedback)
        self._iteration_circuits = iter_circuits
        # Create and return the result
        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=phase_fraction,
            evolution_time=self.settings().get("evolution_time"),
            bits_msb_first=bits,
        )

    def create_iteration_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        evolution_builder: TimeEvolutionBuilder,
        circuit_mapper: ControlledEvolutionCircuitMapper,
        iteration: int,
        total_iterations: int,
        phase_correction: float = 0.0,
    ) -> Circuit:
        """Construct a single IQPE iteration circuit.

        Args:
            state_preparation: Trial-state preparation circuit that prepares the initial state on the system qubits.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            evolution_builder: The time evolution builder to use.
            circuit_mapper: The controlled evolution circuit mapper to use.
            iteration: Current iteration index (0-based), where 0 corresponds to the most-significant bit.
            total_iterations: Total number of phase bits to measure across all iterations.
            phase_correction: Feedback phase angle to apply before controlled evolution, defaults to 0.0.

        Returns:
            A quantum circuit implementing one IQPE iteration.

        """
        _validate_iteration_inputs(iteration, total_iterations)
        # Build the base circuit with registers
        num_system_qubits = qubit_hamiltonian.num_qubits
        # Import the Q# code from the IterativePhaseEstimation.qs file

        time_evolution_unitary = self._create_time_evolution(
            qubit_hamiltonian, self.settings().get("evolution_time"), evolution_builder
        )
        controlled_evolution = ControlledTimeEvolutionUnitary(
            time_evolution_unitary=time_evolution_unitary, control_indices=[0]
        )
        power = 2 ** (total_iterations - iteration - 1)
        ctrl_evol_circuit = self._create_ctrl_time_evol_circuit(controlled_evolution, power, circuit_mapper)

        if state_preparation._qsharp_op and ctrl_evol_circuit._qsharp_op:  # noqa: SLF001
            state_prep_op = state_preparation._qsharp_op  # noqa: SLF001
            ctrl_evol_op = ctrl_evol_circuit._qsharp_op  # noqa: SLF001
            code = (Path(__file__).parent / "IterativePhaseEstimation.qs").read_text()
            qsharp.eval(code)
            iqpe_iter_qsc = qsharp.circuit(
                qdk.code.MakeIQPECircuit,
                state_prep_op,
                ctrl_evol_op,
                phase_correction,
                0,
                [1 + i for i in range(num_system_qubits)],  # target qubits
            )
            iqpe_iter_qir = qsharp.compile(
                qdk.code.MakeIQPECircuit,
                state_prep_op,
                ctrl_evol_op,
                phase_correction,
                0,
                [1 + i for i in range(num_system_qubits)],  # target qubits
            )

            return Circuit(qsharp=iqpe_iter_qsc, qir=iqpe_iter_qir)

        state_prep_qc = None
        ctrl_evol_qc = None
        try:
            state_prep_qc = state_preparation.get_qiskit_circuit()
            ctrl_evol_qc = ctrl_evol_circuit.get_qiskit_circuit()
        except ImportError as err:
            raise RuntimeError(
                "Failed to create iteration circuit: Qiskit is not available to construct the circuits from QASM."
            ) from err

        if state_prep_qc is not None and ctrl_evol_qc is not None:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3  # noqa: PLC0415

            # Parse the state preparation circuit from QASM
            ancilla = QuantumRegister(1, "ancilla")
            system_target = QuantumRegister(num_system_qubits, "system")
            classical = ClassicalRegister(1, f"c{iteration}")
            circuit = QuantumCircuit(ancilla, system_target, classical)
            circuit.append(state_prep_qc.to_gate(), system_target)
            control = ancilla[0]
            target_qubits = list(system_target)
            circuit.h(control)

            # Apply phase correction if provided
            if phase_correction:
                circuit.rz(phase_correction, control)

            # Append the controlled evolution circuit
            circuit.append(ctrl_evol_qc.to_gate(), [control, *target_qubits])
            circuit.h(control)
            circuit.measure(control, classical[0])

            return Circuit(qasm=qasm3.dumps(circuit))

        raise RuntimeError(
            "Failed to create iteration circuit: Q# operations or Qiskit dependencies are not available."
        )

    def get_circuits(self) -> list[Circuit]:
        """Get the list of iteration circuits generated during algorithm execution.

        Returns:
            List of quantum circuits for each IQPE iteration.

        Raises:
            ValueError: If no iteration circuits are available.

        """
        if self._iteration_circuits is not None:
            return self._iteration_circuits
        raise ValueError("No iteration circuits have been generated. Please run the algorithm first.")

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "iterative"


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
