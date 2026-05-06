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

from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.phase_estimation.builder.iterative_builder import IterativePhaseEstimationBuilder
from qdk_chemistry.data import (
    Circuit,
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
        shots_per_bit: int = 3,
    ):
        """Initialize IterativePhaseEstimation with the given settings.

        Args:
            num_bits: The number of phase bits to estimate. Default to -1; user needs to set a valid value.
            shots_per_bit: The number of shots to execute per measuring a bit in the iterative phase estimation.

        """
        Logger.trace_entering()
        super().__init__(num_bits=num_bits)
        self._settings = IterativePhaseEstimationSettings()
        self._settings.set("num_bits", num_bits)
        self._settings.set("shots_per_bit", shots_per_bit)
        self._iteration_circuits: list[Circuit] | None = None

    def _create_builder(self) -> IterativePhaseEstimationBuilder:
        """Create an IterativePhaseEstimationBuilder with settings propagated from this algorithm.

        Returns:
            An IterativePhaseEstimationBuilder instance configured with matching
            unitary_builder, circuit_mapper, and num_bits settings.

        """
        builder = IterativePhaseEstimationBuilder(num_bits=self.settings().get("num_bits"))
        builder.settings().update("unitary_builder", self.settings().get("unitary_builder"))
        builder.settings().update("circuit_mapper", self.settings().get("circuit_mapper"))
        return builder

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run the iterative phase estimation algorithm with the given state preparation circuit and qubit Hamiltonian.

        Args:
            state_preparation: The state preparation circuit.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            QpeResult: The result of the phase estimation.

        """
        # Create nested algorithms from settings
        circuit_executor = self._create_nested("circuit_executor")
        builder = self._create_builder()
        # Initialize the parameters
        phase_feedback = 0.0
        bits: list[int] = []
        iter_circuits: list[Circuit] = []

        # Iterate over the number of phase bits
        for iteration in range(self.settings().get("num_bits")):
            # Create the iteration circuit via the builder
            iteration_circuit = builder.build_iteration_circuit(
                state_preparation=state_preparation,
                qubit_hamiltonian=qubit_hamiltonian,
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

        if isinstance(self.unitary_builder, TimeEvolutionBuilder):
            evolution_time = self.unitary_builder.settings().get("time")
            return QpeResult.from_phase_fraction(
                method=self.name(),
                phase_fraction=phase_fraction,
                evolution_time=evolution_time,
                bits_msb_first=bits,
            )
        raise NotImplementedError(
            "IQPE result construction currently only supports post-processing from time evolution. "
            f"Got {type(self.unitary_builder)} instead."
        )

    def create_iteration_circuit(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitHamiltonian,
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
        builder = self._create_builder()
        return builder.build_iteration_circuit(
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
            iteration=iteration,
            total_iterations=total_iterations,
            phase_correction=phase_correction,
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
