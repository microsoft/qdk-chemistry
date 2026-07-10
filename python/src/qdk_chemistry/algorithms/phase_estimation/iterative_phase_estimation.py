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
import numpy as np

from qdk_chemistry.data import (
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitOperator,
)
from qdk_chemistry.utils import Logger

from .base import PhaseEstimation, PhaseEstimationSettings
from .circuit_builder.base import IterativeQpeCircuitBuilder

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
        shots_per_bit: int = 3,
    ):
        """Initialize IterativePhaseEstimation with the given settings.

        Args:
            shots_per_bit: The number of shots to execute per measuring a bit in the iterative phase estimation.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = IterativePhaseEstimationSettings()
        self._settings.set("shots_per_bit", shots_per_bit)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
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
        circuit_builder = self._create_nested("qpe_circuit_builder")
        if not isinstance(circuit_builder, IterativeQpeCircuitBuilder):
            raise TypeError(
                f"Expected qpe_circuit_builder to be an instance of IterativeQpeCircuitBuilder, "
                f"but got {type(circuit_builder)} instead."
            )

        # Resolve container before running iterations
        unitary_builder = circuit_builder._create_nested("unitary_builder")  # noqa: SLF001
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        container = unitary_rep.get_container()

        num_bits = circuit_builder.settings().get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")
        # Initialize the parameters
        phase_feedback = 0.0
        bits: list[int] = []

        # Iterate over the number of phase bits
        for iteration in range(num_bits):
            # Create the iteration circuit via the builder
            circuit_builder.settings().update("phase_correction", phase_feedback)
            circuit_builder.settings().update("num_iteration", iteration)
            iteration_circuits = circuit_builder._run_impl(  # noqa: SLF001
                state_preparation=state_preparation, qubit_hamiltonian=qubit_hamiltonian
            )
            iteration_circuit = iteration_circuits[0]
            Logger.info(f"Iteration {iteration + 1} / {num_bits}: circuit generated.")
            # Run the iteration circuit on the simulator
            executor_data = circuit_executor.run(
                iteration_circuit, shots=self.settings().get("shots_per_bit"), noise=noise
            )
            bitstring_result = executor_data.bitstring_counts
            Logger.info(f"Iteration {iteration + 1} / {num_bits}: Measurement results: {bitstring_result}")
            # Phase bit through majority vote
            measured_bit = 0 if bitstring_result.get("0", 0) >= bitstring_result.get("1", 0) else 1
            Logger.debug(f"Majority measured bit: {measured_bit}")
            # Store the measured bit
            bits.append(measured_bit)

            # Update the phase feedback for next iteration
            phase_feedback = phase_feedback / 2.0 + np.pi * measured_bit / 2.0

        # Compute the final phase fraction
        phase_fraction = phase_feedback / np.pi

        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=phase_fraction,
            eigenvalue_from_phase=container.eigenvalue_from_phase,
            bits_msb_first=bits,
        )

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "qdk_iterative"
