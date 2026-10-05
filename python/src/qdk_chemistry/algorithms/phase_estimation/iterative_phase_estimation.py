"""Iterative phase estimation implementation.

This module implements the Kitaev-style iterative quantum phase estimation (IQPE)
algorithm, which measures phase bits sequentially from least-significant to most-significant
using a single ancilla qubit and adaptive feedback corrections. The first iteration applies
the largest controlled power :math:`U^{2^{n-1}}` (``n`` = ``num_bits``) and therefore measures
the least-significant bit; subsequent iterations proceed toward the most-significant bit.
The returned `QpeResult.bits_msb_first` reverses this execution order
into the conventional most-significant-first bitstring.

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
from .circuit_builder.iterative_builder import QdkIterativeQpeCircuitBuilder

__all__: list[str] = ["IterativePhaseEstimation", "IterativePhaseEstimationSettings"]


class IterativePhaseEstimationSettings(PhaseEstimationSettings):
    """Settings for the Iterative Phase Estimation algorithm."""

    def __init__(self):
        """Initialize the sampling and execution-mode settings for Iterative Phase Estimation."""
        super().__init__()
        self._set_default(
            "shots_per_bit",
            "int",
            3,
            "Samples per phase bit; combined mode votes inside one whole-circuit execution.",
        )
        self._set_default(
            "combine_iterations",
            "bool",
            False,
            "Run all rounds in one adaptive circuit with in-circuit voting. Requires the QDK iterative builder.",
        )


class IterativePhaseEstimation(PhaseEstimation):
    """Iterative Phase Estimation algorithm implementation."""

    def __init__(
        self,
        shots_per_bit: int = 3,
        combine_iterations: bool = False,
    ):
        """Initialize IterativePhaseEstimation with the given settings.

        Args:
            shots_per_bit: Samples per phase bit. When ``combine_iterations`` is enabled,
                overrides the builder's ``mid_shots`` to vote inside a single whole-circuit execution.
            combine_iterations: Run every round in one adaptive circuit, executed once.
                Requires the QDK iterative circuit builder. Default to False.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = IterativePhaseEstimationSettings()
        self._settings.set("shots_per_bit", shots_per_bit)
        self._settings.set("combine_iterations", combine_iterations)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run the iterative phase estimation algorithm with the given state preparation circuit and Hamiltonian.

        Args:
            state_preparation: The state preparation circuit.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            QpeResult: The result of the phase estimation.

        Raises:
            ValueError: If ``shots_per_bit`` or ``num_bits`` is not positive, combined mode is requested
                with an unsupported builder, or combined mode is enabled only on the nested builder.

        """
        settings = self.settings()
        shots_per_bit = settings.get("shots_per_bit")
        combine_iterations = settings.get("combine_iterations")
        if shots_per_bit <= 0:
            raise ValueError(f"shots_per_bit must be a positive integer. Got {shots_per_bit}.")

        # Create nested algorithms from settings
        circuit_executor = self._create_nested("circuit_executor")
        circuit_builder = self._create_nested("qpe_circuit_builder")
        if not isinstance(circuit_builder, IterativeQpeCircuitBuilder):
            raise TypeError(
                f"Expected qpe_circuit_builder to be an instance of IterativeQpeCircuitBuilder, "
                f"but got {type(circuit_builder)} instead."
            )

        builder_settings = circuit_builder.settings()
        if isinstance(circuit_builder, QdkIterativeQpeCircuitBuilder):
            if builder_settings.get("combine_iterations") and not combine_iterations:
                raise ValueError(
                    "Set combine_iterations=True on IterativePhaseEstimation, not only on qpe_circuit_builder."
                )
            builder_settings.update("combine_iterations", combine_iterations)
            if combine_iterations:
                builder_settings.update("mid_shots", shots_per_bit)
        elif combine_iterations:
            raise ValueError(
                f"combine_iterations=True requires a QDK iterative QPE circuit builder; got {circuit_builder.name()!r}."
            )

        # Resolve container before running iterations
        unitary_builder = circuit_builder._create_nested("unitary_builder")  # noqa: SLF001
        unitary_rep = unitary_builder.run(qubit_hamiltonian)
        container = unitary_rep.get_container()

        num_bits = builder_settings.get("num_bits")
        if num_bits <= 0:
            raise ValueError(f"num_bits must be a positive integer. Got {num_bits}.")

        # Full single-circuit IQPE with in-circuit classical feedback (Adaptive-profile targets).
        if combine_iterations:
            return self._run_single_circuit(
                circuit_builder=circuit_builder,
                circuit_executor=circuit_executor,
                state_preparation=state_preparation,
                qubit_hamiltonian=qubit_hamiltonian,
                container=container,
                num_bits=num_bits,
                shots_per_bit=shots_per_bit,
                noise=noise,
            )

        # Initialize the parameters
        phase_feedback = 0.0
        bits: list[int] = []

        # Iterate over the number of phase bits
        for iteration in range(num_bits):
            # Create the iteration circuit via the builder
            circuit_builder.settings().update("phase_correction", -phase_feedback)
            circuit_builder.settings().update("num_iteration", iteration)
            iteration_circuits = circuit_builder._run_impl(  # noqa: SLF001
                state_preparation=state_preparation, qubit_hamiltonian=qubit_hamiltonian
            )
            iteration_circuit = iteration_circuits[0]
            Logger.info(f"Iteration {iteration + 1} / {num_bits}: circuit generated.")
            # Run the iteration circuit on the simulator
            executor_data = circuit_executor.run(iteration_circuit, shots=shots_per_bit, noise=noise)
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
            bits_msb_first=bits[::-1],
        )

    def _run_single_circuit(
        self,
        *,
        circuit_builder: IterativeQpeCircuitBuilder,
        circuit_executor,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
        container,
        num_bits: int,
        shots_per_bit: int,
        noise: QuantumErrorProfile | None,
    ) -> QpeResult:
        """Run the full IQPE as a single circuit with in-circuit classical feedback.

        The builder produces one circuit that performs every round using mid-circuit
        measurement and classical feed-forward. The estimator sets the builder's
        ``mid_shots`` to ``shots_per_bit``, so every round uses the same sample count
        as the per-bit path and feeds its majority bit forward. The executor runs the
        whole circuit once, and its voted bitstring is decoded as
        ``int(bitstring_msb_first, 2) / 2**num_bits``.

        Args:
            circuit_builder: The iterative circuit builder configured with ``combine_iterations`` enabled.
            circuit_executor: The circuit executor used to run the circuit.
            state_preparation: The state preparation circuit.
            qubit_hamiltonian: The qubit Hamiltonian for which to estimate the phase.
            container: The unitary container providing ``eigenvalue_from_phase``.
            num_bits: The number of phase bits to estimate.
            shots_per_bit: The validated number of internal samples per phase bit.
            noise: The quantum error profile to simulate noise, defaults to None.

        Returns:
            QpeResult: The result of the phase estimation.

        Raises:
            RuntimeError: If the executor returns no measurement results.

        """
        full_circuit = circuit_builder.run(state_preparation=state_preparation, qubit_hamiltonian=qubit_hamiltonian)[0]
        Logger.info(
            "combine_iterations=True: running the whole circuit once (shots=1); "
            f"shots_per_bit={shots_per_bit} samples per phase bit, voted inside the circuit."
        )
        executor_data = circuit_executor.run(full_circuit, shots=1, noise=noise)
        counts = executor_data.bitstring_counts
        if not counts:
            raise RuntimeError("No measurement results returned from the circuit executor.")

        # Each shot returns one voted bitstring, MSB-first, as in the standard QPE path.
        bitstring_msb_first = min(counts, key=lambda b: (-counts[b], b))
        Logger.info(f"Voted bitstring (MSB first): {bitstring_msb_first}")
        phase_fraction = int(bitstring_msb_first, 2) / (2**num_bits)

        return QpeResult.from_phase_fraction(
            method=self.name(),
            phase_fraction=phase_fraction,
            eigenvalue_from_phase=container.eigenvalue_from_phase,
            bits_msb_first=[int(c) for c in bitstring_msb_first],
            bitstring_msb_first=bitstring_msb_first,
        )

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "qdk_iterative"
