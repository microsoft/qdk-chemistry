r"""Robust (randomized) phase estimation driver.

This module implements ``RobustPhaseEstimation``, a phase-estimation algorithm
suited to *randomized* time-evolution channels (qDRIFT / partially randomized),
where the textbook iterative-QPE trick of coherently powering a single unitary
draw is invalid. Instead it samples the Hadamard-test signal
:math:`g(t) = \langle\psi|e^{-iHt}|\psi\rangle` at a geometric ladder of
evolution times, averaging shots into a complex signal per round, and refines
the eigenphase with a robust angle-consistency update.

References:
    Günther, J., Witteveen, F., et al. (2025). Phase estimation with partially
    randomized time evolution. PRX Quantum 7, 020332. arXiv:2503.05647.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import numpy as np

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitOperator,
    Settings,
)
from qdk_chemistry.utils import Logger
from qdk_chemistry.utils.rpe import (
    energy_from_rpe_angle,
    expectation_from_counts,
    qdrift_phase_to_energy,
    rpe_angle_update,
    wrap_to_principal,
)

from .base import PhaseEstimation
from .circuit_builder.robust_builder import (
    RobustPhaseEstimationCircuitBuilder,
    RobustPhaseEstimationCircuitSet,
)

__all__: list[str] = ["RobustPhaseEstimation", "RobustPhaseEstimationSettings"]


class RobustPhaseEstimationSettings(Settings):
    """Settings for the Robust Phase Estimation algorithm."""

    def __init__(self) -> None:
        """Initialize circuit-builder and executor references."""
        super().__init__()
        self._set_default(
            "robust_phase_estimation_circuit_builder",
            "algorithm_ref",
            AlgorithmRef("robust_phase_estimation_circuit_builder", "qdk"),
            "Circuit builder that owns the RPE schedule and circuit-generation settings.",
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
            "Backend used to execute generated Hadamard-test circuits.",
        )


class RobustPhaseEstimation(PhaseEstimation):
    """Robust phase estimation for randomized time-evolution channels."""

    def __init__(
        self,
        robust_phase_estimation_circuit_builder: AlgorithmRef | None = None,
        circuit_executor: AlgorithmRef | None = None,
    ) -> None:
        """Initialize robust phase estimation orchestration.

        Args:
            robust_phase_estimation_circuit_builder: Optional reference to the RPE circuit builder.
            circuit_executor: Optional reference to the circuit execution backend.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = RobustPhaseEstimationSettings()
        if robust_phase_estimation_circuit_builder is not None:
            self._settings.set("robust_phase_estimation_circuit_builder", robust_phase_estimation_circuit_builder)
        if circuit_executor is not None:
            self._settings.set("circuit_executor", circuit_executor)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Run robust phase estimation and return the recovered energy.

        Args:
            state_preparation: Circuit preparing the trial state on the system qubits.
            qubit_hamiltonian: Qubit Hamiltonian whose eigenenergy is estimated.
            noise: Noise model. Not supported in this MVP and ignored if provided.

        Returns:
            QpeResult: Result carrying the resolved energy.

        """
        Logger.trace_entering()
        circuit_set = self.build_circuit_set(state_preparation, qubit_hamiltonian)
        return self.execute_circuit_set(circuit_set, noise=noise)

    def build_circuit_set(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Build the public lazy circuit collection configured for this estimator.

        Args:
            state_preparation: Circuit preparing the trial state on the system qubits.
            qubit_hamiltonian: Qubit Hamiltonian whose eigenenergy is estimated.

        Returns:
            Lazy, re-iterable robust phase estimation circuit collection.

        Raises:
            TypeError: If the configured nested algorithm is not an RPE circuit builder.

        """
        circuit_builder = self._create_nested("robust_phase_estimation_circuit_builder")
        if not isinstance(circuit_builder, RobustPhaseEstimationCircuitBuilder):
            raise TypeError(
                "Expected robust_phase_estimation_circuit_builder to be an instance of "
                f"RobustPhaseEstimationCircuitBuilder, got {type(circuit_builder)} instead."
            )
        return circuit_builder.run(state_preparation, qubit_hamiltonian)

    def execute_circuit_set(
        self,
        circuit_set: RobustPhaseEstimationCircuitSet,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Execute a public RPE circuit set and return the recovered energy.

        Args:
            circuit_set: Lazy circuit collection returned by an RPE circuit builder.
            noise: Noise model. Not supported in this MVP and ignored if provided.

        Returns:
            QpeResult carrying the resolved energy.

        Raises:
            TypeError: If ``circuit_set`` is not a robust phase estimation circuit set.

        """
        if not isinstance(circuit_set, RobustPhaseEstimationCircuitSet):
            raise TypeError(
                f"circuit_set must be an instance of RobustPhaseEstimationCircuitSet, got {type(circuit_set)} instead."
            )
        if noise is not None:
            Logger.warning("RobustPhaseEstimation does not support noise yet; ignoring the noise model.")

        Logger.info(
            f"RobustPhaseEstimation: lambda={circuit_set.lambda_norm:.6g}, "
            f"base_time={circuit_set.base_time:.6g}, rounds={circuit_set.num_rounds}, "
            f"builder={circuit_set.unitary_builder_category}, correction={circuit_set.energy_correction}, "
            f"eps_rpe={circuit_set.epsilon_rpe:.3g}, eps_unitary={circuit_set.epsilon_unitary:.3g}."
        )

        circuit_executor = self._create_nested("circuit_executor")
        theta = 0.0
        for round_data in circuit_set.rounds:
            real_accumulator = 0.0
            imag_accumulator = 0.0
            for experiment in circuit_set.iter_round(round_data.round_index):
                real_data = circuit_executor.run(
                    experiment.x_circuit,
                    shots=experiment.circuit_multiplicity,
                )
                imag_data = circuit_executor.run(
                    experiment.y_circuit,
                    shots=experiment.circuit_multiplicity,
                )
                real_accumulator += expectation_from_counts(real_data.bitstring_counts)
                imag_accumulator += expectation_from_counts(imag_data.bitstring_counts)

            real_part = real_accumulator / float(round_data.num_draws)
            imag_part = imag_accumulator / float(round_data.num_draws)
            measured_phase = float(np.angle(complex(real_part, imag_part)))
            theta = rpe_angle_update(theta, measured_phase, round_data.round_index)
            Logger.debug(
                f"Round {round_data.round_index}: shots={round_data.shots_per_basis}, "
                f"samples={round_data.scheduled_samples}, phi={measured_phase:.6f}, theta={theta:.6f}."
            )

        energy = self._resolve_energy(
            theta,
            circuit_set.base_time,
            circuit_set.num_rounds - 1,
            circuit_set.lambda_norm,
            circuit_set.final_samples,
            correction=circuit_set.energy_correction,
        )
        metadata = {
            "lambda": circuit_set.lambda_norm,
            "base_time": circuit_set.base_time,
            "num_rounds": circuit_set.num_rounds,
            "target_accuracy": circuit_set.target_accuracy,
            "epsilon_rpe": circuit_set.epsilon_rpe,
            "epsilon_unitary": circuit_set.epsilon_unitary,
            "unitary_accuracy_fraction": circuit_set.unitary_accuracy_fraction,
            "error_budget_mode": circuit_set.error_budget_mode,
            "unitary_builder": circuit_set.unitary_builder_category,
            "energy_correction": circuit_set.energy_correction,
            "requested_seed": circuit_set.requested_seed,
            "root_seed": circuit_set.root_seed,
        }
        return QpeResult.from_energy(
            method=self.name(),
            energy=energy,
            evolution_time=circuit_set.base_time,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_energy(
        theta: float,
        base_time: float,
        total_rounds: int,
        lambda_norm: float,
        final_samples: int,
        *,
        correction: str,
    ) -> float:
        """Map the recovered per-base-time phase to an energy."""
        if correction != "qdrift_tangent":
            return energy_from_rpe_angle(theta, base_time)
        # Apply the qDRIFT tangent de-biasing at the final (largest-time) round,
        # using the unwrapped phase consistent with the principal per-base phase.
        principal = wrap_to_principal(theta)
        final_time = (2**total_rounds) * base_time
        final_phase = principal * (2**total_rounds)
        return qdrift_phase_to_energy(final_phase, final_time, lambda_norm, final_samples)

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "qdk_robust"
