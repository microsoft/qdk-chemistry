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

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QpeResult,
    QuantumErrorProfile,
    QubitOperator,
    RobustPhaseEstimationCircuitSet,
    Settings,
)
from qdk_chemistry.utils import Logger

from .base import PhaseEstimation
from .circuit_builder.robust_builder import RobustPhaseEstimationCircuitBuilder

__all__: list[str] = ["RobustPhaseEstimation", "RobustPhaseEstimationSettings"]


def _wrap_to_principal(angle: float) -> float:
    """Wrap an angle into the principal interval ``[-pi, pi)``."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _rpe_angle_update(previous_angle: float, measured_phase: float, round_index: int) -> float:
    """Select the measured-phase alias closest to the previous RPE estimate."""
    if round_index < 0:
        raise ValueError(f"round_index must be non-negative, received {round_index}.")
    scale = 2**round_index
    best_candidate = measured_phase / scale
    best_diff = (best_candidate - previous_angle + np.pi) % (2 * np.pi) - np.pi
    best_distance = abs(float(best_diff))
    for alias_index in range(1, scale):
        candidate = (measured_phase + 2 * np.pi * alias_index) / scale
        diff = (candidate - previous_angle + np.pi) % (2 * np.pi) - np.pi
        distance = abs(float(diff))
        if distance < best_distance:
            best_distance = distance
            best_candidate = candidate
    return float(best_candidate)


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
            noise: Optional noise profile applied to every circuit execution.

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
        """Build the public on-demand circuit collection configured for this estimator.

        Args:
            state_preparation: Circuit preparing the trial state on the system qubits.
            qubit_hamiltonian: Qubit Hamiltonian whose eigenenergy is estimated.

        Returns:
            Re-iterable robust phase estimation circuit collection that generates pairs on demand.

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
            circuit_set: On-demand circuit collection returned by an RPE circuit builder.
            noise: Optional noise profile applied to every X/Y circuit execution.

        Returns:
            QpeResult carrying the resolved energy.

        Raises:
            TypeError: If ``circuit_set`` is not a robust phase estimation circuit set.

        """
        if not isinstance(circuit_set, RobustPhaseEstimationCircuitSet):
            raise TypeError(
                f"circuit_set must be an instance of RobustPhaseEstimationCircuitSet, got {type(circuit_set)} instead."
            )
        Logger.info(
            f"RobustPhaseEstimation: lambda={circuit_set.lambda_norm:.6g}, "
            f"base_time={circuit_set.base_time:.6g}, rounds={circuit_set.num_rounds}, "
            f"builder={circuit_set.unitary_builder_category}, correction={circuit_set.energy_correction}, "
            f"eps_rpe={circuit_set.epsilon_rpe:.3g}, eps_unitary={circuit_set.epsilon_unitary:.3g}."
        )

        requested_executor_seed, executor_root_seed = self._resolve_executor_seed_configuration()
        shared_executor = self._create_executor(None) if executor_root_seed is None else None
        theta = 0.0
        for round_data in circuit_set.rounds:
            real_accumulator = 0.0
            imag_accumulator = 0.0
            for experiment in circuit_set.iter_round(round_data.round_index):
                real_seed = self._measurement_seed(
                    executor_root_seed,
                    round_data.round_index,
                    experiment.draw_index,
                    basis_index=0,
                )
                imag_seed = self._measurement_seed(
                    executor_root_seed,
                    round_data.round_index,
                    experiment.draw_index,
                    basis_index=1,
                )
                real_executor = shared_executor if shared_executor is not None else self._create_executor(real_seed)
                imag_executor = shared_executor if shared_executor is not None else self._create_executor(imag_seed)
                real_data = real_executor.run(
                    experiment.x_circuit,
                    shots=experiment.circuit_multiplicity,
                    noise=noise,
                )
                imag_data = imag_executor.run(
                    experiment.y_circuit,
                    shots=experiment.circuit_multiplicity,
                    noise=noise,
                )
                basis_expectations: list[float] = []
                for execution_data in (real_data, imag_data):
                    counts = execution_data.bitstring_counts
                    num_zero = int(counts.get("0", 0))
                    num_one = int(counts.get("1", 0))
                    total = num_zero + num_one
                    basis_expectations.append((num_zero - num_one) / total if total else 0.0)
                real_accumulator += basis_expectations[0]
                imag_accumulator += basis_expectations[1]

            real_part = real_accumulator / float(round_data.num_draws)
            imag_part = imag_accumulator / float(round_data.num_draws)
            measured_phase = float(np.angle(complex(real_part, imag_part)))
            theta = _rpe_angle_update(theta, measured_phase, round_data.round_index)
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
            "requested_executor_seed": requested_executor_seed,
            "executor_root_seed": executor_root_seed,
        }
        return QpeResult.from_energy(
            method=self.name(),
            energy=energy,
            evolution_time=circuit_set.base_time,
            metadata=metadata,
        )

    def _resolve_executor_seed_configuration(self) -> tuple[int | None, int | None]:
        """Return the configured executor seed and the concrete root used for measurement streams."""
        executor_ref = self._settings.get("circuit_executor")
        if executor_ref.settings is None or not executor_ref.settings.has("seed"):
            return None, None
        requested_seed = int(executor_ref.settings.get("seed"))
        if requested_seed >= 0:
            return requested_seed, requested_seed
        root_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        return requested_seed, root_seed

    @staticmethod
    def _measurement_seed(
        root_seed: int | None,
        round_index: int,
        draw_index: int | None,
        *,
        basis_index: int,
    ) -> int | None:
        """Derive an independent reproducible executor seed for one measurement stream."""
        if root_seed is None:
            return None
        draw_component = 0 if draw_index is None else draw_index + 1
        sequence = np.random.SeedSequence([root_seed, round_index, draw_component, basis_index])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def _create_executor(self, seed: int | None) -> CircuitExecutor:
        """Create the configured circuit executor, optionally overriding its seed on an independent copy."""
        if seed is None:
            executor = self._create_nested("circuit_executor")
        else:
            executor_ref = self._settings.get("circuit_executor")
            if executor_ref.settings is None or not executor_ref.settings.has("seed"):
                raise RuntimeError("Cannot override the seed of a circuit executor without a seed setting.")
            settings = Settings.from_json(executor_ref.settings.to_json())
            settings.set("seed", seed)
            from qdk_chemistry.algorithms import create  # noqa: PLC0415

            executor = create(executor_ref.algorithm_type, executor_ref.algorithm_name, **settings.to_dict())
        if not isinstance(executor, CircuitExecutor):
            raise TypeError(f"Expected circuit_executor to be a CircuitExecutor, got {type(executor)} instead.")
        return executor

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
        if base_time <= 0.0:
            raise ValueError(f"base_time must be positive, received {base_time}.")
        if correction != "qdrift_tangent":
            return -_wrap_to_principal(theta) / base_time
        # Apply the qDRIFT tangent de-biasing at the final (largest-time) round,
        # using the unwrapped phase consistent with the principal per-base phase.
        if final_samples < 1:
            raise ValueError(f"final_samples must be at least 1, received {final_samples}.")
        principal = _wrap_to_principal(theta)
        final_time = (2**total_rounds) * base_time
        final_phase = principal * (2**total_rounds)
        step_angle = lambda_norm * final_time / final_samples
        denominator = np.tan(step_angle)
        if lambda_norm == 0.0 or abs(denominator) < 1e-12:
            return -final_phase / final_time
        return float(-lambda_norm * np.tan(final_phase / final_samples) / denominator)

    def name(self) -> str:
        """Return the name of the phase estimation algorithm."""
        return "qdk_robust"
