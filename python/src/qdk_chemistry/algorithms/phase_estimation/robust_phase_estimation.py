r"""Robust phase estimation for deterministic and randomized time evolution.

The algorithm samples the Hadamard-test signal
:math:`g(t) = \langle\psi|e^{-iHt}|\psi\rangle` on a geometric time ladder and
refines the eigenphase through robust angle-consistency updates. Randomized
builders use one independently seeded unitary draw per scheduled experiment,
with that same draw shared by the X- and Y-basis circuits.

References:
    Günther, J., Witteveen, F., et al. (2025). Phase estimation with partially
    randomized time evolution. PRX Quantum 7, 020332. arXiv:2503.05647.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.algorithms.circuit_executor.base import CircuitExecutor
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    CircuitExecutorData,
    QpeResult,
    QuantumErrorProfile,
    QubitOperator,
    RobustPhaseEstimationCircuitSet,
    RobustPhaseEstimationExperimentSpec,
    Settings,
)
from qdk_chemistry.utils import Logger

from .base import PhaseEstimation
from .circuit_builder.robust_builder import RobustPhaseEstimationCircuitBuilder

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__: list[str] = ["RobustPhaseEstimation", "RobustPhaseEstimationSettings"]


@dataclass(frozen=True)
class _RpeExecutionResult:
    """Execution results paired with their stable experiment identity."""

    experiment_spec: RobustPhaseEstimationExperimentSpec
    x_result: CircuitExecutorData
    y_result: CircuitExecutorData


def _rpe_angle_update(previous_angle: float, measured_phase: float, round_index: int) -> float:
    """Select the measured-phase alias closest to the previous RPE estimate.

    Args:
        previous_angle: Previous estimate of the phase at the base evolution time, in radians.
        measured_phase: Measured phase at the current round's evolution time, in radians.
        round_index: Nonnegative number of time doublings since the base round.

    Returns:
        The closest alias at the base time; the result need not lie in the principal interval.

    Raises:
        ValueError: If the round index is negative.

    """
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
    """Settings for the robust phase estimation algorithm."""

    def __init__(self) -> None:
        """Initialize the robust QPE circuit builder and executor references."""
        super().__init__()
        self._set_default(
            "qpe_circuit_builder",
            "algorithm_ref",
            AlgorithmRef("qpe_circuit_builder", "qdk_robust"),
            "Robust QPE builder that schedules and constructs X/Y circuit pairs.",
        )
        self._set_default(
            "circuit_executor",
            "algorithm_ref",
            AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
            "Backend used to execute generated Hadamard-test circuits.",
        )


class RobustPhaseEstimation(PhaseEstimation):
    """Robust phase estimation for deterministic and randomized evolution."""

    def __init__(
        self,
        qpe_circuit_builder: AlgorithmRef | None = None,
        circuit_executor: AlgorithmRef | None = None,
    ) -> None:
        """Initialize robust phase estimation orchestration.

        Args:
            qpe_circuit_builder: Optional robust-builder reference, including its nested scheduler configuration.
            circuit_executor: Optional backend reference used to execute all X/Y experiments.

        """
        Logger.trace_entering()
        super().__init__()
        self._settings = RobustPhaseEstimationSettings()
        if qpe_circuit_builder is not None:
            self._settings.set("qpe_circuit_builder", qpe_circuit_builder)
        if circuit_executor is not None:
            self._settings.set("circuit_executor", circuit_executor)

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Schedule, stream, and post-process one robust phase estimation run.

        The builder is scheduled once and its ``iter_build`` path is consumed
        lazily. Calling its eager ``run`` instead would materialize the complete
        circuit list and discard the workload needed for execution metadata.

        Args:
            state_preparation: Circuit preparing the input state for each experiment.
            qubit_hamiltonian: Hamiltonian whose evolution phases are estimated.
            noise: Optional error profile forwarded unchanged to every basis-circuit execution.

        Returns:
            The energy estimate with resolved schedule, error-budget, and seed metadata.

        """
        Logger.trace_entering()
        circuit_builder = self._create_circuit_builder()
        circuit_set = circuit_builder.schedule(state_preparation, qubit_hamiltonian)
        return self._execute_with_builder(circuit_builder, circuit_set, noise=noise)

    def schedule_circuit_set(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Resolve one reproducible workload without constructing circuits.

        Args:
            state_preparation: Circuit to store as the workload's input-state preparation.
            qubit_hamiltonian: Hamiltonian used to resolve the schedule.

        Returns:
            A serializable circuit set that can be reused without drawing new scheduling entropy.

        """
        return self._create_circuit_builder().schedule(state_preparation, qubit_hamiltonian)

    def execute_circuit_set(
        self,
        circuit_set: RobustPhaseEstimationCircuitSet,
        *,
        noise: QuantumErrorProfile | None = None,
    ) -> QpeResult:
        """Build and execute a previously scheduled RPE workload.

        Args:
            circuit_set: Recorded workload to replay with its original round settings and unitary seeds.
            noise: Optional error profile forwarded to every X/Y execution.

        Returns:
            The energy estimate and metadata, including the measurement root seed chosen for this execution.

        Raises:
            TypeError: If the supplied workload is not an RPE circuit set.

        """
        if not isinstance(circuit_set, RobustPhaseEstimationCircuitSet):
            raise TypeError(f"circuit_set must be a RobustPhaseEstimationCircuitSet, got {type(circuit_set)} instead.")
        return self._execute_with_builder(self._create_circuit_builder(), circuit_set, noise=noise)

    def _create_circuit_builder(self) -> RobustPhaseEstimationCircuitBuilder:
        """Create and validate the configured robust QPE circuit builder.

        Returns:
            A fresh robust builder initialized from the algorithm's settings.

        Raises:
            TypeError: If the nested algorithm is not a robust QPE circuit builder.

        """
        circuit_builder = self._create_nested("qpe_circuit_builder")
        if not isinstance(circuit_builder, RobustPhaseEstimationCircuitBuilder):
            raise TypeError(
                "Expected qpe_circuit_builder to be a RobustPhaseEstimationCircuitBuilder, "
                f"got {type(circuit_builder)} instead."
            )
        return circuit_builder

    def _execute_with_builder(
        self,
        circuit_builder: RobustPhaseEstimationCircuitBuilder,
        circuit_set: RobustPhaseEstimationCircuitSet,
        *,
        noise: QuantumErrorProfile | None,
    ) -> QpeResult:
        """Stream one workload through execution and post-processing.

        Args:
            circuit_builder: Builder whose shared ``iter_build`` implementation constructs each pair on demand.
            circuit_set: Previously scheduled workload; no scheduling is repeated.
            noise: Optional profile forwarded to the executor.

        Returns:
            The reconstructed energy and execution metadata.

        """
        Logger.info(
            f"RobustPhaseEstimation: lambda={circuit_set.lambda_norm:.6g}, "
            f"base_time={circuit_set.base_time:.6g}, rounds={circuit_set.num_rounds}, "
            f"builder={circuit_set.unitary_builder_category}, correction={circuit_set.energy_correction}, "
            f"eps_rpe={circuit_set.epsilon_rpe:.3g}, eps_unitary={circuit_set.epsilon_unitary:.3g}."
        )
        execution_results, requested_executor_seed, executor_root_seed = self._execute_experiments(
            circuit_builder.iter_build(circuit_set),
            noise=noise,
        )
        return self._post_process(
            circuit_set,
            execution_results,
            requested_executor_seed=requested_executor_seed,
            executor_root_seed=executor_root_seed,
        )

    def _execute_experiments(
        self,
        experiments: Iterator[tuple[RobustPhaseEstimationExperimentSpec, Circuit, Circuit]],
        *,
        noise: QuantumErrorProfile | None,
    ) -> tuple[tuple[_RpeExecutionResult, ...], int | None, int | None]:
        """Execute streamed X/Y circuit pairs while preserving experiment identities.

        Args:
            experiments: Ordered specification/X-circuit/Y-circuit triples produced lazily by the builder.
            noise: Optional error profile passed to each execution.

        Returns:
            Paired results, the requested executor seed, and its resolved root; unsupported seeds remain ``None``.

        """
        executor_ref = self._settings.get("circuit_executor")
        requested_executor_seed = None
        executor_root_seed = None
        if executor_ref.settings is not None and executor_ref.settings.has("seed"):
            requested_executor_seed = int(executor_ref.settings.get("seed"))
            executor_root_seed = (
                requested_executor_seed
                if requested_executor_seed >= 0
                else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
            )
        shared_executor = self._create_executor(None) if executor_root_seed is None else None
        execution_results: list[_RpeExecutionResult] = []
        for experiment_spec, x_circuit, y_circuit in experiments:
            measurement_seeds: list[int | None] = [None, None]
            if executor_root_seed is not None:
                draw_component = 0 if experiment_spec.draw_index is None else experiment_spec.draw_index + 1
                for basis_index in (0, 1):
                    sequence = np.random.SeedSequence(
                        [executor_root_seed, experiment_spec.round_index, draw_component, basis_index]
                    )
                    measurement_seeds[basis_index] = int(sequence.generate_state(1, dtype=np.uint32)[0])
            real_executor = (
                shared_executor if shared_executor is not None else self._create_executor(measurement_seeds[0])
            )
            imag_executor = (
                shared_executor if shared_executor is not None else self._create_executor(measurement_seeds[1])
            )
            execution_results.append(
                _RpeExecutionResult(
                    experiment_spec=experiment_spec,
                    x_result=real_executor.run(x_circuit, shots=experiment_spec.shots, noise=noise),
                    y_result=imag_executor.run(y_circuit, shots=experiment_spec.shots, noise=noise),
                )
            )
        return tuple(execution_results), requested_executor_seed, executor_root_seed

    def _post_process(
        self,
        circuit_set: RobustPhaseEstimationCircuitSet,
        execution_results: tuple[_RpeExecutionResult, ...],
        *,
        requested_executor_seed: int | None,
        executor_root_seed: int | None,
    ) -> QpeResult:
        """Reconstruct the round signals and resolve the final energy.

        Each draw contributes equally to the round's X/Y expectations, even if
        returned count totals differ. Empty counts contribute zero.

        Args:
            circuit_set: Schedule defining round times, experiment identities, and the energy correction.
            execution_results: X/Y counts associated with their experiment specifications, in any order.
            requested_executor_seed: Original measurement-seed setting, or ``None`` when unsupported.
            executor_root_seed: Concrete measurement root used for this execution, or ``None`` when unsupported.

        Returns:
            An energy estimate with the resolved workload and measurement-seed metadata.

        Raises:
            RuntimeError: If a round has a different number of results than its manifest specifies.

        """
        theta = 0.0
        for round_data in circuit_set.rounds:
            round_results = tuple(
                result for result in execution_results if result.experiment_spec.round_index == round_data.round_index
            )
            if len(round_results) != round_data.num_draws:
                raise RuntimeError(
                    f"Round {round_data.round_index} expected {round_data.num_draws} execution results, "
                    f"received {len(round_results)}."
                )
            signal_sums = [0.0, 0.0]
            for result in round_results:
                for basis_index, execution_data in enumerate((result.x_result, result.y_result)):
                    counts = execution_data.bitstring_counts
                    num_zero = int(counts.get("0", 0))
                    num_one = int(counts.get("1", 0))
                    total = num_zero + num_one
                    signal_sums[basis_index] += (num_zero - num_one) / total if total else 0.0
            real_part, imag_part = (component / float(round_data.num_draws) for component in signal_sums)
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

    def _create_executor(self, seed: int | None) -> CircuitExecutor:
        """Create the configured executor, optionally overriding its seed.

        Args:
            seed: Measurement seed override, or ``None`` to retain the configured executor settings.

        Returns:
            A fresh circuit executor without modifying the source algorithm reference.

        Raises:
            RuntimeError: If a seed override is requested for an executor without a seed setting.
            TypeError: If the nested algorithm is not a circuit executor.

        """
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
        """Map the recovered per-base-time phase to an energy.

        Args:
            theta: Recovered phase at the base evolution time, in radians.
            base_time: Positive base evolution time.
            total_rounds: Final round index, equal to the number of doublings after round zero.
            lambda_norm: Hamiltonian coefficient one-norm used by the qDRIFT correction.
            final_samples: Reference qDRIFT sample count for the final round.
            correction: Resolved mapping; ``"qdrift_tangent"`` uses the tangent correction, otherwise linear.

        Returns:
            The energy estimate obtained from the principal base-time phase.

        Raises:
            ValueError: If the base time is nonpositive or the tangent correction has no final samples.

        """
        if base_time <= 0.0:
            raise ValueError(f"base_time must be positive, received {base_time}.")
        if correction == "qdrift_tangent" and final_samples < 1:
            raise ValueError(f"final_samples must be at least 1, received {final_samples}.")
        principal = float((theta + np.pi) % (2 * np.pi) - np.pi)
        if correction != "qdrift_tangent":
            return -principal / base_time
        final_time = (2**total_rounds) * base_time
        final_phase = principal * (2**total_rounds)
        step_angle = lambda_norm * final_time / final_samples
        denominator = np.tan(step_angle)
        if lambda_norm == 0.0 or abs(denominator) < 1e-12:
            return -final_phase / final_time
        return float(-lambda_norm * np.tan(final_phase / final_samples) / denominator)

    def name(self) -> str:
        """Return the robust phase estimation algorithm name.

        Returns:
            ``"qdk_robust"``.

        """
        return "qdk_robust"
