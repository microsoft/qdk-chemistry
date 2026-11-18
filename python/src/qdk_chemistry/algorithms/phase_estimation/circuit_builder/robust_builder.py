"""Circuit construction for robust phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitOperator,
    RobustPhaseEstimationExperimentSpec,
    RobustPhaseEstimationSchedule,
    Settings,
)

from ..rpe_experiment_scheduler import RobustPhaseEstimationExperimentScheduler, _validate_unit_power
from .base import QpeCircuitBuilder

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "RobustPhaseEstimationCircuitBuilder",
    "RobustPhaseEstimationCircuitBuilderSettings",
]


class RobustPhaseEstimationCircuitBuilderSettings(Settings):
    """Settings for robust phase estimation circuit construction."""

    def __init__(self) -> None:
        """Initialize the nested RPE experiment scheduler."""
        super().__init__()
        self._set_default(
            "experiment_scheduler",
            "algorithm_ref",
            AlgorithmRef("rpe_experiment_scheduler", "qdk"),
            "Scheduler that resolves the reproducible RPE circuit workload.",
        )


class RobustPhaseEstimationCircuitBuilder(QpeCircuitBuilder):
    """QDK circuit builder for robust phase estimation.

    ``run`` schedules once and returns the complete circuit list through
    ``_run_impl``. Streaming supports bounded-memory execution and replay.
    Override ``iter_build`` to customize circuit construction for both paths.
    """

    def __init__(self, experiment_scheduler: AlgorithmRef | None = None) -> None:
        """Initialize robust phase estimation circuit construction.

        Args:
            experiment_scheduler: Optional reference configuring workload scheduling; defaults to the QDK scheduler.

        """
        super().__init__()
        self._settings = RobustPhaseEstimationCircuitBuilderSettings()
        if experiment_scheduler is not None:
            self._settings.set("experiment_scheduler", experiment_scheduler)

    def iter_build(
        self,
        schedule: RobustPhaseEstimationSchedule,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> Iterator[tuple[RobustPhaseEstimationExperimentSpec, Circuit, Circuit]]:
        """Build scheduled X/Y circuit pairs one at a time.

        This is the shared construction path for eager ``run`` and
        streamed RPE execution. Reiteration uses the same recorded draw seeds.

        Args:
            schedule: Canonical evolution parameters and draw seeds.
            state_preparation: Live circuit used to prepare each experiment.
            qubit_hamiltonian: Hamiltonian matching the schedule fingerprint.

        Yields:
            Each experiment specification followed by its X and Y circuits, built from the same unitary draw.

        Raises:
            TypeError: If the schedule or inputs have the wrong type, or a declared power is not an integer.
            ValueError: If the input identity, nested power, or executed sample count disagrees with the schedule.

        """
        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        if not isinstance(schedule, RobustPhaseEstimationSchedule):
            raise TypeError("schedule must be a RobustPhaseEstimationSchedule.")
        if not isinstance(state_preparation, Circuit) or not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("Construction requires a Circuit and a QubitOperator.")
        hamiltonian = schedule.validate_and_copy_hamiltonian(qubit_hamiltonian)
        unitary_configuration = schedule.unitary_builder_configuration
        _validate_unit_power(unitary_configuration.settings)
        unitary_settings = unitary_configuration.settings.to_dict()
        hadamard_configuration = schedule.hadamard_test_circuit_builder_configuration
        hadamard_settings = hadamard_configuration.settings.to_dict()
        x_builder = create(
            hadamard_configuration.algorithm_type,
            hadamard_configuration.algorithm_name,
            **(hadamard_settings | {"test_basis": "X"}),
        )
        y_builder = create(
            hadamard_configuration.algorithm_type,
            hadamard_configuration.algorithm_name,
            **(hadamard_settings | {"test_basis": "Y"}),
        )
        for experiment_spec in schedule.experiment_specs:
            round_data = schedule.rounds[experiment_spec.round_index]
            updates: dict[str, object] = {"time": round_data.evolution_time}
            draw_index = experiment_spec.draw_index if experiment_spec.draw_index is not None else 0
            seed = round_data.draw_seeds[draw_index]
            if seed is not None:
                updates["seed"] = seed
            if schedule.unitary_builder_category == "qdrift":
                updates["num_samples"] = round_data.scheduled_samples
            unitary_builder = create(
                unitary_configuration.algorithm_type,
                unitary_configuration.algorithm_name,
                **(unitary_settings | updates),
            )
            if schedule.unitary_builder_category == "qdrift":
                sample_resolver = getattr(unitary_builder, "_resolve_num_samples", None)
                if (
                    callable(sample_resolver)
                    and sample_resolver(hamiltonian, round_data.evolution_time) != round_data.scheduled_samples
                ):
                    raise ValueError("Executed qDRIFT sample count must match the schedule.")
            unitary = unitary_builder.run(hamiltonian)
            x_circuit = x_builder.run(state_preparation, unitary)
            y_circuit = y_builder.run(state_preparation, unitary)
            yield experiment_spec, x_circuit, y_circuit

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        """Schedule once and return the canonical flat circuit list.

        Args:
            state_preparation: Circuit preparing the input state for each Hadamard test.
            qubit_hamiltonian: Hamiltonian used to schedule and construct the evolution circuits.

        Returns:
            All scheduled X/Y pairs flattened in manifest order; use ``iter_build`` for lazy construction.

        Raises:
            TypeError: If the configured scheduler has the wrong algorithm type.

        """
        scheduler = self._create_nested("experiment_scheduler")
        if not isinstance(scheduler, RobustPhaseEstimationExperimentScheduler):
            raise TypeError(
                "Expected experiment_scheduler to be a RobustPhaseEstimationExperimentScheduler, "
                f"got {type(scheduler)} instead."
            )
        schedule = scheduler.run(qubit_hamiltonian)
        circuits: list[Circuit] = []
        for _, x_circuit, y_circuit in self.iter_build(schedule, state_preparation, qubit_hamiltonian):
            circuits.extend((x_circuit, y_circuit))
        return circuits

    def name(self) -> str:
        """Return the QDK robust circuit-builder name.

        Returns:
            ``"qdk_robust"``.

        """
        return "qdk_robust"
