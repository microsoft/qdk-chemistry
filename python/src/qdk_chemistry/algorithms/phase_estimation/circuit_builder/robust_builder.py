"""Circuit construction for robust phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitOperator,
    RobustPhaseEstimationExperimentSpec,
    RobustPhaseEstimationSchedule,
    Settings,
)
from qdk_chemistry.data.robust_phase_estimation import _HamiltonianSnapshot

from ..rpe_experiment_scheduler import RobustPhaseEstimationExperimentScheduler, _AlgorithmSnapshot
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

    ``run`` returns the complete circuit list through ``_run_impl``. Scheduling
    and streaming are also exposed for bounded-memory execution and replay.
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

    def schedule(
        self,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationSchedule:
        """Resolve one reproducible RPE workload without constructing circuits.

        Args:
            qubit_hamiltonian: Hamiltonian whose evolution is scheduled.

        Returns:
            An input-free schedule, including concrete randomized-draw seeds.

        Raises:
            TypeError: If the configured scheduler has the wrong algorithm type.

        """
        scheduler = self._create_nested("experiment_scheduler")
        if not isinstance(scheduler, RobustPhaseEstimationExperimentScheduler):
            raise TypeError(
                "Expected experiment_scheduler to be a RobustPhaseEstimationExperimentScheduler, "
                f"got {type(scheduler)} instead."
            )
        return scheduler.run(qubit_hamiltonian)

    def iter_build(
        self,
        schedule: RobustPhaseEstimationSchedule,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> Iterator[tuple[RobustPhaseEstimationExperimentSpec, Circuit, Circuit]]:
        """Build scheduled X/Y circuit pairs one at a time.

        This is the shared construction path for eager ``run``/``build`` and
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
        if not isinstance(schedule, RobustPhaseEstimationSchedule):
            raise TypeError("schedule must be a RobustPhaseEstimationSchedule.")
        if not isinstance(state_preparation, Circuit) or not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("Construction requires a Circuit and a QubitOperator.")
        snapshot = _HamiltonianSnapshot.from_operator(qubit_hamiltonian)
        if snapshot.content_hash() != schedule.hamiltonian_hash:
            raise ValueError("Hamiltonian does not match the schedule fingerprint.")
        hamiltonian = snapshot.to_operator()
        if not np.isclose(hamiltonian.schatten_norm, schedule.lambda_norm, rtol=1e-12, atol=0.0):
            raise ValueError("Hamiltonian norm does not match the schedule lambda_norm.")
        unitary_configuration = _AlgorithmSnapshot.from_ref(schedule.unitary_builder_configuration)
        unitary_configuration.validate_unit_power()
        hadamard_configuration = _AlgorithmSnapshot.from_ref(schedule.hadamard_test_circuit_builder_configuration)
        x_builder = hadamard_configuration.with_updates(test_basis="X").create()
        y_builder = hadamard_configuration.with_updates(test_basis="Y").create()
        for experiment_spec in schedule.experiment_specs:
            round_data = schedule.rounds[experiment_spec.round_index]
            updates: dict[str, object] = {"time": round_data.evolution_time}
            draw_index = experiment_spec.draw_index if experiment_spec.draw_index is not None else 0
            seed = round_data.draw_seeds[draw_index]
            if seed is not None:
                updates["seed"] = seed
            if schedule.unitary_builder_category == "qdrift":
                updates["num_samples"] = round_data.scheduled_samples
            unitary_builder = unitary_configuration.create(**updates)
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

    def build(
        self, schedule: RobustPhaseEstimationSchedule, state_preparation: Circuit, qubit_hamiltonian: QubitOperator
    ) -> list[Circuit]:
        """Materialize the canonical flat circuit list for one RPE workload.

        Args:
            schedule: Previously resolved schedule to build without rescheduling.
            state_preparation: Live input-state circuit.
            qubit_hamiltonian: Hamiltonian matching the schedule.

        Returns:
            All circuits in manifest order, with X immediately followed by Y for each experiment.

        """
        circuits: list[Circuit] = []
        for _, x_circuit, y_circuit in self.iter_build(schedule, state_preparation, qubit_hamiltonian):
            circuits.extend((x_circuit, y_circuit))
        return circuits

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

        """
        return self.build(self.schedule(qubit_hamiltonian), state_preparation, qubit_hamiltonian)

    def name(self) -> str:
        """Return the QDK robust circuit-builder name.

        Returns:
            ``"qdk_robust"``.

        """
        return "qdk_robust"
