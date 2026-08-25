"""Circuit construction for robust phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitOperator,
    RobustPhaseEstimationCircuitSet,
    RobustPhaseEstimationExperimentSpec,
    Settings,
)
from qdk_chemistry.data.robust_phase_estimation import (
    _AlgorithmConfiguration as _AlgorithmSnapshot,
)

from ..experiment_scheduler import RobustPhaseEstimationExperimentScheduler
from .base import QpeCircuitBuilder

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "QdkRobustPhaseEstimationCircuitBuilder",
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
    """Abstract circuit builder for robust phase estimation."""

    def __init__(self, experiment_scheduler: AlgorithmRef | None = None) -> None:
        """Initialize robust phase estimation circuit construction."""
        super().__init__()
        self._settings = RobustPhaseEstimationCircuitBuilderSettings()
        if experiment_scheduler is not None:
            self._settings.set("experiment_scheduler", experiment_scheduler)

    def schedule(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Resolve one reproducible RPE workload."""
        scheduler = self._create_nested("experiment_scheduler")
        if not isinstance(scheduler, RobustPhaseEstimationExperimentScheduler):
            raise TypeError(
                "Expected experiment_scheduler to be a RobustPhaseEstimationExperimentScheduler, "
                f"got {type(scheduler)} instead."
            )
        return scheduler.run(state_preparation, qubit_hamiltonian)

    def iter_build(
        self,
        circuit_set: RobustPhaseEstimationCircuitSet,
    ) -> Iterator[tuple[RobustPhaseEstimationExperimentSpec, Circuit, Circuit]]:
        """Build scheduled X/Y circuit pairs one at a time."""
        if not isinstance(circuit_set, RobustPhaseEstimationCircuitSet):
            raise TypeError(f"circuit_set must be a RobustPhaseEstimationCircuitSet, got {type(circuit_set)} instead.")
        hadamard_configuration = _AlgorithmSnapshot.from_ref(circuit_set.hadamard_test_circuit_builder_configuration)
        for experiment_spec in circuit_set.experiment_specs:
            round_data = circuit_set.rounds[experiment_spec.round_index]
            unitary_configuration = _AlgorithmSnapshot.from_ref(round_data.unitary_builder_configuration)
            unitary_configuration.validate_unit_power()
            if experiment_spec.draw_seed is not None and unitary_configuration.has_setting("seed"):
                unitary_configuration = unitary_configuration.with_updates(seed=experiment_spec.draw_seed)
            unitary = unitary_configuration.create().run(circuit_set.qubit_hamiltonian)
            x_circuit = (
                hadamard_configuration.with_updates(test_basis="X")
                .create()
                .run(
                    circuit_set.state_preparation,
                    unitary,
                )
            )
            y_circuit = (
                hadamard_configuration.with_updates(test_basis="Y")
                .create()
                .run(
                    circuit_set.state_preparation,
                    unitary,
                )
            )
            yield experiment_spec, x_circuit, y_circuit

    def build(self, circuit_set: RobustPhaseEstimationCircuitSet) -> list[Circuit]:
        """Materialize the canonical flat circuit list for one RPE workload."""
        circuits: list[Circuit] = []
        for _, x_circuit, y_circuit in self.iter_build(circuit_set):
            circuits.extend((x_circuit, y_circuit))
        return circuits

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        """Schedule and build robust phase estimation circuits."""


class QdkRobustPhaseEstimationCircuitBuilder(RobustPhaseEstimationCircuitBuilder):
    """QDK implementation of robust phase estimation circuit construction."""

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> list[Circuit]:
        """Schedule once and return the canonical flat circuit list."""
        return self.build(self.schedule(state_preparation, qubit_hamiltonian))

    def name(self) -> str:
        """Return the QDK robust circuit-builder name."""
        return "qdk_robust"
