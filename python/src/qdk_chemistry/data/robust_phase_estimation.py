"""Serializable data structures for robust phase estimation circuit workloads."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import h5py

from qdk_chemistry._core.data import AlgorithmRef, Settings
from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.circuit import Circuit

__all__ = [
    "RobustPhaseEstimationExperiment",
    "RobustPhaseEstimationRound",
    "RobustPhaseEstimationSchedule",
]


@dataclass(frozen=True)
class _AlgorithmConfiguration:
    """Immutable serialized algorithm reference."""

    algorithm_type: str
    """Registry type of the snapshotted algorithm."""

    algorithm_name: str
    """Registered implementation name of the snapshotted algorithm."""

    settings_json: str
    """Serialized settings used to reconstruct independent algorithm references."""

    @classmethod
    def from_ref(cls, ref: AlgorithmRef) -> _AlgorithmConfiguration:
        """Snapshot a resolved algorithm reference."""
        if ref.settings is None:
            raise ValueError(
                f"Cannot snapshot unresolved algorithm reference '{ref.algorithm_type}/{ref.algorithm_name}'."
            )
        return cls(ref.algorithm_type, ref.algorithm_name, ref.settings.to_json())

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _AlgorithmConfiguration:
        """Restore a serialized algorithm reference."""
        return cls(
            algorithm_type=str(data["algorithm_type"]),
            algorithm_name=str(data["algorithm_name"]),
            settings_json=str(data["settings_json"]),
        )

    def to_ref(self) -> AlgorithmRef:
        """Return an independent algorithm reference."""
        return AlgorithmRef(
            self.algorithm_type,
            self.algorithm_name,
            settings=Settings.from_json(self.settings_json),
        )

    def to_json(self) -> dict[str, str]:
        """Return a JSON-safe representation."""
        return {
            "algorithm_type": self.algorithm_type,
            "algorithm_name": self.algorithm_name,
            "settings_json": self.settings_json,
        }


def _write_json_payload(group: h5py.Group, data: dict[str, Any]) -> None:
    """Write a JSON payload to an HDF5 group."""
    group.create_dataset("payload", data=json.dumps(data), dtype=h5py.string_dtype(encoding="utf-8"))


def _read_json_payload(group: h5py.Group) -> dict[str, Any]:
    """Read a JSON payload from an HDF5 group."""
    return json.loads(group["payload"].asstr()[()])


class RobustPhaseEstimationRound(DataClass):
    """Serializable circuit-generation metadata for one RPE round.

    Attributes:
        round_index: Zero-based position in the geometric RPE schedule.
        evolution_time: Evolution time assigned to the round.
        shots_per_basis: Number of measurement shots scheduled for each X/Y basis.
        num_draws: Number of independent unitary circuit draws generated for the round.
        scheduled_samples: Unitary sample count assigned by the RPE schedule.
        circuit_multiplicity: Number of executions represented by each generated circuit.
        draw_seeds: Concrete random seeds, or an empty tuple for deterministic evolution.

    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for RPE round metadata."""
        return "robust_phase_estimation_round"

    _serialization_version = "0.1.0"

    def __init__(
        self,
        *,
        round_index: int,
        evolution_time: float,
        shots_per_basis: int,
        num_draws: int,
        scheduled_samples: int,
        circuit_multiplicity: int,
        draw_seeds: tuple[int, ...],
        unitary_builder_configuration: AlgorithmRef,
    ) -> None:
        """Initialize one round of an RPE circuit schedule."""
        self.round_index = int(round_index)
        self.evolution_time = float(evolution_time)
        self.shots_per_basis = int(shots_per_basis)
        self.num_draws = int(num_draws)
        self.scheduled_samples = int(scheduled_samples)
        self.circuit_multiplicity = int(circuit_multiplicity)
        self.draw_seeds = tuple(int(seed) for seed in draw_seeds)
        self._unitary_builder_configuration = _AlgorithmConfiguration.from_ref(unitary_builder_configuration)
        super().__init__()

    @property
    def unitary_builder_configuration(self) -> AlgorithmRef:
        """Return an independent copy of the per-round unitary-builder configuration."""
        return self._unitary_builder_configuration.to_ref()

    def _payload(self) -> dict[str, Any]:
        """Return identifying data without the serialization version."""
        return {
            "round_index": self.round_index,
            "evolution_time": self.evolution_time,
            "shots_per_basis": self.shots_per_basis,
            "num_draws": self.num_draws,
            "scheduled_samples": self.scheduled_samples,
            "circuit_multiplicity": self.circuit_multiplicity,
            "draw_seeds": list(self.draw_seeds),
            "unitary_builder_configuration": self._unitary_builder_configuration.to_json(),
        }

    def _hash_update(self, h) -> None:
        """Feed identifying round data into the hasher."""
        _hash_str(h, self.data_type_name())
        _hash_arg(h, self._payload())

    def get_summary(self) -> str:
        """Return a human-readable round summary."""
        return (
            f"Robust phase estimation round {self.round_index}: "
            f"time={self.evolution_time:.6g}, draws={self.num_draws}, shots/basis={self.shots_per_basis}"
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe round representation."""
        return self._add_json_version(self._payload())

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the round to an HDF5 group."""
        self._add_hdf5_version(group)
        _write_json_payload(group, self._payload())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> RobustPhaseEstimationRound:
        """Restore a round from JSON data."""
        cls._validate_json_version(cls._serialization_version, json_data)
        configuration = _AlgorithmConfiguration.from_json(json_data["unitary_builder_configuration"])
        return cls(
            round_index=json_data["round_index"],
            evolution_time=json_data["evolution_time"],
            shots_per_basis=json_data["shots_per_basis"],
            num_draws=json_data["num_draws"],
            scheduled_samples=json_data["scheduled_samples"],
            circuit_multiplicity=json_data["circuit_multiplicity"],
            draw_seeds=tuple(json_data["draw_seeds"]),
            unitary_builder_configuration=configuration.to_ref(),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RobustPhaseEstimationRound:
        """Restore a round from an HDF5 group."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        payload = _read_json_payload(group)
        payload["version"] = cls._serialization_version
        return cls.from_json(payload)


class RobustPhaseEstimationExperiment(DataClass):
    """Serializable materialized X/Y Hadamard-test circuit pair and metadata.

    Attributes:
        round_index: Zero-based index of the round that produced the pair.
        evolution_time: Evolution time implemented by both circuits.
        shots_per_basis: Total shots scheduled per basis for the round.
        draw_index: Zero-based randomized draw index, or ``None`` for deterministic evolution.
        draw_seed: Random seed used for the unitary draw, or ``None`` for deterministic evolution.
        circuit_multiplicity: Number of executions represented by each circuit.
        x_circuit: Hadamard-test circuit measured in the X basis.
        y_circuit: Hadamard-test circuit measured in the Y basis.

    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for materialized RPE experiments."""
        return "robust_phase_estimation_experiment"

    _serialization_version = "0.1.0"

    def __init__(
        self,
        *,
        round_index: int,
        evolution_time: float,
        shots_per_basis: int,
        draw_index: int | None,
        draw_seed: int | None,
        circuit_multiplicity: int,
        x_circuit: Circuit,
        y_circuit: Circuit,
        unitary_builder_configuration: AlgorithmRef,
    ) -> None:
        """Initialize one materialized RPE experiment."""
        self.round_index = int(round_index)
        self.evolution_time = float(evolution_time)
        self.shots_per_basis = int(shots_per_basis)
        self.draw_index = int(draw_index) if draw_index is not None else None
        self.draw_seed = int(draw_seed) if draw_seed is not None else None
        self.circuit_multiplicity = int(circuit_multiplicity)
        self.x_circuit = x_circuit
        self.y_circuit = y_circuit
        self._unitary_builder_configuration = _AlgorithmConfiguration.from_ref(unitary_builder_configuration)
        super().__init__()

    @property
    def unitary_builder_configuration(self) -> AlgorithmRef:
        """Return an independent copy of the unitary-builder configuration used."""
        return self._unitary_builder_configuration.to_ref()

    def _payload(self) -> dict[str, Any]:
        """Return identifying data without the serialization version."""
        return {
            "round_index": self.round_index,
            "evolution_time": self.evolution_time,
            "shots_per_basis": self.shots_per_basis,
            "draw_index": self.draw_index,
            "draw_seed": self.draw_seed,
            "circuit_multiplicity": self.circuit_multiplicity,
            "x_circuit": self.x_circuit.to_json(),
            "y_circuit": self.y_circuit.to_json(),
            "unitary_builder_configuration": self._unitary_builder_configuration.to_json(),
        }

    def _hash_update(self, h) -> None:
        """Feed identifying experiment data into the hasher."""
        _hash_str(h, self.data_type_name())
        _hash_arg(h, self.round_index)
        _hash_arg(h, self.evolution_time)
        _hash_arg(h, self.shots_per_basis)
        _hash_arg(h, self.draw_index)
        _hash_arg(h, self.draw_seed)
        _hash_arg(h, self.circuit_multiplicity)
        _hash_arg(h, self.x_circuit)
        _hash_arg(h, self.y_circuit)
        _hash_arg(h, self._unitary_builder_configuration.to_json())

    def get_summary(self) -> str:
        """Return a human-readable experiment summary."""
        return (
            f"Robust phase estimation experiment: round={self.round_index}, "
            f"draw={self.draw_index}, multiplicity={self.circuit_multiplicity}"
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe experiment representation."""
        return self._add_json_version(self._payload())

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the experiment to an HDF5 group."""
        self._add_hdf5_version(group)
        _write_json_payload(group, self._payload())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> RobustPhaseEstimationExperiment:
        """Restore an experiment from JSON data."""
        cls._validate_json_version(cls._serialization_version, json_data)
        configuration = _AlgorithmConfiguration.from_json(json_data["unitary_builder_configuration"])
        return cls(
            round_index=json_data["round_index"],
            evolution_time=json_data["evolution_time"],
            shots_per_basis=json_data["shots_per_basis"],
            draw_index=json_data.get("draw_index"),
            draw_seed=json_data.get("draw_seed"),
            circuit_multiplicity=json_data["circuit_multiplicity"],
            x_circuit=Circuit.from_json(json_data["x_circuit"]),
            y_circuit=Circuit.from_json(json_data["y_circuit"]),
            unitary_builder_configuration=configuration.to_ref(),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RobustPhaseEstimationExperiment:
        """Restore an experiment from an HDF5 group."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        payload = _read_json_payload(group)
        payload["version"] = cls._serialization_version
        return cls.from_json(payload)


class RobustPhaseEstimationSchedule(DataClass):
    """Serializable RPE workload recipe without materialized circuits.

    Attributes:
        rounds: Resolved round schedule in execution order.
        lambda_norm: One-norm of the qubit Hamiltonian coefficients.
        base_time: Evolution time used for round zero.
        target_accuracy: Requested absolute accuracy of the final energy estimate.
        epsilon_rpe: Energy tolerance used to determine the number of RPE rounds.
        epsilon_unitary: Resolved accuracy parameter supplied to the unitary builder.
        unitary_accuracy_fraction: Resolved legacy non-Trotter accuracy-budget fraction.
        error_budget_mode: Mode used to resolve RPE and unitary accuracy parameters.
        unitary_builder_category: Category of the configured unitary builder.
        energy_correction: Resolved phase-to-energy conversion method.
        requested_seed: Root seed requested in the builder settings.
        root_seed: Concrete root seed for randomized draws, or ``None`` for deterministic evolution.

    """

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for RPE schedules."""
        return "robust_phase_estimation_schedule"

    _serialization_version = "0.1.0"

    def __init__(
        self,
        *,
        rounds: tuple[RobustPhaseEstimationRound, ...],
        lambda_norm: float,
        base_time: float,
        target_accuracy: float,
        epsilon_rpe: float,
        epsilon_unitary: float,
        unitary_accuracy_fraction: float,
        error_budget_mode: str,
        unitary_builder_category: str,
        energy_correction: str,
        requested_seed: int,
        root_seed: int | None,
        hadamard_test_circuit_builder_configuration: AlgorithmRef,
    ) -> None:
        """Initialize a serializable RPE schedule."""
        self.rounds = tuple(rounds)
        self.lambda_norm = float(lambda_norm)
        self.base_time = float(base_time)
        self.target_accuracy = float(target_accuracy)
        self.epsilon_rpe = float(epsilon_rpe)
        self.epsilon_unitary = float(epsilon_unitary)
        self.unitary_accuracy_fraction = float(unitary_accuracy_fraction)
        self.error_budget_mode = str(error_budget_mode)
        self.unitary_builder_category = str(unitary_builder_category)
        self.energy_correction = str(energy_correction)
        self.requested_seed = int(requested_seed)
        self.root_seed = int(root_seed) if root_seed is not None else None
        self._hadamard_test_circuit_builder_configuration = _AlgorithmConfiguration.from_ref(
            hadamard_test_circuit_builder_configuration
        )
        super().__init__()

    @property
    def num_rounds(self) -> int:
        """Return the number of RPE rounds."""
        return len(self.rounds)

    @property
    def final_samples(self) -> int:
        """Return the unitary sample count scheduled for the final round."""
        return self.rounds[-1].scheduled_samples if self.rounds else 1

    @property
    def hadamard_test_circuit_builder_configuration(self) -> AlgorithmRef:
        """Return an independent Hadamard-test circuit-builder configuration."""
        return self._hadamard_test_circuit_builder_configuration.to_ref()

    def _payload(self) -> dict[str, Any]:
        """Return identifying data without the serialization version."""
        return {
            "rounds": [round_data.to_json() for round_data in self.rounds],
            "lambda_norm": self.lambda_norm,
            "base_time": self.base_time,
            "target_accuracy": self.target_accuracy,
            "epsilon_rpe": self.epsilon_rpe,
            "epsilon_unitary": self.epsilon_unitary,
            "unitary_accuracy_fraction": self.unitary_accuracy_fraction,
            "error_budget_mode": self.error_budget_mode,
            "unitary_builder_category": self.unitary_builder_category,
            "energy_correction": self.energy_correction,
            "requested_seed": self.requested_seed,
            "root_seed": self.root_seed,
            "hadamard_test_circuit_builder_configuration": (
                self._hadamard_test_circuit_builder_configuration.to_json()
            ),
        }

    def _hash_update(self, h) -> None:
        """Feed identifying schedule data into the hasher."""
        _hash_str(h, self.data_type_name())
        _hash_arg(h, self._payload())

    def get_summary(self) -> str:
        """Return a human-readable schedule summary."""
        return (
            f"Robust phase estimation schedule: rounds={self.num_rounds}, "
            f"builder={self.unitary_builder_category}, target_accuracy={self.target_accuracy:.6g}"
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe schedule representation."""
        return self._add_json_version(self._payload())

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the schedule to an HDF5 group."""
        self._add_hdf5_version(group)
        _write_json_payload(group, self._payload())

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> RobustPhaseEstimationSchedule:
        """Restore a schedule from JSON data."""
        cls._validate_json_version(cls._serialization_version, json_data)
        configuration = _AlgorithmConfiguration.from_json(json_data["hadamard_test_circuit_builder_configuration"])
        return cls(
            rounds=tuple(RobustPhaseEstimationRound.from_json(item) for item in json_data["rounds"]),
            lambda_norm=json_data["lambda_norm"],
            base_time=json_data["base_time"],
            target_accuracy=json_data["target_accuracy"],
            epsilon_rpe=json_data["epsilon_rpe"],
            epsilon_unitary=json_data["epsilon_unitary"],
            unitary_accuracy_fraction=json_data["unitary_accuracy_fraction"],
            error_budget_mode=json_data["error_budget_mode"],
            unitary_builder_category=json_data["unitary_builder_category"],
            energy_correction=json_data["energy_correction"],
            requested_seed=json_data["requested_seed"],
            root_seed=json_data.get("root_seed"),
            hadamard_test_circuit_builder_configuration=configuration.to_ref(),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RobustPhaseEstimationSchedule:
        """Restore a schedule from an HDF5 group."""
        cls._validate_hdf5_version(cls._serialization_version, group)
        payload = _read_json_payload(group)
        payload["version"] = cls._serialization_version
        return cls.from_json(payload)
