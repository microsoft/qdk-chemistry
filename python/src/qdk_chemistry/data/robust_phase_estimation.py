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
import numpy as np

from qdk_chemistry._core.data import AlgorithmRef, Settings
from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.qubit_operator import QubitOperator

__all__ = [
    "RobustPhaseEstimationCircuitSet",
    "RobustPhaseEstimationExperimentSpec",
    "RobustPhaseEstimationRound",
]


@dataclass(frozen=True)
class _AlgorithmConfiguration:
    """Immutable serialized algorithm reference."""

    algorithm_type: str
    algorithm_name: str
    settings_json: str

    @classmethod
    def from_ref(cls, ref: AlgorithmRef) -> _AlgorithmConfiguration:
        """Snapshot a resolved algorithm reference.

        Args:
            ref: Algorithm reference whose settings have been resolved.

        Returns:
            An immutable copy of the algorithm identity and serialized settings.

        Raises:
            ValueError: If the reference has no resolved settings.

        """
        if ref.settings is None:
            raise ValueError(
                f"Cannot snapshot unresolved algorithm reference '{ref.algorithm_type}/{ref.algorithm_name}'."
            )
        return cls(ref.algorithm_type, ref.algorithm_name, ref.settings.to_json())

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _AlgorithmConfiguration:
        """Restore a serialized algorithm reference.

        Args:
            data: Algorithm identity and settings previously written by ``to_json``.

        Returns:
            The immutable algorithm configuration.

        """
        return cls(
            algorithm_type=str(data["algorithm_type"]),
            algorithm_name=str(data["algorithm_name"]),
            settings_json=str(data["settings_json"]),
        )

    def to_ref(self) -> AlgorithmRef:
        """Return an independent algorithm reference.

        Returns:
            A resolved reference with a fresh copy of the stored settings.

        """
        return AlgorithmRef(
            self.algorithm_type,
            self.algorithm_name,
            settings=Settings.from_json(self.settings_json),
        )

    def has_setting(self, key: str) -> bool:
        """Check whether the snapshotted settings contain a key.

        Args:
            key: Setting name to look up.

        Returns:
            Whether the setting exists in this configuration.

        """
        return Settings.from_json(self.settings_json).has(key)

    def with_updates(self, **updates: object) -> _AlgorithmConfiguration:
        """Return an independent configuration with selected settings updated.

        Args:
            **updates: Existing setting names and their replacement values.

        Returns:
            A new configuration; the original settings remain unchanged.

        Raises:
            ValueError: If an update names a setting the algorithm does not define.
            RuntimeError: If the restored reference unexpectedly has no settings.

        """
        ref = self.to_ref()
        if ref.settings is None:
            raise RuntimeError("Algorithm configuration unexpectedly has no settings.")
        for key, value in updates.items():
            if not ref.settings.has(key):
                raise ValueError(
                    f"Algorithm '{self.algorithm_type}/{self.algorithm_name}' does not define setting '{key}'."
                )
            ref.settings.set(key, value)
        return self.from_ref(ref)

    def create(self) -> Any:
        """Create a fresh algorithm instance from this configuration.

        Returns:
            The registered algorithm initialized with the snapshotted settings.

        """
        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        settings = Settings.from_json(self.settings_json)
        return create(self.algorithm_type, self.algorithm_name, **settings.to_dict())

    def validate_unit_power(self) -> None:
        """Require RPE to be the sole owner of the evolution power schedule.

        Raises:
            TypeError: If a declared ``power`` is not an integer.
            ValueError: If a declared ``power`` is not one.

        """
        settings = Settings.from_json(self.settings_json)
        if not settings.has("power"):
            return
        power = settings.get("power")
        if not isinstance(power, int):
            raise TypeError(f"unitary_builder power must be an integer, got {type(power).__name__}.")
        if power != 1:
            raise ValueError(
                "Robust phase estimation controls evolution powers through its round-time schedule; "
                f"unitary_builder power must be 1, got {power}."
            )

    def to_json(self) -> dict[str, str]:
        """Return a JSON-safe representation.

        Returns:
            Algorithm identity and serialized settings for nesting in a workload.

        """
        return {
            "algorithm_type": self.algorithm_type,
            "algorithm_name": self.algorithm_name,
            "settings_json": self.settings_json,
        }


@dataclass(frozen=True)
class RobustPhaseEstimationExperimentSpec:
    """Execution count and stable identity for one planned X/Y Hadamard-test pair.

    Attributes:
        experiment_index: Zero-based position of the pair in the full workload.
        round_index: Zero-based RPE round containing this pair.
        draw_index: Randomized draw number within the round, or ``None`` for deterministic evolution.
        draw_seed: Concrete unitary seed shared by X and Y, or ``None`` for deterministic evolution.
        shots: Executions of each basis circuit; randomized draws use one shot per basis.

    """

    experiment_index: int
    round_index: int
    draw_index: int | None
    draw_seed: int | None
    shots: int

    def __post_init__(self) -> None:
        """Validate experiment coordinates and execution count.

        Raises:
            ValueError: If an index is negative, draw metadata is incomplete, or the shot count is nonpositive.

        """
        if self.experiment_index < 0:
            raise ValueError("experiment_index must be non-negative.")
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative.")
        if self.draw_index is not None and self.draw_index < 0:
            raise ValueError("draw_index must be non-negative when provided.")
        if (self.draw_index is None) != (self.draw_seed is None):
            raise ValueError("draw_index and draw_seed must either both be set or both be None.")
        if self.shots < 1:
            raise ValueError("shots must be at least 1.")

    @property
    def x_circuit_index(self) -> int:
        """Return the X-circuit position in the canonical flat circuit list.

        Returns:
            Twice the experiment index, since every pair is ordered X then Y.

        """
        return 2 * self.experiment_index

    @property
    def y_circuit_index(self) -> int:
        """Return the Y-circuit position in the canonical flat circuit list.

        Returns:
            The position immediately after the corresponding X circuit.

        """
        return self.x_circuit_index + 1


@dataclass(frozen=True, init=False)
class RobustPhaseEstimationRound:
    """Read-only time, sample, draw, and execution metadata for one RPE round."""

    round_index: int
    evolution_time: float
    shots_per_basis: int
    num_draws: int
    scheduled_samples: int
    _unitary_builder_configuration: _AlgorithmConfiguration

    def __init__(
        self,
        *,
        round_index: int,
        evolution_time: float,
        shots_per_basis: int,
        num_draws: int,
        scheduled_samples: int,
        unitary_builder_configuration: AlgorithmRef,
    ) -> None:
        """Initialize one round of an RPE circuit schedule.

        Args:
            round_index: Zero-based position in the time-doubling ladder.
            evolution_time: Positive evolution time for this round.
            shots_per_basis: Total executions of the X circuits, and separately of the Y circuits.
            num_draws: Number of X/Y pairs; one for deterministic evolution, one per shot for randomized evolution.
            scheduled_samples: Reference qDRIFT sample count assigned to this round.
            unitary_builder_configuration: Resolved builder settings, copied when the round is created.

        Raises:
            ValueError: If the index is negative, time or counts are nonpositive, or settings are unresolved.

        """
        object.__setattr__(self, "round_index", int(round_index))
        object.__setattr__(self, "evolution_time", float(evolution_time))
        object.__setattr__(self, "shots_per_basis", int(shots_per_basis))
        object.__setattr__(self, "num_draws", int(num_draws))
        object.__setattr__(self, "scheduled_samples", int(scheduled_samples))
        object.__setattr__(
            self,
            "_unitary_builder_configuration",
            _AlgorithmConfiguration.from_ref(unitary_builder_configuration),
        )
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative.")
        if self.evolution_time <= 0.0:
            raise ValueError("evolution_time must be positive.")
        if self.shots_per_basis < 1:
            raise ValueError("shots_per_basis must be at least 1.")
        if self.num_draws < 1:
            raise ValueError("num_draws must be at least 1.")
        if self.scheduled_samples < 1:
            raise ValueError("scheduled_samples must be at least 1.")

    @property
    def unitary_builder_configuration(self) -> AlgorithmRef:
        """Return an independent copy of the per-round unitary-builder configuration.

        Returns:
            A resolved reference whose settings can be changed without changing this round.

        """
        return self._unitary_builder_configuration.to_ref()


class RobustPhaseEstimationCircuitSet(DataClass):
    """Serializable RPE workload and execution manifest."""

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for RPE circuit workloads.

        Returns:
            ``"robust_phase_estimation_circuit_set"``.

        """
        return "robust_phase_estimation_circuit_set"

    _serialization_version = "0.1.0"

    def __init__(
        self,
        *,
        rounds: tuple[RobustPhaseEstimationRound, ...],
        experiment_specs: tuple[RobustPhaseEstimationExperimentSpec, ...],
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
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
        """Initialize a reproducible RPE circuit workload without building circuits.

        Args:
            rounds: Ordered round metadata, including the base round at index zero.
            experiment_specs: Ordered X/Y pair identities and per-basis execution counts.
            state_preparation: Circuit preparing the input state for each experiment.
            qubit_hamiltonian: Hamiltonian to evolve; its coefficients and metadata are copied.
            lambda_norm: Coefficient one-norm used to construct the time ladder.
            base_time: Base evolution time for the time-doubling ladder.
            target_accuracy: Requested final energy accuracy.
            epsilon_rpe: Energy tolerance used to choose the number of rounds.
            epsilon_unitary: Full-unitary error budget assigned by the scheduler.
            unitary_accuracy_fraction: Legacy fractional allocation recorded by the selected budget mode.
            error_budget_mode: Name of the scheduler's resolved error-allocation policy.
            unitary_builder_category: Evolution category used for scheduling and post-processing.
            energy_correction: Resolved linear or qDRIFT-tangent phase-to-energy mapping.
            requested_seed: Requested unitary-draw seed; a negative value requests entropy.
            root_seed: Concrete randomized-draw root, or ``None`` for deterministic evolution.
            hadamard_test_circuit_builder_configuration: Resolved Hadamard-test builder settings to snapshot.

        Raises:
            TypeError: If the preparation or Hamiltonian has the wrong type, or a builder power is not an integer.
            ValueError: If settings are unresolved or the round, power, draw, and shot metadata are inconsistent.

        """
        if not isinstance(state_preparation, Circuit):
            raise TypeError("state_preparation must be a Circuit.")
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError("qubit_hamiltonian must be a QubitOperator.")
        self.rounds = tuple(rounds)
        self.experiment_specs = tuple(experiment_specs)
        self.state_preparation = state_preparation
        self.qubit_hamiltonian = QubitOperator(
            pauli_strings=list(qubit_hamiltonian.pauli_strings),
            coefficients=np.array(qubit_hamiltonian.coefficients, dtype=np.complex128, copy=True),
            encoding=qubit_hamiltonian.encoding,
            fermion_mode_order=qubit_hamiltonian.fermion_mode_order,
            term_partition=qubit_hamiltonian.term_partition,
            tapering=qubit_hamiltonian.tapering,
        )
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
        self._validate_manifest()
        super().__init__()

    @property
    def num_rounds(self) -> int:
        """Return the number of RPE rounds.

        Returns:
            The number of scheduled rounds, including round zero.

        """
        return len(self.rounds)

    @property
    def final_samples(self) -> int:
        """Return the unitary sample count for the final round.

        Returns:
            The reference sample count used by the qDRIFT energy correction.

        """
        return self.rounds[-1].scheduled_samples

    @property
    def hadamard_test_circuit_builder_configuration(self) -> AlgorithmRef:
        """Return an independent Hadamard-test builder configuration.

        Returns:
            A resolved reference with a fresh copy of the stored settings.

        """
        return self._hadamard_test_circuit_builder_configuration.to_ref()

    def experiment_specs_for_round(
        self,
        round_index: int,
    ) -> tuple[RobustPhaseEstimationExperimentSpec, ...]:
        """Return the planned experiments for one round.

        Args:
            round_index: Zero-based round to select.

        Returns:
            That round's experiment specifications in their original order.

        Raises:
            IndexError: If the index is outside the scheduled rounds.

        """
        if round_index < 0 or round_index >= self.num_rounds:
            raise IndexError(f"round_index must be in [0, {self.num_rounds - 1}], got {round_index}.")
        return tuple(spec for spec in self.experiment_specs if spec.round_index == round_index)

    def rebind(
        self,
        state_preparation: Circuit,
    ) -> RobustPhaseEstimationCircuitSet:
        """Return the workload with its live state-preparation circuit rebound.

        Args:
            state_preparation: Replacement preparation, for example a live callable restored after loading metadata.

        Returns:
            A new workload retaining the original schedule, settings, and draw seeds without rescheduling.

        Raises:
            TypeError: If the replacement preparation is not a circuit.

        """
        return type(self)(
            rounds=self.rounds,
            experiment_specs=self.experiment_specs,
            state_preparation=state_preparation,
            qubit_hamiltonian=self.qubit_hamiltonian,
            lambda_norm=self.lambda_norm,
            base_time=self.base_time,
            target_accuracy=self.target_accuracy,
            epsilon_rpe=self.epsilon_rpe,
            epsilon_unitary=self.epsilon_unitary,
            unitary_accuracy_fraction=self.unitary_accuracy_fraction,
            error_budget_mode=self.error_budget_mode,
            unitary_builder_category=self.unitary_builder_category,
            energy_correction=self.energy_correction,
            requested_seed=self.requested_seed,
            root_seed=self.root_seed,
            hadamard_test_circuit_builder_configuration=self.hadamard_test_circuit_builder_configuration,
        )

    def _validate_manifest(self) -> None:
        """Validate round, experiment, and execution-count consistency.

        Raises:
            TypeError: If a declared unitary-builder power is not an integer.
            ValueError: If indices, powers, draw metadata, or shot totals are inconsistent.

        """
        if not self.rounds:
            raise ValueError("rounds must contain at least one RPE round.")
        expected_round_indices = tuple(range(len(self.rounds)))
        round_indices = tuple(round_data.round_index for round_data in self.rounds)
        if round_indices != expected_round_indices:
            raise ValueError("round_index values must be contiguous and match round order.")
        if not self.experiment_specs:
            raise ValueError("experiment_specs must contain at least one X/Y pair.")
        expected_experiment_indices = tuple(range(len(self.experiment_specs)))
        experiment_indices = tuple(spec.experiment_index for spec in self.experiment_specs)
        if experiment_indices != expected_experiment_indices:
            raise ValueError("experiment_index values must be contiguous and match manifest order.")

        for round_data in self.rounds:
            _AlgorithmConfiguration.from_ref(round_data.unitary_builder_configuration).validate_unit_power()
            specs = tuple(spec for spec in self.experiment_specs if spec.round_index == round_data.round_index)
            if len(specs) != round_data.num_draws:
                raise ValueError(
                    f"Round {round_data.round_index} declares {round_data.num_draws} draws but has {len(specs)} specs."
                )
            if sum(spec.shots for spec in specs) != round_data.shots_per_basis:
                raise ValueError(f"Round {round_data.round_index} experiment shots do not sum to shots_per_basis.")
            randomized = specs[0].draw_index is not None
            if randomized:
                if tuple(spec.draw_index for spec in specs) != tuple(range(round_data.num_draws)):
                    raise ValueError(f"Round {round_data.round_index} randomized draw indices must be contiguous.")
                if any(spec.shots != 1 for spec in specs):
                    raise ValueError("Randomized RPE experiments must represent one shot per circuit draw.")
            elif len(specs) != 1:
                raise ValueError("Deterministic RPE rounds must contain exactly one experiment spec.")

        if any(spec.round_index >= self.num_rounds for spec in self.experiment_specs):
            raise ValueError("Every experiment spec must reference an existing round.")

    def _metadata_payload(self) -> dict[str, Any]:
        """Return metadata serialized directly by the circuit set.

        Returns:
            JSON-safe schedule metadata, excluding the preparation and Hamiltonian.

        """
        return {
            "rounds": [
                {
                    "round_index": round_data.round_index,
                    "evolution_time": round_data.evolution_time,
                    "shots_per_basis": round_data.shots_per_basis,
                    "num_draws": round_data.num_draws,
                    "scheduled_samples": round_data.scheduled_samples,
                    "unitary_builder_configuration": _AlgorithmConfiguration.from_ref(
                        round_data.unitary_builder_configuration
                    ).to_json(),
                }
                for round_data in self.rounds
            ],
            "experiment_specs": [
                {
                    "experiment_index": spec.experiment_index,
                    "round_index": spec.round_index,
                    "draw_index": spec.draw_index,
                    "draw_seed": spec.draw_seed,
                    "shots": spec.shots,
                }
                for spec in self.experiment_specs
            ],
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

    def _payload(self) -> dict[str, Any]:
        """Return the complete JSON-safe circuit-set payload.

        Returns:
            Schedule metadata with serialized preparation and Hamiltonian inputs.

        """
        return {
            **self._metadata_payload(),
            "state_preparation": self.state_preparation.to_json(),
            "qubit_hamiltonian": self.qubit_hamiltonian.to_json(),
        }

    def _hash_update(self, h) -> None:
        """Feed identifying circuit-set data into the hasher.

        Args:
            h: Hasher updated with the data type and complete serialized payload.

        """
        _hash_str(h, self.data_type_name())
        _hash_arg(h, self._payload())

    def get_summary(self) -> str:
        """Return a human-readable workload summary.

        Returns:
            The number of rounds and planned X/Y pairs.

        """
        return (
            f"Robust phase estimation circuit set: rounds={self.num_rounds}, experiments={len(self.experiment_specs)}"
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe circuit-set representation.

        Returns:
            The versioned payload, including inputs, settings, and concrete draw seeds.

        """
        return self._add_json_version(self._payload())

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write the circuit set and nested values to HDF5.

        Args:
            group: Destination HDF5 group for metadata and nested input datasets.

        """
        self._add_hdf5_version(group)
        group.create_dataset(
            "payload", data=json.dumps(self._metadata_payload()), dtype=h5py.string_dtype(encoding="utf-8")
        )
        self.state_preparation.to_hdf5(group.create_group("state_preparation"))
        self.qubit_hamiltonian.to_hdf5(group.create_group("qubit_hamiltonian"))

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Restore a circuit set from decoded nested values.

        Args:
            payload: Decoded schedule metadata.
            state_preparation: Restored input-state circuit.
            qubit_hamiltonian: Restored Hamiltonian.

        Returns:
            A validated workload preserving the recorded schedule and seeds.

        """
        hadamard_configuration = _AlgorithmConfiguration.from_json(
            payload["hadamard_test_circuit_builder_configuration"]
        )
        rounds = []
        for data in payload["rounds"]:
            configuration = _AlgorithmConfiguration.from_json(data["unitary_builder_configuration"])
            rounds.append(
                RobustPhaseEstimationRound(
                    round_index=int(data["round_index"]),
                    evolution_time=float(data["evolution_time"]),
                    shots_per_basis=int(data["shots_per_basis"]),
                    num_draws=int(data["num_draws"]),
                    scheduled_samples=int(data["scheduled_samples"]),
                    unitary_builder_configuration=configuration.to_ref(),
                )
            )
        experiment_specs = tuple(
            RobustPhaseEstimationExperimentSpec(
                experiment_index=int(data["experiment_index"]),
                round_index=int(data["round_index"]),
                draw_index=int(data["draw_index"]) if data.get("draw_index") is not None else None,
                draw_seed=int(data["draw_seed"]) if data.get("draw_seed") is not None else None,
                shots=int(data["shots"]),
            )
            for data in payload["experiment_specs"]
        )
        return cls(
            rounds=tuple(rounds),
            experiment_specs=experiment_specs,
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
            lambda_norm=float(payload["lambda_norm"]),
            base_time=float(payload["base_time"]),
            target_accuracy=float(payload["target_accuracy"]),
            epsilon_rpe=float(payload["epsilon_rpe"]),
            epsilon_unitary=float(payload["epsilon_unitary"]),
            unitary_accuracy_fraction=float(payload["unitary_accuracy_fraction"]),
            error_budget_mode=str(payload["error_budget_mode"]),
            unitary_builder_category=str(payload["unitary_builder_category"]),
            energy_correction=str(payload["energy_correction"]),
            requested_seed=int(payload["requested_seed"]),
            root_seed=int(payload["root_seed"]) if payload.get("root_seed") is not None else None,
            hadamard_test_circuit_builder_configuration=hadamard_configuration.to_ref(),
        )

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> RobustPhaseEstimationCircuitSet:
        """Restore a circuit set from JSON data.

        Args:
            json_data: Versioned payload produced by ``to_json``.

        Returns:
            A workload with the original settings and seeds; no circuits are built.

        Raises:
            ValueError: If the serialization version or workload metadata is invalid.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls._from_payload(
            json_data,
            Circuit.from_json(json_data["state_preparation"]),
            QubitOperator.from_json(json_data["qubit_hamiltonian"]),
        )

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RobustPhaseEstimationCircuitSet:
        """Restore a circuit set from HDF5 data.

        Args:
            group: HDF5 group previously populated by ``to_hdf5``.

        Returns:
            A workload with the original settings and seeds; no circuits are built.

        Raises:
            ValueError: If the serialization version or workload metadata is invalid.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls._from_payload(
            json.loads(group["payload"].asstr()[()]),
            Circuit.from_hdf5(group["state_preparation"]),
            QubitOperator.from_hdf5(group["qubit_hamiltonian"]),
        )
