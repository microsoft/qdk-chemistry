"""Serializable data structures for robust phase estimation circuit workloads."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

import qdk_chemistry.data
from qdk_chemistry._core.data import AlgorithmRef, Settings
from qdk_chemistry.data._hashing import _hash_arg, _hash_str
from qdk_chemistry.data.base import DataClass
from qdk_chemistry.data.circuit import Circuit
from qdk_chemistry.data.qubit_operator import QubitOperator

__all__ = [
    "RobustPhaseEstimationCircuitSet",
    "RobustPhaseEstimationExperimentSpec",
    "RobustPhaseEstimationRound",
    "RobustPhaseEstimationSchedule",
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


@dataclass(frozen=True)
class RobustPhaseEstimationRound:
    """Canonical evolution and execution parameters for one RPE round.

    Attributes:
        round_index: Zero-based position in the time-doubling ladder.
        evolution_time: Finite positive evolution time, stored only here.
        shots_per_basis: Total executions of each basis in this round.
        scheduled_samples: Resolved qDRIFT sample count, or reference count for other families.
        draw_seeds: One seed per unitary draw, with a single optional seed for deterministic evolution.

    """

    round_index: int
    evolution_time: float
    shots_per_basis: int
    scheduled_samples: int
    draw_seeds: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the canonical round values.

        Raises:
            ValueError: If times, indices, counts, or seeds are invalid.

        """
        for name, minimum in (("round_index", 0), ("shots_per_basis", 1), ("scheduled_samples", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}.")
        if not np.isfinite(self.evolution_time) or self.evolution_time <= 0.0:
            raise ValueError("evolution_time must be finite and positive.")
        object.__setattr__(self, "evolution_time", float(self.evolution_time))
        object.__setattr__(self, "draw_seeds", tuple(self.draw_seeds))
        if not self.draw_seeds:
            raise ValueError("draw_seeds must contain at least one draw.")
        if any(
            seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0)
            for seed in self.draw_seeds
        ):
            raise ValueError("Draw seeds must be nonnegative integers or None.")

    @property
    def num_draws(self) -> int:
        """Return the number of recorded unitary draws.

        Returns:
            The number of entries in the immutable seed tuple.

        """
        return len(self.draw_seeds)


@dataclass(frozen=True)
class _HamiltonianSnapshot:
    """Canonical immutable serialization of replay input data."""

    payload_json: str

    @classmethod
    def from_operator(cls, operator: QubitOperator) -> _HamiltonianSnapshot:
        """Snapshot coefficients, labels, and metadata without retaining mutable input data.

        Args:
            operator: Hamiltonian to snapshot.

        Returns:
            Canonical serialized input data.

        """
        snapshot = QubitOperator(
            pauli_strings=list(operator.pauli_strings),
            coefficients=np.array(operator.coefficients, dtype=np.complex128, copy=True),
            encoding=operator.encoding,
            fermion_mode_order=operator.fermion_mode_order,
            term_partition=operator.term_partition,
            tapering=operator.tapering,
        )
        return cls(json.dumps(snapshot.to_json(), sort_keys=True, allow_nan=False))

    def to_operator(self) -> QubitOperator:
        """Restore an independent Hamiltonian.

        Returns:
            A fresh operator whose nested values do not alias the snapshot.

        """
        return QubitOperator.from_json(json.loads(self.payload_json))

    def content_hash(self) -> str:
        """Return the complete fingerprint of the canonical input.

        Returns:
            A hexadecimal SHA-256 fingerprint.

        """
        return hashlib.sha256(self.payload_json.encode()).hexdigest()


class RobustPhaseEstimationSchedule(DataClass):
    """Immutable RPE experiment parameters, independent of live circuit inputs."""

    @staticmethod
    def data_type_name() -> str:
        """Return the wire-format identifier for an RPE schedule.

        Returns:
            ``"robust_phase_estimation_schedule"``.

        """
        return "robust_phase_estimation_schedule"

    _serialization_version = "0.1.0"

    def __init__(
        self,
        *,
        rounds: tuple[RobustPhaseEstimationRound, ...],
        hamiltonian_hash: str,
        lambda_norm: float,
        target_accuracy: float,
        epsilon_rpe: float,
        epsilon_unitary: float,
        unitary_accuracy_fraction: float,
        error_budget_mode: str,
        unitary_builder_category: str,
        energy_correction: str,
        requested_seed: int,
        root_seed: int | None,
        unitary_builder_configuration: AlgorithmRef,
        hadamard_test_circuit_builder_configuration: AlgorithmRef,
    ) -> None:
        """Store the resolved experiment schedule without live input data or algorithms.

        Args:
            rounds: Canonical times, shot counts, sample counts, and seeds for each round.
            hamiltonian_hash: Fingerprint tying the schedule to its Hamiltonian input.
            lambda_norm: Coefficient one-norm used to construct the time ladder.
            target_accuracy: Requested final energy accuracy.
            epsilon_rpe: Energy tolerance used to choose the number of rounds.
            epsilon_unitary: Full-unitary error budget assigned by the scheduler.
            unitary_accuracy_fraction: Legacy fractional allocation recorded by the selected budget mode.
            error_budget_mode: Name of the scheduler's resolved error-allocation policy.
            unitary_builder_category: Evolution category used for scheduling and post-processing.
            energy_correction: Resolved linear or qDRIFT-tangent phase-to-energy mapping.
            requested_seed: Requested unitary-draw seed; a negative value requests entropy.
            root_seed: Concrete randomized-draw root, or ``None`` for deterministic evolution.
            unitary_builder_configuration: Shared builder settings, excluding round-owned time, seed, and samples.
            hadamard_test_circuit_builder_configuration: Resolved Hadamard-test builder settings to snapshot.

        Raises:
            ValueError: If settings are unresolved, values are invalid, or the time ladder and draws disagree.

        """
        self.rounds = tuple(rounds)
        self.hamiltonian_hash = str(hamiltonian_hash)
        self.lambda_norm = float(lambda_norm)
        self.target_accuracy = float(target_accuracy)
        self.epsilon_rpe = float(epsilon_rpe)
        self.epsilon_unitary = float(epsilon_unitary)
        self.unitary_accuracy_fraction = float(unitary_accuracy_fraction)
        self.error_budget_mode = str(error_budget_mode)
        self.unitary_builder_category = str(unitary_builder_category)
        self.energy_correction = str(energy_correction)
        self.requested_seed = int(requested_seed)
        self.root_seed = int(root_seed) if root_seed is not None else None
        self._unitary_builder_configuration = _AlgorithmConfiguration.from_ref(unitary_builder_configuration)
        self._hadamard_test_circuit_builder_configuration = _AlgorithmConfiguration.from_ref(
            hadamard_test_circuit_builder_configuration
        )
        self._validate_schedule()
        super().__init__()

    @property
    def base_time(self) -> float:
        """Return the canonical time of round zero.

        Returns:
            The first round's evolution time, never stored separately.

        """
        return self.rounds[0].evolution_time

    @property
    def unitary_builder_configuration(self) -> AlgorithmRef:
        """Return independent shared evolution settings.

        Returns:
            A copied reference without round-specific time, sample count, or seed.

        """
        return self._unitary_builder_configuration.to_ref()

    @property
    def experiment_specs(self) -> tuple[RobustPhaseEstimationExperimentSpec, ...]:
        """Derive the canonical experiment identities and per-draw shots.

        Returns:
            X/Y pair specifications in round and draw order.

        """
        randomized = self.unitary_builder_category in ("qdrift", "partial_randomized")
        specs: list[RobustPhaseEstimationExperimentSpec] = []
        for round_data in self.rounds:
            for draw_index, seed in enumerate(round_data.draw_seeds):
                specs.append(
                    RobustPhaseEstimationExperimentSpec(
                        experiment_index=len(specs),
                        round_index=round_data.round_index,
                        draw_index=draw_index if randomized else None,
                        draw_seed=seed if randomized else None,
                        shots=1 if randomized else round_data.shots_per_basis,
                    )
                )
        return tuple(specs)

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

    def _validate_schedule(self) -> None:
        """Validate canonical times, reconstruction metadata, and draw counts.

        Raises:
            ValueError: If stored values disagree or duplicate round-owned parameters.

        """
        if not self.rounds:
            raise ValueError("rounds must contain at least one RPE round.")
        if len(self.hamiltonian_hash) != 64 or any(char not in "0123456789abcdef" for char in self.hamiltonian_hash):
            raise ValueError("hamiltonian_hash must be a SHA-256 fingerprint.")
        for field in ("lambda_norm", "target_accuracy", "epsilon_rpe", "epsilon_unitary", "unitary_accuracy_fraction"):
            value = getattr(self, field)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be finite and nonnegative.")
        if self.target_accuracy == 0.0 or self.epsilon_rpe == 0.0:
            raise ValueError("Accuracy targets must be positive.")
        if self.unitary_accuracy_fraction > 1.0:
            raise ValueError("unitary_accuracy_fraction must not exceed one.")
        if self.unitary_builder_category not in ("deterministic_or_exact", "qdrift", "partial_randomized"):
            raise ValueError("Unsupported schedule evolution category.")
        if self.energy_correction not in ("linear", "qdrift_tangent"):
            raise ValueError("Unsupported schedule energy correction.")
        settings = Settings.from_json(self._unitary_builder_configuration.settings_json)
        for key in ("time", "seed", "num_samples"):
            if settings.has(key):
                raise ValueError(f"Builder {key} belongs to the rounds and must not be duplicated in shared settings.")
        expected_round_indices = tuple(range(len(self.rounds)))
        round_indices = tuple(round_data.round_index for round_data in self.rounds)
        if round_indices != expected_round_indices:
            raise ValueError("round_index values must be contiguous and match round order.")
        randomized = self.unitary_builder_category in ("qdrift", "partial_randomized")
        if randomized and (self.root_seed is None or self.root_seed < 0):
            raise ValueError("Randomized schedules require a nonnegative root_seed.")
        for round_data in self.rounds:
            expected_time = self.base_time * 2**round_data.round_index
            if not np.isfinite(expected_time) or not np.isclose(
                round_data.evolution_time, expected_time, rtol=1e-12, atol=0.0
            ):
                raise ValueError("Each round's evolution_time must match base_time * 2**round_index.")
            if randomized:
                if round_data.num_draws != round_data.shots_per_basis or None in round_data.draw_seeds:
                    raise ValueError("Randomized shots_per_basis must equal the number of concrete draw seeds.")
            elif round_data.num_draws != 1:
                raise ValueError("Deterministic rounds must contain exactly one draw.")

    def _payload(self) -> dict[str, Any]:
        """Return only the canonical schedule fields.

        Returns:
            JSON-safe schedule values, excluding derived positions, counts, and live inputs.

        """
        return {
            "rounds": [
                {
                    "round_index": round_data.round_index,
                    "evolution_time": round_data.evolution_time,
                    "shots_per_basis": round_data.shots_per_basis,
                    "scheduled_samples": round_data.scheduled_samples,
                    "draw_seeds": list(round_data.draw_seeds),
                }
                for round_data in self.rounds
            ],
            "hamiltonian_hash": self.hamiltonian_hash,
            "lambda_norm": self.lambda_norm,
            "target_accuracy": self.target_accuracy,
            "epsilon_rpe": self.epsilon_rpe,
            "epsilon_unitary": self.epsilon_unitary,
            "unitary_accuracy_fraction": self.unitary_accuracy_fraction,
            "error_budget_mode": self.error_budget_mode,
            "unitary_builder_category": self.unitary_builder_category,
            "energy_correction": self.energy_correction,
            "requested_seed": self.requested_seed,
            "root_seed": self.root_seed,
            "unitary_builder_configuration": self._unitary_builder_configuration.to_json(),
            "hadamard_test_circuit_builder_configuration": (
                self._hadamard_test_circuit_builder_configuration.to_json()
            ),
        }

    def _hash_update(self, h) -> None:
        """Feed the canonical schedule data into the hasher.

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
            f"Robust phase estimation schedule: rounds={self.num_rounds}, "
            f"experiments={sum(round_data.num_draws for round_data in self.rounds)}"
        )

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-safe schedule representation.

        Returns:
            The versioned schedule, shared settings, and concrete draw seeds.

        """
        return self._add_json_version(self._payload())

    def to_hdf5(self, group: h5py.Group) -> None:
        """Write canonical schedule values to HDF5.

        Args:
            group: Destination HDF5 group for the schedule payload.

        """
        self._add_hdf5_version(group)
        group.create_dataset("payload", data=json.dumps(self._payload()), dtype=h5py.string_dtype(encoding="utf-8"))

    @classmethod
    def _from_payload(
        cls,
        payload: dict[str, Any],
    ) -> RobustPhaseEstimationSchedule:
        """Restore the schedule through the same constructor validation as new schedules.

        Args:
            payload: Decoded schedule metadata.

        Returns:
            A validated immutable schedule.

        """
        values = dict(payload)
        values.pop("version", None)
        try:
            values["rounds"] = tuple(RobustPhaseEstimationRound(**data) for data in values["rounds"])
            for key in ("unitary_builder_configuration", "hadamard_test_circuit_builder_configuration"):
                values[key] = _AlgorithmConfiguration.from_json(values[key]).to_ref()
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid schedule fields: {error}") from error

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> RobustPhaseEstimationSchedule:
        """Restore a schedule from JSON data.

        Args:
            json_data: Versioned payload produced by ``to_json``.

        Returns:
            A workload with the original settings and seeds; no circuits are built.

        Raises:
            ValueError: If the serialization version or workload metadata is invalid.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls._from_payload(json_data)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RobustPhaseEstimationSchedule:
        """Restore a schedule from HDF5 data.

        Args:
            group: HDF5 group previously populated by ``to_hdf5``.

        Returns:
            A workload with the original settings and seeds; no circuits are built.

        Raises:
            ValueError: If the serialization version or workload metadata is invalid.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls._from_payload(json.loads(group["payload"].asstr()[()]))


class RobustPhaseEstimationCircuitSet(DataClass):
    """Optional replay bundle containing a schedule and its input data."""

    _serialization_version = "0.2.0"

    @staticmethod
    def data_type_name() -> str:
        """Return the replay-bundle wire identifier.

        Returns:
            The identifier used by existing circuit-set files.

        """
        return "robust_phase_estimation_circuit_set"

    def __init__(
        self,
        *,
        schedule: RobustPhaseEstimationSchedule,
        state_preparation: Circuit,
        qubit_hamiltonian: qdk_chemistry.data.QubitOperator,
    ) -> None:
        """Bundle immutable scheduling data with inputs for reproducible replay.

        Args:
            schedule: Resolved schedule, without duplicating any of its parameters.
            state_preparation: Live input preparation or a serialized representation of it.
            qubit_hamiltonian: Hamiltonian matching the schedule, defensively snapshotted.

        Raises:
            TypeError: If the schedule or inputs have the wrong types.
            ValueError: If the Hamiltonian does not match the schedule fingerprint or norm.

        """
        if not isinstance(schedule, RobustPhaseEstimationSchedule):
            raise TypeError("schedule must be a RobustPhaseEstimationSchedule.")
        if not isinstance(state_preparation, Circuit):
            raise TypeError("state_preparation must be a Circuit.")
        if not isinstance(qubit_hamiltonian, qdk_chemistry.data.QubitOperator):
            raise TypeError("qubit_hamiltonian must be a QubitOperator.")
        self.schedule = schedule
        self.state_preparation = state_preparation
        self._hamiltonian_snapshot = _HamiltonianSnapshot.from_operator(qubit_hamiltonian)
        if self._hamiltonian_snapshot.content_hash() != schedule.hamiltonian_hash:
            raise ValueError("Hamiltonian does not match the schedule fingerprint.")
        norm = float(np.sum(np.abs(qubit_hamiltonian.coefficients)))
        if not np.isclose(norm, schedule.lambda_norm, rtol=1e-12, atol=0.0):
            raise ValueError("Hamiltonian norm does not match the schedule lambda_norm.")
        super().__init__()

    @property
    def qubit_hamiltonian(self) -> QubitOperator:
        """Return a defensive copy of the stored Hamiltonian.

        Returns:
            Independent coefficients, Pauli strings, and metadata.

        """
        return self._hamiltonian_snapshot.to_operator()

    def rebind(self, state_preparation: Circuit) -> RobustPhaseEstimationCircuitSet:
        """Replace the live preparation without rescheduling or changing seeds.

        Args:
            state_preparation: Replacement live circuit, for example after loading QIR-only metadata.

        Returns:
            A new replay bundle sharing the immutable schedule.

        """
        return type(self)(
            schedule=self.schedule, state_preparation=state_preparation, qubit_hamiltonian=self.qubit_hamiltonian
        )

    def get_summary(self) -> str:
        """Describe the schedule contained in this replay bundle.

        Returns:
            A brief replay-bundle description.

        """
        return f"Robust phase estimation replay bundle: {self.schedule.num_rounds} rounds"

    def _hash_update(self, h) -> None:
        """Hash the immutable replay contents.

        Args:
            h: Hasher receiving the schedule and serialized input data.

        """
        _hash_str(h, self.data_type_name())
        _hash_arg(h, self.to_json())

    def to_json(self) -> dict[str, Any]:
        """Serialize a self-contained replay bundle.

        Returns:
            The versioned schedule and input data.

        """
        return self._add_json_version(
            {
                "schedule": self.schedule.to_json(),
                "state_preparation": self.state_preparation.to_json(),
                "qubit_hamiltonian": json.loads(self._hamiltonian_snapshot.payload_json),
            }
        )

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> RobustPhaseEstimationCircuitSet:
        """Load a replay bundle and validate its schedule and input identity.

        Args:
            json_data: Versioned replay data.

        Returns:
            A validated replay bundle.

        """
        cls._validate_json_version(cls._serialization_version, json_data)
        return cls(
            schedule=RobustPhaseEstimationSchedule.from_json(json_data["schedule"]),
            state_preparation=Circuit.from_json(json_data["state_preparation"]),
            qubit_hamiltonian=QubitOperator.from_json(json_data["qubit_hamiltonian"]),
        )

    def to_hdf5(self, group: h5py.Group) -> None:
        """Store the schedule and inputs in separate HDF5 groups.

        Args:
            group: Destination replay group.

        """
        self._add_hdf5_version(group)
        self.schedule.to_hdf5(group.create_group("schedule"))
        self.state_preparation.to_hdf5(group.create_group("state_preparation"))
        self.qubit_hamiltonian.to_hdf5(group.create_group("qubit_hamiltonian"))

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> RobustPhaseEstimationCircuitSet:
        """Load and validate separately stored scheduling and replay data.

        Args:
            group: HDF5 group containing a replay bundle.

        Returns:
            A replay bundle with validated schedule and input identity.

        """
        cls._validate_hdf5_version(cls._serialization_version, group)
        return cls(
            schedule=RobustPhaseEstimationSchedule.from_hdf5(group["schedule"]),
            state_preparation=Circuit.from_hdf5(group["state_preparation"]),
            qubit_hamiltonian=QubitOperator.from_hdf5(group["qubit_hamiltonian"]),
        )
