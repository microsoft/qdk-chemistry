"""Workload scheduling and circuit construction for robust phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

import qdk_chemistry.data
from qdk_chemistry.algorithms.base import create_from_ref
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.qdrift_error import _qdrift_lambda
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator, Settings

from .base import QpeCircuitBuilder

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "RobustPhaseEstimationCircuitBuilder",
    "RobustPhaseEstimationCircuitBuilderSettings",
    "RobustPhaseEstimationCircuitSet",
    "RobustPhaseEstimationExperimentSpec",
    "RobustPhaseEstimationRound",
    "RobustPhaseEstimationSchedule",
]

_UNSET_BUDGET_VALUE = -1.0
_DEFAULT_RPE_EPSILON_UNITARY = 0.85
_RPE_BUILDER_CATEGORIES = {
    "trotter": "deterministic_or_exact",
    "qdrift": "qdrift",
    "partially_randomized": "partial_randomized",
    "zassenhaus": "deterministic_or_exact",
}


def _round_sample_floor(round_index: int, base_time_ratio: float) -> int:
    """Round the scaled qDRIFT ladder up without converting its potentially large integer counts to floats."""
    numerator, denominator = max(1.0, base_time_ratio**2).as_integer_ratio()
    return (2 ** (2 * round_index + 1) * numerator + denominator - 1) // denominator


def _validate_unit_power(settings: Settings) -> None:
    """Require the RPE schedule to be the sole source of evolution powers."""
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


@dataclass(frozen=True)
class RobustPhaseEstimationExperimentSpec:
    """One X/Y pair, identified by its round and optional randomized draw."""

    experiment_index: int
    round_index: int
    draw_index: int | None
    draw_seed: int | None
    shots: int

    def __post_init__(self) -> None:
        """Validate the pair coordinates and shot count."""
        if self.experiment_index < 0 or self.round_index < 0:
            raise ValueError("experiment_index and round_index must be non-negative.")
        if self.draw_index is not None and self.draw_index < 0:
            raise ValueError("draw_index must be non-negative when provided.")
        if (self.draw_index is None) != (self.draw_seed is None):
            raise ValueError("draw_index and draw_seed must either both be set or both be None.")
        if self.shots < 1:
            raise ValueError("shots must be at least 1.")

    @property
    def x_circuit_index(self) -> int:
        """Position of the X circuit in the flat list returned by the builder."""
        return 2 * self.experiment_index

    @property
    def y_circuit_index(self) -> int:
        """Position of the Y circuit immediately after its X counterpart."""
        return self.x_circuit_index + 1


@dataclass(frozen=True)
class RobustPhaseEstimationRound:
    """Evolution time, per-basis shots, unitary seeds, and the per-draw sample count for pure qDRIFT only."""

    round_index: int
    evolution_time: float
    shots_per_basis: int
    scheduled_samples: int | None
    draw_seeds: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Check the evolution time and counts, and freeze the draw seeds."""
        for name, minimum in (("round_index", 0), ("shots_per_basis", 1), ("scheduled_samples", 1)):
            value = getattr(self, name)
            if name == "scheduled_samples" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}.")
        if not np.isfinite(self.evolution_time) or self.evolution_time <= 0.0:
            raise ValueError("evolution_time must be finite and positive.")
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
        """Number of distinct unitary draws in this round."""
        return len(self.draw_seeds)


def _restore_builder_ref(configuration: tuple[str, str, str]) -> AlgorithmRef:
    """Restore an independent reference from the builder identity and immutable settings snapshot."""
    algorithm_type, algorithm_name, settings_json = configuration
    return AlgorithmRef(algorithm_type, algorithm_name, settings=Settings.from_json(settings_json))


def _canonical_coefficients(operator: QubitOperator) -> tuple[np.ndarray, float]:
    """Freeze coefficients at replay precision before taking their one-norm."""
    coefficients = np.array(operator.coefficients, dtype=np.complex128, copy=True)
    return coefficients, float(np.sum(np.abs(coefficients)))


def _snapshot_hamiltonian(operator: QubitOperator, coefficients: np.ndarray) -> tuple[str, str]:
    """Capture the normalized coefficients and metadata, returning the payload and its fingerprint."""
    snapshot = QubitOperator(
        pauli_strings=list(operator.pauli_strings),
        coefficients=coefficients,
        encoding=operator.encoding,
        fermion_mode_order=operator.fermion_mode_order,
        term_partition=operator.term_partition,
        tapering=operator.tapering,
    )
    payload = json.dumps(snapshot.to_json(), sort_keys=True, allow_nan=False)
    return payload, hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class RobustPhaseEstimationSchedule:
    """In-memory RPE plan, independent of live circuits and executor settings."""

    rounds: tuple[RobustPhaseEstimationRound, ...]
    hamiltonian_hash: str
    lambda_norm: float
    target_accuracy: float
    epsilon_rpe: float
    epsilon_unitary: float
    unitary_accuracy_fraction: float
    error_budget_mode: str
    unitary_builder_category: str
    energy_correction: str
    requested_seed: int
    root_seed: int | None
    _unitary_builder: tuple[str, str, str] = field(repr=False)
    _hadamard_builder: tuple[str, str, str] = field(repr=False)

    def __post_init__(self) -> None:
        """Validate the time ladder, draw counts, and shared reconstruction settings."""
        object.__setattr__(self, "rounds", tuple(self.rounds))
        if not self.rounds:
            raise ValueError("rounds must contain at least one RPE round.")
        if len(self.hamiltonian_hash) != 64 or any(char not in "0123456789abcdef" for char in self.hamiltonian_hash):
            raise ValueError("hamiltonian_hash must be a SHA-256 fingerprint.")
        for name in ("lambda_norm", "target_accuracy", "epsilon_rpe", "epsilon_unitary", "unitary_accuracy_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if self.target_accuracy == 0.0 or self.epsilon_rpe == 0.0:
            raise ValueError("Accuracy targets must be positive.")
        if self.unitary_accuracy_fraction > 1.0:
            raise ValueError("unitary_accuracy_fraction must not exceed one.")
        if self.unitary_builder_category not in ("deterministic_or_exact", "qdrift", "partial_randomized"):
            raise ValueError("Unsupported schedule evolution category.")
        if self.energy_correction not in ("linear", "qdrift_tangent"):
            raise ValueError("Unsupported schedule energy correction.")
        if self.energy_correction == "qdrift_tangent" and self.unitary_builder_category != "qdrift":
            raise ValueError("The qDRIFT tangent correction requires a qDRIFT schedule.")
        settings = self.unitary_builder_configuration.settings
        for key in ("time", "seed", "num_samples"):
            if settings.has(key):
                raise ValueError(f"Builder {key} belongs to the rounds and must not be duplicated in shared settings.")
        randomized = self.unitary_builder_category in ("qdrift", "partial_randomized")
        if randomized and (self.root_seed is None or self.root_seed < 0):
            raise ValueError("Randomized schedules require a nonnegative root_seed.")
        for index, round_data in enumerate(self.rounds):
            if round_data.round_index != index:
                raise ValueError("round_index values must be contiguous and match round order.")
            if (round_data.scheduled_samples is not None) != (self.unitary_builder_category == "qdrift"):
                raise ValueError("scheduled_samples must be set for pure qDRIFT and None for other evolution families.")
            expected_time = self.base_time * 2**index
            if not np.isfinite(expected_time) or not np.isclose(
                round_data.evolution_time, expected_time, rtol=1e-12, atol=0.0
            ):
                raise ValueError("Each round's evolution_time must match base_time * 2**round_index.")
            if randomized:
                if round_data.num_draws != round_data.shots_per_basis or None in round_data.draw_seeds:
                    raise ValueError("Randomized shots_per_basis must equal the number of concrete draw seeds.")
            elif round_data.draw_seeds != (None,):
                raise ValueError("Deterministic rounds must have draw_seeds=(None,).")

    @staticmethod
    def hamiltonian_fingerprint(qubit_hamiltonian: QubitOperator) -> str:
        """Identify a Hamiltonian by coefficients, labels, and metadata, independent of coefficient dtype."""
        coefficients, _ = _canonical_coefficients(qubit_hamiltonian)
        _, fingerprint = _snapshot_hamiltonian(qubit_hamiltonian, coefficients)
        return fingerprint

    def validate_and_copy_hamiltonian(self, qubit_hamiltonian: QubitOperator) -> QubitOperator:
        """Check the scheduled identity and return a defensive copy for execution."""
        return QubitOperator.from_json(json.loads(_validated_snapshot(self, qubit_hamiltonian)))

    @property
    def unitary_builder_configuration(self) -> AlgorithmRef:
        """Independent shared evolution settings, excluding round-owned time, seed, and samples."""
        return _restore_builder_ref(self._unitary_builder)

    @property
    def hadamard_test_circuit_builder_configuration(self) -> AlgorithmRef:
        """Independent resolved settings for the X/Y circuit builder."""
        return _restore_builder_ref(self._hadamard_builder)

    @property
    def base_time(self) -> float:
        """Evolution time of the first round."""
        return self.rounds[0].evolution_time

    @property
    def num_rounds(self) -> int:
        """Number of rounds, including round zero."""
        return len(self.rounds)

    @property
    def final_samples(self) -> int | None:
        """Pure qDRIFT sample count for the final round, or None for other evolution families."""
        return self.rounds[-1].scheduled_samples

    @property
    def experiment_specs(self) -> tuple[RobustPhaseEstimationExperimentSpec, ...]:
        """Canonical X/Y pairs in round and draw order."""
        return tuple(_iter_experiment_specs(self))

    def experiment_specs_for_round(self, round_index: int) -> tuple[RobustPhaseEstimationExperimentSpec, ...]:
        """Return the specified round's pairs, preserving their global indices."""
        if round_index < 0 or round_index >= self.num_rounds:
            raise IndexError(f"round_index must be in [0, {self.num_rounds - 1}], got {round_index}.")
        return tuple(spec for spec in _iter_experiment_specs(self) if spec.round_index == round_index)


def _iter_experiment_specs(schedule: RobustPhaseEstimationSchedule) -> Iterator[RobustPhaseEstimationExperimentSpec]:
    """Derive draw identities lazily from the recorded round seeds."""
    randomized = schedule.unitary_builder_category in ("qdrift", "partial_randomized")
    experiment_index = 0
    for round_data in schedule.rounds:
        for draw_index, seed in enumerate(round_data.draw_seeds):
            yield RobustPhaseEstimationExperimentSpec(
                experiment_index,
                round_data.round_index,
                draw_index if randomized else None,
                seed if randomized else None,
                1 if randomized else round_data.shots_per_basis,
            )
            experiment_index += 1


def _validated_snapshot(schedule: RobustPhaseEstimationSchedule, hamiltonian: QubitOperator) -> str:
    """Take a defensive snapshot and check it against the schedule before binding or executing it."""
    coefficients, norm = _canonical_coefficients(hamiltonian)
    payload, fingerprint = _snapshot_hamiltonian(hamiltonian, coefficients)
    if fingerprint != schedule.hamiltonian_hash:
        raise ValueError("Hamiltonian does not match the schedule fingerprint.")
    if not np.isclose(norm, schedule.lambda_norm, rtol=1e-12, atol=0.0):
        raise ValueError("Hamiltonian norm does not match the schedule lambda_norm.")
    return payload


@dataclass(frozen=True, init=False)
class RobustPhaseEstimationCircuitSet:
    """An in-memory schedule bound to a live input circuit and an independent Hamiltonian snapshot."""

    schedule: RobustPhaseEstimationSchedule
    state_preparation: Circuit
    _hamiltonian_snapshot: str = field(repr=False)

    def __init__(
        self,
        *,
        schedule: RobustPhaseEstimationSchedule,
        state_preparation: Circuit,
        qubit_hamiltonian: qdk_chemistry.data.QubitOperator,
    ) -> None:
        """Bind the live preparation and a checked Hamiltonian snapshot to the plan."""
        if not isinstance(schedule, RobustPhaseEstimationSchedule):
            raise TypeError("schedule must be a RobustPhaseEstimationSchedule.")
        if not isinstance(state_preparation, Circuit) or not isinstance(
            qubit_hamiltonian, qdk_chemistry.data.QubitOperator
        ):
            raise TypeError("Replay requires a Circuit and a QubitOperator.")
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "state_preparation", state_preparation)
        object.__setattr__(self, "_hamiltonian_snapshot", _validated_snapshot(schedule, qubit_hamiltonian))

    @property
    def qubit_hamiltonian(self) -> QubitOperator:
        """A fresh copy of the Hamiltonian for this replay."""
        return QubitOperator.from_json(json.loads(self._hamiltonian_snapshot))

    def rebind(self, state_preparation: Circuit) -> RobustPhaseEstimationCircuitSet:
        """Use another live preparation without choosing new rounds or draws."""
        return type(self)(
            schedule=self.schedule, state_preparation=state_preparation, qubit_hamiltonian=self.qubit_hamiltonian
        )


class RobustPhaseEstimationCircuitBuilderSettings(Settings):
    """Settings for scheduling and building robust phase estimation circuits."""

    def __init__(self) -> None:
        """Initialize nested algorithms and RPE schedule settings."""
        super().__init__()
        self._set_default(
            "unitary_builder",
            "algorithm_ref",
            AlgorithmRef("hamiltonian_unitary_builder", "qdrift"),
            "Time-evolution builder used to realize U(t); sized per round with power fixed at 1.",
        )
        self._set_default(
            "hadamard_test_circuit_builder",
            "algorithm_ref",
            AlgorithmRef("hadamard_test_circuit_builder", "qdk"),
            "Circuit builder used to generate X- and Y-basis Hadamard tests.",
        )
        self._set_default(
            "target_accuracy", "double", 1e-3, "Requested absolute accuracy epsilon on the final energy estimate."
        )
        self._set_default(
            "base_time",
            "double",
            0.0,
            "Base evolution time tau. 0.0 selects pi/(2*lambda) automatically; "
            "explicit positive values must satisfy tau*lambda < pi.",
        )
        self._set_default(
            "unitary_accuracy_fraction",
            "double",
            _UNSET_BUDGET_VALUE,
            "Optional legacy fraction of target_accuracy assigned to a non-Trotter unitary builder; "
            "omitted partially randomized builders use an independent unitary tolerance. Must be below 1.",
        )
        self._set_default(
            "epsilon_rpe",
            "double",
            _UNSET_BUDGET_VALUE,
            "Optional explicit RPE energy tolerance for non-Trotter builders. Set together with epsilon_unitary.",
        )
        self._set_default(
            "epsilon_unitary",
            "double",
            _UNSET_BUDGET_VALUE,
            "Positive full-unitary tolerance. Trotter and partially randomized builders default to 0.85.",
        )
        self._set_default(
            "energy_correction",
            "string",
            "auto",
            "Phase-to-energy map: 'auto', 'linear', or 'qdrift_tangent' (qDRIFT builder only).",
            ["auto", "linear", "qdrift_tangent"],
        )
        self._set_default(
            "seed",
            "int",
            -1,
            "Random seed for evolution draws. Use -1 to choose one entropy-backed seed per circuit set.",
        )
        self._set_default(
            "max_qdrift_samples",
            "int",
            1_000_000,
            "Maximum samples per scheduled qDRIFT circuit. Increase explicitly only when resources permit.",
        )


class RobustPhaseEstimationCircuitBuilder(QpeCircuitBuilder):
    """QDK circuit builder for robust phase estimation.

    ``run`` schedules once and returns the complete circuit list through
    ``_run_impl``. Streaming supports bounded-memory execution and replay.
    Override ``iter_build`` to customize circuit construction for both paths.
    """

    def __init__(
        self,
        *,
        target_accuracy: float = 1e-3,
        base_time: float = 0.0,
        unitary_accuracy_fraction: float | None = None,
        energy_correction: str = "auto",
        seed: int = -1,
        epsilon_rpe: float | None = None,
        epsilon_unitary: float | None = None,
        unitary_builder: AlgorithmRef | None = None,
        hadamard_test_circuit_builder: AlgorithmRef | None = None,
        max_qdrift_samples: int = 1_000_000,
    ) -> None:
        """Initialize robust phase estimation scheduling and circuit construction.

        Args:
            target_accuracy: Requested absolute accuracy of the final energy estimate.
            base_time: Base evolution time; zero selects the coefficient-norm default.
            unitary_accuracy_fraction: Optional legacy fractional error allocation for non-Trotter evolution.
            energy_correction: Phase-to-energy mapping, or ``"auto"`` to select it from the evolution family.
            seed: Unitary-draw root seed; a negative value requests entropy once per randomized workload.
            epsilon_rpe: Optional explicit energy tolerance, paired with ``epsilon_unitary`` for supported families.
            epsilon_unitary: Optional full-evolution error tolerance passed to the builder's accuracy setting.
            unitary_builder: Reference to a time-evolution builder; its power must be one.
            hadamard_test_circuit_builder: Reference to the builder used for each X/Y circuit pair.
            max_qdrift_samples: Positive per-circuit sample ceiling for qDRIFT, including nested accuracy sizing.

        """
        super().__init__()
        self._settings = RobustPhaseEstimationCircuitBuilderSettings()
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("base_time", base_time)
        self._settings.set("energy_correction", energy_correction)
        self._settings.set("seed", seed)
        self._settings.set("max_qdrift_samples", max_qdrift_samples)
        if unitary_accuracy_fraction is not None:
            self._settings.set("unitary_accuracy_fraction", unitary_accuracy_fraction)
        if epsilon_rpe is not None:
            self._settings.set("epsilon_rpe", epsilon_rpe)
        if epsilon_unitary is not None:
            self._settings.set("epsilon_unitary", epsilon_unitary)
        if unitary_builder is not None:
            self._settings.set("unitary_builder", unitary_builder)
        if hadamard_test_circuit_builder is not None:
            self._settings.set("hadamard_test_circuit_builder", hadamard_test_circuit_builder)

    def build_schedule(self, qubit_hamiltonian: QubitOperator) -> RobustPhaseEstimationSchedule:
        """Lock settings and resolve a reproducible workload without building circuits.

        Deterministic rounds use one multi-shot X/Y pair. Randomized rounds use one independently
        seeded unitary per shot, shared by both bases. All sample counts are checked before drawing seeds.

        Args:
            qubit_hamiltonian: Hermitian Hamiltonian used to resolve evolution times and builder settings.

        Returns:
            An in-memory schedule of rounds and draw seeds that can be passed directly to ``iter_build``.

        Raises:
            KeyError: If a configured builder is not registered.
            TypeError: If the Hamiltonian is not a QubitOperator or the builder is not a time-evolution builder.
            ValueError: If the Hamiltonian is non-Hermitian or a builder, correction, power, time, or budget is invalid.

        """
        self._settings.lock()
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError(
                f"The {self.name()!r} RPE builder takes a QubitOperator, got {type(qubit_hamiltonian).__name__}."
            )
        if not qubit_hamiltonian.is_hermitian():
            raise ValueError("RPE requires a Hermitian Hamiltonian; coefficients have nonzero imaginary parts.")
        for setting in ("target_accuracy", "unitary_accuracy_fraction", "epsilon_rpe", "epsilon_unitary"):
            value = float(self._settings.get(setting))
            if not np.isfinite(value):
                raise ValueError(f"{setting} must be finite, received {value}.")
        epsilon_total = float(self._settings.get("target_accuracy"))
        if epsilon_total <= 0.0:
            raise ValueError(f"target_accuracy (epsilon) must be positive, received {epsilon_total}.")
        base_time = float(self._settings.get("base_time"))
        if not np.isfinite(base_time) or base_time < 0.0:
            raise ValueError(f"base_time must be finite and non-negative, received {base_time}.")
        max_qdrift_samples = int(self._settings.get("max_qdrift_samples"))
        if max_qdrift_samples < 1:
            raise ValueError("max_qdrift_samples must be positive.")

        unitary_ref = self._settings.get("unitary_builder")
        unitary_builder = self._create_nested("unitary_builder")
        unitary_settings = unitary_builder.settings()
        _validate_unit_power(unitary_settings)

        if not isinstance(unitary_builder, TimeEvolutionBuilder):
            raise TypeError(
                "RPE requires a TimeEvolutionBuilder; "
                f"'{unitary_ref.algorithm_type}/{unitary_ref.algorithm_name}' "
                "does not represent supported time evolution. Block encodings and quantum walks are not supported."
            )
        builder_name = unitary_builder.name()
        if builder_name not in _RPE_BUILDER_CATEGORIES:
            supported = ", ".join(sorted(_RPE_BUILDER_CATEGORIES))
            raise ValueError(f"Unsupported RPE unitary builder {builder_name!r}. Supported builders: {supported}.")
        category = _RPE_BUILDER_CATEGORIES[builder_name]
        correction = str(self._settings.get("energy_correction"))
        if correction == "auto":
            correction = "qdrift_tangent" if category == "qdrift" else "linear"
        elif correction == "qdrift_tangent" and category != "qdrift":
            raise ValueError(
                f"energy_correction='qdrift_tangent' requires the qDRIFT unitary builder, got {builder_name!r}."
            )
        fraction, epsilon_rpe, epsilon_unitary, budget_mode = self._resolve_budget(
            category,
            epsilon_total,
            is_trotter=builder_name == "trotter",
        )

        coefficients, lambda_norm = _canonical_coefficients(qubit_hamiltonian)
        automatic_base_time = float((np.pi / 2.0) / lambda_norm) if lambda_norm > 0.0 else 1.0
        if base_time == 0.0:
            base_time = automatic_base_time
        if base_time * lambda_norm >= np.pi:
            raise ValueError(
                "base_time must satisfy base_time * lambda_norm < pi to avoid energy aliasing; "
                f"got base_time={base_time:.6g} and lambda_norm={lambda_norm:.6g}."
            )

        if epsilon_rpe <= 0.0:
            raise ValueError(f"epsilon must be positive, received {epsilon_rpe}.")
        if lambda_norm < 0.0:
            raise ValueError(f"lambda_norm must be non-negative, received {lambda_norm}.")
        energy_resolution_scale = float(np.pi / (2.0 * base_time))
        total_round = (
            0
            if lambda_norm == 0.0 or energy_resolution_scale <= epsilon_rpe
            else int(np.ceil(np.log2(energy_resolution_scale / epsilon_rpe)))
        )
        round_samples: list[int] | None = None
        if category == "qdrift":
            base_time_ratio = base_time / automatic_base_time if lambda_norm > 0.0 else 1.0
            round_samples = [
                _round_sample_floor(round_index, base_time_ratio) for round_index in range(total_round + 1)
            ]
            if unitary_settings.has("target_accuracy") and unitary_settings.get("target_accuracy") > 0.0:
                sample_resolver = getattr(unitary_builder, "_resolve_num_samples_from_lambda", None)
                if not callable(sample_resolver):
                    raise TypeError("qDRIFT builders with target_accuracy must support sample-count resolution.")
                sampled_lambda = _qdrift_lambda(qubit_hamiltonian, unitary_settings.get("weight_threshold"))
                for round_index, minimum_samples in enumerate(round_samples):
                    resolved_samples = sample_resolver(
                        sampled_lambda, float((2**round_index) * base_time), minimum_samples
                    )
                    if not isinstance(resolved_samples, int) or resolved_samples < minimum_samples:
                        raise ValueError(
                            "Resolved qDRIFT samples must be an integer at least as large as the RPE count."
                        )
                    round_samples[round_index] = resolved_samples
            peak_samples = max(round_samples)
            if peak_samples > max_qdrift_samples:
                raise ValueError(
                    f"qDRIFT round {round_samples.index(peak_samples)} requires {peak_samples} samples, exceeding "
                    f"max_qdrift_samples={max_qdrift_samples}. Raise max_qdrift_samples explicitly only "
                    "if sufficient resources are available."
                )
        randomized = category in ("qdrift", "partial_randomized")
        requested_seed = int(self._settings.get("seed"))
        root_seed = None
        if randomized:
            root_seed = (
                requested_seed
                if requested_seed >= 0
                else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
            )

        rounds: list[RobustPhaseEstimationRound] = []
        for round_index in range(total_round + 1):
            shots = int(np.ceil(np.e * (11 + 4 * (total_round - round_index))))
            draw_seeds: tuple[int | None, ...] = (None,)
            if root_seed is not None:
                draw_seeds = tuple(
                    int(np.random.SeedSequence([root_seed, round_index, draw_index]).generate_state(1, np.uint32)[0])
                    for draw_index in range(shots)
                )
            rounds.append(
                RobustPhaseEstimationRound(
                    round_index=round_index,
                    evolution_time=float((2**round_index) * base_time),
                    shots_per_basis=shots,
                    scheduled_samples=round_samples[round_index] if round_samples is not None else None,
                    draw_seeds=draw_seeds,
                )
            )

        shared_settings = json.loads(unitary_settings.to_json())
        for key in ("time", "seed", "num_samples"):
            shared_settings.pop(key, None)
        if category != "qdrift" and "target_accuracy" in shared_settings:
            shared_settings["target_accuracy"] = epsilon_unitary
        hadamard_builder = self._create_nested("hadamard_test_circuit_builder")
        hadamard_settings = hadamard_builder.settings()
        if hadamard_settings.has("controlled_circuit_mapper"):
            mapper = create_from_ref(hadamard_settings, "controlled_circuit_mapper")
            hadamard_settings.set(
                "controlled_circuit_mapper", AlgorithmRef(mapper.type_name(), mapper.name(), settings=mapper.settings())
            )
        _, fingerprint = _snapshot_hamiltonian(qubit_hamiltonian, coefficients)
        return RobustPhaseEstimationSchedule(
            rounds=tuple(rounds),
            hamiltonian_hash=fingerprint,
            lambda_norm=lambda_norm,
            target_accuracy=epsilon_total,
            epsilon_rpe=epsilon_rpe,
            epsilon_unitary=epsilon_unitary,
            unitary_accuracy_fraction=fraction,
            error_budget_mode=budget_mode,
            unitary_builder_category=category,
            energy_correction=correction,
            requested_seed=requested_seed,
            root_seed=root_seed,
            _unitary_builder=(unitary_ref.algorithm_type, builder_name, json.dumps(shared_settings)),
            _hadamard_builder=(hadamard_builder.type_name(), hadamard_builder.name(), hadamard_settings.to_json()),
        )

    def _resolve_budget(
        self,
        category: str,
        epsilon_total: float,
        *,
        is_trotter: bool,
    ) -> tuple[float, float, float, str]:
        """Resolve the fraction, RPE energy tolerance, unitary tolerance, and budget mode."""
        configured_fraction = float(self._settings.get("unitary_accuracy_fraction"))
        explicit_rpe = float(self._settings.get("epsilon_rpe"))
        explicit_unitary = float(self._settings.get("epsilon_unitary"))

        if is_trotter:
            if configured_fraction != _UNSET_BUDGET_VALUE:
                raise ValueError(
                    "unitary_accuracy_fraction is not supported for Trotter RPE; "
                    "set target_accuracy and optional epsilon_unitary instead."
                )
            if explicit_rpe != _UNSET_BUDGET_VALUE:
                raise ValueError(
                    "epsilon_rpe is not configurable for Trotter RPE; target_accuracy sets the RPE energy tolerance."
                )
            if explicit_unitary != _UNSET_BUDGET_VALUE and explicit_unitary <= 0.0:
                raise ValueError("epsilon_unitary must be positive for Trotter RPE.")
            epsilon_unitary = (
                _DEFAULT_RPE_EPSILON_UNITARY if explicit_unitary == _UNSET_BUDGET_VALUE else explicit_unitary
            )
            return 0.0, epsilon_total, epsilon_unitary, "independent_trotter"

        if (
            category == "partial_randomized"
            and configured_fraction == _UNSET_BUDGET_VALUE
            and explicit_rpe == _UNSET_BUDGET_VALUE
        ):
            if explicit_unitary != _UNSET_BUDGET_VALUE and explicit_unitary <= 0.0:
                raise ValueError("epsilon_unitary must be positive for partially randomized RPE.")
            epsilon_unitary = (
                _DEFAULT_RPE_EPSILON_UNITARY if explicit_unitary == _UNSET_BUDGET_VALUE else explicit_unitary
            )
            if epsilon_unitary >= np.sin(np.pi / 3.0):
                raise ValueError("epsilon_unitary must be smaller than sin(pi/3) for branch-safe RPE.")
            return 0.0, epsilon_total, epsilon_unitary, "independent_partial_randomized"

        if configured_fraction != _UNSET_BUDGET_VALUE and configured_fraction >= 1.0:
            raise ValueError("unitary_accuracy_fraction must be smaller than 1 so RPE keeps a positive energy budget.")
        fraction = 0.5 if configured_fraction == _UNSET_BUDGET_VALUE else max(configured_fraction, 0.0)
        has_explicit_budget = explicit_rpe != _UNSET_BUDGET_VALUE or explicit_unitary != _UNSET_BUDGET_VALUE

        if category == "qdrift":
            if has_explicit_budget:
                raise ValueError("Explicit epsilon_rpe/epsilon_unitary budgets are not supported for pure qDRIFT.")
            fraction = 0.0

        if has_explicit_budget:
            if explicit_rpe <= 0.0 or explicit_unitary <= 0.0:
                raise ValueError("epsilon_rpe and epsilon_unitary must both be positive when set explicitly.")
            if explicit_unitary >= np.sin(np.pi / 3.0):
                raise ValueError("epsilon_unitary must be smaller than sin(pi/3) for branch-safe RPE.")
            propagated_bound = (2.0 / np.pi) * explicit_rpe * np.arcsin(explicit_unitary)
            if propagated_bound > epsilon_total * (1.0 + 1e-12):
                raise ValueError(
                    "Explicit error budgets do not meet target_accuracy: "
                    f"(2/pi) * epsilon_rpe * arcsin(epsilon_unitary) = {propagated_bound:.6g} "
                    f"> {epsilon_total:.6g}."
                )
            return fraction, explicit_rpe, explicit_unitary, "explicit"

        epsilon_unitary = fraction * epsilon_total
        if epsilon_unitary >= np.sin(np.pi / 3.0):
            raise ValueError("epsilon_unitary must be smaller than sin(pi/3) for branch-safe RPE.")
        epsilon_rpe = (1.0 - fraction) * epsilon_total
        return fraction, epsilon_rpe, epsilon_unitary, "fraction"

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
        sampled_lambda: float | None = None
        for experiment_spec in _iter_experiment_specs(schedule):
            round_data = schedule.rounds[experiment_spec.round_index]
            updates: dict[str, object] = {"time": round_data.evolution_time}
            if experiment_spec.draw_seed is not None:
                updates["seed"] = experiment_spec.draw_seed
            if round_data.scheduled_samples is not None:
                updates["num_samples"] = round_data.scheduled_samples
            unitary_builder = create(
                unitary_configuration.algorithm_type,
                unitary_configuration.algorithm_name,
                **(unitary_settings | updates),
            )
            if round_data.scheduled_samples is not None:
                sample_resolver = getattr(unitary_builder, "_resolve_num_samples_from_lambda", None)
                if callable(sample_resolver):
                    if sampled_lambda is None:
                        sampled_lambda = _qdrift_lambda(hamiltonian, unitary_builder.settings().get("weight_threshold"))
                    if sample_resolver(sampled_lambda, round_data.evolution_time) != round_data.scheduled_samples:
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

        """
        schedule = self.build_schedule(qubit_hamiltonian)
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
