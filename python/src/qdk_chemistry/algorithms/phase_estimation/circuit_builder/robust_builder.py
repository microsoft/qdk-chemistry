"""Lazy circuit generation for robust phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator, Settings
from qdk_chemistry.utils.rpe import num_rounds, qdrift_schedule

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "QdkRobustPhaseEstimationCircuitBuilder",
    "RobustPhaseEstimationCircuitBuilder",
    "RobustPhaseEstimationCircuitBuilderFactory",
    "RobustPhaseEstimationCircuitBuilderSettings",
    "RobustPhaseEstimationCircuitSet",
    "RobustPhaseEstimationExperiment",
    "RobustPhaseEstimationRound",
]


@dataclass(frozen=True)
class _AlgorithmSnapshot:
    """Immutable snapshot of an algorithm reference and its nested settings."""

    algorithm_type: str
    algorithm_name: str
    settings_json: str

    @classmethod
    def from_ref(cls, ref: AlgorithmRef) -> _AlgorithmSnapshot:
        """Capture an algorithm reference without retaining mutable settings."""
        if ref.settings is None:
            raise ValueError(
                f"Cannot snapshot unresolved algorithm reference '{ref.algorithm_type}/{ref.algorithm_name}'."
            )
        return cls(ref.algorithm_type, ref.algorithm_name, ref.settings.to_json())

    def to_ref(self) -> AlgorithmRef:
        """Reconstruct an independent algorithm reference."""
        return AlgorithmRef(
            self.algorithm_type,
            self.algorithm_name,
            settings=Settings.from_json(self.settings_json),
        )

    def has_setting(self, key: str) -> bool:
        """Return whether the snapshotted settings contain ``key``."""
        settings = Settings.from_json(self.settings_json)
        return settings.has(key)

    def with_updates(self, **updates: object) -> _AlgorithmSnapshot:
        """Return a new snapshot with existing setting values updated."""
        ref = self.to_ref()
        if ref.settings is None:
            raise RuntimeError("Algorithm snapshot unexpectedly has no settings.")
        for key, value in updates.items():
            if not ref.settings.has(key):
                raise ValueError(
                    f"Algorithm '{self.algorithm_type}/{self.algorithm_name}' does not define setting '{key}'."
                )
            ref.settings.set(key, value)
        return self.from_ref(ref)

    def create(self) -> Algorithm:
        """Create a fresh configured algorithm from this snapshot."""
        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        settings = Settings.from_json(self.settings_json)
        return create(self.algorithm_type, self.algorithm_name, **settings.to_dict())


@dataclass(frozen=True)
class RobustPhaseEstimationRound:
    """Circuit-generation metadata for one robust phase estimation round."""

    round_index: int
    evolution_time: float
    shots_per_basis: int
    num_draws: int
    scheduled_samples: int
    circuit_multiplicity: int
    draw_seeds: tuple[int, ...]
    _unitary_builder_snapshot: _AlgorithmSnapshot = field(repr=False)

    @property
    def unitary_builder_configuration(self) -> AlgorithmRef:
        """Return an independent copy of the per-round unitary-builder configuration."""
        return self._unitary_builder_snapshot.to_ref()


@dataclass(frozen=True)
class RobustPhaseEstimationExperiment:
    """One lazily generated X/Y Hadamard-test circuit pair."""

    round_index: int
    evolution_time: float
    shots_per_basis: int
    draw_index: int | None
    draw_seed: int | None
    circuit_multiplicity: int
    x_circuit: Circuit
    y_circuit: Circuit
    _unitary_builder_snapshot: _AlgorithmSnapshot = field(repr=False)

    @property
    def unitary_builder_configuration(self) -> AlgorithmRef:
        """Return an independent copy of the exact unitary-builder configuration used."""
        return self._unitary_builder_snapshot.to_ref()


@dataclass(frozen=True)
class RobustPhaseEstimationCircuitSet:
    """Re-iterable lazy circuit collection for robust phase estimation."""

    rounds: tuple[RobustPhaseEstimationRound, ...]
    lambda_norm: float
    base_time: float
    target_accuracy: float
    epsilon_rpe: float
    epsilon_unitary: float
    unitary_accuracy_fraction: float
    error_budget_mode: str
    unitary_builder_category: str
    energy_correction: str
    requested_seed: int
    root_seed: int | None
    _state_preparation: Circuit = field(repr=False)
    _qubit_hamiltonian: QubitOperator = field(repr=False)
    _hadamard_builder_snapshot: _AlgorithmSnapshot = field(repr=False)

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
        """Return an independent copy of the Hadamard-test circuit-builder configuration."""
        return self._hadamard_builder_snapshot.to_ref()

    def __iter__(self) -> Iterator[RobustPhaseEstimationExperiment]:
        """Generate every circuit pair lazily in round and draw order."""
        for round_data in self.rounds:
            yield from self._iter_round(round_data)

    def iter_round(self, round_index: int) -> Iterator[RobustPhaseEstimationExperiment]:
        """Generate circuit pairs lazily for one round.

        Args:
            round_index: Zero-based round index.

        Returns:
            Iterator over the round's circuit-pair experiments.

        Raises:
            IndexError: If ``round_index`` is outside the circuit set.

        """
        if round_index < 0 or round_index >= len(self.rounds):
            raise IndexError(f"round_index must be in [0, {len(self.rounds) - 1}], got {round_index}.")
        yield from self._iter_round(self.rounds[round_index])

    def _iter_round(self, round_data: RobustPhaseEstimationRound) -> Iterator[RobustPhaseEstimationExperiment]:
        if round_data.draw_seeds:
            draws: tuple[tuple[int | None, int | None], ...] = tuple(enumerate(round_data.draw_seeds))
        else:
            draws = ((None, None),)

        for draw_index, draw_seed in draws:
            unitary_snapshot = _AlgorithmSnapshot.from_ref(round_data.unitary_builder_configuration)
            if draw_seed is not None and unitary_snapshot.has_setting("seed"):
                unitary_snapshot = unitary_snapshot.with_updates(seed=draw_seed)

            unitary_builder = unitary_snapshot.create()
            unitary = unitary_builder.run(self._qubit_hamiltonian)

            x_builder = self._hadamard_builder_snapshot.with_updates(test_basis="X").create()
            x_circuit = x_builder.run(self._state_preparation, unitary)
            y_builder = self._hadamard_builder_snapshot.with_updates(test_basis="Y").create()
            y_circuit = y_builder.run(self._state_preparation, unitary)

            yield RobustPhaseEstimationExperiment(
                round_index=round_data.round_index,
                evolution_time=round_data.evolution_time,
                shots_per_basis=round_data.shots_per_basis,
                draw_index=draw_index,
                draw_seed=draw_seed,
                circuit_multiplicity=round_data.circuit_multiplicity,
                x_circuit=x_circuit,
                y_circuit=y_circuit,
                _unitary_builder_snapshot=unitary_snapshot,
            )


class RobustPhaseEstimationCircuitBuilderSettings(Settings):
    """Settings for robust phase estimation circuit generation."""

    def __init__(self) -> None:
        """Initialize nested circuit algorithms and RPE schedule settings."""
        super().__init__()
        self._set_default(
            "unitary_builder",
            "algorithm_ref",
            AlgorithmRef("hamiltonian_unitary_builder", "qdrift"),
            "Time-evolution builder used to realize U(t); sized per round.",
        )
        self._set_default(
            "hadamard_test_circuit_builder",
            "algorithm_ref",
            AlgorithmRef("hadamard_test_circuit_builder", "qdk"),
            "Circuit builder used to generate X- and Y-basis Hadamard tests.",
        )
        self._set_default(
            "target_accuracy",
            "double",
            1e-3,
            "Requested absolute accuracy epsilon on the final energy estimate.",
        )
        self._set_default(
            "base_time",
            "double",
            0.0,
            "Base evolution time tau. 0.0 selects pi/(2*lambda) automatically.",
        )
        self._set_default(
            "unitary_accuracy_fraction",
            "double",
            0.5,
            "Fraction of target_accuracy assigned to the unitary builder; ignored for pure qDRIFT.",
        )
        self._set_default(
            "epsilon_rpe",
            "double",
            0.0,
            "Optional explicit RPE energy tolerance. Set together with epsilon_unitary.",
        )
        self._set_default(
            "epsilon_unitary",
            "double",
            0.0,
            "Optional explicit dimensionless unitary signal tolerance. Set together with epsilon_rpe.",
        )
        self._set_default(
            "energy_correction",
            "string",
            "auto",
            "Phase-to-energy map: 'auto', 'linear', or 'qdrift_tangent'.",
            ["auto", "linear", "qdrift_tangent"],
        )
        self._set_default(
            "seed",
            "int",
            -1,
            "Random seed for evolution draws. Use -1 to choose one entropy-backed seed per circuit set.",
        )


class RobustPhaseEstimationCircuitBuilder(Algorithm):
    """Abstract circuit builder for robust phase estimation."""

    def __init__(
        self,
        target_accuracy: float = 1e-3,
        base_time: float = 0.0,
        unitary_accuracy_fraction: float = 0.5,
        energy_correction: str = "auto",
        seed: int = -1,
        epsilon_rpe: float = 0.0,
        epsilon_unitary: float = 0.0,
        unitary_builder: AlgorithmRef | None = None,
        hadamard_test_circuit_builder: AlgorithmRef | None = None,
    ) -> None:
        """Initialize robust phase estimation circuit generation.

        Args:
            target_accuracy: Requested absolute accuracy on the final energy.
            base_time: Base evolution time; ``0.0`` selects ``pi/(2*lambda)``.
            unitary_accuracy_fraction: Fraction of ``target_accuracy`` assigned to unitary synthesis.
            energy_correction: Phase-to-energy map: ``"auto"``, ``"linear"``, or ``"qdrift_tangent"``.
            seed: Root random seed; ``-1`` chooses one entropy-backed seed per circuit set.
            epsilon_rpe: Optional explicit RPE energy tolerance.
            epsilon_unitary: Optional explicit unitary signal tolerance.
            unitary_builder: Optional time-evolution builder reference.
            hadamard_test_circuit_builder: Optional Hadamard-test circuit-builder reference.

        """
        super().__init__()
        self._settings = RobustPhaseEstimationCircuitBuilderSettings()
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("base_time", base_time)
        self._settings.set("unitary_accuracy_fraction", unitary_accuracy_fraction)
        self._settings.set("energy_correction", energy_correction)
        self._settings.set("seed", seed)
        self._settings.set("epsilon_rpe", epsilon_rpe)
        self._settings.set("epsilon_unitary", epsilon_unitary)
        if unitary_builder is not None:
            self._settings.set("unitary_builder", unitary_builder)
        if hadamard_test_circuit_builder is not None:
            self._settings.set("hadamard_test_circuit_builder", hadamard_test_circuit_builder)

    def type_name(self) -> str:
        """Return the robust phase estimation circuit-builder type name."""
        return "robust_phase_estimation_circuit_builder"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Build a lazy robust phase estimation circuit set."""


class RobustPhaseEstimationCircuitBuilderFactory(AlgorithmFactory):
    """Factory for robust phase estimation circuit builders."""

    def algorithm_type_name(self) -> str:
        """Return the robust phase estimation circuit-builder type name."""
        return "robust_phase_estimation_circuit_builder"

    def default_algorithm_name(self) -> str:
        """Return the default QDK robust circuit-builder name."""
        return "qdk"


class QdkRobustPhaseEstimationCircuitBuilder(RobustPhaseEstimationCircuitBuilder):
    """QDK implementation of lazy robust phase estimation circuit generation."""

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Resolve the RPE schedule and return a lazy circuit set.

        Args:
            state_preparation: Circuit preparing the trial state.
            qubit_hamiltonian: Qubit Hamiltonian whose eigenenergy will be estimated.

        Returns:
            Lazy, re-iterable robust phase estimation circuit collection.

        """
        unitary_ref = self._settings.get("unitary_builder")
        hadamard_ref = self._settings.get("hadamard_test_circuit_builder")
        unitary_snapshot = _AlgorithmSnapshot.from_ref(unitary_ref)
        hadamard_snapshot = _AlgorithmSnapshot.from_ref(hadamard_ref)

        category = self._classify_builder(unitary_snapshot)
        correction = self._select_correction(category)
        epsilon_total = float(self._settings.get("target_accuracy"))
        fraction, epsilon_rpe, epsilon_unitary, budget_mode = self._resolve_budget(category, epsilon_total)

        lambda_norm = float(np.sum(np.abs(np.asarray(qubit_hamiltonian.coefficients, dtype=float))))
        base_time = float(self._settings.get("base_time"))
        if base_time <= 0.0:
            base_time = float(np.pi / (2.0 * lambda_norm)) if lambda_norm > 0.0 else 1.0

        total_round = num_rounds(lambda_norm, epsilon_rpe)
        randomized = category in ("qdrift", "partial_randomized")
        requested_seed = int(self._settings.get("seed"))
        root_seed = self._resolve_root_seed(requested_seed) if randomized else None

        rounds: list[RobustPhaseEstimationRound] = []
        for round_index in range(total_round + 1):
            shots, samples = qdrift_schedule(total_round, round_index)
            evolution_time = float((2**round_index) * base_time)
            updates: dict[str, object] = {"time": evolution_time}
            if category == "qdrift" and unitary_snapshot.has_setting("num_samples"):
                updates["num_samples"] = int(samples)
            elif unitary_snapshot.has_setting("target_accuracy"):
                updates["target_accuracy"] = float(epsilon_unitary)
            if not randomized and requested_seed >= 0 and unitary_snapshot.has_setting("seed"):
                updates["seed"] = requested_seed + round_index
            round_snapshot = unitary_snapshot.with_updates(**updates)

            if randomized:
                assert root_seed is not None
                draw_seeds = tuple(self._derive_seed(root_seed, round_index, draw) for draw in range(shots))
                num_draws = shots
                multiplicity = 1
            else:
                draw_seeds = ()
                num_draws = 1
                multiplicity = shots

            rounds.append(
                RobustPhaseEstimationRound(
                    round_index=round_index,
                    evolution_time=evolution_time,
                    shots_per_basis=shots,
                    num_draws=num_draws,
                    scheduled_samples=samples,
                    circuit_multiplicity=multiplicity,
                    draw_seeds=draw_seeds,
                    _unitary_builder_snapshot=round_snapshot,
                )
            )

        return RobustPhaseEstimationCircuitSet(
            rounds=tuple(rounds),
            lambda_norm=lambda_norm,
            base_time=base_time,
            target_accuracy=epsilon_total,
            epsilon_rpe=epsilon_rpe,
            epsilon_unitary=epsilon_unitary,
            unitary_accuracy_fraction=fraction,
            error_budget_mode=budget_mode,
            unitary_builder_category=category,
            energy_correction=correction,
            requested_seed=requested_seed,
            root_seed=root_seed,
            _state_preparation=state_preparation,
            _qubit_hamiltonian=qubit_hamiltonian,
            _hadamard_builder_snapshot=hadamard_snapshot,
        )

    @staticmethod
    def _classify_builder(snapshot: _AlgorithmSnapshot) -> str:
        """Classify the unitary builder from its supported settings."""
        if snapshot.has_setting("num_random_samples"):
            return "partial_randomized"
        if snapshot.has_setting("num_samples"):
            return "qdrift"
        return "deterministic_or_exact"

    def _select_correction(self, category: str) -> str:
        """Resolve the configured phase-to-energy correction."""
        mode = str(self._settings.get("energy_correction"))
        if mode != "auto":
            return mode
        return "qdrift_tangent" if category == "qdrift" else "linear"

    def _resolve_budget(self, category: str, epsilon_total: float) -> tuple[float, float, float, str]:
        """Resolve and validate the RPE and unitary error budgets."""
        fraction = min(max(float(self._settings.get("unitary_accuracy_fraction")), 0.0), 1.0)
        explicit_rpe = float(self._settings.get("epsilon_rpe"))
        explicit_unitary = float(self._settings.get("epsilon_unitary"))
        has_explicit_budget = explicit_rpe > 0.0 or explicit_unitary > 0.0

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
        epsilon_rpe = (1.0 - fraction) * epsilon_total
        if epsilon_rpe <= 0.0:
            epsilon_rpe = epsilon_total
        return fraction, epsilon_rpe, epsilon_unitary, "fraction"

    @staticmethod
    def _resolve_root_seed(requested_seed: int) -> int:
        """Return a concrete root seed for one circuit set."""
        if requested_seed >= 0:
            return requested_seed
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])

    @staticmethod
    def _derive_seed(root_seed: int, round_index: int, draw_index: int) -> int:
        """Derive one independent reproducible unitary-builder seed."""
        sequence = np.random.SeedSequence([root_seed, round_index, draw_index])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def name(self) -> str:
        """Return the QDK robust circuit-builder name."""
        return "qdk"
