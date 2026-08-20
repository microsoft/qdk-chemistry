"""On-demand circuit generation for robust phase estimation."""

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
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitOperator,
    RobustPhaseEstimationExperiment,
    RobustPhaseEstimationRound,
    RobustPhaseEstimationSchedule,
    Settings,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "QdkRobustPhaseEstimationCircuitBuilder",
    "RobustPhaseEstimationCircuitBuilder",
    "RobustPhaseEstimationCircuitBuilderFactory",
    "RobustPhaseEstimationCircuitBuilderSettings",
    "RobustPhaseEstimationCircuitSet",
]

_UNSET_BUDGET_VALUE = -1.0
_DEFAULT_RPE_EPSILON_UNITARY = 0.85
_SUPPORTED_RPE_CATEGORIES = frozenset({"deterministic_or_exact", "trotter", "qdrift", "partial_randomized"})


def _num_rounds(lambda_norm: float, epsilon: float) -> int:
    """Return the number of RPE time-doubling rounds after the base round."""
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, received {epsilon}.")
    if lambda_norm < 0.0:
        raise ValueError(f"lambda_norm must be non-negative, received {lambda_norm}.")
    if lambda_norm <= epsilon:
        return 0
    return int(np.ceil(np.log2(lambda_norm / epsilon)))


def _qdrift_schedule(total_rounds: int, round_index: int) -> tuple[int, int]:
    """Return the per-basis shot count and qDRIFT sample count for one round."""
    shots = int(np.ceil(np.e * (11 + 4 * (total_rounds - round_index))))
    samples = 2 ** (2 * round_index + 1)
    return shots, samples


@dataclass(frozen=True)
class _AlgorithmSnapshot:
    """Immutable snapshot of an algorithm reference and its nested settings."""

    algorithm_type: str
    """Registry type of the snapshotted algorithm."""

    algorithm_name: str
    """Registered implementation name of the snapshotted algorithm."""

    settings_json: str
    """Serialized settings used to reconstruct independent algorithm instances."""

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

    def get_setting(self, key: str) -> object:
        """Return one setting value from an independent settings copy."""
        settings = Settings.from_json(self.settings_json)
        return settings.get(key)

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
class RobustPhaseEstimationCircuitSet:
    """Re-iterable robust phase estimation circuit collection that generates pairs on demand."""

    rounds: tuple[RobustPhaseEstimationRound, ...]
    """Resolved round schedule in execution order."""

    lambda_norm: float
    """One-norm of the qubit Hamiltonian coefficients."""

    base_time: float
    """Evolution time used for round zero."""

    target_accuracy: float
    """Requested absolute accuracy of the final energy estimate."""

    epsilon_rpe: float
    """Energy tolerance used to determine the number of RPE rounds."""

    epsilon_unitary: float
    """Resolved full-evolution tolerance before builder-specific mapping."""

    unitary_accuracy_fraction: float
    """Resolved legacy fraction used to partition non-Trotter accuracy budgets."""

    error_budget_mode: str
    """Mode used to resolve the RPE and unitary accuracy parameters."""

    unitary_builder_category: str
    """Category of the configured unitary builder."""

    energy_correction: str
    """Resolved phase-to-energy conversion method."""

    requested_seed: int
    """Root seed requested in the builder settings, where ``-1`` requests entropy."""

    root_seed: int | None
    """Concrete root seed for randomized draws, or ``None`` for deterministic evolution."""

    _state_preparation: Circuit = field(repr=False)
    """Trial-state circuit retained for on-demand circuit generation."""

    _qubit_hamiltonian: QubitOperator = field(repr=False)
    """Qubit Hamiltonian retained for on-demand unitary construction."""

    _hadamard_builder_snapshot: _AlgorithmSnapshot = field(repr=False)
    """Hadamard-test builder configuration retained for circuit generation."""

    @classmethod
    def from_schedule(
        cls,
        schedule: RobustPhaseEstimationSchedule,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Bind a serialized schedule to live inputs for on-demand circuit generation.

        Args:
            schedule: Serialized RPE workload recipe.
            state_preparation: Circuit preparing the trial state.
            qubit_hamiltonian: Qubit Hamiltonian used to build scheduled unitaries.

        Returns:
            A runtime circuit set that materializes circuits from ``schedule`` on demand.

        """
        for round_data in schedule.rounds:
            _validate_unitary_builder_power(_AlgorithmSnapshot.from_ref(round_data.unitary_builder_configuration))
        return cls(
            rounds=schedule.rounds,
            lambda_norm=schedule.lambda_norm,
            base_time=schedule.base_time,
            target_accuracy=schedule.target_accuracy,
            epsilon_rpe=schedule.epsilon_rpe,
            epsilon_unitary=schedule.epsilon_unitary,
            unitary_accuracy_fraction=schedule.unitary_accuracy_fraction,
            error_budget_mode=schedule.error_budget_mode,
            unitary_builder_category=schedule.unitary_builder_category,
            energy_correction=schedule.energy_correction,
            requested_seed=schedule.requested_seed,
            root_seed=schedule.root_seed,
            _state_preparation=state_preparation,
            _qubit_hamiltonian=qubit_hamiltonian,
            _hadamard_builder_snapshot=_AlgorithmSnapshot.from_ref(
                schedule.hadamard_test_circuit_builder_configuration
            ),
        )

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

    @property
    def schedule(self) -> RobustPhaseEstimationSchedule:
        """Return a serializable schedule containing no materialized circuits."""
        return RobustPhaseEstimationSchedule(
            rounds=self.rounds,
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

    def __iter__(self) -> Iterator[RobustPhaseEstimationExperiment]:
        """Generate every circuit pair on demand in round and draw order."""
        for round_data in self.rounds:
            yield from self._iter_round(round_data)

    def iter_round(self, round_index: int) -> Iterator[RobustPhaseEstimationExperiment]:
        """Generate circuit pairs on demand for one round.

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

    def get_experiment(self, round_index: int, draw_index: int | None = None) -> RobustPhaseEstimationExperiment:
        """Generate one X/Y circuit pair for execution or resource estimation.

        Args:
            round_index: Zero-based round index.
            draw_index: Zero-based randomized draw index. Use ``None`` for deterministic evolution.

        Returns:
            The requested circuit-pair experiment and its execution metadata.

        Raises:
            IndexError: If ``round_index`` or ``draw_index`` is outside the circuit set.
            ValueError: If ``draw_index`` does not match the round's deterministic or randomized schedule.

        """
        if round_index < 0 or round_index >= len(self.rounds):
            raise IndexError(f"round_index must be in [0, {len(self.rounds) - 1}], got {round_index}.")
        round_data = self.rounds[round_index]
        if round_data.draw_seeds:
            if draw_index is None:
                raise ValueError("draw_index is required for randomized evolution.")
            if draw_index < 0 or draw_index >= len(round_data.draw_seeds):
                raise IndexError(f"draw_index must be in [0, {len(round_data.draw_seeds) - 1}], got {draw_index}.")
            draw_seed = round_data.draw_seeds[draw_index]
        else:
            if draw_index is not None:
                raise ValueError("draw_index must be None for deterministic evolution.")
            draw_seed = None

        unitary_snapshot = _AlgorithmSnapshot.from_ref(round_data.unitary_builder_configuration)
        _validate_unitary_builder_power(unitary_snapshot)
        if draw_seed is not None and unitary_snapshot.has_setting("seed"):
            unitary_snapshot = unitary_snapshot.with_updates(seed=draw_seed)

        unitary_builder = unitary_snapshot.create()
        unitary = unitary_builder.run(self._qubit_hamiltonian)

        x_builder = self._hadamard_builder_snapshot.with_updates(test_basis="X").create()
        x_circuit = x_builder.run(self._state_preparation, unitary)
        y_builder = self._hadamard_builder_snapshot.with_updates(test_basis="Y").create()
        y_circuit = y_builder.run(self._state_preparation, unitary)

        return RobustPhaseEstimationExperiment(
            round_index=round_data.round_index,
            evolution_time=round_data.evolution_time,
            shots_per_basis=round_data.shots_per_basis,
            draw_index=draw_index,
            draw_seed=draw_seed,
            circuit_multiplicity=round_data.circuit_multiplicity,
            x_circuit=x_circuit,
            y_circuit=y_circuit,
            unitary_builder_configuration=unitary_snapshot.to_ref(),
        )

    def get_circuit(self, round_index: int, basis: str, draw_index: int | None = None) -> Circuit:
        """Generate one concrete X- or Y-basis circuit.

        Args:
            round_index: Zero-based round index.
            basis: Hadamard-test basis, either ``"X"`` or ``"Y"``.
            draw_index: Zero-based randomized draw index. Use ``None`` for deterministic evolution.

        Returns:
            The requested concrete circuit, suitable for execution or QRE.

        Raises:
            ValueError: If ``basis`` is not ``"X"`` or ``"Y"``.

        """
        normalized_basis = basis.upper()
        if normalized_basis not in ("X", "Y"):
            raise ValueError(f"basis must be 'X' or 'Y', got {basis!r}.")
        experiment = self.get_experiment(round_index, draw_index)
        return experiment.x_circuit if normalized_basis == "X" else experiment.y_circuit

    def _iter_round(self, round_data: RobustPhaseEstimationRound) -> Iterator[RobustPhaseEstimationExperiment]:
        if round_data.draw_seeds:
            draw_indices: tuple[int | None, ...] = tuple(range(len(round_data.draw_seeds)))
        else:
            draw_indices = (None,)

        for draw_index in draw_indices:
            yield self.get_experiment(round_data.round_index, draw_index)


class RobustPhaseEstimationCircuitBuilderSettings(Settings):
    """Settings for robust phase estimation circuit generation."""

    def __init__(self) -> None:
        """Initialize nested circuit algorithms and RPE schedule settings."""
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
            "target_accuracy",
            "double",
            1e-3,
            "Requested absolute accuracy epsilon on the final energy estimate.",
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
            "omitted partially randomized builders use an independent unitary tolerance.",
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
        unitary_accuracy_fraction: float | None = None,
        energy_correction: str = "auto",
        seed: int = -1,
        epsilon_rpe: float | None = None,
        epsilon_unitary: float | None = None,
        unitary_builder: AlgorithmRef | None = None,
        hadamard_test_circuit_builder: AlgorithmRef | None = None,
    ) -> None:
        """Initialize robust phase estimation circuit generation.

        Args:
            target_accuracy: Requested absolute accuracy on the final energy.
            base_time: Round-zero time; positive values require ``base_time*lambda < pi`` and ``0.0`` is automatic.
            unitary_accuracy_fraction: Optional legacy non-Trotter fraction of ``target_accuracy``.
            energy_correction: Phase-to-energy map: ``"auto"``, ``"linear"``, or ``"qdrift_tangent"``.
            seed: Root random seed; ``-1`` chooses one entropy-backed seed per circuit set.
            epsilon_rpe: Optional explicit RPE energy tolerance for non-Trotter builders.
            epsilon_unitary: Optional unitary tolerance; Trotter and partially randomized builders default to ``0.85``.
            unitary_builder: Optional time-evolution builder reference whose ``power`` must be ``1``.
            hadamard_test_circuit_builder: Optional Hadamard-test circuit-builder reference.

        """
        super().__init__()
        self._settings = RobustPhaseEstimationCircuitBuilderSettings()
        self._settings.set("target_accuracy", target_accuracy)
        self._settings.set("base_time", base_time)
        self._settings.set("energy_correction", energy_correction)
        self._settings.set("seed", seed)
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

    def type_name(self) -> str:
        """Return the robust phase estimation circuit-builder type name."""
        return "robust_phase_estimation_circuit_builder"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Build an on-demand robust phase estimation circuit set."""


class RobustPhaseEstimationCircuitBuilderFactory(AlgorithmFactory):
    """Factory for robust phase estimation circuit builders."""

    def algorithm_type_name(self) -> str:
        """Return the robust phase estimation circuit-builder type name."""
        return "robust_phase_estimation_circuit_builder"

    def default_algorithm_name(self) -> str:
        """Return the default QDK robust circuit-builder name."""
        return "qdk"


class QdkRobustPhaseEstimationCircuitBuilder(RobustPhaseEstimationCircuitBuilder):
    """QDK implementation of on-demand robust phase estimation circuit generation."""

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Resolve the RPE schedule and return an on-demand circuit set.

        Args:
            state_preparation: Circuit preparing the trial state.
            qubit_hamiltonian: Qubit Hamiltonian whose eigenenergy will be estimated.

        Returns:
            Re-iterable robust phase estimation circuit collection that generates pairs on demand.

        """
        unitary_ref = self._settings.get("unitary_builder")
        hadamard_ref = self._settings.get("hadamard_test_circuit_builder")
        unitary_snapshot = _AlgorithmSnapshot.from_ref(unitary_ref)
        hadamard_snapshot = _AlgorithmSnapshot.from_ref(hadamard_ref)
        _validate_unitary_builder_power(unitary_snapshot)

        unitary_builder = unitary_snapshot.create()
        declared_category = self._resolve_rpe_category(unitary_snapshot, unitary_builder)
        category = "deterministic_or_exact" if declared_category == "trotter" else declared_category
        correction = self._select_correction(category)
        epsilon_total = float(self._settings.get("target_accuracy"))
        fraction, epsilon_rpe, epsilon_unitary, budget_mode = self._resolve_budget(
            category,
            epsilon_total,
            is_trotter=declared_category == "trotter",
        )
        nested_epsilon_unitary = self._resolve_rpe_target_accuracy(
            unitary_snapshot,
            unitary_builder,
            epsilon_unitary,
        )

        lambda_norm = float(np.sum(np.abs(np.asarray(qubit_hamiltonian.coefficients, dtype=float))))
        base_time = float(self._settings.get("base_time"))
        if base_time <= 0.0:
            base_time = float(np.pi / (2.0 * lambda_norm)) if lambda_norm > 0.0 else 1.0
        elif base_time * lambda_norm >= np.pi:
            raise ValueError(
                "base_time must satisfy base_time * lambda_norm < pi to avoid energy aliasing; "
                f"got base_time={base_time:.6g} and lambda_norm={lambda_norm:.6g}."
            )

        total_round = _num_rounds(lambda_norm, epsilon_rpe)
        randomized = category in ("qdrift", "partial_randomized")
        requested_seed = int(self._settings.get("seed"))
        root_seed = self._resolve_root_seed(requested_seed) if randomized else None

        rounds: list[RobustPhaseEstimationRound] = []
        for round_index in range(total_round + 1):
            shots, samples = _qdrift_schedule(total_round, round_index)
            evolution_time = float((2**round_index) * base_time)
            updates: dict[str, object] = {"time": evolution_time}
            if category == "qdrift" and unitary_snapshot.has_setting("num_samples"):
                updates["num_samples"] = int(samples)
            elif unitary_snapshot.has_setting("target_accuracy"):
                updates["target_accuracy"] = nested_epsilon_unitary
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
                    unitary_builder_configuration=round_snapshot.to_ref(),
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
    def _resolve_rpe_category(snapshot: _AlgorithmSnapshot, builder: Algorithm) -> str:
        """Return and validate the unitary builder's declared RPE category."""
        category_resolver = getattr(builder, "rpe_category", None)
        if not callable(category_resolver):
            raise TypeError(
                f"Unitary builder '{snapshot.algorithm_type}/{snapshot.algorithm_name}' must implement rpe_category()."
            )
        category = category_resolver()
        if not isinstance(category, str):
            raise TypeError(
                f"Unitary builder '{snapshot.algorithm_type}/{snapshot.algorithm_name}' returned a non-string "
                "RPE category."
            )
        if category not in _SUPPORTED_RPE_CATEGORIES:
            supported = ", ".join(sorted(_SUPPORTED_RPE_CATEGORIES))
            raise ValueError(
                f"Unitary builder '{snapshot.algorithm_type}/{snapshot.algorithm_name}' returned unsupported "
                f"RPE category {category!r}; expected one of: {supported}."
            )
        return category

    @staticmethod
    def _resolve_rpe_target_accuracy(
        snapshot: _AlgorithmSnapshot,
        builder: Algorithm,
        epsilon_unitary: float,
    ) -> float:
        """Map and validate the RPE unitary tolerance for a nested builder."""
        target_resolver = getattr(builder, "rpe_target_accuracy", None)
        if target_resolver is None:
            return epsilon_unitary
        if not callable(target_resolver):
            raise TypeError(
                f"Unitary builder '{snapshot.algorithm_type}/{snapshot.algorithm_name}' defines a non-callable "
                "rpe_target_accuracy attribute."
            )
        target_accuracy = target_resolver(epsilon_unitary)
        if not isinstance(target_accuracy, int | float) or not np.isfinite(target_accuracy):
            raise TypeError(
                f"Unitary builder '{snapshot.algorithm_type}/{snapshot.algorithm_name}' returned an invalid "
                "RPE target accuracy."
            )
        if target_accuracy < 0.0:
            raise ValueError(
                f"Unitary builder '{snapshot.algorithm_type}/{snapshot.algorithm_name}' returned a negative "
                "RPE target accuracy."
            )
        return float(target_accuracy)

    def _select_correction(self, category: str) -> str:
        """Resolve the configured phase-to-energy correction."""
        mode = str(self._settings.get("energy_correction"))
        if mode != "auto":
            return mode
        return "qdrift_tangent" if category == "qdrift" else "linear"

    def _resolve_budget(
        self,
        category: str,
        epsilon_total: float,
        *,
        is_trotter: bool,
    ) -> tuple[float, float, float, str]:
        """Resolve and validate the RPE and unitary error budgets."""
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

        fraction = 0.5 if configured_fraction == _UNSET_BUDGET_VALUE else min(max(configured_fraction, 0.0), 1.0)
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


def _validate_unitary_builder_power(snapshot: _AlgorithmSnapshot) -> None:
    """Require RPE to be the sole owner of the evolution power schedule."""
    if not snapshot.has_setting("power"):
        return
    power = snapshot.get_setting("power")
    if not isinstance(power, int):
        raise TypeError(f"unitary_builder power must be an integer, got {type(power).__name__}.")
    if power != 1:
        raise ValueError(
            "Robust phase estimation controls evolution powers through its round-time schedule; "
            f"unitary_builder power must be 1, got {power}."
        )
