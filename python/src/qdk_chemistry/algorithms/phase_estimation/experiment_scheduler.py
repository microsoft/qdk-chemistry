"""Workload scheduling for robust phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitOperator,
    RobustPhaseEstimationCircuitSet,
    RobustPhaseEstimationExperimentSpec,
    RobustPhaseEstimationRound,
    Settings,
)
from qdk_chemistry.data.robust_phase_estimation import (
    _AlgorithmConfiguration as _AlgorithmSnapshot,
)

__all__ = [
    "QdkRobustPhaseEstimationExperimentScheduler",
    "RobustPhaseEstimationExperimentScheduler",
    "RobustPhaseEstimationExperimentSchedulerFactory",
    "RobustPhaseEstimationExperimentSchedulerSettings",
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


class RobustPhaseEstimationExperimentSchedulerSettings(Settings):
    """Settings for robust phase estimation workload scheduling."""

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


class RobustPhaseEstimationExperimentScheduler(Algorithm):
    """Abstract workload scheduler for robust phase estimation."""

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
        """Initialize robust phase estimation workload scheduling."""
        super().__init__()
        self._settings = RobustPhaseEstimationExperimentSchedulerSettings()
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
        """Return the RPE experiment-scheduler type name."""
        return "rpe_experiment_scheduler"

    @abstractmethod
    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Resolve and return a reproducible RPE workload."""


class RobustPhaseEstimationExperimentSchedulerFactory(AlgorithmFactory):
    """Factory for robust phase estimation experiment schedulers."""

    def algorithm_type_name(self) -> str:
        """Return the RPE experiment-scheduler type name."""
        return "rpe_experiment_scheduler"

    def default_algorithm_name(self) -> str:
        """Return the default QDK scheduler name."""
        return "qdk"


class QdkRobustPhaseEstimationExperimentScheduler(RobustPhaseEstimationExperimentScheduler):
    """QDK implementation of reproducible robust phase estimation scheduling."""

    def _run_impl(
        self,
        state_preparation: Circuit,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationCircuitSet:
        """Resolve rounds, randomized draws, and execution metadata."""
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
        experiment_specs: list[RobustPhaseEstimationExperimentSpec] = []
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
                for draw_index, draw_seed in enumerate(draw_seeds):
                    experiment_specs.append(
                        RobustPhaseEstimationExperimentSpec(
                            experiment_index=len(experiment_specs),
                            round_index=round_index,
                            draw_index=draw_index,
                            draw_seed=draw_seed,
                            shots=1,
                        )
                    )
                num_draws = shots
            else:
                experiment_specs.append(
                    RobustPhaseEstimationExperimentSpec(
                        experiment_index=len(experiment_specs),
                        round_index=round_index,
                        draw_index=None,
                        draw_seed=None,
                        shots=shots,
                    )
                )
                num_draws = 1

            rounds.append(
                RobustPhaseEstimationRound(
                    round_index=round_index,
                    evolution_time=evolution_time,
                    shots_per_basis=shots,
                    num_draws=num_draws,
                    scheduled_samples=samples,
                    unitary_builder_configuration=round_snapshot.to_ref(),
                )
            )

        return RobustPhaseEstimationCircuitSet(
            rounds=tuple(rounds),
            experiment_specs=tuple(experiment_specs),
            state_preparation=state_preparation,
            qubit_hamiltonian=qubit_hamiltonian,
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
            hadamard_test_circuit_builder_configuration=hadamard_snapshot.to_ref(),
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
        """Return the QDK scheduler name."""
        return "qdk"


def _validate_unitary_builder_power(snapshot: _AlgorithmSnapshot) -> None:
    """Require RPE to be the sole owner of the evolution power schedule."""
    snapshot.validate_unit_power()
