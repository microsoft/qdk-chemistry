"""Workload scheduling for robust phase estimation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json

import numpy as np

from qdk_chemistry.algorithms.base import Algorithm, AlgorithmFactory
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import TimeEvolutionBuilder
from qdk_chemistry.data import (
    AlgorithmRef,
    QubitOperator,
    RobustPhaseEstimationRound,
    RobustPhaseEstimationSchedule,
    Settings,
)
from qdk_chemistry.data.robust_phase_estimation import _AlgorithmConfiguration, _HamiltonianSnapshot

__all__ = [
    "RobustPhaseEstimationExperimentScheduler",
    "RobustPhaseEstimationExperimentSchedulerFactory",
    "RobustPhaseEstimationExperimentSchedulerSettings",
]

_UNSET_BUDGET_VALUE = -1.0
_DEFAULT_RPE_EPSILON_UNITARY = 0.85
_RPE_BUILDER_CATEGORIES = {
    "trotter": "deterministic_or_exact",
    "qdrift": "qdrift",
    "partially_randomized": "partial_randomized",
    "zassenhaus": "deterministic_or_exact",
}


class _AlgorithmSnapshot(_AlgorithmConfiguration):
    """Algorithm-layer operations on independent configuration snapshots."""

    @classmethod
    def from_ref(cls, ref: AlgorithmRef) -> _AlgorithmSnapshot:
        """Capture a resolved reference for algorithm construction.

        Args:
            ref: Reference with resolved settings to copy.

        Returns:
            A configuration snapshot with algorithm-layer operations.

        """
        configuration = _AlgorithmConfiguration.from_ref(ref)
        return cls(configuration.algorithm_type, configuration.algorithm_name, configuration.settings_json)

    def has_setting(self, key: str) -> bool:
        """Check whether the configuration defines a setting.

        Args:
            key: Setting name to look up.

        Returns:
            Whether this setting is present.

        """
        return Settings.from_json(self.settings_json).has(key)

    def with_updates(self, **updates: object) -> _AlgorithmSnapshot:
        """Prepare a new configuration without mutating the stored snapshot.

        Args:
            **updates: Defined setting names and replacement values.

        Returns:
            An independent updated configuration.

        Raises:
            ValueError: If an update refers to an undefined setting.

        """
        settings = Settings.from_json(self.settings_json)
        for key, value in updates.items():
            if not settings.has(key):
                raise ValueError(
                    f"Algorithm '{self.algorithm_type}/{self.algorithm_name}' does not define setting '{key}'."
                )
            settings.set(key, value)
        return type(self)(self.algorithm_type, self.algorithm_name, settings.to_json())

    def create(self, **updates: object) -> Algorithm:
        """Construct a fresh registered algorithm from this configuration.

        Args:
            **updates: Round-specific settings applied through the algorithm's settings schema.

        Returns:
            An algorithm initialized with independent settings.

        """
        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        settings = Settings.from_json(self.settings_json)
        values = settings.to_dict()
        values.update(updates)
        return create(self.algorithm_type, self.algorithm_name, **values)

    def validate_unit_power(self) -> None:
        """Require the RPE schedule to be the sole source of evolution powers.

        Raises:
            TypeError: If the configured power is not an integer.
            ValueError: If the configured power is not one.

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
        self._set_default(
            "max_qdrift_samples",
            "int",
            1_000_000,
            "Maximum samples per scheduled qDRIFT circuit. Increase explicitly only when resources permit.",
        )


class RobustPhaseEstimationExperimentScheduler(Algorithm):
    """QDK implementation of reproducible robust phase estimation scheduling."""

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
        max_qdrift_samples: int = 1_000_000,
    ) -> None:
        """Initialize robust phase estimation workload scheduling.

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
        self._settings = RobustPhaseEstimationExperimentSchedulerSettings()
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

    def type_name(self) -> str:
        """Return the RPE experiment-scheduler type name.

        Returns:
            ``"rpe_experiment_scheduler"``.

        """
        return "rpe_experiment_scheduler"

    def _run_impl(
        self,
        qubit_hamiltonian: QubitOperator,
    ) -> RobustPhaseEstimationSchedule:
        """Resolve rounds, randomized draws, and execution metadata.

        The workload includes round zero and every subsequent time doubling.
        Deterministic rounds use one multi-shot X/Y pair. Randomized rounds use
        one independently seeded unitary per shot, shared by the two bases.
        For nonzero Hamiltonians, the final evolution time is at least
        ``pi / (2 * epsilon_rpe)``, including when ``base_time`` is explicit.
        qDRIFT counts include any tighter nested accuracy request and must not
        exceed ``max_qdrift_samples``. Counts are never silently clamped.

        Args:
            qubit_hamiltonian: Hamiltonian used to choose the norm, time ladder, and nested builder settings.

        Returns:
            A reproducible schedule with rounds, shared settings, and concrete draw seeds, without live inputs.

        Raises:
            TypeError: If the configured builder is not a time-evolution builder.
            ValueError: If the builder name, power, base time, or error-budget settings are unsupported or invalid.

        """
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
        hadamard_ref = self._settings.get("hadamard_test_circuit_builder")
        unitary_snapshot = _AlgorithmSnapshot.from_ref(unitary_ref)
        hadamard_snapshot = _AlgorithmSnapshot.from_ref(hadamard_ref)
        unitary_snapshot.validate_unit_power()

        unitary_builder = unitary_snapshot.create()
        if not isinstance(unitary_builder, TimeEvolutionBuilder):
            raise TypeError(
                "RPE requires a TimeEvolutionBuilder; "
                f"'{unitary_snapshot.algorithm_type}/{unitary_snapshot.algorithm_name}' "
                "does not represent supported time evolution. Block encodings and quantum walks are not supported."
            )
        builder_name = unitary_snapshot.algorithm_name
        if builder_name not in _RPE_BUILDER_CATEGORIES:
            supported = ", ".join(sorted(_RPE_BUILDER_CATEGORIES))
            raise ValueError(f"Unsupported RPE unitary builder {builder_name!r}. Supported builders: {supported}.")
        category = _RPE_BUILDER_CATEGORIES[builder_name]
        correction = str(self._settings.get("energy_correction"))
        if correction == "auto":
            correction = "qdrift_tangent" if category == "qdrift" else "linear"
        fraction, epsilon_rpe, epsilon_unitary, budget_mode = self._resolve_budget(
            category,
            epsilon_total,
            is_trotter=builder_name == "trotter",
        )

        lambda_norm = float(np.sum(np.abs(np.asarray(qubit_hamiltonian.coefficients, dtype=float))))
        if base_time == 0.0:
            base_time = float(np.pi / (2.0 * lambda_norm)) if lambda_norm > 0.0 else 1.0
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
        if category == "qdrift":
            final_scheduled_samples = 2 ** (2 * total_round + 1)
            if final_scheduled_samples > max_qdrift_samples:
                raise ValueError(
                    f"qDRIFT round {total_round} requires at least {final_scheduled_samples} samples, exceeding "
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
            samples = 2 ** (2 * round_index + 1)
            evolution_time = float((2**round_index) * base_time)
            if category == "qdrift" and unitary_snapshot.has_setting("num_samples"):
                if (
                    unitary_snapshot.has_setting("target_accuracy")
                    and unitary_builder.settings().get("target_accuracy") > 0.0
                ):
                    sample_resolver = getattr(unitary_builder, "_resolve_num_samples", None)
                    if not callable(sample_resolver):
                        raise TypeError("qDRIFT builders with target_accuracy must support sample-count resolution.")
                    unitary_builder.settings().set("time", evolution_time)
                    unitary_builder.settings().set("num_samples", samples)
                    resolved_samples = sample_resolver(qubit_hamiltonian, evolution_time)
                    if not isinstance(resolved_samples, int) or resolved_samples < samples:
                        raise ValueError(
                            "Resolved qDRIFT samples must be an integer at least as large as the RPE count."
                        )
                    samples = resolved_samples
                if samples > max_qdrift_samples:
                    raise ValueError(
                        f"qDRIFT round {round_index} requires {samples} samples, exceeding "
                        f"max_qdrift_samples={max_qdrift_samples}. Raise max_qdrift_samples explicitly only "
                        "if sufficient resources are available."
                    )
            draw_seeds: list[int | None] = []
            if randomized:
                assert root_seed is not None
                for draw_index in range(shots):
                    sequence = np.random.SeedSequence([root_seed, round_index, draw_index])
                    draw_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
                    draw_seeds.append(draw_seed)
            else:
                seed = None
                if unitary_snapshot.has_setting("seed"):
                    configured_seed = int(unitary_builder.settings().get("seed"))
                    seed = requested_seed + round_index if requested_seed >= 0 else configured_seed
                    if seed < 0:
                        seed = None
                draw_seeds.append(seed)

            rounds.append(
                RobustPhaseEstimationRound(
                    round_index=round_index,
                    evolution_time=evolution_time,
                    shots_per_basis=shots,
                    scheduled_samples=samples,
                    draw_seeds=tuple(draw_seeds),
                )
            )

        shared_settings = json.loads(unitary_snapshot.settings_json)
        for key in ("time", "seed", "num_samples"):
            shared_settings.pop(key, None)
        if category != "qdrift" and "target_accuracy" in shared_settings:
            shared_settings["target_accuracy"] = epsilon_unitary
        shared_configuration = _AlgorithmConfiguration(
            unitary_snapshot.algorithm_type, unitary_snapshot.algorithm_name, json.dumps(shared_settings)
        )
        return RobustPhaseEstimationSchedule(
            rounds=tuple(rounds),
            hamiltonian_hash=_HamiltonianSnapshot.from_operator(qubit_hamiltonian).content_hash(),
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
            unitary_builder_configuration=shared_configuration.to_ref(),
            hadamard_test_circuit_builder_configuration=hadamard_snapshot.to_ref(),
        )

    def _resolve_budget(
        self,
        category: str,
        epsilon_total: float,
        *,
        is_trotter: bool,
    ) -> tuple[float, float, float, str]:
        """Resolve and validate the RPE and unitary error budgets.

        Args:
            category: Normalized evolution category used to choose a budget policy.
            epsilon_total: Requested final energy accuracy.
            is_trotter: Whether to use the independent Trotter tolerance policy.

        Returns:
            The legacy fraction, RPE energy tolerance, unitary tolerance, and budget-mode name.

        Raises:
            ValueError: If the configured tolerances or legacy options are invalid for the selected policy.

        """
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

    def name(self) -> str:
        """Return the QDK scheduler name.

        Returns:
            ``"qdk"``.

        """
        return "qdk"


class RobustPhaseEstimationExperimentSchedulerFactory(AlgorithmFactory):
    """Factory for robust phase estimation experiment schedulers."""

    def algorithm_type_name(self) -> str:
        """Return the RPE experiment-scheduler type name.

        Returns:
            ``"rpe_experiment_scheduler"``.

        """
        return "rpe_experiment_scheduler"

    def default_algorithm_name(self) -> str:
        """Return the default QDK scheduler name.

        Returns:
            ``"qdk"``.

        """
        return "qdk"
