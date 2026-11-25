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
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.qdrift_error import _qdrift_lambda
from qdk_chemistry.data import (
    AlgorithmRef,
    QubitOperator,
    RobustPhaseEstimationRound,
    RobustPhaseEstimationSchedule,
    Settings,
)

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


def _round_sample_floor(round_index: int, base_time_ratio: float) -> int:
    """Round the scaled qDRIFT ladder up without converting its potentially large integer counts to floats."""
    numerator, denominator = max(1.0, base_time_ratio**2).as_integer_ratio()
    return (2 ** (2 * round_index + 1) * numerator + denominator - 1) // denominator


def _validate_unit_power(settings: Settings) -> None:
    """Require the RPE schedule to be the sole source of evolution powers.

    Args:
        settings: Resolved unitary-builder settings.

    Raises:
        TypeError: If the configured power is not an integer.
        ValueError: If the configured power is not one.

    """
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
        Deterministic rounds use one multi-shot X/Y pair without a draw seed.
        Randomized rounds use one independently seeded unitary per shot, shared by the two bases.
        For nonzero Hamiltonians, the final evolution time is at least
        ``pi / (2 * epsilon_rpe)``, including when ``base_time`` is explicit.
        qDRIFT counts include any tighter nested accuracy request and must not
        exceed ``max_qdrift_samples``. Every count is checked before any draw seed
        is generated, and counts are never silently clamped.

        Args:
            qubit_hamiltonian: Hermitian Hamiltonian used to choose the norm, time ladder, and nested builder settings.

        Returns:
            A reproducible schedule with rounds, shared settings, and concrete draw seeds, without live inputs.

        Raises:
            KeyError: If the configured unitary builder is not registered.
            TypeError: If the Hamiltonian is not a QubitOperator or the builder is not a time-evolution builder.
            ValueError: If the Hamiltonian is non-Hermitian or a builder, correction, power, time, or budget is invalid.

        """
        if not isinstance(qubit_hamiltonian, QubitOperator):
            raise TypeError(
                f"The {self.name()!r} RPE scheduler takes a QubitOperator, got {type(qubit_hamiltonian).__name__}."
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
        hadamard_ref = self._settings.get("hadamard_test_circuit_builder")
        # Create first so an unregistered builder name raises the registry's KeyError.
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

        lambda_norm = float(qubit_hamiltonian.schatten_norm)
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
        base_time_ratio = base_time / automatic_base_time if category == "qdrift" and lambda_norm > 0.0 else 1.0
        round_samples = [_round_sample_floor(round_index, base_time_ratio) for round_index in range(total_round + 1)]
        if category == "qdrift":
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
        for round_index, samples in enumerate(round_samples):
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
                    scheduled_samples=samples,
                    draw_seeds=draw_seeds,
                )
            )

        shared_settings = json.loads(unitary_settings.to_json())
        for key in ("time", "seed", "num_samples"):
            shared_settings.pop(key, None)
        if category != "qdrift" and "target_accuracy" in shared_settings:
            shared_settings["target_accuracy"] = epsilon_unitary
        shared_configuration = AlgorithmRef(
            unitary_ref.algorithm_type,
            builder_name,
            settings=Settings.from_json(json.dumps(shared_settings)),
        )
        return RobustPhaseEstimationSchedule(
            rounds=tuple(rounds),
            hamiltonian_hash=RobustPhaseEstimationSchedule.hamiltonian_fingerprint(qubit_hamiltonian),
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
            unitary_builder_configuration=shared_configuration,
            hadamard_test_circuit_builder_configuration=hadamard_ref,
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
