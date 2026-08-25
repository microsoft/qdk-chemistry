"""Tests for robust phase estimation scheduling and circuit construction."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice, pairwise
from math import ceil, e, pi
from typing import TYPE_CHECKING

import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.partially_randomized import (
    PartiallyRandomized,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder import (
    QdkRobustPhaseEstimationCircuitBuilder,
    RobustPhaseEstimationCircuitBuilder,
)
from qdk_chemistry.algorithms.phase_estimation.experiment_scheduler import (
    QdkRobustPhaseEstimationExperimentScheduler,
    RobustPhaseEstimationExperimentScheduler,
    _AlgorithmSnapshot,
    _num_rounds,
    _qdrift_schedule,
)
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    QubitOperator,
    RobustPhaseEstimationCircuitSet,
    RobustPhaseEstimationRound,
    SettingNotFoundError,
    Settings,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class _FakeUnitary:
    """Unitary marker used to verify X/Y circuit pairing."""

    seed: int | None


class _FakeUnitaryBuilder:
    """Record one on-demand unitary construction."""

    def __init__(self, settings: Settings, records: list[dict[str, object]], rpe_category: str) -> None:
        self._settings = settings
        self._records = records
        self._rpe_category = rpe_category

    def rpe_category(self) -> str:
        """Return the category of the replaced unitary builder."""
        return self._rpe_category

    def rpe_target_accuracy(self, epsilon_unitary: float) -> float:
        """Map an RPE unitary tolerance using the replaced builder's contract."""
        if self._rpe_category != "partial_randomized":
            return epsilon_unitary
        split = float(self._settings.get("accuracy_split"))
        split = min(max(split, 1e-6), 1.0 - 1e-6)
        return epsilon_unitary / ((split**0.5) + ((1.0 - split) ** 0.5))

    def run(self, qubit_hamiltonian: QubitOperator) -> _FakeUnitary:
        """Record settings and return a unitary marker."""
        record = self._settings.to_dict()
        record["num_qubits"] = qubit_hamiltonian.num_qubits
        self._records.append(record)
        seed = int(self._settings.get("seed")) if self._settings.has("seed") else None
        return _FakeUnitary(seed)


class _FakeHadamardBuilder:
    """Record one basis circuit built from a unitary marker."""

    def __init__(self, settings: Settings, records: list[tuple[str, _FakeUnitary]]) -> None:
        self._settings = settings
        self._records = records

    def run(self, state_preparation: Circuit, unitary: _FakeUnitary) -> Circuit:
        """Record the basis and shared unitary, then return a QASM circuit."""
        assert isinstance(state_preparation, Circuit)
        basis = str(self._settings.get("test_basis"))
        self._records.append((basis, unitary))
        return Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n")


class _RenamedTrotter(Trotter):
    """Trotter implementation registered under a non-built-in name."""

    def name(self) -> str:
        """Return a custom registry name."""
        return "renamed_trotter_for_rpe_test"


@pytest.fixture
def rpe_problem() -> tuple[Circuit, QubitOperator]:
    """Return a minimal state-preparation circuit and Hamiltonian."""
    state_preparation = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n")
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    return state_preparation, hamiltonian


@pytest.fixture
def recording_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]]:
    """Replace snapshot instantiation with recording unitary and Hadamard builders."""
    unitary_records: list[dict[str, object]] = []
    hadamard_records: list[tuple[str, _FakeUnitary]] = []

    def create_snapshot(snapshot: _AlgorithmSnapshot):
        settings = Settings.from_json(snapshot.settings_json)
        if snapshot.algorithm_type == "hamiltonian_unitary_builder":
            categories = {
                "trotter": "trotter",
                "qdrift": "qdrift",
                "partially_randomized": "partial_randomized",
            }
            return _FakeUnitaryBuilder(settings, unitary_records, categories[snapshot.algorithm_name])
        if snapshot.algorithm_type == "hadamard_test_circuit_builder":
            return _FakeHadamardBuilder(settings, hadamard_records)
        raise AssertionError(f"Unexpected algorithm type: {snapshot.algorithm_type}")

    monkeypatch.setattr(_AlgorithmSnapshot, "create", create_snapshot)
    return unitary_records, hadamard_records


def _copy_with_rounds(
    circuit_set: RobustPhaseEstimationCircuitSet,
    rounds: tuple[RobustPhaseEstimationRound, ...],
) -> RobustPhaseEstimationCircuitSet:
    """Copy a circuit set with replacement round metadata."""
    return RobustPhaseEstimationCircuitSet(
        rounds=rounds,
        experiment_specs=circuit_set.experiment_specs,
        state_preparation=circuit_set.state_preparation,
        qubit_hamiltonian=circuit_set.qubit_hamiltonian,
        lambda_norm=circuit_set.lambda_norm,
        base_time=circuit_set.base_time,
        target_accuracy=circuit_set.target_accuracy,
        epsilon_rpe=circuit_set.epsilon_rpe,
        epsilon_unitary=circuit_set.epsilon_unitary,
        unitary_accuracy_fraction=circuit_set.unitary_accuracy_fraction,
        error_budget_mode=circuit_set.error_budget_mode,
        unitary_builder_category=circuit_set.unitary_builder_category,
        energy_correction=circuit_set.energy_correction,
        requested_seed=circuit_set.requested_seed,
        root_seed=circuit_set.root_seed,
        hadamard_test_circuit_builder_configuration=(circuit_set.hadamard_test_circuit_builder_configuration),
    )


@pytest.mark.parametrize(
    ("lambda_norm", "epsilon", "expected"),
    [(1.0, 1.0, 0), (0.5, 1.0, 0), (8.0, 1.0, 3), (10.0, 1.0, 4)],
)
def test_num_rounds(lambda_norm: float, epsilon: float, expected: int) -> None:
    """The scheduler resolves the expected number of time-doubling rounds."""
    assert _num_rounds(lambda_norm, epsilon) == expected


def test_num_rounds_rejects_nonpositive_epsilon() -> None:
    """RPE scheduling requires a positive energy tolerance."""
    with pytest.raises(ValueError, match="epsilon"):
        _num_rounds(1.0, 0.0)


def test_qdrift_schedule_formula_and_monotonicity() -> None:
    """RPE shots decrease while qDRIFT samples increase over the ladder."""
    total_rounds = 5
    schedules = [_qdrift_schedule(total_rounds, round_index) for round_index in range(total_rounds + 1)]
    shots = [schedule[0] for schedule in schedules]
    samples = [schedule[1] for schedule in schedules]

    assert schedules[0] == (ceil(e * (11 + 4 * total_rounds)), 2)
    assert shots == sorted(shots, reverse=True)
    assert samples == sorted(samples)
    assert all(samples[round_index] == 2 ** (2 * round_index + 1) for round_index in range(total_rounds + 1))


def test_scheduler_and_builder_are_registered_and_scheduling_is_lazy(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """The registered scheduler creates metadata without constructing circuits."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    scheduler = create("rpe_experiment_scheduler", "qdk", target_accuracy=0.5, seed=7)
    builder = create("qpe_circuit_builder", "qdk_robust")

    circuit_set = scheduler.run(state_preparation, hamiltonian)

    assert isinstance(scheduler, RobustPhaseEstimationExperimentScheduler)
    assert isinstance(builder, RobustPhaseEstimationCircuitBuilder)
    assert isinstance(circuit_set, RobustPhaseEstimationCircuitSet)
    assert unitary_records == []
    assert hadamard_records == []
    assert len(circuit_set.experiment_specs) == sum(round_data.num_draws for round_data in circuit_set.rounds)


def test_robust_builder_rejects_standard_qpe_settings() -> None:
    """The shared builder type retains variant-specific settings schemas."""
    with pytest.raises(SettingNotFoundError):
        create("qpe_circuit_builder", "qdk_robust", num_bits=10)


@pytest.mark.parametrize("base_time", [pi, 1.1 * pi])
def test_explicit_base_time_rejects_aliasing_energy_interval(
    rpe_problem: tuple[Circuit, QubitOperator],
    base_time: float,
) -> None:
    """Explicit base times must distinguish every energy in the Hamiltonian norm bound."""
    state_preparation, hamiltonian = rpe_problem
    scheduler = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, base_time=base_time)

    with pytest.raises(ValueError, match=r"base_time \* lambda_norm < pi"):
        scheduler.run(state_preparation, hamiltonian)


def test_explicit_base_time_below_aliasing_limit_is_retained(
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """A safe explicit base time remains unchanged in the generated workload."""
    state_preparation, hamiltonian = rpe_problem
    base_time = pi * (1.0 - 1e-12)

    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, base_time=base_time).run(
        state_preparation, hamiltonian
    )

    assert circuit_set.base_time == pytest.approx(base_time)
    assert circuit_set.rounds[0].evolution_time == pytest.approx(base_time)


@pytest.mark.parametrize("epsilon_unitary", [None, 0.5])
def test_renamed_trotter_uses_same_rpe_policy(
    rpe_problem: tuple[Circuit, QubitOperator],
    epsilon_unitary: float | None,
) -> None:
    """A custom Trotter registry name preserves category-driven RPE behavior."""
    state_preparation, hamiltonian = rpe_problem
    algorithms.register(_RenamedTrotter)
    try:
        circuit_sets = []
        for builder_name in ("trotter", "renamed_trotter_for_rpe_test"):
            circuit_sets.append(
                QdkRobustPhaseEstimationExperimentScheduler(
                    target_accuracy=0.01,
                    epsilon_unitary=epsilon_unitary,
                    unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", builder_name),
                ).run(state_preparation, hamiltonian)
            )
    finally:
        algorithms.unregister("hamiltonian_unitary_builder", "renamed_trotter_for_rpe_test")

    builtin, renamed = circuit_sets
    expected_unitary_accuracy = 0.85 if epsilon_unitary is None else epsilon_unitary
    assert builtin.error_budget_mode == renamed.error_budget_mode == "independent_trotter"
    assert builtin.epsilon_rpe == renamed.epsilon_rpe == pytest.approx(0.01)
    assert builtin.epsilon_unitary == renamed.epsilon_unitary == pytest.approx(expected_unitary_accuracy)
    assert builtin.unitary_builder_category == renamed.unitary_builder_category == "deterministic_or_exact"
    assert builtin.num_rounds == renamed.num_rounds


def test_default_partial_randomized_random_cost_scales_quadratically(
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """Default PR sample growth follows the expected inverse-square RPE scaling."""
    state_preparation, _ = rpe_problem
    hamiltonian = QubitOperator(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
    random_rotation_counts: list[int] = []

    for target_accuracy in (0.1, 0.05, 0.025, 0.0125):
        circuit_set = QdkRobustPhaseEstimationExperimentScheduler(
            target_accuracy=target_accuracy,
            seed=7,
            unitary_builder=AlgorithmRef(
                "hamiltonian_unitary_builder",
                "partially_randomized",
                weight_threshold=0.75,
                num_random_samples=1,
                trotter_order=2,
            ),
        ).run(state_preparation, hamiltonian)
        final_round = circuit_set.rounds[-1]
        settings = final_round.unitary_builder_configuration.settings
        assert settings is not None
        partial_builder = PartiallyRandomized(**settings.to_dict())
        terms = hamiltonian.get_real_coefficients(tolerance=1e-12, sort_by_magnitude=True)
        random_terms = terms[1:]
        num_divisions = partial_builder._resolve_num_divisions(hamiltonian, final_round.evolution_time)
        block_samples = partial_builder._resolve_block_samples(random_terms, final_round.evolution_time, num_divisions)

        assert circuit_set.epsilon_rpe == pytest.approx(target_accuracy)
        assert circuit_set.epsilon_unitary == pytest.approx(0.85)
        assert settings.get("target_accuracy") == pytest.approx(0.85 / (2.0**0.5))
        random_rotation_counts.append(num_divisions * block_samples)

    ratios = [current / previous for previous, current in pairwise(random_rotation_counts)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios)


@pytest.mark.parametrize("builder_name", ["trotter", "qdrift", "partially_randomized"])
@pytest.mark.parametrize("power_strategy", ["repeat", "rescale"])
def test_nested_unitary_power_must_be_one(
    rpe_problem: tuple[Circuit, QubitOperator],
    builder_name: str,
    power_strategy: str,
) -> None:
    """RPE rejects every built-in unitary builder that applies an additional power."""
    state_preparation, hamiltonian = rpe_problem
    unitary_builder = AlgorithmRef(
        "hamiltonian_unitary_builder",
        builder_name,
        power=2,
        power_strategy=power_strategy,
    )
    scheduler = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, unitary_builder=unitary_builder)

    with pytest.raises(ValueError, match="unitary_builder power must be 1"):
        scheduler.run(state_preparation, hamiltonian)


@pytest.mark.parametrize("builder_name", ["trotter", "qdrift", "partially_randomized"])
def test_explicit_nested_unitary_power_one_is_accepted(
    rpe_problem: tuple[Circuit, QubitOperator],
    builder_name: str,
) -> None:
    """An explicit unit power preserves each built-in RPE evolution path."""
    state_preparation, hamiltonian = rpe_problem
    unitary_builder = AlgorithmRef("hamiltonian_unitary_builder", builder_name, power=1)

    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(
        target_accuracy=0.5,
        unitary_builder=unitary_builder,
    ).run(state_preparation, hamiltonian)

    assert all(round_data.unitary_builder_configuration.settings.get("power") == 1 for round_data in circuit_set.rounds)


def test_circuit_set_rejects_rebound_nested_unitary_power(
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """Reconstructing a workload cannot bypass the nested-power invariant."""
    state_preparation, hamiltonian = rpe_problem
    original = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5).run(state_preparation, hamiltonian)
    first_round = original.rounds[0]
    invalid_round = RobustPhaseEstimationRound(
        round_index=first_round.round_index,
        evolution_time=first_round.evolution_time,
        shots_per_basis=first_round.shots_per_basis,
        num_draws=first_round.num_draws,
        scheduled_samples=first_round.scheduled_samples,
        unitary_builder_configuration=AlgorithmRef("hamiltonian_unitary_builder", "trotter", power=2),
    )

    with pytest.raises(ValueError, match="unitary_builder power must be 1"):
        _copy_with_rounds(original, (invalid_round, *original.rounds[1:]))


def test_deterministic_round_builds_one_multi_shot_pair(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A deterministic round builds one shared-unitary pair measured many times."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    scheduler = QdkRobustPhaseEstimationExperimentScheduler(
        target_accuracy=0.5,
        seed=7,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
    )
    circuit_set = scheduler.run(state_preparation, hamiltonian)
    builder = QdkRobustPhaseEstimationCircuitBuilder()

    spec, _, _ = next(builder.iter_build(circuit_set))

    expected_shots, expected_samples = _qdrift_schedule(circuit_set.num_rounds - 1, 0)
    assert circuit_set.rounds[0].shots_per_basis == expected_shots
    assert circuit_set.rounds[0].scheduled_samples == expected_samples
    assert circuit_set.rounds[0].num_draws == 1
    assert spec.draw_index is None
    assert spec.draw_seed is None
    assert spec.shots == expected_shots
    assert len(unitary_records) == 1
    assert [basis for basis, _ in hadamard_records] == ["X", "Y"]
    assert hadamard_records[0][1] is hadamard_records[1][1]


def test_randomized_round_builds_independent_pairs_on_demand(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A randomized round builds independent seeded pairs only as requested."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, seed=11).run(
        state_preparation, hamiltonian
    )
    builder = QdkRobustPhaseEstimationCircuitBuilder()

    experiments = list(islice(builder.iter_build(circuit_set), 2))
    expected_specs = circuit_set.experiment_specs_for_round(0)[:2]

    assert [experiment[0] for experiment in experiments] == list(expected_specs)
    assert [record["seed"] for record in unitary_records] == [spec.draw_seed for spec in expected_specs]
    assert len(unitary_records) == 2
    assert len(hadamard_records) == 4
    assert hadamard_records[0][1] is hadamard_records[1][1]
    assert hadamard_records[2][1] is hadamard_records[3][1]


def test_build_returns_canonical_flat_circuit_list(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Eager construction matches the standard QPE list contract and manifest positions."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, seed=17).run(
        state_preparation, hamiltonian
    )
    circuits = QdkRobustPhaseEstimationCircuitBuilder().build(circuit_set)

    assert len(circuits) == 2 * len(circuit_set.experiment_specs)
    for spec in circuit_set.experiment_specs:
        assert isinstance(circuits[spec.x_circuit_index], Circuit)
        assert isinstance(circuits[spec.y_circuit_index], Circuit)
    assert len(unitary_records) == len(circuit_set.experiment_specs)
    assert len(hadamard_records) == len(circuits)


def test_run_matches_standard_qpe_list_contract(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """The normal algorithm entry point schedules once and returns a flat circuit list."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    scheduler_ref = AlgorithmRef("rpe_experiment_scheduler", "qdk", target_accuracy=0.5, seed=17)
    builder = QdkRobustPhaseEstimationCircuitBuilder(experiment_scheduler=scheduler_ref)

    circuits = builder.run(state_preparation, hamiltonian)

    assert isinstance(circuits, list)
    assert len(circuits) == len(hadamard_records) == 2 * len(unitary_records)


def test_streamed_pair_supports_qre(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A selected streamed circuit pair converts directly to QRE applications."""
    from qdk.qre.application import OpenQASMApplication  # noqa: PLC0415

    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, seed=17).run(
        state_preparation, hamiltonian
    )
    spec, x_circuit, y_circuit = next(islice(QdkRobustPhaseEstimationCircuitBuilder().iter_build(circuit_set), 1, 2))

    assert spec == circuit_set.experiment_specs[1]
    assert isinstance(x_circuit.get_qre_application(), OpenQASMApplication)
    assert isinstance(y_circuit.get_qre_application(), OpenQASMApplication)
    assert unitary_records[-1]["seed"] == spec.draw_seed
    assert hadamard_records[-2][1] is hadamard_records[-1][1]


def test_stream_reiteration_replays_seeded_draws(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Re-iterating one workload rebuilds the same randomized draw sequence."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, _ = recording_builders
    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, seed=13).run(
        state_preparation, hamiltonian
    )
    builder = QdkRobustPhaseEstimationCircuitBuilder()

    first = next(builder.iter_build(circuit_set))[0]
    second = next(builder.iter_build(circuit_set))[0]

    assert first.draw_seed == second.draw_seed
    assert [record["seed"] for record in unitary_records] == [first.draw_seed, first.draw_seed]


def test_serialized_workload_rebinds_and_replays_seeded_draw(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A round-tripped workload regenerates the same draw after rebinding live inputs."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    original = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, seed=23).run(
        state_preparation, hamiltonian
    )
    restored = RobustPhaseEstimationCircuitSet.from_json(json.loads(json.dumps(original.to_json())))
    rebound = restored.rebind(state_preparation)

    spec, _, _ = next(islice(QdkRobustPhaseEstimationCircuitBuilder().iter_build(rebound), 2, 3))

    assert spec.draw_seed == original.experiment_specs[2].draw_seed
    assert unitary_records[-1]["seed"] == spec.draw_seed
    assert hadamard_records[-2][1] is hadamard_records[-1][1]


@pytest.mark.parametrize("suffix", ["json", "hdf5"])
def test_serialized_circuit_set_remains_lazy_and_builds_on_demand(
    tmp_path: Path,
    suffix: str,
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A loaded workload retains inputs and materializes only through the builder."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    original = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, seed=11).run(
        state_preparation, hamiltonian
    )
    filename = tmp_path / f"sample.robust_phase_estimation_circuit_set.{suffix}"
    original.to_file(filename, suffix)

    restored = RobustPhaseEstimationCircuitSet.from_file(filename, suffix)

    assert restored.content_hash() == original.content_hash()
    assert unitary_records == []
    assert hadamard_records == []

    spec, _, _ = next(QdkRobustPhaseEstimationCircuitBuilder().iter_build(restored))

    assert spec.draw_seed == restored.experiment_specs[0].draw_seed
    assert len(unitary_records) == 1
    assert [basis for basis, _ in hadamard_records] == ["X", "Y"]


def test_entropy_seed_is_concretized_once_per_circuit_set(
    monkeypatch: pytest.MonkeyPatch,
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """The nondeterministic sentinel becomes one replayable root seed."""
    state_preparation, hamiltonian = rpe_problem
    monkeypatch.setattr(
        QdkRobustPhaseEstimationExperimentScheduler,
        "_resolve_root_seed",
        staticmethod(lambda _seed: 1234),
    )

    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, seed=-1).run(
        state_preparation, hamiltonian
    )

    assert circuit_set.requested_seed == -1
    assert circuit_set.root_seed == 1234
    assert circuit_set.experiment_specs[0].draw_seed == QdkRobustPhaseEstimationExperimentScheduler._derive_seed(
        1234, 0, 0
    )


def test_round_configuration_is_defensive_and_round_index_is_validated(
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """Configuration access returns copies and invalid round indices fail clearly."""
    state_preparation, hamiltonian = rpe_problem
    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(
        target_accuracy=0.5,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
    ).run(state_preparation, hamiltonian)
    round_zero = circuit_set.rounds[0]
    first_config = round_zero.unitary_builder_configuration
    assert first_config.settings is not None
    first_config.settings.set("time", 99.0)

    second_config = round_zero.unitary_builder_configuration

    assert second_config.settings is not None
    assert second_config.settings.get("time") == pytest.approx(round_zero.evolution_time)
    with pytest.raises(IndexError, match="round_index"):
        circuit_set.experiment_specs_for_round(circuit_set.num_rounds)
