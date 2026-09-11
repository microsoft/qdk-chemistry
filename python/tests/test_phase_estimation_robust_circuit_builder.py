"""Tests for robust phase estimation scheduling and circuit construction."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import islice, pairwise
from math import ceil, e, pi
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.base import HamiltonianUnitaryBuilder
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


class _FakeUnitaryBuilder(Trotter):
    """Record one on-demand unitary construction."""

    def __init__(self, settings: Settings, records: list[dict[str, object]], evolution_category: str) -> None:
        self._settings = settings
        self._records = records
        self._evolution_category = evolution_category

    def evolution_category(self) -> str:
        """Return the category of the replaced unitary builder."""
        return self._evolution_category

    def target_accuracy_from_unitary_tolerance(self, epsilon_unitary: float) -> float:
        """Map the unitary tolerance using the replaced builder's contract."""
        if self._evolution_category != "partial_randomized":
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
    [(0.0, 1.0, 0), (0.5, 1.0, 0), (1.0, 1.0, 0), (1.01, 1.0, 1), (8.0, 1.0, 3), (8.01, 1.0, 4)],
)
def test_num_rounds(
    rpe_problem: tuple[Circuit, QubitOperator], lambda_norm: float, epsilon: float, expected: int
) -> None:
    """The workload includes the base round and the expected number of doubling rounds."""
    state_preparation, _ = rpe_problem
    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=epsilon, seed=7).run(
        state_preparation, QubitOperator(pauli_strings=["Z"], coefficients=np.array([lambda_norm]))
    )

    assert circuit_set.num_rounds == expected + 1
    assert [round_data.round_index for round_data in circuit_set.rounds] == list(range(expected + 1))
    assert [round_data.evolution_time for round_data in circuit_set.rounds] == pytest.approx(
        [circuit_set.base_time * 2**round_index for round_index in range(expected + 1)]
    )


@pytest.mark.parametrize("epsilon", [0.0, -0.1])
def test_num_rounds_rejects_nonpositive_epsilon(rpe_problem: tuple[Circuit, QubitOperator], epsilon: float) -> None:
    """RPE scheduling requires a positive energy tolerance."""
    scheduler = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=epsilon)
    with pytest.raises(ValueError, match="epsilon"):
        scheduler.run(*rpe_problem)


def test_qdrift_schedule_formula_and_monotonicity(rpe_problem: tuple[Circuit, QubitOperator]) -> None:
    """RPE shots decrease while qDRIFT samples increase over the ladder."""
    state_preparation, _ = rpe_problem
    total_rounds = 5
    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=1.0, seed=7).run(
        state_preparation, QubitOperator(pauli_strings=["Z"], coefficients=np.array([32.0]))
    )
    shots = [round_data.shots_per_basis for round_data in circuit_set.rounds]
    samples = [round_data.scheduled_samples for round_data in circuit_set.rounds]

    assert circuit_set.num_rounds == total_rounds + 1
    assert shots == [ceil(e * (11 + 4 * (total_rounds - round_index))) for round_index in range(total_rounds + 1)]
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


@pytest.mark.parametrize(
    ("builder_name", "expected_json_hash", "expected_content_hash"),
    [
        (
            "trotter",
            "1345344da1d74542a51e4dfa2319db794646f6535159d1baa1cfd13e14654a2a",
            "d238cd0d2e24ec25901e7dac4456400564d654a147a6df558c2de2a2b53fc83c",
        ),
        (
            "qdrift",
            "2ad586f371b4adc44fc0917312bbac281e915a6acaeb1fc815a39bb77a07c6d2",
            "1b0fc46fb14cbcbb82322e3217874fdff6ffb735048f68066eedf91469f1bb94",
        ),
        (
            "partially_randomized",
            "129ee2ccc824591e93c488e7eadc197fbc51944335fa1fc202336488b9495b4e",
            "97762fb02d33ac1ec96bd45188cde4f4b7a39b53e36ee4b82824003ee1bb6804",
        ),
    ],
)
def test_builtin_schedule_preserves_serialized_baseline(
    rpe_problem: tuple[Circuit, QubitOperator],
    builder_name: str,
    expected_json_hash: str,
    expected_content_hash: str,
) -> None:
    """The capability refactor preserves fixed-seed schedules and their serialized format."""
    state_preparation, hamiltonian = rpe_problem
    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(
        target_accuracy=0.5,
        seed=17,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", builder_name),
    ).run(state_preparation, hamiltonian)
    payload = json.dumps(circuit_set.to_json(), sort_keys=True, separators=(",", ":"), allow_nan=False)

    assert hashlib.sha256(payload.encode()).hexdigest() == expected_json_hash
    assert circuit_set.content_hash(truncate_chars=0) == expected_content_hash


@pytest.mark.parametrize("quantum_walk", [False, True])
def test_scheduler_rejects_block_encoding_builders(
    rpe_problem: tuple[Circuit, QubitOperator], quantum_walk: bool
) -> None:
    """Block encodings and walks do not inherit time-evolution capabilities."""
    assert not hasattr(HamiltonianUnitaryBuilder, "evolution_category")
    assert not hasattr(HamiltonianUnitaryBuilder, "target_accuracy_from_unitary_tolerance")
    scheduler = QdkRobustPhaseEstimationExperimentScheduler(
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=quantum_walk)
    )

    with pytest.raises(TypeError, match="requires a TimeEvolutionBuilder"):
        scheduler.run(*rpe_problem)


@pytest.mark.parametrize("category", [None, 1, "unsupported"])
def test_scheduler_rejects_invalid_evolution_category(
    rpe_problem: tuple[Circuit, QubitOperator], monkeypatch: pytest.MonkeyPatch, category: object
) -> None:
    """An invalid capability result fails before any round is scheduled."""
    monkeypatch.setattr(Trotter, "evolution_category", lambda _self: category)
    scheduler = QdkRobustPhaseEstimationExperimentScheduler(
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter")
    )
    error_type = ValueError if isinstance(category, str) else TypeError

    with pytest.raises(error_type, match="evolution category"):
        scheduler.run(*rpe_problem)


@pytest.mark.parametrize("target_accuracy", ["0.5", float("nan"), float("inf"), -0.1])
def test_scheduler_rejects_invalid_target_accuracy_conversion(
    rpe_problem: tuple[Circuit, QubitOperator], monkeypatch: pytest.MonkeyPatch, target_accuracy: object
) -> None:
    """A builder's tolerance conversion must return a finite nonnegative number."""
    monkeypatch.setattr(Trotter, "target_accuracy_from_unitary_tolerance", lambda _self, _epsilon: target_accuracy)
    scheduler = QdkRobustPhaseEstimationExperimentScheduler(
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter")
    )
    error_type = ValueError if target_accuracy == -0.1 else TypeError

    with pytest.raises(error_type, match="RPE target accuracy"):
        scheduler.run(*rpe_problem)


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

    expected_shots = ceil(e * (11 + 4 * (circuit_set.num_rounds - 1)))
    expected_samples = 2
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
    builder = QdkRobustPhaseEstimationCircuitBuilder()
    circuits = builder.build(circuit_set)

    assert len(circuits) == 2 * len(circuit_set.experiment_specs)
    for spec in circuit_set.experiment_specs:
        assert isinstance(circuits[spec.x_circuit_index], Circuit)
        assert isinstance(circuits[spec.y_circuit_index], Circuit)
    assert len(unitary_records) == len(circuit_set.experiment_specs)
    assert len(hadamard_records) == len(circuits)

    eager_records = list(unitary_records)
    unitary_records.clear()
    streamed = list(builder.iter_build(circuit_set))

    assert unitary_records == eager_records
    assert [spec for spec, _, _ in streamed] == list(circuit_set.experiment_specs)
    assert [circuit.content_hash() for circuit in circuits] == [
        circuit.content_hash() for _, x_circuit, y_circuit in streamed for circuit in (x_circuit, y_circuit)
    ]


def test_run_matches_standard_qpe_list_contract(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal algorithm entry point schedules once and returns a flat circuit list."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    scheduler_ref = AlgorithmRef("rpe_experiment_scheduler", "qdk", target_accuracy=0.5, seed=17)
    builder = QdkRobustPhaseEstimationCircuitBuilder(experiment_scheduler=scheduler_ref)
    original_schedule = builder.schedule
    scheduled_workloads: list[RobustPhaseEstimationCircuitSet] = []

    def schedule(state_preparation: Circuit, qubit_hamiltonian: QubitOperator) -> RobustPhaseEstimationCircuitSet:
        circuit_set = original_schedule(state_preparation, qubit_hamiltonian)
        scheduled_workloads.append(circuit_set)
        return circuit_set

    monkeypatch.setattr(builder, "schedule", schedule)

    circuits = builder.run(state_preparation, hamiltonian)

    assert len(scheduled_workloads) == 1
    assert isinstance(circuits, list)
    assert len(circuits) == len(hadamard_records) == 2 * len(unitary_records)
    assert len(circuits) == 2 * len(scheduled_workloads[0].experiment_specs)


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
    original_seed_sequence = np.random.SeedSequence
    entropy_calls: list[int] = []

    def seed_sequence(entropy: int | list[int] | None = None) -> np.random.SeedSequence:
        if entropy is None:
            entropy_calls.append(1234)
            return original_seed_sequence(1234)
        return original_seed_sequence(entropy)

    monkeypatch.setattr(np.random, "SeedSequence", seed_sequence)

    circuit_set = QdkRobustPhaseEstimationExperimentScheduler(target_accuracy=0.5, seed=-1).run(
        state_preparation, hamiltonian
    )

    expected_root = int(original_seed_sequence(1234).generate_state(1, dtype=np.uint32)[0])
    expected_draw = int(original_seed_sequence([expected_root, 0, 0]).generate_state(1, dtype=np.uint32)[0])
    assert entropy_calls == [1234]
    assert circuit_set.requested_seed == -1
    assert circuit_set.root_seed == expected_root
    assert circuit_set.experiment_specs[0].draw_seed == expected_draw


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
