"""Tests for robust phase estimation scheduling and circuit construction."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
import warnings
from dataclasses import FrozenInstanceError, dataclass, fields, is_dataclass, replace
from itertools import islice, pairwise
from math import ceil, e, pi
from unittest.mock import patch

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.partially_randomized import (
    PartiallyRandomized,
)
from qdk_chemistry.algorithms.hamiltonian_unitary_builder.time_evolution.trotter import Trotter
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder import (
    RobustPhaseEstimationCircuitBuilder,
    RobustPhaseEstimationCircuitSet,
    RobustPhaseEstimationSchedule,
)
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    DataClass,
    FermiHubbardModelHamiltonianDescription,
    LatticeGeometry,
    QubitOperator,
    SettingNotFoundError,
    Settings,
    SettingsAreLocked,
)


@dataclass(frozen=True)
class _FakeUnitary:
    """Unitary marker used to verify X/Y circuit pairing."""

    seed: int | None
    time: float


def _fake_qasm(basis: str, unitary: _FakeUnitary) -> str:
    """Return a QASM program that identifies one basis and unitary draw."""
    return f"OPENQASM 3.0;\n// {basis} time={unitary.time!r} seed={unitary.seed}\nqubit[1] q;\n"


class _FakeUnitaryBuilder:
    """Record one on-demand unitary construction."""

    def __init__(self, settings: Settings, records: list[dict[str, object]]) -> None:
        self._settings = settings
        self._records = records

    def run(self, qubit_hamiltonian: QubitOperator) -> _FakeUnitary:
        """Record settings and return a unitary marker."""
        record = self._settings.to_dict()
        record["num_qubits"] = qubit_hamiltonian.num_qubits
        self._records.append(record)
        seed = int(self._settings.get("seed")) if self._settings.has("seed") else None
        return _FakeUnitary(seed, float(self._settings.get("time")))


class _FakeHadamardBuilder:
    """Record one basis circuit built from a unitary marker."""

    def __init__(self, settings: Settings, records: list[tuple[str, _FakeUnitary]]) -> None:
        self._settings = settings
        self._records = records

    def run(self, state_preparation: Circuit, unitary: _FakeUnitary) -> Circuit:
        """Record the basis and shared unitary, then return a QASM circuit identifying both."""
        assert isinstance(state_preparation, Circuit)
        basis = str(self._settings.get("test_basis"))
        self._records.append((basis, unitary))
        return Circuit(qasm=_fake_qasm(basis, unitary))


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
    """Replace factory construction with recording unitary and Hadamard builders."""
    unitary_records: list[dict[str, object]] = []
    hadamard_records: list[tuple[str, _FakeUnitary]] = []
    original_create = algorithms.create

    def create_algorithm(algorithm_type: str, algorithm_name: str, **settings_values: object) -> object:
        settings = original_create(algorithm_type, algorithm_name, **settings_values).settings()
        if algorithm_type == "hamiltonian_unitary_builder":
            return _FakeUnitaryBuilder(settings, unitary_records)
        if algorithm_type == "hadamard_test_circuit_builder":
            return _FakeHadamardBuilder(settings, hadamard_records)
        raise AssertionError(f"Unexpected algorithm type: {algorithm_type}")

    monkeypatch.setattr(algorithms, "create", create_algorithm)
    return unitary_records, hadamard_records


@pytest.mark.parametrize(
    ("lambda_norm", "epsilon", "expected"),
    [(0.0, 1.0, 0), (0.5, 1.0, 0), (1.0, 1.0, 0), (1.01, 1.0, 1), (8.0, 1.0, 3), (8.01, 1.0, 4)],
)
def test_num_rounds(lambda_norm: float, epsilon: float, expected: int) -> None:
    """The workload includes the base round and the expected number of doubling rounds."""
    circuit_set = RobustPhaseEstimationCircuitBuilder(target_accuracy=epsilon, seed=7).build_schedule(
        QubitOperator(pauli_strings=["Z"], coefficients=np.array([lambda_norm]))
    )

    assert circuit_set.num_rounds == expected + 1
    assert [round_data.round_index for round_data in circuit_set.rounds] == list(range(expected + 1))
    assert [round_data.evolution_time for round_data in circuit_set.rounds] == pytest.approx(
        [circuit_set.base_time * 2**round_index for round_index in range(expected + 1)]
    )


@pytest.mark.parametrize("epsilon", [0.0, -0.1])
def test_num_rounds_rejects_nonpositive_epsilon(rpe_problem: tuple[Circuit, QubitOperator], epsilon: float) -> None:
    """RPE scheduling requires a positive energy tolerance."""
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=epsilon)
    with pytest.raises(ValueError, match="epsilon"):
        builder.build_schedule(rpe_problem[1])


@pytest.mark.parametrize("builder_name", ["trotter", "qdrift", "partially_randomized"])
@pytest.mark.parametrize("setting", ["target_accuracy", "unitary_accuracy_fraction", "epsilon_rpe", "epsilon_unitary"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_schedule_rejects_nonfinite_budget_settings(
    rpe_problem: tuple[Circuit, QubitOperator],
    builder_name: str,
    setting: str,
    value: float,
) -> None:
    """Scheduling inputs must be finite before any family-specific budget resolution."""
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", builder_name),
    )
    builder.settings().set(setting, value)

    with pytest.raises(ValueError, match=rf"{setting}.*finite"):
        builder.build_schedule(rpe_problem[1])


def test_schedule_rejects_nan_explicit_budget(rpe_problem: tuple[Circuit, QubitOperator]) -> None:
    """A pair of NaN tolerances must not fall back to a finite fractional budget."""
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.1,
        epsilon_rpe=float("nan"),
        epsilon_unitary=float("nan"),
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "partially_randomized"),
    )

    with pytest.raises(ValueError, match=r"epsilon_rpe.*finite"):
        builder.build_schedule(rpe_problem[1])


def test_qdrift_schedule_formula_and_monotonicity() -> None:
    """Check the full ground-state stress-test ladder without constructing circuits."""
    total_rounds = 7
    circuit_set = RobustPhaseEstimationCircuitBuilder(target_accuracy=1e-2, seed=42).build_schedule(
        QubitOperator(pauli_strings=["XX", "ZZ"], coefficients=np.array([0.25, 0.5]))
    )
    shots = [round_data.shots_per_basis for round_data in circuit_set.rounds]
    samples = [round_data.scheduled_samples for round_data in circuit_set.rounds]

    assert circuit_set.num_rounds == total_rounds + 1
    assert len(circuit_set.experiment_specs) == 548
    assert shots == [ceil(e * (11 + 4 * (total_rounds - round_index))) for round_index in range(total_rounds + 1)]
    assert shots == sorted(shots, reverse=True)
    assert samples == sorted(samples)
    assert all(samples[round_index] == 2 ** (2 * round_index + 1) for round_index in range(total_rounds + 1))


@pytest.mark.parametrize(
    ("base_time", "counts"),
    [(0.0, (2, 8, 32, 128, 512)), (1.0, (2, 8, 32, 128, 512)), (3.0, (8, 30, 117, 467))],
)
def test_qdrift_sample_floor_includes_base_time(base_time: float, counts: tuple[int, ...]) -> None:
    """Only longer base times raise the reference floor; equality with the cap is allowed."""
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.1, base_time=base_time, seed=7, max_qdrift_samples=counts[-1]
    )
    schedule = builder.build_schedule(QubitOperator(["ZI", "IZ"], [0.6, 0.4]))

    assert tuple(round_data.scheduled_samples for round_data in schedule.rounds) == counts


def test_qdrift_long_base_time_checks_scaled_ceiling() -> None:
    """An explicit time may exceed the ceiling even when the old round-only floor fits."""
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.1, base_time=3.0, max_qdrift_samples=466)
    with pytest.raises(ValueError, match="467 samples.*max_qdrift_samples=466"):
        builder.build_schedule(QubitOperator(["ZI", "IZ"], [0.6, 0.4]))


def test_qdrift_replay_uses_recorded_sample_counts(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Construction uses the given schedule without recomputing its sample ladder."""
    state_preparation, hamiltonian = rpe_problem
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.1, base_time=3.0, seed=7)
    schedule = builder.build_schedule(hamiltonian)
    original_counts = (2, 8, 32, 128)
    restored = replace(
        schedule,
        rounds=tuple(
            replace(round_data, scheduled_samples=count)
            for round_data, count in zip(schedule.rounds, original_counts, strict=True)
        ),
    )
    assert tuple(round_data.scheduled_samples for round_data in restored.rounds) == original_counts
    assert restored.experiment_specs == schedule.experiment_specs
    spec, _, _ = next(builder.iter_build(restored, state_preparation, hamiltonian))
    unitary_records, _ = recording_builders
    assert unitary_records[0]["num_samples"] == original_counts[0]
    assert unitary_records[0]["seed"] == spec.draw_seed


@pytest.mark.parametrize("nested_accuracy", [0.0, 0.25, 10.0])
def test_qdrift_schedule_records_effective_samples(
    rpe_problem: tuple[Circuit, QubitOperator], nested_accuracy: float
) -> None:
    """Scheduled samples include nested accuracy sizing and match the constructed draw."""
    state_preparation, hamiltonian = rpe_problem
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        seed=7,
        unitary_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            "qdrift",
            target_accuracy=nested_accuracy,
            merge_duplicate_terms=False,
        ),
    )

    circuit_set = builder.build_schedule(hamiltonian)

    for round_data in circuit_set.rounds:
        configuration = circuit_set.unitary_builder_configuration
        unitary_builder = create(
            configuration.algorithm_type,
            configuration.algorithm_name,
            **configuration.settings.to_dict(),
            time=round_data.evolution_time,
            num_samples=round_data.scheduled_samples,
        )
        container = unitary_builder.run(hamiltonian).get_container()

        assert round_data.scheduled_samples == len(container.step_terms)
        assert not configuration.settings.has("num_samples")
        assert configuration.settings.get("target_accuracy") == nested_accuracy
        assert round_data.scheduled_samples >= 2 ** (2 * round_data.round_index + 1)
    assert circuit_set.final_samples == circuit_set.rounds[-1].scheduled_samples


def test_qdrift_schedule_rejects_default_oversized_workload() -> None:
    """Default chemistry-scale sampling fails before constructing a huge circuit."""
    builder = RobustPhaseEstimationCircuitBuilder(seed=7)
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[100.0])

    with pytest.raises(ValueError, match=r"34359738368.*max_qdrift_samples=1000000"):
        builder.build_schedule(hamiltonian)


@pytest.mark.parametrize("nested_accuracy", [0.0, 0.25])
def test_qdrift_schedule_reports_ceiling_beyond_native_integers(nested_accuracy: float) -> None:
    """Counts too large for native settings still raise the explicit sample-ceiling error."""
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=1e-10,
        seed=7,
        max_qdrift_samples=50,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "qdrift", target_accuracy=nested_accuracy),
    )

    with pytest.raises(ValueError, match=r"round 34 requires \d+ samples, exceeding max_qdrift_samples=50"):
        builder.build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=[1.0]))


@pytest.mark.parametrize("limit", [0, -1])
def test_qdrift_schedule_rejects_nonpositive_sample_limit(
    rpe_problem: tuple[Circuit, QubitOperator], limit: int
) -> None:
    """A sample ceiling must be a positive integer, not a disabling sentinel."""
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, max_qdrift_samples=limit)

    with pytest.raises(ValueError, match=r"max_qdrift_samples.*positive"):
        builder.build_schedule(rpe_problem[1])


@pytest.mark.parametrize("limit", [8, 9])
def test_qdrift_schedule_accepts_sample_limit_boundary(rpe_problem: tuple[Circuit, QubitOperator], limit: int) -> None:
    """An allowed sample count is unchanged at or below the configured ceiling."""
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, max_qdrift_samples=limit)

    circuit_set = builder.build_schedule(rpe_problem[1])

    assert circuit_set.final_samples == 8


def test_qdrift_schedule_rejects_samples_above_limit(rpe_problem: tuple[Circuit, QubitOperator]) -> None:
    """A ladder count above the ceiling raises rather than silently clamping."""
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, max_qdrift_samples=7)

    with pytest.raises(ValueError, match=r"8.*max_qdrift_samples=7"):
        builder.build_schedule(rpe_problem[1])


@pytest.mark.parametrize("sparse_reference", [False, True])
def test_qdrift_schedule_limit_uses_effective_samples(
    rpe_problem: tuple[Circuit, QubitOperator], sparse_reference: bool
) -> None:
    """Nested accuracy sizing uses the builder's resolved defaults and fails before drawing any seed."""
    unitary_builder = AlgorithmRef("hamiltonian_unitary_builder", "qdrift", target_accuracy=0.25)
    if sparse_reference:
        override = {"target_accuracy": json.loads(unitary_builder.settings.to_json())["target_accuracy"]}
        unitary_builder = AlgorithmRef(
            "hamiltonian_unitary_builder", "qdrift", settings=Settings.from_json(json.dumps(override))
        )
        assert not unitary_builder.settings.has("num_samples")
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5, seed=7, max_qdrift_samples=50, unitary_builder=unitary_builder
    )

    with (
        patch.object(QubitOperator, "to_json", side_effect=AssertionError("Rejected workloads need no snapshot")),
        patch.object(np.random, "SeedSequence", wraps=np.random.SeedSequence) as seed_sequence,
        pytest.raises(ValueError, match=r"79.*max_qdrift_samples=50"),
    ):
        builder.build_schedule(rpe_problem[1])
    assert seed_sequence.call_count == 0


def test_qdrift_schedule_accepts_explicit_higher_sample_limit() -> None:
    """An explicit ceiling permits a large metadata-only schedule without clamping."""
    builder = RobustPhaseEstimationCircuitBuilder(seed=7, max_qdrift_samples=2**35)
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[100.0])

    circuit_set = builder.build_schedule(hamiltonian)

    assert circuit_set.final_samples == 2**35


def test_registered_builder_schedules_without_building_circuits(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """The registered robust builder can plan a workload without constructing any circuits."""
    _, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    builder = create("qpe_circuit_builder", "qdk_robust", target_accuracy=0.5, seed=7)

    circuit_set = builder.build_schedule(hamiltonian)

    assert isinstance(builder, RobustPhaseEstimationCircuitBuilder)
    assert (builder.type_name(), builder.name()) == ("qpe_circuit_builder", "qdk_robust")
    with pytest.raises(SettingsAreLocked):
        builder.settings().set("seed", 42)
    assert isinstance(circuit_set, RobustPhaseEstimationSchedule)
    assert not hasattr(circuit_set, "state_preparation")
    assert not hasattr(circuit_set, "qubit_hamiltonian")
    assert unitary_records == []
    assert hadamard_records == []
    assert len(circuit_set.experiment_specs) == sum(round_data.num_draws for round_data in circuit_set.rounds)


def test_robust_builder_rejects_standard_qpe_settings() -> None:
    """The shared builder type retains variant-specific settings schemas."""
    with pytest.raises(SettingNotFoundError):
        create("qpe_circuit_builder", "qdk_robust", num_bits=10)


@pytest.mark.parametrize("base_time", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_schedule_rejects_invalid_base_time(rpe_problem: tuple[Circuit, QubitOperator], base_time: float) -> None:
    """Only finite, nonnegative times are valid, with zero selecting automatic time."""
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, base_time=base_time)

    with pytest.raises(ValueError, match=r"base_time.*finite and non-negative"):
        builder.build_schedule(rpe_problem[1])


@pytest.mark.parametrize("base_time", [pi, 1.1 * pi])
def test_explicit_base_time_rejects_aliasing_energy_interval(
    rpe_problem: tuple[Circuit, QubitOperator],
    base_time: float,
) -> None:
    """Explicit base times must distinguish every energy in the Hamiltonian norm bound."""
    state_preparation, hamiltonian = rpe_problem
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, base_time=base_time)

    with pytest.raises(ValueError, match=r"base_time \* lambda_norm < pi"):
        builder.build_schedule(hamiltonian)


def test_explicit_base_time_below_aliasing_limit_is_retained(
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """A safe explicit base time remains unchanged in the generated workload."""
    state_preparation, hamiltonian = rpe_problem
    base_time = pi * (1.0 - 1e-12)

    circuit_set = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, base_time=base_time).build_schedule(
        hamiltonian
    )

    assert circuit_set.base_time == pytest.approx(base_time)
    assert circuit_set.rounds[0].evolution_time == pytest.approx(base_time)


def test_schedule_rejects_unrecognized_builder_name(rpe_problem: tuple[Circuit, QubitOperator]) -> None:
    """RPE requires an explicit scheduling policy for each supported registered builder name."""
    _, hamiltonian = rpe_problem
    algorithms.register(_RenamedTrotter)
    try:
        builder = RobustPhaseEstimationCircuitBuilder(
            target_accuracy=0.5,
            unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "renamed_trotter_for_rpe_test"),
        )
        with pytest.raises(ValueError, match=r"Unsupported RPE unitary builder.*renamed_trotter_for_rpe_test"):
            builder.build_schedule(hamiltonian)
    finally:
        algorithms.unregister("hamiltonian_unitary_builder", "renamed_trotter_for_rpe_test")


@pytest.mark.parametrize("overrides", [{}, {"order": 2}])
def test_schedule_records_resolved_default_builder(overrides: dict[str, int]) -> None:
    """An empty unitary builder name dispatches and replays as the factory's default Trotter builder."""
    hamiltonian = QubitOperator(["Z"], [1.0])

    def schedule(name: str) -> RobustPhaseEstimationSchedule:
        return RobustPhaseEstimationCircuitBuilder(
            target_accuracy=0.5, seed=7, unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", name, **overrides)
        ).build_schedule(hamiltonian)

    implicit = schedule("")
    assert implicit.unitary_builder_configuration.algorithm_name == "trotter"
    assert implicit == schedule("trotter")


@pytest.mark.parametrize("batch_size", [None, 7])
def test_schedule_snapshots_resolved_hadamard_settings(batch_size: int | None) -> None:
    """Sparse Hadamard and mapper settings keep their overrides and are independent of returned copies."""
    overrides = Settings.from_json('{"test_basis":"Y"}')
    if batch_size is not None:
        mapper = AlgorithmRef(
            "controlled_circuit_mapper",
            "pauli_sequence",
            settings=Settings.from_json(json.dumps({"max_hamming_weight_phasing_batch_size": batch_size})),
        )
        overrides._set_default("controlled_circuit_mapper", "algorithm_ref", mapper)
    builder = RobustPhaseEstimationCircuitBuilder(
        seed=7,
        target_accuracy=0.5,
        hadamard_test_circuit_builder=AlgorithmRef("hadamard_test_circuit_builder", "qdk", settings=overrides),
    )

    schedule = builder.build_schedule(QubitOperator(["Z"], [1.0]))

    settings = schedule.hadamard_test_circuit_builder_configuration.settings
    assert settings.get("test_basis") == "Y"
    mapper_settings = settings.get("controlled_circuit_mapper").settings
    assert mapper_settings.has("control_indices")
    assert mapper_settings.get("max_hamming_weight_phasing_batch_size") == (-1 if batch_size is None else batch_size)
    mapper_settings.set("max_hamming_weight_phasing_batch_size", 42)
    another_mapper = schedule.hadamard_test_circuit_builder_configuration.settings.get("controlled_circuit_mapper")
    assert another_mapper.settings.get("max_hamming_weight_phasing_batch_size") == (
        -1 if batch_size is None else batch_size
    )


@pytest.mark.parametrize("builder_name", ["trotter", "qdrift", "partially_randomized"])
def test_builtin_schedule_preserves_experiment_baseline(
    rpe_problem: tuple[Circuit, QubitOperator],
    builder_name: str,
) -> None:
    """The canonical format preserves the established times, shots, and draw-seed policy."""
    _, hamiltonian = rpe_problem
    schedule = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        seed=17,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", builder_name),
    ).build_schedule(hamiltonian)
    assert [round_data.evolution_time for round_data in schedule.rounds] == pytest.approx([pi / 2, pi])
    assert [round_data.shots_per_basis for round_data in schedule.rounds] == [41, 30]
    expected_samples: list[int | None] = [2, 8] if builder_name == "qdrift" else [None, None]
    assert [round_data.scheduled_samples for round_data in schedule.rounds] == expected_samples
    assert schedule.final_samples == expected_samples[-1]
    if builder_name == "trotter":
        assert [round_data.draw_seeds for round_data in schedule.rounds] == [(None,), (None,)]
    else:
        for round_data in schedule.rounds:
            expected = tuple(
                int(np.random.SeedSequence([17, round_data.round_index, index]).generate_state(1, dtype=np.uint32)[0])
                for index in range(round_data.shots_per_basis)
            )
            assert round_data.draw_seeds == expected


def test_partial_schedule_preserves_additive_accuracy(rpe_problem: tuple[Circuit, QubitOperator]) -> None:
    """Partially randomized snapshots retain the additive tolerance and configured sample floor."""
    schedule = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        seed=17,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "partially_randomized"),
    ).build_schedule(rpe_problem[1])
    configuration = schedule.unitary_builder_configuration
    assert configuration.settings.get("target_accuracy") == pytest.approx(schedule.epsilon_unitary)
    assert configuration.settings.get("num_random_samples") == 100
    builder = create(configuration.algorithm_type, configuration.algorithm_name, **configuration.settings.to_dict())
    assert sum(builder._split_accuracy()) == pytest.approx(schedule.epsilon_unitary)


@pytest.mark.parametrize("quantum_walk", [False, True])
def test_schedule_rejects_block_encoding_builders(
    rpe_problem: tuple[Circuit, QubitOperator], quantum_walk: bool
) -> None:
    """Block encodings and walks are not supported time-evolution builders."""
    builder = RobustPhaseEstimationCircuitBuilder(
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "lcu", quantum_walk=quantum_walk)
    )

    with pytest.raises(TypeError, match="requires a TimeEvolutionBuilder"):
        builder.build_schedule(rpe_problem[1])


def test_schedule_rejects_unregistered_builder_name(rpe_problem: tuple[Circuit, QubitOperator]) -> None:
    """A misspelled builder name reports the registry's available builders."""
    builder = RobustPhaseEstimationCircuitBuilder(
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "does_not_exist")
    )

    with pytest.raises(KeyError, match="'does_not_exist' not found"):
        builder.build_schedule(rpe_problem[1])


def test_schedule_rejects_model_hamiltonian_description() -> None:
    """Scheduling requires the Pauli coefficients that a model description does not carry."""
    model = FermiHubbardModelHamiltonianDescription(LatticeGeometry.chain(2), t=1.0, u=4.0)

    with pytest.raises(TypeError, match="takes a QubitOperator, got FermiHubbardModelHamiltonianDescription"):
        RobustPhaseEstimationCircuitBuilder().build_schedule(model)


def test_schedule_requires_hermitian_coefficients() -> None:
    """Complex Hermitian coefficients schedule without casting, while non-Hermitian inputs are rejected."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        schedule = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=7).build_schedule(
            QubitOperator(pauli_strings=["Z"], coefficients=np.array([1.0 + 0.0j]))
        )
    assert schedule.lambda_norm == 1.0

    with pytest.raises(ValueError, match="Hermitian"):
        RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5).build_schedule(
            QubitOperator(pauli_strings=["Z"], coefficients=np.array([1.0j]))
        )


@pytest.mark.parametrize(
    ("builder_name", "category", "budget_mode", "correction", "randomized"),
    [
        ("trotter", "deterministic_or_exact", "independent_trotter", "linear", False),
        ("qdrift", "qdrift", "fraction", "qdrift_tangent", True),
        ("partially_randomized", "partial_randomized", "independent_partial_randomized", "linear", True),
        ("zassenhaus", "deterministic_or_exact", "fraction", "linear", False),
    ],
)
def test_registered_builder_name_preserves_rpe_policy(
    rpe_problem: tuple[Circuit, QubitOperator],
    builder_name: str,
    category: str,
    budget_mode: str,
    correction: str,
    randomized: bool,
) -> None:
    """Known builder names retain their accuracy, correction, and randomized-draw policies."""
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", builder_name),
    )

    schedule = builder.build_schedule(rpe_problem[1])

    assert schedule.unitary_builder_category == category
    assert schedule.error_budget_mode == budget_mode
    assert schedule.energy_correction == correction
    for round_data in schedule.rounds:
        assert round_data.num_draws == (round_data.shots_per_basis if randomized else 1)
        if not randomized:
            assert round_data.draw_seeds == (None,)


def test_default_partial_randomized_random_cost_scales_quadratically() -> None:
    """PR sample growth follows inverse-square RPE scaling when RPE keeps the configured unit floor."""
    hamiltonian = QubitOperator(pauli_strings=["X", "Z"], coefficients=[1.0, 0.5])
    random_rotation_counts: list[int] = []

    for target_accuracy in (0.1, 0.05, 0.025, 0.0125):
        circuit_set = RobustPhaseEstimationCircuitBuilder(
            target_accuracy=target_accuracy,
            seed=7,
            unitary_builder=AlgorithmRef(
                "hamiltonian_unitary_builder",
                "partially_randomized",
                weight_threshold=0.75,
                num_random_samples=1,
                trotter_order=2,
            ),
        ).build_schedule(hamiltonian)
        final_round = circuit_set.rounds[-1]
        settings = circuit_set.unitary_builder_configuration.settings
        assert settings is not None
        assert settings.get("num_random_samples") == 1
        partial_builder = PartiallyRandomized(**settings.to_dict())
        terms = hamiltonian.get_real_coefficients(tolerance=1e-12, sort_by_magnitude=True)
        random_terms = terms[1:]
        num_divisions = partial_builder._resolve_num_divisions(hamiltonian, final_round.evolution_time)
        block_samples = partial_builder._resolve_block_samples(random_terms, final_round.evolution_time, num_divisions)

        assert circuit_set.epsilon_rpe == pytest.approx(target_accuracy)
        assert circuit_set.epsilon_unitary == pytest.approx(0.85)
        assert settings.get("target_accuracy") == pytest.approx(0.85)
        assert sum(partial_builder._split_accuracy()) == pytest.approx(0.85)
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
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, unitary_builder=unitary_builder)

    with pytest.raises(ValueError, match="unitary_builder power must be 1"):
        builder.build_schedule(hamiltonian)


@pytest.mark.parametrize("builder_name", ["trotter", "qdrift", "partially_randomized"])
def test_explicit_nested_unitary_power_one_is_accepted(
    rpe_problem: tuple[Circuit, QubitOperator],
    builder_name: str,
) -> None:
    """An explicit unit power preserves each built-in RPE evolution path."""
    state_preparation, hamiltonian = rpe_problem
    unitary_builder = AlgorithmRef("hamiltonian_unitary_builder", builder_name, power=1)

    circuit_set = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        unitary_builder=unitary_builder,
    ).build_schedule(hamiltonian)

    assert circuit_set.unitary_builder_configuration.settings.get("power") == 1


def test_builder_rejects_rebound_nested_unitary_power(
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """A rebuilt workload is subject to the builder's nested-power policy."""
    state_preparation, hamiltonian = rpe_problem
    original = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5).build_schedule(hamiltonian)
    configuration = original.unitary_builder_configuration
    settings = configuration.settings
    settings.set("power", 2)
    rebuilt = replace(
        original, _unitary_builder=(configuration.algorithm_type, configuration.algorithm_name, settings.to_json())
    )

    with pytest.raises(ValueError, match="unitary_builder power must be 1"):
        next(RobustPhaseEstimationCircuitBuilder().iter_build(rebuilt, state_preparation, hamiltonian))


@pytest.mark.parametrize("field", ["coefficients", "pauli_strings"])
def test_builder_rejects_changed_hamiltonian(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
    field: str,
) -> None:
    """Explicit build inputs must match the Hamiltonian from which the schedule was resolved."""
    state_preparation, hamiltonian = rpe_problem
    schedule = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5).build_schedule(hamiltonian)
    if field == "coefficients":
        hamiltonian.coefficients[0] = 2.0
    else:
        hamiltonian.pauli_strings[0] = "X"

    with pytest.raises(ValueError, match="fingerprint"):
        next(RobustPhaseEstimationCircuitBuilder().iter_build(schedule, state_preparation, hamiltonian))

    unitary_records, hadamard_records = recording_builders
    assert not unitary_records
    assert not hadamard_records


def test_builder_rejects_inconsistent_scheduled_samples(rpe_problem: tuple[Circuit, QubitOperator]) -> None:
    """A corrupted qDRIFT count cannot silently alter tangent-correction bookkeeping."""
    state_preparation, hamiltonian = rpe_problem
    schedule = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "qdrift", target_accuracy=0.25),
    ).build_schedule(hamiltonian)
    corrupted = replace(schedule, rounds=(replace(schedule.rounds[0], scheduled_samples=2), *schedule.rounds[1:]))

    with pytest.raises(ValueError, match="sample count"):
        next(RobustPhaseEstimationCircuitBuilder().iter_build(corrupted, state_preparation, hamiltonian))


def test_deterministic_round_builds_one_multi_shot_pair(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A deterministic round builds one shared-unitary pair measured many times."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        seed=7,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
    )
    circuit_set = builder.build_schedule(hamiltonian)

    spec, _, _ = next(builder.iter_build(circuit_set, state_preparation, hamiltonian))

    expected_shots = ceil(e * (11 + 4 * (circuit_set.num_rounds - 1)))
    assert circuit_set.rounds[0].shots_per_basis == expected_shots
    assert circuit_set.rounds[0].scheduled_samples is None
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
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=11)
    circuit_set = builder.build_schedule(hamiltonian)

    with patch.object(
        RobustPhaseEstimationSchedule, "experiment_specs", property(lambda _: pytest.fail("Eager manifest"))
    ):
        experiments = list(islice(builder.iter_build(circuit_set, state_preparation, hamiltonian), 2))
    expected_specs = circuit_set.experiment_specs_for_round(0)[:2]

    assert [experiment[0] for experiment in experiments] == list(expected_specs)
    assert [record["seed"] for record in unitary_records] == [spec.draw_seed for spec in expected_specs]
    assert len(unitary_records) == 2
    assert len(hadamard_records) == 4
    assert hadamard_records[0][1] is hadamard_records[1][1]
    assert hadamard_records[2][1] is hadamard_records[3][1]


def test_basis_builders_are_created_once_per_stream(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """X/Y builders are reused across draws without changing same-draw circuit pairing."""
    state_preparation, hamiltonian = rpe_problem
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=7)
    schedule = builder.build_schedule(hamiltonian)
    original_create = algorithms.create
    created_bases: list[str] = []

    def record_creation(algorithm_type: str, algorithm_name: str, **settings_values: object) -> object:
        if algorithm_type == "hadamard_test_circuit_builder":
            created_bases.append(str(settings_values["test_basis"]))
        return original_create(algorithm_type, algorithm_name, **settings_values)

    monkeypatch.setattr(algorithms, "create", record_creation)
    experiments = list(islice(builder.iter_build(schedule, state_preparation, hamiltonian), 3))

    unitary_records, hadamard_records = recording_builders
    assert len(experiments) == len(unitary_records) == 3
    assert created_bases == ["X", "Y"]
    assert [basis for basis, _ in hadamard_records] == ["X", "Y"] * 3
    for index in range(0, len(hadamard_records), 2):
        assert hadamard_records[index][1] is hadamard_records[index + 1][1]


def test_run_returns_canonical_flat_circuit_list(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eager construction schedules once and places each draw's X/Y circuits at their manifest indices."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=17)
    original_schedule = builder.build_schedule
    scheduled_workloads: list[RobustPhaseEstimationSchedule] = []

    def schedule(qubit_hamiltonian: QubitOperator) -> RobustPhaseEstimationSchedule:
        scheduled_workloads.append(original_schedule(qubit_hamiltonian))
        return scheduled_workloads[-1]

    monkeypatch.setattr(builder, "build_schedule", schedule)

    circuits = builder.run(state_preparation, hamiltonian)

    assert len(scheduled_workloads) == 1
    circuit_set = scheduled_workloads[0]
    assert isinstance(circuits, list)
    assert len(circuits) == len(hadamard_records) == 2 * len(circuit_set.experiment_specs)
    for spec in circuit_set.experiment_specs:
        unitary = _FakeUnitary(spec.draw_seed, circuit_set.rounds[spec.round_index].evolution_time)
        assert circuits[spec.x_circuit_index].get_qasm() == _fake_qasm("X", unitary)
        assert circuits[spec.y_circuit_index].get_qasm() == _fake_qasm("Y", unitary)

    eager_records = list(unitary_records)
    unitary_records.clear()
    streamed = list(builder.iter_build(circuit_set, state_preparation, hamiltonian))

    assert unitary_records == eager_records
    assert [spec for spec, _, _ in streamed] == list(circuit_set.experiment_specs)
    assert [circuit.get_qasm() for circuit in circuits] == [
        circuit.get_qasm() for _, x_circuit, y_circuit in streamed for circuit in (x_circuit, y_circuit)
    ]


def test_stream_reiteration_replays_seeded_draws(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Re-iterating one workload rebuilds the same randomized draw sequence."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, _ = recording_builders
    builder = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=13)
    circuit_set = builder.build_schedule(hamiltonian)

    first = next(builder.iter_build(circuit_set, state_preparation, hamiltonian))[0]
    second = next(builder.iter_build(circuit_set, state_preparation, hamiltonian))[0]

    assert first.draw_seed == second.draw_seed
    assert [record["seed"] for record in unitary_records] == [first.draw_seed, first.draw_seed]


def test_workload_rebinds_and_replays_seeded_draw(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Rebinding a different live preparation preserves the schedule and paired unitary draw."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    original = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=23).build_schedule(hamiltonian)
    replay = RobustPhaseEstimationCircuitSet(
        schedule=original, state_preparation=state_preparation, qubit_hamiltonian=hamiltonian
    )
    other_preparation = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nx q[0];\n")
    rebound = replay.rebind(other_preparation)
    assert rebound.state_preparation is other_preparation
    assert rebound.schedule is replay.schedule

    spec, _, _ = next(
        islice(
            RobustPhaseEstimationCircuitBuilder().iter_build(
                rebound.schedule, rebound.state_preparation, rebound.qubit_hamiltonian
            ),
            2,
            3,
        )
    )

    assert spec.draw_seed == original.experiment_specs[2].draw_seed
    assert unitary_records[-1]["seed"] == spec.draw_seed
    assert hadamard_records[-2][1] is hadamard_records[-1][1]


def test_circuit_set_remains_lazy_and_builds_on_demand(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Binding live inputs to a schedule does not construct any circuits."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    original = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=11).build_schedule(hamiltonian)
    replay = RobustPhaseEstimationCircuitSet(
        schedule=original, state_preparation=state_preparation, qubit_hamiltonian=hamiltonian
    )
    assert unitary_records == []
    assert hadamard_records == []

    spec, _, _ = next(
        RobustPhaseEstimationCircuitBuilder().iter_build(
            replay.schedule, replay.state_preparation, replay.qubit_hamiltonian
        )
    )

    assert spec.draw_seed == replay.schedule.experiment_specs[0].draw_seed
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

    circuit_set = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=-1).build_schedule(hamiltonian)

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
    circuit_set = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
    ).build_schedule(hamiltonian)
    first_config = circuit_set.unitary_builder_configuration
    assert first_config.settings is not None
    first_config.settings.set("power", 99)

    second_config = circuit_set.unitary_builder_configuration

    assert second_config.settings is not None
    assert second_config.settings.get("power") == 1
    assert not second_config.settings.has("time")
    with pytest.raises(IndexError, match="round_index"):
        circuit_set.experiment_specs_for_round(circuit_set.num_rounds)


@pytest.fixture
def workload(rpe_problem: tuple[Circuit, QubitOperator]) -> RobustPhaseEstimationCircuitSet:
    """Bind a two-round randomized schedule to live inputs."""
    preparation, operator = rpe_problem
    schedule = RobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=17).build_schedule(operator)
    return RobustPhaseEstimationCircuitSet(schedule=schedule, state_preparation=preparation, qubit_hamiltonian=operator)


def test_workload_records_are_frozen_in_memory_dataclasses(workload: RobustPhaseEstimationCircuitSet) -> None:
    """Scheduling and replay need immutable records, not a file-format API."""
    schedule = workload.schedule
    spec = schedule.experiment_specs_for_round(1)[0]
    for record in (workload, schedule, schedule.rounds[0], spec):
        assert is_dataclass(record)
        assert not isinstance(record, DataClass)
        assert not any(hasattr(record, name) for name in ("to_file", "from_file", "to_json", "to_hdf5", "content_hash"))
        with pytest.raises(FrozenInstanceError):
            setattr(record, fields(record)[0].name, None)
    assert not hasattr(schedule, "state_preparation")
    assert not hasattr(schedule, "qubit_hamiltonian")
    assert spec.x_circuit_index == 2 * schedule.rounds[0].num_draws
    assert spec.y_circuit_index == spec.x_circuit_index + 1


def test_workload_isolated_from_mutable_hamiltonians(
    workload: RobustPhaseEstimationCircuitSet, rpe_problem: tuple[Circuit, QubitOperator]
) -> None:
    """Neither the original inputs nor a returned Hamiltonian can mutate future replays."""
    _, original = rpe_problem
    equivalent = QubitOperator(["Z"], np.array([1.0], dtype=np.complex128))
    assert workload.schedule.hamiltonian_fingerprint(equivalent) == workload.schedule.hamiltonian_hash
    original.coefficients[0] = 99.0
    original.pauli_strings[0] = "X"
    retrieved = workload.qubit_hamiltonian
    retrieved.coefficients[0] = 50.0
    retrieved.pauli_strings[0] = "Y"
    checked = workload.schedule.validate_and_copy_hamiltonian(equivalent)
    checked.coefficients[0] = 12.0
    assert equivalent.coefficients[0] == 1.0
    assert workload.qubit_hamiltonian.coefficients[0] == 1.0
    assert workload.qubit_hamiltonian.pauli_strings == ["Z"]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.complex64])
def test_schedule_and_replay_use_the_same_coefficient_precision(dtype: type) -> None:
    """Widening the actual input values must not change the time ladder, sample counts, or input checks."""
    coefficients = np.array([0.6, 0.4, 0.1], dtype=dtype)
    labels = ["ZI", "IZ", "XX"]
    operator = QubitOperator(labels, coefficients)
    canonical = QubitOperator(labels, coefficients.astype(np.complex128))
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        seed=17,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "qdrift", target_accuracy=0.25),
    )

    schedule = builder.build_schedule(operator)

    assert schedule.lambda_norm == canonical.schatten_norm
    assert schedule == builder.build_schedule(canonical)
    copy = schedule.validate_and_copy_hamiltonian(operator)
    assert schedule.validate_and_copy_hamiltonian(copy).schatten_norm == schedule.lambda_norm


@pytest.mark.parametrize("time", [0.0, -1.0, float("nan"), float("inf")])
def test_workload_round_rejects_invalid_time(workload: RobustPhaseEstimationCircuitSet, time: float) -> None:
    """Evolution times must be finite and positive."""
    with pytest.raises(ValueError, match="evolution_time"):
        replace(workload.schedule.rounds[0], evolution_time=time)


@pytest.mark.parametrize("name", ["round_index", "shots_per_basis", "scheduled_samples"])
@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_workload_round_rejects_invalid_counts(
    workload: RobustPhaseEstimationCircuitSet, name: str, value: object
) -> None:
    """Counts must be integers in range, not booleans or rounded floats."""
    with pytest.raises(ValueError, match=name):
        replace(workload.schedule.rounds[0], **{name: value})


@pytest.mark.parametrize("name", ["time", "seed", "num_samples"])
def test_workload_rejects_duplicated_round_settings(workload: RobustPhaseEstimationCircuitSet, name: str) -> None:
    """The shared builder cannot override a round's time, seed, or samples."""
    schedule = workload.schedule
    config = ("hamiltonian_unitary_builder", "qdrift", json.dumps({name: 2}))
    with pytest.raises(ValueError, match=name):
        replace(schedule, _unitary_builder=config)


@pytest.mark.parametrize(
    ("family", "invalid_samples"), [("qdrift", None), ("partially_randomized", 2), ("trotter", 2), ("zassenhaus", 2)]
)
def test_schedule_validates_qdrift_only_sample_counts(family: str, invalid_samples: int | None) -> None:
    """A schedule can report a per-draw count only when the builder will actually execute it."""
    schedule = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=1.0, seed=7, unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", family)
    ).build_schedule(QubitOperator(["Z"], np.array([1.0])))
    with pytest.raises(ValueError, match="scheduled_samples must be set for pure qDRIFT"):
        replace(schedule, rounds=(replace(schedule.rounds[0], scheduled_samples=invalid_samples), *schedule.rounds[1:]))


@pytest.mark.parametrize("family", ["trotter", "zassenhaus"])
def test_deterministic_schedule_rejects_seeded_draws(family: str) -> None:
    """The experiment spec and the circuit builder must agree that a deterministic draw has no seed."""
    schedule = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=1.0, unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", family)
    ).build_schedule(QubitOperator(["Z"], np.array([1.0])))
    assert schedule.rounds[0].draw_seeds == (None,)
    with pytest.raises(ValueError, match=r"draw_seeds=\(None,\)"):
        replace(schedule, rounds=(replace(schedule.rounds[0], draw_seeds=(123,)), *schedule.rounds[1:]))


def test_workload_validates_ladder_draws_and_correction(workload: RobustPhaseEstimationCircuitSet) -> None:
    """Invalid combinations fail when an in-memory schedule is constructed."""
    schedule = workload.schedule
    first, second = schedule.rounds
    with pytest.raises(ValueError, match="evolution_time"):
        replace(schedule, rounds=(first, replace(second, evolution_time=0.2)))
    with pytest.raises(ValueError, match="shots_per_basis"):
        replace(schedule, rounds=(replace(first, shots_per_basis=1), second))
    with pytest.raises(ValueError, match="qDRIFT tangent correction requires a qDRIFT schedule"):
        replace(schedule, unitary_builder_category="partial_randomized")
    invalid_norm = replace(schedule, lambda_norm=2.0)
    with pytest.raises(ValueError, match="lambda_norm"):
        invalid_norm.validate_and_copy_hamiltonian(workload.qubit_hamiltonian)
    with pytest.raises(ValueError, match="lambda_norm"):
        RobustPhaseEstimationCircuitSet(
            schedule=invalid_norm,
            state_preparation=workload.state_preparation,
            qubit_hamiltonian=workload.qubit_hamiltonian,
        )
