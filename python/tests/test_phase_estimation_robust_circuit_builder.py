"""Tests for on-demand robust phase estimation circuit generation."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import islice

import pytest

from qdk_chemistry.algorithms import create
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder import (
    QdkRobustPhaseEstimationCircuitBuilder,
    RobustPhaseEstimationCircuitBuilder,
    RobustPhaseEstimationCircuitSet,
    _AlgorithmSnapshot,
)
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator, RobustPhaseEstimationSchedule, Settings
from qdk_chemistry.utils.rpe import qdrift_schedule


@dataclass(frozen=True)
class _FakeUnitary:
    """Unitary marker used to verify X/Y circuit pairing."""

    seed: int | None


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
            return _FakeUnitaryBuilder(settings, unitary_records)
        if snapshot.algorithm_type == "hadamard_test_circuit_builder":
            return _FakeHadamardBuilder(settings, hadamard_records)
        raise AssertionError(f"Unexpected algorithm type: {snapshot.algorithm_type}")

    monkeypatch.setattr(_AlgorithmSnapshot, "create", create_snapshot)
    return unitary_records, hadamard_records


def test_builder_is_registered_and_does_not_build_eagerly(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """The registered builder resolves a schedule without constructing circuits."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    builder = create("robust_phase_estimation_circuit_builder", "qdk", target_accuracy=0.5, seed=7)

    circuit_set = builder.run(state_preparation, hamiltonian)

    assert isinstance(builder, RobustPhaseEstimationCircuitBuilder)
    assert isinstance(circuit_set, RobustPhaseEstimationCircuitSet)
    assert circuit_set.num_rounds > 0
    hadamard_ref = builder.settings().get("hadamard_test_circuit_builder")
    assert hadamard_ref.settings is not None
    assert unitary_records == []
    assert hadamard_records == []

    schedule = circuit_set.schedule
    assert isinstance(schedule, RobustPhaseEstimationSchedule)
    assert schedule.num_rounds == circuit_set.num_rounds
    assert unitary_records == []
    assert hadamard_records == []


def test_deterministic_round_yields_one_pair_with_shot_multiplicity(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A deterministic round builds one shared-unitary pair measured many times."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    builder = QdkRobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        seed=7,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
    )
    circuit_set = builder.run(state_preparation, hamiltonian)
    round_zero = circuit_set.rounds[0]

    experiments = list(circuit_set.iter_round(0))

    expected_shots, expected_samples = qdrift_schedule(circuit_set.num_rounds - 1, 0)
    assert round_zero.shots_per_basis == expected_shots
    assert round_zero.scheduled_samples == expected_samples
    assert round_zero.num_draws == 1
    assert round_zero.circuit_multiplicity == expected_shots
    assert round_zero.draw_seeds == ()
    assert len(experiments) == 1
    assert experiments[0].draw_index is None
    assert experiments[0].draw_seed is None
    assert experiments[0].circuit_multiplicity == expected_shots
    assert len(unitary_records) == 1
    assert unitary_records[0]["target_accuracy"] == pytest.approx(0.85)
    assert [basis for basis, _ in hadamard_records] == ["X", "Y"]
    assert hadamard_records[0][1] is hadamard_records[1][1]


def test_randomized_round_generates_independent_pairs_on_demand(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A randomized round builds independent seeded pairs only as requested."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    builder = QdkRobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=11)
    circuit_set = builder.run(state_preparation, hamiltonian)
    round_zero = circuit_set.rounds[0]

    experiments = list(islice(circuit_set.iter_round(0), 2))

    assert round_zero.num_draws == round_zero.shots_per_basis
    assert round_zero.circuit_multiplicity == 1
    assert len(round_zero.draw_seeds) == round_zero.num_draws
    assert len(set(round_zero.draw_seeds)) == round_zero.num_draws
    assert [experiment.draw_seed for experiment in experiments] == list(round_zero.draw_seeds[:2])
    assert [record["seed"] for record in unitary_records] == list(round_zero.draw_seeds[:2])
    assert len(unitary_records) == 2
    assert len(hadamard_records) == 4
    assert hadamard_records[0][1] is hadamard_records[1][1]
    assert hadamard_records[2][1] is hadamard_records[3][1]


def test_direct_randomized_experiment_and_circuit_access_support_qre(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Users can materialize one seeded pair or one QRE-compatible circuit directly."""
    from qdk.qre.application import OpenQASMApplication  # noqa: PLC0415

    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    circuit_set = QdkRobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=17).run(
        state_preparation, hamiltonian
    )

    experiment = circuit_set.get_experiment(0, 1)

    assert experiment.draw_index == 1
    assert experiment.draw_seed == circuit_set.rounds[0].draw_seeds[1]
    assert unitary_records[0]["seed"] == experiment.draw_seed
    assert hadamard_records[0][1] is hadamard_records[1][1]
    assert isinstance(experiment.x_circuit.get_qre_application(), OpenQASMApplication)

    y_circuit = circuit_set.get_circuit(0, "y", 1)
    assert isinstance(y_circuit.get_qre_application(), OpenQASMApplication)
    assert unitary_records[1]["seed"] == experiment.draw_seed


def test_direct_experiment_access_validates_round_draw_and_basis(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Direct access rejects invalid coordinates before constructing circuits."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    randomized_set = QdkRobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=19).run(
        state_preparation, hamiltonian
    )
    deterministic_set = QdkRobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
    ).run(state_preparation, hamiltonian)

    with pytest.raises(ValueError, match="required for randomized"):
        randomized_set.get_experiment(0)
    with pytest.raises(IndexError, match="draw_index"):
        randomized_set.get_experiment(0, randomized_set.rounds[0].num_draws)
    with pytest.raises(ValueError, match="must be None"):
        deterministic_set.get_experiment(0, 0)
    with pytest.raises(ValueError, match="basis"):
        randomized_set.get_circuit(0, "Z", 0)

    assert unitary_records == []
    assert hadamard_records == []


def test_circuit_set_reiteration_replays_seeded_draws(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """Re-iterating one set rebuilds the same randomized draw sequence."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, _ = recording_builders
    circuit_set = QdkRobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=13).run(
        state_preparation, hamiltonian
    )

    first = next(circuit_set.iter_round(0))
    second = next(circuit_set.iter_round(0))

    assert first.draw_seed == second.draw_seed
    assert [record["seed"] for record in unitary_records] == [first.draw_seed, first.draw_seed]


def test_serialized_schedule_rebinds_and_replays_seeded_draw(
    rpe_problem: tuple[Circuit, QubitOperator],
    recording_builders: tuple[list[dict[str, object]], list[tuple[str, _FakeUnitary]]],
) -> None:
    """A round-tripped schedule regenerates the same draw after binding live inputs."""
    state_preparation, hamiltonian = rpe_problem
    unitary_records, hadamard_records = recording_builders
    original = QdkRobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=23).run(state_preparation, hamiltonian)
    restored_schedule = RobustPhaseEstimationSchedule.from_json(json.loads(json.dumps(original.schedule.to_json())))

    rebound = RobustPhaseEstimationCircuitSet.from_schedule(restored_schedule, state_preparation, hamiltonian)
    experiment = rebound.get_experiment(0, 2)

    assert rebound.schedule.content_hash() == original.schedule.content_hash()
    assert experiment.draw_seed == original.rounds[0].draw_seeds[2]
    assert unitary_records[0]["seed"] == experiment.draw_seed
    assert hadamard_records[0][1] is hadamard_records[1][1]


def test_entropy_seed_is_concretized_once_per_circuit_set(
    monkeypatch: pytest.MonkeyPatch,
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """The nondeterministic sentinel becomes one replayable root seed."""
    state_preparation, hamiltonian = rpe_problem
    monkeypatch.setattr(QdkRobustPhaseEstimationCircuitBuilder, "_resolve_root_seed", staticmethod(lambda _seed: 1234))

    circuit_set = QdkRobustPhaseEstimationCircuitBuilder(target_accuracy=0.5, seed=-1).run(
        state_preparation, hamiltonian
    )

    assert circuit_set.requested_seed == -1
    assert circuit_set.root_seed == 1234
    assert circuit_set.rounds[0].draw_seeds[0] == QdkRobustPhaseEstimationCircuitBuilder._derive_seed(1234, 0, 0)


def test_round_configuration_is_defensive_and_round_index_is_validated(
    rpe_problem: tuple[Circuit, QubitOperator],
) -> None:
    """Configuration access returns copies and invalid round indices fail clearly."""
    state_preparation, hamiltonian = rpe_problem
    circuit_set = QdkRobustPhaseEstimationCircuitBuilder(
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
        next(circuit_set.iter_round(circuit_set.num_rounds))
