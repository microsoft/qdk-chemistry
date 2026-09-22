"""Tests for serializable robust phase estimation workload data."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    DataClass,
    QubitOperator,
    RobustPhaseEstimationCircuitSet,
    RobustPhaseEstimationExperimentSpec,
    RobustPhaseEstimationRound,
    RobustPhaseEstimationSchedule,
    Settings,
)
from qdk_chemistry.data.robust_phase_estimation import _AlgorithmConfiguration, _HamiltonianSnapshot


def _algorithm_ref(algorithm_type: str, algorithm_name: str, **values: object) -> AlgorithmRef:
    """Create a resolved algorithm reference for serialization tests."""
    settings = Settings()
    for key, value in values.items():
        if isinstance(value, bool):
            setting_type = "bool"
        elif isinstance(value, int):
            setting_type = "int"
        elif isinstance(value, float):
            setting_type = "double"
        else:
            setting_type = "string"
        settings._set_default(key, setting_type, value)
    return AlgorithmRef(algorithm_type, algorithm_name, settings=settings)


def _round() -> RobustPhaseEstimationRound:
    """Create representative randomized round metadata."""
    return RobustPhaseEstimationRound(
        round_index=0,
        evolution_time=1.25,
        shots_per_basis=7,
        scheduled_samples=32,
        draw_seeds=tuple(range(101, 108)),
    )


def _experiment_specs() -> tuple[RobustPhaseEstimationExperimentSpec, ...]:
    """Create representative randomized experiment metadata."""
    return tuple(
        RobustPhaseEstimationExperimentSpec(
            experiment_index=draw_index,
            round_index=0,
            draw_index=draw_index,
            draw_seed=101 + draw_index,
            shots=1,
        )
        for draw_index in range(7)
    )


def _schedule(
    *,
    rounds: tuple[RobustPhaseEstimationRound, ...] | None = None,
    category: str = "partial_randomized",
) -> RobustPhaseEstimationSchedule:
    """Create a schedule without live circuit or Hamiltonian inputs."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=np.asarray([2.5]))
    return RobustPhaseEstimationSchedule(
        rounds=(_round(),) if rounds is None else rounds,
        hamiltonian_hash=_HamiltonianSnapshot.from_operator(hamiltonian).content_hash(),
        lambda_norm=2.5,
        target_accuracy=0.01,
        epsilon_rpe=0.005,
        epsilon_unitary=0.005,
        unitary_accuracy_fraction=0.5,
        error_budget_mode="fraction",
        unitary_builder_category=category,
        energy_correction="linear",
        requested_seed=11,
        root_seed=11,
        unitary_builder_configuration=_algorithm_ref("hamiltonian_unitary_builder", "partially_randomized", power=1),
        hadamard_test_circuit_builder_configuration=_algorithm_ref(
            "hadamard_test_circuit_builder", "qdk", test_basis="X"
        ),
    )


def _circuit_set() -> RobustPhaseEstimationCircuitSet:
    """Package optional replay inputs alongside a canonical schedule."""
    return RobustPhaseEstimationCircuitSet(
        schedule=_schedule(),
        state_preparation=Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n"),
        qubit_hamiltonian=QubitOperator(pauli_strings=["Z"], coefficients=np.asarray([2.5])),
    )


def test_experiment_spec_defines_canonical_pair_indices() -> None:
    """An experiment identity determines its canonical X/Y circuit positions."""
    spec = _experiment_specs()[3]

    assert spec.x_circuit_index == 6
    assert spec.y_circuit_index == 7


def test_schedule_and_replay_bundle_are_independent_data_classes() -> None:
    """A schedule can exist and serialize without its optional live replay inputs."""
    circuit_set = _circuit_set()

    assert isinstance(circuit_set, DataClass)
    assert isinstance(circuit_set.schedule, DataClass)
    assert not isinstance(circuit_set.schedule.rounds[0], DataClass)
    assert not isinstance(circuit_set.schedule.experiment_specs[0], DataClass)
    assert circuit_set.data_type_name() == "robust_phase_estimation_circuit_set"
    with pytest.raises(AttributeError, match="Cannot modify immutable"):
        circuit_set.state_preparation = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n")
    with pytest.raises(FrozenInstanceError):
        circuit_set.schedule.rounds[0].shots_per_basis = 9


@pytest.mark.parametrize("suffix", ["json", "hdf5"])
def test_circuit_set_file_roundtrip(tmp_path: Path, suffix: str) -> None:
    """The workload, manifest, inputs, and nested settings survive file round trips."""
    circuit_set = _circuit_set()
    filename = tmp_path / f"sample.robust_phase_estimation_circuit_set.{suffix}"

    circuit_set.to_file(filename, suffix)
    restored = RobustPhaseEstimationCircuitSet.from_file(filename, suffix)

    assert restored.content_hash() == circuit_set.content_hash()
    assert restored.schedule.rounds == circuit_set.schedule.rounds
    assert restored.schedule.experiment_specs == circuit_set.schedule.experiment_specs
    assert restored.state_preparation.get_qasm() == circuit_set.state_preparation.get_qasm()
    assert restored.qubit_hamiltonian.content_hash() == circuit_set.qubit_hamiltonian.content_hash()
    assert restored.schedule.hadamard_test_circuit_builder_configuration.settings.get("test_basis") == "X"


@pytest.mark.parametrize("suffix", ["json", "hdf5"])
def test_schedule_file_roundtrip_without_inputs(tmp_path: Path, suffix: str) -> None:
    """A standalone schedule serializes without materializing or storing input circuits."""
    schedule = _schedule()
    filename = tmp_path / f"sample.robust_phase_estimation_schedule.{suffix}"

    schedule.to_file(filename, suffix)
    restored = RobustPhaseEstimationSchedule.from_file(filename, suffix)

    assert restored.content_hash() == schedule.content_hash()
    assert restored.rounds == schedule.rounds
    assert restored.experiment_specs == _experiment_specs()
    assert not hasattr(restored, "qubit_hamiltonian")
    assert not hasattr(restored, "state_preparation")


def test_schedule_payload_stores_each_experiment_parameter_once() -> None:
    """Times, counts, and identities have one canonical source rather than competing copies."""
    schedule = _schedule()
    payload = schedule.to_json()

    assert "base_time" not in payload
    assert "experiment_specs" not in payload
    assert "qubit_hamiltonian" not in payload
    assert "state_preparation" not in payload
    for round_data in payload["rounds"]:
        assert "unitary_builder_configuration" not in round_data
        assert "num_draws" not in round_data
    settings = schedule.unitary_builder_configuration.settings
    assert all(not settings.has(key) for key in ("time", "seed", "num_samples"))
    assert schedule.base_time == schedule.rounds[0].evolution_time
    assert schedule.rounds[0].num_draws == len(schedule.rounds[0].draw_seeds)


@pytest.mark.parametrize("version", ["0.1.0", "999.0.0"])
def test_circuit_set_serialization_guards_version(version: str) -> None:
    """Circuit-set deserialization rejects incompatible wire versions."""
    payload = _circuit_set().to_json()
    payload["version"] = version

    with pytest.raises(RuntimeError, match="version"):
        RobustPhaseEstimationCircuitSet.from_json(payload)


def test_schedule_configuration_is_defensive() -> None:
    """A caller cannot mutate the shared algorithm settings in a schedule."""
    schedule = _schedule()
    first = schedule.unitary_builder_configuration
    first.settings.set("power", 99)

    second = schedule.unitary_builder_configuration

    assert second.settings.get("power") == 1
    assert not second.settings.has("time")


def test_data_configuration_has_no_algorithm_operations() -> None:
    """Data snapshots serialize references without constructing or configuring algorithms."""
    configuration = _AlgorithmConfiguration.from_ref(_algorithm_ref("test", "test", value=1))

    for operation in ("create", "with_updates", "validate_unit_power", "has_setting"):
        assert not hasattr(configuration, operation)
    assert configuration.to_ref().settings.get("value") == 1


@pytest.mark.parametrize("field", ["coefficients", "pauli_strings"])
def test_circuit_set_hamiltonian_snapshot_is_defensive(field: str) -> None:
    """Mutating a retrieved Hamiltonian must not alter a recorded workload or its hash."""
    circuit_set = _circuit_set()
    original_hash = circuit_set.content_hash()
    hamiltonian = circuit_set.qubit_hamiltonian

    if field == "coefficients":
        hamiltonian.coefficients[0] = 99.0
    else:
        hamiltonian.pauli_strings[0] = "X"

    assert circuit_set.qubit_hamiltonian.coefficients[0] == pytest.approx(2.5)
    assert circuit_set.qubit_hamiltonian.pauli_strings == ["Z"]
    assert circuit_set.content_hash() == original_hash


def test_replay_snapshot_isolated_from_original_input() -> None:
    """Mutation of the caller's Hamiltonian cannot affect replay inputs or content hashes."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[2.5])
    replay = RobustPhaseEstimationCircuitSet(
        schedule=_schedule(),
        state_preparation=Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n"),
        qubit_hamiltonian=hamiltonian,
    )
    original_hash = replay.content_hash()

    hamiltonian.coefficients[0] = 99.0
    hamiltonian.pauli_strings[0] = "X"
    serialized = replay.to_json()
    serialized["qubit_hamiltonian"]["pauli_strings"][0] = "Y"

    assert replay.qubit_hamiltonian.coefficients[0] == pytest.approx(2.5)
    assert replay.qubit_hamiltonian.pauli_strings == ["Z"]
    assert replay.content_hash() == original_hash


@pytest.mark.parametrize("field", ["base_time", "evolution_time", "builder_time"])
@pytest.mark.parametrize("suffix", ["json", "hdf5"])
def test_circuit_set_rejects_conflicting_serialized_times(tmp_path: Path, field: str, suffix: str) -> None:
    """Loading must reject a schedule whose construction and reconstruction times disagree."""
    payload = _circuit_set().to_json()
    schedule_data = payload["schedule"]
    if field == "base_time":
        schedule_data["base_time"] = 0.2
    elif field == "evolution_time":
        schedule_data["rounds"].append({**schedule_data["rounds"][0], "round_index": 1, "evolution_time": 0.2})
    else:
        configuration = schedule_data["unitary_builder_configuration"]
        settings = json.loads(configuration["settings_json"])
        settings["time"] = 0.2
        configuration["settings_json"] = json.dumps(settings)

    if suffix == "json":
        with pytest.raises(ValueError, match="time"):
            RobustPhaseEstimationCircuitSet.from_json(payload)
    else:
        filename = tmp_path / "invalid.robust_phase_estimation_circuit_set.hdf5"
        _circuit_set().to_file(filename, "hdf5")
        schedule_data.pop("version")
        with h5py.File(filename, "r+") as handle:
            handle["schedule/payload"][()] = json.dumps(schedule_data)
        with pytest.raises(ValueError, match="time"):
            RobustPhaseEstimationCircuitSet.from_file(filename, "hdf5")


@pytest.mark.parametrize("time", [0.0, -1.0, float("nan"), float("inf")])
def test_round_rejects_invalid_time(time: float) -> None:
    """Every constructor path rejects nonfinite or nonpositive evolution times."""
    with pytest.raises(ValueError, match="evolution_time"):
        replace(_round(), evolution_time=time)


@pytest.mark.parametrize("field", ["round_index", "shots_per_basis", "scheduled_samples"])
@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_round_rejects_invalid_counts(field: str, value: object) -> None:
    """Invalid integer fields must not be silently truncated during deserialization."""
    payload = _schedule().to_json()
    payload["rounds"][0][field] = value

    with pytest.raises(ValueError, match=field):
        RobustPhaseEstimationSchedule.from_json(payload)


@pytest.mark.parametrize("field", ["seed", "num_samples"])
def test_schedule_rejects_duplicated_round_settings(field: str) -> None:
    """Builder snapshots cannot supply another seed or sample count for the same round."""
    payload = _schedule().to_json()
    configuration = payload["unitary_builder_configuration"]
    settings = json.loads(configuration["settings_json"])
    settings[field] = 42
    configuration["settings_json"] = json.dumps(settings)

    with pytest.raises(ValueError, match=field):
        RobustPhaseEstimationSchedule.from_json(payload)


def test_bundle_rejects_inconsistent_reconstruction_norm() -> None:
    """A stored reconstruction norm must describe the bundle's actual Hamiltonian."""
    payload = _circuit_set().to_json()
    payload["schedule"]["lambda_norm"] = 3.0

    with pytest.raises(ValueError, match="lambda_norm"):
        RobustPhaseEstimationCircuitSet.from_json(payload)


def test_manifest_rejects_inconsistent_shot_total() -> None:
    """Randomized shot totals must match the number of canonical unitary seeds."""
    round_data = replace(_round(), shots_per_basis=8)

    with pytest.raises(ValueError, match="shots_per_basis"):
        _schedule(rounds=(round_data,))


def test_deterministic_manifest_uses_one_multi_shot_pair() -> None:
    """A deterministic round maps its complete basis workload to one circuit pair."""
    round_data = RobustPhaseEstimationRound(
        round_index=0,
        evolution_time=1.25,
        shots_per_basis=7,
        scheduled_samples=32,
        draw_seeds=(None,),
    )
    spec = RobustPhaseEstimationExperimentSpec(
        experiment_index=0,
        round_index=0,
        draw_index=None,
        draw_seed=None,
        shots=7,
    )

    schedule = _schedule(rounds=(round_data,), category="deterministic_or_exact")

    assert schedule.experiment_specs_for_round(0) == (spec,)


def test_rebind_replaces_live_inputs_without_rescheduling() -> None:
    """Rebinding preserves the manifest and concrete randomized seeds."""
    circuit_set = _circuit_set()
    state_preparation = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nx q[0];\n")

    rebound = circuit_set.rebind(state_preparation)

    assert rebound.schedule is circuit_set.schedule
    assert rebound.schedule.experiment_specs == circuit_set.schedule.experiment_specs
    assert rebound.state_preparation.get_qasm() == state_preparation.get_qasm()
