"""Tests for serializable robust phase estimation workload data."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
import pytest

from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    DataClass,
    QubitOperator,
    RobustPhaseEstimationCircuitSet,
    RobustPhaseEstimationExperiment,
    RobustPhaseEstimationRound,
    RobustPhaseEstimationSchedule,
    Settings,
)


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
        round_index=2,
        evolution_time=1.25,
        shots_per_basis=7,
        num_draws=7,
        scheduled_samples=32,
        circuit_multiplicity=1,
        draw_seeds=(101, 102, 103, 104, 105, 106, 107),
        unitary_builder_configuration=_algorithm_ref(
            "hamiltonian_unitary_builder", "partially_randomized", time=1.25, seed=101
        ),
    )


def _schedule() -> RobustPhaseEstimationSchedule:
    """Create a representative schedule."""
    return RobustPhaseEstimationSchedule(
        rounds=(_round(),),
        lambda_norm=2.5,
        base_time=0.2,
        target_accuracy=0.01,
        epsilon_rpe=0.005,
        epsilon_unitary=0.005,
        unitary_accuracy_fraction=0.5,
        error_budget_mode="fraction",
        unitary_builder_category="partial_randomized",
        energy_correction="linear",
        requested_seed=11,
        root_seed=11,
        hadamard_test_circuit_builder_configuration=_algorithm_ref(
            "hadamard_test_circuit_builder", "qdk", test_basis="X"
        ),
    )


def _circuit_set() -> RobustPhaseEstimationCircuitSet:
    """Create a representative serializable lazy circuit set."""
    return RobustPhaseEstimationCircuitSet.from_schedule(
        _schedule(),
        Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n"),
        QubitOperator(pauli_strings=["Z"], coefficients=np.asarray([2.5])),
    )


def test_circuit_set_is_immutable_data_class() -> None:
    """The circuit-builder output is immutable data with a guarded wire type."""
    circuit_set = _circuit_set()

    assert isinstance(circuit_set, DataClass)
    assert circuit_set.data_type_name() == "robust_phase_estimation_circuit_set"
    with pytest.raises(AttributeError, match="Cannot modify immutable"):
        circuit_set.state_preparation = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n")


@pytest.mark.parametrize("suffix", ["json", "hdf5"])
def test_circuit_set_file_roundtrip(tmp_path: Path, suffix: str) -> None:
    """The lazy algorithm output preserves its schedule and source inputs across files."""
    circuit_set = _circuit_set()
    filename = tmp_path / f"sample.robust_phase_estimation_circuit_set.{suffix}"

    circuit_set.to_file(filename, suffix)
    restored = RobustPhaseEstimationCircuitSet.from_file(filename, suffix)

    assert restored.content_hash() == circuit_set.content_hash()
    assert restored.schedule.content_hash() == circuit_set.schedule.content_hash()
    assert restored.state_preparation.get_qasm() == circuit_set.state_preparation.get_qasm()
    assert restored.qubit_hamiltonian.content_hash() == circuit_set.qubit_hamiltonian.content_hash()


def test_circuit_set_serialization_guards_type_and_version(tmp_path: Path) -> None:
    """Circuit-set files reject the wrong data type and incompatible versions."""
    circuit_set = _circuit_set()
    filename = tmp_path / "sample.robust_phase_estimation_circuit_set.json"
    circuit_set.to_json_file(filename)

    with pytest.raises(ValueError, match="robust_phase_estimation_schedule"):
        RobustPhaseEstimationSchedule.from_json_file(filename)

    payload = circuit_set.to_json()
    payload["version"] = "999.0.0"
    with pytest.raises(RuntimeError, match="version"):
        RobustPhaseEstimationCircuitSet.from_json(payload)


def test_schedule_is_immutable_data_class_without_circuits() -> None:
    """The schedule is lightweight immutable metadata rather than a circuit collection."""
    schedule = _schedule()

    assert isinstance(schedule, DataClass)
    assert "x_circuit" not in str(schedule.to_json())
    assert "y_circuit" not in str(schedule.to_json())
    with pytest.raises(AttributeError, match="Cannot modify immutable"):
        schedule.target_accuracy = 0.1


@pytest.mark.parametrize("suffix", ["json", "hdf5"])
def test_schedule_file_roundtrip(tmp_path: Path, suffix: str) -> None:
    """Schedule metadata and nested builder settings survive JSON and HDF5 files."""
    schedule = _schedule()
    filename = tmp_path / f"sample.robust_phase_estimation_schedule.{suffix}"

    schedule.to_file(filename, suffix)
    restored = RobustPhaseEstimationSchedule.from_file(filename, suffix)

    assert restored.content_hash() == schedule.content_hash()
    assert restored.rounds[0].draw_seeds == schedule.rounds[0].draw_seeds
    assert restored.rounds[0].unitary_builder_configuration.settings is not None
    assert restored.rounds[0].unitary_builder_configuration.settings.get("seed") == 101
    assert restored.hadamard_test_circuit_builder_configuration.settings is not None
    assert restored.hadamard_test_circuit_builder_configuration.settings.get("test_basis") == "X"


@pytest.mark.parametrize("suffix", ["json", "hdf5"])
def test_materialized_experiment_file_roundtrip_supports_qre(tmp_path: Path, suffix: str) -> None:
    """A selected circuit pair retains metadata and remains QRE-compatible after loading."""
    from qdk.qre.application import OpenQASMApplication  # noqa: PLC0415

    qasm = "OPENQASM 3.0;\nqubit[1] q;\n"
    experiment = RobustPhaseEstimationExperiment(
        round_index=2,
        evolution_time=1.25,
        shots_per_basis=7,
        draw_index=3,
        draw_seed=104,
        circuit_multiplicity=1,
        x_circuit=Circuit(qasm=qasm),
        y_circuit=Circuit(qasm=qasm),
        unitary_builder_configuration=_round().unitary_builder_configuration,
    )
    filename = tmp_path / f"sample.robust_phase_estimation_experiment.{suffix}"

    experiment.to_file(filename, suffix)
    restored = RobustPhaseEstimationExperiment.from_file(filename, suffix)

    assert restored.content_hash() == experiment.content_hash()
    assert restored.draw_seed == experiment.draw_seed
    assert restored.unitary_builder_configuration.settings is not None
    assert restored.unitary_builder_configuration.settings.get("seed") == 101
    assert isinstance(restored.x_circuit.get_qre_application(), OpenQASMApplication)
    assert isinstance(restored.y_circuit.get_qre_application(), OpenQASMApplication)
