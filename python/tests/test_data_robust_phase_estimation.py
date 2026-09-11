"""Tests for serializable robust phase estimation workload data."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from dataclasses import FrozenInstanceError
from pathlib import Path

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
        round_index=0,
        evolution_time=1.25,
        shots_per_basis=7,
        num_draws=7,
        scheduled_samples=32,
        unitary_builder_configuration=_algorithm_ref(
            "hamiltonian_unitary_builder", "partially_randomized", time=1.25, seed=101
        ),
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


def _circuit_set(
    *,
    rounds: tuple[RobustPhaseEstimationRound, ...] | None = None,
    experiment_specs: tuple[RobustPhaseEstimationExperimentSpec, ...] | None = None,
) -> RobustPhaseEstimationCircuitSet:
    """Create a representative serializable RPE workload."""
    return RobustPhaseEstimationCircuitSet(
        rounds=(_round(),) if rounds is None else rounds,
        experiment_specs=_experiment_specs() if experiment_specs is None else experiment_specs,
        state_preparation=Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n"),
        qubit_hamiltonian=QubitOperator(pauli_strings=["Z"], coefficients=np.asarray([2.5])),
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


def test_experiment_spec_defines_canonical_pair_indices() -> None:
    """An experiment identity determines its canonical X/Y circuit positions."""
    spec = _experiment_specs()[3]

    assert spec.x_circuit_index == 6
    assert spec.y_circuit_index == 7


def test_circuit_set_is_only_independent_rpe_data_class() -> None:
    """Nested RPE metadata is immutable without defining additional wire types."""
    circuit_set = _circuit_set()

    assert isinstance(circuit_set, DataClass)
    assert not isinstance(circuit_set.rounds[0], DataClass)
    assert not isinstance(circuit_set.experiment_specs[0], DataClass)
    assert circuit_set.data_type_name() == "robust_phase_estimation_circuit_set"
    with pytest.raises(AttributeError, match="Cannot modify immutable"):
        circuit_set.state_preparation = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n")
    with pytest.raises(FrozenInstanceError):
        circuit_set.rounds[0].shots_per_basis = 9


@pytest.mark.parametrize("suffix", ["json", "hdf5"])
def test_circuit_set_file_roundtrip(tmp_path: Path, suffix: str) -> None:
    """The workload, manifest, inputs, and nested settings survive file round trips."""
    circuit_set = _circuit_set()
    filename = tmp_path / f"sample.robust_phase_estimation_circuit_set.{suffix}"

    circuit_set.to_file(filename, suffix)
    restored = RobustPhaseEstimationCircuitSet.from_file(filename, suffix)

    assert restored.content_hash() == circuit_set.content_hash()
    assert restored.rounds == circuit_set.rounds
    assert restored.experiment_specs == circuit_set.experiment_specs
    assert restored.state_preparation.get_qasm() == circuit_set.state_preparation.get_qasm()
    assert restored.qubit_hamiltonian.content_hash() == circuit_set.qubit_hamiltonian.content_hash()
    assert restored.hadamard_test_circuit_builder_configuration.settings.get("test_basis") == "X"


def test_circuit_set_serialization_guards_version() -> None:
    """Circuit-set deserialization rejects incompatible wire versions."""
    payload = _circuit_set().to_json()
    payload["version"] = "999.0.0"

    with pytest.raises(RuntimeError, match="version"):
        RobustPhaseEstimationCircuitSet.from_json(payload)


def test_round_configuration_is_defensive() -> None:
    """A caller cannot mutate the algorithm settings stored by round metadata."""
    round_data = _round()
    first = round_data.unitary_builder_configuration
    first.settings.set("time", 99.0)

    second = round_data.unitary_builder_configuration

    assert second.settings.get("time") == pytest.approx(round_data.evolution_time)


def test_manifest_rejects_inconsistent_shot_total() -> None:
    """Per-experiment shots must reconstruct each round's declared basis workload."""
    specs = list(_experiment_specs())
    specs[-1] = RobustPhaseEstimationExperimentSpec(
        experiment_index=6,
        round_index=0,
        draw_index=6,
        draw_seed=107,
        shots=2,
    )

    with pytest.raises(ValueError, match="do not sum"):
        _circuit_set(experiment_specs=tuple(specs))


def test_deterministic_manifest_uses_one_multi_shot_pair() -> None:
    """A deterministic round maps its complete basis workload to one circuit pair."""
    round_data = RobustPhaseEstimationRound(
        round_index=0,
        evolution_time=1.25,
        shots_per_basis=7,
        num_draws=1,
        scheduled_samples=32,
        unitary_builder_configuration=_algorithm_ref("hamiltonian_unitary_builder", "trotter", time=1.25),
    )
    spec = RobustPhaseEstimationExperimentSpec(
        experiment_index=0,
        round_index=0,
        draw_index=None,
        draw_seed=None,
        shots=7,
    )

    circuit_set = _circuit_set(rounds=(round_data,), experiment_specs=(spec,))

    assert circuit_set.experiment_specs_for_round(0) == (spec,)


def test_rebind_replaces_live_inputs_without_rescheduling() -> None:
    """Rebinding preserves the manifest and concrete randomized seeds."""
    circuit_set = _circuit_set()
    state_preparation = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nx q[0];\n")

    rebound = circuit_set.rebind(state_preparation)

    assert rebound.experiment_specs == circuit_set.experiment_specs
    assert rebound.rounds == circuit_set.rounds
    assert rebound.state_preparation.get_qasm() == state_preparation.get_qasm()
