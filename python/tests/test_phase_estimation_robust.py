"""Tests for robust phase estimation (qdk_robust).

Fast tests inject circuit-generation and execution fakes at the same public
builder/executor boundaries used by production. Slow tests exercise the full
QDK circuit stack against exact diagonalization.
"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qdk_chemistry import algorithms
from qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder import (
    RobustPhaseEstimationCircuitBuilder,
    RobustPhaseEstimationCircuitSet,
    RobustPhaseEstimationSchedule,
)
from qdk_chemistry.algorithms.phase_estimation.robust_phase_estimation import (
    RobustPhaseEstimation,
    _rpe_angle_update,
    _RpeExecutionResult,
)
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    CircuitExecutorData,
    QuantumErrorProfile,
    QubitOperator,
    Settings,
    SettingsAreLocked,
    UnitaryRepresentation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_HAS_QSHARP = importlib.util.find_spec("qdk.qsharp") is not None
_RUN_SLOW_TESTS = os.getenv("QDK_CHEMISTRY_RUN_SLOW_TESTS", "").lower() in {"1", "true", "yes"}
_RANDOMIZED_ACCURACY_MARKS = (
    pytest.mark.slow,
    pytest.mark.skipif(
        not _RUN_SLOW_TESTS,
        reason="Skipping slow randomized accuracy test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
    ),
)
_DUMMY_STATE_PREPARATION = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n")

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def test_rpe_angle_update_picks_alias_closest_to_previous() -> None:
    """Phase reconstruction selects the alias nearest the previous estimate."""
    measured_phase = 0.4
    assert _rpe_angle_update(3.3, measured_phase, round_index=1) == pytest.approx((measured_phase + 2 * np.pi) / 2)
    assert _rpe_angle_update(0.0, measured_phase, round_index=1) == pytest.approx(measured_phase / 2)


def test_rpe_angle_update_round_zero_returns_measured_phase() -> None:
    """The base round has no aliases and retains its measured phase."""
    assert _rpe_angle_update(0.0, 0.3, round_index=0) == pytest.approx(0.3)


def test_rpe_angle_update_rejects_negative_round() -> None:
    """Phase reconstruction rejects a negative round index."""
    with pytest.raises(ValueError, match="round_index"):
        _rpe_angle_update(0.0, 0.1, round_index=-1)


def _exhaustive_alias_update(previous_angle: float, measured_phase: float, round_index: int) -> float:
    """Return the closest alias by checking every candidate in index order."""
    scale = 2**round_index
    best_candidate = measured_phase / scale
    best_distance = abs(float((best_candidate - previous_angle + np.pi) % (2 * np.pi) - np.pi))
    for alias_index in range(1, scale):
        candidate = (measured_phase + 2 * np.pi * alias_index) / scale
        distance = abs(float((candidate - previous_angle + np.pi) % (2 * np.pi) - np.pi))
        if distance < best_distance:
            best_distance, best_candidate = distance, candidate
    return float(best_candidate)


@pytest.mark.parametrize("round_index", [0, 1, 3, 8])
def test_rpe_angle_update_matches_exhaustive_alias_search(round_index: int) -> None:
    """Constant-time alias selection matches an exhaustive search, including equidistant aliases."""
    scale = 2**round_index
    rng = np.random.default_rng(round_index)
    cases = [(float(previous), float(measured)) for previous, measured in rng.uniform(-2 * np.pi, 2 * np.pi, (500, 2))]
    cases += [((0.25 + np.pi * (2 * alias + 1)) / scale, 0.25) for alias in range(min(scale, 8))]

    for previous_angle, measured_phase in cases:
        expected = _exhaustive_alias_update(previous_angle, measured_phase, round_index)
        assert _rpe_angle_update(previous_angle, measured_phase, round_index) == expected


def test_resolve_energy_inverts_linear_phase() -> None:
    """Linear phase-to-energy conversion preserves the sign convention."""
    base_time = np.pi / 2
    energy = RobustPhaseEstimation._resolve_energy(
        -0.75 * base_time,
        base_time,
        0,
        1.0,
        1,
        correction="linear",
    )
    assert energy == pytest.approx(0.75)


def test_resolve_energy_rejects_nonpositive_time() -> None:
    """Phase-to-energy conversion requires a positive base time."""
    with pytest.raises(ValueError, match="base_time"):
        RobustPhaseEstimation._resolve_energy(0.1, 0.0, 0, 1.0, 1, correction="linear")


@pytest.mark.parametrize("theta", [-np.pi, np.pi, 3 * np.pi])
def test_resolve_energy_preserves_principal_interval_boundary(theta: float) -> None:
    """Both representations of the phase cut map to the closed negative-angle endpoint."""
    assert RobustPhaseEstimation._resolve_energy(theta, 1.0, 0, 1.0, 1, correction="linear") == pytest.approx(np.pi)


def _qdrift_forward_phase(energy: float, lambda_norm: float, evolution_time: float, num_samples: int) -> float:
    """Return the expected qDRIFT signal phase for one eigenenergy."""
    step_angle = lambda_norm * evolution_time / num_samples
    return -num_samples * np.arctan((energy / lambda_norm) * np.tan(step_angle))


@pytest.mark.parametrize("energy", [0.5, -0.5, 0.123])
def test_resolve_energy_inverts_qdrift_phase(energy: float) -> None:
    """The qDRIFT tangent correction removes the finite-sample phase bias."""
    lambda_norm, base_time, num_samples = 1.0, 0.3, 8
    phase = _qdrift_forward_phase(energy, lambda_norm, base_time, num_samples)
    recovered = RobustPhaseEstimation._resolve_energy(
        phase,
        base_time,
        0,
        lambda_norm,
        num_samples,
        correction="qdrift_tangent",
    )
    assert recovered == pytest.approx(energy, abs=1e-9)


def test_qdrift_correction_beats_linear_and_bias_shrinks() -> None:
    """Tangent correction removes qDRIFT bias that shrinks with sample count."""
    energy, lambda_norm, base_time = 0.6, 1.0, 0.5
    linear_errors: list[float] = []
    for num_samples in (2, 8, 32, 128):
        phase = _qdrift_forward_phase(energy, lambda_norm, base_time, num_samples)
        corrected = RobustPhaseEstimation._resolve_energy(
            phase,
            base_time,
            0,
            lambda_norm,
            num_samples,
            correction="qdrift_tangent",
        )
        linear_error = abs((-phase / base_time) - energy)
        assert abs(corrected - energy) < linear_error
        linear_errors.append(linear_error)
    assert linear_errors == sorted(linear_errors, reverse=True)


@pytest.mark.parametrize("energy", [0.75, 0.25, -0.6, 1.1])
def test_rpe_math_recovers_energy_from_ideal_signal(energy: float) -> None:
    """Ideal geometric-ladder phases reconstruct the corresponding eigenenergy."""
    lambda_norm = 1.5
    base_time = np.pi / (2 * lambda_norm)
    total_rounds = int(np.ceil(np.log2(lambda_norm / 1e-3)))
    theta = 0.0
    for round_index in range(total_rounds + 1):
        evolution_time = (2**round_index) * base_time
        measured_phase = float(np.angle(np.exp(-1j * energy * evolution_time)))
        theta = _rpe_angle_update(theta, measured_phase, round_index)

    recovered = RobustPhaseEstimation._resolve_energy(
        theta,
        base_time,
        total_rounds,
        lambda_norm,
        1,
        correction="linear",
    )
    assert recovered == pytest.approx(energy, abs=1e-3)


def _dense_from_pauli(pauli_strings: list[str], coefficients: list[float]) -> np.ndarray:
    """Build the dense matrix of a Pauli-sum Hamiltonian (little-endian labels)."""
    dim = 2 ** len(pauli_strings[0])
    matrix = np.zeros((dim, dim), dtype=complex)
    for label, coeff in zip(pauli_strings, coefficients, strict=True):
        term = np.array([[1]], dtype=complex)
        for char in label:
            term = np.kron(term, _PAULI[char])
        matrix += coeff * term
    return matrix


@dataclass(frozen=True)
class _FakeUnitary:
    """Minimal unitary marker carrying the configured time and seed."""

    time: float
    seed: int | None


class _FakeUnitaryBuilder:
    """Build fake unitaries and record exact settings."""

    def __init__(self, settings: Settings, records: list[dict[str, object]]) -> None:
        self._settings = settings
        self._records = records

    def run(self, qubit_hamiltonian: QubitOperator) -> _FakeUnitary:
        """Record one build and return its configured time and seed."""
        record = self._settings.to_dict()
        record["num_qubits"] = qubit_hamiltonian.num_qubits
        self._records.append(record)
        seed = int(self._settings.get("seed")) if self._settings.has("seed") else None
        return _FakeUnitary(time=float(self._settings.get("time")), seed=seed)


class _FakeHadamardCircuitBuilder:
    """Associate generated circuits with the basis and unitary they represent."""

    def __init__(self, settings: Settings, contexts: dict[int, tuple[str, object]]) -> None:
        self._settings = settings
        self._contexts = contexts

    def run(self, state_preparation: Circuit, unitary: object) -> Circuit:
        """Return a circuit linked to its basis and shared unitary."""
        assert isinstance(state_preparation, Circuit)
        circuit = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\n")
        self._contexts[id(circuit)] = (str(self._settings.get("test_basis")), unitary)
        return circuit


class _FakeExecutorData:
    """Minimal executor result containing bitstring counts."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.bitstring_counts = counts


class _FakeExecutor:
    """Evaluate generated circuit contexts with a supplied signal function."""

    def __init__(
        self,
        contexts: dict[int, tuple[str, object]],
        expectation: Callable[[object, str], float],
        resolution: int = 2_000_000,
    ) -> None:
        self._contexts = contexts
        self._expectation = expectation
        self._resolution = resolution
        self.shot_calls: list[int] = []
        self.noise_calls: list[QuantumErrorProfile | None] = []
        self.seed_calls: list[int | None] = []

    def run(
        self,
        circuit: Circuit,
        *,
        shots: int,
        noise: QuantumErrorProfile | None = None,
    ) -> _FakeExecutorData:
        """Convert the configured expectation into deterministic counts."""
        self.shot_calls.append(shots)
        self.noise_calls.append(noise)
        basis, unitary = self._contexts[id(circuit)]
        expectation = self._expectation(unitary, basis)
        n0 = round((1.0 + expectation) / 2.0 * self._resolution)
        return _FakeExecutorData({"0": int(n0), "1": int(self._resolution - n0)})


def _make_builder(
    *,
    target_accuracy: float,
    unitary_builder_name: str = "trotter",
    unitary_builder_kwargs: dict[str, object] | None = None,
    base_time: float = 0.0,
    unitary_accuracy_fraction: float | None = None,
    energy_correction: str = "auto",
    seed: int = 7,
    epsilon_rpe: float | None = None,
    epsilon_unitary: float | None = None,
    max_qdrift_samples: int = 1_000_000,
) -> RobustPhaseEstimationCircuitBuilder:
    """Create a robust circuit builder with the requested scheduling and evolution settings."""
    return RobustPhaseEstimationCircuitBuilder(
        target_accuracy=target_accuracy,
        base_time=base_time,
        unitary_accuracy_fraction=unitary_accuracy_fraction,
        energy_correction=energy_correction,
        seed=seed,
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
        max_qdrift_samples=max_qdrift_samples,
        unitary_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            unitary_builder_name,
            **(unitary_builder_kwargs or {}),
        ),
    )


def _install_test_stack(
    monkeypatch: pytest.MonkeyPatch,
    driver: RobustPhaseEstimation,
    circuit_builder: RobustPhaseEstimationCircuitBuilder,
    expectation: Callable[[object, str], float],
    *,
    use_real_unitary_builder: bool = False,
    resolution: int = 2_000_000,
) -> tuple[list[dict[str, object]], _FakeExecutor]:
    """Install fake circuit/execution boundaries around the real RPE orchestration."""
    contexts: dict[int, tuple[str, object]] = {}
    unitary_records: list[dict[str, object]] = []
    executor = _FakeExecutor(contexts, expectation, resolution=resolution)
    original_create = algorithms.create

    def create_algorithm(algorithm_type: str, algorithm_name: str, **settings_values: object) -> object:
        algorithm = original_create(algorithm_type, algorithm_name, **settings_values)
        settings = algorithm.settings()
        if algorithm_type == "hamiltonian_unitary_builder":
            if use_real_unitary_builder:
                return algorithm
            return _FakeUnitaryBuilder(settings, unitary_records)
        if algorithm_type == "hadamard_test_circuit_builder":
            return _FakeHadamardCircuitBuilder(settings, contexts)
        raise AssertionError(f"Unexpected algorithm type: {algorithm_type}")

    def create_nested(setting_key: str):
        if setting_key == "qpe_circuit_builder":
            return circuit_builder
        if setting_key == "circuit_executor":
            return executor
        raise KeyError(setting_key)

    def create_executor(seed: int | None):
        executor.seed_calls.append(seed)
        return executor

    monkeypatch.setattr(algorithms, "create", create_algorithm)
    monkeypatch.setattr(driver, "_create_nested", create_nested)
    monkeypatch.setattr(driver, "_create_executor", create_executor)
    return unitary_records, executor


def _ideal_expectation(energy: float, signal_factor: complex = 1.0 + 0.0j) -> Callable[[object, str], float]:
    """Return an ideal Hadamard expectation function for a known eigenenergy."""

    def expectation(unitary: object, basis: str) -> float:
        assert isinstance(unitary, _FakeUnitary)
        signal = signal_factor * np.exp(-1j * energy * unitary.time)
        return float(signal.real) if basis == "X" else float(signal.imag)

    return expectation


def test_post_process_uses_experiment_identity_after_reordering() -> None:
    """Result tuple order does not replace the manifest's round and basis identity."""
    energy = 0.2
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    circuit_set = _make_builder(target_accuracy=0.5, energy_correction="linear").build_schedule(hamiltonian)
    execution_results: list[_RpeExecutionResult] = []
    resolution = 2_000_000
    for spec in circuit_set.experiment_specs:
        evolution_time = circuit_set.rounds[spec.round_index].evolution_time
        signal = np.exp(-1j * energy * evolution_time)
        basis_results = []
        for expectation in (float(signal.real), float(signal.imag)):
            num_zero = round((1.0 + expectation) * resolution / 2.0)
            basis_results.append(_FakeExecutorData({"0": num_zero, "1": resolution - num_zero}))
        execution_results.append(_RpeExecutionResult(spec, basis_results[0], basis_results[1]))

    result = RobustPhaseEstimation()._post_process(
        circuit_set,
        tuple(reversed(execution_results)),
        requested_executor_seed=None,
        executor_root_seed=None,
    )

    assert result.resolved_energy == pytest.approx(energy, abs=1e-6)


@pytest.mark.parametrize("energy", [-0.2, 0.2])
@pytest.mark.parametrize(("base_time", "nested_accuracy"), [(0.0, 0.25), (3.0, 0.0), (np.pi - 1e-3, 0.0), (3.0, 0.25)])
def test_post_process_uses_effective_qdrift_samples(energy: float, base_time: float, nested_accuracy: float) -> None:
    """Noiseless qDRIFT signals track the correct branch through the final tangent inversion."""
    hamiltonian = QubitOperator(pauli_strings=["ZI", "IZ"], coefficients=[0.6, 0.4])
    circuit_set = _make_builder(
        target_accuracy=0.1,
        base_time=base_time,
        unitary_builder_name="qdrift",
        unitary_builder_kwargs={"target_accuracy": nested_accuracy},
    ).build_schedule(hamiltonian)
    execution_results: list[_RpeExecutionResult] = []
    resolution = 1_000_000_000
    for round_data in circuit_set.rounds:
        configuration = circuit_set.unitary_builder_configuration
        builder = algorithms.create(
            configuration.algorithm_type,
            configuration.algorithm_name,
            **configuration.settings.to_dict(),
            time=round_data.evolution_time,
            num_samples=round_data.scheduled_samples,
        )
        actual_samples = builder._resolve_num_samples(hamiltonian, round_data.evolution_time)
        step_angle = circuit_set.lambda_norm * round_data.evolution_time / actual_samples
        signal = (np.cos(step_angle) - 1j * energy / circuit_set.lambda_norm * np.sin(step_angle)) ** actual_samples
        basis_results = []
        for expectation in (float(signal.real), float(signal.imag)):
            num_zero = round((1.0 + expectation) * resolution / 2.0)
            basis_results.append(_FakeExecutorData({"0": num_zero, "1": resolution - num_zero}))
        execution_results.extend(
            _RpeExecutionResult(spec, basis_results[0], basis_results[1])
            for spec in circuit_set.experiment_specs_for_round(round_data.round_index)
        )

    result = RobustPhaseEstimation()._post_process(
        circuit_set,
        tuple(execution_results),
        requested_executor_seed=None,
        executor_root_seed=None,
    )

    assert result.resolved_energy == pytest.approx(energy, abs=1e-8)


def test_post_process_averages_each_randomized_draw() -> None:
    """Different count totals do not change the equal weighting of randomized draws."""
    circuit_set = _make_builder(
        target_accuracy=1.0, unitary_builder_name="qdrift", energy_correction="linear"
    ).build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=np.array([1.0])))
    assert circuit_set.num_rounds == 1
    assert len(circuit_set.experiment_specs) == 30
    results = tuple(
        _RpeExecutionResult(
            spec,
            CircuitExecutorData({"0": 1} if index < 15 else {"1": 100}, 1 if index < 15 else 100, "test"),
            CircuitExecutorData({"0": 1}, 1, "test"),
        )
        for index, spec in enumerate(circuit_set.experiment_specs)
    )

    result = RobustPhaseEstimation()._post_process(
        circuit_set, results, requested_executor_seed=None, executor_root_seed=None
    )

    assert result.resolved_energy == pytest.approx(-1.0)


@pytest.mark.parametrize("energy", [0.4, -0.3, 0.75, 0.0])
def test_driver_recovers_energy_exact_mode(monkeypatch: pytest.MonkeyPatch, energy: float) -> None:
    """Linear RPE recovers an injected ideal energy through builder/executor composition."""
    hamiltonian = QubitOperator(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    builder = _make_builder(target_accuracy=1e-4, energy_correction="linear")
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(energy))

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)


@pytest.mark.parametrize("energy", [0.4, -0.3, 0.75])
def test_driver_recovers_energy_qdrift_mode(monkeypatch: pytest.MonkeyPatch, energy: float) -> None:
    """The qDRIFT tangent map leaves an injected ideal signal effectively unchanged."""
    hamiltonian = QubitOperator(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    builder = _make_builder(
        target_accuracy=1e-4,
        unitary_builder_name="qdrift",
        energy_correction="qdrift_tangent",
        max_qdrift_samples=2**29,
    )
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(energy))

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)


@pytest.mark.parametrize("mutate_input", [False, True])
def test_qdrift_tangent_correction_uses_sampled_norm(monkeypatch: pytest.MonkeyPatch, mutate_input: bool) -> None:
    """Thresholded qDRIFT corrects with the sampled-term norm of the input snapshot that built the circuits."""
    pauli_strings, coefficients = ["ZI", "IZ"], [0.6, 0.4]
    hamiltonian = QubitOperator(pauli_strings=pauli_strings, coefficients=coefficients)
    _, eigenvectors = np.linalg.eigh(_dense_from_pauli(pauli_strings, coefficients))
    readout = _classical_expectation(eigenvectors[:, 0])

    def expectation(unitary: object, basis: str) -> float:
        if mutate_input:
            hamiltonian.coefficients[0] = 0.9
        return readout(unitary, basis)

    builder = _make_builder(
        target_accuracy=0.5,
        unitary_builder_name="qdrift",
        unitary_builder_kwargs={"weight_threshold": 0.5},
        energy_correction="qdrift_tangent",
    )
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, expectation, use_real_unitary_builder=True)

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    assert result.metadata["lambda"] == pytest.approx(1.0)
    assert result.resolved_energy == pytest.approx(-0.6, abs=1e-6)


def test_qdrift_tangent_replay_resolves_omitted_builder_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """A qDRIFT schedule without explicit builder defaults replays with the default sampling threshold."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[0.5])
    builder = _make_builder(target_accuracy=0.5, unitary_builder_name="qdrift", energy_correction="qdrift_tangent")
    driver = RobustPhaseEstimation()
    _install_test_stack(
        monkeypatch, driver, builder, _classical_expectation(np.array([0.0, 1.0])), use_real_unitary_builder=True
    )
    schedule = builder.build_schedule(hamiltonian)
    sparse = replace(schedule, _unitary_builder=("hamiltonian_unitary_builder", "qdrift", Settings().to_json()))
    workload = RobustPhaseEstimationCircuitSet(
        schedule=sparse, state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian
    )
    assert not sparse.unitary_builder_configuration.settings.has("weight_threshold")

    result = driver.execute_circuit_set(workload)

    assert result.resolved_energy == pytest.approx(-0.5, abs=1e-5)


def test_driver_uses_explicit_base_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A builder-provided base time is honored during execution."""
    energy = 0.6
    hamiltonian = QubitOperator(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    builder = _make_builder(target_accuracy=1e-4, base_time=np.pi / 4, energy_correction="linear")
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(energy))

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)
    assert result.metadata["base_time"] == pytest.approx(np.pi / 4)


def test_driver_handles_empty_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty executor counts contribute a guarded zero expectation."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    builder = _make_builder(target_accuracy=0.5, energy_correction="linear")
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(0.2), resolution=0)

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(0.0)


def test_robust_phase_estimation_name() -> None:
    """The robust estimator retains its registered name."""
    assert RobustPhaseEstimation().name() == "qdk_robust"


def test_energy_correction_allows_linear_qdrift_override() -> None:
    """An explicit linear map overrides qDRIFT's automatic tangent correction."""
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        seed=7,
        energy_correction="linear",
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "qdrift"),
    )
    circuit_set = builder.build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=np.array([1.0])))

    assert circuit_set.energy_correction == "linear"


@pytest.mark.parametrize("builder_name", ["trotter", "partially_randomized", "zassenhaus"])
def test_qdrift_tangent_requires_qdrift_builder(builder_name: str) -> None:
    """The tangent map models pure qDRIFT and is rejected for other evolution families."""
    builder = RobustPhaseEstimationCircuitBuilder(
        target_accuracy=0.5,
        energy_correction="qdrift_tangent",
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", builder_name),
    )

    with pytest.raises(ValueError, match=f"qdrift_tangent.*{builder_name}"):
        builder.build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=np.array([1.0])))


@pytest.mark.parametrize("base_time", [0.0, np.pi / 32.0, 3.0 * np.pi / 4.0])
def test_non_trotter_product_budget_meets_target_accuracy(monkeypatch: pytest.MonkeyPatch, base_time: float) -> None:
    """Non-Trotter explicit RPE and unitary tolerances retain their product bound."""
    epsilon_total = 0.1
    epsilon_unitary = 0.5
    epsilon_rpe = np.pi * epsilon_total / (2.0 * np.arcsin(epsilon_unitary))
    energy = 0.3
    phase_error = np.arcsin(epsilon_unitary)
    signal_factor = np.sqrt(1.0 - epsilon_unitary**2) * np.exp(1j * phase_error)
    hamiltonian = QubitOperator(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    builder = _make_builder(
        target_accuracy=epsilon_total,
        unitary_builder_name="partially_randomized",
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
        energy_correction="linear",
        base_time=base_time,
    )
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(energy, signal_factor))

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    metadata = result.metadata
    final_round = metadata["num_rounds"] - 1
    final_time = (2**final_round) * metadata["base_time"]
    exact_energy_bound = phase_error / final_time
    propagated_energy_bound = (2.0 / np.pi) * epsilon_rpe * phase_error
    assert abs(signal_factor - 1.0) == pytest.approx(epsilon_unitary)
    assert metadata["epsilon_unitary"] == pytest.approx(epsilon_unitary)
    assert metadata["epsilon_rpe"] == pytest.approx(epsilon_rpe)
    assert metadata["error_budget_mode"] == "explicit"
    assert metadata["energy_correction"] == "linear"
    assert metadata["unitary_builder"] == "partial_randomized"
    assert abs(result.resolved_energy - energy) == pytest.approx(exact_energy_bound, rel=1e-3)
    assert propagated_energy_bound == pytest.approx(epsilon_total)
    assert exact_energy_bound <= propagated_energy_bound


@pytest.mark.parametrize("builder_name", ["trotter", "partially_randomized"])
@pytest.mark.parametrize("base_time", [0.0, np.pi / 32.0])
def test_base_time_preserves_energy_accuracy_with_bounded_phase_bias(
    monkeypatch: pytest.MonkeyPatch, builder_name: str, base_time: float
) -> None:
    """A permitted per-round unitary phase bias respects the energy target at any base time."""
    energy = 0.3
    epsilon = 0.01
    epsilon_unitary = 0.5
    phase_bias = 0.4
    signal_factor = np.exp(1j * phase_bias)
    assert abs(signal_factor - 1.0) < epsilon_unitary
    hamiltonian = QubitOperator(pauli_strings=["I", "Z"], coefficients=[0.65, -0.35])
    builder = _make_builder(
        target_accuracy=epsilon,
        unitary_builder_name=builder_name,
        epsilon_unitary=epsilon_unitary,
        base_time=base_time,
        energy_correction="linear",
    )
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(energy, signal_factor))

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    final_time = result.metadata["base_time"] * 2 ** (result.metadata["num_rounds"] - 1)
    energy_error = abs(result.resolved_energy - energy)
    assert energy_error == pytest.approx(phase_bias / final_time, rel=1e-3)
    assert energy_error <= epsilon


def test_non_trotter_explicit_fraction_retains_clamping() -> None:
    """Non-Trotter routing clamps an explicit negative fraction instead of treating it as omitted."""
    builder = _make_builder(
        target_accuracy=0.1,
        unitary_builder_name="partially_randomized",
        unitary_accuracy_fraction=-0.25,
    )

    fraction, epsilon_rpe, epsilon_unitary, budget_mode = builder._resolve_budget(
        "partial_randomized",
        0.1,
        is_trotter=False,
    )

    assert fraction == pytest.approx(0.0)
    assert epsilon_rpe == pytest.approx(0.1)
    assert epsilon_unitary == pytest.approx(0.0)
    assert budget_mode == "fraction"


def test_partial_builder_retains_explicit_legacy_fraction() -> None:
    """An explicit legacy fraction still selects the fractional PR budget route."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    builder = _make_builder(
        target_accuracy=0.1,
        unitary_builder_name="partially_randomized",
        unitary_accuracy_fraction=0.25,
    )

    circuit_set = builder.build_schedule(hamiltonian)

    assert circuit_set.unitary_accuracy_fraction == pytest.approx(0.25)
    assert circuit_set.epsilon_rpe == pytest.approx(0.075)
    assert circuit_set.epsilon_unitary == pytest.approx(0.025)
    assert circuit_set.error_budget_mode == "fraction"
    settings = circuit_set.unitary_builder_configuration.settings
    assert settings is not None
    assert settings.get("target_accuracy") == pytest.approx(0.025)


@pytest.mark.parametrize("unitary_builder_name", ["partially_randomized", "zassenhaus"])
@pytest.mark.parametrize("fraction", [0.25, np.nextafter(np.sin(np.pi / 3.0), 0.0)])
def test_non_trotter_fraction_accepts_branch_safe_budget(unitary_builder_name: str, fraction: float) -> None:
    """Fraction budgets below the branch-safety limit preserve the requested split."""
    builder = _make_builder(
        target_accuracy=1.0,
        unitary_builder_name=unitary_builder_name,
        unitary_accuracy_fraction=fraction,
    )

    schedule = builder.build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=[1.0]))

    assert schedule.epsilon_unitary == pytest.approx(fraction)
    assert schedule.epsilon_rpe == pytest.approx(1.0 - fraction)


@pytest.mark.parametrize("unitary_builder_name", ["partially_randomized", "zassenhaus"])
@pytest.mark.parametrize("fraction", [np.sin(np.pi / 3.0), np.nextafter(np.sin(np.pi / 3.0), 1.0), 0.9])
def test_non_trotter_fraction_rejects_branch_unsafe_budget(unitary_builder_name: str, fraction: float) -> None:
    """Fraction budgets enforce the same strict unitary limit as explicit budgets."""
    builder = _make_builder(
        target_accuracy=1.0,
        unitary_builder_name=unitary_builder_name,
        unitary_accuracy_fraction=fraction,
    )

    with pytest.raises(ValueError, match="epsilon_unitary must be smaller"):
        builder.build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=[1.0]))


@pytest.mark.parametrize("unitary_builder_name", ["partially_randomized", "zassenhaus"])
@pytest.mark.parametrize("fraction", [1.0, 1.5])
def test_non_trotter_fraction_rejects_unit_fraction(unitary_builder_name: str, fraction: float) -> None:
    """A fraction of one or more would leave the RPE ladder without an energy budget."""
    builder = _make_builder(
        target_accuracy=0.1,
        unitary_builder_name=unitary_builder_name,
        unitary_accuracy_fraction=fraction,
    )

    with pytest.raises(ValueError, match="unitary_accuracy_fraction must be smaller than 1"):
        builder.build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=[1.0]))


@pytest.mark.parametrize("unitary_builder_name", ["partially_randomized", "zassenhaus"])
@pytest.mark.parametrize("epsilon_rpe", [0.0, -0.5])
def test_non_trotter_rejects_nonpositive_explicit_rpe_budget(unitary_builder_name: str, epsilon_rpe: float) -> None:
    """Only the unset sentinel omits an RPE tolerance; any other explicit value must be positive."""
    builder = _make_builder(
        target_accuracy=0.1,
        unitary_builder_name=unitary_builder_name,
        epsilon_rpe=epsilon_rpe,
    )

    with pytest.raises(ValueError, match="must both be positive"):
        builder.build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=[1.0]))


def test_trotter_uses_independent_default_tolerances() -> None:
    """Trotter uses the full energy target for RPE and an independent unitary tolerance."""
    target_accuracy = 1e-2
    hamiltonian = QubitOperator(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    builder = _make_builder(target_accuracy=target_accuracy)

    circuit_set = builder.build_schedule(hamiltonian)

    assert circuit_set.epsilon_rpe == pytest.approx(target_accuracy)
    assert circuit_set.epsilon_unitary == pytest.approx(0.85)
    assert circuit_set.unitary_accuracy_fraction == pytest.approx(0.0)
    assert circuit_set.error_budget_mode == "independent_trotter"
    assert circuit_set.num_rounds == 8
    ref = circuit_set.unitary_builder_configuration
    assert ref.settings is not None
    assert ref.settings.get("target_accuracy") == pytest.approx(0.85)


@pytest.mark.parametrize("epsilon_unitary", [0.5, 0.9, 2.5])
def test_trotter_accepts_positive_unitary_tolerance(epsilon_unitary: float) -> None:
    """A positive Trotter unitary tolerance is forwarded without additional policy checks."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    builder = _make_builder(target_accuracy=1e-2, epsilon_unitary=epsilon_unitary)

    circuit_set = builder.build_schedule(hamiltonian)

    assert circuit_set.epsilon_unitary == pytest.approx(epsilon_unitary)
    ref = circuit_set.unitary_builder_configuration
    assert ref.settings is not None
    assert ref.settings.get("target_accuracy") == pytest.approx(epsilon_unitary)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("unitary_accuracy_fraction", 0.5, "unitary_accuracy_fraction is not supported"),
        ("epsilon_rpe", 1e-2, "epsilon_rpe is not configurable"),
        ("epsilon_unitary", 0.0, "epsilon_unitary must be positive"),
        ("epsilon_unitary", -0.5, "epsilon_unitary must be positive"),
    ],
)
def test_trotter_rejects_legacy_or_nonpositive_tolerances(
    setting: str,
    value: float,
    message: str,
) -> None:
    """Trotter rejects dimensional legacy routing and nonpositive sizing inputs."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    builder = _make_builder(target_accuracy=1e-2)
    builder.settings().set(setting, value)

    with pytest.raises(ValueError, match=message):
        builder.build_schedule(hamiltonian)


@pytest.mark.parametrize("epsilon_unitary", [None, 0.5])
@pytest.mark.parametrize("split", [0.1, 0.5, 0.9])
def test_partial_builder_receives_independent_unitary_budget(epsilon_unitary: float | None, split: float) -> None:
    """Partially randomized rounds and standalone builders share the additive accuracy contract."""
    hamiltonian = QubitOperator(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    epsilon_total = 1e-2
    builder = _make_builder(
        target_accuracy=epsilon_total,
        unitary_builder_name="partially_randomized",
        unitary_builder_kwargs={"accuracy_split": split},
        energy_correction="linear",
        seed=5,
        epsilon_unitary=epsilon_unitary,
    )

    circuit_set = builder.build_schedule(hamiltonian)

    outer_epsilon_unitary = 0.85 if epsilon_unitary is None else epsilon_unitary
    assert circuit_set.epsilon_rpe == pytest.approx(epsilon_total)
    assert circuit_set.epsilon_unitary == pytest.approx(outer_epsilon_unitary)
    assert circuit_set.unitary_accuracy_fraction == pytest.approx(0.0)
    assert circuit_set.error_budget_mode == "independent_partial_randomized"
    for round_data in circuit_set.rounds:
        ref = circuit_set.unitary_builder_configuration
        assert ref.settings is not None
        assert ref.settings.get("target_accuracy") == pytest.approx(outer_epsilon_unitary)
        nested_builder = algorithms.create(ref.algorithm_type, ref.algorithm_name, **ref.settings.to_dict())
        assert sum(nested_builder._split_accuracy()) == pytest.approx(outer_epsilon_unitary)
        assert not ref.settings.has("num_samples")
        assert round_data.num_draws == round_data.shots_per_basis
        draw_seeds = [spec.draw_seed for spec in circuit_set.experiment_specs_for_round(round_data.round_index)]
        assert len(set(draw_seeds)) == round_data.shots_per_basis


@pytest.mark.parametrize(
    ("epsilon_unitary", "message"),
    [
        (0.0, "epsilon_unitary must be positive"),
        (-0.5, "epsilon_unitary must be positive"),
        (np.sin(np.pi / 3.0), "epsilon_unitary must be smaller"),
    ],
)
def test_partial_builder_rejects_invalid_independent_unitary_budget(
    epsilon_unitary: float,
    message: str,
) -> None:
    """Independent PR routing rejects nonpositive and branch-unsafe tolerances."""
    builder = _make_builder(
        target_accuracy=0.1,
        unitary_builder_name="partially_randomized",
        epsilon_unitary=epsilon_unitary,
    )

    with pytest.raises(ValueError, match=message):
        builder.build_schedule(QubitOperator(pauli_strings=["Z"], coefficients=[1.0]))


@pytest.mark.parametrize(
    ("unitary_name", "randomized", "expected_seed_hash"),
    [
        ("trotter", False, "90d4cf4b544ca3a5e9bc7afe04cd16d27529d13bb56345bd127bfc5d2a799513"),
        ("qdrift", True, "25aa67aeea22d3285959c7e622164302d90bb3d2a3026574678c1c2b95a7084a"),
    ],
)
def test_executor_uses_manifest_shots(
    monkeypatch: pytest.MonkeyPatch,
    unitary_name: str,
    randomized: bool,
    expected_seed_hash: str,
) -> None:
    """Execution honors deterministic multi-shot specs and randomized one-shot draws."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    builder = _make_builder(target_accuracy=0.5, unitary_builder_name=unitary_name)
    driver = RobustPhaseEstimation()
    _, executor = _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(0.2))

    circuit_set = driver.schedule_circuit_set(_DUMMY_STATE_PREPARATION, hamiltonian)
    original_specs = circuit_set.schedule.experiment_specs

    def unexpected_schedule(
        _self: RobustPhaseEstimationCircuitBuilder, _hamiltonian: QubitOperator
    ) -> RobustPhaseEstimationSchedule:
        pytest.fail("Replay must not build a new schedule.")

    monkeypatch.setattr(RobustPhaseEstimationCircuitBuilder, "build_schedule", unexpected_schedule)
    result = driver.execute_circuit_set(circuit_set)

    assert circuit_set.schedule.experiment_specs == original_specs
    with pytest.raises(SettingsAreLocked):
        driver.settings().set("circuit_executor", driver.settings().get("circuit_executor"))
    if randomized:
        expected = [1 for round_data in circuit_set.schedule.rounds for _ in range(2 * round_data.num_draws)]
    else:
        expected = [shots for round_data in circuit_set.schedule.rounds for shots in (round_data.shots_per_basis,) * 2]
    assert executor.shot_calls == expected
    assert len(executor.seed_calls) == len(expected)
    seed_bytes = np.asarray(executor.seed_calls, dtype="<u4").tobytes()
    assert hashlib.sha256(seed_bytes).hexdigest() == expected_seed_hash
    assert len(set(executor.seed_calls)) == len(executor.seed_calls)
    assert result.metadata["requested_executor_seed"] == 42
    assert result.metadata["executor_root_seed"] == 42


@pytest.mark.parametrize("replay", [False, True])
def test_failed_execution_keeps_settings_locked(monkeypatch: pytest.MonkeyPatch, replay: bool) -> None:
    """Normal execution and replay lock settings before execution, even when it fails."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    builder = _make_builder(target_accuracy=0.5)
    driver = RobustPhaseEstimation()
    monkeypatch.setattr(driver, "_create_circuit_builder", lambda: builder)
    circuit_set = driver.schedule_circuit_set(_DUMMY_STATE_PREPARATION, hamiltonian)

    def fail_execution(*_args: object, **_kwargs: object) -> None:
        with pytest.raises(SettingsAreLocked):
            driver.settings().set("circuit_executor", driver.settings().get("circuit_executor"))
        raise RuntimeError("test execution failure")

    monkeypatch.setattr(driver, "_execute_with_builder", fail_execution)

    if replay:
        with pytest.raises(RuntimeError, match="test execution failure"):
            driver.execute_circuit_set(circuit_set)
    else:
        with pytest.raises(RuntimeError, match="test execution failure"):
            driver.run(_DUMMY_STATE_PREPARATION, hamiltonian)

    with pytest.raises(SettingsAreLocked):
        driver.settings().set("circuit_executor", driver.settings().get("circuit_executor"))


def test_executor_entropy_seed_is_resolved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """One entropy-backed measurement root is shared by the entire execution."""
    driver = RobustPhaseEstimation(
        circuit_executor=AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=-1)
    )
    builder = _make_builder(target_accuracy=0.5)
    _, executor = _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(0.2))
    original_seed_sequence = np.random.SeedSequence
    entropy_calls: list[int] = []

    def seed_sequence(entropy: int | list[int] | None = None) -> np.random.SeedSequence:
        if entropy is None:
            entropy_calls.append(1234)
            return original_seed_sequence(1234)
        return original_seed_sequence(entropy)

    monkeypatch.setattr(np.random, "SeedSequence", seed_sequence)
    result = driver.run(_DUMMY_STATE_PREPARATION, QubitOperator(pauli_strings=["Z"], coefficients=np.array([1.0])))

    assert entropy_calls == [1234]
    assert result.metadata["requested_executor_seed"] == -1
    assert result.metadata["executor_root_seed"] == int(
        original_seed_sequence(1234).generate_state(1, dtype=np.uint32)[0]
    )
    assert len(set(executor.seed_calls)) == len(executor.seed_calls) == 4


def test_executor_forwards_noise_to_every_x_y_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same noise profile reaches every X and Y circuit execution."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    builder = _make_builder(target_accuracy=0.5)
    driver = RobustPhaseEstimation()
    _, executor = _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(0.2))
    noise = QuantumErrorProfile(name="test noise")

    circuit_set = driver.schedule_circuit_set(_DUMMY_STATE_PREPARATION, hamiltonian)
    driver.execute_circuit_set(circuit_set, noise=noise)

    expected_calls = 2 * sum(round_data.num_draws for round_data in circuit_set.schedule.rounds)
    assert executor.noise_calls == [noise] * expected_calls


_NONCOMMUTING_PAULIS = ["ZI", "XI", "IZ", "ZZ"]
_NONCOMMUTING_COEFFS = [1.0, 0.8, 0.5, 0.3]

_H2_STO3G_PAULIS = [
    "ZIZI",
    "YYYY",
    "XXYY",
    "IIII",
    "XXXX",
    "IIIZ",
    "IZII",
    "IIZI",
    "ZIII",
    "ZIIZ",
    "IIZZ",
    "IZZI",
    "ZZII",
    "IZIZ",
    "YYXX",
]
_H2_STO3G_COEFFS = [
    0.19176479,
    0.04104867,
    0.04104867,
    -0.5734373,
    0.04104867,
    0.23708567,
    0.23708567,
    -0.46083546,
    -0.46083546,
    0.18168163,
    0.14063296,
    0.18168163,
    0.14063296,
    0.18454294,
    0.04104867,
]


def _noncommuting_ground_state_problem() -> tuple[QubitOperator, np.ndarray, float]:
    """Return a noncommuting two-qubit Hamiltonian, ground vector, and ground energy."""
    hamiltonian = QubitOperator(pauli_strings=_NONCOMMUTING_PAULIS, coefficients=_NONCOMMUTING_COEFFS)
    dense = _dense_from_pauli(_NONCOMMUTING_PAULIS, _NONCOMMUTING_COEFFS)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return hamiltonian, eigenvectors[:, 0], float(eigenvalues[0])


def _h2_sto3g_ground_state_problem() -> tuple[QubitOperator, np.ndarray, float]:
    """Return the repository's four-qubit H2/STO-3G ground-state problem."""
    hamiltonian = QubitOperator(pauli_strings=_H2_STO3G_PAULIS, coefficients=_H2_STO3G_COEFFS)
    dense = np.asarray(hamiltonian.to_matrix(sparse=False), dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return hamiltonian, eigenvectors[:, 0], float(eigenvalues[0])


def _materialize_container(container) -> np.ndarray:
    """Materialize a Pauli product formula container as a dense unitary."""
    num_qubits = container.num_qubits
    dim = 2**num_qubits
    identity = np.eye(dim, dtype=complex)
    step = identity.copy()
    for term in container.step_terms:
        labels = ["I"] * num_qubits
        for qubit, op in term.pauli_term.items():
            labels[num_qubits - 1 - qubit] = op
        pauli = _dense_from_pauli(["".join(labels)], [1.0])
        step = (np.cos(term.angle) * identity - 1j * np.sin(term.angle) * pauli) @ step
    return np.linalg.matrix_power(step, container.step_reps)


def _classical_expectation(ground_vector: np.ndarray) -> Callable[[object, str], float]:
    """Return an exact expectation function for real unitary representations."""

    def expectation(unitary: object, basis: str) -> float:
        assert isinstance(unitary, UnitaryRepresentation)
        dense_unitary = _materialize_container(unitary.get_container())
        signal = complex(ground_vector.conj() @ (dense_unitary @ ground_vector))
        return float(signal.real) if basis == "X" else float(signal.imag)

    return expectation


@pytest.mark.parametrize(
    ("builder_name", "builder_kwargs", "expected_category", "expected_correction"),
    [
        ("trotter", {"order": 2}, "deterministic_or_exact", "linear"),
        pytest.param("qdrift", {}, "qdrift", "qdrift_tangent", marks=_RANDOMIZED_ACCURACY_MARKS),
        pytest.param(
            "partially_randomized",
            {"weight_threshold": 0.5, "trotter_order": 2, "num_random_samples": 1},
            "partial_randomized",
            "linear",
            marks=_RANDOMIZED_ACCURACY_MARKS,
        ),
    ],
)
def test_robust_qpe_within_target_accuracy_classical_signal(
    monkeypatch: pytest.MonkeyPatch,
    builder_name: str,
    builder_kwargs: dict[str, object],
    expected_category: str,
    expected_correction: str,
) -> None:
    """All supported evolution categories recover the GSE within target accuracy."""
    epsilon = 0.1
    hamiltonian, ground_vector, ground_energy = _noncommuting_ground_state_problem()
    builder = _make_builder(
        target_accuracy=epsilon,
        unitary_builder_name=builder_name,
        unitary_builder_kwargs=builder_kwargs,
        energy_correction="auto",
        seed=7,
    )
    driver = RobustPhaseEstimation()
    _install_test_stack(
        monkeypatch,
        driver,
        builder,
        _classical_expectation(ground_vector),
        use_real_unitary_builder=True,
    )

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(ground_energy, abs=epsilon)
    assert result.metadata["unitary_builder"] == expected_category
    assert result.metadata["energy_correction"] == expected_correction


@pytest.mark.parametrize(
    ("problem", "epsilon_total"),
    [
        (_noncommuting_ground_state_problem, 0.1),
        (_noncommuting_ground_state_problem, 1e-3),
        (_h2_sto3g_ground_state_problem, 1e-3),
    ],
    ids=["noncommuting-tenth-hartree", "noncommuting-one-millihartree", "h2-sto3g-one-millihartree"],
)
def test_independent_tolerances_bound_trotter_ground_energy(
    monkeypatch: pytest.MonkeyPatch,
    problem: Callable[[], tuple[QubitOperator, np.ndarray, float]],
    epsilon_total: float,
) -> None:
    """Independent defaults bound real order-two Trotter ground-energy estimates."""
    epsilon_rpe = epsilon_total
    epsilon_unitary = 0.85
    hamiltonian, ground_vector, ground_energy = problem()
    lambda_norm = hamiltonian.schatten_norm
    builder = _make_builder(
        target_accuracy=epsilon_total,
        unitary_builder_name="trotter",
        unitary_builder_kwargs={"order": 2},
        energy_correction="linear",
    )
    driver = RobustPhaseEstimation()
    _install_test_stack(
        monkeypatch,
        driver,
        builder,
        _classical_expectation(ground_vector),
        use_real_unitary_builder=True,
    )

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    final_round = int(np.ceil(np.log2(lambda_norm / epsilon_rpe)))
    final_time = (2**final_round) * np.pi / (2.0 * lambda_norm)
    ladder_bound = np.arcsin(epsilon_unitary) / final_time
    product_bound = (2.0 / np.pi) * epsilon_rpe * np.arcsin(epsilon_unitary)
    energy_error = abs(result.resolved_energy - ground_energy)
    assert result.metadata["num_rounds"] == final_round + 1
    assert result.metadata["unitary_builder"] == "deterministic_or_exact"
    assert result.metadata["epsilon_rpe"] == pytest.approx(epsilon_total)
    assert result.metadata["epsilon_unitary"] == pytest.approx(epsilon_unitary)
    assert result.metadata["error_budget_mode"] == "independent_trotter"
    assert product_bound < epsilon_total
    assert energy_error <= ladder_bound <= epsilon_total


_TWO_QUBIT_PAULIS = ["XX", "ZZ"]
_TWO_QUBIT_COEFFS = [0.25, 0.5]


def _ground_state_problem() -> tuple[QubitOperator, np.ndarray, float]:
    """Return a two-qubit Hamiltonian, ground eigenvector, and ground energy."""
    hamiltonian = QubitOperator(pauli_strings=_TWO_QUBIT_PAULIS, coefficients=_TWO_QUBIT_COEFFS)
    dense = _dense_from_pauli(_TWO_QUBIT_PAULIS, _TWO_QUBIT_COEFFS)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return hamiltonian, eigenvectors[:, 0], float(eigenvalues[0])


def _make_state_prep(state_vector: np.ndarray, num_qubits: int) -> Circuit:
    """Build a Q# state-preparation circuit for a real state vector."""
    from qdk_chemistry.data.circuit import QsharpFactoryData  # noqa: PLC0415
    from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: PLC0415

    params = {
        "rowMap": list(reversed(range(num_qubits))),
        "stateVector": [float(x) for x in np.real(state_vector)],
        "expansionOps": [],
        "numQubits": num_qubits,
    }
    factory = QsharpFactoryData(program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=params)
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(params)
    return Circuit(qsharp_factory=factory, qsharp_op=qsharp_op)


def _registered_driver(
    *,
    target_accuracy: float,
    unitary_builder: AlgorithmRef,
    energy_correction: str,
    seed: int = -1,
    unitary_accuracy_fraction: float | None = None,
) -> RobustPhaseEstimation:
    """Create a registered robust estimator with nested builder and executor configuration."""
    from qdk_chemistry.algorithms import create  # noqa: PLC0415

    builder_ref = AlgorithmRef(
        "qpe_circuit_builder",
        "qdk_robust",
        target_accuracy=target_accuracy,
        energy_correction=energy_correction,
        seed=seed,
        unitary_builder=unitary_builder,
    )
    if unitary_accuracy_fraction is not None:
        builder_ref.set("unitary_accuracy_fraction", unitary_accuracy_fraction)
    return create(
        "phase_estimation",
        "qdk_robust",
        qpe_circuit_builder=builder_ref,
        circuit_executor=AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=7),
    )


@pytest.mark.skipif(not _HAS_QSHARP, reason="requires Q#")
def test_robust_qpe_registered() -> None:
    """The robust driver and circuit builders resolve through the registry."""
    from qdk_chemistry.algorithms import create  # noqa: PLC0415

    assert create("hadamard_test_circuit_builder", "qdk").name() == "qdk"
    assert create("phase_estimation", "qdk_robust").name() == "qdk_robust"
    assert create("qpe_circuit_builder", "qdk_robust").name() == "qdk_robust"


@pytest.mark.skipif(not _HAS_QSHARP, reason="requires Q#")
def test_robust_circuit_builder_direct_pair_supports_qre() -> None:
    """Built circuits support QIR export; an in-memory workload keeps the live Q# preparation."""
    from qdk.qre.application import QIRApplication, QSharpApplication  # noqa: PLC0415

    from qdk_chemistry.algorithms import create  # noqa: PLC0415

    hamiltonian, ground_vector, _ = _ground_state_problem()
    state_preparation = _make_state_prep(ground_vector, num_qubits=2)
    builder = create(
        "qpe_circuit_builder",
        "qdk_robust",
        target_accuracy=1.0,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
        energy_correction="linear",
    )
    circuit_set = builder.build_schedule(hamiltonian)
    _, x_circuit, y_circuit = next(builder.iter_build(circuit_set, state_preparation, hamiltonian))

    assert isinstance(x_circuit.get_qre_application(), QSharpApplication)
    assert isinstance(y_circuit.get_qre_application(), QSharpApplication)

    restored_x = Circuit.from_json(x_circuit.to_json())
    restored_y = Circuit.from_json(y_circuit.to_json())
    assert isinstance(restored_x.get_qre_application(), QIRApplication)
    assert isinstance(restored_y.get_qre_application(), QIRApplication)

    replay = RobustPhaseEstimationCircuitSet(
        schedule=circuit_set, state_preparation=state_preparation, qubit_hamiltonian=hamiltonian
    )
    assert replay.state_preparation is state_preparation
    _, replay_x, replay_y = next(
        builder.iter_build(replay.schedule, replay.state_preparation, replay.qubit_hamiltonian)
    )
    assert isinstance(replay_x.get_qre_application(), QSharpApplication)
    assert isinstance(replay_y.get_qre_application(), QSharpApplication)


@pytest.mark.skipif(not _HAS_QSHARP, reason="requires Q#")
@pytest.mark.parametrize(
    ("family", "dtype"),
    [("trotter", np.float64), ("qdrift", np.float64), ("trotter", np.float32), ("qdrift", np.complex64)],
)
def test_seeded_simulator_replays_full_result_in_memory(
    family: str, dtype: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh execution and replay must agree on the full QPE result without scheduling twice."""
    coefficients = np.array([0.6, 0.4], dtype=dtype)
    hamiltonian = QubitOperator(["ZI", "IZ"], coefficients)
    preparation = _make_state_prep(np.array([1.0, 0.0, 0.0, 0.0]), num_qubits=2)
    driver = _registered_driver(
        target_accuracy=1.0,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", family),
        energy_correction="auto",
        seed=17,
    )
    expected = driver.run(preparation, hamiltonian)
    workload = driver.schedule_circuit_set(preparation, hamiltonian)
    assert (
        expected.metadata["lambda"]
        == workload.schedule.lambda_norm
        == np.sum(np.abs(coefficients.astype(np.complex128)))
    )
    monkeypatch.setattr(
        RobustPhaseEstimationCircuitBuilder, "build_schedule", lambda *_: pytest.fail("Replay rescheduled")
    )
    assert driver.execute_circuit_set(workload).to_json() == expected.to_json()


@pytest.mark.skipif(not _HAS_QSHARP, reason="requires Q#")
def test_robust_qpe_deterministic_control_recovers_gse() -> None:
    """Exact commuting-term evolution and an exact ground state recover the GSE."""
    hamiltonian, ground_vector, ground_energy = _ground_state_problem()
    state_preparation = _make_state_prep(ground_vector, num_qubits=2)
    driver = _registered_driver(
        target_accuracy=1e-3,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "trotter"),
        energy_correction="linear",
    )

    result = driver.run(state_preparation=state_preparation, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(ground_energy, abs=2e-3)


@pytest.mark.skipif(not _HAS_QSHARP, reason="requires Q#")
@pytest.mark.slow
@pytest.mark.parametrize(
    "target_accuracy",
    [
        pytest.param(5e-2, id="ci"),
        pytest.param(
            1e-2,
            id="stress",
            marks=pytest.mark.skipif(
                os.getenv("QDK_CHEMISTRY_RUN_RPE_STRESS_TESTS", "").lower() not in {"1", "true", "yes"},
                reason="Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 and QDK_CHEMISTRY_RUN_RPE_STRESS_TESTS=1 for the full run.",
            ),
        ),
    ],
)
def test_robust_qpe_qdrift_recovers_gse(target_accuracy: float) -> None:
    """End-to-end randomized evolution recovers the GSE within tolerance."""
    hamiltonian, ground_vector, ground_energy = _ground_state_problem()
    trial = ground_vector + 0.1 * np.roll(ground_vector, 1)
    trial = trial / np.linalg.norm(trial)
    state_preparation = _make_state_prep(trial, num_qubits=2)
    driver = _registered_driver(
        target_accuracy=target_accuracy,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "qdrift"),
        energy_correction="qdrift_tangent",
        seed=42,
        unitary_accuracy_fraction=0.0,
    )

    result = driver.run(state_preparation=state_preparation, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(ground_energy, abs=5e-3)
