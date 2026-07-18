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

import importlib.util
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qdk_chemistry.algorithms.phase_estimation.circuit_builder.robust_builder import (
    QdkRobustPhaseEstimationCircuitBuilder,
    _AlgorithmSnapshot,
)
from qdk_chemistry.algorithms.phase_estimation.robust_phase_estimation import RobustPhaseEstimation
from qdk_chemistry.data import AlgorithmRef, Circuit, QubitOperator, Settings, UnitaryRepresentation
from qdk_chemistry.utils.rpe import num_rounds

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


def _has_robust_stack() -> bool:
    """Return True when the registry can build the robust circuit stack."""
    if not _HAS_QSHARP:
        return False
    try:
        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        create("hadamard_test_circuit_builder", "qdk")
        create("robust_phase_estimation_circuit_builder", "qdk")
        create("phase_estimation", "qdk_robust")
    except (KeyError, RuntimeError, ValueError):
        return False
    return True


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

    def run(self, circuit: Circuit, *, shots: int) -> _FakeExecutorData:
        """Convert the configured expectation into deterministic counts."""
        self.shot_calls.append(shots)
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
    unitary_accuracy_fraction: float = 0.5,
    energy_correction: str = "auto",
    seed: int = 7,
    epsilon_rpe: float = 0.0,
    epsilon_unitary: float = 0.0,
) -> QdkRobustPhaseEstimationCircuitBuilder:
    """Create a directly configurable robust circuit builder for tests."""
    return QdkRobustPhaseEstimationCircuitBuilder(
        target_accuracy=target_accuracy,
        base_time=base_time,
        unitary_accuracy_fraction=unitary_accuracy_fraction,
        energy_correction=energy_correction,
        seed=seed,
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
        unitary_builder=AlgorithmRef(
            "hamiltonian_unitary_builder",
            unitary_builder_name,
            **(unitary_builder_kwargs or {}),
        ),
    )


def _install_test_stack(
    monkeypatch: pytest.MonkeyPatch,
    driver: RobustPhaseEstimation,
    circuit_builder: QdkRobustPhaseEstimationCircuitBuilder,
    expectation: Callable[[object, str], float],
    *,
    use_real_unitary_builder: bool = False,
) -> tuple[list[dict[str, object]], _FakeExecutor]:
    """Install fake circuit/execution boundaries around the real RPE orchestration."""
    contexts: dict[int, tuple[str, object]] = {}
    unitary_records: list[dict[str, object]] = []
    executor = _FakeExecutor(contexts, expectation)
    original_create = _AlgorithmSnapshot.create

    def create_snapshot(snapshot: _AlgorithmSnapshot):
        settings = Settings.from_json(snapshot.settings_json)
        if snapshot.algorithm_type == "hamiltonian_unitary_builder":
            if use_real_unitary_builder:
                return original_create(snapshot)
            return _FakeUnitaryBuilder(settings, unitary_records)
        if snapshot.algorithm_type == "hadamard_test_circuit_builder":
            return _FakeHadamardCircuitBuilder(settings, contexts)
        raise AssertionError(f"Unexpected algorithm type: {snapshot.algorithm_type}")

    def create_nested(setting_key: str):
        if setting_key == "robust_phase_estimation_circuit_builder":
            return circuit_builder
        if setting_key == "circuit_executor":
            return executor
        raise KeyError(setting_key)

    monkeypatch.setattr(_AlgorithmSnapshot, "create", create_snapshot)
    monkeypatch.setattr(driver, "_create_nested", create_nested)
    return unitary_records, executor


def _ideal_expectation(energy: float, signal_factor: complex = 1.0 + 0.0j) -> Callable[[object, str], float]:
    """Return an ideal Hadamard expectation function for a known eigenenergy."""

    def expectation(unitary: object, basis: str) -> float:
        assert isinstance(unitary, _FakeUnitary)
        signal = signal_factor * np.exp(-1j * energy * unitary.time)
        return float(signal.real) if basis == "X" else float(signal.imag)

    return expectation


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
    )
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(energy))

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)


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


def test_robust_phase_estimation_name() -> None:
    """The robust estimator retains its registered name."""
    assert RobustPhaseEstimation().name() == "qdk_robust"


def test_energy_correction_auto_selection() -> None:
    """Auto correction maps only pure randomized-product evolution to the tangent map."""
    auto = QdkRobustPhaseEstimationCircuitBuilder()
    assert auto._select_correction("qdrift") == "qdrift_tangent"
    assert auto._select_correction("partial_randomized") == "linear"
    assert auto._select_correction("deterministic_or_exact") == "linear"
    forced = QdkRobustPhaseEstimationCircuitBuilder(energy_correction="qdrift_tangent")
    assert forced._select_correction("partial_randomized") == "qdrift_tangent"


def test_product_budget_meets_target_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit RPE and unitary tolerances yield the requested deterministic bound."""
    epsilon_total = 0.1
    epsilon_unitary = 0.5
    epsilon_rpe = np.pi * epsilon_total / (2.0 * np.arcsin(epsilon_unitary))
    energy = 0.3
    phase_error = np.arcsin(epsilon_unitary)
    signal_factor = np.sqrt(1.0 - epsilon_unitary**2) * np.exp(1j * phase_error)
    hamiltonian = QubitOperator(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    builder = _make_builder(
        target_accuracy=epsilon_total,
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
        energy_correction="linear",
    )
    driver = RobustPhaseEstimation()
    _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(energy, signal_factor))

    result = driver.run(state_preparation=_DUMMY_STATE_PREPARATION, qubit_hamiltonian=hamiltonian)

    metadata = result.metadata
    final_round = num_rounds(1.0, epsilon_rpe)
    final_time = (2**final_round) * np.pi / 2.0
    exact_energy_bound = phase_error / final_time
    propagated_energy_bound = (2.0 / np.pi) * epsilon_rpe * phase_error
    assert abs(signal_factor - 1.0) == pytest.approx(epsilon_unitary)
    assert metadata["epsilon_unitary"] == pytest.approx(epsilon_unitary)
    assert metadata["epsilon_rpe"] == pytest.approx(epsilon_rpe)
    assert metadata["error_budget_mode"] == "explicit"
    assert metadata["energy_correction"] == "linear"
    assert metadata["unitary_builder"] == "deterministic_or_exact"
    assert metadata["num_rounds"] == final_round + 1
    assert abs(result.resolved_energy - energy) == pytest.approx(exact_energy_bound, rel=1e-3)
    assert propagated_energy_bound == pytest.approx(epsilon_total)
    assert exact_energy_bound <= propagated_energy_bound


def test_partial_builder_receives_unitary_budget() -> None:
    """Every partially randomized round exposes the unitary target accuracy and independent seeds."""
    hamiltonian = QubitOperator(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    epsilon_total = 1e-2
    builder = _make_builder(
        target_accuracy=epsilon_total,
        unitary_builder_name="partially_randomized",
        energy_correction="linear",
        seed=5,
    )

    circuit_set = builder.run(_DUMMY_STATE_PREPARATION, hamiltonian)

    epsilon_unitary = 0.5 * epsilon_total
    for round_data in circuit_set.rounds:
        ref = round_data.unitary_builder_configuration
        assert ref.settings is not None
        assert ref.settings.get("target_accuracy") == pytest.approx(epsilon_unitary)
        assert not ref.settings.has("num_samples")
        assert round_data.num_draws == round_data.shots_per_basis
        assert len(set(round_data.draw_seeds)) == round_data.shots_per_basis


@pytest.mark.parametrize(("unitary_name", "randomized"), [("trotter", False), ("qdrift", True)])
def test_executor_uses_declared_circuit_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
    unitary_name: str,
    randomized: bool,
) -> None:
    """Execution honors deterministic shot multiplicity and randomized one-shot draws."""
    hamiltonian = QubitOperator(pauli_strings=["Z"], coefficients=[1.0])
    builder = _make_builder(target_accuracy=0.5, unitary_builder_name=unitary_name)
    driver = RobustPhaseEstimation()
    _, executor = _install_test_stack(monkeypatch, driver, builder, _ideal_expectation(0.2))

    circuit_set = driver.build_circuit_set(_DUMMY_STATE_PREPARATION, hamiltonian)
    driver.execute_circuit_set(circuit_set)

    if randomized:
        expected = [1 for round_data in circuit_set.rounds for _ in range(2 * round_data.num_draws)]
    else:
        expected = [shots for round_data in circuit_set.rounds for shots in (round_data.shots_per_basis,) * 2]
    assert executor.shot_calls == expected


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
    ("epsilon_total", "epsilon_unitary"),
    [(0.1, 0.5), (1e-3, 0.5)],
    ids=["tenth-hartree", "one-millihartree"],
)
def test_product_budget_bounds_noncommuting_trotter_ground_energy(
    monkeypatch: pytest.MonkeyPatch,
    epsilon_total: float,
    epsilon_unitary: float,
) -> None:
    """Product budgets bound real order-two Trotter ground-energy estimates."""
    epsilon_rpe = np.pi * epsilon_total / (2.0 * np.arcsin(epsilon_unitary))
    hamiltonian, ground_vector, ground_energy = _noncommuting_ground_state_problem()
    lambda_norm = float(np.sum(np.abs(_NONCOMMUTING_COEFFS)))
    builder = _make_builder(
        target_accuracy=epsilon_total,
        unitary_builder_name="trotter",
        unitary_builder_kwargs={"order": 2},
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
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

    final_round = num_rounds(lambda_norm, epsilon_rpe)
    final_time = (2**final_round) * np.pi / (2.0 * lambda_norm)
    ladder_bound = np.arcsin(epsilon_unitary) / final_time
    product_bound = (2.0 / np.pi) * epsilon_rpe * np.arcsin(epsilon_unitary)
    energy_error = abs(result.resolved_energy - ground_energy)
    assert result.metadata["num_rounds"] == final_round + 1
    assert result.metadata["unitary_builder"] == "deterministic_or_exact"
    assert result.metadata["error_budget_mode"] == "explicit"
    assert product_bound == pytest.approx(epsilon_total)
    assert energy_error <= ladder_bound <= epsilon_total


def test_product_budget_reaches_one_millihartree_for_h2_sto3g(monkeypatch: pytest.MonkeyPatch) -> None:
    """The explicit product budget reaches one millihartree on H2/STO-3G."""
    epsilon_total = 1e-3
    epsilon_unitary = 0.5
    epsilon_rpe = np.pi * epsilon_total / (2.0 * np.arcsin(epsilon_unitary))
    hamiltonian, ground_vector, ground_energy = _h2_sto3g_ground_state_problem()
    lambda_norm = float(np.sum(np.abs(_H2_STO3G_COEFFS)))
    builder = _make_builder(
        target_accuracy=epsilon_total,
        unitary_builder_name="trotter",
        unitary_builder_kwargs={"order": 2},
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
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

    final_round = num_rounds(lambda_norm, epsilon_rpe)
    final_time = (2**final_round) * np.pi / (2.0 * lambda_norm)
    ladder_bound = np.arcsin(epsilon_unitary) / final_time
    product_bound = (2.0 / np.pi) * epsilon_rpe * np.arcsin(epsilon_unitary)
    energy_error = abs(result.resolved_energy - ground_energy)
    assert result.metadata["num_rounds"] == final_round + 1
    assert result.metadata["unitary_builder"] == "deterministic_or_exact"
    assert result.metadata["error_budget_mode"] == "explicit"
    assert product_bound == pytest.approx(epsilon_total)
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
) -> RobustPhaseEstimation:
    """Create a registered robust estimator with nested builder and executor configuration."""
    from qdk_chemistry.algorithms import create  # noqa: PLC0415

    builder_ref = AlgorithmRef(
        "robust_phase_estimation_circuit_builder",
        "qdk",
        target_accuracy=target_accuracy,
        unitary_accuracy_fraction=0.0,
        energy_correction=energy_correction,
        seed=seed,
        unitary_builder=unitary_builder,
    )
    return create(
        "phase_estimation",
        "qdk_robust",
        robust_phase_estimation_circuit_builder=builder_ref,
        circuit_executor=AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=7),
    )


@pytest.mark.skipif(not _has_robust_stack(), reason="requires Q# and the registered robust circuit stack")
def test_robust_qpe_registered() -> None:
    """The driver and its dedicated circuit builder resolve through the registry."""
    from qdk_chemistry.algorithms import create  # noqa: PLC0415

    assert create("phase_estimation", "qdk_robust").name() == "qdk_robust"
    assert create("robust_phase_estimation_circuit_builder", "qdk").name() == "qdk"


@pytest.mark.skipif(not _has_robust_stack(), reason="requires Q# and the registered robust circuit stack")
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


@pytest.mark.skipif(not _has_robust_stack(), reason="requires Q# and the registered robust circuit stack")
@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_SLOW_TESTS,
    reason="Skipping slow Q# randomized integration test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
)
def test_robust_qpe_qdrift_recovers_gse() -> None:
    """End-to-end randomized evolution recovers the GSE within tolerance."""
    hamiltonian, ground_vector, ground_energy = _ground_state_problem()
    trial = ground_vector + 0.1 * np.roll(ground_vector, 1)
    trial = trial / np.linalg.norm(trial)
    state_preparation = _make_state_prep(trial, num_qubits=2)
    driver = _registered_driver(
        target_accuracy=1e-2,
        unitary_builder=AlgorithmRef("hamiltonian_unitary_builder", "qdrift"),
        energy_correction="qdrift_tangent",
        seed=42,
    )

    result = driver.run(state_preparation=state_preparation, qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(ground_energy, abs=5e-3)
