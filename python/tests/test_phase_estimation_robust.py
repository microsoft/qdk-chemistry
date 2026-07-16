"""Tests for robust phase estimation (qdk_robust).

Two layers:

* Driver-wiring tests inject an ideal signal ``g(t) = e^{-iEt}`` through fake
  nested algorithms, exercising the real ``RobustPhaseEstimation`` loop, sign
  conventions, and energy map without a simulator.
* End-to-end tests run the full qDRIFT + Hadamard-test + simulator stack against
  an exact-diagonalization ground truth; they require Q# and the merged Hadamard
  test, and are skipped otherwise.

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import os
from typing import cast

import numpy as np
import pytest

from qdk_chemistry.algorithms.phase_estimation.robust_phase_estimation import RobustPhaseEstimation
from qdk_chemistry.data import QubitHamiltonian
from qdk_chemistry.utils.rpe import num_rounds, qdrift_schedule

_HAS_QSHARP = importlib.util.find_spec("qdk.qsharp") is not None
_RUN_SLOW_TESTS = os.getenv("QDK_CHEMISTRY_RUN_SLOW_TESTS", "").lower() in {"1", "true", "yes"}
_RANDOMIZED_ACCURACY_MARKS = (
    pytest.mark.slow,
    pytest.mark.skipif(
        not _RUN_SLOW_TESTS,
        reason="Skipping slow randomized accuracy test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
    ),
)

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
    """Return True when the registry can build the robust stack (post-#405 build)."""
    if not _HAS_QSHARP:
        return False
    try:
        from qdk_chemistry.algorithms import create  # noqa: PLC0415

        create("hadamard_test")
        create("phase_estimation", "qdk_robust")
    except (KeyError, RuntimeError, ValueError):
        return False
    return True


# =============================================================================
# Layer 1: driver-wiring tests with an injected ideal signal (no simulator)
# =============================================================================
class _FakeSettings:
    """Minimal stand-in for an algorithm Settings object."""

    def __init__(self, supported_keys: set[str] | None = None) -> None:
        self._values: dict[str, object] = {}
        self._supported_keys = frozenset(supported_keys or ())

    def set(self, key: str, value: object) -> None:
        self._values[key] = value

    def get(self, key: str) -> object:
        return self._values.get(key)

    def has(self, key: str) -> bool:
        return key in self._supported_keys

    def update(self, *args: object) -> None:
        if len(args) == 2:
            self._values[str(args[0])] = args[1]

    def lock(self) -> None:
        pass


class _FakeUnitary:
    def __init__(self, time: float) -> None:
        self.time = time


class _FakeBuilder:
    """Fake time-evolution builder that records the requested evolution time."""

    def __init__(self) -> None:
        self._settings = _FakeSettings({"time", "target_accuracy", "seed"})

    def settings(self) -> _FakeSettings:
        return self._settings

    def run(self, qubit_hamiltonian: object) -> _FakeUnitary:  # noqa: ARG002
        return _FakeUnitary(float(cast("float", self._settings.get("time"))))


class _FakeExecutorData:
    def __init__(self, counts: dict[str, int]) -> None:
        self.bitstring_counts = counts


class _FakeHadamardTest:
    """Fake Hadamard test returning counts for a scaled ideal signal."""

    def __init__(self, energy: float, resolution: int = 2_000_000, signal_factor: complex = 1.0 + 0.0j) -> None:
        self._settings = _FakeSettings({"test_basis"})
        self._energy = energy
        self._resolution = resolution
        self._signal_factor = signal_factor

    def settings(self) -> _FakeSettings:
        return self._settings

    def run(self, state_preparation: object, unitary: _FakeUnitary, shots: int) -> _FakeExecutorData:  # noqa: ARG002
        signal = self._signal_factor * np.exp(-1j * self._energy * unitary.time)
        basis = self._settings.get("test_basis")
        expectation = signal.real if basis == "X" else signal.imag
        n0 = round((1.0 + expectation) / 2.0 * self._resolution)
        n1 = self._resolution - n0
        return _FakeExecutorData({"0": int(n0), "1": int(n1)})


def _patch_with_ideal_signal(
    monkeypatch: pytest.MonkeyPatch,
    driver: RobustPhaseEstimation,
    energy: float,
    signal_factor: complex = 1.0 + 0.0j,
) -> None:
    """Replace the driver's nested-algorithm factory with ideal-signal fakes."""

    def fake_create_nested(setting_key: str):
        if setting_key == "unitary_builder":
            return _FakeBuilder()
        if setting_key == "hadamard_test":
            return _FakeHadamardTest(energy, signal_factor=signal_factor)
        raise KeyError(setting_key)

    monkeypatch.setattr(driver, "_create_nested", fake_create_nested)


@pytest.mark.parametrize("energy", [0.4, -0.3, 0.75, 0.0])
def test_driver_recovers_energy_exact_mode(monkeypatch: pytest.MonkeyPatch, energy: float) -> None:
    """energy_correction='linear': the RPE loop + linear energy map recover the injected energy."""
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    driver = RobustPhaseEstimation(target_accuracy=1e-4, energy_correction="linear")
    _patch_with_ideal_signal(monkeypatch, driver, energy)

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)
    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)


@pytest.mark.parametrize("energy", [0.4, -0.3, 0.75])
def test_driver_recovers_energy_qdrift_mode(monkeypatch: pytest.MonkeyPatch, energy: float) -> None:
    """energy_correction='qdrift_tangent': the tangent de-biasing leaves an ideal signal unchanged."""
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    driver = RobustPhaseEstimation(target_accuracy=1e-4, energy_correction="qdrift_tangent")
    _patch_with_ideal_signal(monkeypatch, driver, energy)

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)
    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)


def test_driver_uses_explicit_base_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user-provided base_time is honoured and still recovers the energy."""
    energy = 0.6
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    driver = RobustPhaseEstimation(target_accuracy=1e-4, base_time=np.pi / 4, energy_correction="linear")
    _patch_with_ideal_signal(monkeypatch, driver, energy)

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)
    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)
    assert result.metadata["base_time"] == pytest.approx(np.pi / 4)


def test_robust_phase_estimation_name() -> None:
    assert RobustPhaseEstimation().name() == "qdk_robust"


def test_energy_correction_auto_selection() -> None:
    """'auto' maps qDRIFT to the tangent map and every other family to linear."""
    auto = RobustPhaseEstimation()
    assert auto._select_correction("qdrift") == "qdrift_tangent"
    assert auto._select_correction("partial_randomized") == "linear"
    assert auto._select_correction("deterministic_or_exact") == "linear"
    # An explicit mode overrides the inference.
    forced = RobustPhaseEstimation(energy_correction="qdrift_tangent")
    assert forced._select_correction("partial_randomized") == "qdrift_tangent"


def test_product_budget_meets_target_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit RPE and unitary tolerances yield the requested deterministic bound."""
    epsilon_total = 0.1
    epsilon_unitary = 0.5
    epsilon_rpe = np.pi * epsilon_total / (2.0 * np.arcsin(epsilon_unitary))
    energy = 0.3
    phase_error = np.arcsin(epsilon_unitary)
    signal_factor = np.sqrt(1.0 - epsilon_unitary**2) * np.exp(1j * phase_error)

    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    driver = RobustPhaseEstimation(
        target_accuracy=epsilon_total,
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
        energy_correction="linear",
    )
    _patch_with_ideal_signal(monkeypatch, driver, energy, signal_factor)

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)
    metadata = result.metadata
    lambda_norm = 1.0  # |0.5| + |0.5|
    final_round = num_rounds(lambda_norm, epsilon_rpe)
    base_time = np.pi / (2.0 * lambda_norm)
    final_time = (2**final_round) * base_time
    exact_energy_bound = phase_error / final_time
    propagated_energy_bound = (2.0 / np.pi) * epsilon_rpe * phase_error

    assert abs(signal_factor - 1.0) == pytest.approx(epsilon_unitary)
    assert metadata["epsilon_unitary"] == pytest.approx(epsilon_unitary)
    assert metadata["epsilon_rpe"] == pytest.approx(epsilon_rpe)
    assert metadata["error_budget_mode"] == "explicit"
    assert metadata["energy_correction"] == "linear"
    assert metadata["unitary_builder"] == "deterministic_or_exact"
    # Round count is sized from epsilon_rpe, not the full target_accuracy.
    assert metadata["num_rounds"] == final_round + 1
    assert abs(result.resolved_energy - energy) == pytest.approx(exact_energy_bound, rel=1e-3)
    assert propagated_energy_bound == pytest.approx(epsilon_total)
    assert exact_energy_bound <= propagated_energy_bound


def test_partial_builder_receives_unitary_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partially-randomized builder gets target_accuracy=epsilon_unitary per round (no num_samples)."""
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    driver = RobustPhaseEstimation(
        target_accuracy=1e-2, unitary_accuracy_fraction=0.5, energy_correction="linear", seed=5
    )
    records: list[dict[str, object]] = []

    class _RecordingPartialBuilder:
        """Fake builder that classifies as partial and records each round's settings."""

        def __init__(self) -> None:
            self._settings = _FakeSettings({"time", "target_accuracy", "seed", "num_random_samples"})

        def settings(self) -> _FakeSettings:
            return self._settings

        def run(self, qubit_hamiltonian: object) -> _FakeUnitary:  # noqa: ARG002
            records.append(dict(self._settings._values))
            return _FakeUnitary(float(cast("float", self._settings.get("time"))))

    def fake_create_nested(setting_key: str):
        if setting_key == "unitary_builder":
            return _RecordingPartialBuilder()
        if setting_key == "hadamard_test":
            return _FakeHadamardTest(0.3)
        raise KeyError(setting_key)

    monkeypatch.setattr(driver, "_create_nested", fake_create_nested)
    driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)

    assert records, "unitary builder was never run"
    eps_unitary = 0.5e-2
    tau = float(np.pi / (2 * 1.0))
    total = num_rounds(1.0, eps_unitary)  # eps_rpe == eps_unitary here (fraction 0.5, target 1e-2)
    # Every build carries the per-round unitary budget and never the qDRIFT knob.
    for record in records:
        assert record["target_accuracy"] == pytest.approx(eps_unitary)
        assert "num_samples" not in record  # partial path must not use the qDRIFT sample knob
    # Fresh-draw-per-shot: each round issues `shots` independent builds at that
    # round's evolution time, every one seeded independently.
    builds_by_time: dict[float, list[dict[str, object]]] = {}
    for record in records:
        builds_by_time.setdefault(float(cast("float", record["time"])), []).append(record)
    for round_index in range(total + 1):
        round_time = (2**round_index) * tau
        round_builds = builds_by_time.get(round_time)
        assert round_builds is not None, f"round {round_index} issued no builds"
        expected_shots, _ = qdrift_schedule(total, round_index)
        assert len(round_builds) == expected_shots  # one fresh draw per shot
        seeds = [record["seed"] for record in round_builds]
        assert len(set(seeds)) == expected_shots  # every draw seeded independently


def test_sample_signal_randomized_averages_independent_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    """Randomized rounds draw a fresh circuit per shot and average the per-draw signals."""
    driver = RobustPhaseEstimation(seed=7)
    build_seeds: list[int] = []

    def fake_build(qh, t, samples, seed, m, category, eps, *, explicit_seed=False):  # noqa: ARG001
        assert explicit_seed is True  # per-draw seed used verbatim
        build_seeds.append(seed)
        return ("unitary", seed)

    def fake_sample(state, unitary, shots, basis):  # noqa: ARG001
        assert shots == 1  # one measurement per fresh draw
        value = (unitary[1] % 100) / 100.0
        return value if basis == "X" else -value

    monkeypatch.setattr(driver, "_build_unitary", fake_build)
    monkeypatch.setattr(driver, "_sample_signal", fake_sample)

    shots = 6
    real_part, imag_part = driver._sample_signal_randomized(object(), object(), 1.5, 8, shots, 7, 3, "qdrift", 0.0)

    assert len(build_seeds) == shots  # one fresh draw per shot
    assert len(set(build_seeds)) == shots  # every draw is independent
    expected_real = float(np.mean([(s % 100) / 100.0 for s in build_seeds]))
    assert real_part == pytest.approx(expected_real)  # averaged, not a single draw
    assert imag_part == pytest.approx(-expected_real)  # X and Y share each draw


def test_sample_signal_randomized_nondeterministic_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """With seed < 0 the round still issues one fresh (RNG-seeded) draw per shot."""
    driver = RobustPhaseEstimation(seed=-1)
    build_seeds: list[int] = []

    def fake_build(qh, t, samples, seed, m, category, eps, *, explicit_seed=False):  # noqa: ARG001
        build_seeds.append(seed)
        return ("unitary", 0)

    def fake_sample(state, unitary, shots, basis):  # noqa: ARG001
        return 0.0

    monkeypatch.setattr(driver, "_build_unitary", fake_build)
    monkeypatch.setattr(driver, "_sample_signal", fake_sample)

    driver._sample_signal_randomized(object(), object(), 1.0, 8, 4, -1, 0, "qdrift", 0.0)

    assert len(build_seeds) == 4  # still draws once per shot
    assert all(seed == -1 for seed in build_seeds)  # non-deterministic RNG per build


# =============================================================================
# Layer 1b: precision contract via a noiseless classical signal (no Q#)
# =============================================================================
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


def _noncommuting_ground_state_problem() -> tuple[QubitHamiltonian, np.ndarray, float]:
    """Return (H, ground eigenvector, ground energy) for a non-commuting 2-qubit H.

    ``ZI`` and ``XI`` anti-commute, so the deterministic Trotter part has genuine
    error to resolve; with ``weight_threshold=0.5`` the partially randomized split
    is ``H_D = {ZI, XI, IZ}`` and ``H_R = {ZZ}`` (lambda_R = 0.3 << lambda = 2.6).
    """
    hamiltonian = QubitHamiltonian(pauli_strings=_NONCOMMUTING_PAULIS, coefficients=_NONCOMMUTING_COEFFS)
    dense = _dense_from_pauli(_NONCOMMUTING_PAULIS, _NONCOMMUTING_COEFFS)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return hamiltonian, eigenvectors[:, 0], float(eigenvalues[0])


def _h2_sto3g_ground_state_problem() -> tuple[QubitHamiltonian, np.ndarray, float]:
    """Return the repository's 4-qubit H2/STO-3G Hamiltonian and exact ground state."""
    hamiltonian = QubitHamiltonian(pauli_strings=_H2_STO3G_PAULIS, coefficients=_H2_STO3G_COEFFS)
    dense = np.asarray(hamiltonian.to_matrix(sparse=False), dtype=complex)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return hamiltonian, eigenvectors[:, 0], float(eigenvalues[0])


def _materialize_container(container) -> np.ndarray:
    """Materialize a ``PauliProductFormulaContainer`` as a dense unitary matrix.

    Applies the product of exponentiated Pauli terms once and raises it to the
    container's ``step_reps`` power. Each ``e^{-i*angle*P}`` uses the closed form
    ``cos(angle) I - i sin(angle) P`` (valid because every Pauli string squares to
    the identity), so no matrix exponential is required.
    """
    num_qubits = container.num_qubits
    dim = 2**num_qubits
    identity = np.eye(dim, dtype=complex)
    step = identity.copy()
    for term in container.step_terms:
        labels = ["I"] * num_qubits
        for qubit, op in term.pauli_term.items():
            labels[num_qubits - 1 - qubit] = op  # little-endian: qubit 0 is rightmost
        pauli = _dense_from_pauli(["".join(labels)], [1.0])
        angle = term.angle
        step = (np.cos(angle) * identity - 1j * np.sin(angle) * pauli) @ step
    return np.linalg.matrix_power(step, container.step_reps)


def _classical_signal_sampler(ground_vector: np.ndarray):
    """Return a noiseless replacement for ``RobustPhaseEstimation._sample_signal``.

    It materializes the real per-round unitary and returns the exact Hadamard-test
    expectation (X basis -> ``Re<psi|U|psi>``, Y basis -> ``Im<psi|U|psi>``),
    removing shot noise so the test isolates the algorithm's systematic and
    single-draw error rather than statistical fluctuation.
    """

    def _sample(state_preparation: object, unitary, shots: int, test_basis: str) -> float:  # noqa: ARG001
        dense_unitary = _materialize_container(unitary.get_container())
        signal = complex(ground_vector.conj() @ (dense_unitary @ ground_vector))
        return float(signal.real) if test_basis == "X" else float(signal.imag)

    return _sample


@pytest.mark.parametrize(
    ("builder_name", "builder_kwargs", "expected_category", "expected_correction"),
    [
        ("trotter", {"order": 2}, "deterministic_or_exact", "linear"),
        pytest.param(
            "qdrift",
            {},
            "qdrift",
            "qdrift_tangent",
            marks=_RANDOMIZED_ACCURACY_MARKS,
        ),
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
    """All three builders recover the GSE within ``target_accuracy`` (the public contract).

    A noiseless classical Hadamard signal (each per-round unitary materialized
    densely) replaces the Q# simulator, so the test runs without Q# and isolates the
    algorithm's error from shot noise. For the randomized builders every one of a
    round's shots draws a fresh circuit, so the classical signal is averaged over
    independent draws (the faithful expected signal), exactly as on hardware. The
    seed is fixed; because qDRIFT and partially randomized are stochastic, the
    assertion is the public contract ``abs <= epsilon`` rather than a tighter
    per-seed value.

    Measured errors at this seed (epsilon=0.1): trotter ~3.7e-4, qDRIFT ~7e-3,
    partial ~2e-4. Averaging a fresh draw per shot makes the qDRIFT error the
    shot-averaged (expected-signal) error; the earlier frozen single-draw path was
    both larger (~1.2e-2 typical) and prone to rare catastrophic phase-wrap
    failures on high-significance rounds (max ~5 over 15 seeds).
    """
    from qdk_chemistry.data import AlgorithmRef  # noqa: PLC0415

    epsilon = 0.1
    hamiltonian, ground_vector, ground_energy = _noncommuting_ground_state_problem()
    driver = RobustPhaseEstimation(
        target_accuracy=epsilon, unitary_accuracy_fraction=0.5, energy_correction="auto", seed=7
    )
    driver.settings().set(
        "unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", builder_name, **builder_kwargs)
    )
    monkeypatch.setattr(driver, "_sample_signal", _classical_signal_sampler(ground_vector))

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)

    assert result.resolved_energy == pytest.approx(ground_energy, abs=epsilon)
    assert result.metadata["unitary_builder"] == expected_category
    assert result.metadata["energy_correction"] == expected_correction


@pytest.mark.parametrize(
    ("epsilon_total", "epsilon_unitary"),
    [(0.1, 0.5), (1e-3, 0.5)],
    ids=["tenth-hartree", "one-millihartree"],
)
def test_product_budget_bounds_noncommuting_trotter_ground_energy(
    monkeypatch: pytest.MonkeyPatch, epsilon_total: float, epsilon_unitary: float
) -> None:
    """Product budgets bound real order-2 Trotter ground-energy estimates."""
    from qdk_chemistry.data import AlgorithmRef  # noqa: PLC0415

    epsilon_rpe = np.pi * epsilon_total / (2.0 * np.arcsin(epsilon_unitary))
    hamiltonian, ground_vector, ground_energy = _noncommuting_ground_state_problem()
    lambda_norm = float(np.sum(np.abs(_NONCOMMUTING_COEFFS)))

    driver = RobustPhaseEstimation(
        target_accuracy=epsilon_total,
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
        energy_correction="linear",
    )
    driver.settings().set("unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter", order=2))
    monkeypatch.setattr(driver, "_sample_signal", _classical_signal_sampler(ground_vector))

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)

    final_round = num_rounds(lambda_norm, epsilon_rpe)
    base_time = np.pi / (2.0 * lambda_norm)
    final_time = (2**final_round) * base_time
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
    from qdk_chemistry.data import AlgorithmRef  # noqa: PLC0415

    epsilon_total = 1e-3
    epsilon_unitary = 0.5
    epsilon_rpe = np.pi * epsilon_total / (2.0 * np.arcsin(epsilon_unitary))
    hamiltonian, ground_vector, ground_energy = _h2_sto3g_ground_state_problem()
    lambda_norm = float(np.sum(np.abs(_H2_STO3G_COEFFS)))

    driver = RobustPhaseEstimation(
        target_accuracy=epsilon_total,
        epsilon_rpe=epsilon_rpe,
        epsilon_unitary=epsilon_unitary,
        energy_correction="linear",
    )
    driver.settings().set("unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter", order=2))
    monkeypatch.setattr(driver, "_sample_signal", _classical_signal_sampler(ground_vector))

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)

    final_round = num_rounds(lambda_norm, epsilon_rpe)
    base_time = np.pi / (2.0 * lambda_norm)
    final_time = (2**final_round) * base_time
    ladder_bound = np.arcsin(epsilon_unitary) / final_time
    product_bound = (2.0 / np.pi) * epsilon_rpe * np.arcsin(epsilon_unitary)
    energy_error = abs(result.resolved_energy - ground_energy)

    assert result.metadata["num_rounds"] == final_round + 1
    assert result.metadata["unitary_builder"] == "deterministic_or_exact"
    assert result.metadata["error_budget_mode"] == "explicit"
    assert product_bound == pytest.approx(epsilon_total)
    assert energy_error <= ladder_bound <= epsilon_total


# =============================================================================
# Layer 2: end-to-end against exact diagonalization (requires Q# + #405 build)
# =============================================================================
_TWO_QUBIT_PAULIS = ["XX", "ZZ"]
_TWO_QUBIT_COEFFS = [0.25, 0.5]


def _ground_state_problem() -> tuple[QubitHamiltonian, np.ndarray, float]:
    """Return (Hamiltonian, ground eigenvector, ground energy) for the 2-qubit benchmark."""
    hamiltonian = QubitHamiltonian(pauli_strings=_TWO_QUBIT_PAULIS, coefficients=_TWO_QUBIT_COEFFS)
    dense = _dense_from_pauli(_TWO_QUBIT_PAULIS, _TWO_QUBIT_COEFFS)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    ground_energy = float(eigenvalues[0])
    ground_vector = eigenvectors[:, 0]
    return hamiltonian, ground_vector, ground_energy


def _make_state_prep(state_vector: np.ndarray, num_qubits: int):
    """Build a Q# state-preparation circuit for the given (real) state vector."""
    from qdk_chemistry.data import Circuit  # noqa: PLC0415
    from qdk_chemistry.data.circuit import QsharpFactoryData  # noqa: PLC0415
    from qdk_chemistry.utils.qsharp import QSHARP_UTILS  # noqa: PLC0415

    params = {
        "rowMap": list(reversed(range(num_qubits))),
        "stateVector": [float(x) for x in np.real(state_vector)],
        "expansionOps": [],
        "numQubits": num_qubits,
    }
    factories = QsharpFactoryData(program=QSHARP_UTILS.StatePreparation.MakeStatePreparationCircuit, parameter=params)
    qsharp_op = QSHARP_UTILS.StatePreparation.MakeStatePreparationOp(params)
    return Circuit(qsharp_factory=factories, qsharp_op=qsharp_op)


@pytest.mark.skipif(not _has_robust_stack(), reason="requires Q# and the merged Hadamard test (#405)")
def test_robust_qpe_registered() -> None:
    """The driver resolves through the registry under the name qdk_robust."""
    from qdk_chemistry.algorithms import create  # noqa: PLC0415

    driver = create("phase_estimation", "qdk_robust")
    assert driver.name() == "qdk_robust"


@pytest.mark.skipif(not _has_robust_stack(), reason="requires Q# and the merged Hadamard test (#405)")
def test_robust_qpe_deterministic_control_recovers_gse() -> None:
    """Exact (commuting-term Trotter) evolution + exact ground state recovers the GSE."""
    from qdk_chemistry.algorithms import create  # noqa: PLC0415
    from qdk_chemistry.data import AlgorithmRef  # noqa: PLC0415

    hamiltonian, ground_vector, ground_energy = _ground_state_problem()
    state_prep = _make_state_prep(ground_vector, num_qubits=2)

    driver = create(
        "phase_estimation",
        "qdk_robust",
        target_accuracy=1e-3,
        unitary_accuracy_fraction=0.0,
        energy_correction="linear",
    )
    # XX and ZZ commute, so Trotter is exact for this Hamiltonian.
    driver.settings().set("unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter"))
    driver.settings().set(
        "hadamard_test",
        AlgorithmRef(
            "hadamard_test",
            "qdk",
            circuit_executor=AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=7),
        ),
    )

    result = driver.run(state_preparation=state_prep, qubit_hamiltonian=hamiltonian)
    # Measured abs error ~9e-5 for exact evolution; tolerance keeps margin and stays sub-chemical-accuracy.
    assert result.resolved_energy == pytest.approx(ground_energy, abs=2e-3)


@pytest.mark.skipif(not _has_robust_stack(), reason="requires Q# and the merged Hadamard test (#405)")
@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_SLOW_TESTS,
    reason="Skipping slow Q# qDRIFT integration test. Set QDK_CHEMISTRY_RUN_SLOW_TESTS=1 to enable.",
)
def test_robust_qpe_qdrift_recovers_gse() -> None:
    """End-to-end qDRIFT run with a high-overlap trial state recovers the GSE within tolerance.

    Calibrated from a measured run at this seed: abs error ~7e-4 (overlap ~0.99),
    well within chemical accuracy.
    """
    from qdk_chemistry.algorithms import create  # noqa: PLC0415
    from qdk_chemistry.data import AlgorithmRef  # noqa: PLC0415

    hamiltonian, ground_vector, ground_energy = _ground_state_problem()
    # Perturb the exact eigenvector to model an imperfect (high-overlap) trial state.
    trial = ground_vector + 0.1 * np.roll(ground_vector, 1)
    trial = trial / np.linalg.norm(trial)
    state_prep = _make_state_prep(trial, num_qubits=2)

    driver = create(
        "phase_estimation",
        "qdk_robust",
        target_accuracy=1e-2,
        unitary_accuracy_fraction=0.0,
        energy_correction="qdrift_tangent",
        seed=42,
    )
    driver.settings().set("unitary_builder", AlgorithmRef("hamiltonian_unitary_builder", "qdrift"))
    driver.settings().set(
        "hadamard_test",
        AlgorithmRef(
            "hadamard_test",
            "qdk",
            circuit_executor=AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=7),
        ),
    )

    result = driver.run(state_preparation=state_prep, qubit_hamiltonian=hamiltonian)
    assert result.resolved_energy == pytest.approx(ground_energy, abs=5e-3)
