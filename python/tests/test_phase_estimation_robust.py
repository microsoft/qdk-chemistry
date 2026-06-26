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

import numpy as np
import pytest

from qdk_chemistry.algorithms.phase_estimation.robust_phase_estimation import RobustPhaseEstimation
from qdk_chemistry.data import QubitHamiltonian

_HAS_QSHARP = importlib.util.find_spec("qdk.qsharp") is not None

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

    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        self._values[key] = value

    def get(self, key: str) -> object:
        return self._values.get(key)

    def has(self, key: str) -> bool:  # noqa: ARG002 - fakes accept any setting
        return True

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
        self._settings = _FakeSettings()

    def settings(self) -> _FakeSettings:
        return self._settings

    def run(self, qubit_hamiltonian: object) -> _FakeUnitary:  # noqa: ARG002
        return _FakeUnitary(float(self._settings.get("time")))


class _FakeExecutorData:
    def __init__(self, counts: dict[str, int]) -> None:
        self.bitstring_counts = counts


class _FakeHadamardTest:
    """Fake Hadamard test returning counts for the ideal signal g(t) = e^{-iEt}."""

    def __init__(self, energy: float, resolution: int = 2_000_000) -> None:
        self._settings = _FakeSettings()
        self._energy = energy
        self._resolution = resolution

    def settings(self) -> _FakeSettings:
        return self._settings

    def run(self, state_preparation: object, unitary: _FakeUnitary, shots: int) -> _FakeExecutorData:  # noqa: ARG002
        signal = np.exp(-1j * self._energy * unitary.time)
        basis = self._settings.get("test_basis")
        expectation = signal.real if basis == "X" else signal.imag
        n0 = round((1.0 + expectation) / 2.0 * self._resolution)
        n1 = self._resolution - n0
        return _FakeExecutorData({"0": int(n0), "1": int(n1)})


def _patch_with_ideal_signal(monkeypatch: pytest.MonkeyPatch, driver: RobustPhaseEstimation, energy: float) -> None:
    """Replace the driver's nested-algorithm factory with ideal-signal fakes."""

    def fake_create_nested(setting_key: str):
        if setting_key == "unitary_builder":
            return _FakeBuilder()
        if setting_key == "hadamard_test":
            return _FakeHadamardTest(energy)
        raise KeyError(setting_key)

    monkeypatch.setattr(driver, "_create_nested", fake_create_nested)


@pytest.mark.parametrize("energy", [0.4, -0.3, 0.75, 0.0])
def test_driver_recovers_energy_exact_mode(monkeypatch: pytest.MonkeyPatch, energy: float) -> None:
    """randomized=False: the RPE loop + linear energy map recover the injected energy."""
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    driver = RobustPhaseEstimation(target_accuracy=1e-4, randomized=False)
    _patch_with_ideal_signal(monkeypatch, driver, energy)

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)
    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)


@pytest.mark.parametrize("energy", [0.4, -0.3, 0.75])
def test_driver_recovers_energy_qdrift_mode(monkeypatch: pytest.MonkeyPatch, energy: float) -> None:
    """randomized=True: the tangent de-biasing leaves an ideal signal essentially unchanged."""
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    driver = RobustPhaseEstimation(target_accuracy=1e-4, randomized=True)
    _patch_with_ideal_signal(monkeypatch, driver, energy)

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)
    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)


def test_driver_uses_explicit_base_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user-provided base_time is honoured and still recovers the energy."""
    energy = 0.6
    hamiltonian = QubitHamiltonian(pauli_strings=["ZZ", "XX"], coefficients=[0.5, 0.5])
    driver = RobustPhaseEstimation(target_accuracy=1e-4, base_time=np.pi / 4, randomized=False)
    _patch_with_ideal_signal(monkeypatch, driver, energy)

    result = driver.run(state_preparation=object(), qubit_hamiltonian=hamiltonian)
    assert result.resolved_energy == pytest.approx(energy, abs=1e-3)
    assert result.evolution_time == pytest.approx(np.pi / 4)


def test_robust_phase_estimation_name() -> None:
    assert RobustPhaseEstimation().name() == "qdk_robust"


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

    driver = create("phase_estimation", "qdk_robust", target_accuracy=1e-3, randomized=False)
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

    driver = create("phase_estimation", "qdk_robust", target_accuracy=1e-2, randomized=True, seed=42)
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
