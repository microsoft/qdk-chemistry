"""Tests for EvolveAndMeasure state-preparation composition."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import qdk_chemistry.algorithms.time_evolution.measure_simulation.base as measure_base

if TYPE_CHECKING:
    from collections.abc import Callable
from qdk_chemistry.algorithms.time_evolution.measure_simulation import EvolveAndMeasure
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    DrivenQubitHamiltonian,
    FlatPartition,
    QubitHamiltonian,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT_AER, QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME
from qdk_chemistry.utils.qsharp import QSHARP_UTILS


def _constant_drive(_t: float) -> float:
    """Drive function that always returns 1.0."""
    return 1.0


def _identity_state_prep(num_qubits: int) -> Circuit:
    """Create a trivial state-prep circuit that leaves |0...0> unchanged."""
    params = {"pauliExponents": [], "pauliCoefficients": [], "repetitions": 1}
    targets = list(range(num_qubits))
    return Circuit(
        qsharp_op=QSHARP_UTILS.PauliExp.MakeRepPauliExpOp(params),
        qsharp_factory=QsharpFactoryData(
            program=QSHARP_UTILS.PauliExp.MakeRepPauliExpCircuit,
            parameter={"evo_params": params, "target_indices": targets},
        ),
    )


def test_prepend_state_prep_circuit_composes_qsharp_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper should compose state preparation and evolution through Q# operations."""
    algo = EvolveAndMeasure()
    state_prep_op = object()
    evolution_op = object()

    monkeypatch.setattr(
        measure_base,
        "QSHARP_UTILS",
        SimpleNamespace(
            CircuitComposition=SimpleNamespace(
                MakeSequentialCircuit="sequential-circuit",
                MakeSequentialOp=lambda first, second: ("sequential-op", first, second),
            )
        ),
    )

    state_prep = Circuit(
        qir='attributes #0 = { "required_num_qubits"="1" }',
        qsharp_op=state_prep_op,
        encoding="jordan-wigner",
    )
    evolution = Circuit(
        qir='attributes #0 = { "required_num_qubits"="1" }',
        qsharp_op=evolution_op,
        encoding="jordan-wigner",
    )

    combined = algo._prepend_state_prep_circuit(state_prep, evolution, num_qubits=1)

    assert combined._qsharp_factory is not None
    assert combined._qsharp_factory.program == "sequential-circuit"
    assert combined._qsharp_factory.parameter == {
        "first": state_prep_op,
        "second": evolution_op,
        "targets": [0],
    }
    assert combined._qsharp_op == ("sequential-op", state_prep_op, evolution_op)
    assert combined.encoding == "jordan-wigner"


def test_prepend_state_prep_circuit_requires_qsharp_operations() -> None:
    """The helper should fail fast when either circuit lacks a Q# operation handle."""
    algo = EvolveAndMeasure()
    state_prep = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n")
    evolution = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nx q[0];\n")

    with pytest.raises(RuntimeError, match="requires Q# operations"):
        algo._prepend_state_prep_circuit(state_prep, evolution, num_qubits=1)


def test_evolve_and_measure_eigenvalue_remains_constant() -> None:
    """Run an example EvolveAndMeasure workflow."""
    partition = FlatPartition(strategy="commuting", groups=[[0]])
    h0 = QubitHamiltonian(["ZZ"], np.array([0.0]), term_partition=partition)
    h1 = QubitHamiltonian(["ZZ"], np.array([1.0]), term_partition=partition)

    def square_wave(t: float) -> float:
        """Alternates +1/-1 based on which time step we are in."""
        step = int(t)
        return 1.0 if step % 2 == 0 else -1.0

    td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=square_wave)
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    algo = EvolveAndMeasure()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=100, order=1),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
    )
    algo.settings().set("shots", 1024)
    algo.settings().set("total_time", 100.0)
    algo.settings().set("dt", 1.0)

    state_prep = _identity_state_prep(num_qubits=2)
    measurements = algo.run(td_hamiltonian, observables=[observable], state_prep=state_prep)

    for measurement in measurements:
        assert measurement[0].energy_expectation_value == pytest.approx(1.0, abs=0.2)


@pytest.mark.skipif(
    not QDK_CHEMISTRY_HAS_QISKIT_AER or not QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME,
    reason="Qiskit Aer or IBM Runtime not available",
)
def test_evolve_and_measure_with_device_backend() -> None:
    """Run EvolveAndMeasure with a device_backend_name string."""
    partition = FlatPartition(strategy="commuting", groups=[[0]])
    h0 = QubitHamiltonian(["ZZ"], np.array([0.0]), term_partition=partition)
    h1 = QubitHamiltonian(["ZZ"], np.array([1.0]), term_partition=partition)
    td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=_constant_drive)
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    algo = EvolveAndMeasure()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=1, order=1),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qiskit_aer_simulator", device_backend_name="fake_manila"),
    )
    algo.settings().set("shots", 1024)

    state_prep = _identity_state_prep(num_qubits=2)
    measurements = algo.run(td_hamiltonian, observables=[observable], state_prep=state_prep)

    for measurement in measurements:
        # With device noise the expectation value should still be close to 1.0
        assert measurement[0].energy_expectation_value == pytest.approx(1.0, abs=0.5)


# ---------------------------------------------------------------------------
# Error-path tests for _run_impl input validation
# ---------------------------------------------------------------------------


class TestEvolveAndMeasureValidation:
    """Tests for EvolveAndMeasure input validation."""

    def _make_hamiltonian(self, num_qubits: int = 2) -> QubitHamiltonian:
        labels = ["Z" + "I" * (num_qubits - 1)]
        return QubitHamiltonian(labels, np.array([1.0]))

    def _dummy_state_prep(self) -> Circuit:
        return _identity_state_prep(num_qubits=2)

    def test_mismatched_observable_num_qubits_raises(self):
        """Observables with different num_qubits from Hamiltonians should raise ValueError."""
        algo = EvolveAndMeasure()
        h2 = self._make_hamiltonian(num_qubits=2)
        obs3 = self._make_hamiltonian(num_qubits=3)
        td = DrivenQubitHamiltonian(h2, h2, drive=_constant_drive)
        with pytest.raises(ValueError, match="same number of qubits"):
            algo.run(td, observables=[obs3], state_prep=self._dummy_state_prep())

    def test_get_circuit_before_run_raises(self):
        """get_circuit() before run() should raise ValueError."""
        algo = EvolveAndMeasure()
        with pytest.raises(ValueError, match="No evolution circuit"):
            algo.get_circuit()


# ---------------------------------------------------------------------------
# Time-dependent Trotter tests with smooth driving functions
# ---------------------------------------------------------------------------


def _run_smooth_drive_test(
    drive: Callable[[float], float],
    total_time: float,
    num_divisions: int,
    *,
    dt: float = 0.1,
    expected_zz: float = 1.0,
    atol: float = 0.2,
) -> None:
    """Helper: run EvolveAndMeasure with a given smooth drive and check ZZ expectation."""
    partition = FlatPartition(strategy="commuting", groups=[[0]])
    h0 = QubitHamiltonian(["ZZ"], np.array([0.0]), term_partition=partition)
    h1 = QubitHamiltonian(["ZZ"], np.array([1.0]), term_partition=partition)
    td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=drive)
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    algo = EvolveAndMeasure()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=num_divisions, order=1),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
    )
    algo.settings().set("shots", 1024)
    algo.settings().set("total_time", total_time)
    algo.settings().set("dt", dt)

    state_prep = _identity_state_prep(num_qubits=2)
    measurements = algo.run(td_hamiltonian, observables=[observable], state_prep=state_prep)

    for measurement in measurements:
        assert measurement[0].energy_expectation_value == pytest.approx(expected_zz, abs=atol)


def test_evolve_and_measure_sinusoidal_drive() -> None:
    """Sinusoidal drive on a ZZ-only Hamiltonian starting from |00>.

    H(t) = sin(t)*ZZ.  The |00> state is a +1 eigenstate of ZZ, so time
    evolution is a global phase and the ZZ expectation stays at +1.
    """
    _run_smooth_drive_test(drive=np.sin, total_time=2 * np.pi, num_divisions=50)


def test_evolve_and_measure_exponential_decay_drive() -> None:
    """Exponentially decaying drive on a ZZ-only Hamiltonian starting from |00>.

    H(t) = exp(-t)*ZZ.  Same eigenstate argument — expectation remains +1.
    """

    def exp_decay(t: float) -> float:
        return float(np.exp(-t))

    _run_smooth_drive_test(drive=exp_decay, total_time=3.0, num_divisions=30)


def test_evolve_and_measure_linear_ramp_drive() -> None:
    """Linear ramp drive on a ZZ-only Hamiltonian starting from |00>.

    H(t) = t*ZZ.  Same eigenstate argument — expectation remains +1.
    """

    def linear_ramp(t: float) -> float:
        return t

    _run_smooth_drive_test(drive=linear_ramp, total_time=5.0, num_divisions=50)
