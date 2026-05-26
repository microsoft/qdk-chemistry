"""Tests for EulerIntegrator state-preparation composition."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import qdk_chemistry.algorithms.time_evolution.hamiltonian_simulation.base as hamiltonian_sim_base

if TYPE_CHECKING:
    from collections.abc import Callable
from qdk_chemistry.algorithms.state_preparation import identity_state_prep
from qdk_chemistry.algorithms.time_evolution.hamiltonian_simulation import EulerIntegrator
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    DrivenQubitHamiltonian,
    FlatPartition,
    QubitHamiltonian,
)
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT_AER, QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME


def _constant_drive(_t: float) -> float:
    """Drive function that always returns 1.0."""
    return 1.0


def test_prepend_state_prep_circuit_composes_qsharp_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper should compose state preparation and evolution through Q# operations."""
    algo = EulerIntegrator()
    state_prep_op = object()
    evolution_op = object()

    monkeypatch.setattr(
        hamiltonian_sim_base,
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
    algo = EulerIntegrator()
    state_prep = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n")
    evolution = Circuit(qasm="OPENQASM 3.0;\nqubit[1] q;\nx q[0];\n")

    with pytest.raises(RuntimeError, match="requires Q# operations"):
        algo._prepend_state_prep_circuit(state_prep, evolution, num_qubits=1)


def test_euler_integrator_eigenvalue_remains_constant() -> None:
    """Run an example EulerIntegrator workflow."""
    partition = FlatPartition(strategy="commuting", groups=[[0]])
    h0 = QubitHamiltonian(["ZZ"], np.array([0.0]), term_partition=partition)
    h1 = QubitHamiltonian(["ZZ"], np.array([1.0]), term_partition=partition)

    def square_wave(t: float) -> float:
        """Alternates +1/-1 based on which time step we are in."""
        step = int(t)
        return 1.0 if step % 2 == 0 else -1.0

    td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=square_wave)
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    algo = EulerIntegrator()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=100, order=1),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
    )
    algo.settings().set("total_time", 100.0)
    algo.settings().set("dt", 1.0)

    state_prep = identity_state_prep(num_qubits=2)
    measurements = algo.run(td_hamiltonian, observables=[observable], state_prep=state_prep, shots=1024)

    for measurement in measurements:
        assert measurement[0].energy_expectation_value == pytest.approx(1.0, abs=0.2)


@pytest.mark.skipif(
    not QDK_CHEMISTRY_HAS_QISKIT_AER or not QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME,
    reason="Qiskit Aer or IBM Runtime not available",
)
def test_euler_integrator_with_device_backend() -> None:
    """Run EulerIntegrator with a device_backend_name string."""
    partition = FlatPartition(strategy="commuting", groups=[[0]])
    h0 = QubitHamiltonian(["ZZ"], np.array([0.0]), term_partition=partition)
    h1 = QubitHamiltonian(["ZZ"], np.array([1.0]), term_partition=partition)
    td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=_constant_drive)
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    algo = EulerIntegrator()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=1, order=1),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qiskit_aer_simulator", device_backend_name="fake_manila"),
    )
    algo.settings().set("total_time", 1.0)
    algo.settings().set("dt", 1.0)

    state_prep = identity_state_prep(num_qubits=2)
    measurements = algo.run(td_hamiltonian, observables=[observable], state_prep=state_prep, shots=1024)

    for measurement in measurements:
        # With device noise the expectation value should still be close to 1.0
        assert measurement[0].energy_expectation_value == pytest.approx(1.0, abs=0.5)


# ---------------------------------------------------------------------------
# Error-path tests for _run_impl input validation
# ---------------------------------------------------------------------------


class TestEulerIntegratorValidation:
    """Tests for EulerIntegrator input validation."""

    def _make_hamiltonian(self, num_qubits: int = 2) -> QubitHamiltonian:
        labels = ["Z" + "I" * (num_qubits - 1)]
        return QubitHamiltonian(labels, np.array([1.0]))

    def _dummy_state_prep(self) -> Circuit:
        return identity_state_prep(num_qubits=2)

    def test_mismatched_observable_num_qubits_raises(self):
        """Observables with different num_qubits from Hamiltonians should raise ValueError."""
        algo = EulerIntegrator()
        h2 = self._make_hamiltonian(num_qubits=2)
        obs3 = self._make_hamiltonian(num_qubits=3)
        td = DrivenQubitHamiltonian(h2, h2, drive=_constant_drive)
        with pytest.raises(ValueError, match="same number of qubits"):
            algo.run(td, observables=[obs3], state_prep=self._dummy_state_prep())

    def test_get_circuit_before_run_raises(self):
        """get_circuit() before run() should raise ValueError."""
        algo = EulerIntegrator()
        with pytest.raises(ValueError, match="No evolution circuit"):
            algo.get_circuit()

    def test_dt_exceeds_total_time_raises(self):
        """Dt > total_time should raise ValueError."""
        h = self._make_hamiltonian(num_qubits=2)
        td = DrivenQubitHamiltonian(h, h, drive=_constant_drive)
        algo = EulerIntegrator()
        algo.settings().set(
            "evolution_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=1, order=1)
        )
        algo.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator"))
        algo.settings().set("total_time", 1.0)
        algo.settings().set("dt", 2.0)
        with pytest.raises(ValueError, match="must match not exceed"):
            algo.run(td, observables=[self._make_hamiltonian()], state_prep=self._dummy_state_prep())

    def test_dt_zero_raises(self):
        """Dt = 0 should raise ValueError."""
        h = self._make_hamiltonian(num_qubits=2)
        td = DrivenQubitHamiltonian(h, h, drive=_constant_drive)
        algo = EulerIntegrator()
        algo.settings().set(
            "evolution_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=1, order=1)
        )
        algo.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator"))
        algo.settings().set("total_time", 1.0)
        algo.settings().set("dt", 0.0)
        with pytest.raises(ValueError, match="must be nonzero"):
            algo.run(td, observables=[self._make_hamiltonian()], state_prep=self._dummy_state_prep())

    def test_dt_negative_raises(self):
        """Dt with opposite sign to total_time should raise ValueError."""
        h = self._make_hamiltonian(num_qubits=2)
        td = DrivenQubitHamiltonian(h, h, drive=_constant_drive)
        algo = EulerIntegrator()
        algo.settings().set(
            "evolution_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=1, order=1)
        )
        algo.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator"))
        algo.settings().set("total_time", 1.0)
        algo.settings().set("dt", -0.5)
        with pytest.raises(ValueError, match="must match the sign"):
            algo.run(td, observables=[self._make_hamiltonian()], state_prep=self._dummy_state_prep())

    def test_total_time_zero_raises(self):
        """total_time = 0 should raise ValueError."""
        h = self._make_hamiltonian(num_qubits=2)
        td = DrivenQubitHamiltonian(h, h, drive=_constant_drive)
        algo = EulerIntegrator()
        algo.settings().set(
            "evolution_builder", AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=1, order=1)
        )
        algo.settings().set("circuit_executor", AlgorithmRef("circuit_executor", "qdk_full_state_simulator"))
        algo.settings().set("total_time", 0.0)
        algo.settings().set("dt", 0.1)
        with pytest.raises(ValueError, match="total_time must be nonzero"):
            algo.run(td, observables=[self._make_hamiltonian()], state_prep=self._dummy_state_prep())

    def test_propagator_setting_via_create_kwargs(self):
        """Propagator AlgorithmRef should be configurable through create() kwargs."""
        from qdk_chemistry.algorithms import registry  # noqa: PLC0415

        algo = registry.create(
            "hamiltonian_simulation",
            "euler_integrator",
            propagator=AlgorithmRef("propagator", "magnus", order=2),
        )
        ref = algo.settings().get("propagator")
        assert ref.algorithm_type == "propagator"
        assert ref.algorithm_name == "magnus"


# ---------------------------------------------------------------------------
# Time-dependent Trotter tests with smooth driving functions
# ---------------------------------------------------------------------------


def _run_smooth_drive_test(
    drive: Callable[[float], float],
    total_time: float,
    num_divisions: int,
    *,
    dt: float = 0.1,
    expected_z: float = 0.0,
    atol: float = 0.05,
) -> None:
    """Helper: run EulerIntegrator with H(t)=drive(t)*X on one qubit, measure Z.

    Uses a non-commuting Hamiltonian so ⟨Z⟩ depends on the evolution time.
    """
    h0 = QubitHamiltonian(["I"], np.array([0.0]))
    h1 = QubitHamiltonian(["X"], np.array([1.0]))
    td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=drive)
    observable = QubitHamiltonian(["Z"], np.array([1.0]))

    algo = EulerIntegrator()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=num_divisions, order=1),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
    )
    algo.settings().set("total_time", total_time)
    algo.settings().set("dt", dt)

    state_prep = identity_state_prep(num_qubits=1)
    measurements = algo.run(td_hamiltonian, observables=[observable], state_prep=state_prep, shots=10000)

    for measurement in measurements:
        assert measurement[0].energy_expectation_value == pytest.approx(expected_z, abs=atol)


def test_euler_integrator_sinusoidal_drive() -> None:
    """Sinusoidal drive H(t) = sin(t)*X on single qubit, measure Z."""
    total_time = np.pi / 2
    # Exact: ⟨Z⟩ = cos(2∫₀ᵀ sin(t)dt) = cos(2(1 - cos(T)))
    expected_z = float(np.cos(2 * (1 - np.cos(total_time))))
    _run_smooth_drive_test(drive=np.sin, total_time=total_time, num_divisions=1, dt=np.pi / 20, expected_z=expected_z)


def test_euler_integrator_exponential_decay_drive() -> None:
    """Exponential decay drive H(t) = exp(-t)*X on single qubit, measure Z."""

    def exp_decay(t: float) -> float:
        return float(np.exp(-t))

    # Exact: ⟨Z⟩ = cos(2∫₀¹ exp(-t)dt) = cos(2(1 - e⁻¹))
    expected_z = float(np.cos(2 * (1 - np.exp(-1))))
    _run_smooth_drive_test(drive=exp_decay, total_time=1.0, num_divisions=1, dt=0.1, expected_z=expected_z)


def test_euler_integrator_linear_ramp_drive() -> None:
    """Linear ramp drive H(t) = t*X on single qubit, measure Z."""

    def linear_ramp(t: float) -> float:
        return t

    total_time = 1.0
    # Exact: ⟨Z⟩ = cos(2∫₀ᵀ t dt) = cos(T²)
    expected_z = float(np.cos(total_time**2))
    _run_smooth_drive_test(drive=linear_ramp, total_time=total_time, num_divisions=1, dt=0.1, expected_z=expected_z)


def test_euler_integrator_non_commuting_observable() -> None:
    """H(t) = X on single qubit, observable = Z, starting from |0⟩.

    Unlike the ZZ-eigenstate tests, here ⟨Z⟩ depends on evolution time:
    exact ⟨Z⟩ = cos(2t).  At t = π/4, ⟨Z⟩ = 0.  This catches bugs where
    the evolution time is wrong (e.g., doubled → ⟨Z⟩ = -1).
    """
    h0 = QubitHamiltonian(["I"], np.array([0.0]))
    h1 = QubitHamiltonian(["X"], np.array([1.0]))
    td_hamiltonian = DrivenQubitHamiltonian(h0, h1, drive=_constant_drive)
    observable = QubitHamiltonian(["Z"], np.array([1.0]))

    total_time = np.pi / 4
    dt = total_time  # single step — constant drive, so one step is exact

    algo = EulerIntegrator()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=1, order=1),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
    )
    algo.settings().set("total_time", total_time)
    algo.settings().set("dt", dt)

    state_prep = identity_state_prep(num_qubits=1)
    measurements = algo.run(td_hamiltonian, observables=[observable], state_prep=state_prep, shots=10000)

    # exact ⟨Z⟩ = cos(2 * pi/4) = 0
    assert measurements[0][0].energy_expectation_value == pytest.approx(0.0, abs=0.05)
