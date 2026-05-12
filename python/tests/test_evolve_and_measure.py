"""Tests for EvolveAndMeasure state-preparation composition."""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import qdk_chemistry.algorithms.time_evolution.measure_simulation.base as measure_base
from qdk_chemistry.algorithms.time_evolution.measure_simulation import EvolveAndMeasure
from qdk_chemistry.data import (
    AlgorithmRef,
    Circuit,
    FlatPartition,
    PiecewiseConstantQubitHamiltonian,
    QubitHamiltonian,
)
from qdk_chemistry.data.circuit import QsharpFactoryData
from qdk_chemistry.plugins.qiskit import QDK_CHEMISTRY_HAS_QISKIT_AER, QDK_CHEMISTRY_HAS_QISKIT_IBM_RUNTIME
from qdk_chemistry.utils.qsharp import QSHARP_UTILS


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
    hamiltonian_p = QubitHamiltonian(["ZZ"], np.array([1.0]), term_partition=partition)
    hamiltonian_m = QubitHamiltonian(["ZZ"], np.array([-1.0]), term_partition=partition)

    steps = 100
    hamiltonians = [hamiltonian_p, hamiltonian_m] * (steps // 2)
    time_steps = [float(t + 1) for t in range(steps)]
    td_hamiltonian = PiecewiseConstantQubitHamiltonian(hamiltonians, time_steps)
    observable = QubitHamiltonian(["ZZ"], np.array([1.0]))

    algo = EvolveAndMeasure()
    algo.settings().set(
        "evolution_builder",
        AlgorithmRef("hamiltonian_unitary_builder", "trotter", num_divisions=1, order=1),
    )
    algo.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator"),
    )
    algo.settings().set("shots", 1024)

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
    hamiltonian = QubitHamiltonian(["ZZ"], np.array([1.0]), term_partition=partition)
    td_hamiltonian = PiecewiseConstantQubitHamiltonian([hamiltonian], [1.0])
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
        td = PiecewiseConstantQubitHamiltonian([h2], [1.0])
        with pytest.raises(ValueError, match="same number of qubits"):
            algo.run(td, observables=[obs3], state_prep=self._dummy_state_prep())

    def test_get_circuit_before_run_raises(self):
        """get_circuit() before run() should raise ValueError."""
        algo = EvolveAndMeasure()
        with pytest.raises(ValueError, match="No evolution circuit"):
            algo.get_circuit()
